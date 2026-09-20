"""
API contract, readiness, fail-closed, provisioning, and hardening tests.

These tests exercise the real route handlers and dependency logic directly (no ASGI
event loop and no network), using local_stub components and mocks only. They never
download artifacts, load production models, or call external providers.
"""

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from pydantic import ValidationError

# Force local_stub before any application settings are constructed.
os.environ["ENVIRONMENT"] = "local_stub"

from legalrag.api.config import Settings
from legalrag.api.dependencies import (
    PipelineComponents,
    PipelineInitializationError,
    StubBM25Retriever,
    StubCrossEncoderReranker,
    StubDenseRetriever,
    StubGenerator,
    check_readiness,
    create_stub_chunks,
    get_pipeline,
    reset_pipeline_cache,
)
from legalrag.api.main import create_app
from legalrag.api.provisioning import (
    REQUIRED_ARTIFACT_NAMES,
    ArtifactProvisioningError,
    missing_artifacts,
    provision_production_artifacts,
)
from legalrag.api.routes import health_check, query_endpoint, readiness_check
from legalrag.api.schemas import QueryRequest
from legalrag.generation.client import GenerationError, GenerationResult


def make_stub_pipeline(generator=None) -> PipelineComponents:
    """Builds a deterministic local_stub pipeline for route-level tests."""
    chunks = create_stub_chunks()
    return PipelineComponents(
        bm25=StubBM25Retriever(chunk_ids=chunks["chunk_id"].tolist()),
        dense=StubDenseRetriever(chunk_ids=chunks["chunk_id"].tolist()),
        reranker=StubCrossEncoderReranker(),
        chunks=chunks,
        generator=generator or StubGenerator(),
        environment="local_stub",
    )


class _FakeCompletedProcess:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestRequestContract(unittest.TestCase):
    """POST /query must accept ONLY 'question'; all server config is server-side."""

    def test_query_request_exposes_only_question(self):
        self.assertEqual(set(QueryRequest.model_fields.keys()), {"question"})

    def test_openapi_query_schema_exposes_only_question(self):
        schema = create_app().openapi()
        query_schema = schema["components"]["schemas"]["QueryRequest"]
        self.assertEqual(set(query_schema.get("properties", {}).keys()), {"question"})
        self.assertFalse(query_schema.get("additionalProperties", True))

    def test_health_and_readiness_routes_registered(self):
        schema = create_app().openapi()
        self.assertIn("/health", schema["paths"])
        self.assertIn("/ready", schema["paths"])

    def test_error_handlers_map_to_correct_status_codes(self):
        import asyncio

        app = create_app()
        not_ready = asyncio.run(
            app.exception_handlers[PipelineInitializationError](None, PipelineInitializationError("x"))
        )
        generation = asyncio.run(
            app.exception_handlers[GenerationError](None, GenerationError("x", status_code=502))
        )
        self.assertEqual(not_ready.status_code, 503)
        self.assertEqual(generation.status_code, 502)

    def test_server_configuration_cannot_be_supplied(self):
        forbidden = [
            "environment",
            "artifact_dir",
            "hf_token",
            "hf_repo_id",
            "embedding_model_name",
            "reranker_model_name",
            "groq_api_key",
            "groq_model_name",
            "host",
            "port",
            "api_key",
        ]
        for field in forbidden:
            with self.subTest(field=field):
                with self.assertRaises(ValidationError):
                    QueryRequest.model_validate({"question": "valid question", field: "x"})

    def test_empty_question_rejected(self):
        with self.assertRaises(ValidationError):
            QueryRequest.model_validate({"question": "   "})

    def test_missing_question_rejected(self):
        with self.assertRaises(ValidationError):
            QueryRequest.model_validate({})


class TestHealthAndReadiness(unittest.TestCase):
    """Liveness (/health) is cheap; readiness (/ready) verifies the pipeline."""

    def setUp(self):
        reset_pipeline_cache()

    def tearDown(self):
        reset_pipeline_cache()

    def test_health_is_liveness_only(self):
        with patch("legalrag.api.routes.get_pipeline", side_effect=AssertionError("must not load")):
            response = health_check(Settings(environment="local_stub"))
        self.assertEqual(response.status, "ok")
        self.assertEqual(response.environment, "local_stub")

    def test_readiness_local_stub_ready(self):
        response = readiness_check(Settings(environment="local_stub"))
        self.assertEqual(response.status, "ready")

    def test_readiness_production_missing_artifacts_is_503(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                environment="production",
                groq_api_key="gsk_dummy_key_for_test",
                artifact_dir=Path(tmp) / "artifacts",
            )
            with self.assertRaises(HTTPException) as ctx:
                readiness_check(settings)
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertNotIn("artifacts", str(ctx.exception.detail))

    def test_readiness_production_missing_groq_key_is_503(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                environment="production",
                groq_api_key=None,
                artifact_dir=Path(tmp),
            )
            ready, reason = check_readiness(settings)
        self.assertFalse(ready)
        self.assertNotIn("GROQ", reason)

    def test_production_configuration_issues_report_missing_key(self):
        with_key = Settings(environment="production", groq_api_key="gsk_x")
        without_key = Settings(environment="production", groq_api_key=None)
        self.assertEqual(without_key.production_configuration_issues, ["GROQ_API_KEY is not configured."])
        self.assertEqual(with_key.production_configuration_issues, [])

    def test_default_environment_is_production(self):
        self.assertEqual(Settings.model_fields["environment"].default, "production")


class TestQueryEndpoint(unittest.TestCase):
    """POST /query success and proper HTTP error semantics."""

    def test_query_success_uses_court_code_metadata(self):
        pipeline = make_stub_pipeline()
        response = query_endpoint(
            QueryRequest.model_validate({"question": "What are the requirements for anticipatory bail?"}),
            pipeline,
        )
        self.assertEqual(response.status, "ok")
        self.assertGreater(len(response.answer), 10)
        self.assertGreater(len(response.retrieved_chunk_ids), 0)
        self.assertGreater(len(response.citations), 0)

        citation = response.citations[0]
        self.assertEqual(citation.chunk_id, "stub_chunk_0")
        self.assertEqual(citation.court_code, "01")
        # court_name does not exist in the real schema and must not be fabricated.
        self.assertFalse(hasattr(citation, "court_name"))

    def test_query_generation_failure_returns_502(self):
        generator = MagicMock()
        generator.generate.return_value = GenerationResult(
            text=None,
            status="api_error",
            error_message="raw provider detail",
            error_type="APIError",
            model_name="m",
        )
        pipeline = make_stub_pipeline(generator=generator)
        with self.assertRaises(HTTPException) as ctx:
            query_endpoint(QueryRequest.model_validate({"question": "generation failure?"}), pipeline)
        self.assertEqual(ctx.exception.status_code, 502)

    def test_query_generation_exception_is_sanitized_502(self):
        generator = MagicMock()
        generator.generate.side_effect = GenerationError(
            message="provider leaked gsk_super_secret_key", status_code=502
        )
        pipeline = make_stub_pipeline(generator=generator)
        with self.assertRaises(HTTPException) as ctx:
            query_endpoint(QueryRequest.model_validate({"question": "generation failure?"}), pipeline)
        self.assertEqual(ctx.exception.status_code, 502)
        self.assertNotIn("gsk_super_secret_key", str(ctx.exception.detail))

    def test_query_retrieval_failure_returns_sanitized_500(self):
        bm25 = MagicMock()
        bm25.search.side_effect = RuntimeError("failed to read /internal/secret/index")
        pipeline = make_stub_pipeline()
        pipeline.bm25 = bm25
        with self.assertRaises(HTTPException) as ctx:
            query_endpoint(QueryRequest.model_validate({"question": "retrieval failure?"}), pipeline)
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertNotIn("/internal/secret/index", str(ctx.exception.detail))

    def test_query_filters_hallucinated_citations(self):
        generator = MagicMock()
        generator.generate.return_value = GenerationResult(
            text="Holding citing valid [Chunk ID: stub_chunk_0] and fake [Chunk ID: chunk_fake_999].",
            status="success",
            model_name="stub-model",
        )
        pipeline = make_stub_pipeline(generator=generator)
        response = query_endpoint(QueryRequest.model_validate({"question": "citations?"}), pipeline)
        cited_ids = [c.chunk_id for c in response.citations]
        self.assertIn("stub_chunk_0", cited_ids)
        self.assertNotIn("chunk_fake_999", cited_ids)


class TestProductionFailClosed(unittest.TestCase):
    """Production can never silently return stub components."""

    def setUp(self):
        reset_pipeline_cache()

    def tearDown(self):
        reset_pipeline_cache()

    def test_local_stub_loads_stub_components(self):
        pipeline = get_pipeline(Settings(environment="local_stub"))
        self.assertEqual(pipeline.environment, "local_stub")
        self.assertIsInstance(pipeline.generator, StubGenerator)

    def test_production_loads_production_components(self):
        production = PipelineComponents(
            bm25=MagicMock(),
            dense=MagicMock(),
            reranker=MagicMock(),
            chunks=create_stub_chunks(),
            generator=MagicMock(),
            environment="production",
        )
        with patch("legalrag.api.dependencies.load_production_pipeline", return_value=production):
            pipeline = get_pipeline(
                Settings(environment="production", groq_api_key="gsk_x", artifact_dir=Path("/tmp/none"))
            )
        self.assertIs(pipeline, production)
        self.assertEqual(pipeline.environment, "production")
        self.assertNotIsInstance(pipeline.generator, StubGenerator)

    def test_production_refuses_stub_components(self):
        stub = make_stub_pipeline()
        with patch("legalrag.api.dependencies.load_production_pipeline", return_value=stub):
            with self.assertRaises(PipelineInitializationError):
                get_pipeline(
                    Settings(environment="production", groq_api_key="gsk_x", artifact_dir=Path("/tmp/none"))
                )

    def test_production_initialization_failure_raises_and_caches_error(self):
        settings = Settings(
            environment="production", groq_api_key="gsk_x", artifact_dir=Path("/tmp/definitely-missing")
        )
        with self.assertRaises(PipelineInitializationError):
            get_pipeline(settings)
        # Second call must fail fast (fail closed) rather than retry or return stubs.
        with self.assertRaises(PipelineInitializationError):
            get_pipeline(settings)


class TestConcurrentInitialization(unittest.TestCase):
    """Concurrent first requests must initialize the heavy pipeline exactly once."""

    def setUp(self):
        reset_pipeline_cache()

    def tearDown(self):
        reset_pipeline_cache()

    def test_concurrent_initialization_single_load(self):
        calls = {"count": 0}

        def slow_loader():
            time.sleep(0.2)
            calls["count"] += 1
            return make_stub_pipeline()

        n_threads = 8
        barrier = threading.Barrier(n_threads)
        results = []

        def worker():
            barrier.wait()
            results.append(get_pipeline(Settings(environment="local_stub")))

        with patch("legalrag.api.dependencies.load_stub_pipeline", side_effect=slow_loader):
            threads = [threading.Thread(target=worker) for _ in range(n_threads)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(calls["count"], 1)
        self.assertEqual(len(results), n_threads)
        self.assertTrue(all(result is results[0] for result in results))


class TestArtifactProvisioning(unittest.TestCase):
    """Startup provisioning reuses present artifacts and downloads only missing ones."""

    def _settings(self, artifact_dir: Path) -> Settings:
        return Settings(
            environment="production",
            groq_api_key="gsk_x",
            hf_repo_id="user/legalrag-artifacts",
            hf_token="hf_secret_token_1234567890",
            artifact_dir=artifact_dir,
        )

    def test_missing_artifacts_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self.assertEqual(set(missing_artifacts(tmp_path)), set(REQUIRED_ARTIFACT_NAMES))
            for name in REQUIRED_ARTIFACT_NAMES:
                (tmp_path / name).write_bytes(b"data")
            self.assertEqual(missing_artifacts(tmp_path), [])

    def test_existing_artifacts_are_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for name in REQUIRED_ARTIFACT_NAMES:
                (tmp_path / name).write_bytes(b"data")

            def must_not_run(cmd, env, timeout):
                raise AssertionError("provisioning must not run when artifacts exist")

            provision_production_artifacts(self._settings(tmp_path), runner=must_not_run)

    def test_missing_artifacts_are_downloaded_without_token_in_argv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            captured = {}

            def runner(cmd, env, timeout):
                captured["cmd"] = list(cmd)
                captured["env"] = dict(env)
                target = Path(cmd[cmd.index("--target-dir") + 1])
                for name in REQUIRED_ARTIFACT_NAMES:
                    (target / name).write_bytes(b"downloaded")
                return _FakeCompletedProcess(returncode=0)

            settings = self._settings(tmp_path)
            provision_production_artifacts(settings, runner=runner)

            self.assertEqual(missing_artifacts(tmp_path), [])
            self.assertIn("--repo-id", captured["cmd"])
            self.assertNotIn(settings.hf_token, captured["cmd"])
            self.assertNotIn(settings.hf_token, " ".join(captured["cmd"]))
            self.assertEqual(captured["env"].get("HF_TOKEN"), settings.hf_token)

    def test_missing_repo_id_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                environment="production",
                groq_api_key="gsk_x",
                hf_repo_id=None,
                artifact_dir=Path(tmp),
            )
            with self.assertRaises(ArtifactProvisioningError):
                provision_production_artifacts(settings)

    def test_download_failure_raises_and_does_not_leak_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(Path(tmp))

            def runner(cmd, env, timeout):
                return _FakeCompletedProcess(
                    returncode=1, stderr=f"auth failed for {settings.hf_token}"
                )

            with self.assertRaises(ArtifactProvisioningError) as ctx:
                provision_production_artifacts(settings, runner=runner)
            self.assertNotIn(settings.hf_token, str(ctx.exception))

    def test_incomplete_download_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(Path(tmp))
            with self.assertRaises(ArtifactProvisioningError):
                provision_production_artifacts(
                    settings, runner=lambda cmd, env, timeout: _FakeCompletedProcess(returncode=0)
                )


if __name__ == "__main__":
    unittest.main()
