"""
HTTP/ASGI endpoint tests for the LegalRAG FastAPI service (Phase 6).

These tests drive the real ASGI application (routing, request validation, dependency
injection, exception handlers, response schemas) through ``fastapi.testclient``. They
never load production artifacts, download models, or call external providers: every
pipeline used here is either the built-in ``local_stub`` pipeline or an explicitly
injected stub/failing component.

Note: ``TestClient`` is instantiated without the context manager so the application
``lifespan`` (production artifact provisioning) is never executed.
"""

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock

os.environ["ENVIRONMENT"] = "local_stub"

from fastapi.testclient import TestClient

from legalrag.api import routes as routes_module
from legalrag.api.config import Settings, get_settings
from legalrag.api.dependencies import (
    PipelineComponents,
    PipelineInitializationError,
    StubBM25Retriever,
    StubCrossEncoderReranker,
    StubDenseRetriever,
    StubGenerator,
    create_stub_chunks,
    get_pipeline,
    reset_pipeline_cache,
)
from legalrag.api.main import create_app
from legalrag.generation.client import GenerationError, GenerationResult


def make_stub_pipeline(generator=None) -> PipelineComponents:
    """Builds a deterministic local_stub pipeline."""
    chunks = create_stub_chunks()
    return PipelineComponents(
        bm25=StubBM25Retriever(chunk_ids=chunks["chunk_id"].tolist()),
        dense=StubDenseRetriever(chunk_ids=chunks["chunk_id"].tolist()),
        reranker=StubCrossEncoderReranker(),
        chunks=chunks,
        generator=generator or StubGenerator(),
        environment="local_stub",
    )


def production_settings(artifact_dir: Path = Path("/tmp/legalrag-missing-artifacts")) -> Settings:
    """Settings that point at a non-existent artifact directory (fail-closed path)."""
    return Settings(
        environment="production",
        groq_api_key="gsk_dummy_key_for_tests",
        artifact_dir=artifact_dir,
    )


class EndpointTestCase(unittest.TestCase):
    """Base class providing a TestClient with per-test dependency overrides."""

    def setUp(self):
        reset_pipeline_cache()
        self.addCleanup(reset_pipeline_cache)
        self.app = create_app()
        # No context manager => application lifespan (artifact provisioning) never runs.
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)

    def override_pipeline(self, pipeline) -> None:
        """Binds ``Depends(_pipeline_dependency)`` to a fixed pipeline instance."""

        def _provider() -> PipelineComponents:
            return pipeline

        self.app.dependency_overrides[routes_module._pipeline_dependency] = _provider

    def override_pipeline_error(self, error: Exception) -> None:
        """Binds the pipeline dependency to always raise ``error`` (not-ready path)."""

        def _provider() -> PipelineComponents:
            raise error

        self.app.dependency_overrides[routes_module._pipeline_dependency] = _provider

    def override_settings(self, settings: Settings) -> None:
        self.app.dependency_overrides[get_settings] = lambda: settings


class TestQueryRequestBodyContract(EndpointTestCase):
    """POST /query must bind exactly the documented ``{"question": ...}`` body."""

    def test_openapi_request_body_is_query_request_schema(self):
        schema = self.app.openapi()
        body_schema = schema["paths"]["/query"]["post"]["requestBody"]["content"][
            "application/json"
        ]["schema"]
        self.assertEqual(body_schema, {"$ref": "#/components/schemas/QueryRequest"})

    def test_openapi_query_request_schema_exposes_only_question(self):
        schema = self.app.openapi()
        properties = schema["components"]["schemas"]["QueryRequest"]["properties"]
        self.assertEqual(set(properties.keys()), {"question"})

    def test_server_settings_is_not_an_accepted_body_field(self):
        """Regression: server configuration must never be bindable from the request body."""
        schema = self.app.openapi()
        body_schema = schema["paths"]["/query"]["post"]["requestBody"]["content"][
            "application/json"
        ]["schema"]
        self.assertNotIn("settings", body_schema)
        self.assertNotIn(
            "Body_query_endpoint_query_post", schema["components"]["schemas"]
        )

    def test_documented_question_body_is_accepted(self):
        """Regression: the documented ``{"question": ...}`` payload must not be rejected."""
        self.override_pipeline(make_stub_pipeline())
        response = self.client.post(
            "/query",
            json={"question": "What are the conditions for granting anticipatory bail?"},
        )
        self.assertEqual(response.status_code, 200, response.text)

    def test_legacy_wrapped_request_body_is_rejected(self):
        """Guards against regressing to the accidental ``{"request": {...}}`` wrapper."""
        self.override_pipeline(make_stub_pipeline())
        response = self.client.post(
            "/query", json={"request": {"question": "What are the conditions for bail?"}}
        )
        self.assertEqual(response.status_code, 422, response.text)

    def test_client_cannot_inject_server_settings(self):
        """A client must not be able to supply Settings (e.g. force 'local_stub')."""
        self.override_pipeline(make_stub_pipeline())
        response = self.client.post(
            "/query",
            json={
                "question": "What are the conditions for granting anticipatory bail?",
                "settings": {"environment": "local_stub"},
            },
        )
        self.assertEqual(response.status_code, 422, response.text)


class TestQueryRequestValidation(EndpointTestCase):
    """Invalid requests must fail with HTTP 422 before any pipeline work happens."""

    def test_empty_question_is_422(self):
        self.override_pipeline(make_stub_pipeline())
        response = self.client.post("/query", json={"question": "   "})
        self.assertEqual(response.status_code, 422)

    def test_missing_question_is_422(self):
        self.override_pipeline(make_stub_pipeline())
        response = self.client.post("/query", json={})
        self.assertEqual(response.status_code, 422)

    def test_malformed_json_is_422(self):
        self.override_pipeline(make_stub_pipeline())
        response = self.client.post(
            "/query", content="{not valid json", headers={"content-type": "application/json"}
        )
        self.assertEqual(response.status_code, 422)

    def test_extra_field_is_422(self):
        self.override_pipeline(make_stub_pipeline())
        response = self.client.post(
            "/query", json={"question": "valid question", "top_k": 10}
        )
        self.assertEqual(response.status_code, 422)

    def test_invalid_type_is_422(self):
        self.override_pipeline(make_stub_pipeline())
        response = self.client.post("/query", json={"question": 123})
        self.assertEqual(response.status_code, 422)

    def test_question_exceeding_max_length_is_422(self):
        self.override_pipeline(make_stub_pipeline())
        response = self.client.post("/query", json={"question": "x" * 5000})
        self.assertEqual(response.status_code, 422)


class TestHealthAndReadinessEndpoints(EndpointTestCase):
    def test_health_is_liveness_only_and_never_loads_pipeline(self):
        def _must_not_load() -> PipelineComponents:
            raise AssertionError("GET /health must not initialize the retrieval pipeline")

        self.app.dependency_overrides[routes_module._pipeline_dependency] = _must_not_load
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "environment": "local_stub"})

    def test_ready_reports_ready_for_local_stub(self):
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")

    def test_ready_is_503_when_production_artifacts_are_missing(self):
        self.override_settings(production_settings())
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "Service is not ready."})

    def test_ready_is_503_without_groq_key_in_production(self):
        self.override_settings(
            Settings(environment="production", groq_api_key=None, artifact_dir=Path("/tmp/x"))
        )
        response = self.client.get("/ready")
        self.assertEqual(response.status_code, 503)


class TestQueryErrorSemantics(EndpointTestCase):
    """Retrieval/reranking errors -> 500; generation errors -> 502; not-ready -> 503."""

    def test_successful_grounded_response(self):
        self.override_pipeline(make_stub_pipeline())
        response = self.client.post(
            "/query",
            json={"question": "What are the requirements for anticipatory bail under Section 438?"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["answer"])
        self.assertTrue(payload["citations"])
        self.assertTrue(payload["retrieved_chunk_ids"])
        for key in ("retrieve_ms", "rerank_ms", "generate_ms", "total_ms"):
            self.assertGreaterEqual(payload[key], 0.0)
        for citation in payload["citations"]:
            self.assertIn(citation["chunk_id"], payload["retrieved_chunk_ids"])

    def test_retrieval_failure_is_500_and_sanitized(self):
        pipeline = make_stub_pipeline()
        pipeline.bm25 = MagicMock()
        pipeline.bm25.search.side_effect = RuntimeError("failed reading /srv/secret/index")
        self.override_pipeline(pipeline)
        response = self.client.post("/query", json={"question": "retrieval failure case"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Document retrieval failed."})
        self.assertNotIn("secret", response.text)

    def test_reranker_failure_is_500_and_sanitized(self):
        pipeline = make_stub_pipeline()
        pipeline.reranker = MagicMock()
        pipeline.reranker.rerank.side_effect = RuntimeError("cross encoder cuda oom")
        self.override_pipeline(pipeline)
        response = self.client.post("/query", json={"question": "reranking failure case"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Candidate reranking failed."})

    def test_generation_failure_is_502_and_sanitized(self):
        generator = MagicMock()
        generator.generate.side_effect = GenerationError(
            message="provider leaked gsk_super_secret_key", status_code=502
        )
        self.override_pipeline(make_stub_pipeline(generator=generator))
        response = self.client.post("/query", json={"question": "generation failure case"})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json(), {"detail": "Answer generation failed."})
        self.assertNotIn("gsk_super_secret_key", response.text)

    def test_generation_timeout_is_502(self):
        generator = MagicMock()
        generator.generate.side_effect = TimeoutError("provider timed out")
        self.override_pipeline(make_stub_pipeline(generator=generator))
        response = self.client.post("/query", json={"question": "generation timeout case"})
        self.assertEqual(response.status_code, 502)

    def test_unusable_generation_result_is_502(self):
        generator = MagicMock()
        generator.generate.return_value = GenerationResult(
            text=None, status="empty_response", error_type="EmptyResponseError", model_name="m"
        )
        self.override_pipeline(make_stub_pipeline(generator=generator))
        response = self.client.post("/query", json={"question": "empty generation case"})
        self.assertEqual(response.status_code, 502)

    def test_pipeline_not_initialized_is_503(self):
        self.override_pipeline_error(PipelineInitializationError("artifacts unavailable"))
        response = self.client.post("/query", json={"question": "pipeline not ready case"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "Service is not ready."})

    def test_query_never_falls_back_from_production_to_stub(self):
        """
        In production with missing artifacts the dependency must fail closed (503),
        never silently serve local_stub components.
        """
        reset_pipeline_cache()
        self.addCleanup(reset_pipeline_cache)

        def _production_provider() -> PipelineComponents:
            return get_pipeline(production_settings())

        self.app.dependency_overrides[routes_module._pipeline_dependency] = _production_provider
        response = self.client.post("/query", json={"question": "production fallback probe"})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("stub", response.text.lower())


class TestCitationHandling(EndpointTestCase):
    """Citation extraction/resolution through the HTTP endpoint."""

    def _pipeline_with_answer(self, answer_text) -> PipelineComponents:
        generator = MagicMock()
        generator.generate.return_value = GenerationResult(
            text=answer_text, status="success", model_name="stub-model"
        )
        return make_stub_pipeline(generator=generator)

    def test_valid_citations_are_resolved(self):
        self.override_pipeline(
            self._pipeline_with_answer(
                "Holding per [Chunk ID: stub_chunk_0] and [Chunk ID: stub_chunk_1]."
            )
        )
        payload = self.client.post("/query", json={"question": "citation resolution case"}).json()
        cited = [c["chunk_id"] for c in payload["citations"]]
        self.assertEqual(cited, ["stub_chunk_0", "stub_chunk_1"])

    def test_malformed_citations_are_ignored(self):
        self.override_pipeline(
            self._pipeline_with_answer("See [Chunk ID: ] and [Chunk ID and stray [Chunk ID x].")
        )
        response = self.client.post("/query", json={"question": "malformed citation case"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["citations"], [])

    def test_missing_citations_yield_no_citations(self):
        self.override_pipeline(
            self._pipeline_with_answer("A grounded answer with no explicit citation markers.")
        )
        response = self.client.post("/query", json={"question": "missing citation case"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["citations"], [])

    def test_hallucinated_chunk_ids_are_dropped(self):
        self.override_pipeline(
            self._pipeline_with_answer(
                "Valid [Chunk ID: stub_chunk_0] but fabricated [Chunk ID: chunk_fake_999]."
            )
        )
        payload = self.client.post("/query", json={"question": "hallucinated citation case"}).json()
        cited = [c["chunk_id"] for c in payload["citations"]]
        self.assertEqual(cited, ["stub_chunk_0"])


if __name__ == "__main__":
    unittest.main()
