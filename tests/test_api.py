"""
Unit and integration tests for the LegalRAG FastAPI application and API routes.

Tests endpoint behaviors in 'local_stub' environment:
- Health check (GET /health)
- Service root (GET /)
- Query endpoint (POST /query) with grounded answers, citations, and latency attribution
- Request payload validation (422 for empty/missing questions)
- Error shielding (graceful handling of retrieval and generation errors)
"""

import unittest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from legalrag.api.config import Settings, get_settings
from legalrag.api.dependencies import (
    PipelineComponents,
    StubBM25Retriever,
    StubCrossEncoderReranker,
    StubDenseRetriever,
    StubGenerator,
    create_stub_chunks,
    get_pipeline,
    reset_pipeline_cache,
)
from legalrag.api.main import create_app
from legalrag.generation.client import GenerationResult


class TestAPIEndpoints(unittest.TestCase):
    """Test suite for FastAPI endpoints using TestClient in local_stub mode."""

    def setUp(self):
        """Set up test client with clean settings and pipeline singletons."""
        reset_pipeline_cache()
        self.app = create_app()
        self.client = TestClient(self.app)

    def tearDown(self):
        """Clean up pipeline cache after each test."""
        reset_pipeline_cache()

    def test_health_endpoint(self):
        """Tests GET /health returns status 'ok' and environment mode."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["environment"], "local_stub")

    def test_root_endpoint(self):
        """Tests GET / returns API metadata and valid documentation links."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], "LegalRAG API")
        self.assertEqual(data["environment"], "local_stub")
        self.assertIn("docs_url", data)

    def test_query_endpoint_success(self):
        """Tests POST /query returns grounded answer, citations, chunk IDs, and latency timings."""
        payload = {"question": "What are the requirements for anticipatory bail under Section 438 CrPC?"}
        response = self.client.post("/query", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIsInstance(data["answer"], str)
        self.assertGreater(len(data["answer"]), 10)

        # Verify citation structure
        self.assertIsInstance(data["citations"], list)
        self.assertGreater(len(data["citations"]), 0)
        first_citation = data["citations"][0]
        self.assertEqual(first_citation["chunk_id"], "stub_chunk_0")
        self.assertIsNotNone(first_citation["court_code"])
        self.assertIsNotNone(first_citation["title"])

        # Verify retrieved chunk IDs
        self.assertIsInstance(data["retrieved_chunk_ids"], list)
        self.assertGreater(len(data["retrieved_chunk_ids"]), 0)

        # Verify latency tracking
        self.assertIn("retrieve_ms", data)
        self.assertIn("rerank_ms", data)
        self.assertIn("generate_ms", data)
        self.assertIn("total_ms", data)
        self.assertGreaterEqual(data["total_ms"], 0.0)

    def test_query_validation_empty_question(self):
        """Tests POST /query rejects whitespace-only or empty question with 422."""
        response = self.client.post("/query", json={"question": "   "})
        self.assertEqual(response.status_code, 422)

    def test_query_validation_missing_field(self):
        """Tests POST /query rejects missing question key with 422."""
        response = self.client.post("/query", json={})
        self.assertEqual(response.status_code, 422)

    def test_query_generation_error_shield(self):
        """
        Tests that LLM generation failures (API errors, rate limits) are shielded
        and return status='generation_error' without leaking raw exceptions or crashing.
        """
        # Create a mock generator that returns a failed GenerationResult
        mock_generator = MagicMock()
        mock_generator.generate.return_value = GenerationResult(
            text=None,
            status="rate_limited",
            error_message="Upstream rate limit exceeded (429)",
            error_type="ClientError",
        )

        stub_chunks = create_stub_chunks()
        custom_pipeline = PipelineComponents(
            bm25=StubBM25Retriever(chunk_ids=stub_chunks["chunk_id"].tolist()),
            dense=StubDenseRetriever(chunk_ids=stub_chunks["chunk_id"].tolist()),
            reranker=StubCrossEncoderReranker(),
            chunks=stub_chunks,
            generator=mock_generator,
            environment="local_stub",
        )

        self.app.dependency_overrides[get_pipeline] = lambda: custom_pipeline

        response = self.client.post("/query", json={"question": "Test question on error handling?"})
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "generation_error")
        self.assertEqual(data["answer"], "An error occurred while generating the legal answer. Please try again later.")
        self.assertEqual(data["citations"], [])

        self.app.dependency_overrides.clear()

    def test_query_retrieval_error_shield(self):
        """
        Tests that retrieval failures are gracefully caught and return status='retrieval_error'.
        """
        mock_bm25 = MagicMock()
        mock_bm25.search.side_effect = RuntimeError("FAISS index unavailable")

        stub_chunks = create_stub_chunks()
        custom_pipeline = PipelineComponents(
            bm25=mock_bm25,
            dense=StubDenseRetriever(chunk_ids=stub_chunks["chunk_id"].tolist()),
            reranker=StubCrossEncoderReranker(),
            chunks=stub_chunks,
            generator=StubGenerator(),
            environment="local_stub",
        )

        self.app.dependency_overrides[get_pipeline] = lambda: custom_pipeline

        response = self.client.post("/query", json={"question": "Test question on retrieval failure?"})
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "retrieval_error")
        self.assertEqual(data["answer"], "An error occurred during document retrieval. Please try again later.")

        self.app.dependency_overrides.clear()

    def test_citation_filtering_hallucinated_ids(self):
        """
        Tests that cited chunk IDs not present in retrieved context are filtered out.
        """
        mock_generator = MagicMock()
        mock_generator.generate.return_value = GenerationResult(
            text="Holding citing valid [Chunk ID: stub_chunk_0] and fake [Chunk ID: chunk_fake_999].",
            status="success",
            model_name="stub-model",
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            finish_reason="stop",
        )

        stub_chunks = create_stub_chunks()
        custom_pipeline = PipelineComponents(
            bm25=StubBM25Retriever(chunk_ids=stub_chunks["chunk_id"].tolist()),
            dense=StubDenseRetriever(chunk_ids=stub_chunks["chunk_id"].tolist()),
            reranker=StubCrossEncoderReranker(),
            chunks=stub_chunks,
            generator=mock_generator,
            environment="local_stub",
        )

        self.app.dependency_overrides[get_pipeline] = lambda: custom_pipeline

        response = self.client.post("/query", json={"question": "Test hallucinated citation?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()

        cited_ids = [c["chunk_id"] for c in data["citations"]]
        self.assertIn("stub_chunk_0", cited_ids)
        self.assertNotIn("chunk_fake_999", cited_ids)

        self.app.dependency_overrides.clear()

    def test_settings_production_validation(self):
        """
        Tests that production environment requires GEMINI_API_KEY.
        """
        with self.assertRaises(ValueError):
            Settings(environment="production", gemini_api_key=None)

        # Setting the key should succeed
        valid_settings = Settings(
            environment="production",
            gemini_api_key="AIzaSyDummyKeyForTestingOnly",
        )
        self.assertEqual(valid_settings.environment, "production")
        self.assertTrue(valid_settings.is_production)
        self.assertFalse(valid_settings.is_local_stub)


if __name__ == "__main__":
    unittest.main()
