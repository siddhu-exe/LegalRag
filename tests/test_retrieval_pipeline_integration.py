"""
Scaled retrieval-cascade integration tests (Phase 4/5).

Builds real ``bm25.pkl`` / ``dense.index`` / ``legal_chunks.parquet`` artifacts over a
synthetic corpus large enough that every cascade stage saturates (BM25 top-50, Dense
top-50, RRF top-50, CrossEncoder top-5). BM25 and FAISS are real; the embedding model,
cross-encoder, and generator are deterministic stubs so no production weights are
downloaded and no paid API is called.

The production-sized 538,079-chunk artifacts are intentionally *not* used here (they are
not present locally); this exercises the same code paths at a smaller scale.
"""

import os
import pickle
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["ENVIRONMENT"] = "local_stub"

from fastapi.testclient import TestClient

from legalrag.api import routes as routes_module
from legalrag.api.config import Settings
from legalrag.api.dependencies import (
    PipelineComponents,
    StubCrossEncoderReranker,
    load_production_pipeline,
    reset_pipeline_cache,
)
from legalrag.api.main import create_app
from legalrag.generation.client import GenerationResult
from legalrag.retrieval.bm25 import tokenize_legal_text
from legalrag.retrieval.dense import DenseRetriever
from legalrag.retrieval.fusion import reciprocal_rank_fusion

NUM_CHUNKS = 240
TOP_K = 50
RRF_K = 60
RERANK_TOP_K = 5

# Distinct legal themes so lexical (BM25) and dense matching are both non-degenerate.
THEMES = [
    "anticipatory bail under section 438 crpc requires reasonable apprehension of arrest",
    "dishonour of cheque under section 138 negotiable instruments act statutory notice",
    "writ petition under article 226 constitution enforcement of fundamental rights",
    "principles of natural justice audi alteram partem in departmental enquiry",
    "interference with arbitral award under section 34 patent illegality",
    "maintenance under section 125 crpc sufficient means of the husband",
    "quashing of fir under section 482 crpc absence of cognizable offence",
    "specific performance of contract under specific relief act readiness willingness",
    "service jurisprudence seniority promotion and administrative exigency",
    "compensation under motor vehicles act contributory negligence",
]


def build_scaled_corpus(num_chunks: int = NUM_CHUNKS):
    """Returns (chunk_ids, texts, dataframe) for a deterministic synthetic corpus."""
    chunk_ids, texts, rows = [], [], []
    for index in range(num_chunks):
        theme = THEMES[index % len(THEMES)]
        chunk_id = f"CNR{index:06d}_chunk_0"
        text = (
            f"{theme}. The court in judgment number {index} considered the pleadings and "
            f"exhibits and recorded its findings under statute {100 + (index % 400)}."
        )
        chunk_ids.append(chunk_id)
        texts.append(text)
        rows.append(
            {
                "chunk_id": chunk_id,
                "cnr": f"CNR{index:06d}",
                "chunk_index": 0,
                "text": text,
                "court_code": f"{index % 25:02d}",
                "decision_date": f"20{index % 20:02d}-01-15",
                "title": f"Case {index} v. State",
            }
        )
    return chunk_ids, texts, pd.DataFrame(rows)


class ScaledFakeEmbedder:
    """
    Deterministic bag-of-tokens embedder (normalized) used in place of
    SentenceTransformer so no model weights are downloaded.
    """

    def __init__(self, vocab):
        self._index = {token: i for i, token in enumerate(vocab)}
        self.dim = len(self._index)

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(
        self,
        texts,
        batch_size: int = 64,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        **kwargs,
    ) -> np.ndarray:
        vectors = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            for token in tokenize_legal_text(text):
                position = self._index.get(token)
                if position is not None:
                    vec[position] += 1.0
            norm = float(np.linalg.norm(vec))
            if normalize_embeddings and norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.vstack(vectors).astype(np.float32)


class ContextEchoGenerator:
    """Deterministic generator that cites the chunk IDs present in its own prompt."""

    def __init__(self):
        self.model_name = "context-echo"

    def generate(self, system_prompt, user_prompt, temperature=0.0, max_tokens=1024):
        pattern = re.compile(r"\[Chunk ID:\s*([^\]|\s][^\]|]*?)\s*(?:\||\])")
        cited = list(dict.fromkeys(pattern.findall(user_prompt)))[:2]
        text = "Held, per the retrieved judgments " + " ".join(
            f"[Chunk ID: {cid}]" for cid in cited
        )
        return GenerationResult(text=text, status="success", model_name=self.model_name)


class ScaledCascadeTestCase(unittest.TestCase):
    """Shared real-artifact pipeline built over the scaled synthetic corpus."""

    def setUp(self):
        reset_pipeline_cache()
        self.addCleanup(reset_pipeline_cache)
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.artifact_dir = Path(self._tmpdir.name)

        self.chunk_ids, self.texts, self.chunks_df = build_scaled_corpus()
        self.embedder = ScaledFakeEmbedder(
            sorted({token for text in self.texts for token in tokenize_legal_text(text)})
        )

        self._write_artifacts()
        self.pipeline = self._load_pipeline()
        self.pipeline.reranker = StubCrossEncoderReranker()
        self.pipeline.generator = ContextEchoGenerator()
        self.known_ids = set(self.chunk_ids)

    def _write_artifacts(self):
        from rank_bm25 import BM25Okapi

        model = BM25Okapi([tokenize_legal_text(t) for t in self.texts], k1=1.5, b=0.75)
        with open(self.artifact_dir / "bm25.pkl", "wb") as handle:
            pickle.dump({"bm25": model, "chunk_ids": list(self.chunk_ids)}, handle, protocol=4)

        dense = DenseRetriever(model_name="scaled-fake-bge")
        dense._model = self.embedder
        embeddings = dense.encode(self.texts, normalize=True)
        dense.build_index(embeddings, list(self.chunk_ids))
        dense.save_index(self.artifact_dir / "dense.index")

        self.chunks_df.to_parquet(self.artifact_dir / "legal_chunks.parquet", index=False)

    def _load_pipeline(self) -> PipelineComponents:
        settings = Settings(
            environment="production",
            groq_api_key="gsk_dummy_key_for_tests",
            artifact_dir=self.artifact_dir,
            model_warmup_enabled=False,
        )
        pipeline = load_production_pipeline(settings)
        pipeline.dense._model = self.embedder
        pipeline.dense.validate()
        pipeline.bm25.validate()
        return pipeline

    def question(self) -> str:
        return "anticipatory bail under section 438 crpc apprehension of arrest"

    def retrieve_stages(self, question=None):
        question = question or self.question()
        bm25_scored = self.pipeline.bm25.search(question, top_k=TOP_K)
        dense_scored = self.pipeline.dense.search(question, top_k=TOP_K)
        fused = reciprocal_rank_fusion(
            [[cid for cid, _ in bm25_scored], [cid for cid, _ in dense_scored]],
            k=RRF_K,
            top_n=TOP_K,
        )
        return bm25_scored, dense_scored, fused


class TestCascadeStageContract(ScaledCascadeTestCase):
    def test_retrievers_are_usable_not_merely_non_null(self):
        self.assertIsNotNone(self.pipeline.bm25.model)
        self.assertIsNotNone(self.pipeline.dense.index)
        self.assertEqual(self.pipeline.dense.index.ntotal, len(self.chunk_ids))
        self.assertEqual(self.pipeline.bm25.model.corpus_size, len(self.chunk_ids))

    def test_every_stage_returns_exact_expected_counts(self):
        bm25_scored, dense_scored, fused = self.retrieve_stages()

        self.assertEqual(len(bm25_scored), TOP_K, "BM25 must return exactly top-50")
        self.assertEqual(len(dense_scored), TOP_K, "Dense must return exactly top-50")
        self.assertEqual(len(fused), TOP_K, "RRF must return exactly top-50")

        chunk_text = dict(zip(self.chunk_ids, self.texts))
        candidates = [(cid, chunk_text[cid]) for cid, _ in fused]
        reranked = self.pipeline.reranker.rerank(
            query=self.question(), candidates=candidates, top_k=RERANK_TOP_K
        )
        self.assertEqual(len(reranked), RERANK_TOP_K, "Reranker must return exactly top-5")

    def test_all_stage_ids_are_valid_unique_and_resolve(self):
        bm25_scored, dense_scored, fused = self.retrieve_stages()
        for stage in (bm25_scored, dense_scored, fused):
            ids = [cid for cid, _ in stage]
            self.assertEqual(len(ids), len(set(ids)), "stage returned duplicate chunk ids")
            for cid in ids:
                self.assertIn(cid, self.known_ids)
        retrieved = {cid for cid, _ in bm25_scored + dense_scored}
        self.assertTrue({cid for cid, _ in fused} <= retrieved)

    def test_all_scores_are_finite(self):
        bm25_scored, dense_scored, fused = self.retrieve_stages()
        for stage in (bm25_scored, dense_scored, fused):
            for cid, score in stage:
                self.assertIsNotNone(score)
                self.assertIsInstance(score, float)
                self.assertTrue(np.isfinite(score), f"non-finite score for {cid}")

    def test_relevant_chunk_is_ranked_first(self):
        bm25_scored, dense_scored, _ = self.retrieve_stages()
        # Every third chunk carries the "anticipatory bail" theme; at least one must lead.
        top_text = dict(zip(self.chunk_ids, self.texts))[bm25_scored[0][0]]
        self.assertIn("anticipatory bail", top_text)
        texts_by_id = dict(zip(self.chunk_ids, self.texts))
        self.assertIn("anticipatory bail", texts_by_id[dense_scored[0][0]])

    def test_retrieval_is_deterministic(self):
        first = self.retrieve_stages()
        second = self.retrieve_stages()
        self.assertEqual(first, second)

    def test_unmatched_query_does_not_return_silently_empty(self):
        bm25_scored, dense_scored, fused = self.retrieve_stages(
            "completely unrelated zoological taxonomy of marine invertebrates"
        )
        self.assertEqual(len(bm25_scored), TOP_K)
        self.assertEqual(len(dense_scored), TOP_K)
        self.assertTrue(fused)


class TestFullCascadeThroughQueryEndpoint(ScaledCascadeTestCase):
    def _client(self) -> TestClient:
        app = create_app()

        def _provider() -> PipelineComponents:
            return self.pipeline

        app.dependency_overrides[routes_module._pipeline_dependency] = _provider
        client = TestClient(app)
        self.addCleanup(client.close)
        return client

    def test_query_endpoint_returns_grounded_top5_over_real_retrieval(self):
        client = self._client()
        response = client.post("/query", json={"question": self.question()})
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()

        self.assertEqual(payload["status"], "ok")
        self.assertLessEqual(len(payload["retrieved_chunk_ids"]), RERANK_TOP_K)
        self.assertTrue(payload["retrieved_chunk_ids"])
        for cid in payload["retrieved_chunk_ids"]:
            self.assertIn(cid, self.known_ids)
        self.assertTrue(payload["citations"])
        for citation in payload["citations"]:
            self.assertIn(citation["chunk_id"], payload["retrieved_chunk_ids"])
            self.assertIn(citation["chunk_id"], self.known_ids)

    def test_production_loader_metadata_order_matches_bm25_and_dense(self):
        # BM25 artifact order, dense mapping order and parquet order must all be identical.
        self.assertEqual(self.pipeline.bm25.chunk_ids, self.chunk_ids)
        self.assertEqual(self.pipeline.dense.chunk_ids, self.chunk_ids)
        self.assertEqual(self.pipeline.chunks["chunk_id"].tolist(), self.chunk_ids)


class TestCitationResolutionIntegration(ScaledCascadeTestCase):
    """Grounding: generated citations resolve against legal_chunks.parquet metadata."""

    def test_context_echo_generator_citations_resolve(self):
        bm25_scored, dense_scored, fused = self.retrieve_stages()
        chunk_text = dict(zip(self.chunk_ids, self.texts))
        context_blocks = [
            {"chunk_id": cid, "text": chunk_text[cid]}
            for cid, _ in fused[:RERANK_TOP_K]
        ]
        from legalrag.generation.prompts import build_rag_prompt, extract_citations

        prompt = build_rag_prompt(self.question(), context_blocks)
        result = self.pipeline.generator.generate("system", prompt)
        cited = extract_citations(result.text)

        self.assertTrue(cited)
        metadata = self.pipeline.chunks.set_index("chunk_id")
        for cid in cited:
            self.assertIn(cid, metadata.index)


if __name__ == "__main__":
    unittest.main()
