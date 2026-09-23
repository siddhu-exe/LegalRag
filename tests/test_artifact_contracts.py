"""
Artifact/model serialization contract tests for the production retrieval stack.

Covers the DenseRetriever FAISS/chunk-id contract, the production pipeline loader's
fail-closed validation, and a complete retrieval path (real BM25 + real FAISS + real RRF)
driven by deterministic fake embedding/reranker/generation components. No production
models, downloads, or paid APIs are touched.
"""

import os
import pickle
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from legalrag.api.config import Settings
from legalrag.api.dependencies import (
    PipelineInitializationError,
    get_pipeline,
    load_production_pipeline,
    reset_pipeline_cache,
    validate_pipeline_components,
)
from legalrag.api.dependencies import PipelineComponents
from legalrag.api.routes import query_endpoint
from legalrag.api.schemas import QueryRequest
from legalrag.generation.client import GenerationResult
from legalrag.retrieval.bm25 import BM25Retriever, tokenize_legal_text
from legalrag.retrieval.dense import DEFAULT_QUERY_INSTRUCTION, DenseRetriever
from legalrag.retrieval.fusion import reciprocal_rank_fusion
from legalrag.retrieval.reranker import CrossEncoderReranker

_CONTEXT_ID_RE = re.compile(r"\[Chunk ID:\s*([^\]|\s][^\]|]*?)\s*(?:\||\])")

CORPUS = [
    (
        "PHHC010000012020_chunk_0",
        "Anticipatory bail under Section 438 CrPC may be granted when the applicant apprehends "
        "arrest in a non-bailable offence and the court is satisfied that custodial interrogation "
        "is not required.",
    ),
    (
        "PHHC010000012020_chunk_1",
        "The conditions under Section 438(2) CrPC include attendance before the investigating "
        "officer and non-interference with the investigation.",
    ),
    (
        "DLHC020000022021_chunk_0",
        "Section 138 of the Negotiable Instruments Act, 1881 makes the dishonour of a cheque for "
        "insufficiency of funds a penal offence after statutory notice within 30 days.",
    ),
    (
        "DLHC020000022021_chunk_1",
        "The complainant must prove the debt or liability, presentment of the cheque, and service "
        "of the statutory demand notice to sustain a Section 138 complaint.",
    ),
    (
        "KHC030000032022_chunk_0",
        "A writ petition under Article 226 of the Constitution lies for the enforcement of "
        "fundamental rights and for any other purpose against the State.",
    ),
    (
        "KHC030000032022_chunk_1",
        "The High Court may decline relief under Article 226 where an alternative efficacious "
        "statutory remedy exists.",
    ),
]


# Deterministic token -> dimension map built from the test corpus vocabulary.
_VOCAB = {
    token: index
    for index, token in enumerate(
        sorted({token for _chunk_id, text in CORPUS for token in tokenize_legal_text(text)})
    )
}


class FakeEmbedder:
    """
    Deterministic vocabulary-overlap embedder that mimics SentenceTransformer.encode
    without downloading any model weights.
    """

    def __init__(self, dim=None):
        self.dim = int(dim) if dim else len(_VOCAB)

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
                index = _VOCAB.get(token)
                if index is not None and index < self.dim:
                    vec[index] += 1.0
            norm = float(np.linalg.norm(vec))
            if normalize_embeddings and norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.vstack(vectors).astype(np.float32)


class ContextEchoGenerator:
    """Generation stub that cites the chunk IDs actually present in its prompt."""

    def __init__(self):
        self.model_name = "context-echo"

    def generate(self, system_prompt, user_prompt, temperature=0.0, max_tokens=1024):
        cited = list(dict.fromkeys(_CONTEXT_ID_RE.findall(user_prompt)))[:2]
        text = "Held, based on the retrieved judgments " + " ".join(
            f"[Chunk ID: {cid}]" for cid in cited
        )
        return GenerationResult(text=text, status="success", model_name=self.model_name)


def build_chunks_dataframe():
    rows = []
    for chunk_id, text in CORPUS:
        cnr = chunk_id.rsplit("_chunk_", 1)[0]
        rows.append(
            {
                "chunk_id": chunk_id,
                "cnr": cnr,
                "chunk_index": int(chunk_id.rsplit("_", 1)[1]),
                "text": text,
                "court_code": "01",
                "decision_date": "2020-01-15",
                "title": f"{cnr} v. State",
            }
        )
    return pd.DataFrame(rows)


def build_bm25_model(texts):
    from rank_bm25 import BM25Okapi

    return BM25Okapi([tokenize_legal_text(t) for t in texts], k1=1.5, b=0.75)


def _write_artifacts(artifact_dir: Path, bm25_state, chunk_ids, dim=None):
    """Writes bm25.pkl / dense.index / legal_chunks.parquet in production layout."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with open(artifact_dir / "bm25.pkl", "wb") as handle:
        pickle.dump(bm25_state, handle)

    dense = DenseRetriever(model_name="fake-bge")
    dense._model = FakeEmbedder(dim=dim)
    texts = build_chunks_dataframe()["text"].tolist()
    embeddings = dense.encode(texts, normalize=True)
    dense.build_index(embeddings.astype(np.float32), chunk_ids)
    dense.save_index(artifact_dir / "dense.index")

    build_chunks_dataframe().to_parquet(artifact_dir / "legal_chunks.parquet", index=False)


def production_settings(artifact_dir: Path, warmup: bool = False) -> Settings:
    return Settings(
        environment="production",
        groq_api_key="gsk_dummy_key_for_test",
        artifact_dir=artifact_dir,
        model_warmup_enabled=warmup,
    )


class TestDenseArtifactContract(unittest.TestCase):
    """dense.index and legal_chunks.parquet must agree on vector/chunk-id counts."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.artifact_dir = Path(self._tmpdir.name)
        self.chunk_ids = [cid for cid, _ in CORPUS]

    def _save_index(self, count: int, dim=None) -> Path:
        retriever = DenseRetriever(model_name="fake-bge")
        retriever._model = FakeEmbedder(dim=dim)
        texts = [text for _, text in CORPUS][:count]
        embeddings = retriever.encode(texts, normalize=True)
        retriever.build_index(embeddings, self.chunk_ids[:count])
        path = self.artifact_dir / "dense.index"
        retriever.save_index(path)
        return path

    def test_load_index_and_search_resolves_chunk_ids(self):
        path = self._save_index(len(self.chunk_ids))
        retriever = DenseRetriever.load_index(path, chunk_ids=self.chunk_ids)
        retriever._model = FakeEmbedder()

        retriever.validate()
        results = retriever.search("anticipatory bail section 438 crpc", top_k=3)

        self.assertTrue(results)
        self.assertTrue(all(cid in self.chunk_ids for cid, _ in results))
        self.assertEqual(results[0][0], "PHHC010000012020_chunk_0")
        for _cid, score in results:
            self.assertIsInstance(score, float)

    def test_load_index_rejects_vector_count_mismatch(self):
        path = self._save_index(len(self.chunk_ids) - 2)
        with self.assertRaises(ValueError):
            DenseRetriever.load_index(path, chunk_ids=self.chunk_ids)

    def test_validate_rejects_unloaded_index(self):
        with self.assertRaises(ValueError):
            DenseRetriever(chunk_ids=["a", "b"]).validate()

    def test_warmup_accepts_matching_dimension(self):
        path = self._save_index(len(self.chunk_ids), dim=32)
        retriever = DenseRetriever.load_index(path, chunk_ids=self.chunk_ids)
        retriever._model = FakeEmbedder(dim=32)
        retriever.warmup()

    def test_warmup_detects_embedding_dimension_mismatch(self):
        path = self._save_index(len(self.chunk_ids), dim=32)
        retriever = DenseRetriever.load_index(path, chunk_ids=self.chunk_ids)
        retriever._model = FakeEmbedder(dim=16)
        with self.assertRaises(ValueError):
            retriever.warmup()


class _RecordingEmbedder:
    """Captures the exact texts passed to ``encode`` and returns zero vectors."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.encoded_texts: list = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts, **kwargs) -> np.ndarray:
        self.encoded_texts.extend(list(texts))
        return np.zeros((len(list(texts)), self.dim), dtype=np.float32)


class _NullIndex:
    """Minimal FAISS-like index whose ``search`` returns no hits."""

    def __init__(self, dim: int = 8):
        self.d = dim
        self.ntotal = 1

    def search(self, vectors, top_k):
        return np.zeros((1, top_k), dtype=np.float32), -np.ones((1, top_k), dtype=np.int64)


class TestDenseQueryInstructionContract(unittest.TestCase):
    """
    Regression: the frozen ``dense.index`` experiment encoded BGE queries with the
    retrieval instruction while embedding documents without it. ``search()`` must apply the
    instruction to the query only; ``encode()`` (used for documents) must stay untouched.
    """

    def _retriever(self, query_instruction=None):
        embedder = _RecordingEmbedder()
        retriever = (
            DenseRetriever(chunk_ids=["c0"], index=_NullIndex())
            if query_instruction is None
            else DenseRetriever(
                chunk_ids=["c0"], index=_NullIndex(), query_instruction=query_instruction
            )
        )
        retriever._model = embedder
        return retriever, embedder

    def test_default_instruction_prefixes_query(self):
        retriever, embedder = self._retriever()
        retriever.search("essential elements of negligence", top_k=1)
        self.assertEqual(
            embedder.encoded_texts,
            [DEFAULT_QUERY_INSTRUCTION + "essential elements of negligence"],
        )

    def test_document_encoding_is_not_prefixed(self):
        retriever, embedder = self._retriever()
        retriever.encode(["a judgment passage"], normalize=True)
        self.assertEqual(embedder.encoded_texts, ["a judgment passage"])

    def test_instruction_is_idempotent(self):
        retriever, embedder = self._retriever()
        already = DEFAULT_QUERY_INSTRUCTION + "anticipatory bail"
        retriever.search(already, top_k=1)
        self.assertEqual(embedder.encoded_texts, [already])

    def test_empty_instruction_disables_prefixing(self):
        retriever, embedder = self._retriever(query_instruction="")
        retriever.search("raw query text", top_k=1)
        self.assertEqual(embedder.encoded_texts, ["raw query text"])


class TestRerankerContract(unittest.TestCase):
    def test_validate_raises_when_sentence_transformers_missing(self):
        from unittest.mock import patch

        reranker = CrossEncoderReranker()
        with patch("legalrag.retrieval.reranker.CrossEncoder", None):
            with self.assertRaises(ImportError):
                reranker.validate()

    def test_validate_rejects_empty_model_name(self):
        reranker = CrossEncoderReranker(model_name="")
        with self.assertRaises(ValueError):
            reranker.validate()


class TestProductionPipelineLoader(unittest.TestCase):
    """load_production_pipeline must fail closed on any unusable retrieval component."""

    def setUp(self):
        reset_pipeline_cache()
        self.addCleanup(reset_pipeline_cache)
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.artifact_dir = Path(self._tmpdir.name)
        self.chunk_ids = [cid for cid, _ in CORPUS]
        self.model = build_bm25_model([text for _, text in CORPUS])

    def _write_valid(self, bm25_state=None):
        _write_artifacts(
            self.artifact_dir,
            bm25_state
            if bm25_state is not None
            else {"bm25": self.model, "chunk_ids": self.chunk_ids},
            self.chunk_ids,
        )

    def test_loads_production_bm25_dict_format_with_real_model(self):
        self._write_valid()
        pipeline = load_production_pipeline(production_settings(self.artifact_dir))

        self.assertIsNotNone(pipeline.bm25.model)
        self.assertEqual(pipeline.bm25.chunk_ids, self.chunk_ids)
        self.assertEqual(pipeline.dense.index.ntotal, len(self.chunk_ids))
        pipeline.bm25.validate()
        pipeline.dense.validate()

    def test_legacy_bm25_artifact_without_chunk_ids_is_backfilled(self):
        self._write_valid(bm25_state={"bm25": self.model})
        pipeline = load_production_pipeline(production_settings(self.artifact_dir))
        self.assertEqual(pipeline.bm25.chunk_ids, self.chunk_ids)
        pipeline.bm25.validate()

    def test_fails_closed_when_bm25_model_missing(self):
        """Reproduces the original production failure: a bm25.pkl with no usable model."""
        self._write_valid(bm25_state={"chunk_ids": self.chunk_ids})
        with self.assertRaises(PipelineInitializationError):
            get_pipeline(production_settings(self.artifact_dir))

    def test_fails_closed_when_dense_index_count_mismatches(self):
        self._write_valid()
        # Rewrite the FAISS index with fewer vectors than there are chunk_ids.
        dense = DenseRetriever(model_name="fake-bge")
        dense._model = FakeEmbedder()
        embeddings = dense.encode([text for _, text in CORPUS][:3], normalize=True)
        dense.build_index(embeddings, self.chunk_ids[:3])
        dense.save_index(self.artifact_dir / "dense.index")

        with self.assertRaises(PipelineInitializationError):
            get_pipeline(production_settings(self.artifact_dir))

    def test_fails_closed_when_retrieval_ids_do_not_resolve(self):
        self._write_valid(bm25_state={"bm25": self.model, "chunk_ids": ["ghost_chunk_0"] * 6})
        with self.assertRaises(PipelineInitializationError):
            get_pipeline(production_settings(self.artifact_dir))

    def test_validate_pipeline_components_rejects_unloaded_bm25_model(self):
        pipeline = PipelineComponents(
            bm25=BM25Retriever(chunk_ids=self.chunk_ids),
            dense=DenseRetriever(index=object(), chunk_ids=self.chunk_ids),
            reranker=CrossEncoderReranker(),
            chunks=build_chunks_dataframe(),
            generator=object(),
            environment="production",
        )
        with self.assertRaises(ValueError):
            validate_pipeline_components(pipeline)


class TestFullRetrievalPath(unittest.TestCase):
    """
    End-to-end retrieval cascade over real artifacts:
    BM25 top-50 + Dense top-50 -> RRF k=60 -> CrossEncoder-style top-5.
    """

    def setUp(self):
        reset_pipeline_cache()
        self.addCleanup(reset_pipeline_cache)
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.artifact_dir = Path(self._tmpdir.name)
        self.chunk_ids = [cid for cid, _ in CORPUS]
        self.model = build_bm25_model([text for _, text in CORPUS])
        _write_artifacts(
            self.artifact_dir,
            {"bm25": self.model, "chunk_ids": self.chunk_ids},
            self.chunk_ids,
        )

    def _build_pipeline(self) -> PipelineComponents:
        pipeline = load_production_pipeline(production_settings(self.artifact_dir))
        # Inject a deterministic embedder so no model weights are downloaded.
        pipeline.dense._model = FakeEmbedder()
        return pipeline

    def test_bm25_dense_rrf_rerank_returns_valid_ids_and_scores(self):
        from legalrag.api.dependencies import StubCrossEncoderReranker

        pipeline = self._build_pipeline()
        query = "Anticipatory bail under Section 438 CrPC apprehension of arrest"

        bm25_scored = pipeline.bm25.search(query, top_k=50)
        dense_scored = pipeline.dense.search(query, top_k=50)
        self.assertEqual(bm25_scored[0][0], "PHHC010000012020_chunk_0")
        self.assertEqual(dense_scored[0][0], "PHHC010000012020_chunk_0")
        for _cid, score in bm25_scored + dense_scored:
            self.assertIsInstance(score, float)

        fused = reciprocal_rank_fusion(
            [[cid for cid, _ in bm25_scored], [cid for cid, _ in dense_scored]], k=60, top_n=50
        )
        self.assertEqual(fused[0][0], "PHHC010000012020_chunk_0")
        self.assertTrue(all(cid in self.chunk_ids for cid, _ in fused))

        pipeline.reranker = StubCrossEncoderReranker()
        chunk_dict = pipeline.chunks.set_index("chunk_id")["text"].to_dict()
        reranked = pipeline.reranker.rerank(
            query, [(cid, chunk_dict[cid]) for cid, _ in fused], top_k=5
        )
        self.assertEqual(len(reranked), 5)
        self.assertTrue(all(cid in self.chunk_ids for cid, _ in reranked))

    def test_query_endpoint_over_real_retrieval_returns_grounded_citations(self):
        from legalrag.api.dependencies import StubCrossEncoderReranker

        pipeline = self._build_pipeline()
        pipeline.generator = ContextEchoGenerator()
        pipeline.reranker = StubCrossEncoderReranker()

        response = query_endpoint(
            QueryRequest.model_validate(
                {"question": "Anticipatory bail under Section 438 CrPC apprehension of arrest"}
            ),
            pipeline,
        )

        self.assertEqual(response.status, "ok")
        self.assertGreater(len(response.retrieved_chunk_ids), 0)
        self.assertLessEqual(len(response.retrieved_chunk_ids), 5)
        self.assertEqual(response.retrieved_chunk_ids[0], "PHHC010000012020_chunk_0")
        self.assertTrue(all(cid in self.chunk_ids for cid in response.retrieved_chunk_ids))
        self.assertGreater(len(response.citations), 0)
        self.assertTrue(all(c.chunk_id in response.retrieved_chunk_ids for c in response.citations))


if __name__ == "__main__":
    unittest.main()
