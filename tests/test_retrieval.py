"""Tests for BM25 lexical retriever, tokenization, and Reciprocal Rank Fusion (RRF)."""

import os
import pickle
import tempfile
import unittest

from legalrag.retrieval.bm25 import BM25Retriever, tokenize_legal_text
from legalrag.retrieval.fusion import reciprocal_rank_fusion


def _build_bm25_model(corpus_texts):
    """Builds a real BM25Okapi model over tokenized corpus texts, or skips if unavailable."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:  # pragma: no cover - rank_bm25 is a runtime dependency
        return None
    tokenized = [tokenize_legal_text(text) for text in corpus_texts]
    return BM25Okapi(tokenized, k1=1.5, b=0.75)


class TestRetrieval(unittest.TestCase):
    def test_tokenize_legal_text(self):
        text = "Section 138 of the Negotiable Instruments Act, 1881!"
        tokens = tokenize_legal_text(text)
        self.assertIn("section", tokens)
        self.assertIn("138", tokens)
        self.assertIn("negotiable", tokens)
        self.assertIn("instruments", tokens)
        self.assertIn("1881", tokens)

    def test_reciprocal_rank_fusion(self):
        list_bm25 = ["doc_A", "doc_B", "doc_C"]
        list_dense = ["doc_B", "doc_D", "doc_A"]

        # doc_B is rank 2 in bm25 (score: 1/(60+2) = 1/62) and rank 1 in dense (score: 1/(60+1) = 1/61)
        # Total for doc_B = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
        fused = reciprocal_rank_fusion([list_bm25, list_dense], k=60, top_n=3)

        self.assertEqual(len(fused), 3)
        self.assertEqual(fused[0][0], "doc_B")  # doc_B has highest combined rank
        self.assertGreater(fused[0][1], fused[1][1])

    def test_bm25_retriever_mock_or_real(self):
        # Test basic initialization logic without requiring external package if absent
        try:
            corpus = [
                "The respondent committed an offence under Section 138 of NI Act for cheque bounce.",
                "Criminal appeal under Section 302 IPC concerning murder allegations in district court.",
                "Civil writ petition under Article 226 of the Constitution of India challenging service dismissal.",
            ]
            chunk_ids = ["chunk_ni_act", "chunk_ipc_302", "chunk_art_226"]
            retriever = BM25Retriever(chunk_ids=chunk_ids, corpus_texts=corpus)
            results = retriever.search("cheque bounce under section 138", top_k=2)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0][0], "chunk_ni_act")
        except ImportError:
            # Expected when rank_bm25 is not yet installed in local environment
            pass


class TestBM25ArtifactContract(unittest.TestCase):
    """
    Regression coverage for the production bm25.pkl serialization contract.

    Production artifacts are a dict keyed by "bm25" (legacy in-repo writers used "model").
    The original bug loaded neither, producing a retriever with model=None whose first
    search failed at request time instead of failing closed at readiness.
    """

    CORPUS = [
        "Anticipatory bail under Section 438 CrPC may be granted on reasonable apprehension of arrest.",
        "Dishonour of a cheque under Section 138 of the Negotiable Instruments Act attracts penal liability.",
        "A writ petition under Article 226 of the Constitution lies for enforcement of fundamental rights.",
    ]
    CHUNK_IDS = ["chunk_bail_438", "chunk_ni_138", "chunk_art_226"]

    def setUp(self):
        self.model = _build_bm25_model(self.CORPUS)
        if self.model is None:
            self.skipTest("rank-bm25 is not installed")
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.path = os.path.join(self._tmpdir.name, "bm25.pkl")

    def _write_state(self, state):
        with open(self.path, "wb") as handle:
            pickle.dump(state, handle)

    def test_load_production_dict_uses_bm25_key(self):
        """Regression: the production dict format must populate a usable BM25 model."""
        self._write_state({"bm25": self.model, "chunk_ids": list(self.CHUNK_IDS)})

        retriever = BM25Retriever.load(self.path)

        self.assertIsNotNone(retriever.model)
        self.assertEqual(retriever.chunk_ids, self.CHUNK_IDS)
        results = retriever.search("anticipatory bail section 438", top_k=1)
        self.assertEqual(results[0][0], "chunk_bail_438")

    def test_load_legacy_model_key_format(self):
        """Backward compatibility: older in-repo writers used the "model" key."""
        self._write_state({"model": self.model, "chunk_ids": list(self.CHUNK_IDS)})

        retriever = BM25Retriever.load(self.path)

        self.assertIsNotNone(retriever.model)
        self.assertEqual(retriever.search("cheque dishonour 138", top_k=1)[0][0], "chunk_ni_138")

    def test_load_raw_bm25okapi_object(self):
        """The original Kaggle notebook pickled a bare BM25Okapi object with no chunk_ids."""
        self._write_state(self.model)

        retriever = BM25Retriever.load(self.path)

        self.assertIsNotNone(retriever.model)
        self.assertEqual(retriever.chunk_ids, [])
        retriever.chunk_ids = list(self.CHUNK_IDS)
        self.assertEqual(retriever.search("article 226 writ", top_k=1)[0][0], "chunk_art_226")

    def test_load_dict_without_model_key_raises(self):
        """A dict with neither "bm25" nor "model" must fail closed, never model=None."""
        self._write_state({"chunk_ids": list(self.CHUNK_IDS)})
        with self.assertRaises(ValueError):
            BM25Retriever.load(self.path)

    def test_load_non_dict_non_model_state_raises(self):
        self._write_state(["not", "a", "valid", "artifact"])
        with self.assertRaises(TypeError):
            BM25Retriever.load(self.path)

    def test_save_writes_canonical_and_legacy_keys(self):
        retriever = BM25Retriever(chunk_ids=list(self.CHUNK_IDS), corpus_texts=self.CORPUS)
        retriever.save(self.path)

        with open(self.path, "rb") as handle:
            state = pickle.load(handle)

        self.assertIsNot(state["bm25"], None)
        self.assertIs(state["model"], state["bm25"])
        self.assertEqual(state["chunk_ids"], self.CHUNK_IDS)

        # Round-trip through the public loader.
        loaded = BM25Retriever.load(self.path)
        self.assertEqual(loaded.search("anticipatory bail 438", top_k=1)[0][0], "chunk_bail_438")

    def test_save_without_model_raises(self):
        retriever = BM25Retriever(chunk_ids=["only_id"])
        with self.assertRaises(ValueError):
            retriever.save(self.path)

    def test_validate_detects_chunk_id_count_mismatch(self):
        retriever = BM25Retriever(bm25_model=self.model, chunk_ids=["only_one_id"])
        with self.assertRaises(ValueError):
            retriever.validate()

    def test_search_raises_on_inconsistent_lengths(self):
        retriever = BM25Retriever(bm25_model=self.model, chunk_ids=["only_one_id"])
        with self.assertRaises(ValueError):
            retriever.search("anticipatory bail", top_k=1)

    def test_validate_accepts_consistent_retriever(self):
        retriever = BM25Retriever(bm25_model=self.model, chunk_ids=list(self.CHUNK_IDS))
        retriever.validate()
        self.assertEqual(len(retriever.chunk_ids), self.model.corpus_size)


if __name__ == "__main__":
    unittest.main()
