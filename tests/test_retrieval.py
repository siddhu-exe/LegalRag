"""Tests for BM25 lexical retriever, tokenization, and Reciprocal Rank Fusion (RRF)."""

import unittest
from legalrag.retrieval.bm25 import BM25Retriever, tokenize_legal_text
from legalrag.retrieval.fusion import reciprocal_rank_fusion


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


if __name__ == "__main__":
    unittest.main()
