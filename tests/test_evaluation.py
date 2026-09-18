"""Tests for evaluation metrics, evidence matching, and failure taxonomy."""

import unittest
from legalrag.evaluation.grounding import find_gold_chunks
from legalrag.evaluation.metrics import (
    calculate_recall_at_k,
    classify_failure_mode,
    calculate_generation_metrics,
)


class TestEvaluation(unittest.TestCase):
    def test_find_gold_chunks_exact(self):
        chunks = [
            ("chunk_1", "The appellant was sentenced to rigorous imprisonment for a period of ten years."),
            ("chunk_2", "The high court dismissed the petition on the ground of limitation."),
        ]
        supporting_text = "sentenced to rigorous imprisonment for a period of ten years"
        matched = find_gold_chunks(supporting_text, chunks)
        self.assertEqual(matched, ["chunk_1"])

    def test_calculate_recall_at_k(self):
        retrieved = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
        gold = ["doc_3"]

        self.assertEqual(calculate_recall_at_k(retrieved, gold, k=1), 0.0)
        self.assertEqual(calculate_recall_at_k(retrieved, gold, k=3), 1.0)
        self.assertEqual(calculate_recall_at_k(retrieved, gold, k=5), 1.0)

    def test_classify_failure_mode_retrieval_failure(self):
        top_5 = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
        gold = ["doc_99"]  # not in top 5
        eval_rec = {"answer_relevance": 4, "faithfulness": 4, "citation_correctness": 4, "hallucination": False}

        cat = classify_failure_mode(eval_rec, top_5, gold)
        self.assertEqual(cat, "Retrieval Failure")

    def test_classify_failure_mode_success(self):
        top_5 = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
        gold = ["doc_2"]  # in top 5
        eval_rec = {"answer_relevance": 4, "faithfulness": 4, "citation_correctness": 4, "hallucination": False}

        cat = classify_failure_mode(eval_rec, top_5, gold)
        self.assertEqual(cat, "No Major Failure (Success)")

    def test_classify_failure_mode_hallucination(self):
        top_5 = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
        gold = ["doc_2"]
        eval_rec = {"answer_relevance": 4, "faithfulness": 2, "citation_correctness": 4, "hallucination": True}

        cat = classify_failure_mode(eval_rec, top_5, gold)
        self.assertEqual(cat, "Generation Grounding Failure")

    def test_calculate_generation_metrics(self):
        records = [
            {"answer_relevance": 4, "faithfulness": 4, "citation_correctness": 4, "overall_score": 4, "unsupported_claim_rate": 0.0, "hallucination": False},
            {"answer_relevance": 2, "faithfulness": 2, "citation_correctness": 2, "overall_score": 2, "unsupported_claim_rate": 0.5, "hallucination": True},
        ]
        metrics = calculate_generation_metrics(records)

        self.assertEqual(metrics["count"], 2)
        self.assertEqual(metrics["answer_relevance"], 3.0)
        self.assertEqual(metrics["faithfulness"], 3.0)
        self.assertEqual(metrics["unsupported_claim_rate"], 0.25)
        self.assertEqual(metrics["hallucination_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
