"""Evaluation, evidence verification, and metric computation for LegalRAG."""

from legalrag.evaluation.grounding import find_gold_chunks
from legalrag.evaluation.metrics import (
    calculate_recall_at_k,
    classify_failure_mode,
    calculate_generation_metrics,
)

__all__ = [
    "find_gold_chunks",
    "calculate_recall_at_k",
    "classify_failure_mode",
    "calculate_generation_metrics",
]
