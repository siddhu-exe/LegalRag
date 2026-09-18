"""
Evaluation metrics computation and failure mode taxonomy classifier for LegalRAG.
"""

from typing import List, Dict, Any, Set, Optional


def calculate_recall_at_k(
    retrieved_chunk_ids: List[str],
    gold_chunk_ids: List[str],
    k: int,
) -> float:
    """
    Computes binary Recall@K: whether at least one gold chunk ID appears
    within the top-K retrieved chunk IDs.

    Args:
        retrieved_chunk_ids: Ranked list of retrieved chunk IDs.
        gold_chunk_ids: Ground-truth gold chunk IDs containing evidence.
        k: Cutoff rank.

    Returns:
        1.0 if any gold chunk is present in top-K, 0.0 otherwise.
    """
    if not gold_chunk_ids:
        return 0.0

    top_k_retrieved = set(retrieved_chunk_ids[:k])
    gold_set = set(gold_chunk_ids)

    return 1.0 if bool(top_k_retrieved.intersection(gold_set)) else 0.0


def classify_failure_mode(
    eval_record: Dict[str, Any],
    top_5_retrieved: List[str],
    gold_chunks: List[str],
) -> str:
    """
    Categorizes an evaluation query into the failure taxonomy:
    1. Retrieval Failure (target gold chunks omitted from top-5 context)
    2. Generation Grounding Failure (context present, but model hallucinated ungrounded facts)
    3. Low Answer Relevance (context present, but answer failed to directly answer query)
    4. Context Selection / Citation (context present, but improper chunk citation/attribution)
    5. No Major Failure (Success)

    Args:
        eval_record: Dict containing judge scores (answer_relevance, faithfulness, citation_correctness, hallucination).
        top_5_retrieved: Top 5 retrieved chunk IDs provided to LLM.
        gold_chunks: Gold chunk IDs containing answer evidence.

    Returns:
        Category name string.
    """
    gold_set = set(gold_chunks)
    retrieved_set = set(top_5_retrieved[:5])
    retrieval_success = bool(retrieved_set.intersection(gold_set))

    if not retrieval_success:
        return "Retrieval Failure"

    hallucination = eval_record.get("hallucination", False)
    faithfulness = eval_record.get("faithfulness", 4)
    relevance = eval_record.get("answer_relevance", 4)
    citation = eval_record.get("citation_correctness", 4)

    if hallucination or faithfulness <= 2:
        return "Generation Grounding Failure"
    elif relevance <= 2:
        return "Low Answer Relevance"
    elif citation <= 2:
        return "Context Selection / Citation"
    else:
        return "No Major Failure (Success)"


def calculate_generation_metrics(evaluation_records: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregates judge scores across all evaluation records.

    Args:
        evaluation_records: List of judge evaluation dicts.

    Returns:
        Dictionary of mean scores and proportions.
    """
    if not evaluation_records:
        return {}

    n = len(evaluation_records)
    total_relevance = sum(r.get("answer_relevance", 0) for r in evaluation_records)
    total_faithfulness = sum(r.get("faithfulness", 0) for r in evaluation_records)
    total_citation = sum(r.get("citation_correctness", 0) for r in evaluation_records)
    total_overall = sum(r.get("overall_score", 0) for r in evaluation_records)
    total_unsupported = sum(r.get("unsupported_claim_rate", 0.0) for r in evaluation_records)
    total_hallucinations = sum(1 for r in evaluation_records if r.get("hallucination", False))

    return {
        "count": n,
        "answer_relevance": total_relevance / n,
        "faithfulness": total_faithfulness / n,
        "citation_correctness": total_citation / n,
        "overall_score": total_overall / n,
        "unsupported_claim_rate": total_unsupported / n,
        "hallucination_rate": total_hallucinations / n,
    }
