"""
Reciprocal Rank Fusion (RRF) for combining lexical and dense candidate rankings.
"""

from collections import defaultdict
from typing import Dict, List, Tuple


def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = 60,
    top_n: int = 50,
) -> List[Tuple[str, float]]:
    """
    Combines multiple ranked candidate lists into a single consolidated ranking
    using Reciprocal Rank Fusion:
        Score_RRF(doc) = sum(1 / (k + rank_i(doc)))

    Args:
        ranked_lists: A list of ranked candidate ID lists (e.g. [bm25_top50, dense_top50]).
        k: Smoothing constant (standard default: 60).
        top_n: Maximum number of candidate items to return.

    Returns:
        List of (doc_id, rrf_score) tuples sorted descending by fused score.
    """
    rrf_scores: Dict[str, float] = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            rrf_scores[doc_id] += 1.0 / (k + (rank + 1))

    # Sort candidates by combined RRF score descending
    sorted_candidates = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return sorted_candidates[:top_n]
