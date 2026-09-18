"""
Evidence verification and gold chunk attribution engine.

Matches candidate supporting text against corpus chunks using exact
and multi-scale sliding-window word sequence alignment.
"""

import re
from typing import List, Tuple, Optional


def normalize_text_for_matching(text: str) -> str:
    """Normalizes text by removing non-alphanumeric characters and lowercasing."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def find_gold_chunks(
    supporting_text: str,
    chunks: List[Tuple[str, str]],
    window_sizes: Optional[List[int]] = None,
) -> List[str]:
    """
    Identifies gold chunk IDs that contain the supporting evidence text.

    Uses a hierarchical matching cascade:
    1. Exact normalized substring match of the full supporting passage.
    2. Multi-scale sliding window matching across word sequence windows (default: [60, 40, 25, 15, 10]).

    Args:
        supporting_text: Evidence passage extracted during question formulation.
        chunks: List of (chunk_id, chunk_text) tuples for the target document.
        window_sizes: Optional list of word window sizes for sliding-window search.

    Returns:
        List of matched chunk IDs.
    """
    if window_sizes is None:
        window_sizes = [60, 40, 25, 15, 10]

    norm_supporting = normalize_text_for_matching(supporting_text)
    if not norm_supporting:
        return []

    matched_chunk_ids = []

    # 1. Exact full-passage substring matching
    for chunk_id, chunk_text in chunks:
        norm_chunk = normalize_text_for_matching(chunk_text)
        if norm_supporting in norm_chunk:
            matched_chunk_ids.append(chunk_id)

    if matched_chunk_ids:
        return list(dict.fromkeys(matched_chunk_ids))

    # 2. Sliding window matching
    words = norm_supporting.split()
    total_words = len(words)

    for w_size in window_sizes:
        if total_words < w_size:
            continue

        # Check slices of length w_size
        step = max(1, w_size // 2)
        for i in range(0, total_words - w_size + 1, step):
            phrase = " ".join(words[i : i + w_size])
            for chunk_id, chunk_text in chunks:
                norm_chunk = normalize_text_for_matching(chunk_text)
                if phrase in norm_chunk and chunk_id not in matched_chunk_ids:
                    matched_chunk_ids.append(chunk_id)

        if matched_chunk_ids:
            break

    return list(dict.fromkeys(matched_chunk_ids))
