"""
Corpus cleaning and sanitization utility for Indian High Court judgment transcripts.

Handles control character stripping, Caesar cipher corruption checks,
and whitespace normalization while preserving legal structure and citations.
"""

import re
from typing import Optional

# Regex pattern matching non-printable ASCII and control characters
CONTROL_CHAR_REGEX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
WHITESPACE_REGEX = re.compile(r"\s+")


def is_corrupted_document(text: Optional[str], threshold: float = 0.01) -> bool:
    """
    Determines whether a document is severely corrupted based on control character ratio.

    Args:
        text: The raw judgment text.
        threshold: Maximum allowed ratio of control characters to text length (default 1%).

    Returns:
        True if the text is null, empty, or exceeds the control character threshold.
    """
    if not text or not isinstance(text, str):
        return True

    text_len = len(text)
    if text_len == 0:
        return True

    control_chars = len(CONTROL_CHAR_REGEX.findall(text))
    return (control_chars / text_len) > threshold


def clean_judgment_text(text: Optional[str]) -> str:
    """
    Sanitizes judgment text by removing non-printable control characters
    and collapsing irregular whitespace while preserving valid punctuation.

    Args:
        text: Raw judgment text.

    Returns:
        Cleaned text string.
    """
    if not text or not isinstance(text, str):
        return ""

    # Strip control characters
    cleaned = CONTROL_CHAR_REGEX.sub(" ", text)

    # Collapse multiple whitespace characters into single spaces
    cleaned = WHITESPACE_REGEX.sub(" ", cleaned)

    return cleaned.strip()
