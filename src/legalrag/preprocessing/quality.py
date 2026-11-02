"""
Text quality measures for the data quality report (plan task C2).

Pure functions, no I/O. The OCR measures are simple heuristics, not a trained detector and
not validated on this data. The report prints their distributions and a review sample so the
owner can read flagged and unflagged texts and move the thresholds before trusting a share.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional

_TOKEN = re.compile(r"[A-Za-z]+")
_VOWELS = frozenset("aeiouyAEIOUY")
# Five or more single letters separated by single spaces, such as "t h e c o u r t". Enumerations
# like "(a) (b) (c)" and references like "Section 12A" do not match, because the letters touch
# brackets or digits.
_SPACED_LETTERS = re.compile(r"(?<![A-Za-z0-9(])(?:[A-Za-z] ){4,}[A-Za-z](?![A-Za-z0-9)])")
# Characters expected in English legal prose; everything else counts as a symbol.
_PROSE_PUNCTUATION = frozenset(".,;:()[]{}'\"-/&%?!_`’‘“”–—§")


@dataclass(frozen=True)
class OcrThresholds:
    """A text is flagged when any measure goes above its limit. Starting values, not validated."""

    max_non_ascii_share: float = 0.05
    max_replacement_chars: int = 5
    max_spaced_letter_share: float = 0.02
    max_vowelless_share: float = 0.05
    max_symbol_share: float = 0.15

    def to_dict(self) -> Dict[str, float]:
        return dict(vars(self))


def ocr_noise_metrics(text: str) -> Dict[str, float]:
    """Measures that rise when text came from poor scans or broken extraction.

    non_ascii_share         share of characters above U+007F (also high for scripts other than English)
    replacement_chars       count of U+FFFD, the character used for undecodable bytes
    spaced_letter_share     share of characters inside runs of 5+ single letters ("t h e c o u r t"), a sign of split words
    vowelless_share         share of letter words of 5+ letters with no vowel and not all capitals
    symbol_share            share of characters that are not letters, digits, spaces or normal punctuation
    """
    length = len(text)
    if length == 0:
        return {
            "chars": 0, "non_ascii_share": 0.0, "replacement_chars": 0,
            "spaced_letter_share": 0.0, "vowelless_share": 0.0, "symbol_share": 0.0,
        }
    tokens = _TOKEN.findall(text)
    token_count = len(tokens)
    spaced = sum(len(m.group(0)) for m in _SPACED_LETTERS.finditer(text))
    vowelless = sum(1 for t in tokens if len(t) >= 5 and not t.isupper() and not (_VOWELS & set(t)))
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    symbols = sum(1 for ch in text if not (ch.isalnum() or ch.isspace() or ch in _PROSE_PUNCTUATION))
    return {
        "chars": length,
        "non_ascii_share": non_ascii / length,
        "replacement_chars": text.count("�"),
        "spaced_letter_share": spaced / length,
        "vowelless_share": vowelless / token_count if token_count else 0.0,
        "symbol_share": symbols / length,
    }


def ocr_flags(metrics: Dict[str, float], limits: Optional[OcrThresholds] = None) -> Dict[str, bool]:
    """Which individual limits a text breaks."""
    limits = limits or OcrThresholds()
    return {
        "non_ascii": metrics["non_ascii_share"] > limits.max_non_ascii_share,
        "replacement_chars": metrics["replacement_chars"] > limits.max_replacement_chars,
        "spaced_letters": metrics["spaced_letter_share"] > limits.max_spaced_letter_share,
        "vowelless_words": metrics["vowelless_share"] > limits.max_vowelless_share,
        "symbols": metrics["symbol_share"] > limits.max_symbol_share,
    }


def is_ocr_suspect(metrics: Dict[str, float], limits: Optional[OcrThresholds] = None) -> bool:
    return any(ocr_flags(metrics, limits).values())


def has_repeated_4gram(text: str, head_chars: int = 3000, min_count: int = 3) -> bool:
    """True when any 4-word phrase appears ``min_count``+ times in the first ``head_chars``.

    Same measure as scripts/profile_bombay_hc.py, so the share can be compared with the
    54.1% in docs/DATA_PROFILE_BOMBAY.md. Ordinary legal phrases can trigger it, so it is an
    upper limit on extraction repeats.
    """
    words = text[:head_chars].split()
    grams = Counter(" ".join(words[i : i + 4]) for i in range(max(len(words) - 4, 0)))
    return bool(grams) and max(grams.values()) >= min_count


def in_sample(key: str, rate: float, salt: str = "") -> bool:
    """Deterministic pseudo-random inclusion by hash, stable across runs and machines."""
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    digest = hashlib.md5(f"{salt}|{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < rate


YEAR_PERIODS = ("before 2010", "2010-14", "2015-19", "2020-23", "2024+", "unknown")


def year_period(year: Optional[int]) -> str:
    """The same year bands as docs/DATA_PROFILE_BOMBAY.md."""
    if year is None:
        return "unknown"
    if year < 2010:
        return "before 2010"
    if year < 2015:
        return "2010-14"
    if year < 2020:
        return "2015-19"
    if year < 2024:
        return "2020-23"
    return "2024+"
