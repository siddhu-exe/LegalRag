"""
Statistics for LegalRAG evaluation (plan task C3).

Every reported metric gets an interval, and a system is called better than another only when
the paired interval for the difference excludes zero.

Why clusters: several benchmark questions can come from the same judgment, so questions are
not independent. Pass ``groups`` (for example the source CNR) and the bootstrap resamples whole
groups instead of single questions. Without groups, each question is its own group.

Everything is seeded and deterministic. Only numpy is needed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_RESAMPLES = 10_000
_CHUNK = 500  # resamples per vectorised block, to bound memory


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval."""

    estimate: float
    low: float
    high: float
    n: int
    method: str
    confidence: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        return {key: _json_safe(value) for key, value in asdict(self).items()}

    def format(self, digits: int = 3) -> str:
        return f"{self.estimate:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"


@dataclass(frozen=True)
class PairedComparison:
    """Paired difference ``a - b`` over the same questions."""

    difference: Interval
    p_value: float
    p_method: str
    verdict: str  # "a better", "b better" or "no clear difference"
    n: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "difference": self.difference.to_dict(),
            "p_value": _json_safe(self.p_value),
            "p_method": self.p_method,
            "verdict": self.verdict,
            "n": self.n,
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return _json_safe(value.item())
    return value


def _check_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")


def _group_codes(n: int, groups: Optional[Sequence[Any]]) -> np.ndarray:
    if groups is None:
        return np.arange(n)
    if len(groups) != n:
        raise ValueError(f"groups has {len(groups)} items but there are {n} values.")
    _, codes = np.unique(np.asarray([str(g) for g in groups]), return_inverse=True)
    return codes


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a proportion. Use it when items are independent."""
    _check_confidence(confidence)
    if n <= 0:
        return (float("nan"), float("nan"))
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n.")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    low = 0.0 if successes == 0 else max(0.0, centre - half)  # exact at the edges
    high = 1.0 if successes == n else min(1.0, centre + half)
    return (low, high)


def _bootstrap_means(
    group_sums: np.ndarray,
    group_counts: np.ndarray,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Means of ``n_resamples`` resamples of whole groups (ratio of resampled sums)."""
    n_groups = len(group_sums)
    out = np.empty(n_resamples, dtype=float)
    done = 0
    while done < n_resamples:
        block = min(_CHUNK, n_resamples - done)
        picks = rng.integers(0, n_groups, size=(block, n_groups))
        out[done : done + block] = group_sums[picks].sum(axis=1) / group_counts[picks].sum(axis=1)
        done += block
    return out


def _group_totals(values: np.ndarray, codes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_groups = int(codes.max()) + 1
    sums = np.bincount(codes, weights=values, minlength=n_groups)
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    return sums, counts


def bootstrap_ci(
    values: Sequence[float],
    groups: Optional[Sequence[Any]] = None,
    confidence: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval for a mean, resampling whole groups when given."""
    _check_confidence(confidence)
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0, "bootstrap", confidence)
    codes = _group_codes(data.size, groups)
    sums, counts = _group_totals(data, codes)
    rng = np.random.default_rng(seed)
    means = _bootstrap_means(sums, counts, n_resamples, rng)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    method = "cluster_bootstrap" if groups is not None else "bootstrap"
    return Interval(float(data.mean()), float(low), float(high), int(data.size), method, confidence)


def summarize(
    values: Sequence[float],
    groups: Optional[Sequence[Any]] = None,
    confidence: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Interval:
    """Mean with an interval. Binary 0/1 values without clusters use the Wilson interval.

    Everything else uses the (cluster) bootstrap.
    """
    data = np.asarray(values, dtype=float)
    if data.size and groups is None and set(np.unique(data)) <= {0.0, 1.0}:
        low, high = wilson_interval(int(data.sum()), int(data.size), confidence)
        return Interval(float(data.mean()), low, high, int(data.size), "wilson", confidence)
    return bootstrap_ci(data, groups, confidence, n_resamples, seed)


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    groups: Optional[Sequence[Any]] = None,
    confidence: float = 0.95,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> PairedComparison:
    """Paired bootstrap for the mean difference ``a - b`` over the same questions.

    The p-value is two-sided, from the share of resampled differences on each side of zero
    (with +1 smoothing). The verdict follows the interval, not the p-value: ``a better`` only
    when the whole interval is above zero.
    """
    _check_confidence(confidence)
    first, second = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if first.shape != second.shape:
        raise ValueError("a and b must have the same length and be aligned by question.")
    if first.size == 0:
        raise ValueError("Cannot compare empty lists.")
    diff = first - second
    codes = _group_codes(diff.size, groups)
    sums, counts = _group_totals(diff, codes)
    rng = np.random.default_rng(seed)
    means = _bootstrap_means(sums, counts, n_resamples, rng)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])

    at_or_below = (np.sum(means <= 0.0) + 1) / (n_resamples + 1)
    at_or_above = (np.sum(means >= 0.0) + 1) / (n_resamples + 1)
    p_value = float(min(1.0, 2.0 * min(at_or_below, at_or_above)))

    method = "cluster_bootstrap" if groups is not None else "bootstrap"
    interval = Interval(float(diff.mean()), float(low), float(high), int(diff.size), method, confidence)
    return PairedComparison(interval, p_value, method, verdict_from_interval(interval), int(diff.size))


def verdict_from_interval(difference: Interval) -> str:
    """Claim a difference only when the interval for ``a - b`` excludes zero."""
    if difference.low > 0.0:
        return "a better"
    if difference.high < 0.0:
        return "b better"
    return "no clear difference"


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> Dict[str, Any]:
    """Exact McNemar test for paired 0/1 outcomes (for example hit or miss at K).

    Counts only the questions where the two systems disagree. It assumes questions are
    independent, so with clustered questions use it as a cross-check of ``paired_bootstrap``.
    """
    first, second = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    if first.shape != second.shape:
        raise ValueError("a and b must have the same length.")
    a_only = int(np.sum((first == 1) & (second == 0)))
    b_only = int(np.sum((first == 0) & (second == 1)))
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(a_only, b_only)
        tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2.0**discordant
        p_value = min(1.0, 2.0 * tail)
    return {"a_only": a_only, "b_only": b_only, "discordant": discordant, "p_value": p_value}


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values, in the original order.

    Use when comparing several systems or cut-offs, so a lucky difference is not reported.
    """
    count = len(p_values)
    order = sorted(range(count), key=lambda i: p_values[i])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def percent_agreement(a: Sequence[Any], b: Sequence[Any]) -> float:
    if len(a) != len(b):
        raise ValueError("Label lists must have the same length.")
    if not len(a):
        return float("nan")
    return float(np.mean([x == y for x, y in zip(a, b)]))


def cohens_kappa(a: Sequence[Any], b: Sequence[Any], weights: Optional[str] = None) -> float:
    """Cohen's kappa for two raters. ``weights`` may be None, "linear" or "quadratic".

    Weighted kappa treats labels as ordered (sorted order) and suits 1-4 judge scores.
    Returns nan when kappa is undefined (both raters used one identical single label).
    """
    if len(a) != len(b):
        raise ValueError("Label lists must have the same length.")
    if weights not in (None, "linear", "quadratic"):
        raise ValueError("weights must be None, 'linear' or 'quadratic'.")
    if not len(a):
        return float("nan")
    categories = sorted(set(a) | set(b))
    index = {label: i for i, label in enumerate(categories)}
    size = len(categories)
    observed = np.zeros((size, size))
    for x, y in zip(a, b):
        observed[index[x], index[y]] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))

    if weights is None or size == 1:
        penalty = 1.0 - np.eye(size)
    else:
        distance = np.abs(np.subtract.outer(np.arange(size), np.arange(size))) / (size - 1)
        penalty = distance if weights == "linear" else distance**2
    expected_disagreement = float((penalty * expected).sum())
    if expected_disagreement == 0.0:
        return float("nan")
    return 1.0 - float((penalty * observed).sum()) / expected_disagreement


def kappa_interval(
    a: Sequence[Any],
    b: Sequence[Any],
    weights: Optional[str] = None,
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    seed: int = 0,
) -> Interval:
    """Cohen's kappa with a percentile bootstrap interval over items."""
    _check_confidence(confidence)
    estimate = cohens_kappa(a, b, weights)
    n = len(a)
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0, "bootstrap", confidence)
    rng = np.random.default_rng(seed)
    first, second = list(a), list(b)
    draws: List[float] = []
    for _ in range(n_resamples):
        picks = rng.integers(0, n, size=n)
        value = cohens_kappa([first[i] for i in picks], [second[i] for i in picks], weights)
        if not math.isnan(value):
            draws.append(value)
    if not draws:
        return Interval(estimate, float("nan"), float("nan"), n, "bootstrap", confidence)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    return Interval(estimate, float(low), float(high), n, "bootstrap", confidence)
