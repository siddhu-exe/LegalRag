"""
Split training data by judgment, and prove no judgment is in two splits (plan task C10).

A reranker learns from (question, passage) pairs. If passages from one judgment are in both training and test
data, the test score measures memory of that judgment, not ranking skill. So the unit of splitting is the
judgment (its CNR), never the question or the passage, and every passage that appears in a split, positive or
negative, must come from a judgment that belongs to that split.

This module needs no PyTorch. ``assert_no_leakage`` raises ``LeakageError`` listing the offending judgments, and
the training script calls it before it loads a model.

Judgments the corpus build removed as near-duplicates of another judgment are mapped to the kept one first
(``canonical``), so two copies of the same judgment cannot land on opposite sides of the split.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

SPLITS = ("train", "val", "test")


class LeakageError(ValueError):
    """A judgment (or passage) appears in more than one split."""


def canonical_cnr(cnr: str, duplicate_links: Optional[Mapping[str, str]] = None) -> str:
    """The CNR the judgment is known by after near-duplicate merging (case and spaces ignored)."""
    key = "".join(str(cnr).split()).upper()
    links = {"".join(k.split()).upper(): "".join(v.split()).upper() for k, v in (duplicate_links or {}).items()}
    seen = set()
    while key in links and key not in seen:  # follow chains, guarding against cycles
        seen.add(key)
        key = links[key]
    return key


@dataclass
class SplitAssignment:
    """Which judgments are in which split."""

    train: Set[str] = field(default_factory=set)
    val: Set[str] = field(default_factory=set)
    test: Set[str] = field(default_factory=set)
    forced_test: Set[str] = field(default_factory=set)
    seed: int = 0
    test_fraction: float = 0.0
    val_fraction: float = 0.0

    def split_of(self, cnr: str) -> Optional[str]:
        for name in SPLITS:
            if cnr in getattr(self, name):
                return name
        return None

    def sha256(self) -> str:
        payload = json.dumps({n: sorted(getattr(self, n)) for n in SPLITS}, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed, "test_fraction": self.test_fraction, "val_fraction": self.val_fraction,
            "counts": {n: len(getattr(self, n)) for n in SPLITS}, "sha256": self.sha256(),
            "forced_test": sorted(self.forced_test), **{n: sorted(getattr(self, n)) for n in SPLITS},
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SplitAssignment":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        assignment = cls(**{n: set(data[n]) for n in SPLITS}, forced_test=set(data.get("forced_test", [])),
                         seed=data.get("seed", 0), test_fraction=data.get("test_fraction", 0.0), val_fraction=data.get("val_fraction", 0.0))
        if data.get("sha256") and data["sha256"] != assignment.sha256():
            raise LeakageError(f"{path} was edited after it was written (its checksum no longer matches).")
        assert_disjoint(assignment)
        return assignment


def assert_disjoint(assignment: SplitAssignment) -> None:
    """No judgment is assigned to two splits."""
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = getattr(assignment, first) & getattr(assignment, second)
        if shared:
            raise LeakageError(f"{len(shared)} judgment(s) are in both {first} and {second}: {sorted(shared)[:5]}")


def split_judgments(
    cnrs: Iterable[str],
    test_fraction: float = 0.15,
    val_fraction: float = 0.10,
    seed: int = 42,
    forced_test: Iterable[str] = (),
) -> SplitAssignment:
    """Assigns each distinct judgment to train, val or test, reproducibly.

    ``forced_test`` judgments (for example the ones behind the evaluation benchmark) are always test, even
    if that makes the test share larger than ``test_fraction``. The rest are shuffled with ``seed`` and cut
    into exact counts (rounded to the nearest judgment; at least one in each split when there are enough).
    """
    if not 0.0 <= test_fraction < 1.0 or not 0.0 <= val_fraction < 1.0 or test_fraction + val_fraction >= 1.0:
        raise ValueError("test_fraction and val_fraction must be in [0, 1) and sum to less than 1.")
    everything = sorted(set(cnrs))
    forced = set(forced_test) & set(everything)
    pool = [c for c in everything if c not in forced]
    random.Random(seed).shuffle(pool)

    total = len(everything)
    want_test = max(0, round(total * test_fraction) - len(forced))
    want_val = round(total * val_fraction)
    if total >= 3:
        if test_fraction > 0 and not forced:
            want_test = max(1, want_test)
        if val_fraction > 0:
            want_val = max(1, want_val)
    want_test = min(want_test, max(0, len(pool) - 1)) if pool else 0
    want_val = min(want_val, max(0, len(pool) - want_test - 1)) if pool else 0

    assignment = SplitAssignment(forced_test=forced, seed=seed, test_fraction=test_fraction, val_fraction=val_fraction)
    assignment.test = forced | set(pool[:want_test])
    assignment.val = set(pool[want_test : want_test + want_val])
    assignment.train = set(pool[want_test + want_val :])
    assert_disjoint(assignment)
    return assignment


@dataclass
class SplitData:
    """Examples assigned to each split, plus what was dropped to keep the splits clean."""

    train: List[Dict[str, Any]] = field(default_factory=list)
    val: List[Dict[str, Any]] = field(default_factory=list)
    test: List[Dict[str, Any]] = field(default_factory=list)
    dropped_negatives: Dict[str, int] = field(default_factory=dict)
    dropped_examples: Dict[str, int] = field(default_factory=dict)


def text_fingerprint(text: str) -> str:
    """Identifies a passage by its words, ignoring case and spacing."""
    return hashlib.sha1(" ".join(text.split()).lower().encode("utf-8")).hexdigest()


def _drop_text_shared_with_higher_priority_splits(data: "SplitData") -> None:
    """Removes passages whose text also appears in a split that must stay clean (test, then val).

    Stock paragraphs recur across judgments. Rather than stop the run, the copy in the lower-priority split is
    removed (the test split is never touched): a negative is dropped, and a positive is dropped with its
    example if that leaves the example without a positive or without a negative. Counted in the report.
    """
    protected: set = set()
    for name in ("test", "val", "train"):
        examples = getattr(data, name)
        kept = []
        for example in examples:
            positives = [p for p in example["positives"] if text_fingerprint(p["text"]) not in protected]
            negatives = [n for n in example["negatives"] if text_fingerprint(n["text"]) not in protected]
            removed_negatives = len(example["negatives"]) - len(negatives)
            if removed_negatives:
                key = f"{name}_negative_text_also_in_a_cleaner_split"
                data.dropped_negatives[key] = data.dropped_negatives.get(key, 0) + removed_negatives
            if len(positives) < len(example["positives"]):
                data.dropped_examples["positive_text_also_in_a_cleaner_split"] = data.dropped_examples.get("positive_text_also_in_a_cleaner_split", 0) + 1
                continue
            if not negatives:
                data.dropped_examples["no_negatives_left"] = data.dropped_examples.get("no_negatives_left", 0) + 1
                continue
            kept.append({**example, "positives": positives, "negatives": negatives})
        setattr(data, name, kept)
        for example in kept:
            protected.update(text_fingerprint(p["text"]) for p in example["positives"])
            protected.update(text_fingerprint(n["text"]) for n in example["negatives"])


def partition_examples(
    examples: Sequence[Mapping[str, Any]],
    assignment: SplitAssignment,
    duplicate_links: Optional[Mapping[str, str]] = None,
) -> SplitData:
    """Puts each example in the split of its judgment and removes any passage that could cross a boundary.

    A negative passage whose judgment belongs to a different split than the example is dropped (it would carry
    text across the boundary). A passage whose exact text also appears in test (or in val, for train) is dropped
    too. An example left with no negatives is dropped and counted. Examples whose judgment is not in the
    assignment are dropped and counted, never guessed into a split.
    """
    data = SplitData()
    for example in examples:
        home = assignment.split_of(canonical_cnr(example["cnr"], duplicate_links))
        if home is None:
            data.dropped_examples["unassigned_judgment"] = data.dropped_examples.get("unassigned_judgment", 0) + 1
            continue
        kept = []
        for negative in example["negatives"]:
            where = assignment.split_of(canonical_cnr(negative["cnr"], duplicate_links))
            if where == home:
                kept.append(negative)
            else:
                key = f"{home}_example_with_{where or 'unassigned'}_negative"
                data.dropped_negatives[key] = data.dropped_negatives.get(key, 0) + 1
        if not kept:
            data.dropped_examples["no_negatives_left"] = data.dropped_examples.get("no_negatives_left", 0) + 1
            continue
        getattr(data, home).append({**example, "negatives": kept})
    _drop_text_shared_with_higher_priority_splits(data)
    return data


def _passages(example: Mapping[str, Any]) -> List[Tuple[str, str, str]]:
    """(chunk_id, cnr, text) for every passage in an example."""
    items = [(p["chunk_id"], example["cnr"], p["text"]) for p in example["positives"]]
    items += [(n["chunk_id"], n["cnr"], n["text"]) for n in example["negatives"]]
    return items


def assert_no_leakage(
    data: SplitData,
    assignment: SplitAssignment,
    duplicate_links: Optional[Mapping[str, str]] = None,
) -> Dict[str, int]:
    """Raises ``LeakageError`` unless every judgment, chunk and passage text is confined to one split.

    Checks (1) the assignment is disjoint, (2) every passage in a split comes from a judgment assigned to
    that split, (3) no chunk id is in two splits, and (4) no identical passage text is in two splits (this
    catches the same text under different ids). Returns counts for the report.
    """
    assert_disjoint(assignment)
    seen_chunk: Dict[str, str] = {}
    seen_text: Dict[str, str] = {}
    judgments: Dict[str, Set[str]] = {n: set() for n in SPLITS}
    for name in SPLITS:
        for example in getattr(data, name):
            for chunk_id, cnr, text in _passages(example):
                canonical = canonical_cnr(cnr, duplicate_links)
                if assignment.split_of(canonical) != name:
                    raise LeakageError(f"{name} contains a passage from judgment {cnr}, which belongs to {assignment.split_of(canonical)}.")
                judgments[name].add(canonical)
                if seen_chunk.setdefault(chunk_id, name) != name:
                    raise LeakageError(f"chunk {chunk_id} appears in both {seen_chunk[chunk_id]} and {name}.")
                fingerprint = text_fingerprint(text)
                if seen_text.setdefault(fingerprint, name) != name:
                    raise LeakageError(f"the same passage text appears in both {seen_text[fingerprint]} and {name} (chunk {chunk_id}).")
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = judgments[first] & judgments[second]
        if shared:
            raise LeakageError(f"judgment(s) {sorted(shared)[:5]} appear in both {first} and {second}.")
    return {name: len(judgments[name]) for name in SPLITS}
