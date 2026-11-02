"""
Benchmark file loading shared by the evaluation scripts (plan task C3).

Canonical fields per question (JSON list, JSONL, or an object with a "questions" list):
    question_id     unique id (required)
    question        the question text (required)
    gold_cnrs       CNRs of the judgments that answer it   -> judgment-level recall
    gold_chunk_ids  chunk ids that hold the evidence       -> chunk-level recall
    source_cnr      judgment the question was written from; used as the gold CNR when
                    gold_cnrs is absent, and as the cluster for the bootstrap
    question_type, bench, year   optional, used for breakdown tables
    supporting_text  optional reference passage; shown to the generation judge for relevance only
At least one of gold_cnrs, gold_chunk_ids or source_cnr is required.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_benchmark(path: Path) -> List[Dict[str, Any]]:
    """Reads and validates a benchmark file. Exits with a clear message if it is malformed."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        items = data["questions"] if isinstance(data, dict) and "questions" in data else data
    except json.JSONDecodeError:
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(items, list) or not items:
        raise SystemExit(f"{path}: expected a non-empty list of questions.")

    seen: set = set()
    cleaned: List[Dict[str, Any]] = []
    for position, raw in enumerate(items):
        if not isinstance(raw, dict):
            raise SystemExit(f"{path}: item {position} is not an object.")
        qid, question = raw.get("question_id"), raw.get("question")
        if qid in (None, "") or not isinstance(question, str) or not question.strip():
            raise SystemExit(f"{path}: item {position} needs 'question_id' and 'question'.")
        qid = str(qid)
        if qid in seen:
            raise SystemExit(f"{path}: duplicate question_id {qid!r}.")
        seen.add(qid)

        gold_cnrs = [str(c) for c in raw.get("gold_cnrs") or []]
        source = raw.get("source_cnr")
        if not gold_cnrs and source:
            gold_cnrs = [str(source)]
        gold_chunks = [str(c) for c in raw.get("gold_chunk_ids") or []]
        if not gold_cnrs and not gold_chunks:
            raise SystemExit(f"{path}: question {qid!r} has no gold_cnrs, gold_chunk_ids or source_cnr.")
        item = dict(raw)
        item.update(
            question_id=qid,
            gold_cnrs=gold_cnrs,
            gold_chunk_ids=gold_chunks,
            group=str(source or (gold_cnrs[0] if gold_cnrs else qid)),
        )
        cleaned.append(item)
    return cleaned


def sample_benchmark(items: List[Dict[str, Any]], size: Optional[int], seed: int) -> List[Dict[str, Any]]:
    """A seeded random subset in the original order. The first rows of a file are not a sample."""
    if size is None or size >= len(items):
        return items
    chosen = set(random.Random(seed).sample(range(len(items)), size))
    return [item for index, item in enumerate(items) if index in chosen]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
