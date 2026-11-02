"""
Retrieval evaluation with confidence intervals and paired comparisons (plan task C3).

Runs the retrieval cascade once per question and scores four systems from that one pass:
``bm25``, ``dense``, ``hybrid_rrf`` (RRF of the two) and ``hybrid_rerank`` (cross-encoder over
the RRF top-50). No LLM is called, so this costs nothing but time. It needs the real artifacts
and about 10.5 GB of RAM on your PC; ``--pipeline local_stub`` runs the same code on stubs.

Benchmark file: a JSON list, or JSONL, with these fields per question:
    question_id     unique id (required)
    question        the question text (required)
    gold_cnrs       list of CNRs of the judgments that answer it   -> judgment-level recall
    gold_chunk_ids  list of chunk ids that hold the evidence       -> chunk-level recall
    source_cnr      judgment the question was written from. Used as the gold CNR when gold_cnrs
                    is absent, and as the cluster for the bootstrap.
    question_type, bench, year   optional, used for breakdown tables
At least one of gold_cnrs, gold_chunk_ids or source_cnr is required per question. The v1
gold_eval.json is not guaranteed to match this schema; convert it first.

Definitions:
    Recall@K (judgment level)  1 if any of the top-K retrieved chunks belongs to a gold judgment.
    Recall@K (chunk level)     1 if any of the top-K retrieved chunks is a gold chunk.
    MRR@10                     1 / rank of the first hit within the top 10, else 0.
Every metric has a 95% interval. Questions from the same source judgment are resampled together
(cluster bootstrap). Each system is compared with ``--baseline`` by a paired bootstrap on the
same questions; p-values are Holm-adjusted across the comparisons at each level, and a system is
called better only when the interval for the difference excludes zero.

Outputs (in --out-dir, default results/eval_v2/):
    retrieval_per_question.jsonl   one line per question, saved as soon as it is done (resumable)
    retrieval_metrics.json         all numbers
    retrieval_metrics.md           tables

Examples:
    python scripts/run_retrieval_eval.py --benchmark data/eval_v2/benchmark.json
    python scripts/run_retrieval_eval.py --benchmark data/eval_v2/benchmark.json --sample 100
    python scripts/run_retrieval_eval.py --benchmark data/eval_v2/benchmark.json --metrics-only
    python scripts/run_retrieval_eval.py --benchmark tests_only.json --pipeline local_stub
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from legalrag.evaluation.runtime import JsonlCheckpoint, ensure_run_config
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from legalrag.evaluation.runtime import JsonlCheckpoint, ensure_run_config  # noqa: E402

from legalrag.evaluation.benchmark import file_sha256, load_benchmark, sample_benchmark  # noqa: E402,F401
from legalrag.evaluation.stats import (  # noqa: E402
    holm_adjust,
    mcnemar_exact,
    paired_bootstrap,
    summarize,
)
from legalrag.retrieval.fusion import reciprocal_rank_fusion  # noqa: E402

SYSTEMS = ("bm25", "dense", "hybrid_rrf", "hybrid_rerank")
DEFAULT_KS = (1, 5, 10, 50)
MRR_CUTOFF = 10
CONTEXT_SIZE = 5  # passages kept per system for the generation evaluation
DEFAULT_OUT_DIR = Path("results/eval_v2")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class ChunkLookup:
    """Fast chunk id -> text/cnr/metadata lookup over the chunk table."""

    FIELDS = ("text", "cnr", "court_code", "decision_date")

    def __init__(self, chunks: Any):
        columns = [c for c in self.FIELDS if c in chunks.columns]
        self._frame = chunks.set_index("chunk_id")[columns]

    def get_many(self, chunk_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        wanted = list(dict.fromkeys(chunk_ids))
        present = self._frame.index.intersection(wanted)
        return self._frame.loc[present].to_dict(orient="index")


def retrieve_all_systems(pipeline: Any, lookup: ChunkLookup, question: str, depth: int = 50) -> Dict[str, Any]:
    """Runs the cascade once and returns the ranked chunk ids of all four systems."""
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()
    bm25_ids = [cid for cid, _ in pipeline.bm25.search(question, top_k=depth)]
    timings["bm25_ms"] = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    dense_ids = [cid for cid, _ in pipeline.dense.search(question, top_k=depth)]
    timings["dense_ms"] = (time.perf_counter() - t0) * 1000.0
    fused_ids = [cid for cid, _ in reciprocal_rank_fusion([bm25_ids, dense_ids], k=60, top_n=depth)]

    meta = lookup.get_many(bm25_ids + dense_ids + fused_ids)
    t0 = time.perf_counter()
    candidates = [(cid, meta[cid].get("text", "")) for cid in fused_ids if cid in meta]
    reranked = pipeline.reranker.rerank(query=question, candidates=candidates, top_k=depth) if candidates else []
    timings["rerank_ms"] = (time.perf_counter() - t0) * 1000.0
    if fused_ids and not reranked:
        # Falling back to the fused order here would make hybrid_rerank equal hybrid_rrf and hide the
        # fault as "no difference". Raise so run_retrieval records a system error for this question.
        raise RuntimeError("reranker returned no results")
    rerank_ids = [cid for cid, _ in reranked]

    ranked = {"bm25": bm25_ids, "dense": dense_ids, "hybrid_rrf": fused_ids, "hybrid_rerank": rerank_ids}
    return {
        "systems": {
            name: {"chunk_ids": ids, "cnrs": [str(meta.get(cid, {}).get("cnr")) for cid in ids]}
            for name, ids in ranked.items()
        },
        "context": {
            name: [
                {
                    "chunk_id": cid,
                    "text": meta.get(cid, {}).get("text", ""),
                    "court_code": meta.get(cid, {}).get("court_code"),
                    "decision_date": meta.get(cid, {}).get("decision_date"),
                }
                for cid in ids[:CONTEXT_SIZE]
            ]
            for name, ids in ranked.items()
        },
        "latency_ms": {k: round(v, 1) for k, v in timings.items()},
    }


def load_pipeline(mode: str) -> Tuple[Any, ChunkLookup]:
    """Loads retrieval components only. No Groq key is needed for retrieval evaluation."""
    if mode == "local_stub":
        from legalrag.api.dependencies import load_stub_pipeline

        pipeline = load_stub_pipeline()
    else:
        import pandas as pd

        from legalrag.api.config import get_settings
        from legalrag.api.dependencies import PipelineComponents, validate_pipeline_components
        from legalrag.retrieval.bm25 import BM25Retriever
        from legalrag.retrieval.dense import DenseRetriever
        from legalrag.retrieval.reranker import CrossEncoderReranker

        settings = get_settings()
        missing = [p for p in (settings.bm25_path, settings.dense_index_path, settings.chunks_path) if not p.is_file()]
        if missing:
            raise SystemExit("Missing artifacts: " + ", ".join(str(p) for p in missing))
        chunks = pd.read_parquet(settings.chunks_path)
        bm25 = BM25Retriever.load(settings.bm25_path)
        if not bm25.chunk_ids:
            bm25.chunk_ids = chunks["chunk_id"].tolist()
        dense = DenseRetriever.load_index(
            index_path=settings.dense_index_path,
            chunk_ids=chunks["chunk_id"].tolist(),
            model_name=settings.embedding_model_name,
        )
        reranker = CrossEncoderReranker(model_name=settings.reranker_model_name)
        pipeline = PipelineComponents(bm25, dense, reranker, chunks, generator=None, environment="production")
        validate_pipeline_components(pipeline)
    return pipeline, ChunkLookup(pipeline.chunks)


def run_retrieval(
    items: Sequence[Dict[str, Any]],
    pipeline: Any,
    lookup: ChunkLookup,
    checkpoint: JsonlCheckpoint,
    depth: int,
    retry_errors: bool = False,
    log: Any = print,
) -> None:
    """Retrieves for every question not yet done, saving each result immediately."""
    retry = ["error"] if retry_errors else []
    todo = [item for item in items if not checkpoint.is_done(item["question_id"], retry)]
    log(f"{len(items) - len(todo)} questions already done, {len(todo)} to run.")
    started = time.perf_counter()
    for number, item in enumerate(todo, start=1):
        try:
            result = retrieve_all_systems(pipeline, lookup, item["question"], depth)
            record = {"item_id": item["question_id"], "status": "ok", **result}
        except Exception as exc:  # noqa: BLE001 - a retrieval fault is a system error, not a miss
            record = {"item_id": item["question_id"], "status": "error", "error_type": type(exc).__name__}
        checkpoint.append(record)
        if number % 10 == 0 or number == len(todo):
            log(f"  {number}/{len(todo)} done, {time.perf_counter() - started:,.0f}s elapsed")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def first_hit_rank(ranked: Sequence[str], gold: Sequence[str]) -> Optional[int]:
    """1-based rank of the first retrieved id that is gold, or None."""
    gold_set = set(gold)
    for position, value in enumerate(ranked, start=1):
        if value in gold_set:
            return position
    return None


def metric_names(ks: Sequence[int]) -> List[str]:
    return [f"recall@{k}" for k in ks] + [f"mrr@{MRR_CUTOFF}"]


def metric_values(rank: Optional[int], ks: Sequence[int]) -> Dict[str, float]:
    values = {f"recall@{k}": 1.0 if rank is not None and rank <= k else 0.0 for k in ks}
    values[f"mrr@{MRR_CUTOFF}"] = 1.0 / rank if rank is not None and rank <= MRR_CUTOFF else 0.0
    return values


def evaluate(
    items: Sequence[Dict[str, Any]],
    records: Dict[str, Dict[str, Any]],
    ks: Sequence[int] = DEFAULT_KS,
    systems: Sequence[str] = SYSTEMS,
    baseline: str = "hybrid_rrf",
    breakdown_fields: Sequence[str] = ("question_type", "bench"),
    breakdown_k: int = 5,
    n_resamples: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Scores every system and compares each with the baseline on the same questions."""
    if baseline not in systems:
        raise ValueError(f"baseline {baseline!r} is not one of {list(systems)}.")
    usable = [i for i in items if records.get(i["question_id"], {}).get("status") == "ok"]
    errors = [i["question_id"] for i in items if records.get(i["question_id"], {}).get("status") == "error"]
    missing = [i["question_id"] for i in items if i["question_id"] not in records]

    levels: Dict[str, Any] = {}
    for level, gold_key, rank_key in (("judgment", "gold_cnrs", "cnrs"), ("chunk", "gold_chunk_ids", "chunk_ids")):
        rows = [i for i in usable if i[gold_key]]
        if not rows:
            continue
        groups = [i["group"] for i in rows]
        per_system: Dict[str, Dict[str, List[float]]] = {}
        for name in systems:
            collected: Dict[str, List[float]] = {m: [] for m in metric_names(ks)}
            for item in rows:
                ranked = records[item["question_id"]]["systems"][name][rank_key]
                for metric, value in metric_values(first_hit_rank(ranked, item[gold_key]), ks).items():
                    collected[metric].append(value)
            per_system[name] = collected

        summary = {
            name: {
                metric: summarize(values, groups, confidence, n_resamples, seed).to_dict()
                for metric, values in metrics.items()
            }
            for name, metrics in per_system.items()
        }

        comparisons: List[Dict[str, Any]] = []
        for name in systems:
            if name == baseline:
                continue
            for metric in metric_names(ks):
                a, b = per_system[name][metric], per_system[baseline][metric]
                paired = paired_bootstrap(a, b, groups, confidence, n_resamples, seed)
                entry = {"system": name, "baseline": baseline, "metric": metric, **paired.to_dict()}
                if metric.startswith("recall@"):
                    entry["mcnemar"] = mcnemar_exact([int(v) for v in a], [int(v) for v in b])
                comparisons.append(entry)
        for entry, adjusted in zip(comparisons, holm_adjust([c["p_value"] for c in comparisons])):
            entry["p_value_holm"] = adjusted
            entry["verdict"] = _verdict_text(entry, name_a=entry["system"], name_b=baseline)

        levels[level] = {
            "n": len(rows),
            "n_clusters": len(set(groups)),
            "systems": summary,
            "comparisons": comparisons,
        }

    primary = "judgment" if "judgment" in levels else ("chunk" if levels else None)
    breakdown: Dict[str, Any] = {}
    if primary:
        gold_key, rank_key = ("gold_cnrs", "cnrs") if primary == "judgment" else ("gold_chunk_ids", "chunk_ids")
        for field in breakdown_fields:
            buckets: Dict[str, List[Dict[str, Any]]] = {}
            for item in usable:
                if item[gold_key] and item.get(field) not in (None, ""):
                    buckets.setdefault(str(item[field]), []).append(item)
            if not buckets:
                continue
            breakdown[field] = {}
            for value, rows in sorted(buckets.items()):
                groups = [r["group"] for r in rows]
                entry = {"n": len(rows), "metric": f"recall@{breakdown_k}", "systems": {}}
                for name in systems:
                    hits = [
                        metric_values(first_hit_rank(records[r["question_id"]]["systems"][name][rank_key], r[gold_key]), [breakdown_k])[f"recall@{breakdown_k}"]
                        for r in rows
                    ]
                    entry["systems"][name] = summarize(hits, groups, confidence, n_resamples, seed).to_dict()
                breakdown[field][value] = entry

    return {
        "n_questions": len(items),
        "n_scored": len(usable),
        "retrieval_errors": errors,
        "not_run": missing,
        "levels": levels,
        "breakdown": {"level": primary, "k": breakdown_k, "fields": breakdown},
        "settings": {
            "ks": list(ks), "mrr_cutoff": MRR_CUTOFF, "baseline": baseline, "systems": list(systems),
            "n_resamples": n_resamples, "seed": seed, "confidence": confidence,
            "interval_methods": "cluster bootstrap by source judgment",
        },
    }


def _verdict_text(entry: Dict[str, Any], name_a: str, name_b: str) -> str:
    """Reads the verdict from the interval; a Holm-adjusted p above 0.05 downgrades a claim."""
    verdict = entry["verdict"]
    if verdict == "no clear difference":
        return verdict
    if entry["p_value_holm"] > 0.05:
        return "no clear difference (not significant after Holm adjustment)"
    return f"{name_a} better" if verdict == "a better" else f"{name_b} better"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(interval: Dict[str, Any]) -> str:
    if interval.get("estimate") is None:
        return "n/a"
    return f"{interval['estimate']:.3f} [{interval['low']:.3f}, {interval['high']:.3f}]"


def render_markdown(result: Dict[str, Any]) -> str:
    settings = result["settings"]
    lines = [
        "# Retrieval evaluation",
        "",
        f"Questions: {result['n_questions']} ({result['n_scored']} scored, "
        f"{len(result['retrieval_errors'])} retrieval errors, {len(result['not_run'])} not run).",
        f"Intervals: {int(settings['confidence'] * 100)}% {settings['interval_methods']}, "
        f"{settings['n_resamples']:,} resamples, seed {settings['seed']}.",
        "",
    ]
    if result["retrieval_errors"]:
        lines += ["> **Warning:** some questions hit system errors and are excluded. Re-run with --retry-errors.", ""]
    names = metric_names(settings["ks"])
    for level, data in result["levels"].items():
        lines += [
            f"## {level.capitalize()}-level recall (n = {data['n']}, {data['n_clusters']} source judgments)",
            "",
            "| System | " + " | ".join(names) + " |",
            "|---|" + "---|" * len(names),
        ]
        for system, metrics in data["systems"].items():
            lines.append(f"| {system} | " + " | ".join(_fmt(metrics[m]) for m in names) + " |")
        lines += [
            "",
            f"### Paired comparison against `{settings['baseline']}` ({level} level)",
            "",
            "Difference = system minus baseline. A claim needs an interval that excludes zero "
            "and a Holm-adjusted p-value of 0.05 or less.",
            "",
            "| System | Metric | Difference [95% CI] | p (Holm) | Verdict |",
            "|---|---|---|---|---|",
        ]
        for c in data["comparisons"]:
            lines.append(
                f"| {c['system']} | {c['metric']} | {_fmt(c['difference'])} | {c['p_value_holm']:.3f} | {c['verdict']} |"
            )
        lines.append("")
    for field, groups in result["breakdown"]["fields"].items():
        k = result["breakdown"]["k"]
        systems = list(next(iter(groups.values()))["systems"])
        lines += [
            f"## Recall@{k} by {field} ({result['breakdown']['level']} level)",
            "",
            "Small groups have wide intervals; read n first.",
            "",
            f"| {field} | n | " + " | ".join(systems) + " |",
            "|---|---|" + "---|" * len(systems),
        ]
        for value, entry in groups.items():
            lines.append(f"| {value} | {entry['n']} | " + " | ".join(_fmt(entry["systems"][s]) for s in systems) + " |")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--pipeline", choices=("production", "local_stub"), default="production")
    parser.add_argument("--depth", type=int, default=50, help="ranked list depth per system")
    parser.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    parser.add_argument("--baseline", choices=SYSTEMS, default="hybrid_rrf")
    parser.add_argument("--breakdown", nargs="*", default=["question_type", "bench"])
    parser.add_argument("--breakdown-k", type=int, default=5)
    parser.add_argument("--sample", type=int, default=None, help="random subset of N questions (seeded)")
    parser.add_argument("--metrics-only", action="store_true", help="skip retrieval; score saved results")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--force-resume", action="store_true")
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if max(args.ks) > args.depth:
        raise SystemExit(f"--depth ({args.depth}) must be at least the largest K ({max(args.ks)}).")
    items = sample_benchmark(load_benchmark(args.benchmark), args.sample, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = JsonlCheckpoint(args.out_dir / "retrieval_per_question.jsonl")

    if not args.metrics_only:
        ensure_run_config(
            args.out_dir / "retrieval_run_config.json",
            {"benchmark_sha256": file_sha256(args.benchmark), "pipeline": args.pipeline, "depth": args.depth},
            force=args.force_resume,
        )
        pipeline, lookup = load_pipeline(args.pipeline)
        run_retrieval(items, pipeline, lookup, checkpoint, args.depth, args.retry_errors)

    result = evaluate(
        items, checkpoint.records, args.ks, SYSTEMS, args.baseline, args.breakdown, args.breakdown_k,
        args.n_resamples, args.seed,
    )
    (args.out_dir / "retrieval_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out_dir / "retrieval_metrics.md").write_text(render_markdown(result), encoding="utf-8")
    print(f"Wrote {args.out_dir / 'retrieval_metrics.json'} and retrieval_metrics.md")


if __name__ == "__main__":
    main()
