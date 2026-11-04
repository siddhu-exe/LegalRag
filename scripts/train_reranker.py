"""
Fine-tunes the cross-encoder reranker with a split by judgment, and exports it to ONNX (plan task C10).

Run on your PC (the RTX A1000 with 8 GB should handle the default 512-token, 4-group batches in fp16; this has
not been measured). Needs the full requirements.txt (torch, transformers).

Training data is a JSONL file of questions with positive and negative passages; see
``legalrag/training/reranker.py`` for the exact shape. Every passage carries its judgment's CNR so the data can
be split by judgment. The judgments behind your evaluation benchmark (--benchmark) are forced into the test
split, so the benchmark is only ever test data. Before any model is loaded the script re-checks that no
judgment, chunk or passage text is in two splits and stops if one is.

    python scripts/train_reranker.py train --examples data/eval_v2/rerank_train.jsonl \\
        --benchmark data/eval_v2/benchmark.json --duplicate-links artifacts/v2/corpus/duplicate_links.parquet \\
        --out-dir artifacts/v2/reranker --dry-run                     # splits and counts only, nothing written
    python scripts/train_reranker.py train --examples ... --benchmark ... --out-dir artifacts/v2/reranker --export-onnx
    python scripts/train_reranker.py export --model artifacts/v2/reranker/best --out-dir artifacts/v2/onnx/reranker

Then serve it with RERANKER_BACKEND=onnx, or point RERANKER_MODEL_NAME at the saved folder, and run
scripts/run_retrieval_eval.py to compare it with the current reranker using the paired test. Keep it only if
that test supports it (docs/IMPROVEMENT_PLAN.md, rule 2). The report's own comparison is a first look.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

try:
    from legalrag.training.reranker import TrainingConfig, benchmark_judgments, load_examples, sha256_file
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from legalrag.training.reranker import TrainingConfig, benchmark_judgments, load_examples, sha256_file  # noqa: E402

from legalrag.training.splits import (  # noqa: E402
    SPLITS,
    SplitAssignment,
    assert_no_leakage,
    canonical_cnr,
    partition_examples,
    split_judgments,
)

STAGING = ".training_staging"
OUTPUTS = ("best", "splits.json", "train_report.json", "onnx")


def assert_v2_dir(path: Path) -> None:
    if "v2" not in Path(path).resolve().parts and "v2" not in Path(path).parts:
        raise SystemExit(f"Refusing to write to {path}: the output directory must contain a 'v2' part (for example artifacts/v2/reranker).")


def read_duplicate_links(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    import pyarrow.parquet as pq

    table = pq.read_table(str(path), columns=["duplicate_cnr", "kept_cnr"]).to_pylist()
    return {row["duplicate_cnr"]: row["kept_cnr"] for row in table}


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    """Loads, splits by judgment, drops cross-boundary passages and proves the splits are clean."""
    examples = load_examples(args.examples)
    links = read_duplicate_links(args.duplicate_links)
    forced = [canonical_cnr(c, links) for c in benchmark_judgments(args.benchmark)] if args.benchmark else []

    if args.splits:
        assignment = SplitAssignment.load(args.splits)
        missing_forced = [c for c in forced if assignment.split_of(c) in ("train", "val")]
        if missing_forced:
            raise SystemExit(f"{args.splits} puts {len(missing_forced)} benchmark judgment(s) in train or val: {missing_forced[:5]}. Re-split.")
    else:
        judgments = {canonical_cnr(e["cnr"], links) for e in examples} | {canonical_cnr(n["cnr"], links) for e in examples for n in e["negatives"]}
        assignment = split_judgments(judgments, args.test_fraction, args.val_fraction, args.seed, forced_test=forced)
    data = partition_examples(examples, assignment, links)
    judgments_per_split = assert_no_leakage(data, assignment, links)  # raises LeakageError, which stops the run
    return {
        "examples": examples, "assignment": assignment, "data": data, "links": links,
        "summary": {
            "examples_file": str(args.examples), "examples_sha256": sha256_file(args.examples), "examples_read": len(examples),
            "benchmark_judgments_forced_to_test": len(assignment.forced_test),
            "judgments": {n: len(getattr(assignment, n)) for n in SPLITS},
            "judgments_with_examples": judgments_per_split,
            "examples_per_split": {n: len(getattr(data, n)) for n in SPLITS},
            "dropped_negatives": data.dropped_negatives, "dropped_examples": data.dropped_examples,
            "split_sha256": assignment.sha256(), "leakage_check": "passed",
        },
    }


def config_from(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        base_model=args.base_model, epochs=args.epochs, batch_groups=args.batch_groups, grad_accum=args.grad_accum,
        negatives_per_positive=args.negatives, learning_rate=args.lr, max_length=args.max_length, fp16=not args.no_fp16,
        seed=args.seed, max_train_groups=args.max_train_groups, warmup_ratio=args.warmup_ratio,
    )


def run_train(args: argparse.Namespace, log: Any = print) -> Dict[str, Any]:
    assert_v2_dir(args.out_dir)
    prepared = prepare(args)
    summary = prepared["summary"]
    log(json.dumps(summary, indent=2))
    if summary["examples_per_split"]["train"] == 0:
        raise SystemExit("No training examples remain after the split by judgment.")
    if args.dry_run:
        log("Dry run: nothing was written and no model was loaded.")
        return {"dry_run": True, **summary}

    existing = [name for name in OUTPUTS if (args.out_dir / name).exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"{args.out_dir} already holds {', '.join(existing)}. Use --overwrite or a new --out-dir.")
    from legalrag.training.reranker import train

    stage = args.out_dir / STAGING
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True, exist_ok=True)
    try:
        report = train(prepared["data"], stage, config_from(args), args.device, log)
        prepared["assignment"].save(stage / "splits.json")
        report.update(split=summary, command=list(sys.argv), versions=_versions())
        if args.benchmark:
            report["benchmark_sha256"] = sha256_file(args.benchmark)
        if args.export_onnx:
            from legalrag.retrieval.onnx_export import export_reranker

            report["onnx_verification"] = export_reranker(str(stage / "best"), stage / "onnx", max_length=args.max_length)
        report["best_model_dir"] = str(args.out_dir / "best")
        (stage / "train_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        for name in OUTPUTS:  # the report moves last, so a crash never leaves a report beside missing files
            target = args.out_dir / name
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
        for name in [n for n in OUTPUTS if n != "train_report.json"] + ["train_report.json"]:
            if (stage / name).exists():
                (stage / name).replace(args.out_dir / name)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return report


def _versions() -> Mapping[str, str]:
    versions = {}
    for name in ("torch", "transformers", "numpy"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            versions[name] = "unknown"
    return versions


def run_export(args: argparse.Namespace) -> Dict[str, Any]:
    assert_v2_dir(args.out_dir)
    from legalrag.retrieval.onnx_export import export_reranker

    report = export_reranker(str(args.model), args.out_dir, max_length=args.max_length)
    print(json.dumps({"exported_to": str(args.out_dir), "verification": report}, indent=2))
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="split by judgment, fine-tune, evaluate, optionally export")
    train.add_argument("--examples", type=Path, required=True)
    train.add_argument("--out-dir", type=Path, default=Path("artifacts/v2/reranker"))
    train.add_argument("--benchmark", type=Path, default=None, help="benchmark file; its judgments become test-only")
    train.add_argument("--duplicate-links", type=Path, default=None, help="duplicate_links.parquet from the corpus build")
    train.add_argument("--splits", type=Path, default=None, help="reuse a saved splits.json instead of splitting again")
    train.add_argument("--test-fraction", type=float, default=0.15)
    train.add_argument("--val-fraction", type=float, default=0.10)
    train.add_argument("--base-model", default=TrainingConfig.base_model)
    train.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    train.add_argument("--batch-groups", type=int, default=TrainingConfig.batch_groups)
    train.add_argument("--grad-accum", type=int, default=TrainingConfig.grad_accum)
    train.add_argument("--negatives", type=int, default=TrainingConfig.negatives_per_positive)
    train.add_argument("--lr", type=float, default=TrainingConfig.learning_rate)
    train.add_argument("--warmup-ratio", type=float, default=TrainingConfig.warmup_ratio)
    train.add_argument("--max-length", type=int, default=TrainingConfig.max_length)
    train.add_argument("--max-train-groups", type=int, default=None, help="train on only the first N examples (a trial)")
    train.add_argument("--no-fp16", action="store_true")
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--export-onnx", action="store_true", help="also export the best model to <out-dir>/onnx and verify it")
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--overwrite", action="store_true")

    export = commands.add_parser("export", help="export a trained model to ONNX and verify it")
    export.add_argument("--model", type=Path, required=True)
    export.add_argument("--out-dir", type=Path, required=True)
    export.add_argument("--max-length", type=int, default=512)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    run_train(args) if args.command == "train" else run_export(args)


if __name__ == "__main__":
    main()
