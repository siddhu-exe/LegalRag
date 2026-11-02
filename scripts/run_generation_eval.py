"""
Generation evaluation with an independent judge (plan task C3). THIS SCRIPT CALLS A PAID LLM.

For each benchmark question it takes the top-5 passages saved by ``run_retrieval_eval.py``
for one system (default ``hybrid_rerank``, the deployed cascade), asks the *deployed generator*
for a grounded answer, and has a *different* judge model score it. Then it summarises every
metric with a 95% interval.

Safety rules (docs/IMPROVEMENT_PLAN.md, ground rules 5 and 8):
  * It prints an estimated cost first and asks you to confirm (``--yes`` to skip, ``--dry-run``
    to only print). Prices are never assumed: pass your provider's prices to see dollars.
  * Every item is written to ``generation_per_item.jsonl`` as soon as it is done. Re-running the
    same command resumes; ``--retry-errors`` repeats items that hit system errors.
  * The judge model must be named with ``--judge-model`` and must differ from the generator, or
    the script stops. Resuming with a different generator or judge is refused.
  * Provider and parsing failures are system errors: counted and reported, never scored as bad
    answers. Refusals are detected without the judge and reported as a refusal rate beside quality.

Judge scores are estimates. To report how far they can be trusted, label items yourself:
    1. python scripts/run_generation_eval.py ... --export-label-sheet 50    (no judge scores shown)
    2. fill in the blank score fields of generation_label_sheet.json by hand
    3. python scripts/run_generation_eval.py ... --metrics-only --human-labels filled.json
It then reports Cohen's kappa with an interval for each metric. The labels must be yours.

Needs GROQ_API_KEY for both the generator and the judge. The generator defaults to the
GROQ_MODEL_NAME environment variable, like the API.

Examples:
    python scripts/run_generation_eval.py --benchmark data/eval_v2/benchmark.json \\
        --retrieval-results results/eval_v2/retrieval_per_question.jsonl \\
        --judge-model <a-model-that-is-not-the-generator> --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from legalrag.evaluation.runtime import JsonlCheckpoint
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from legalrag.evaluation.runtime import JsonlCheckpoint  # noqa: E402

from legalrag.evaluation.benchmark import file_sha256, load_benchmark, sample_benchmark  # noqa: E402
from legalrag.evaluation.judge import (  # noqa: E402
    JUDGE_SYSTEM_PROMPT,
    JudgeVerdict,
    LLMJudge,
    build_judge_prompt,
    check_independence,
    is_refusal,
    label_failure,
    summarize_generation,
)
from legalrag.evaluation.runtime import (  # noqa: E402
    build_cost_line,
    confirm_or_abort,
    ensure_run_config,
    format_cost_estimate,
)
from legalrag.evaluation.stats import kappa_interval, percent_agreement  # noqa: E402
from legalrag.generation.client import GenerationError  # noqa: E402
from legalrag.generation.prompts import SYSTEM_PROMPT, build_rag_prompt, extract_citations  # noqa: E402

DEFAULT_OUT_DIR = Path("results/eval_v2")
DEFAULT_GENERATOR_MODEL = "qwen/qwen3.8-27b"  # same default as the API settings
SCORE_FIELDS = ("answer_relevance", "faithfulness", "citation_correctness")
HUMAN_FIELDS = SCORE_FIELDS + ("hallucination",)


# ---------------------------------------------------------------------------
# One item
# ---------------------------------------------------------------------------


def process_item(
    item: Dict[str, Any],
    retrieval: Dict[str, Any],
    system: str,
    generator: Any,
    judge: LLMJudge,
    generator_model: str,
) -> Dict[str, Any]:
    """Generates and judges one answer. Never raises on provider faults."""
    context = retrieval["context"][system]
    ranked = retrieval["systems"][system]
    context_ids = [block["chunk_id"] for block in context]
    user_prompt = build_rag_prompt(item["question"], context)

    record: Dict[str, Any] = {
        "item_id": item["question_id"],
        "question": item["question"],
        "group": item["group"],
        "system": system,
        "generator_model": generator_model,
        "judge_model": judge.judge_model,
        "context_chunk_ids": context_ids,
        "generation_ok": False,
        "gen_error_type": None,
        "answer": None,
        "refused": False,
        "cited_chunk_ids": [],
        "citations_in_context": None,
        "judge_status": "skipped_generation_error",
        "judge_error_type": None,
        "scores": None,
        "tokens": {"gen_prompt": 0, "gen_completion": 0, "judge_prompt": 0, "judge_completion": 0},
    }

    t0 = time.perf_counter()
    try:
        generated = generator.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt, temperature=0.0)
        if not getattr(generated, "is_success", False) or generated.text is None:
            raise GenerationError("generation was not successful", error_type="GenerationFailed")
    except GenerationError as exc:
        record["gen_error_type"] = exc.error_type
    except Exception as exc:  # noqa: BLE001 - class name only, never provider text
        record["gen_error_type"] = type(exc).__name__
    else:
        record.update(generation_ok=True, answer=generated.text)
        record["tokens"]["gen_prompt"] = generated.prompt_tokens or 0
        record["tokens"]["gen_completion"] = generated.completion_tokens or 0
        cited = extract_citations(generated.text)
        record["cited_chunk_ids"] = cited
        record["citations_in_context"] = all(c in context_ids for c in cited) if cited else None
        record["refused"] = is_refusal(generated.text)
    record["generation_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)

    verdict: Optional[JudgeVerdict] = None
    if record["generation_ok"] and record["refused"]:
        record["judge_status"] = "skipped_refusal"
    elif record["generation_ok"]:
        t0 = time.perf_counter()
        verdict = judge.judge(item["question"], context, record["answer"], item.get("supporting_text"))
        record["judge_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        record["judge_status"] = verdict.status
        record["judge_error_type"] = verdict.error_type
        record["tokens"]["judge_prompt"] = verdict.prompt_tokens
        record["tokens"]["judge_completion"] = verdict.completion_tokens
        if verdict.ok:
            record["scores"] = verdict.scores.to_eval_record()

    if item["gold_cnrs"]:
        top_ids, gold_ids = [str(c) for c in ranked["cnrs"][:5]], item["gold_cnrs"]
    else:
        top_ids, gold_ids = ranked["chunk_ids"][:5], item["gold_chunk_ids"]
    record["retrieval_hit"] = bool(set(top_ids) & set(gold_ids))
    record["failure_label"] = label_failure(
        record["generation_ok"], record["refused"], verdict, top_ids, gold_ids
    )
    good = record["generation_ok"] and record["judge_status"] in ("ok", "skipped_refusal")
    record["status"] = "ok" if good else "error"
    return record


# ---------------------------------------------------------------------------
# Cost estimate
# ---------------------------------------------------------------------------


def estimate_run_cost(
    todo: Sequence[Dict[str, Any]],
    retrievals: Dict[str, Dict[str, Any]],
    system: str,
    args: argparse.Namespace,
) -> str:
    """Builds the same prompts the run will use and estimates tokens. Judge cost is an upper bound."""
    gen_prompts, judge_prompts = [], []
    placeholder_answer = "x" * (4 * args.expected_gen_output_tokens)
    for item in todo:
        context = retrievals[item["question_id"]]["context"][system]
        gen_prompts.append(SYSTEM_PROMPT + build_rag_prompt(item["question"], context))
        judge_prompts.append(JUDGE_SYSTEM_PROMPT + build_judge_prompt(item["question"], context, placeholder_answer))
    lines = [
        build_cost_line("generation", gen_prompts, args.expected_gen_output_tokens, args.gen_price_in, args.gen_price_out),
        build_cost_line("judge (max)", judge_prompts, args.expected_judge_output_tokens, args.judge_price_in, args.judge_price_out),
    ]
    return format_cost_estimate(lines)


# ---------------------------------------------------------------------------
# Human agreement
# ---------------------------------------------------------------------------


def export_label_sheet(records: Sequence[Dict[str, Any]], contexts: Dict[str, List[Dict[str, Any]]], size: int, seed: int) -> List[Dict[str, Any]]:
    """A random sample of answered items for hand labelling. Judge scores are left out on purpose."""
    pool = sorted((r for r in records if r.get("generation_ok") and not r.get("refused")), key=lambda r: r["item_id"])
    chosen = random.Random(seed).sample(pool, min(size, len(pool)))
    return [
        {
            "item_id": r["item_id"],
            "question": r["question"],
            "context": contexts.get(r["item_id"], []),
            "answer": r["answer"],
            "instructions": "Score 1 (worst) to 4 (best) against the context only. hallucination: true if the answer states a case, section, quote or fact that is not in the context.",
            "answer_relevance": None,
            "faithfulness": None,
            "citation_correctness": None,
            "hallucination": None,
            "notes": "",
        }
        for r in chosen
    ]


def judge_agreement(records: Sequence[Dict[str, Any]], human: Sequence[Dict[str, Any]], seed: int = 0) -> Dict[str, Any]:
    """Cohen's kappa between the judge and your labels, with intervals. Skips unlabelled items."""
    by_id = {r["item_id"]: r for r in records if r.get("scores")}
    report: Dict[str, Any] = {"labelled_items": len(human), "metrics": {}}
    for field in HUMAN_FIELDS:
        pairs = [
            (by_id[h["item_id"]]["scores"][field], h[field])
            for h in human
            if h.get("item_id") in by_id and h.get(field) is not None
        ]
        if not pairs:
            continue
        judge_labels = [a for a, _ in pairs]
        human_labels = [b for _, b in pairs]
        entry: Dict[str, Any] = {
            "n": len(pairs),
            "percent_agreement": percent_agreement(judge_labels, human_labels),
            "kappa": kappa_interval(judge_labels, human_labels, None, seed=seed).to_dict(),
        }
        if field in SCORE_FIELDS:
            entry["kappa_quadratic_weighted"] = kappa_interval(judge_labels, human_labels, "quadratic", seed=seed).to_dict()
        report["metrics"][field] = entry
    return report


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(interval: Dict[str, Any]) -> str:
    if interval.get("estimate") is None:
        return "n/a"
    return f"{interval['estimate']:.3f} [{interval['low']:.3f}, {interval['high']:.3f}]"


def render_markdown(result: Dict[str, Any]) -> str:
    meta, summary = result["run"], result["summary"]
    counts = summary["counts"]
    lines = [
        "# Generation evaluation",
        "",
        f"System: `{meta['system']}`. Generator: `{meta['generator_model']}`. Judge: `{meta['judge_model']}`.",
        f"Questions: {counts['total']} total, {counts['answered']} answered, {counts['refused']} refused, "
        f"{counts['judged']} judged, {counts['generation_errors']} generation errors, {counts['judge_errors']} judge errors.",
        "Intervals are 95% (cluster bootstrap by source judgment). Judge scores are estimates; see the agreement section.",
        "",
    ]
    for warning in meta.get("independence_warnings", []) + summary["warnings"]:
        lines.append(f"> **Warning:** {warning}")
    if lines[-1].startswith(">"):
        lines.append("")
    lines += ["## Metrics", "", "| Metric | Value [95% CI] | n |", "|---|---|---|"]
    for name, interval in summary["metrics"].items():
        lines.append(f"| {name} | {_fmt(interval)} | {interval['n']} |")
    lines += ["", "Quality metrics use judged answers only. `refusal_rate` is over all items without a generation error.", ""]
    labels = summary["failure_labels"]
    lines += [
        f"## First-pass failure labels (n = {labels['denominator']})",
        "",
        "Automatic labels. The real causes come from reading the failures.",
        "",
        "| Label | Count | Share [95% Wilson] |",
        "|---|---|---|",
    ]
    for label, row in labels["labels"].items():
        share = "n/a" if row["share"] is None else f"{row['share']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        lines.append(f"| {label} | {row['count']} | {share} |")
    agreement = result.get("judge_agreement")
    if agreement:
        lines += ["", f"## Judge agreement with human labels ({agreement['labelled_items']} labelled items)", "",
                  "| Metric | n | Agreement | Kappa [95% CI] | Weighted kappa (quadratic) |", "|---|---|---|---|---|"]
        for name, row in agreement["metrics"].items():
            weighted = _fmt(row["kappa_quadratic_weighted"]) if "kappa_quadratic_weighted" in row else "-"
            lines.append(f"| {name} | {row['n']} | {row['percent_agreement']:.3f} | {_fmt(row['kappa'])} | {weighted} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--retrieval-results", type=Path, required=True, help="retrieval_per_question.jsonl")
    parser.add_argument("--system", default="hybrid_rerank", choices=("bm25", "dense", "hybrid_rrf", "hybrid_rerank"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--judge-model", required=True, help="must differ from the generator model")
    parser.add_argument("--generator-model", default=os.getenv("GROQ_MODEL_NAME", DEFAULT_GENERATOR_MODEL))
    parser.add_argument("--question-model", default=None, help="model that wrote the questions (recorded; warns on overlap)")
    parser.add_argument("--sample", type=int, default=None, help="random subset of N questions (seeded)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--dry-run", action="store_true", help="print the estimate and stop")
    parser.add_argument("--yes", action="store_true", help="do not ask before spending")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--force-resume", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between items (rate limits)")
    parser.add_argument("--metrics-only", action="store_true", help="summarise saved items; no LLM calls")
    parser.add_argument("--export-label-sheet", type=int, default=None, metavar="N")
    parser.add_argument("--human-labels", type=Path, default=None, help="your filled label sheet (JSON list)")
    for prefix in ("gen", "judge"):
        parser.add_argument(f"--{prefix}-price-in", type=float, default=None, help="USD per million input tokens")
        parser.add_argument(f"--{prefix}-price-out", type=float, default=None, help="USD per million output tokens")
    parser.add_argument("--expected-gen-output-tokens", type=int, default=400)
    parser.add_argument("--expected-judge-output-tokens", type=int, default=150)
    return parser.parse_args(argv)


def load_retrievals(path: Path) -> Dict[str, Dict[str, Any]]:
    checkpoint = JsonlCheckpoint(path)
    return {k: v for k, v in checkpoint.records.items() if v.get("status") == "ok"}


def build_clients(args: argparse.Namespace) -> tuple:
    from legalrag.generation.client import LegalGenerationClient

    generator = LegalGenerationClient(model_name=args.generator_model)
    judge_client = LegalGenerationClient(model_name=args.judge_model, max_retries=2)
    return generator, LLMJudge(judge_client, args.generator_model, args.question_model)


def main(argv: Optional[Sequence[str]] = None, clients: Optional[tuple] = None) -> None:
    args = parse_args(argv)
    warnings = check_independence(args.judge_model, args.generator_model, args.question_model)  # stops on same model
    items = sample_benchmark(load_benchmark(args.benchmark), args.sample, args.seed)
    retrievals = load_retrievals(args.retrieval_results)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = JsonlCheckpoint(out / "generation_per_item.jsonl")

    if not args.metrics_only:
        usable = [i for i in items if i["question_id"] in retrievals]
        skipped = len(items) - len(usable)
        if skipped:
            print(f"{skipped} questions have no saved retrieval result and are skipped.")
        retry = ["error"] if args.retry_errors else []
        todo = [i for i in usable if not checkpoint.is_done(i["question_id"], retry)]
        print(f"{len(usable) - len(todo)} items already done, {len(todo)} to run.")
        if todo:
            print(estimate_run_cost(todo, retrievals, args.system, args))
            if args.dry_run:
                print("Dry run: nothing was sent.")
                return
            confirm_or_abort(
                f"About to send {len(todo)} generation and up to {len(todo)} judge requests "
                f"(generator {args.generator_model}, judge {args.judge_model}).",
                args.yes,
            )
        elif args.dry_run:
            print("Nothing to run.")
            return
        # Checked even when nothing is left to run, so old results are never relabelled with new models.
        ensure_run_config(
            out / "generation_run_config.json",
            {
                "benchmark_sha256": file_sha256(args.benchmark),
                "system": args.system,
                "generator_model": args.generator_model,
                "judge_model": args.judge_model,
            },
            force=args.force_resume,
        )
        if todo:
            generator, judge = clients if clients is not None else build_clients(args)
            warnings = judge.warnings
            for number, item in enumerate(todo, start=1):
                record = process_item(item, retrievals[item["question_id"]], args.system, generator, judge, args.generator_model)
                checkpoint.append(record)
                if number % 10 == 0 or number == len(todo):
                    print(f"  {number}/{len(todo)} done")
                if args.sleep:
                    time.sleep(args.sleep)

    wanted = {i["question_id"] for i in items}
    records = [r for r in checkpoint.values() if r["item_id"] in wanted]
    contexts = {k: v["context"][args.system] for k, v in retrievals.items()}

    if args.export_label_sheet:
        sheet = export_label_sheet(records, contexts, args.export_label_sheet, args.seed)
        (out / "generation_label_sheet.json").write_text(json.dumps(sheet, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {len(sheet)} items to {out / 'generation_label_sheet.json'} (judge scores not shown).")

    # Report the models that actually produced the saved items, not just the ones named now.
    generators = sorted({r["generator_model"] for r in records}) or [args.generator_model]
    judges = sorted({r["judge_model"] for r in records}) or [args.judge_model]
    warnings = list(warnings)
    if len(generators) > 1 or len(judges) > 1:
        warnings.append(f"Saved items mix models (generators {generators}, judges {judges}); do not quote as one result.")
    result: Dict[str, Any] = {
        "run": {
            "system": args.system,
            "generator_model": ", ".join(generators),
            "judge_model": ", ".join(judges),
            "question_model": args.question_model,
            "independence_warnings": warnings,
            "benchmark_sha256": file_sha256(args.benchmark),
            "n_resamples": args.n_resamples,
            "seed": args.seed,
        },
        "summary": summarize_generation(records, args.n_resamples, args.seed),
    }
    if args.human_labels:
        human = json.loads(args.human_labels.read_text(encoding="utf-8"))
        result["judge_agreement"] = judge_agreement(records, human, args.seed)
    (out / "generation_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "generation_metrics.md").write_text(render_markdown(result), encoding="utf-8")
    print(f"Wrote {out / 'generation_metrics.json'} and generation_metrics.md")


if __name__ == "__main__":
    main()
