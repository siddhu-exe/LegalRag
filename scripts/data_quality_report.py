"""
Data quality report for the v2 corpus (plan task C2).

Reads the output of ``scripts/build_corpus_v2.py`` and writes, by default to
``results/data_profile/``:

    data_quality_report.md     tables for OCR noise, duplicates, missing dates, coverage by bench
    data_quality_report.json   the same numbers
    ocr_review_sample.csv      flagged and unflagged texts side by side, to calibrate the OCR flags

How to read it:
  * Counts over all records are exact. Text measures (OCR noise, repeated phrases) run on a
    reproducible hash-based sample (``--sample-rate``) and carry 95% Wilson intervals.
  * The OCR flags are heuristics with starting thresholds, not a validated detector. Read the
    review sample and move the thresholds (``--max-*`` options) before you quote a share.
  * Dropped records are only visible through ``build_report.json``, which is echoed here.

Example:
    python scripts/data_quality_report.py --corpus-dir artifacts/v2/corpus
    python scripts/data_quality_report.py --corpus-dir artifacts/v2/corpus --sample-rate 1.0
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np

try:
    from legalrag.evaluation.stats import wilson_interval
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from legalrag.evaluation.stats import wilson_interval  # noqa: E402

from legalrag.preprocessing.corpus_v2 import ReservoirSampler, text_fingerprint  # noqa: E402
from legalrag.preprocessing.quality import (  # noqa: E402
    YEAR_PERIODS,
    OcrThresholds,
    has_repeated_4gram,
    in_sample,
    ocr_flags,
    ocr_noise_metrics,
    year_period,
)

DEFAULT_CORPUS_DIR = Path("artifacts/v2/corpus")
DEFAULT_OUT_DIR = Path("results/data_profile")
PROFILE_PATH = Path("results/data_profile/bombay_profile.json")
FEATURES = ("non_ascii_share", "replacement_chars", "spaced_letter_share", "vowelless_share", "symbol_share")
BLANK_OUTCOMES = ("", "other")


def share(count: int, total: int, exact: bool = True) -> Dict[str, Any]:
    """A count and its share. Sampled shares get a Wilson interval; exact ones do not need one."""
    entry: Dict[str, Any] = {"count": count, "total": total, "share": count / total if total else None}
    if not exact and total:
        low, high = wilson_interval(count, total)
        entry.update(ci_low=low, ci_high=high)
    return entry


class QualityAccumulator:
    """Collects the report numbers from corpus rows, one row at a time."""

    def __init__(
        self,
        thresholds: OcrThresholds,
        sample_rate: float = 0.1,
        residual_duplicate_check: bool = True,
        review_size: int = 20,
        seed: int = 42,
    ):
        self.thresholds = thresholds
        self.sample_rate = sample_rate
        self.residual_duplicate_check = residual_duplicate_check
        self.seed = seed
        self.records = Counter()
        self.bench: Dict[tuple, Dict[str, Any]] = {}
        self.periods: Dict[str, Counter] = defaultdict(Counter)
        self.dates = defaultdict(Counter)
        self.metadata = defaultdict(Counter)
        self.sampled = Counter()
        self.feature_values: Dict[str, List[float]] = {name: [] for name in FEATURES}
        self.ocr_by_bench: Dict[str, Counter] = defaultdict(Counter)
        self.ocr_reasons = Counter()
        self.repeat_removed_rows = Counter()
        self.residual_repeat = Counter()
        self.flagged_review = ReservoirSampler(review_size, seed)
        self.clean_review = ReservoirSampler(review_size, seed + 1)
        self._seen_cnr: set = set()
        self._fingerprints: Dict[str, str] = {}
        self.duplicate_cnr_rows = 0
        self.residual_duplicates = 0
        self.kept_bench: Dict[str, str] = {}

    def add(self, row: Dict[str, Any]) -> None:
        searched = bool(row.get("in_search_index"))
        bench_name = row.get("bench_name") or "(none)"
        key = (row.get("source_config") or "?", bench_name)
        cnr = str(row.get("cnr"))

        self.records["all"] += 1
        self.records["search" if searched else "lookup_only"] += 1
        if cnr in self._seen_cnr:
            self.duplicate_cnr_rows += 1
        self._seen_cnr.add(cnr)

        entry = self.bench.setdefault(key, {"records": 0, "search": 0, "chars": 0, "lengths": [], "years": []})
        entry["records"] += 1
        year = row.get("judgment_year")
        self.periods[year_period(year)]["all"] += 1
        if searched:
            entry["search"] += 1
            entry["chars"] += int(row.get("clean_chars") or 0)
            entry["lengths"].append(int(row.get("clean_chars") or 0))
            self.periods[year_period(year)]["search"] += 1
            self.kept_bench[cnr] = bench_name
        if year:
            entry["years"].append(int(year))

        d = self.dates
        d["decision_date_status"][row.get("decision_date_status")] += 1
        d["cnr_date_status"][row.get("cnr_date_status")] += 1
        d["date_source"][row.get("date_source") or "none"] += 1
        if not row.get("judgment_date"):
            d["missing_by_bench"][bench_name] += 1
            d["missing"]["all"] += 1
        d["conflict"]["all"] += int(bool(row.get("date_conflict")))
        d["year_conflict"]["all"] += int(bool(row.get("year_conflict")))
        d["conflict_by_bench"][bench_name] += int(bool(row.get("date_conflict")))
        d["rows_by_bench"][bench_name] += 1

        m = self.metadata
        m["blank_judge"]["all"] += int(not row.get("judge"))
        m["blank_title"]["all"] += int(not row.get("title"))
        m["case_type_source"][row.get("case_type_source") or "none"] += 1
        m["blank_or_other_outcome"]["all"] += int((row.get("outcome") or "").strip().lower() in BLANK_OUTCOMES)

        if (row.get("repeat_chars_removed") or 0) > 0:
            self.repeat_removed_rows["all"] += 1

        text = row.get("text") or ""
        if searched and self.residual_duplicate_check:
            fingerprint = text_fingerprint(text, "prefix", 2000)
            if fingerprint in self._fingerprints:
                self.residual_duplicates += 1
            else:
                self._fingerprints[fingerprint] = cnr

        if in_sample(cnr, self.sample_rate, salt=f"quality-{self.seed}"):
            self._add_text_sample(row, text, bench_name)

    def _add_text_sample(self, row: Dict[str, Any], text: str, bench_name: str) -> None:
        self.sampled["all"] += 1
        metrics = ocr_noise_metrics(text)
        for name in FEATURES:
            self.feature_values[name].append(float(metrics[name]))
        flags = ocr_flags(metrics, self.thresholds)
        suspect = any(flags.values())
        self.ocr_by_bench[bench_name]["sampled"] += 1
        if suspect:
            self.sampled["ocr_suspect"] += 1
            self.ocr_by_bench[bench_name]["suspect"] += 1
            for reason, hit in flags.items():
                self.ocr_reasons[reason] += int(hit)
        if has_repeated_4gram(text):
            self.residual_repeat["all"] += 1
        sample_row = {
            "cnr": row.get("cnr"),
            "bench": row.get("bench_name"),
            "ocr_suspect": suspect,
            **{name: round(float(metrics[name]), 4) for name in FEATURES},
            "text_start": " ".join(text[:500].split()),
        }
        (self.flagged_review if suspect else self.clean_review).add(sample_row)

    def result(self) -> Dict[str, Any]:
        total = self.records["all"]
        sampled = self.sampled["all"]

        coverage = []
        for (source, bench_name), entry in sorted(self.bench.items(), key=lambda kv: -kv[1]["records"]):
            years = entry["years"]
            coverage.append({
                "source_config": source,
                "bench_name": bench_name,
                "records": entry["records"],
                "search_records": entry["search"],
                "search_share": entry["search"] / entry["records"] if entry["records"] else None,
                "search_chars": entry["chars"],
                "median_search_chars": statistics.median(entry["lengths"]) if entry["lengths"] else None,
                "first_year": min(years) if years else None,
                "last_year": max(years) if years else None,
            })

        features = {}
        for name, values in self.feature_values.items():
            if values:
                features[name] = {
                    "mean": float(np.mean(values)),
                    "p95": float(np.percentile(values, 95)),
                    "p99": float(np.percentile(values, 99)),
                    "max": float(np.max(values)),
                }

        d = self.dates
        return {
            "records": dict(self.records),
            "coverage_by_bench": coverage,
            "coverage_by_period": {
                p: {"records": self.periods[p]["all"], "search_records": self.periods[p]["search"]}
                for p in YEAR_PERIODS if self.periods[p]["all"]
            },
            "missing_and_odd_dates": {
                "missing_judgment_date": share(d["missing"]["all"], total),
                "date_conflict": share(d["conflict"]["all"], total),
                "year_conflict": share(d["year_conflict"]["all"], total),
                "decision_date_status": dict(d["decision_date_status"]),
                "cnr_date_status": dict(d["cnr_date_status"]),
                "date_source": dict(d["date_source"]),
                "missing_by_bench": {
                    b: share(d["missing_by_bench"][b], n) for b, n in sorted(d["rows_by_bench"].items())
                },
                "conflict_by_bench": {
                    b: share(d["conflict_by_bench"][b], n) for b, n in sorted(d["rows_by_bench"].items())
                },
            },
            "metadata": {
                "blank_judge": share(self.metadata["blank_judge"]["all"], total),
                "blank_title": share(self.metadata["blank_title"]["all"], total),
                "blank_or_other_outcome": share(self.metadata["blank_or_other_outcome"]["all"], total),
                "case_type_source": dict(self.metadata["case_type_source"]),
            },
            "text_quality": {
                "sample_rate": self.sample_rate,
                "sampled_records": sampled,
                "ocr_thresholds": self.thresholds.to_dict(),
                "ocr_suspect": share(self.sampled["ocr_suspect"], sampled, exact=False),
                "ocr_flag_reasons": {k: share(v, sampled, exact=False) for k, v in sorted(self.ocr_reasons.items())},
                "ocr_feature_distribution": features,
                "ocr_suspect_by_bench": {
                    b: share(c["suspect"], c["sampled"], exact=False) for b, c in sorted(self.ocr_by_bench.items())
                },
                "repeated_4gram_in_first_3000_chars": share(self.residual_repeat["all"], sampled, exact=False),
                "records_where_repeat_cleaning_removed_text": share(self.repeat_removed_rows["all"], total),
            },
            "duplicates": {
                "repeated_cnr_rows_in_output": self.duplicate_cnr_rows,
                "residual_near_duplicates_in_search_index": (
                    self.residual_duplicates if self.residual_duplicate_check else None
                ),
                "residual_check_ran": self.residual_duplicate_check,
            },
        }

    def review_rows(self) -> List[Dict[str, Any]]:
        return self.flagged_review.items + self.clean_review.items


def read_rows(path: Path, columns: Optional[Sequence[str]] = None) -> Iterator[Dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(str(path))
    available = set(parquet.schema_arrow.names)
    wanted = [c for c in columns if c in available] if columns else None
    for batch in parquet.iter_batches(batch_size=2048, columns=wanted):
        yield from batch.to_pylist()


def link_statistics(links_path: Path, kept_bench: Dict[str, str]) -> Dict[str, Any]:
    """Near-duplicate groups from the link table written by the builder."""
    if not links_path.exists():
        return {"available": False}
    per_kept: Counter = Counter()
    for row in read_rows(links_path, ["kept_cnr"]):
        per_kept[row["kept_cnr"]] += 1
    by_bench: Counter = Counter()
    for kept, count in per_kept.items():
        by_bench[kept_bench.get(kept, "(not in search index)")] += count
    return {
        "available": True,
        "removed_records": sum(per_kept.values()),
        "kept_records_with_duplicates": len(per_kept),
        "largest_group_removed": max(per_kept.values()) if per_kept else 0,
        "removed_by_bench_of_kept_record": dict(by_bench.most_common()),
    }


def profile_reconciliation(profile_path: Path, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Profile counts next to built counts. Differences are expected: the profile cut on raw length."""
    if not profile_path.exists():
        return None
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    bombay_expected = sum(b["reasoned_records"] for b in profile.get("benches", []))
    sc_expected = profile.get("supreme_court", {}).get("reasoned_records")
    built = Counter()
    for row in result["coverage_by_bench"]:
        built[row["source_config"]] += row["search_records"]
    return {
        "profile_min_chars": profile.get("reasoned_min_chars"),
        "bombay": {"profile_reasoned_records": bombay_expected, "built_search_records": built.get("high_courts", 0)},
        "supreme_court": {"profile_reasoned_records": sc_expected, "built_search_records": built.get("supreme_court", 0)},
        "note": (
            "The profile cut on raw text_length; the build cuts on cleaned length and removes near duplicates, "
            "so built counts are expected to be somewhat lower."
        ),
    }


def analyze(corpus_dir: Path, args: argparse.Namespace) -> tuple:
    accumulator = QualityAccumulator(
        OcrThresholds(
            args.max_non_ascii_share, args.max_replacement_chars, args.max_spaced_letter_share,
            args.max_vowelless_share, args.max_symbol_share,
        ),
        sample_rate=args.sample_rate,
        residual_duplicate_check=not args.skip_residual_duplicate_check,
        review_size=args.review_size,
        seed=args.seed,
    )
    for name in ("search_corpus.parquet", "lookup_only.parquet"):
        path = corpus_dir / name
        if not path.exists():
            raise SystemExit(f"{path} not found. Run scripts/build_corpus_v2.py first.")
        print(f"reading {path}")
        for row in read_rows(path):
            accumulator.add(row)
    result = accumulator.result()
    result["near_duplicate_links"] = link_statistics(corpus_dir / "duplicate_links.parquet", accumulator.kept_bench)
    report_path = corpus_dir / "build_report.json"
    if report_path.exists():
        build = json.loads(report_path.read_text(encoding="utf-8"))
        result["build"] = {
            "built_at_utc": build.get("built_at_utc"),
            "partial_build": build.get("partial_build"),
            "settings": build.get("settings"),
            "stats": build.get("stats"),
        }
    else:
        result["build"] = None
    result["profile_reconciliation"] = profile_reconciliation(args.profile, result)
    return result, accumulator


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _pct(entry: Dict[str, Any]) -> str:
    if entry["share"] is None:
        return "n/a"
    text = f"{entry['share'] * 100:.1f}%"
    if "ci_low" in entry:
        text += f" [{entry['ci_low'] * 100:.1f}, {entry['ci_high'] * 100:.1f}]"
    return text


def render_markdown(result: Dict[str, Any]) -> str:
    rec = result["records"]
    text_q, dates, meta, dups = result["text_quality"], result["missing_and_odd_dates"], result["metadata"], result["duplicates"]
    lines = [
        "# Data quality report: v2 corpus",
        "",
        f"Records: {rec.get('all', 0):,} ({rec.get('search', 0):,} in the search index, "
        f"{rec.get('lookup_only', 0):,} lookup only).",
        "Counts over all records are exact. Text measures use a "
        f"{text_q['sample_rate'] * 100:.0f}% hash-based sample ({text_q['sampled_records']:,} records) "
        "with 95% Wilson intervals in brackets.",
        "",
    ]
    build = result.get("build")
    if build and build.get("partial_build"):
        lines += ["> **Warning:** this corpus came from a partial build (`--max-rows`). Do not quote these numbers.", ""]
    if build and build.get("stats"):
        stats = build["stats"]
        lines += ["## What the build dropped", "", f"Scanned {stats['scanned']:,} records.", "",
                  "| Reason | Records |", "|---|---|"]
        lines += [f"| {reason} | {count:,} |" for reason, count in stats["dropped_by_reason"].items()]
        lines += ["", f"Counts reconcile: {stats['identity_holds']}. Records shorter than the cut were kept "
                  "for lookup, not dropped.", ""]

    lines += ["## Coverage by bench", "",
              "| Source | Bench | Records | In search index | Share | Median chars (search) | Years |",
              "|---|---|---|---|---|---|---|"]
    for row in result["coverage_by_bench"]:
        median = "n/a" if row["median_search_chars"] is None else f"{row['median_search_chars']:,.0f}"
        share_text = "n/a" if row["search_share"] is None else f"{row['search_share'] * 100:.1f}%"
        years = f"{row['first_year']}-{row['last_year']}" if row["first_year"] else "n/a"
        lines.append(f"| {row['source_config']} | {row['bench_name']} | {row['records']:,} | {row['search_records']:,} | "
                     f"{share_text} | {median} | {years} |")
    lines += ["", "Bench names for Bombay (Appellate Side) and Bombay (Original Side) are inferred from CNR prefixes.",
              "", "## Coverage by period", "", "| Period | Records | In search index |", "|---|---|---|"]
    lines += [f"| {p} | {v['records']:,} | {v['search_records']:,} |" for p, v in result["coverage_by_period"].items()]

    lines += ["", "## Missing and odd dates", "",
              "| Measure | Records | Share |", "|---|---|---|",
              f"| No usable judgment date | {dates['missing_judgment_date']['count']:,} | {_pct(dates['missing_judgment_date'])} |",
              f"| decision_date differs from CNR date | {dates['date_conflict']['count']:,} | {_pct(dates['date_conflict'])} |",
              f"| Year differs | {dates['year_conflict']['count']:,} | {_pct(dates['year_conflict'])} |",
              "", f"decision_date status: {dates['decision_date_status']}. CNR date status: {dates['cnr_date_status']}. "
              f"Date source used: {dates['date_source']}.",
              "", "A very high `unparseable` count for CNR dates means the assumed CNR format is wrong.",
              "", "| Bench | No usable date | Date conflict |", "|---|---|---|"]
    for bench, entry in dates["missing_by_bench"].items():
        lines.append(f"| {bench} | {_pct(entry)} | {_pct(dates['conflict_by_bench'][bench])} |")

    lines += ["", "## OCR noise and repeated text (sample)", "",
              f"Thresholds (starting values, not validated): {text_q['ocr_thresholds']}.", "",
              "| Measure | Share |", "|---|---|",
              f"| Flagged by any OCR measure | {_pct(text_q['ocr_suspect'])} |"]
    lines += [f"| - flagged by {reason} | {_pct(entry)} |" for reason, entry in text_q["ocr_flag_reasons"].items()]
    lines += [
        f"| Repeated 4-word phrase 3+ times in first 3,000 chars (after cleaning) | {_pct(text_q['repeated_4gram_in_first_3000_chars'])} |",
        f"| Records where repeat cleaning removed text (all records) | {_pct(text_q['records_where_repeat_cleaning_removed_text'])} |",
        "",
        "The profile measured 54.1% for the repeated-phrase test before cleaning; ordinary legal phrases also trigger it, "
        "so treat it as an upper limit.",
        "", "| Measure | Mean | 95th percentile | 99th percentile | Max |", "|---|---|---|---|---|",
    ]
    for name, f in text_q["ocr_feature_distribution"].items():
        lines.append(f"| {name} | {f['mean']:.4f} | {f['p95']:.4f} | {f['p99']:.4f} | {f['max']:.4f} |")
    lines += ["", "| Bench | Sampled | Flagged |", "|---|---|---|"]
    for bench, entry in text_q["ocr_suspect_by_bench"].items():
        lines.append(f"| {bench} | {entry['total']:,} | {_pct(entry)} |")
    lines += ["", "Read `ocr_review_sample.csv` (flagged and unflagged texts) before trusting these flags.", ""]

    lines += ["## Duplicates", "",
              f"- Repeated CNR rows in the output: {dups['repeated_cnr_rows_in_output']:,} (should be 0).",
              ("- Near duplicates still in the search index after the build: "
               f"{dups['residual_near_duplicates_in_search_index']:,} (should be 0)." if dups["residual_check_ran"]
               else "- Residual near-duplicate check was skipped.")]
    links = result.get("near_duplicate_links", {})
    if links.get("available"):
        lines += [f"- Near duplicates removed by the build: {links['removed_records']:,}, linked to "
                  f"{links['kept_records_with_duplicates']:,} kept records (largest group: {links['largest_group_removed']:,} removed).",
                  f"- Removed, by bench of the kept record: {links['removed_by_bench_of_kept_record']}."]

    lines += ["", "## Metadata", "",
              "| Measure | Records | Share |", "|---|---|---|",
              f"| Judge not recorded | {meta['blank_judge']['count']:,} | {_pct(meta['blank_judge'])} |",
              f"| Title blank | {meta['blank_title']['count']:,} | {_pct(meta['blank_title'])} |",
              f"| Outcome blank or `other` | {meta['blank_or_other_outcome']['count']:,} | {_pct(meta['blank_or_other_outcome'])} |",
              "", f"Case type source: {meta['case_type_source']}. Outcome statistics are incomplete, so do not present them as complete.", ""]
    if build and build.get("stats") and build["stats"].get("top_unmapped_case_codes"):
        lines += ["Most common parsed case-type codes with no readable name yet: "
                  f"{build['stats']['top_unmapped_case_codes']}.", ""]
    recon = result.get("profile_reconciliation")
    if recon:
        lines += ["## Against the earlier profile", "",
                  f"Profile cut: {recon['profile_min_chars']} characters on raw length.", "",
                  "| Part | Profile (reasoned) | Built (search index) |", "|---|---|---|",
                  f"| Bombay HC | {recon['bombay']['profile_reasoned_records']:,} | {recon['bombay']['built_search_records']:,} |",
                  f"| Supreme Court | {recon['supreme_court']['profile_reasoned_records']:,} | {recon['supreme_court']['built_search_records']:,} |",
                  "", recon["note"], ""]
    return "\n".join(lines)


def write_review_sample(path: Path, rows: List[Dict[str, Any]]) -> None:
    columns = ["cnr", "bench", "ocr_suspect", *FEATURES, "text_start"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    defaults = OcrThresholds()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--profile", type=Path, default=PROFILE_PATH, help="earlier profile JSON for comparison (optional)")
    parser.add_argument("--sample-rate", type=float, default=0.1, help="share of records used for text measures")
    parser.add_argument("--skip-residual-duplicate-check", action="store_true")
    parser.add_argument("--review-size", type=int, default=20, help="flagged and unflagged texts in the review sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-non-ascii-share", type=float, default=defaults.max_non_ascii_share)
    parser.add_argument("--max-replacement-chars", type=int, default=defaults.max_replacement_chars)
    parser.add_argument("--max-spaced-letter-share", type=float, default=defaults.max_spaced_letter_share)
    parser.add_argument("--max-vowelless-share", type=float, default=defaults.max_vowelless_share)
    parser.add_argument("--max-symbol-share", type=float, default=defaults.max_symbol_share)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    result, accumulator = analyze(args.corpus_dir, args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "data_quality_report.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (args.out_dir / "data_quality_report.md").write_text(render_markdown(result), encoding="utf-8")
    write_review_sample(args.out_dir / "ocr_review_sample.csv", accumulator.review_rows())
    print(f"Wrote {args.out_dir / 'data_quality_report.md'}, data_quality_report.json and ocr_review_sample.csv")


if __name__ == "__main__":
    main()
