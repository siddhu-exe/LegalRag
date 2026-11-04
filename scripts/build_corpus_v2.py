"""
Builds the v2 corpus (plan task C1): Bombay High Court plus Supreme Court.

Reads the dataset's parquet files, applies the rules in ``legalrag.preprocessing.corpus_v2``
and writes, under ``artifacts/v2/corpus/`` by default:

    search_corpus.parquet     reasoned judgments (clean_chars >= --min-chars), with text
    lookup_only.parquet       shorter records, kept for case lookup, with text
    duplicate_links.parquet   removed near-duplicate CNR -> kept CNR
    build_report.json         every count, including dropped records by reason, and all settings
    date_disagreements_sample.csv   random sample for choosing the date rule (plan task P3)
    short_orders_sample.csv         random sample for checking the length cut (plan task P3)

It never reads or writes v1 artifacts, and refuses an output directory without a ``v2`` part.
Runs on your PC: it reads about 6 GB of text and keeps only CNRs and fingerprints in memory.

Examples:
    # Local parquet files (any number, in any folder), already downloaded:
    python scripts/build_corpus_v2.py \\
        --high-courts data/hf/high_courts/*.parquet \\
        --supreme-court data/hf/supreme_court/*.parquet

    # Download the parquet files from the Hugging Face Hub first (large):
    python scripts/build_corpus_v2.py --from-hub

    # Quick check on the first 20,000 selected records of each config:
    python scripts/build_corpus_v2.py --from-hub --max-rows 20000 --out-dir artifacts/v2/corpus_smoke
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

try:
    from legalrag.preprocessing.corpus_v2 import (
        BuildConfig,
        CorpusBuilder,
        OUTPUT_FIELDS,
        ReservoirSampler,
        SourceSpec,
        default_sources,
    )
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from legalrag.preprocessing.corpus_v2 import (  # noqa: E402
        BuildConfig,
        CorpusBuilder,
        OUTPUT_FIELDS,
        ReservoirSampler,
        SourceSpec,
        default_sources,
    )

DEFAULT_OUT_DIR = Path("artifacts/v2/corpus")
HUB_REPO = "overthelex/indian-court-decisions"
HUB_SPLITS = ("train", "validation", "test")
READ_COLUMNS = (
    "cnr", "court_code", "bench", "year", "full_text", "title", "judge", "petitioner",
    "respondent", "decision_date", "disposal_nature_normalized", "case_type",
)
OUTPUT_FILES = (
    "search_corpus.parquet", "lookup_only.parquet", "duplicate_links.parquet",
    "build_report.json", "date_disagreements_sample.csv", "short_orders_sample.csv",
)
SNIPPET_CHARS = 400
STAGING_DIRNAME = ".build_staging"


def assert_v2_output_dir(path: Path) -> None:
    """v2 builds must live under a ``v2`` directory so v1 artifacts cannot be overwritten."""
    if "v2" not in path.resolve().parts and "v2" not in path.parts:
        raise SystemExit(
            f"Refusing to write to {path}: the output directory must contain a 'v2' part "
            "(for example artifacts/v2/corpus). v1 artifacts are frozen."
        )


def expand_inputs(patterns: Sequence[str]) -> List[Path]:
    """Expands files, directories and glob patterns into a sorted list of parquet files."""
    found: List[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            found.extend(sorted(path.rglob("*.parquet")))
        elif path.is_file():
            found.append(path)
        else:
            found.extend(Path(p) for p in sorted(glob.glob(pattern, recursive=True)))
    unique = sorted(dict.fromkeys(found))
    return unique


def download_from_hub(config: str, splits: Sequence[str], repo: str) -> List[Path]:
    """Downloads ``<config>/<split>.parquet`` files (network; run on your PC only)."""
    from huggingface_hub import hf_hub_download

    return [
        Path(hf_hub_download(repo_id=repo, filename=f"{config}/{split}.parquet", repo_type="dataset"))
        for split in splits
    ]


class BufferedParquetWriter:
    """Writes rows to one parquet file in reasonably sized row groups."""

    def __init__(self, path: Path, schema: Any, flush_rows: int = 10_000):
        import pyarrow as pa
        import pyarrow.parquet as pq

        self._pa = pa
        self._schema = schema
        self._flush_rows = flush_rows
        self._rows: List[Dict[str, Any]] = []
        self._writer = pq.ParquetWriter(str(path), schema, compression="zstd")
        self.count = 0

    def add(self, row: Dict[str, Any]) -> None:
        self._rows.append(row)
        if len(self._rows) >= self._flush_rows:
            self.flush()

    def flush(self) -> None:
        if self._rows:
            table = self._pa.Table.from_pylist(self._rows, schema=self._schema)
            self._writer.write_table(table)
            self.count += len(self._rows)
            self._rows = []

    def close(self) -> None:
        self.flush()
        self._writer.close()


def arrow_schema(fields: Sequence[Any]) -> Any:
    import pyarrow as pa

    types = {"string": pa.string(), "bool": pa.bool_(), "int32": pa.int32(), "int64": pa.int64()}
    return pa.schema([(name, types[kind]) for name, kind in fields])


REQUIRED_COLUMNS = {
    "high_courts": ("cnr", "full_text", "court_code"),
    "supreme_court": ("cnr", "full_text"),
}


def inspect_files(files: Sequence[Path]) -> Dict[str, Any]:
    """Reads parquet metadata only (no text): rows and column names with types, per file."""
    import pyarrow.parquet as pq

    per_file = []
    for path in files:
        parquet = pq.ParquetFile(str(path))
        per_file.append(
            {
                "path": str(path),
                "rows": parquet.metadata.num_rows,
                "columns": {field.name: str(field.type) for field in parquet.schema_arrow},
            }
        )
    in_every_file = set.intersection(*(set(f["columns"]) for f in per_file)) if per_file else set()
    in_any_file = set().union(*(set(f["columns"]) for f in per_file)) if per_file else set()
    first = per_file[0]["columns"] if per_file else {}
    return {
        "files": per_file,
        "rows": sum(f["rows"] for f in per_file),
        "columns": first,
        "columns_not_in_every_file": sorted(in_any_file - in_every_file),
        "missing_read_columns": [c for c in READ_COLUMNS if c not in in_every_file],
        "_in_every_file": in_every_file,
    }


def schema_problems(config: str, inspection: Dict[str, Any]) -> List[str]:
    """Plain-language problems that stop a build: required columns absent from any file."""
    problems = []
    for column in REQUIRED_COLUMNS.get(config, ("cnr", "full_text")):
        if column in inspection["_in_every_file"]:
            continue
        lacking = [Path(f["path"]).name for f in inspection["files"] if column not in f["columns"]]
        problems.append(
            f"{config}: required column '{column}' is missing from {', '.join(lacking)}. "
            f"Columns found: {', '.join(inspection['columns']) or '(none)'}."
        )
    return problems


def format_schema_report(config: str, inspection: Dict[str, Any]) -> str:
    lines = [f"[{config}] {len(inspection['files'])} file(s), {inspection['rows']:,} rows"]
    lines += [f"  file: {f['path']} ({f['rows']:,} rows)" for f in inspection["files"]]
    lines.append("  columns (name: type):")
    lines += [f"    {name}: {kind}" for name, kind in inspection["columns"].items()]
    if inspection["columns_not_in_every_file"]:
        lines.append(f"  columns missing from some files: {', '.join(inspection['columns_not_in_every_file'])}")
    missing = inspection["missing_read_columns"]
    lines.append(
        "  expected columns not present (they come out empty): " + (", ".join(missing) if missing else "none")
    )
    for column in REQUIRED_COLUMNS.get(config, ("cnr", "full_text")):
        lines.append(f"  required '{column}': {'present' if column in inspection['_in_every_file'] else 'MISSING'}")
    return "\n".join(lines)


def check_schema(sources: Dict[str, List[Path]]) -> bool:
    """Prints the schema report for each config. Writes nothing. Returns False if anything required is missing."""
    ok = True
    for config, files in sources.items():
        if not files:
            continue
        inspection = inspect_files(files)
        print(format_schema_report(config, inspection))
        for problem in schema_problems(config, inspection):
            ok = False
            print(f"  PROBLEM: {problem}")
    return ok


def iter_source_records(
    files: Sequence[Path], spec: SourceSpec, builder: CorpusBuilder, max_rows: Optional[int]
) -> Iterator[Dict[str, Any]]:
    """Streams selected records from parquet files. The court filter runs inside the reader."""
    import pyarrow.dataset as ds

    dataset = ds.dataset([str(path) for path in files], format="parquet")
    columns = [name for name in READ_COLUMNS if name in dataset.schema.names]
    if "cnr" not in columns or "full_text" not in columns:
        raise SystemExit(
            f"{spec.config}: input files need 'cnr' and 'full_text' columns. "
            f"Columns found: {', '.join(dataset.schema.names)}."
        )

    flt = None
    if spec.court_code is not None:
        if "court_code" not in dataset.schema.names:
            raise SystemExit(
                f"{spec.config}: no 'court_code' column to select {spec.court_code}. "
                f"Columns found: {', '.join(dataset.schema.names)}."
            )
        flt = ds.field("court_code") == spec.court_code
    total = dataset.count_rows()
    selected = dataset.count_rows(filter=flt) if flt is not None else total
    builder.note_prefiltered(spec.config, total - selected)

    yielded = 0
    for batch in dataset.scanner(columns=columns, filter=flt, batch_size=2048).to_batches():
        for record in batch.to_pylist():
            if max_rows is not None and yielded >= max_rows:
                return  # rows never read are not counted as scanned; the report flags the cap
            yielded += 1
            yield record


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    return " ".join(text[:limit].split())


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def commit_staged(stage: Path, out_dir: Path) -> None:
    """Moves a finished build from the staging folder into place; the report moves last.

    The old report is removed first, so a crash part-way leaves data with NO report (visibly
    incomplete) and never new data beside an old report.
    """
    report = out_dir / "build_report.json"
    if report.exists():
        report.unlink()
    for name in OUTPUT_FILES:
        if name != "build_report.json":
            os.replace(stage / name, out_dir / name)
    os.replace(stage / "build_report.json", report)


def build(
    sources: Dict[str, List[Path]],
    out_dir: Path,
    config: BuildConfig,
    seed: int = 42,
    date_sample_size: int = 200,
    short_sample_size: int = 100,
    max_rows: Optional[int] = None,
    overwrite: bool = False,
    court_code: str = "27_1",
    command: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Runs the build and returns the report that is also written to build_report.json."""
    assert_v2_output_dir(out_dir)
    # Check the input columns first, from metadata only, so a schema problem leaves no partial outputs.
    problems = []
    for config_name, files in sources.items():
        if files:
            problems += schema_problems(config_name, inspect_files(files))
    if problems:
        raise SystemExit("Cannot build:\n  " + "\n  ".join(problems))
    created_out_dir = not out_dir.exists()
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [name for name in OUTPUT_FILES if (out_dir / name).exists()]
    if existing and not overwrite:
        raise SystemExit(
            f"{out_dir} already holds {', '.join(existing)}. Use --overwrite or a new --out-dir."
        )
    # Everything is written under a staging folder and moved into place only after the whole build
    # succeeds, so a failed or interrupted build cannot leave new partial files beside an old report.
    stage = out_dir / STAGING_DIRNAME
    shutil.rmtree(stage, ignore_errors=True)  # left over from a build that was killed
    stage.mkdir()

    specs = default_sources(court_code)
    builder = CorpusBuilder(config)
    schema = arrow_schema(OUTPUT_FIELDS)
    link_schema = arrow_schema([("duplicate_cnr", "string"), ("kept_cnr", "string"), ("source_config", "string")])
    search_writer = BufferedParquetWriter(stage / "search_corpus.parquet", schema)
    lookup_writer = BufferedParquetWriter(stage / "lookup_only.parquet", schema, flush_rows=50_000)
    link_writer = BufferedParquetWriter(stage / "duplicate_links.parquet", link_schema, flush_rows=50_000)
    date_sampler = ReservoirSampler(date_sample_size, seed)
    short_sampler = ReservoirSampler(short_sample_size, seed)

    input_files: Dict[str, List[str]] = {}
    try:
        for config_name, files in sources.items():
            if not files:
                continue
            spec = specs[config_name]
            input_files[config_name] = [str(path) for path in files]
            print(f"[{config_name}] reading {len(files)} file(s)")
            for record in iter_source_records(files, spec, builder, max_rows):
                result = builder.process(record, spec)
                if result.route == "dropped":
                    if result.drop_reason == "near_duplicate":
                        link_writer.add(
                            {
                                "duplicate_cnr": result.cnr,
                                "kept_cnr": result.duplicate_of,
                                "source_config": config_name,
                            }
                        )
                    continue
                row = result.row
                assert row is not None
                (search_writer if result.route == "search" else lookup_writer).add(row)
                if row["date_conflict"]:
                    date_sampler.add(
                        {
                            "cnr": row["cnr"],
                            "bench": row["bench"],
                            "decision_date": row["decision_date_raw"],
                            "cnr_date": row["cnr_date_raw"],
                            "year_conflict": row["year_conflict"],
                            "text_start": snippet(row["text"]),
                        }
                    )
                if result.route == "lookup_only":
                    short_sampler.add(
                        {
                            "cnr": row["cnr"],
                            "bench": row["bench"],
                            "title": row["title"],
                            "clean_chars": row["clean_chars"],
                            "text": row["text"],
                            "your_label": "",  # filled in by the owner during plan task P3
                        }
                    )
        search_writer.close()
        lookup_writer.close()
        link_writer.close()

        write_csv(
            stage / "date_disagreements_sample.csv",
            date_sampler.items,
            ["cnr", "bench", "decision_date", "cnr_date", "year_conflict", "text_start"],
        )
        write_csv(
            stage / "short_orders_sample.csv",
            short_sampler.items,
            ["cnr", "bench", "title", "clean_chars", "text", "your_label"],
        )

        settings = {
            key: (value.isoformat() if isinstance(value, dt.date) else value)
            for key, value in vars(config).items()
        }
        report = {
            "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "command": list(command) if command else None,
            "settings": settings,
            "bombay_court_code": court_code,
            "max_rows_per_config": max_rows,
            "partial_build": max_rows is not None,
            "input_files": input_files,
            "outputs": {
                "search_corpus_rows": search_writer.count,
                "lookup_only_rows": lookup_writer.count,
                "duplicate_links_rows": link_writer.count,
            },
            "samples": {
                "date_disagreements_seen": date_sampler.seen,
                "date_disagreements_written": len(date_sampler.items),
                "short_orders_seen": short_sampler.seen,
                "short_orders_written": len(short_sampler.items),
                "seed": seed,
            },
            "stats": builder.stats.to_dict(),
            "notes": [
                "bench names for newas and newos are inferred, not confirmed (docs/DATA_PROFILE_BOMBAY.md).",
                "Near duplicates are removed from the search corpus only; lookup-only records are never "
                "de-duplicated because identical short orders in different cases are still different cases.",
                "Drop reasons are exclusive: each scanned record is counted once.",
            ],
        }
        (stage / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        commit_staged(stage, out_dir)
    except BaseException:  # includes Ctrl+C: leave the previous outputs, if any, exactly as they were
        for writer in (search_writer, lookup_writer, link_writer):
            try:
                writer.close()
            except Exception:  # noqa: BLE001 - the original error matters more than a failed close
                pass
        shutil.rmtree(stage, ignore_errors=True)
        if created_out_dir:
            try:
                out_dir.rmdir()  # only succeeds if empty
            except OSError:
                pass
        raise
    shutil.rmtree(stage, ignore_errors=True)
    return report


def _share_text(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f} ({value * 100:.2f}%)"


def print_summary(report: Dict[str, Any]) -> None:
    stats = report["stats"]
    print("\nBuild summary")
    print(f"  scanned records            {stats['scanned']:>12,}")
    for reason, count in stats["dropped_by_reason"].items():
        print(f"  dropped: {reason:<19}{count:>12,}")
    print(f"  kept: search index         {stats['kept_search_index']:>12,}")
    print(f"  kept: lookup only          {stats['kept_lookup_only']:>12,}")
    print(f"  counts reconcile           {stats['identity_holds']}")
    print(f"  date conflicts (kept rows) {stats['date_conflicts']:>12,}")
    print(f"  repeat-cleaned records     {stats['repeat_cleaned_records']:>12,}")
    print("  Rates to compare with docs/DATA_PROFILE_BOMBAY.md:")
    print(f"    near-duplicate share of reasoned  {_share_text(stats['near_duplicate_share_of_reasoned'])}")
    print(f"    date-conflict share of kept       {_share_text(stats['date_conflict_share_of_kept'])}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--high-courts", nargs="*", default=[], help="parquet files, folders or globs")
    parser.add_argument("--supreme-court", nargs="*", default=[], help="parquet files, folders or globs")
    parser.add_argument("--from-hub", action="store_true", help="download the parquet files from the Hub")
    parser.add_argument(
        "--check-schema",
        action="store_true",
        help="print files, row counts, columns and types per config (metadata only) and exit; "
        "writes nothing. With --from-hub the files are downloaded first.",
    )
    parser.add_argument("--hub-repo", default=HUB_REPO)
    parser.add_argument("--splits", nargs="+", default=list(HUB_SPLITS))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--court-code", default="27_1", help="High Court code to select (Bombay = 27_1)")
    parser.add_argument("--min-chars", type=int, default=5000, help="cut for the search index")
    parser.add_argument("--date-rule", choices=("prefer_decision", "prefer_cnr", "agree_only"), default="prefer_decision")
    parser.add_argument("--require-date", action="store_true", help="drop reasoned records with no valid date")
    parser.add_argument("--dedup-mode", choices=("prefix", "exact", "off"), default="prefix")
    parser.add_argument("--dedup-prefix-chars", type=int, default=2000)
    parser.add_argument("--no-repeat-clean", action="store_true")
    parser.add_argument("--repeat-head-chars", type=int, default=3000)
    parser.add_argument("--repeat-min-words", type=int, default=3)
    parser.add_argument("--repeat-max-words", type=int, default=60)
    parser.add_argument("--today", help="YYYY-MM-DD used to reject future dates (default: today)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--date-sample-size", type=int, default=200)
    parser.add_argument("--short-sample-size", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=None, help="cap on selected rows per config (smoke test)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    sources: Dict[str, List[Path]] = {
        "high_courts": expand_inputs(args.high_courts),
        "supreme_court": expand_inputs(args.supreme_court),
    }
    if args.from_hub:
        for config_name in sources:
            if not sources[config_name]:
                sources[config_name] = download_from_hub(config_name, args.splits, args.hub_repo)
    if not any(sources.values()):
        raise SystemExit("No input files. Pass --high-courts/--supreme-court paths or use --from-hub.")
    if args.check_schema:
        raise SystemExit(0 if check_schema(sources) else 1)

    config = BuildConfig(
        min_chars=args.min_chars,
        date_rule=args.date_rule,
        require_date=args.require_date,
        dedup_mode=args.dedup_mode,
        dedup_prefix_chars=args.dedup_prefix_chars,
        clean_repeats=not args.no_repeat_clean,
        repeat_head_chars=args.repeat_head_chars,
        repeat_min_words=args.repeat_min_words,
        repeat_max_words=args.repeat_max_words,
        today=dt.date.fromisoformat(args.today) if args.today else None,
    )
    report = build(
        sources=sources,
        out_dir=args.out_dir,
        config=config,
        seed=args.seed,
        date_sample_size=args.date_sample_size,
        short_sample_size=args.short_sample_size,
        max_rows=args.max_rows,
        overwrite=args.overwrite,
        court_code=args.court_code,
        command=sys.argv,
    )
    print_summary(report)
    print(f"\nWrote {args.out_dir}/build_report.json")


if __name__ == "__main__":
    main()
