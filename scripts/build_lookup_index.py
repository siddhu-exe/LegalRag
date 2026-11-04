"""
Builds the case lookup index from the corpus build output (plan task C7). Run on your PC.

Reads search_corpus.parquet and lookup_only.parquet (metadata columns only, never the judgment text) and
duplicate_links.parquet from the C1 build, and writes one SQLite file:

    python scripts/build_lookup_index.py --corpus-dir artifacts/v2/corpus --out artifacts/v2/lookup.sqlite

Every record the build kept is included, not just the search index, so a judge, a date or a CNR can be
looked up even for short orders. CNRs that the build removed as near-duplicates are included as links to
the record that was kept. The row count is checked against build_report.json, and a partial build
(--max-rows) is refused unless you pass --allow-partial.

Turn the endpoint on with ARTIFACT_DIR=artifacts/v2 and LOOKUP_ENABLED=true.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

try:
    from legalrag.lookup.index import COLUMNS, CaseLookupIndex
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from legalrag.lookup.index import COLUMNS, CaseLookupIndex  # noqa: E402

CORPUS_FILES = ("search_corpus.parquet", "lookup_only.parquet")


def assert_v2_path(path: Path) -> None:
    if "v2" not in path.resolve().parts and "v2" not in path.parts:
        raise SystemExit(f"Refusing to write to {path}: the output path must have a 'v2' directory part (for example artifacts/v2/lookup.sqlite).")


def iter_rows(corpus_dir: Path, batch_rows: int = 20_000) -> Iterator[Dict[str, Any]]:
    import pyarrow.parquet as pq

    for name in CORPUS_FILES:
        path = corpus_dir / name
        if not path.is_file():
            raise SystemExit(f"{path} not found. Run scripts/build_corpus_v2.py first.")
        parquet = pq.ParquetFile(str(path))
        names = set(parquet.schema_arrow.names)
        if "cnr" not in names:
            raise SystemExit(f"{path} has no 'cnr' column. Columns found: {', '.join(parquet.schema_arrow.names)}.")
        columns = [c for c in COLUMNS if c in names]  # the judgment text is never read
        for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
            yield from batch.to_pylist()


def read_links(corpus_dir: Path) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq

    path = corpus_dir / "duplicate_links.parquet"
    if not path.is_file():
        return []
    return pq.read_table(str(path), columns=["duplicate_cnr", "kept_cnr"]).to_pylist()


def build(corpus_dir: Path, out: Path, overwrite: bool = False, allow_partial: bool = False) -> Dict[str, Any]:
    assert_v2_path(out)
    if out.exists() and not overwrite:
        raise SystemExit(f"{out} already exists. Use --overwrite or a new --out.")
    report_path = corpus_dir / "build_report.json"
    expected: Optional[int] = None
    if report_path.is_file():
        build_report = json.loads(report_path.read_text(encoding="utf-8"))
        if build_report.get("partial_build") and not allow_partial:
            raise SystemExit("The corpus build was partial (--max-rows). Rebuild the corpus, or pass --allow-partial for a trial.")
        outputs = build_report.get("outputs", {})
        expected = int(outputs.get("search_corpus_rows", 0)) + int(outputs.get("lookup_only_rows", 0))
    else:
        print("Warning: no build_report.json, so the row count cannot be cross-checked.")

    started = time.perf_counter()
    links = read_links(corpus_dir)
    staging = out.with_name(out.name + ".staging")
    staging.unlink(missing_ok=True)
    index = CaseLookupIndex.build(staging, iter_rows(corpus_dir), links)
    records = len(index)
    index.close()
    if expected is not None and records != expected:
        staging.unlink(missing_ok=True)
        raise SystemExit(f"The index has {records:,} records but build_report.json says the corpus has {expected:,}; not writing it.")
    staging.replace(out)
    report = {
        "out": str(out), "records": records, "links": len(links), "checked_against_build_report": expected is not None,
        "seconds": round(time.perf_counter() - started, 1), "coverage": CaseLookupIndex(out).coverage(),
    }
    print(json.dumps(report, indent=2))
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-dir", type=Path, default=Path("artifacts/v2/corpus"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/v2/lookup.sqlite"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    build(args.corpus_dir, args.out, args.overwrite, args.allow_partial)


if __name__ == "__main__":
    main()
