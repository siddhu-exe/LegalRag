"""
Builds the IPC/BNS, CrPC/BNSS and Evidence Act/BSA mapping data file (plan task C6).

The mapping must come from the OFFICIAL correspondence tables and be checked by a person; this script
does not contain, guess or download any section data. You prepare a CSV from the official table (the
Ministry of Home Affairs "Correspondence Table and Comparison Summary" documents for BNS-IPC and
BNSS-CrPC, and the corresponding table for BSA-Evidence Act) and this script validates it and writes the
JSON file the search uses. A file with no ``--reviewed-by`` is written but the search refuses to load it.

CSV columns (header required; extra columns are ignored):

    pair            IPC-BNS, CRPC-BNSS or IEA-BSA   (or pass --pair for a file with one pair)
    old_section     the old-law section, for example 302, 498A, 438(2). Empty if the new provision is new.
    new_section     the new-law section, for example 103(1). Empty if the old provision was omitted.
    relationship    same, changed, split or merged when both sides are filled; omitted or new otherwise
                    (optional: derived from which side is empty, or "same")
    note            free text, for example "punishment moved to sub-section (2)"

A one-to-many correspondence is several rows. Section numbers are written as in the table; case and
spaces are normalised (498a -> 498A). Every problem in the file is listed at once, with line numbers.

Examples:
    python scripts/build_statute_map.py --csv work/ipc_bns.csv work/crpc_bnss.csv work/iea_bsa.csv \\
        --source-url "<url of each official table>" --retrieved-on 2026-11-04 --out artifacts/v2/statute_map.json

    # after a lawyer or you have checked every row against the official table:
    python scripts/build_statute_map.py --csv work/*.csv --reviewed-by "Your Name" --reviewed-on 2026-11-10 \\
        --source-url "<url>" --out artifacts/v2/statute_map.json

Turn it on with ARTIFACT_DIR=artifacts/v2 and STATUTE_EXPANSION=true.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from legalrag.retrieval.statute_map import FORMAT_VERSION, PAIRS, StatuteMapError, StatuteMap, sha256_of, validate_row
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from legalrag.retrieval.statute_map import (  # noqa: E402
        FORMAT_VERSION, PAIRS, StatuteMapError, StatuteMap, sha256_of, validate_row,
    )


def assert_v2_path(path: Path) -> None:
    if "v2" not in path.resolve().parts and "v2" not in path.parts:
        raise SystemExit(f"Refusing to write to {path}: the output path must have a 'v2' directory part (for example artifacts/v2/statute_map.json).")


def read_csv(path: Path, default_pair: Optional[str]) -> tuple:
    """Returns ``(rows, errors)``. Every row is validated; errors carry the file and line number."""
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = [h.strip() for h in (reader.fieldnames or [])]
        reader.fieldnames = header
        needed = {"old_section", "new_section"}
        if not needed <= set(header):
            return [], [f"{path.name}: header must include {sorted(needed)}; found {header}"]
        if "pair" not in header and not default_pair:
            return [], [f"{path.name}: no 'pair' column and no --pair given"]
        for line, record in enumerate(reader, start=2):
            if not any((v or "").strip() for v in record.values()):
                continue  # a blank line
            raw = {
                "pair": (record.get("pair") or default_pair or "").strip().upper(),
                "old": record.get("old_section"),
                "new": record.get("new_section"),
                "relationship": (record.get("relationship") or "").strip().lower() or None,
                "note": (record.get("note") or "").strip(),
            }
            try:
                row = validate_row(raw)
            except StatuteMapError as exc:
                errors.append(f"{path.name} line {line}: {exc}")
                continue
            rows.append({"pair": row.pair, "old": row.old, "new": row.new, "relationship": row.relationship, "note": row.note})
    return rows, errors


def build(
    csv_paths: Sequence[Path],
    out: Path,
    default_pair: Optional[str] = None,
    source_urls: Sequence[str] = (),
    retrieved_on: Optional[str] = None,
    reviewed_by: Optional[str] = None,
    reviewed_on: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    assert_v2_path(out)
    if out.exists() and not overwrite:
        raise SystemExit(f"{out} already exists. Use --overwrite or a new --out.")
    all_rows: List[Dict[str, Any]] = []
    errors: List[str] = []
    sources: List[Dict[str, Any]] = []
    for path in csv_paths:
        rows, problems = read_csv(Path(path), default_pair)
        all_rows.extend(rows)
        errors.extend(problems)
        sources.append({"file": Path(path).name, "sha256": sha256_of(path), "rows": len(rows)})
    if not all_rows and not errors:
        errors.append("the CSV files contain no rows")

    seen = Counter((r["pair"], r["old"], r["new"]) for r in all_rows)
    errors.extend(f"duplicate row {key} appears {n} times" for key, n in seen.items() if n > 1)
    if errors:
        raise SystemExit("Cannot build the statute map:\n  " + "\n  ".join(errors))

    all_rows.sort(key=lambda r: (r["pair"], r["old"] or "", r["new"] or ""))
    document = {
        "format": FORMAT_VERSION,
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source_urls": list(source_urls),
        "retrieved_on": retrieved_on,
        "csv_files": sources,
        "reviewed_by": (reviewed_by or "").strip() or None,
        "reviewed_on": reviewed_on if reviewed_by else None,
        "rows": all_rows,
    }
    StatuteMap.from_dict(document, require_reviewed=False)  # the file must load before it is written
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, out)

    report = {
        "out": str(out),
        "rows": len(all_rows),
        "rows_by_pair": dict(Counter(r["pair"] for r in all_rows)),
        "rows_by_relationship": dict(Counter(r["relationship"] for r in all_rows)),
        "reviewed_by": document["reviewed_by"],
    }
    print(json.dumps(report, indent=2))
    if not document["reviewed_by"]:
        print("NOT REVIEWED: the search will refuse to load this file until it is rebuilt with --reviewed-by.")
    missing_pairs = sorted(set(PAIRS) - set(report["rows_by_pair"]))
    if missing_pairs:
        print(f"Note: no rows for {', '.join(missing_pairs)}.")
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", nargs="+", type=Path, required=True)
    parser.add_argument("--pair", choices=sorted(PAIRS), default=None, help="pair for CSV files without a 'pair' column")
    parser.add_argument("--out", type=Path, default=Path("artifacts/v2/statute_map.json"))
    parser.add_argument("--source-url", action="append", default=[], help="URL of an official table used (repeat for each)")
    parser.add_argument("--retrieved-on", default=None, help="date the official table was downloaded, YYYY-MM-DD")
    parser.add_argument("--reviewed-by", default=None, help="the person who checked every row against the official table")
    parser.add_argument("--reviewed-on", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    build(args.csv, args.out, args.pair, args.source_url, args.retrieved_on, args.reviewed_by, args.reviewed_on, args.overwrite)


if __name__ == "__main__":
    main()
