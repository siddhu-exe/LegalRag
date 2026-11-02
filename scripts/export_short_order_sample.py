"""
Exports a random sample of short Bombay HC orders for human review (plan task P3).

The review tests whether the 5,000-character search cut drops real legal reasoning. Sampling
uses the dataset's test split, whose membership is set by a hash of the CNR, so it needs no full
download. The reviewer fills in the two blank columns; nothing here labels the rows.

Requires DuckDB, which is not a project dependency:
    pip install duckdb
    python scripts/export_short_order_sample.py
"""

import csv
import random
from pathlib import Path

import duckdb

SOURCE = "hf://datasets/overthelex/indian-court-decisions/high_courts/test.parquet"
BOMBAY_COURT_CODE = "27_1"
REASONED_MIN_CHARS = 5000
SAMPLE_SIZE = 100
SEED = 42
OUTPUT_PATH = Path("data/eval_v2/short_order_review.csv")

BENCH_NAMES = {
    "newas": "Bombay Appellate",
    "newos": "Bombay Original",
    "hcaurdb": "Aurangabad",
    "testcase": "Nagpur",
    "hcbgoa": "Goa",
    "kolhcdb": "Kolhapur",
}


def main() -> None:
    connection = duckdb.connect()
    connection.execute("INSTALL httpfs; LOAD httpfs; SET threads=8;")
    rows = connection.execute(
        f"""
        SELECT cnr, bench, decision_date, title, text_length, full_text
        FROM read_parquet('{SOURCE}')
        WHERE court_code = '{BOMBAY_COURT_CODE}' AND text_length < {REASONED_MIN_CHARS}
        ORDER BY cnr
        """
    ).fetchall()

    sample = random.Random(SEED).sample(rows, SAMPLE_SIZE)
    sample.sort(key=lambda row: row[4])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.writer(output)
        writer.writerow(
            ["has_legal_reasoning", "notes", "chars", "bench", "date", "title", "cnr", "text"]
        )
        for cnr, bench, date, title, length, text in sample:
            writer.writerow(["", "", length, BENCH_NAMES.get(bench, bench), date, title, cnr, text])

    print(f"Sampled {len(sample)} of {len(rows)} short orders. Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
