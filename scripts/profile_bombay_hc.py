"""
Profiles Bombay High Court and Supreme Court coverage in the source dataset.

Queries the Hugging Face parquet files remotely with DuckDB (no full download) and writes
results/data_profile/bombay_profile.json. docs/DATA_PROFILE_BOMBAY.md quotes this output.

Requires DuckDB, which is not a project dependency:
    pip install duckdb
    python scripts/profile_bombay_hc.py
"""

import collections
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import duckdb

BASE = "hf://datasets/overthelex/indian-court-decisions"
SPLITS = ("train", "validation", "test")
BOMBAY_COURT_CODE = "27_1"
REASONED_MIN_CHARS = 5000
SAMPLE_MODULUS = 40
OUTPUT_PATH = Path("results/data_profile/bombay_profile.json")

BENCH_LABELS = {
    "newas": "Bombay, Appellate Side (HCBM01, inferred)",
    "newos": "Bombay, Original Side (HCBM02, inferred)",
    "hcaurdb": "Aurangabad (HCBM03)",
    "testcase": "Nagpur (HCBM04, confirmed from sampled judgment headers)",
    "hcbgoa": "Goa (HCBM05)",
    "kolhcdb": "Kolhapur (HCBM07)",
}


def parquet_source(config: str) -> str:
    """Returns a DuckDB read_parquet expression covering all splits of a config."""
    paths = ",".join(f"'{BASE}/{config}/{split}.parquet'" for split in SPLITS)
    return f"read_parquet([{paths}])"


def fetch_dicts(connection: Any, query: str) -> List[Dict[str, Any]]:
    """Runs a query and returns its rows as dictionaries."""
    cursor = connection.execute(query)
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def profile_text_sample(connection: Any) -> Dict[str, Any]:
    """Measures text quality on a hash-based random sample of the Bombay test split."""
    rows = connection.execute(
        f"""
        SELECT bench, full_text
        FROM read_parquet('{BASE}/high_courts/test.parquet')
        WHERE court_code = '{BOMBAY_COURT_CODE}' AND hash(cnr) % {SAMPLE_MODULUS} = 0
        """
    ).fetchall()

    repeated = 0
    non_english = 0
    fingerprints: collections.Counter = collections.Counter()
    nagpur_headers: collections.Counter = collections.Counter()
    bench_totals: collections.Counter = collections.Counter()

    for bench, text in rows:
        head = text[:3000]
        words = head.split()
        grams = [" ".join(words[i : i + 4]) for i in range(max(len(words) - 4, 0))]
        if grams and max(collections.Counter(grams).values()) >= 3:
            repeated += 1
        if sum(ord(char) > 127 for char in head) / max(len(head), 1) > 0.2:
            non_english += 1
        fingerprints[re.sub(r"\W+", "", text).lower()[:2000]] += 1
        bench_totals[bench] += 1
        if "NAGPUR" in text[:400].upper():
            nagpur_headers[bench] += 1

    total = len(rows)
    return {
        "sample_rows": total,
        "repeated_4gram_share": round(repeated / total, 3),
        "non_english_share": round(non_english / total, 3),
        "near_duplicate_share": round(sum(n - 1 for n in fingerprints.values()) / total, 3),
        "nagpur_in_header_by_bench": {
            bench: f"{nagpur_headers[bench]}/{bench_totals[bench]}" for bench in bench_totals
        },
    }


def main() -> None:
    connection = duckdb.connect()
    connection.execute("INSTALL httpfs; LOAD httpfs; SET threads=8;")
    high_courts = parquet_source("high_courts")
    supreme_court = parquet_source("supreme_court")
    bombay = f"{high_courts} WHERE court_code = '{BOMBAY_COURT_CODE}'"
    keep = f"text_length >= {REASONED_MIN_CHARS}"

    profile: Dict[str, Any] = {"reasoned_min_chars": REASONED_MIN_CHARS}

    profile["benches"] = fetch_dicts(
        connection,
        f"""
        SELECT bench, substr(cnr, 1, 6) AS cnr_prefix, count(*) AS records,
               sum(({keep})::int) AS reasoned_records,
               sum(text_length) AS chars,
               sum(CASE WHEN {keep} THEN text_length ELSE 0 END) AS reasoned_chars,
               median(text_length) AS median_chars,
               sum((text_length < 2000)::int) AS under_2k
        FROM {bombay} GROUP BY 1, 2 ORDER BY records DESC
        """,
    )
    for row in profile["benches"]:
        row["label"] = BENCH_LABELS.get(row["bench"], "unknown")

    profile["years"] = fetch_dicts(
        connection,
        f"""
        SELECT CASE WHEN y < 2010 THEN 'before 2010' WHEN y < 2015 THEN '2010-14'
                    WHEN y < 2020 THEN '2015-19' WHEN y < 2024 THEN '2020-23'
                    ELSE '2024+' END AS period,
               count(*) AS records, sum((text_length >= {REASONED_MIN_CHARS})::int) AS reasoned_records,
               sum(CASE WHEN {keep} THEN text_length ELSE 0 END) AS reasoned_chars
        FROM (SELECT try_cast(substr(split_part(cnr, '_', 3), 1, 4) AS int) AS y, text_length
              FROM {bombay})
        GROUP BY 1 ORDER BY 1
        """,
    )

    profile["metadata"] = fetch_dicts(
        connection,
        f"""
        SELECT count(*) AS records, count(DISTINCT cnr) AS unique_cnr,
               sum((title IS NULL OR title = '')::int) AS blank_title,
               sum((judge IS NULL OR judge = '')::int) AS blank_judge,
               sum((decision_date IS NULL OR decision_date = '')::int) AS blank_date,
               sum((court_name IS NULL OR court_name = '')::int) AS blank_court_name,
               sum((case_type IS NULL OR case_type = '')::int) AS blank_case_type,
               sum((disposal_nature_normalized IS NULL OR disposal_nature_normalized IN ('', 'other'))::int)
                   AS blank_or_other_outcome,
               min(decision_date) AS earliest_date, max(decision_date) AS latest_date
        FROM {bombay}
        """,
    )[0]

    profile["date_check"] = fetch_dicts(
        connection,
        f"""
        SELECT sum((s = d)::int) AS date_matches_cnr,
               sum((s <> d)::int) AS date_differs_from_cnr,
               sum((substr(s, 1, 4) <> substr(d, 1, 4))::int) AS year_differs_from_cnr,
               max(s) AS latest_cnr_date
        FROM (SELECT split_part(cnr, '_', 3) AS s, decision_date AS d FROM {bombay})
        """,
    )[0]

    profile["supreme_court"] = fetch_dicts(
        connection,
        f"""
        SELECT count(*) AS records, min(year) AS first_year, max(year) AS last_year,
               sum(({keep})::int) AS reasoned_records,
               sum(CASE WHEN {keep} THEN text_length ELSE 0 END) AS reasoned_chars
        FROM {supreme_court}
        """,
    )[0]

    profile["court_code_check"] = fetch_dicts(
        connection,
        f"""
        SELECT court_code, substr(cnr, 1, 4) AS cnr_prefix, count(*) AS records
        FROM {high_courts}
        WHERE court_code IN ('10_8', '18_6', '19_16', '1_12', '24_17', '22_18', '16_20',
                             '14_25', '27_1')
        GROUP BY 1, 2 ORDER BY 1, records DESC
        """,
    )

    profile["text_sample"] = profile_text_sample(connection)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(profile, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
