from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


# -----------------------------
# Configuration
# -----------------------------

DATASET = "overthelex/indian-court-decisions"
CONFIG = "high_courts"
SPLIT = "train"

TARGET_TOTAL = 20000
MIN_PER_COURT = 300
MAX_PER_COURT = 1500

OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "legal_judgments_20k.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Load dataset in streaming mode
# -----------------------------

print("Connecting to Hugging Face...")

ds = load_dataset(
    DATASET,
    CONFIG,
    split=SPLIT,
    streaming=True,
)

# Shuffle the stream so we don't simply take the first block
# of records from the source.
ds = ds.shuffle(seed=42, buffer_size=10000)


# -----------------------------
# Collection state
# -----------------------------

records = []
seen_texts = set()
court_counts = {}

FIELDS = [
    "cnr",
    "source",
    "court_code",
    "court_name",
    "bench",
    "year",
    "full_text",
    "text_length",
    "title",
    "judge",
    "petitioner",
    "respondent",
    "decision_date",
    "disposal_nature",
    "disposal_nature_normalized",
    "case_type",
]


# -----------------------------
# Stream and sample
# -----------------------------

print(f"Target: {TARGET_TOTAL:,} unique judgments")

with tqdm(total=TARGET_TOTAL, desc="Collecting") as pbar:

    for row in ds:

        if len(records) >= TARGET_TOTAL:
            break

        text = row.get("full_text")

        if not text:
            continue

        text = str(text).strip()

        # Exact text deduplication
        if text in seen_texts:
            continue

        # Prefer the dataset's official court code.
        # Fall back to CNR prefix if necessary.
        court = row.get("court_code")

        if not court:
            cnr = str(row.get("cnr", ""))
            court = cnr[:4] if cnr else "UNKNOWN"

        current_count = court_counts.get(court, 0)

        # Hard upper bound per court
        if current_count >= MAX_PER_COURT:
            continue

        # Save record
        record = {
            field: row.get(field)
            for field in FIELDS
        }

        # Keep our derived court identifier
        record["sampling_court"] = court

        records.append(record)
        seen_texts.add(text)

        court_counts[court] = current_count + 1

        pbar.update(1)

# -----------------------------
# Save
# -----------------------------

df = pd.DataFrame(records)

df.to_parquet(
    OUTPUT_FILE,
    index=False,
)

print("\n=== COLLECTION COMPLETE ===")
print(f"Rows saved: {len(df):,}")
print(f"Unique texts: {df['full_text'].nunique():,}")
print(f"File: {OUTPUT_FILE}")

print("\n=== COURT DISTRIBUTION ===")
print(
    df["sampling_court"]
    .value_counts()
    .sort_index()
    .to_string()
)
