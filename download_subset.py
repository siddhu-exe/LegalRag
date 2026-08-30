from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

DATASET = "overthelex/indian-court-decisions"

# Stream instead of downloading the entire dataset
ds = load_dataset(
    DATASET,
    "high_courts",
    split="train",
    streaming=True
)

records = []

TARGET = 5000

for row in tqdm(ds, total=TARGET):
    # Keep only fields we actually need
    records.append({
        "cnr": row["cnr"],
        "court_name": row["court_name"],
        "year": row["year"],
        "full_text": row["full_text"],
        "title": row["title"],
        "judge": row["judge"],
        "decision_date": row["decision_date"],
        "case_type": row["case_type"],
        "disposal_nature": row["disposal_nature"],
    })

    if len(records) >= TARGET:
        break

df = pd.DataFrame(records)

df.to_parquet("data/raw/judgments.parquet", index=False)

print(f"Saved {len(df)} judgments")