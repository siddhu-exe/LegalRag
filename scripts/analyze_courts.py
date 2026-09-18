import pandas as pd

df = pd.read_parquet("data/raw/judgments.parquet")

# First 4 characters identify the court in these CNRs
df["cnr_prefix"] = df["cnr"].str[:4]

print("\n=== CNR PREFIX DISTRIBUTION ===")
print(df["cnr_prefix"].value_counts())

print("\n=== UNIQUE CNR PREFIXES ===")
print(df["cnr_prefix"].nunique())

print("\n=== TOP 20 PREFIXES ===")
print(df["cnr_prefix"].value_counts().head(20))
