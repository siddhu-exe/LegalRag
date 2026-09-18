import pandas as pd

PATH = "data/raw/judgments.parquet"

df = pd.read_parquet(PATH)

print("\n=== DATASET SHAPE ===")
print(df.shape)

print("\n=== COLUMNS ===")
print(df.columns.tolist())

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== DUPLICATE ROWS ===")
print(df.duplicated().sum())

print("\n=== TEXT LENGTH ===")
text_lengths = df["full_text"].fillna("").str.len()

print("Min:", text_lengths.min())
print("Max:", text_lengths.max())
print("Mean:", round(text_lengths.mean()))
print("Median:", text_lengths.median())

print("\n=== TEXT LENGTH PERCENTILES ===")
print(text_lengths.quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))

print("\n=== COURTS ===")
print(df["court_name"].value_counts())

print("\n=== YEARS ===")
print(df["year"].value_counts().sort_index())

print("\n=== EMPTY TEXT ===")
print((df["full_text"].fillna("").str.strip() == "").sum())

print("\n=== DUPLICATE TEXT ===")
print(df["full_text"].duplicated().sum())
