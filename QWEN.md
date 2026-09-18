# LegalRAG

Legal-domain RAG (Retrieval-Augmented Generation) system built over Indian High Court judgment texts. The project is currently in the **data/corpus construction and retrieval baselining** phase — no serving or generation layer exists yet.

## Project Overview

- **Source data**: Hugging Face dataset `overthelex/indian-court-decisions`, config `high_courts`, split `train`, consumed in **streaming mode** (never downloaded whole).
- **Pipeline stages** (each stage's output feeds the next):
  1. **Collection** — stream + sample judgments with exact-text deduplication and per-court balancing.
  2. **Profiling / quality analysis** — missing values, text-length distributions, court distribution, corruption detection.
  3. **Cleaning** — drop documents with >1% control characters; normalize whitespace.
  4. **Chunking** — baseline params are **locked**: 1200-char chunks, 200-char overlap, minimum chunk length 100.
  5. **Eval benchmark** — 100 sampled judgments; questions generated with Gemini (`gemini-3.5-flash-lite`, structured JSON via pydantic); gold chunks assigned by exact/fuzzy matching of `supporting_text` against chunks.
  6. **Retrieval baselines** — BM25 done (frozen), dense retrieval (bge-small-en-v1.5 + FAISS) in progress.
- **Two working environments**:
  - **Local** (this repo): lightweight collection/profiling scripts using the `rag/` uv venv.
  - **Kaggle/Colab GPU**: `Notebooks/legalrag.ipynb` — the heavy lifting (50k corpus build, cleaning, chunking, eval generation, embeddings). Notebook paths like `/content/legalrag_corpus/` and `/kaggle/working/` refer to that remote environment, **not** this repo.

## Directory Structure

```
LegalRAG/
├── download_subset.py    # First-pass collector: 5,000 judgments (no dedup/balancing) → data/raw/judgments.parquet
├── collect_corpus.py     # Balanced collector: 20k target, exact dedup, per-court caps (300 min / 1500 max)
├── profile_data.py       # Profiles data/raw/judgments.parquet (shape, nulls, dupes, lengths, courts, years)
├── analyze_courts.py     # CNR-prefix (first 4 chars) distribution analysis
├── Notebooks/
│   └── legalrag.ipynb    # Main experimental pipeline (Kaggle/Colab GPU notebook)
├── data/
│   ├── raw/              # judgments.parquet (5k subset, ~14 MB)
│   └── processed/        # empty (future cleaned/chunked artifacts)
└── rag/                  # uv-managed Python 3.12 virtualenv (gitignored)
```

## Environment

- **Virtualenv**: `rag/` — created with `uv` (Python 3.12.3, system interpreter `/usr/bin/python3`).
- **Local packages**: `datasets`, `pandas`, `pyarrow`, `numpy`, `tqdm`, `huggingface_hub`, `httpx`.
- **Notebook-only packages** (installed via `pip -q install` in cells, not present locally): `torch`, `sentence-transformers`, `faiss-gpu`, `langchain-text-splitters`, `rank-bm25`, `google-genai`, `psutil`.
- There is **no `requirements.txt` or `pyproject.toml`** — dependencies live in the venv and notebook pip cells. If adding local deps, prefer `uv pip install <pkg>` into `rag/`.

## Commands

```bash
# Activate the local environment
source rag/bin/activate

# Collect data (streaming from Hugging Face; writes to data/raw/)
python download_subset.py     # 5k simple subset
python collect_corpus.py      # 20k balanced corpus → data/raw/legal_judgments_20k.parquet

# Profile / analyze
python profile_data.py
python analyze_courts.py
```

Hugging Face auth (`hf auth login` or `login()`) is required for the gated dataset in some environments.

## Development Conventions

- **Random seed 42** everywhere (shuffle, sampling) — keep this consistent for reproducibility.
- **Storage format**: Parquet exclusively, written with `index=False`.
- **Court identity**: use `court_code`; fall back to the first 4 characters of `cnr` if missing (derived `sampling_court` column).
- **Deduplication**: exact `full_text` match via a `seen_texts` set during streaming.
- **Streaming**: always use `load_dataset(..., streaming=True)` with `.shuffle(seed=42, buffer_size=10_000)`; never materialize the full source dataset.
- Scripts are standalone and print analysis directly to stdout (no argparse, no logging config) — follow this style.
- **Metrics**: retrieval quality measured as Recall@K (K = 1, 3, 5, 10) against gold chunk IDs.

## Key Facts & Frozen Results

- **Locked baseline chunking**: 49,629 clean documents → 303,734 chunks (1200 char / 200 overlap / min 100).
- **BM25 baseline (frozen)**: Recall@1 = 0.20, Recall@3 = 0.24, Recall@5 = 0.28, Recall@10 = 0.33.
- **Data corruption quirk**: some source documents are **Caesar-cipher shifted (shift 3)** and/or riddled with control characters. Cleaning rule: drop docs where control chars > 1% of text length; otherwise strip control chars and collapse whitespace.
- **Dense retrieval (in progress)**: `BAAI/bge-small-en-v1.5` with `sentence-transformers` multi-process GPU pooling, `faiss-gpu` for search — notebook ends mid-implementation (embeddings generated, index/eval pending).

## Testing

No test suite exists yet. Verification is done by running the profiling scripts and inspecting printed statistics.

## Current Status / Next Steps

1. Finish dense retrieval baseline (FAISS index + Recall@K evaluation on `gold_eval.json` questions).
2. Compare BM25 vs. dense vs. hybrid retrieval.
3. Bring notebook artifacts (clean corpus, chunks, eval set) back into `data/processed/` locally.
4. Chunking-strategy experiments beyond the locked baseline, then the generation/QA layer.
