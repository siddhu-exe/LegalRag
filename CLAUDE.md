# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LegalRAG is a retrieval-augmented generation (RAG) system built over Indian High Court judgments sourced from the Hugging Face dataset `overthelex/indian-court-decisions` (`high_courts` config). The codebase operates across two environments:
- **Local environment (this repo)**: Lightweight data collection, profiling, exploratory scripts, and notebook generator scripts.
- **Remote GPU environment (Kaggle / Colab)**: Compute-heavy pipeline execution (corpus cleaning, locked baseline chunking, Gemini-powered gold eval benchmark generation, embedding generation with `bge-small-en-v1.5`, FAISS indexing, and Recall@K retrieval benchmarking).

## Environment Setup

- **Local Python Environment**: Virtual environment located at `rag/` (Python 3.12, managed with `uv`).
- **Dependencies**: There is no `requirements.txt` or `pyproject.toml`.
  - Local dependencies: `datasets`, `pandas`, `pyarrow`, `numpy`, `tqdm`, `huggingface_hub`, `httpx`.
  - Adding local packages: `source rag/bin/activate && uv pip install <package>`
  - Remote/Notebook-only dependencies: `torch`, `sentence-transformers`, `faiss-gpu`, `langchain-text-splitters`, `rank-bm25`, `google-genai`, `psutil`.
- **Hugging Face Authentication**: Required for accessing the gated dataset (`hf auth login` or `huggingface_hub.login()`).

## Common Commands

### Activate Environment
```bash
source rag/bin/activate
```

### Data Collection & Profiling
```bash
# Collect 5,000 judgment subset (simple stream to data/raw/judgments.parquet)
python download_subset.py

# Collect 20,000 balanced judgment corpus (with exact-text dedup and 300-1500 per-court bounds)
python collect_corpus.py

# Profile dataset shape, missing values, duplicates, text length distributions, and year/court counts
python profile_data.py

# Analyze CNR prefix court distribution
python analyze_courts.py
```

### Notebook Generation
```bash
# Generates/updates Notebooks/legalrag_refactored.ipynb
python write_notebook.py
```

## Architecture & Pipeline

```
LegalRAG/
├── download_subset.py    # First-pass 5k sample collector → data/raw/judgments.parquet
├── collect_corpus.py     # Balanced 20k collector with exact dedup & per-court limits
├── profile_data.py       # Dataset profiling script (reads data/raw/judgments.parquet)
├── analyze_courts.py     # Analyzes CNR prefix (court code) distribution
├── write_notebook.py     # Programmatic generator for Kaggle/Colab pipeline notebook
├── data/
│   ├── raw/              # Raw collected Parquet files
│   └── processed/        # Cleaned and chunked corpus artifacts
├── Notebooks/
│   ├── legalrag.ipynb            # Original experimental pipeline notebook
│   ├── legalrag_refactored.ipynb # Idempotent, checkpointed pipeline notebook
│   └── legalrag-100k-final.ipynb # 100k scale experimental notebook
└── rag/                  # Local Python virtual environment
```

### Pipeline Stages
1. **Streaming Collection**: Uses Hugging Face `load_dataset(..., streaming=True)` with `.shuffle(seed=42, buffer_size=10_000)`. Never materializes the entire raw dataset locally.
2. **Quality Cleaning**: Drops documents where control characters exceed 1% of text length (handles known corrupt/Caesar-shifted entries); strips remaining control characters and normalizes whitespace.
3. **Chunking (Locked Baseline)**: 1,200 character chunk size, 200 character overlap, minimum chunk length 100 characters.
4. **Eval Benchmark Generation**: 100 sampled judgments evaluated with structured question generation (`gemini-3.5-flash-lite`), mapping `supporting_text` to gold chunk IDs.
5. **Retrieval Baselines**:
   - **BM25**: Tokenized corpus indexed via `rank-bm25`.
   - **Dense**: `BAAI/bge-small-en-v1.5` embeddings (384-dim, normalized) indexed with FAISS (`IndexFlatIP`).
   - **Evaluation Metric**: Recall@K (K = 1, 3, 5, 10) against gold chunk IDs.

## Development Conventions

- **Random Seed**: Always use `seed=42` for shuffling, sampling, and splits.
- **File Storage**: Use Apache Parquet (`.parquet`) for tabular data, always saved with `index=False`. Use JSON for eval benchmarks and metrics.
- **Court Identification**: Use `court_code` column first; fall back to the first 4 characters of `cnr` if `court_code` is missing.
- **Remote vs Local Paths**: Paths starting with `/kaggle/working/` or `/content/` are meant for the remote execution environment and do not exist on the local filesystem.
- **Testing**: There is no automated unit test suite (`pytest`/`unittest`). Verification is done by executing standalone scripts (`profile_data.py`, `analyze_courts.py`) and inspecting stdout metrics.
