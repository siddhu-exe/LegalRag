# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LegalRAG is a retrieval-augmented generation (RAG) system built over Indian High Court judgments sourced from the Hugging Face dataset `overthelex/indian-court-decisions` (`high_courts` config). The codebase operates across two environments:
- **Local environment (this repo)**: Lightweight data collection, profiling, exploratory scripts, modular Python package (`src/legalrag`), and unit tests.
- **Remote GPU environment (Kaggle / Colab)**: Compute-heavy pipeline execution (corpus cleaning, locked baseline chunking, Gemini-powered gold eval benchmark generation, embedding generation with `bge-base-en-v1.5`, FAISS indexing, and Recall@K retrieval benchmarking).

## Environment Setup

- **Local Python Environment**: Virtual environment located at `rag/` (Python 3.12, managed with `uv`).
- **Dependencies**: Defined in `pyproject.toml`, `requirements.txt`, and `requirements-dev.txt`.
  - Install dependencies: `source rag/bin/activate && uv pip install -r requirements.txt`
  - Install editable package: `uv pip install -e .`
  - Adding local packages: `source rag/bin/activate && uv pip install <package>`
- **Hugging Face Authentication**: Required for accessing the gated dataset (`hf auth login` or `huggingface_hub.login()`).

## Common Commands

### Activate Environment
```bash
source rag/bin/activate
```

### Run Unit Tests
```bash
PYTHONPATH=src python -m unittest discover -s tests
# or with pytest
pytest tests/
```

### Data Collection & Profiling Scripts
```bash
# Collect 5,000 judgment subset (simple stream to data/raw/judgments.parquet)
python scripts/download_subset.py

# Collect 20,000 balanced judgment corpus (with exact-text dedup and 300-1500 per-court bounds)
python scripts/collect_corpus.py

# Profile dataset shape, missing values, duplicates, text length distributions, and year/court counts
python scripts/profile_data.py

# Analyze CNR prefix court distribution
python scripts/analyze_courts.py
```

### Notebook Generation
```bash
# Generates/updates Notebooks/legalrag_refactored.ipynb
python scripts/write_notebook.py
```

## Architecture & Code Structure

```
LegalRAG/
├── pyproject.toml              # Build metadata, packaging & tool configs
├── requirements.txt            # Pinned runtime dependencies
├── requirements-dev.txt        # Development & testing tooling
├── src/
│   └── legalrag/               # Core modular library
│       ├── preprocessing/      # Cleaner & locked LegalChunker
│       ├── retrieval/          # BM25, Dense FAISS, RRF fusion, CrossEncoder reranking
│       ├── generation/         # RAG prompt formatting & greedy generation client
│       └── evaluation/         # Gold evidence matching & failure taxonomy metrics
├── tests/                      # Comprehensive unit test suite
├── scripts/
│   ├── download_subset.py      # First-pass 5k sample collector → data/raw/judgments.parquet
│   ├── collect_corpus.py       # Balanced 20k collector with exact dedup & per-court limits
│   ├── profile_data.py         # Dataset profiling script
│   ├── analyze_courts.py       # Analyzes CNR prefix (court code) distribution
│   └── write_notebook.py       # Programmatic generator for Kaggle/Colab pipeline notebook
├── data/
│   ├── raw/                    # Raw collected Parquet files
│   └── processed/              # Cleaned and chunked corpus artifacts
├── Notebooks/
│   ├── legalrag.ipynb          # Original experimental pipeline notebook
│   ├── legalrag_refactored.ipynb # Idempotent, checkpointed pipeline notebook
│   └── legalrag-100k-final.ipynb # 100k scale experimental notebook
└── docs/                       # Research and technical documentation suite
```

### Pipeline Stages
1. **Streaming Collection**: Uses Hugging Face `load_dataset(..., streaming=True)` with `.shuffle(seed=42, buffer_size=10_000)`. Never materializes the entire raw dataset locally.
2. **Quality Cleaning**: Drops documents where control characters exceed 1% of text length (handles known corrupt/Caesar-shifted entries); strips remaining control characters and normalizes whitespace.
3. **Chunking (Locked Baseline)**: 1,200 character chunk size, 200 character overlap, minimum chunk length 100 characters.
4. **Eval Benchmark Generation**: 100 sampled judgments evaluated with structured question generation (`gemini-3.5-flash-lite`), mapping `supporting_text` to gold chunk IDs.
5. **Retrieval Baselines**:
   - **BM25**: Tokenized corpus indexed via `rank-bm25` ($k_1=1.5, b=0.75$).
   - **Dense**: `BAAI/bge-base-en-v1.5` embeddings (768-dim, normalized) indexed with FAISS (`IndexFlatIP`).
   - **Hybrid RRF**: Reciprocal Rank Fusion ($k=60$) combining BM25 and Dense ranking.
   - **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
   - **Evaluation Metric**: Recall@K (K = 1, 3, 5, 10) against gold chunk IDs.

## Development Conventions

- **Random Seed**: Always use `seed=42` for shuffling, sampling, and splits.
- **File Storage**: Use Apache Parquet (`.parquet`) for tabular data, always saved with `index=False`. Use JSON for eval benchmarks and metrics.
- **Court Identification**: Use `court_code` column first; fall back to the first 4 characters of `cnr` if `court_code` is missing.
- **Remote vs Local Paths**: Paths starting with `/kaggle/working/` or `/content/` are meant for the remote execution environment and do not exist on the local filesystem.
