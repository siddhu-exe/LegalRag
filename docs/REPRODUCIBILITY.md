# Reproducibility Guide: LegalRAG

This guide details the exact environment, hardware requirements, dependencies, and execution workflow to reproduce the 100,000-judgment LegalRAG experiment from scratch or resume from intermediate checkpoints.

---

## 1. Hardware Requirements

### Verified Cloud Environment (Kaggle Pipeline)
- **GPUs**: 2 × NVIDIA Tesla T4 (16 GB VRAM each, total 32 GB VRAM).
- **System Memory**: 31.3 GB Host RAM.
- **Disk Storage**: ~30 GB free space (accommodates 100k Parquet shards, 538k chunks, 108 embedding shards, BM25 index, and FAISS index).
- **OS**: Ubuntu 20.04 LTS / 22.04 LTS (x86_64).

### Minimum Local Development Setup (for Subsets e.g., 5k–20k)
- **CPU**: 8+ cores (Intel Core i7/i9, AMD Ryzen 7/9, or Apple Silicon M-series).
- **RAM**: Minimum 16 GB (32 GB recommended).
- **GPU**: Optional for subsets; recommended NVIDIA GPU with 8+ GB VRAM.
- **Disk**: 10 GB free space.

---

## 2. Environment Setup

### Python Version
- **Python**: 3.10 or 3.11

### Virtual Environment Creation
```bash
# Create virtual environment
python3 -m venv rag

# Activate environment
source rag/bin/activate
```

### Dependency Installation
```bash
pip install --upgrade pip

# Install pinned repository dependencies
pip install -r requirements.txt

# Or install core ML, RAG, and API packages directly:
pip install \
    torch==2.1.2 \
    transformers==4.36.2 \
    sentence-transformers==2.3.1 \
    rank-bm25==0.2.2 \
    datasets==2.16.1 \
    pandas==2.1.4 \
    pyarrow==14.0.2 \
    langchain==0.1.0 \
    groq>=0.9.0 \
    fastapi>=0.109.0 \
    uvicorn[standard]>=0.27.0 \
    pydantic>=2.5.0 \
    pydantic-settings>=2.1.0 \
    huggingface-hub>=0.20.0 \
    tqdm==4.66.1 \
    numpy==1.26.3 \
    scipy==1.11.4

# FAISS installation:
# For GPU systems:
pip install faiss-gpu==1.7.2
# For CPU-only systems:
# pip install faiss-cpu==1.7.2
```

---

## 3. Step-by-Step Pipeline Execution

The complete experimental pipeline is consolidated into the standalone notebook `Notebooks/legalrag-100k-final.ipynb` or modular execution scripts.

```text
Step 1: Ingestion & Deduplication
     │
Step 2: Cleaning & Normalization
     │
Step 3: Recursive Character Chunking
     │
Step 4: Lexical BM25 Indexing
     │
Step 5: Multi-GPU Sharded Dense Embedding & FAISS Indexing
     │
Step 6: Evidence-Grounded Benchmark Generation (497 Gold Questions)
     │
Step 7: Retrieval Evaluation (BM25, Dense, Hybrid RRF, Cross-Encoder)
     │
Step 8: RAG Generation (LLM via OmniRoute @ temp=0)
     │
Step 9: Multi-Criteria Judge Evaluation & Failure Analysis
```

---

### Step 1: Corpus Ingestion
Streams 100,000 High Court decisions from Hugging Face:
```python
from datasets import load_dataset
import pandas as pd

dataset = load_dataset(
    "overthelex/indian-court-decisions",
    "high_courts",
    split="train",
    streaming=True
).shuffle(buffer_size=10000, seed=42)

# Deduplicate by text SHA-256 and collect 100,000 records
```

### Step 2: Corpus Cleaning
Sanitizes corrupted control character sequences:
```python
import re

def clean_text(text: str) -> str:
    # Strip non-printable ASCII control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)
    # Collapse irregular whitespace while maintaining single spaces
    return re.sub(r'\s+', ' ', text).strip()
```
*Output*: `legal_judgments_clean.parquet` (485.4 MB).

### Step 3: Text Chunking
Splits documents using locked hyperparameters:
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
```
*Output*: `legal_chunks.parquet` (538,079 chunks, 260.7 MB).

### Step 4: Lexical Index Construction
Builds the `rank-bm25` index over regex-tokenized chunks:
```python
import re
import pickle
from rank_bm25 import BM25Okapi

tokenized_corpus = [re.findall(r'\w+', text.lower()) for text in chunk_texts]
bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

# The chunk_ids must be stored in the same order as the BM25 corpus rows.
with open("bm25.pkl", "wb") as f:
    pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)
```
*Output*: `bm25.pkl` (578.8 MB). The runtime loader (`BM25Retriever.load`) accepts the
canonical `"bm25"` key, the legacy in-repo `"model"` key, and a bare pickled `BM25Okapi`
object (in which case `chunk_ids` are taken from `legal_chunks.parquet`).

### Step 5: Multi-GPU Dense Embedding & FAISS Indexing
Encodes chunks using `BAAI/bge-base-en-v1.5` in shards of 5,000 chunks:
```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
# Distributed sharding -> emb_0000.npy ... emb_0107.npy

# Build FAISS FlatIP index over normalized embeddings
dimension = 768
index = faiss.IndexFlatIP(dimension)
# Add all normalized shards
faiss.write_index(index, "dense.index")
```
*Output*: 108 `.npy` shards (~1.65 GB total) and `dense.index` (1.65 GB).

### Step 6: Benchmark Construction & Evidence Grounding
Generates 500 candidate questions across 5 question types and filters via `find_gold_chunks`:
```python
def find_gold_chunks(supporting_text, chunk_texts, chunk_ids):
    # Matches supporting passages via exact and sliding-window phrase matching
    # Discards candidates failing evidence verification
    ...
```
*Output*: `gold_eval.json` (497 validated evaluation questions).

### Step 7: Retrieval Benchmarking
Evaluates Judgment Recall@K ($K \in \{1, 3, 5, 10, 25, 50\}$) across BM25, Dense, Hybrid RRF, and Cross-Encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
*Outputs*: `bm25_top50.json`, `dense_top50.json`, `hybrid_top50.json`, `reranked_top50.json`.

### Step 8 & 9: RAG Generation & Multi-Criteria Evaluation
Executes greedy generation (`temperature=0`) over top-5 reranked context blocks via OmniRoute and evaluates results with an independent LLM judge.
*Outputs*: `rag_results.json`, `rag_evaluation.json`, `rag_metrics.json`.

---

## 4. Artifact & Checkpoint Catalog

| Artifact Filename | Size | Type | Description |
| :--- | :---: | :---: | :--- |
| `legal_judgments_clean.parquet` | 485.4 MB | Parquet | 100k sanitized High Court judgments |
| `legal_chunks.parquet` | 260.7 MB | Parquet | 538,079 text chunks with CNR and index mapping |
| `evaluation_documents.parquet` | 2.4 MB | Parquet | 500 sampled evaluation judgments |
| `bm25.pkl` | 578.8 MB | Pickle | `{"bm25": BM25Okapi, "chunk_ids": [...]}` lexical index |
| `dense.index` | 1.65 GB | FAISS | FAISS FlatIP vector index over 538k embeddings |
| `dense_index_metadata.json` | 14.2 MB | JSON | Mapping of vector IDs to chunk IDs |
| `emb_0000.npy`–`emb_0107.npy` | ~1.65 GB | NPY | 108 sharded embedding checkpoint files |
| `eval_candidates.json` | 680 KB | JSON | 500 raw candidate evaluation questions |
| `gold_eval.json` | 674 KB | JSON | 497 validated gold benchmark questions |
| `hybrid_top50.json` | 8.2 MB | JSON | Merged BM25 + Dense RRF candidates |
| `reranked_top50.json` | 8.2 MB | JSON | Cross-encoder reranked candidate pool |
| `rag_results.json` | 1.4 MB | JSON | 497 generated RAG answers with context IDs |
| `rag_evaluation.json` | 1.1 MB | JSON | Multi-criteria judge evaluation scores |
| `rag_metrics.json` | 4 KB | JSON | Aggregated score summary and question-type metrics |

---

## 5. Checkpointing & Resume Strategy

The pipeline is fully idempotent. Each stage verifies whether its target artifact exists on disk before running:

```python
import os

if os.path.exists("dense.index"):
    print("Loading existing FAISS index...")
    index = faiss.read_index("dense.index")
else:
    print("Building FAISS index from embedding shards...")
    # Compile shards into index
```

If a GPU execution session disconnects during dense embedding, the script resumes from the last completed shard (`emb_XXXX.npy`) without re-encoding earlier batches.

---

## 6. Resource Consumption & Runtime Estimates

| Pipeline Stage | Verified Runtime (2 × Tesla T4) | Peak RAM | Peak VRAM |
| :--- | :---: | :---: | :---: |
| **Ingestion (100k docs)** | ~18 minutes | 4.2 GB | — |
| **Cleaning & Normalization** | ~3 minutes | 3.8 GB | — |
| **Chunking (538k chunks)** | ~6 minutes | 5.1 GB | — |
| **BM25 Tokenization & Index** | ~4 minutes | 12.4 GB | — |
| **Dense Embedding (BGE-Base)** | ~45 minutes | 8.6 GB | 2 × 11.2 GB |
| **FAISS Index Compilation** | ~2 minutes | 14.8 GB | — |
| **RRF Fusion (500 queries)** | ~15 seconds | 2.1 GB | — |
| **Cross-Encoder Reranking** | ~6 minutes | 3.4 GB | 4.8 GB |
| **RAG Generation (497 q)** | ~12 minutes | 1.2 GB | — |
| **Judge Evaluation (497 q)** | ~15 minutes | 1.2 GB | — |

---

## 7. Verification & Sanity Checks

To verify pipeline integrity after execution:

```bash
# 1. Verify chunk count
python3 -c "import pandas as pd; df = pd.read_parquet('legal_chunks.parquet'); assert len(df) == 538079, f'Expected 538079, got {len(df)}'; print('Chunks OK: 538,079')"

# 2. Verify benchmark count
python3 -c "import json; data = json.load(open('gold_eval.json')); assert len(data) == 497, f'Expected 497, got {len(data)}'; print('Gold benchmark OK: 497')"

# 3. Verify FAISS index vector count
python3 -c "import faiss; index = faiss.read_index('dense.index'); assert index.ntotal == 538079, f'Expected 538079, got {index.ntotal}'; print('FAISS Index OK: 538,079 vectors')"
```

---

## 8. Running the FastAPI Backend Service

The LegalRAG service supports dual runtime modes for zero-overhead local development vs. full production scale:
1. **`production` (fail-closed default)**: Full retrieval cascade across 538,079 chunks using BM25, FAISS IndexFlatIP, Cross-Encoder, and Groq (`llama-3.3-70b-versatile`). If required artifacts or configuration are unavailable, the service reports NOT READY (HTTP 503) and never serves stub responses.
2. **`local_stub` (explicit opt-in)**: Lightweight deterministic mock retrieval and generation for development and resource-constrained environments (e.g. 6 GB RAM laptop, < 100 MB RAM).

### A. Local Development (`local_stub` mode)
No heavy models, disk artifacts, or GPU required. Stub mode must be selected explicitly:
```bash
# Explicitly select local_stub (production is the fail-closed default)
export ENVIRONMENT=local_stub
export API_PORT=7860

# Start server
uvicorn legalrag.api.main:app --host 0.0.0.0 --port 7860
```
- Interactive Swagger UI: `http://localhost:7860/docs`
- Health check: `http://localhost:7860/health`
- Readiness check: `http://localhost:7860/ready`

### B. Production Serving (`production` mode)
Production containers automatically download any missing artifacts from Hugging Face Hub at
startup using `scripts/download_artifacts.py`; existing artifacts are reused. You can also
provision them ahead of time manually:
```bash
# 1. (Optional) Pre-download artifacts from Hugging Face Hub
python scripts/download_artifacts.py --repo-id <hf-username>/<repo-name> --target-dir artifacts

# 2. Set environment variables (HF_REPO_ID is required for startup provisioning)
export ENVIRONMENT=production
export GROQ_API_KEY="your-groq-api-key"
export HF_REPO_ID="<hf-username>/<repo-name>"
export HF_TOKEN="your-huggingface-token"   # only for private repositories
export API_PORT=7860

# 3. Start production server
uvicorn legalrag.api.main:app --host 0.0.0.0 --port 7860
```

### C. Docker Container (Azure Container Apps / Hugging Face Spaces)
Complies with standard container runtime specifications (runs as non-root user `user` with UID `1000`, exposes port `7860`):
```bash
# Build Docker image
docker build -t legalrag-api .

# Run Docker container
docker run -p 7860:7860 \
    -e ENVIRONMENT=production \
    -e GROQ_API_KEY="your-groq-api-key" \
    -e HF_REPO_ID="<hf-username>/<repo-name>" \
    -e HF_TOKEN="your-huggingface-token" \
    legalrag-api
```

The image defaults to `ENVIRONMENT=production` and downloads missing artifacts at startup.
Until the pipeline is initialized, `GET /ready` returns HTTP 503 and `/query` will not serve
stub responses.

### D. Running Unit & Integration Tests
```bash
# Run all unit tests (retrieval, preprocessing, schemas, config, and API endpoints via local stub)
python -m unittest discover -s tests -v
# Or using pytest
pytest tests/ -v
```
