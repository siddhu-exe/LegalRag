# LegalRAG

Production-hardened, hybrid-retrieval RAG (Retrieval-Augmented Generation) system over 100,000 Indian High Court judgment texts with evidence-grounded evaluation and a FastAPI backend service.

## Project Overview

- **Source data**: Hugging Face dataset `overthelex/indian-court-decisions`, config `high_courts`, split `train`, consumed in **streaming mode** (seed 42, shuffle buffer 50,000).
- **Scale**: 100,000 High Court decisions across 24 jurisdictions, cleaned and partitioned into **538,079 chunks**.
- **Multi-Stage Retrieval Cascade**:
  1. **BM25 Lexical Retrieval**: `BM25Okapi` ($k_1=1.5, b=0.75$) with regex tokenization (top-50 pool).
  2. **Dense Vector Retrieval**: `BAAI/bge-base-en-v1.5` (768-dim, normalized) + `faiss.IndexFlatIP` (top-50 pool). Queries are prefixed with the BGE instruction `"Represent this sentence for searching relevant passages: "` (recorded in `dense_index_metadata.json`); passages are not prefixed.
  3. **Reciprocal Rank Fusion (RRF)**: $k=60$, combining BM25 and Dense into top-50 candidate pool.
  4. **Neural Cross-Encoder Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` selecting top-5 RAG context blocks.
- **Generation & LLMOps**:
  - Groq via official `groq` SDK at `temperature=0.0`; model ID from `GROQ_MODEL_NAME`
    (default `qwen/qwen3.8-27b`, must be a model currently served by Groq).
  - Strict exception shielding preventing raw API/rate-limit errors from leaking into answer bodies.
  - Regex citation validation (`[Chunk ID: ...]`) cross-referenced strictly against retrieved context chunks.
- **FastAPI Backend (`src/legalrag/api/`)**:
  - Dual runtime environments: `production` (fail-closed default, full index loading) vs. explicit `local_stub` (< 100 MB RAM for development).
  - High-resolution latency tracking (`retrieve_ms`, `rerank_ms`, `generate_ms`, `total_ms`).
  - Containerized with Docker for Hugging Face Spaces (UID 1000, port 7860).
- **Evaluation Benchmark**: 497 substring-verified gold questions evaluated with multi-criteria LLM judge.

## Directory Structure

```text
LegalRAG/
├── README.md                           # Project presentation and benchmark summary
├── LLMS.md                             # Token-dense full-context summary for AI models
├── CLAUDE.md                           # Developer instructions and coding standards
├── QWEN.md                             # Architecture and system documentation
├── pyproject.toml                      # Standard Python packaging and tool configuration
├── requirements.txt                    # Pinned runtime dependencies
├── requirements-dev.txt                # Development & test tooling
├── Dockerfile                          # Production multi-stage Docker container (HF Spaces UID 1000)
├── .dockerignore                       # Container build exclusion rules
├── .gitignore                          # Data, virtualenv, and checkpoint exclusion rules
├── src/                                # Core modular Python package
│   └── legalrag/
│       ├── api/                        # FastAPI service, configuration, schemas, DI
│       │   ├── config.py               # Pydantic v2 Settings with secret masking & port fallback
│       │   ├── dependencies.py         # Singleton DI provider (local_stub vs production)
│       │   ├── main.py                 # Application factory & metadata routes
│       │   ├── provisioning.py         # Startup Hugging Face artifact provisioning
│       │   ├── routes.py               # /health, /ready, /query: error shielding & latency breakdown
│       │   └── schemas.py              # Pydantic validation request/response schemas
│       ├── preprocessing/              # Text cleaner & locked LegalChunker
│       ├── retrieval/                  # BM25, Dense FAISS, RRF fusion, CrossEncoder reranking
│       ├── generation/                 # RAG prompt templates & Groq generation client
│       └── evaluation/                 # Gold evidence matching & failure taxonomy metrics
├── tests/                              # 133 pytest tests (stubs/mocks only, no real models)
│   ├── test_api.py                     # FastAPI endpoint, validation, and error shielding tests
│   ├── test_api_endpoints.py           # /health, /ready, /query status codes & failure modes
│   ├── test_artifact_contracts.py      # BM25/dense artifact serialization contract regressions
│   ├── test_retrieval.py               # BM25, dense, and RRF fusion tests
│   ├── test_retrieval_pipeline_integration.py  # BM25->Dense->RRF->CrossEncoder (stub) cascade
│   ├── test_cleaner.py / test_chunker.py       # Preprocessing tests
│   ├── test_prompts.py / test_generation_client.py  # Prompts, citations, Groq client
│   ├── test_evaluation.py              # Evidence matching and metrics tests
│   └── test_download_artifacts.py      # Hugging Face artifact provisioning tests
├── scripts/                            # Standalone collection, profiling, and helper scripts
│   ├── download_artifacts.py           # HF Hub artifact downloader for production deployment
│   ├── download_subset.py              # First-pass 5k sample collector
│   ├── collect_corpus.py               # Balanced 20k collector with exact dedup
│   ├── profile_data.py                 # Data profiling and length distribution utility
│   ├── analyze_courts.py               # Court jurisdiction balance analyzer
│   └── write_notebook.py               # Notebook generator and sync script
├── docs/                               # Research and technical documentation suite
│   ├── PROJECT_JOURNEY.md              # Chronological 15-section engineering narrative
│   ├── EXPERIMENT_REPORT.md            # Research-grade 100k experiment report
│   ├── ARCHITECTURE.md                 # System architecture and subsystem specifications
│   └── REPRODUCIBILITY.md              # Hardware, environment, and checkpoint guide
└── Notebooks/
    ├── legalrag-100k-final.ipynb       # 117-cell standalone Kaggle execution notebook
    ├── legalrag_refactored.ipynb       # Idempotent, checkpointed pipeline notebook
    └── legalrag.ipynb                  # Original experimental pipeline notebook
```

## Environment & Execution

### Dual Runtime Modes
1. **`local_stub` (explicit opt-in, dev/test only)**:
   - In-memory mock retrievers and generators (< 100 MB RAM).
   - Zero model downloads, zero GPU requirements.
   - Run locally:
     ```bash
     export ENVIRONMENT=local_stub
     export API_PORT=7860
     uvicorn legalrag.api.main:app --host 0.0.0.0 --port 7860
     ```
2. **`production`**:
   - Full 538k-chunk pipeline with `bm25.pkl`, `dense.index`, `legal_chunks.parquet`, and Groq LLM.
   - Missing artifacts are downloaded from Hugging Face Hub (`siddhu23/LegalRag_Dataset`) at startup; the service reports NOT READY (HTTP 503) if provisioning or initialization fails. Every retrieval component is structurally validated and warmed up before `/ready` returns 200.
   - Run on Azure Container Apps, Hugging Face Spaces, or GPU cloud instance:
     ```bash
     python scripts/download_artifacts.py --repo-id siddhu23/LegalRag_Dataset --target-dir artifacts
     export ENVIRONMENT=production
     export GROQ_API_KEY="your-groq-api-key"
     export GROQ_MODEL_NAME="qwen/qwen3.8-27b"   # must currently be served by Groq
     export HF_REPO_ID="siddhu23/LegalRag_Dataset"
     export API_PORT=7860
     uvicorn legalrag.api.main:app --host 0.0.0.0 --port 7860
     ```

## Key Facts & Frozen 100k Results

- **Scale**: 100,000 High Court decisions -> 538,079 chunks (1200 char chunk size, 200 overlap, min 100).
- **Retrieval Recall (497 Gold Questions)**:
  - BM25 Recall@1: **42.66%** | Recall@50: **63.98%**
  - Dense (BGE-Base) Recall@1: **19.72%** | Recall@50: **44.67%**
  - Hybrid-RRF Recall@1: **32.19%** | Recall@50: **65.79%**
  - Cross-Encoder Reranked Recall@1: **35.61%** | Recall@50: **65.79%**
- **Generation Quality**:
  - Faithfulness: **3.40 / 4**
  - Citation Correctness: **3.25 / 4**
  - Answer Relevance: **3.10 / 4**
  - Unsupported Claim Rate: **14.89%**
  - Retrieval Omission Failure: **51.31%** (primary bottleneck)

## Testing

```bash
# Run all 133 unit & integration tests (local_stub mode; no models, no network)
pytest -q
```

## Production validation (real 538,079-chunk artifacts)

Verified locally against the real `bm25.pkl`, `dense.index`, and `legal_chunks.parquet`
(CPU-only, single process): `BM25Retriever.load()` returns a real `BM25Okapi`;
`check_readiness()` -> `(True, "ready")`; `POST /query` completes real BM25 top-50 +
dense top-50 -> RRF k=60 -> CrossEncoder top-5 retrieval. Mean per-stage latency:
BM25 ≈2,624 ms, dense ≈139 ms, RRF ≈0.1 ms, CrossEncoder ≈1,496 ms (retrieval+rerank
≈4,258 ms); cold pipeline init ≈29.3 s; peak RSS ≈10.5 GB. Generation is not exercised
without a Groq key.
