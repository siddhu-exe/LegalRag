# LegalRAG: Hybrid Retrieval & Evidence-Grounded Question Answering over 100,000 Indian High Court Judgments

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Corpus Size](https://img.shields.io/badge/corpus-100k%20judgments-darkgreen.svg)](https://huggingface.co/datasets/overthelex/indian-court-decisions)
[![Chunks](https://img.shields.io/badge/chunks-538%2C079-green.svg)](#chunking)
[![Benchmark](https://img.shields.io/badge/gold%20benchmark-497%20questions-orange.svg)](#benchmark-construction)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**LegalRAG** is an end-to-end legal information retrieval and evidence-grounded question-answering system evaluated over **100,000 Indian High Court judgments** comprising **538,079 chunks**. 

Standard general-purpose RAG pipelines struggle in legal domains due to exact statutory citations (e.g., *Section 138 NI Act*, *Section 482 CrPC*), domain-specific jargon, lengthy narrative structures, and the severe risks of ungrounded hallucinations. LegalRAG systematically addresses these challenges by combining lexical precision, dense semantic representations, rank fusion, and neural reranking—evaluated against an automated, evidence-verified benchmark.

---

## Live Deployment

| Component | URL / Host | Notes |
| :--- | :--- | :--- |
| **Web Interface** | [legalrag-six.vercel.app](https://legalrag-six.vercel.app) | React 19 + Vite SPA hosted on Vercel |
| **Backend API** | **AWS EC2 `t4.xlarge`** | FastAPI container, `ENVIRONMENT=production`, port `7860` |

The EC2 instance is stopped when idle to avoid continuous hosting costs for the 538k-chunk index.
While it is down, the frontend's readiness gatekeeper (`GET /ready`) shows a standby screen;
start the instance to restore full query service.

---

## Key Highlights

- **Scale**: 100,000 High Court judgments across 24 Indian jurisdictions ingested, sanitized, and partitioned into **538,079 text chunks**.
- **Lexical vs. Dense Discrepancy**: BM25 strongly outperforms out-of-the-box dense retrieval at top-1 judgment recall (**42.66% vs. 19.72%**), proving that statutory citation matching requires exact keyword alignment.
- **Hybrid Fusion Expansion**: Reciprocal Rank Fusion (RRF) achieves **65.79%** candidate recall at top-50, broadening the candidate pool beyond single-retriever limits.
- **Neural Reranking Precision**: Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) boosts top-1 recall by **+3.42 percentage points** to **35.61%**.
- **Evidence-Grounded Benchmark**: 497 validated gold evaluation questions spanning 5 legal inquiry types with substring-verified chunk attribution.
- **Root-Cause Failure Analysis**: Quantifies that **51.3%** of all downstream generation errors stem directly from retrieval omission in top-5 context, pinpointing retrieval recall as the primary bottleneck for legal RAG.

---

## System Architecture

```text
                                User HTTP Request
                          POST /query {"question": "..."}
                                       │
                                       ▼
                         FastAPI Application Factory
                      (Pydantic v2 Request Validation)
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
          BM25 Lexical Retrieval               Dense Vector Retrieval
         (BM25Okapi Index, top-50)           (BAAI/bge-base-en-v1.5, top-50)
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       ▼
                             Reciprocal Rank Fusion
                       RRF Score = ∑ 1 / (60 + Rank_i)
                                  (top-50)
                                       │
                                       ▼
                            Cross-Encoder Reranker
                         (ms-marco-MiniLM-L-6-v2)
                                  (top-5)
                                       │
                                       ▼
                        Strict RAG Prompt Builder &
                        LLM Generation Client Layer
                       (Groq GROQ_MODEL_NAME @ T=0)
                       [Shielded Exception Trapping]
                                       │
                                       ▼
                       Citation Extractor & Grounding Filter
                       (Verifies citations against context)
                                       │
                                       ▼
                                JSON Response
                       - Grounded Answer & Verifiable Citations
                       - Retrieved Top Chunk IDs
                       - Subsystem Latencies (retrieve/rerank/gen/total ms)
                       - Status ('ok' / 'generation_error' / 'retrieval_error')
```

---

## Empirical Benchmark Results

### 1. Retrieval Scoreboard (497 Gold Questions)

| Retriever | Judgment Recall@1 | Judgment Recall@5 | Judgment Recall@10 | Judgment Recall@50 |
| :--- | :---: | :---: | :---: | :---: |
| **BM25 (Lexical)** | **0.4266** (42.66%) | **0.5332** (53.32%) | **0.5775** (57.75%) | 0.6398 (63.98%) |
| **Dense (`bge-base-en-v1.5`)** | 0.1972 (19.72%) | 0.2736 (27.36%) | 0.3340 (33.40%) | 0.4467 (44.67%) |
| **Hybrid-RRF** | 0.3219 (32.19%) | 0.5111 (51.11%) | 0.5614 (56.14%) | **0.6579** (65.79%) |
| **Hybrid-RRF + Cross-Encoder** | 0.3561 (35.61%) | 0.4869 (48.69%) | 0.5594 (55.94%) | **0.6579** (65.79%) |

#### Granular Retrieval Cutoff Comparison
| Cutoff $K$ | BM25 | Dense (BGE) | Hybrid-RRF | Reranked |
| :---: | :---: | :---: | :---: | :---: |
| **@1** | 0.4266 | 0.1972 | 0.3219 | 0.3561 |
| **@3** | 0.5030 | 0.2535 | 0.4527 | 0.4447 |
| **@5** | 0.5332 | 0.2736 | 0.5111 | 0.4869 |
| **@10** | 0.5775 | 0.3340 | 0.5614 | 0.5594 |
| **@25** | 0.6076 | 0.4125 | 0.6177 | 0.6177 |
| **@50** | 0.6398 | 0.4467 | **0.6579** | **0.6579** |

> **Critical Retrieval Engineering Finding (Reranker Domain-Adaptation Trade-off)**:
> Cross-encoder reranking achieves the highest top-1 precision among hybrid methods (+3.42pp over Hybrid-RRF, 0.3219 → 0.3561), placing relevant precedent in the immediate primary attention window. However, it exhibits a slight recall decrease at Recall@5 (0.5111 → 0.4869) and Recall@10 (0.5614 → 0.5594) compared to Hybrid-RRF alone. The root cause is domain divergence: `ms-marco-MiniLM-L-6-v2` is pre-trained on generic MS-MARCO search queries rather than Indian statutory and judicial phrasing. In a production legal pipeline, top-5 context captures sufficient grounding while retaining high top-1 relevance.

---

### 2. Downstream RAG Generation Quality

Evaluated via an automated multi-criteria LLM judge over all 497 validated questions:

| Metric | Score / Value | Score Distribution (1 / 2 / 3 / 4) | Description |
| :--- | :---: | :---: | :--- |
| **Answer Relevance** | **3.0966 / 4** | 88 / 69 / 47 / 293 | Directness and completeness addressing the query |
| **Faithfulness** | **3.4004 / 4** | 53 / 53 / 33 / 358 | Absence of ungrounded factual claims |
| **Citation Correctness** | **3.2515 / 4** | 46 / 100 / 34 / 317 | Syntactic and factual accuracy of chunk citations |
| **Overall Score** | **3.0926 / 4** | 67 / 90 / 70 / 270 | Composite quality synthesis |
| **Unsupported Claim Rate** | **0.1489** (14.89%) | — | Ratio of ungrounded statements to total statements |
| **Hallucination Rate** | **0.2274** (22.74%) | False: 384, True: 113 | Proportion of responses containing false assertions |

---

### 3. Performance by Legal Question Type

| Question Type | Count | Answer Relevance | Faithfulness | Citation Correctness | Unsupported Claim Rate | Hallucination Rate | Overall Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Legal Provision** | 80 | **3.5000** | **3.6625** | **3.6375** | **0.0906** | **0.1500** | **3.4375** |
| **Fact** | 100 | 3.3300 | 3.4400 | 3.2700 | 0.1388 | 0.2300 | 3.1400 |
| **Reasoning** | 149 | 3.0604 | 3.5235 | 3.4027 | 0.1250 | 0.1745 | 3.1007 |
| **Multi-Hop** | 50 | 2.7600 | 3.4000 | 3.2600 | 0.1306 | 0.2400 | 3.0200 |
| **Outcome** | 118 | 2.8136 | 3.0339 | 2.7797 | 0.2347 | 0.3390 | 2.8390 |

---

### 4. Failure Mode Taxonomy

```text
┌─────────────────────────────────────────────────────────────┐
│ Failure Category                      Count      Proportion │
├─────────────────────────────────────────────────────────────┤
│ 1. Retrieval Failure                  255          51.31%   │
│ 2. No Major Failure (Success)         194          39.03%   │
│ 3. Low Answer Relevance                21           4.23%   │
│ 4. Context Selection / Citation        19           3.82%   │
│ 5. Generation Grounding Failure         8           1.61%   │
└─────────────────────────────────────────────────────────────┘
```

- **51.3%** of all generation failures occur because the gold judgment was completely omitted from top-5 context.
- When the correct judgment is present in top-5 context, the model achieves a clean **80.2% success rate** (194/242) with negligible hallucination.

---

## Repository Structure

```text
LegalRAG/
├── README.md                           # Project presentation and benchmark summary
├── LLMS.md                             # Token-dense full-context summary for AI models
├── CLAUDE.md                           # Developer instructions and coding standards
├── QWEN.md                             # Architecture and system documentation
├── pyproject.toml                      # Standard Python packaging and tool configuration
├── requirements.txt                    # Pinned runtime dependencies
├── requirements-dev.txt                # Development & test tooling
├── Dockerfile                          # Production multi-stage Docker container (UID 1000; AWS EC2 & HF Spaces)
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

---

## Quickstart & Serving

### 1. Environment Setup
```bash
git clone https://github.com/Siddharth23052005/LegalRAG.git
cd LegalRAG

python3 -m venv rag
source rag/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Running the FastAPI Backend

LegalRAG features dual runtime modes for zero-overhead local development vs. full production scale:

#### Mode A: Local Development (`local_stub` mode)
Runs instantly on standard laptops (< 100 MB RAM) using deterministic in-memory stubs without loading multi-gigabyte models or FAISS indices:
```bash
# Stub mode must be selected explicitly; production is the fail-closed default.
export ENVIRONMENT=local_stub
export API_PORT=7860

# Start Uvicorn ASGI server
uvicorn legalrag.api.main:app --host 0.0.0.0 --port 7860
```
- Interactive Swagger UI: `http://localhost:7860/docs`
- Health Probe: `http://localhost:7860/health`
- Readiness Probe: `http://localhost:7860/ready`

#### Mode B: Production Serving (`production` mode)
Executes the full 538k-chunk retrieval cascade with Groq generation (`GROQ_MODEL_NAME`).
Production containers download missing artifacts (`bm25.pkl`, `dense.index`,
`legal_chunks.parquet`) from Hugging Face Hub automatically at startup and never fall back
to stub responses:
```bash
# 1. (Optional) Pre-download frozen runtime artifacts from Hugging Face Hub
python scripts/download_artifacts.py --repo-id siddhu23/LegalRag_Dataset --target-dir artifacts

# 2. Configure production environment (HF_REPO_ID required for startup provisioning)
export ENVIRONMENT=production
export GROQ_API_KEY="your-groq-api-key"
export GROQ_MODEL_NAME="qwen/qwen3.8-27b"  # must be a model ID currently served by Groq
export HF_REPO_ID="siddhu23/LegalRag_Dataset"
# Optional: set MODEL_WARMUP_ENABLED=false to skip loading embedding/reranker weights at
# startup (default true, which makes /ready reflect true model usability).
export HF_TOKEN="your-huggingface-token"   # only for private repositories
export API_PORT=7860

# 3. Start production server
uvicorn legalrag.api.main:app --host 0.0.0.0 --port 7860
```

#### Mode C: Docker Container (AWS EC2 / Azure Container Apps / Hugging Face Spaces)
Complies with standard container runtime specification (non-root UID 1000, exposed port 7860):
```bash
docker build -t legalrag-api .
docker run -p 7860:7860 \
    -e ENVIRONMENT=production \
    -e GROQ_API_KEY="your-key" \
    -e GROQ_MODEL_NAME="qwen/qwen3.8-27b" \
    -e HF_REPO_ID="siddhu23/LegalRag_Dataset" \
    legalrag-api
```

The live production backend runs this container on an **AWS EC2 `t4.xlarge`** instance
(see [Live Deployment](#live-deployment)).

### 3. Running Tests
```bash
# Run all 133 unit and integration tests (local stub pipeline only — no models, no network)
pytest -q
# Deterministic across hash seeds; the suite is also runnable per-file in isolation
PYTHONHASHSEED=42 pytest -q
```

### 4. Inspecting Data & Checkpoints
```bash
# Profile existing corpus Parquet files
python scripts/profile_data.py

# Analyze jurisdiction representation across courts
python scripts/analyze_courts.py
```

### 5. Running the Pipeline
Open `Notebooks/legalrag-100k-final.ipynb` in a Jupyter / Kaggle environment (with 2 × Tesla T4 GPUs) to execute the end-to-end ingestion, indexing, retrieval benchmarking, and judge evaluation. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for step-by-step guidance.

### 6. Production Artifact Verification & Measured Latency
The three production artifacts are large (≈2.5 GB combined) and are **not** committed or baked
into the image (see `.gitignore` / `.dockerignore`); they are provisioned from the Hugging Face
repository `siddhu23/LegalRag_Dataset` at startup, or mounted/copied into `artifacts/` manually.
Validate a local copy before deploying:
```bash
# BM25: loads through the production loader (accepts the canonical {"bm25": ..., "chunk_ids": [...]} dict)
PYTHONPATH=src python -c "from legalrag.retrieval.bm25 import BM25Retriever; r = BM25Retriever.load('artifacts/bm25.pkl'); r.validate(); print(type(r.model).__name__, len(r.chunk_ids))"

# Dense: FAISS vector count and dimension must match the 768-dim BGE model
PYTHONPATH=src python -c "import faiss; i = faiss.read_index('artifacts/dense.index'); print(i.ntotal, i.d)"
```

Measured locally against the **real** 538,079-chunk artifacts (CPU-only, single process,
mean over 6 realistic legal queries) during pre-deployment validation:

| Stage | Mean latency |
| :--- | ---: |
| BM25 top-50 | 2,624 ms |
| Dense (BGE-base + FAISS) top-50 | 139 ms |
| RRF (k=60) | 0.1 ms |
| Cross-Encoder top-5 | 1,496 ms |
| Retrieval + rerank total | 4,258 ms |
| Cold pipeline init (artifacts + model warmup) | ≈29.3 s |
| Peak resident memory | ≈10.5 GB |

Generation latency (Groq) is added on top of the retrieval total and is reported per request as
`generate_ms`. The original multi-GPU benchmark timings are in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

> **Troubleshooting — `POST /query` returns 502:** a sanitized 502 always means the generation
> provider call failed. A 404 from Groq (`model_not_found`) means `GROQ_MODEL_NAME` is not a
> model Groq currently serves (for example the decommissioned `llama-3.3-70b-versatile`).
> Check the live list with
> `curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"` and set
> a valid `GROQ_MODEL_NAME`. Retrieval failures surface as 500 and unready service as 503.

---

## Detailed Documentation Suite

For comprehensive deep dives into each subsystem, refer to the documentation suite:
- **[Project Journey](docs/PROJECT_JOURNEY.md)**: Chronological narrative from initial 5k/20k/80k iterations to the final 100k pipeline.
- **[Experiment Report](docs/EXPERIMENT_REPORT.md)**: Formal research paper style report with statistical tables and methodology.
- **[System Architecture](docs/ARCHITECTURE.md)**: Deep dive into chunking, retrieval cascade, prompt templates, and data schemas.
- **[Reproducibility Guide](docs/REPRODUCIBILITY.md)**: Step-by-step instructions, checkpoint catalog, and runtime benchmarks.
- **[LLM Ingestion Brief (LLMS.md)](LLMS.md)**: Optimized concise project digest for language model ingestion.

---

## Roadmap

- [x] **100k High Court Corpus Ingestion & Sanitization** (538k chunks across 24 High Courts)
- [x] **538k Recursive Chunk Indexing** (BM25Okapi + FAISS IndexFlatIP dense vector index)
- [x] **4-Tier Retrieval Benchmarking** (BM25, BGE, Hybrid RRF, Cross-Encoder on 497 gold questions)
- [x] **497-Question Evidence-Grounded Benchmark & Multi-Criteria LLM Judge**
- [x] **FastAPI Backend Service (`src/legalrag/api/`)**: Dual-mode (`local_stub` / `production`), singleton DI, input validation, and latency breakdown
- [x] **Production Exception Shielding & LLMOps**: Sanitized error states, rate limit handling, and hallucinated citation filtering
- [x] **Groq Generation Integration** (official `groq` SDK; model selected by `GROQ_MODEL_NAME`, default `qwen/qwen3.8-27b`)
- [x] **Docker Packaging for Hugging Face Spaces** (Port 7860, UID 1000 non-root user)
- [x] **Live Deployment**: React/Vite interface on Vercel ([legalrag-six.vercel.app](https://legalrag-six.vercel.app)) backed by the FastAPI service on AWS EC2 `t4.xlarge`
- [ ] **Streamlit / Web UI**: Query interface with interactive citation verification, court filtering, and chunk highlight graphs
- [ ] **Vector Quantization (IVFPQ / HNSW)**: Sub-50ms vector search for scale beyond 1M judgments

---

## Citation & Acknowledgments

- **Source Corpus**: [Hugging Face `overthelex/indian-court-decisions`](https://huggingface.co/datasets/overthelex/indian-court-decisions)
- **Embedding Model**: [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5)
- **Reranker Model**: [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2)
