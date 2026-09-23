# LegalRAG: LLM Context & Project Knowledge Base

> **File Purpose**: This document provides a concise, high-density factual summary of the **LegalRAG** repository, architecture, experimental findings, and data schemas for language models and automated agents.

---

## 1. Project Summary

- **Name**: LegalRAG
- **Core Problem**: Legal judgment retrieval and question answering over Indian High Court decisions. General RAG systems fail on legal texts due to exact statutory citations (e.g., Section 138 of NI Act), legal terminology, multi-page judgments, and severe risks of hallucinated jurisprudence.
- **Corpus**: 100,000 unique Indian High Court judgments across 24 state High Courts.
- **Source**: Hugging Face `overthelex/indian-court-decisions` (config: `high_courts`, split: `train`).
- **Total Chunks**: 538,079 chunks (average 5.38 chunks per document).
- **Evaluation Benchmark**: 497 evidence-grounded questions across 5 question types.
- **Key Insight**: Lexical BM25 retrieval outperforms out-of-the-box dense vector retrieval in top-1 recall (**42.66% vs. 19.72%**). Hybrid Reciprocal Rank Fusion (RRF) expands broad candidate recall to **65.79%** at top-50, and neural reranking lifts top-1 recall to **35.61%**. Over **51.3%** of all RAG generation failures are caused directly by context omission at the retrieval stage.

---

## 2. Ingestion & Preprocessing

- **Streaming & Deduplication**: Streamed with `datasets` using `streaming=True`, `buffer_size=10000`, `seed=42`. Deduplication enforced via in-memory SHA-256 text hashes.
- **Corpus Sanitization**: Profiling identified 762 corrupted documents (0.76%) containing non-printable ASCII control characters. Cleaned via regex `[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]` while preserving paragraphs and legal punctuation.
- **Chunking Parameters**:
  - Splitter: `RecursiveCharacterTextSplitter` (separators: `["\n\n", "\n", " ", ""]`)
  - `CHUNK_SIZE`: 1200 characters (~200–250 tokens)
  - `CHUNK_OVERLAP`: 200 characters (~35–40 tokens)
  - `MIN_CHUNK_LENGTH`: 100 characters
  - Result: 538,079 chunks saved in `legal_chunks.parquet`.

---

## 3. Retrieval Architecture

1. **BM25 Lexical Retriever**:
   - Implementation: `rank-bm25` (`BM25Okapi`, $k_1=1.5, b=0.75$).
   - Tokenization: Lowercased alphanumeric regex `\w+`.
   - Artifact: `bm25.pkl` (578.8 MB).
2. **Dense Vector Retriever**:
   - Embedding Model: `BAAI/bge-base-en-v1.5` (768 dimensions, normalized).
   - Index: `faiss.IndexFlatIP` (Cosine similarity over L2-normalized vectors).
   - Artifacts: 108 embedding shards (`emb_0000.npy`–`emb_0107.npy`) and `dense.index` (1.65 GB).
3. **Hybrid Reciprocal Rank Fusion (RRF)**:
   - Formula: $\text{Score}_{\text{RRF}}(d) = \frac{1}{60 + \text{rank}_{\text{BM25}}(d)} + \frac{1}{60 + \text{rank}_{\text{Dense}}(d)}$
   - Merges top-50 BM25 and top-50 Dense candidates into a combined top-50 pool (`hybrid_top50.json`).
4. **Neural Cross-Encoder Reranker**:
   - Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (max length 512).
   - Rescores top-50 hybrid candidates to output top-5 RAG context (`reranked_top50.json`).

---

## 4. Benchmark Construction & Evidence Grounding

- **Candidate Generation**: 500 candidate questions generated across 500 balanced evaluation documents.
- **Evidence Verification (`find_gold_chunks`)**: Validates that supporting text extracted during question generation is verifiably present in source chunks via exact and sliding-window substring matching (windows: 60, 40, 25, 15, 10 words).
- **Candidate Filtering**: 3 candidates dropped due to unanchored text, yielding **497 validated gold questions** (`gold_eval.json`).
- **Benchmark Breakdown by Type**:
  - `reasoning`: 149 questions (29.98%) — judicial rationales, ratio decidendi
  - `outcome`: 118 questions (23.74%) — final orders, sentences, dispositions
  - `fact`: 100 questions (20.12%) — dates, amounts, allegations
  - `legal_provision`: 80 questions (16.10%) — statutory sections, acts, articles
  - `multi_hop`: 50 questions (10.06%) — synthesis across multiple procedural stages

---

## 5. Frozen 100k Benchmark Results

### Retrieval Performance (Judgment Recall@K, 497 Queries)
| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@25 | Recall@50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25** | **0.4266** | **0.5030** | **0.5332** | **0.5775** | 0.6076 | 0.6398 |
| **Dense (BGE-Base)** | 0.1972 | 0.2535 | 0.2736 | 0.3340 | 0.4125 | 0.4467 |
| **Hybrid-RRF** | 0.3219 | 0.4527 | 0.5111 | 0.5614 | 0.6177 | **0.6579** |
| **Hybrid + Reranker** | 0.3561 | 0.4447 | 0.4869 | 0.5594 | 0.6177 | **0.6579** |

### Downstream RAG Generation Evaluation (497 Answers)
- **Answer Relevance**: **3.0966 / 4** (Score counts 1/2/3/4: 88 / 69 / 47 / 293)
- **Faithfulness**: **3.4004 / 4** (Score counts 1/2/3/4: 53 / 53 / 33 / 358)
- **Citation Correctness**: **3.2515 / 4** (Score counts 1/2/3/4: 46 / 100 / 34 / 317)
- **Overall Score**: **3.0926 / 4** (Score counts 1/2/3/4: 67 / 90 / 70 / 270)
- **Unsupported Claim Rate**: **0.1489** (14.89%)
- **Hallucination Rate**: **0.2274** (22.74% — 113 True / 384 False)

### Scores by Question Type
| Type | Count | Relevance | Faithfulness | Citation | Unsupported Rate | Hallucination Rate | Overall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Legal Provision** | 80 | **3.5000** | **3.6625** | **3.6375** | **0.0906** | **0.1500** | **3.4375** |
| **Fact** | 100 | 3.3300 | 3.4400 | 3.2700 | 0.1388 | 0.2300 | 3.1400 |
| **Reasoning** | 149 | 3.0604 | 3.5235 | 3.4027 | 0.1250 | 0.1745 | 3.1007 |
| **Multi-Hop** | 50 | 2.7600 | 3.4000 | 3.2600 | 0.1306 | 0.2400 | 3.0200 |
| **Outcome** | 118 | 2.8136 | 3.0339 | 2.7797 | 0.2347 | 0.3390 | 2.8390 |

---

## 6. Failure Mode Analysis

| Failure Category | Count | % | Mechanism |
| :--- | :---: | :---: | :--- |
| **Retrieval Failure** | **255** | **51.31%** | Target judgment missing from retrieved top-5 context |
| **No Major Failure (Success)** | **194** | **39.03%** | Target judgment present, answer accurate, faithful, properly cited |
| **Low Answer Relevance** | **21** | **4.23%** | Context partially relevant, but model failed to answer directly |
| **Context Selection / Citation** | **19** | **3.82%** | Model cited irrelevant distractor chunks or improper formatting |
| **Generation Grounding Failure** | **8** | **1.61%** | Target judgment present, but model hallucinated extraneous claims |

---

## 7. Production FastAPI Backend & Deployment

- **Application Factory**: `create_app()` in `src/legalrag/api/main.py`.
- **Runtime Dual-Mode**:
  - `production` (fail-closed default): Full pipeline loading `bm25.pkl`, `dense.index`, `legal_chunks.parquet`, BGE embedding model, Cross-Encoder reranker, and connecting to Groq (llama-3.3-70b-versatile). Missing artifacts are provisioned from Hugging Face Hub at startup; if unavailable the service stays NOT READY and never serves stubs.
  - `local_stub` (explicit opt-in): Instant local development on standard laptops (< 100 MB RAM) using deterministic in-memory stubs without loading multi-gigabyte models or FAISS index files.
- **REST Endpoints**:
  - `GET /health` -> lightweight liveness: `HealthResponse(status="ok", environment=...)` (never loads models).
  - `GET /ready` -> readiness: `ReadinessResponse(status="ready")` or HTTP 503 when the pipeline/artifacts are unavailable.
  - `GET /` -> Service root with API metadata, environment, and `/docs` documentation link.
  - `POST /query` -> Executes hybrid retrieval, RRF fusion, cross-encoder reranking, prompt formatting, Groq generation, citation verification, and latency breakdown.
- **Data Transfer Schemas** (`src/legalrag/api/schemas.py`):
  - `QueryRequest`: accepts ONLY `question: str` (min length 3, max length 4000, whitespace-stripped). Extra/server-config fields are rejected with HTTP 422.
  - `QueryResponse`: `answer`, `citations: List[Citation]`, `retrieved_chunk_ids: List[str]`, `retrieve_ms: float`, `rerank_ms: float`, `generate_ms: float`, `total_ms: float`, `status: Literal["ok", "generation_error", "retrieval_error"]`.
  - `Citation`: `chunk_id`, `cnr`, `court_code`, `decision_date`, `title`.
- **HTTP Error Semantics**: 422 invalid request, 500 retrieval/reranking failure, 502 LLM/generation provider failure, 503 pipeline/readiness failure. Error bodies are sanitized (no keys, tokens, paths, or stack traces).
- **Generation & LLMOps**:
  - Model: Groq (`llama-3.3-70b-versatile`) via official `groq` SDK (`LegalGenerationClient`).
  - Strict Exception Shielding: Traps all `GroqAuthenticationError`, `GroqRateLimitError`, `GroqBadRequestError`, `GroqInternalServerError`, `GroqAPIConnectionError`, `GroqAPIError`. Returns sanitized user-facing responses with `status="generation_error"` or `"retrieval_error"` while logging full traces server-side.
  - Citation Grounding Defense: Regex-extracted `[Chunk ID: ...]` citations are cross-validated against retrieved `top_chunk_ids`; hallucinated chunk IDs are filtered out before response serialization.
- **Containerization**:
  - Multi-stage Dockerfile built on `python:3.10-slim`.
  - Non-root user `user` with UID `1000` for Hugging Face Spaces compliance.
  - Default port `7860` with environment aliases (`PORT`, `API_PORT`).

---

## 8. Subsystem Status & Roadmap

- **Completed Core Capabilities**:
  - 100k corpus collection, cleaning, and Parquet serialization.
  - 538k chunking and indexing (BM25 + FAISS Dense).
  - 4-tier retrieval benchmarking and 497-question judge evaluation.
  - Checkpointed execution in `Notebooks/legalrag-100k-final.ipynb`.
  - Production FastAPI backend (`src/legalrag/api/`) with dual runtime modes, input validation, and high-resolution latency tracking.
  - Groq (llama-3.3-70b-versatile) generation client with strict error shielding.
  - Multi-stage Docker containerization and Hub artifact download scripts for Azure Container Apps / Hugging Face Spaces.
- **Planned Application Layer (Phase 2)**:
  - **Streamlit / Web UI**: Interactive legal query interface with citation graphs and court jurisdiction filters.
  - **Quantized Vector Index**: FAISS IVFPQ / HNSW serving for sub-50ms query latency.

---

## 9. Artifact Registry

| Filename | Purpose | Schema / Content |
| :--- | :--- | :--- |
| `legal_judgments_clean.parquet` | Cleaned 100k judgments | `cnr, court_code, court_name, decision_date, year, full_text, ...` |
| `legal_chunks.parquet` | 538,079 text chunks | `chunk_id, cnr, chunk_index, text` |
| `evaluation_documents.parquet` | 500 sampled evaluation docs | Cleaned judgment records sampled round-robin across courts |
| `bm25.pkl` | BM25Okapi lexical index | Pickled `dict` with `bm25` (BM25Okapi) and `chunk_ids`; legacy `model` key is also accepted |
| `dense.index` | FAISS vector index | `faiss.IndexFlatIP` (538,079 x 768-dim normalized vectors) |
| `gold_eval.json` | 497 validated questions | `cnr, question, reference_answer, question_type, supporting_text, gold_chunk_ids` |
| `rag_results.json` | 497 generated answers | `cnr, question, reference_answer, generated_answer, question_type, retrieved_chunk_ids` |
| `rag_evaluation.json` | 497 judge evaluations | `cnr, question_type, answer_relevance, faithfulness, citation_correctness, unsupported_claim_rate, hallucination, overall_score, judge_feedback` |
