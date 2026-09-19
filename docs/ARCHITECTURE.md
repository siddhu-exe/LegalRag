# System Architecture: LegalRAG

## Overview

LegalRAG is a specialized legal information retrieval and question-answering architecture designed for complex, unstructured Indian High Court judgments. The system addresses the core failure modes of standard RAG pipelines—statutory citation mismatch, dense embedding semantic smoothing, context fragmentation, and ungrounded generation—via a hybrid multi-stage retrieval pipeline paired with automated evidence grounding.

---

## High-Level Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                          1. INGESTION & DATA CORPUS                         │
│                                                                             │
│  Hugging Face Dataset ────────► Deterministic Stream ──────► Corpus Cleaner  │
│  (overthelex/high_courts)       (Shuffle buffer 10k, seed 42) (762 sanitized)│
│                                                                      │      │
│                                                          100,000 Judgments  │
│                                                     (legal_judgments_clean) │
└────────────────────────────────────────┬────────────────────────────────────┘
                                         │
┌────────────────────────────────────────▼────────────────────────────────────┐
│                       2. CHUNKING & INDEXING ENGINE                         │
│                                                                             │
│  RecursiveCharacterTextSplitter (1200 char chunk, 200 overlap, min 100 char)│
│  ────────► 538,079 Chunks (legal_chunks.parquet)                            │
│                  │                                       │                  │
│                  ▼                                       ▼                  │
│       BM25 Okapi Index                       BAAI/bge-base-en-v1.5          │
│       (Regex Tokenization)                   Multi-GPU Sharded Embeddings   │
│       (bm25.pkl - 578.8 MB)                  (108 shards -> dense.index)    │
└──────────────────┬───────────────────────────────────────┬──────────────────┘
                   │                                       │
┌──────────────────▼───────────────────────────────────────▼──────────────────┐
│                      3. MULTI-STAGE RETRIEVAL PIPELINE                      │
│                                                                             │
│                   Query: "What was the sentence for IPC 302?"               │
│                                      │                                      │
│                  ┌───────────────────┴───────────────────┐                  │
│                  ▼                                       ▼                  │
│           BM25 Top-50                             Dense Top-50              │
│                  │                                       │                  │
│                  └───────────────────┬───────────────────┘                  │
│                                      ▼                                      │
│                           Reciprocal Rank Fusion                            │
│                           Score = 1/(60+R_bm25) + 1/(60+R_dense)            │
│                                      │                                      │
│                                      ▼                                      │
│                             Top-50 Candidates                               │
│                                      │                                      │
│                                      ▼                                      │
│                         Cross-Encoder Neural Reranker                       │
│                        (ms-marco-MiniLM-L-6-v2 @ 512 max)                   │
│                                      │                                      │
│                                      ▼                                      │
│                           Top-5 Reranked Passages                           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                    4. CONTEXT ASSEMBLY & RAG GENERATION                     │
│                                                                             │
│   Top-5 Structured Context Blocks:                                          │
│   [Chunk ID: {id} | Court: {court} | Date: {date}]                          │
│   {passage_text}                                                            │
│                                      │                                      │
│                                      ▼                                      │
│            LLM Generation Layer (Groq - llama-3.3-70b-versatile)                    │
│            Grounding Instructions, Strict Chunk Attribution, temp=0         │
│            Strict Error Shielding (prevents API error leak into answers)    │
│                                      │                                      │
│                                      ▼                                      │
│                           Grounded Legal Answer                             │
│                           with Verifiable Chunk Citations                   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼──────────────────────────────────────┐
│                  5. BENCHMARK & MULTI-CRITERIA EVALUATION                   │
│                                                                             │
│  497 Gold Questions ──────► find_gold_chunks ──────► LLM Judge Evaluator    │
│  (5 Question Types)         (Evidence Validation)   (Relevance, Faith,      │
│                                                      Citations, Halluc)     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Subsystems

### 1. Ingestion & Preprocessing Subsystem
- **Source**: Hugging Face `overthelex/indian-court-decisions` (configuration: `high_courts`, split: `train`).
- **Streaming Pipeline**: To ingest 100,000 records without exceeding local memory or disk buffers, data is streamed with `streaming=True` and a deterministic shuffle buffer (`buffer_size=10,000`, `seed=42`).
- **Deduplication Engine**: An in-memory hash set tracks SHA-256 digests of document texts to guarantee zero duplicate judgments.
- **Corpus Sanitization**: Documents are inspected for control character contamination. Non-printable ASCII control characters `[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]` are stripped regex-wise while retaining valid structural elements, line breaks, section symbols, and quotes. Whitespace sequences are normalized.

### 2. Chunking & Indexing Pipeline
- **Chunking Engine**: LangChain `RecursiveCharacterTextSplitter`.
  - `CHUNK_SIZE`: 1200 characters (~200–250 tokens), selected to maintain cohesive legal holding contexts.
  - `CHUNK_OVERLAP`: 200 characters (~35–40 tokens), mitigating boundary severance across sentences.
  - `MIN_CHUNK_LENGTH`: 100 characters, filtering orphan headers or artifact lines.
  - Produces **538,079 chunks** across 100k judgments (5.38 chunks/doc average).
- **Identifier Scheme**: Every chunk receives a deterministic unique key: `{cnr}_chunk_{index}`.

### 3. Multi-Stage Retrieval Subsystem
The retrieval architecture executes a 3-tier cascade:

```text
Query ──► [ BM25 (Top 50) ] ──┐
                              ├──► [ RRF Fusion (Top 50) ] ──► [ Cross-Encoder (Top 5) ]
Query ──► [ Dense (Top 50) ] ─┘
```

#### Tier 1A: Lexical Retriever (BM25)
- **Engine**: `rank-bm25` implementing `BM25Okapi` ($k_1=1.5, b=0.75$).
- **Tokenization**: Regex-based alphanumeric lowercasing (`\w+`).
- **Role**: Provides fast, exact keyword anchoring for specific section numbers, act titles, and case identifiers.

#### Tier 1B: Dense Vector Retriever
- **Embedding Model**: `BAAI/bge-base-en-v1.5` (768 dimensions, FP16 inference, normalized).
- **Index Architecture**: `faiss.IndexFlatIP` (Cosine similarity over L2-normalized vectors).
- **Execution**: Distributed across 2 × Tesla T4 GPUs with batch size 64 into 108 `.npy` embedding shards before index compilation.
- **Role**: Captures semantic synonyms, general legal concepts, and paraphrased factual narratives.

#### Tier 2: Reciprocal Rank Fusion (RRF)
- Combines the rank positions from BM25 and Dense retrieval without requiring score calibration:
  $$\text{RRF\_Score}(d) = \frac{1}{k + \text{rank}_{\text{BM25}}(d)} + \frac{1}{k + \text{rank}_{\text{Dense}}(d)}$$
  where $k=60$.
- Emits a consolidated candidate pool of Top 50 chunks.

#### Tier 3: Cross-Encoder Neural Reranker
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Scoring**: Full cross-attention over `(query, passage)` pairs up to 512 tokens.
- **Output**: Top 5 highest-confidence chunks passed to context assembly.

### 4. Context Assembly & Prompt Formatting
Retrieved chunks are structured into clean, attributed context blocks:
```text
[Chunk ID: {chunk_id} | Court: {court_name} | Date: {decision_date}]
{chunk_text}
```

The system prompt strictly instructs the generation model to:
- Rely strictly on provided context passages.
- Cite supporting chunks explicitly using the `[Chunk ID: ...]` syntax.
- Acknowledge when the context is insufficient rather than generating unsupported assertions.

### 5. LLM Generation Layer
- **Interface**: Groq Cloud API via official `groq` SDK (`LegalGenerationClient`).
- **Model**: `llama-3.3-70b-versatile` (configurable via `GROQ_MODEL_NAME`).
- **Generation Parameters**: `temperature=0.0` (greedy decoding for reproducibility and factual consistency), `max_tokens=1024`.
- **Error Shielding**: All upstream API errors (auth, quota/rate-limits, network timeouts, invalid requests) are trapped and isolated into structured `GenerationResult` objects and explicit `status="generation_error"` responses, preventing raw error text from leaking into generated answer bodies.

### 6. FastAPI Backend Service Architecture
The LegalRAG service is built with FastAPI and Pydantic v2, architected around singleton dependency injection and dual runtime modes:

```text
                                HTTP Client Request
                                         │
                                         ▼
                            FastAPI Application Router
                         [Pydantic v2 Input Validation]
                                         │
                                         ▼
                             Dependency Injection Layer
                                 (get_pipeline())
                                         │
                       ┌─────────────────┴─────────────────┐
                       ▼                                   ▼
              `local_stub` Pipeline               `production` Pipeline
            - In-memory mock retrievers         - BM25Okapi disk index (~579 MB)
            - Static mock reranker              - FAISS IndexFlatIP (~1.65 GB)
            - Deterministic mock LLM            - BGE-base SentenceTransformer
            - Zero memory / GPU overhead        - ms-marco CrossEncoder
                                                - Groq LLaMA 3.3 70B SDK
                                         │
                                         ▼
                                 Execution Cascade
                    1. Hybrid Retrieval & RRF Fusion (Top-50)
                    2. Neural Cross-Encoder Reranking (Top-5)
                    3. Context Prompt Construction
                    4. Grounded Generation (Groq LLaMA 3.3 70B)
                    5. Citation Verification & Filtering
                    6. High-Resolution Latency Tracking
                                         │
                                         ▼
                              Structured JSON Response
```

#### A. Dual Runtime Environments
- **`local_stub` (Default)**: Lightweight deterministic mock pipeline tailored for resource-constrained development laptops (< 100 MB RAM). Runs full API validation without loading multi-gigabyte models or FAISS index files.
- **`production`**: Loads precomputed artifacts (`bm25.pkl`, `dense.index`, `legal_chunks.parquet`) and connects to Groq API via `GROQ_API_KEY`.

#### B. API Endpoints
- **`GET /health`**: Health probe returning operational status and active runtime environment (`HealthResponse`).
- **`GET /`**: Service root returning metadata, version, environment, and Swagger documentation link.
- **`POST /query`**: End-to-end multi-stage legal retrieval and grounded answer generation (`QueryRequest` -> `QueryResponse`).

#### C. Request & Response Schemas
- **`QueryRequest`**: Requires `question` (3 to 4,000 characters, whitespace stripped).
- **`QueryResponse`**: Returns `answer`, `citations` list, `retrieved_chunk_ids`, granular latency measurements (`retrieve_ms`, `rerank_ms`, `generate_ms`, `total_ms`), and execution `status` (`ok`, `generation_error`, `retrieval_error`).
- **`Citation`**: Metadata including `chunk_id`, `cnr`, `court_code`, `decision_date`, and `title`.

#### D. Production Exception Shielding & LLMOps
- **Sanitized Failures**: Upstream retrieval or generation errors (e.g. rate limits, network timeouts, index faults) are caught and logged server-side via `logger.exception()`.
- **Response Privacy**: API consumers receive safe, standardized messages (`"An error occurred while generating the legal answer. Please try again later."`) with explicit `status="generation_error"` or `status="retrieval_error"`, preventing secret or stack trace leakage.
- **Citation Hallucination Filter**: Regex-extracted `[Chunk ID: ...]` citations are cross-referenced against `top_chunk_ids`. Hallucinated IDs not present in retrieved context are stripped from the response citations list.

### 7. Containerization & Deployment Specification
- **Base Image**: `python:3.10-slim` with system build utilities (`build-essential`).
- **User Permissions**: Adheres to Hugging Face Spaces requirements by creating and running as non-root user `user` with UID `1000`.
- **Port Binding**: Default port `7860` configured via `api_port` with fallback aliases (`PORT`, `API_PORT`).
- **Healthcheck Probe**: `curl -f http://localhost:7860/health || exit 1`.

### 8. Evidence-Grounded Benchmark & Judge Subsystem
- **Benchmark Construction**: 500 candidate questions generated across 5 question types (Reasoning, Outcome, Fact, Legal Provision, Multi-Hop).
- **Evidence Verification (`find_gold_chunks`)**: Validates that candidate supporting text is anchored in corpus chunks via exact/sliding-window matching. Filtered 3 ungrounded candidates, yielding **497 validated gold questions**.
- **Automated Judge**: Multi-criteria evaluation judging:
  1. *Answer Relevance (1–4)*
  2. *Faithfulness (1–4)*
  3. *Citation Correctness (1–4)*
  4. *Overall Score (1–4)*
  5. *Unsupported Claim Rate (0.0–1.0)*
  6. *Hallucination Rate (Boolean)*

---

## Architectural Boundary: Completed vs. Planned

| Subsystem Component | Status | Implementation Details |
| :--- | :---: | :--- |
| **Corpus Cleaning & Serialization** | **Completed** | Cleaned 100k judgments saved in Parquet shards. |
| **Chunking & Index Generation** | **Completed** | 538k chunks indexed via BM25 (`bm25.pkl`) and FAISS (`dense.index`). |
| **Multi-Stage Retrieval Engine** | **Completed** | BM25 + BGE + RRF + Cross-Encoder fully benchmarked. |
| **RAG Generation & Evaluation** | **Completed** | 497 validated questions evaluated with multi-criteria LLM judge. |
| **FastAPI Backend Service** | **Completed** | REST endpoints (`/health`, `/query`) with granular latency attribution and dual runtime modes (`local_stub` / `production`). |
| **Docker & Cloud Deployment** | **Completed** | Hugging Face Spaces Docker containerization and Hub artifact download automation. |
| **React / Streamlit UI** | *Planned (Phase 2)* | Interactive dashboard with chunk highlighting, citation graph, and court filters. |

---

## Data Schemas & Artifact Catalog

### 1. `legal_judgments_clean.parquet`
```json
{
  "cnr": "STRING (e.g., 'PHHC010839062016_1_2016-01-07')",
  "court_code": "STRING",
  "court_name": "STRING (e.g., 'Punjab and Haryana High Court')",
  "decision_date": "STRING (YYYY-MM-DD)",
  "year": "INTEGER",
  "case_type": "STRING",
  "disposal_nature": "STRING",
  "judge": "STRING",
  "petitioner": "STRING",
  "respondent": "STRING",
  "full_text": "STRING (Sanitized judgment transcript)"
}
```

### 2. `legal_chunks.parquet`
```json
{
  "chunk_id": "STRING (format: '{cnr}_chunk_{index}')",
  "cnr": "STRING",
  "chunk_index": "INTEGER",
  "text": "STRING (Chunk text, 100-1200 characters)"
}
```

### 3. `gold_eval.json` (497 validated benchmark records)
```json
{
  "cnr": "STRING",
  "question": "STRING",
  "reference_answer": "STRING",
  "question_type": "STRING (fact | reasoning | legal_provision | outcome | multi_hop)",
  "supporting_text": "STRING",
  "gold_chunk_ids": ["STRING"]
}
```

### 4. `rag_results.json`
```json
{
  "cnr": "STRING",
  "question": "STRING",
  "reference_answer": "STRING",
  "generated_answer": "STRING",
  "question_type": "STRING",
  "retrieved_chunk_ids": ["STRING"]
}
```

### 5. `rag_evaluation.json`
```json
{
  "cnr": "STRING",
  "question_type": "STRING",
  "answer_relevance": 4,
  "faithfulness": 4,
  "citation_correctness": 4,
  "unsupported_claim_rate": 0.0,
  "hallucination": false,
  "overall_score": 4,
  "judge_feedback": "STRING"
}
```
