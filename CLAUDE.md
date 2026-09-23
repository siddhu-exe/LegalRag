# LegalRAG — Project Context for Claude

This file is read automatically by Claude Code at the start of every session in
this repo. It exists so context doesn't have to be re-explained each time.
Keep it updated as decisions change — treat it as the project's memory, not a
one-time README.

## What this project is

A hybrid-retrieval RAG system over 100,000 Indian High Court judgments, built
as a flagship **AI Engineering** portfolio project (not ML Engineering, not
MLOps — see "Positioning" below). The differentiator is a real evaluation
harness (retrieval scoreboard + LLM-judge generation metrics on a 497-question
gold benchmark), not just a working demo.

Target audience: Indian fresher AI/GenAI engineering job market, 2026.

## Current state (update this section as work progresses)

- [x] 100k judgment corpus ingested, cleaned, chunked (538,079 chunks) — frozen
      artifacts on Kaggle
- [x] BM25 + Dense (BGE-base) + Hybrid RRF + Cross-encoder reranker, all
      benchmarked with Recall@1/5/10/50 on the 497-question set
- [x] 497-question evidence-grounded gold benchmark + LLM-judge eval
      (relevance, faithfulness, citation correctness, unsupported claim rate,
      hallucination rate, overall score)
- [x] Refactored into `src/legalrag/` package: preprocessing/, retrieval/,
      generation/, evaluation/ — with pyproject.toml, unit tests, atomic
      conventional commits
- [x] Fix OmniRoute error-leak bug: Migrated generation to official Groq SDK
      (`groq`, model selected by `GROQ_MODEL_NAME`, default `qwen/qwen3.8-27b`) with typed
      `GenerationResult` and strict exception shielding to prevent raw API error strings
      from leaking into answer fields or judge evaluations
- [x] Add per-stage latency logging: `retrieve_ms`, `rerank_ms`, `generate_ms`,
      and `total_ms` captured monotonically via `time.perf_counter()`
- [x] Build FastAPI backend (`src/legalrag/api/`): `/health`, `/ready`, and `/query`
      endpoints wrapping retrieval, reranking, and generation with dual-mode
      dependency injection (fail-closed `production` vs explicit `local_stub`)
- [x] Deploy preparation: Multi-stage `Dockerfile` (port 7860, non-root user
      UID 1000) and `scripts/download_artifacts.py` for Azure Container Apps / HF Hub
- [ ] Minimal frontend (Streamlit is fine) — query box, answer, cited chunks
      with source metadata, latency shown
- [x] README rewritten with retrieval/generation eval benchmarks, API quickstart,
      and honest analysis of reranker Recall@5/10 trade-offs
- [x] Honest paragraph addressing the reranker Recall@5/10 dip documented in
      README and architecture docs
- [x] Pre-deployment artifact-contract audit against the real production artifacts
      (538,079 chunks: `bm25.pkl`, `dense.index`, `legal_chunks.parquet`): fixed the
      BM25 `"bm25"` serialization mismatch, restored the BGE query instruction in
      `DenseRetriever.search()`, and validated `load_production_pipeline()` /
      `check_readiness()` / `POST /query` locally

## Locked architecture — do not change without explicit discussion

These were deliberately chosen and benchmarked. Don't "improve" them silently;
if something looks suboptimal, flag it and ask before changing.

- **Dataset**: `overthelex/indian-court-decisions`, config `high_courts`,
  split `train`, streamed with `buffer_size=50000, seed=42`
- **Corpus size**: 100,000 judgments — this is FINAL, not an intermediate
  step. Do not re-embed at a different scale. (An earlier 80k run exists as
  throwaway scaffolding, superseded — not a numbered experiment.)
- **Chunking**: `RecursiveCharacterTextSplitter`, chunk_size=1200,
  chunk_overlap=200, min_chunk_length=100. Produces 538,079 chunks.
- **Retrieval cascade**: BM25 (k1=1.5, b=0.75, regex tokenization) top-50 +
  Dense (BAAI/bge-base-en-v1.5, 768-dim, FAISS IndexFlatIP) top-50 → RRF
  fusion (k=60) → top-50 → cross-encoder rerank
  (cross-encoder/ms-marco-MiniLM-L-6-v2) → top-5 → LLM context
- **Generation**: Groq via official `groq` SDK, model ID from `GROQ_MODEL_NAME`
  (default `qwen/qwen3.8-27b`), temperature=0, max_tokens=1024, request timeout
  30 s, bounded retries, strict exception shielding
- **Frozen artifacts** (do not regenerate unless the artifact is provably
  corrupted): `legal_judgments_clean.parquet`, `legal_chunks.parquet`,
  `bm25.pkl`, `dense.index` + embedding shards, `gold_eval.json` (497
  questions), `rag_results.json`, `rag_evaluation.json`

## Known issues & Technical Insights

1. **OmniRoute error leak (FIXED)**: Previous OpenAI/OmniRoute routing errors
   leaked raw exception strings into `generated_answer`. Now resolved with
   `LegalGenerationClient` (`src/legalrag/generation/client.py`) using typed
   `GenerationResult` (`status`, `is_success`, `error_type`), strict error
   shielding in `src/legalrag/api/routes.py`, and sanitized user-facing error
   responses with server-side `logger.exception()` logging.
2. **BM25 artifact serialization mismatch (FIXED)**: the production `bm25.pkl` is a
   pickled dict `{"bm25": <BM25Okapi>, "chunk_ids": [...]}` (the format written by
   `Notebooks/legalrag-100k-final.ipynb`). `BM25Retriever.load()` previously read only
   `state["model"]`, so production silently produced a retriever with `model=None` while
   `chunk_ids` loaded correctly, and every query failed. The loader now reads
   `bm25`, then `model`, then a bare pickled `BM25Okapi`, and fails closed if none of
   those are present. Regression tests live in `tests/test_artifact_contracts.py`.
3. **Dense query instruction was omitted (FIXED)**: the index was built for BGE retrieval
   queries prefixed with `"Represent this sentence for searching relevant passages: "`
   (recorded in `dense_index_metadata.json`). `DenseRetriever.search()` now applies that
   instruction to queries only and keeps `search(..., top_k)` results consistent with the
   experiment that produced `dense.index`. `dense.index` was NOT regenerated.
4. **Groq model decommissioning**: `llama-3.3-70b-versatile` is no longer served by Groq, so
   any deployment using it receives a provider 404 (`model_not_found`), which the service
   correctly sanitizes into HTTP 502. `GROQ_MODEL_NAME` must be set to a currently served
   model ID (default is now `qwen/qwen3.8-27b`); verify with
   `curl https://api.groq.com/openai/v1/models`.
5. **Reranker hurts Recall@5/Recall@10 (Domain-Adaptation Trade-off)**:
   Cross-encoder reranking improved Recall@1 (+3.42pp over hybrid-RRF,
   0.3219 → 0.3561) but *decreased* Recall@5 (0.5111 → 0.4869) and
   Recall@10 (0.5614 → 0.5594) versus hybrid-RRF alone. Root cause:
   `ms-marco-MiniLM-L-6-v2` is trained on general web search / MS-MARCO and
   is not domain-adapted to specialized Indian legal phrasing. This is
   documented honestly in all project literature.

## Positioning — AI Engineering, not ML Engineering / not MLOps

- Say "AI Engineer" / "GenAI Engineer" in resume bullets, READMEs, and any
  pitch. Not "ML Engineer."
- Why: ML Engineers train models from scratch (architectures, gradient
  descent, hyperparameter search). This project applies pre-trained models
  (BGE embeddings, MS-MARCO cross-encoder, an LLM via API) — that's AI
  Engineering: RAG architecture, retrieval engineering, evaluation
  infrastructure, latency, production failure handling.
- Full MLOps (drift monitoring, automated retraining) is NOT needed and
  should not be added — there's no trained-from-scratch model to retrain or
  monitor for drift. Don't scope-creep into this.
- Light LLMOps IS in scope: graceful API failure handling, distinguishing
  system errors from bad answers, basic request/error-rate observability.
  This is achievable at solo-project scale and expected in current AI
  engineer job descriptions.

## Hardware / environment constraints — important for how work gets split

- **Current local machine**: can hold the full real stack (BM25 index + FAISS
  index + BGE-base model + cross-encoder model + torch/transformers). Measured
  with the real 538,079-chunk artifacts: ≈10.5 GB peak RSS, ≈29.3 s cold
  pipeline init, and ≈4.3 s per retrieval+rerank query on CPU (BM25 dominates).
  Unit/integration tests still run entirely on stubs — no model downloads, no
  FAISS index, no network.
- **Historical constraint**: an older 6GB-RAM laptop could not hold the stack
  (~4.15GB just to load), which is why local work was previously restricted to
  stub/toy indexes.
- **Consequence — where code runs**:
  - Local machine: unit/integration tests against stubs; end-to-end validation
    against the real artifacts when present in `artifacts/` (gitignored — never
    committed and never baked into the Docker image).
  - Kaggle: original multi-GPU embedding/generation run and artifact upload.
  - Hugging Face: Hub dataset `siddhu23/LegalRag_Dataset` hosts the large
    artifacts (bm25.pkl ~579MB, dense.index ~1.65GB, legal_chunks.parquet
    ~261MB); Spaces/containers download them at startup.
- When asked to "run the pipeline" or "test retrieval end-to-end," first check
  whether the real artifacts are present in `artifacts/`, then choose stub vs.
  real execution explicitly.

## Package structure

```
src/legalrag/
  api/
    config.py     # Pydantic v2 Settings with secret masking & port fallback
    dependencies.py # Thread-safe singleton DI, fail-closed production vs local_stub
    main.py       # FastAPI application factory & metadata routes
    provisioning.py # Startup HF artifact provisioning (bm25.pkl/dense.index/legal_chunks.parquet)
    routes.py     # /health, /ready, /query with HTTP error semantics & latency breakdown
    schemas.py    # Request/Response models with input validation
  preprocessing/
    cleaner.py    # control-char stripping, >1% corruption detection
    chunker.py    # LegalChunker, locked hyperparameters
  retrieval/
    bm25.py       # BM25Retriever, legal regex tokenization
    dense.py      # DenseRetriever, BGE-base + FAISS IndexFlatIP
    fusion.py     # Reciprocal Rank Fusion, k=60
    reranker.py   # Cross-encoder reranker
  generation/
    prompts.py    # locked RAG context formatting + system prompt
    client.py     # Groq typed client (model via GROQ_MODEL_NAME)
  evaluation/
    grounding.py  # find_gold_chunks evidence matcher
    metrics.py    # calculate_recall_at_k, failure taxonomy, aggregation
tests/            # 133 unit & integration tests, stdlib unittest + pytest + TestClient
```

New work (FastAPI app, latency logging, etc.) should live under
`src/legalrag/api/` or similar, importing the existing retrieval/generation
classes rather than reimplementing pipeline logic.

## Style / working preferences for this project

- Staged collaboration: brainstorm/plan before building.
- Direct answers before elaboration.
- Plain-text, minimal responses preferred generally, but this file itself is
  reference documentation and can stay structured/detailed.
- Preserve existing code style, variable names, and experiment history when
  refactoring — don't silently rewrite working code for taste reasons.
