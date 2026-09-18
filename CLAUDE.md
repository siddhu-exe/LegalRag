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
      generation/, evaluation/ — with pyproject.toml, 20 unit tests, atomic
      conventional commits
- [ ] Fix OmniRoute error-leak bug (see "Known issues" below) — NOT yet fixed
- [ ] Add per-stage latency logging (retrieve_ms / rerank_ms / generate_ms /
      total_ms)
- [ ] Build FastAPI `/query` endpoint wrapping the existing pipeline classes
- [ ] Deploy to Hugging Face Spaces, artifacts hosted on Hugging Face Hub
- [ ] Minimal frontend (Streamlit is fine) — query box, answer, cited chunks
      with source metadata, latency shown
- [ ] README rewritten to lead with eval numbers + live demo link, not a
      feature list
- [ ] Honest paragraph addressing the reranker Recall@5/10 dip (see below)

## Locked architecture — do not change without explicit discussion

These were deliberately chosen and benchmarked. Don't "improve" them silently;
if something looks suboptimal, flag it and ask before changing.

- **Dataset**: `overthelex/indian-court-decisions`, config `high_courts`,
  split `train`, streamed with `buffer_size=10000, seed=42`
- **Corpus size**: 100,000 judgments — this is FINAL, not an intermediate
  step. Do not re-embed at a different scale. (An earlier 80k run exists as
  throwaway scaffolding, superseded — not a numbered experiment.)
- **Chunking**: `RecursiveCharacterTextSplitter`, chunk_size=1200,
  chunk_overlap=200, min_chunk_length=100. Produces 538,079 chunks.
- **Retrieval cascade**: BM25 (k1=1.5, b=0.75, regex tokenization) top-50 +
  Dense (BAAI/bge-base-en-v1.5, 768-dim, FAISS IndexFlatIP) top-50 → RRF
  fusion (k=60) → top-50 → cross-encoder rerank
  (cross-encoder/ms-marco-MiniLM-L-6-v2) → top-5 → LLM context
- **Generation**: OpenAI-compatible client via OmniRoute, temperature=0,
  max_tokens=1024
- **Frozen artifacts** (do not regenerate unless the artifact is provably
  corrupted): `legal_judgments_clean.parquet`, `legal_chunks.parquet`,
  `bm25.pkl`, `dense.index` + embedding shards, `gold_eval.json` (497
  questions), `rag_results.json`, `rag_evaluation.json`

## Known issues — must be fixed, not hidden

1. **OmniRoute error leak (unfixed as of this writing)**: When the OmniRoute
   API returned a routing/model error (e.g. "model no longer available"),
   that error string was saved into `generated_answer` and then scored by the
   LLM judge as a real answer — inflating the hallucination rate and dragging
   down faithfulness. Any new generation code (including the FastAPI
   endpoint) MUST detect API/generation failures explicitly and return a
   distinct error state, never let an error string flow into the answer
   field or get judged as content. This is also the LLMOps / graceful-failure
   story for the writeup — frame it as "found via own eval pipeline, root
   caused, fixed" once done.
2. **Reranker hurts Recall@5/Recall@10**: Cross-encoder reranking improved
   Recall@1 (+3.42pp over hybrid-RRF) but *decreased* Recall@5 (0.5111 →
   0.4869) and Recall@10 (0.5614 → 0.5594) versus hybrid-RRF alone. Likely
   cause: `ms-marco-MiniLM-L-6-v2` is not domain-adapted to legal text. This
   must be stated honestly in the README/writeup, not glossed over — it's a
   good "I read my own numbers critically" signal if framed right, and a red
   flag in an interview if presented as an unambiguous win.

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

- **Local dev laptop**: 6GB RAM, DDR1, very old CPU. Cannot run the real
  pipeline locally — the full stack (BM25 index + FAISS index + BGE-base
  model + cross-encoder model + torch/transformers overhead) needs ~4.15GB
  RAM just to load, leaving no headroom on a 6GB machine, and the CPU
  generation is old enough that PyTorch/FAISS may run very slowly or hit
  missing-instruction-set issues.
- **Consequence — where code runs**:
  - Local laptop: write and test code only, against a small stub/toy index
    (a few hundred fake chunks, no real BM25/FAISS/model weights loaded).
    Used for verifying request/response shapes, error handling, and running
    the existing unit test suite (all tests use stdlib unittest, cheap to
    run anywhere).
  - Kaggle: has the real frozen artifacts and 30GB RAM + 2 GPUs. Used for any
    real integration testing against the actual 538k-chunk index, and for
    uploading artifacts to Hugging Face Hub.
  - Hugging Face: Spaces hosts the deployed FastAPI app (+ frontend); Hub
    hosts the large artifacts (bm25.pkl ~579MB, dense.index ~1.65GB,
    legal_chunks.parquet ~261MB) so the Space downloads them at startup
    rather than needing them baked into a repo or built on a weak machine.
- When asked to "run the pipeline" or "test retrieval end-to-end," check
  which environment the request implies — don't attempt full-index operations
  assuming a local machine that can't hold them.

## Package structure (already built — extend, don't restructure)

```
src/legalrag/
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
    client.py     # OpenAI-compatible greedy decoding interface
  evaluation/
    grounding.py  # find_gold_chunks evidence matcher
    metrics.py    # calculate_recall_at_k, failure taxonomy, aggregation
tests/            # 20 unit tests, stdlib unittest + pytest compatible
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
