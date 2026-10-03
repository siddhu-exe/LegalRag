# LegalRAG Improvement Plan (LLM-executable)

Audience: an AI coding agent (Claude Code, Codex, Qwen, etc.) working in this repo.
Purpose: turn LegalRAG from "strong portfolio demo with overstated claims" into a
system whose evaluation is credible and whose service is safe to expose publicly.

Read `CLAUDE.md` first. Everything below is subordinate to its "Locked architecture"
section. Where a task touches a locked item, the task says so and requires user approval.

---

## 0. Ground rules (apply to every task)

1. **Never overwrite frozen artifacts**: `legal_judgments_clean.parquet`,
   `legal_chunks.parquet`, `bm25.pkl`, `dense.index`, `gold_eval.json`,
   `rag_results.json`, `rag_evaluation.json`. New outputs go to a new directory:
   `artifacts/eval_v2/` (gitignored) and summary JSON/CSV under `results/eval_v2/` (committed, small files only).
2. **Do not change production defaults of the locked retrieval cascade** (reranker model,
   top-k values, RRF k, chunking, embedding model). Implement alternatives as
   *experiments* selectable by config/flag; report numbers; ask the user before
   switching the default.
3. **Real artifacts may not be present locally.** Check `artifacts/` first. If missing,
   download with `python scripts/download_artifacts.py --repo-id siddhu23/LegalRag_Dataset --target-dir artifacts`.
   Unit tests must still run on stubs only (no network, no models).
4. Every new module gets unit tests in `tests/` using stubs. Keep `pytest -q` green.
5. One task = one conventional commit (`feat(eval): ...`, `fix(api): ...`, `docs: ...`).
6. Match existing style: type hints, module docstrings, `logger = logging.getLogger(__name__)`,
   sanitized error messages in `src/legalrag/api/routes.py`.
7. Report numbers honestly. If an experiment does not help, record it as a negative result.
8. Any script that calls a paid LLM API must: checkpoint results to disk after every item,
   be resumable, and print an estimated call count before starting.

---

## Phase P0 — Honesty & repo hygiene (est. 0.5–1 day)

Goal: remove claims an interviewer can disprove in 10 minutes.

| ID | Task | Files | Acceptance criteria |
|----|------|-------|---------------------|
| P0.1 | Fix failure-taxonomy wording. 255/497 = 51.3% of **all questions**; retrieval is 255/303 ≈ **84.2% of failures**. | `README.md`, `docs/EXPERIMENT_REPORT.md`, `LLMS.md`, `CLAUDE.md` | No doc says "51.3% of generation failures". Both numbers stated correctly. |
| P0.2 | Remove/qualify "negligible hallucination when gold in context" unless backed by a conditional table (see P1.6). | same | Claim removed or replaced with measured conditional rate. |
| P0.3 | Rename "Citation Extractor & Grounding Filter (Verifies citations against context)" to what it does: "Citation ID filter (drops IDs not in retrieved context)". | `README.md`, `docs/ARCHITECTURE.md` | Diagram text matches `routes.py` behaviour. |
| P0.4 | Fix EC2 instance type inconsistency (`t3.xlarge` everywhere; `t4.xlarge` does not exist). | `README.md`, docs | `grep -r "t4.xlarge"` returns nothing. |
| P0.5 | Add a `LICENSE` file (MIT, as the badge claims) or remove the badge. Ask user which. | root | Badge link resolves. |
| P0.6 | Remove committed junk: `frontend/.vite/`, decide on `data/raw/judgments.parquet` (move to HF or keep with a note). Add to `.gitignore`. | `.gitignore` | `git ls-files frontend/.vite` empty. |
| P0.7 | Remove stale Streamlit roadmap item (React frontend already shipped). | `README.md` | — |
| P0.8 | State clearly in README that published generation metrics were produced with the OmniRoute experiment model, **not** the deployed Groq model, until P1.2 is done. | `README.md` | Caveat present. |
| P0.9 | Tone down marketing language ("proving", "research-grade", "Critical ... Finding"). Target: README ≤ 50% of current length. | `README.md` | Reads like an engineer's write-up, not an ad. |

---

## Phase P1 — Make the evaluation credible (est. 4–6 days) — HIGHEST PRIORITY

Background: in `Notebooks/legalrag-100k-final.ipynb` the same model
(`OMNIROUTE_MODEL='kaggle'`) generated the questions, the answers, **and** judged them.
The deployed model (`GROQ_MODEL_NAME`, default `qwen/qwen3.8-27b`) was never evaluated.
No confidence intervals. Questions generated from one judgment are often ambiguous
across a 100k corpus.

Create package module `src/legalrag/evaluation/` additions and a CLI at
`scripts/run_eval_v2.py`.

| ID | Task | Details | Acceptance criteria |
|----|------|---------|---------------------|
| P1.1 | **Offline retrieval eval script** | `scripts/run_retrieval_eval.py`: loads real pipeline via `load_production_pipeline()` from `src/legalrag/api/dependencies.py`, runs all 497 `gold_eval.json` questions through BM25 / Dense / RRF / Rerank, computes judgment-level Recall@{1,3,5,10,25,50} using `calculate_recall_at_k` in `src/legalrag/evaluation/metrics.py`. Save per-question ranks to `artifacts/eval_v2/retrieval_runs.parquet`. | Reproduces README retrieval table within ±0.5pp. If not, investigate and report the discrepancy before continuing. |
| P1.2 | **Re-run generation on the deployed model** | For each question, call the real `/query` code path (import functions, don't go over HTTP) with the production Groq model. Save `artifacts/eval_v2/rag_results_groq.json`. | 497 records; failures recorded with `status`, not dropped. |
| P1.3 | **Cross-family judge** | New `src/legalrag/evaluation/judge.py`: reuse the judge rubric from notebook cell "30 RAG evaluation judge" verbatim, but make the judge model configurable (`JUDGE_MODEL`), and it MUST be from a different model family than the generator (e.g. generator = Qwen → judge = Llama/Gemma/GPT-OSS/Claude). Strict JSON parsing, retries, checkpointing. | `judge_model != generator_model` asserted in code. |
| P1.4 | **Human calibration of the judge** | `scripts/export_judge_sample.py` exports a stratified random sample of 50 items (10 per question type, seed 42) to `results/eval_v2/human_labels_template.csv` with columns for the human to fill (faithfulness 1–4, hallucination yes/no). After the user fills it, `scripts/judge_agreement.py` computes Cohen's κ (faithfulness binarized at ≥3, and hallucination) and raw agreement. | Agreement table in `results/eval_v2/judge_agreement.json`. Do NOT fabricate human labels — the user fills them. |
| P1.5 | **Bootstrap confidence intervals** | `src/legalrag/evaluation/stats.py`: `bootstrap_ci(values, n=10000, alpha=0.05, seed=42)` and `paired_bootstrap_diff(a, b, ...)` for comparing two systems on the same questions. Apply to every metric in README tables. | Every README number shows `value [low, high]`. Reranker vs RRF comparisons show paired-difference CI and whether it excludes 0. |
| P1.6 | **Conditional metrics** | Report generation metrics split by `gold_in_top5 ∈ {True, False}`. | Table with hallucination rate & faithfulness for each condition. |
| P1.7 | **Ambiguity audit of the benchmark** | LLM pass (different family from question generator) labels each question `self_contained: bool` = "could a lawyer identify the target judgment from this question alone among 100k judgments?" Human spot-check 30. Report all retrieval metrics on full set AND `self_contained=True` subset. Write labels to `results/eval_v2/question_ambiguity.json`; never modify `gold_eval.json`. | Two retrieval tables in README. |
| P1.8 | **Lexical-overlap bias check** | For each question compute token overlap (Jaccard over `tokenize_legal_text`) with its `supporting_text`. Report BM25 vs Dense Recall@5 bucketed by overlap tercile. | Table showing whether BM25's advantage shrinks on low-overlap questions. README claim about "BM25 proves..." rewritten to match evidence. |
| P1.9 | Update README / EXPERIMENT_REPORT with v2 numbers; keep v1 numbers in a "historical" section. | — | — |

Budget note: P1.2 + P1.3 + P1.7 ≈ 1,500 LLM calls. Print estimate and ask user before running.

---

## Phase P2 — Improve retrieval (est. 5–8 days)

Recall@50 = 65.8% → ~34% of questions are unanswerable no matter what the LLM does.
All items are **experiments**; production default changes need user approval (locked architecture).
Evaluate every experiment with P1.1 harness + paired bootstrap vs current production.

| ID | Experiment | Details |
|----|-----------|---------|
| P2.1 | **Judgment-level aggregation** | After RRF, group chunks by `cnr`, score judgment = max (or sum of top-2) chunk score, take top-N judgments, then pick best chunks within them for context. Pure post-processing, no artifact change. New module `src/legalrag/retrieval/aggregation.py`. |
| P2.2 | **Better reranker** | Compare `cross-encoder/ms-marco-MiniLM-L-6-v2` (current) vs `BAAI/bge-reranker-v2-m3` vs `BAAI/bge-reranker-base`. Measure Recall@1/5/10 and CPU latency per query. Make model name configurable via existing `reranker_model_name` in `src/legalrag/api/config.py`. |
| P2.3 | **Query rewriting / expansion** | LLM rewrites user question into (a) a keyword query for BM25 (statutes, sections, legal terms) and (b) a natural query for dense. Optionally HyDE for dense. Must be behind a flag and measured for added latency. |
| P2.4 | **Metadata filters** | Optional request fields `court_code`, `year_from`, `year_to` in `QueryRequest` (`src/legalrag/api/schemas.py`); filter candidates post-retrieval. Default = no filter (backwards compatible). |
| P2.5 | **RRF weighting** | Try weighted RRF (BM25 weight > dense) given BM25's stronger standalone results. Tiny change in `src/legalrag/retrieval/fusion.py` (add optional `weights` param, default equal → unchanged behaviour). |
| P2.6 | **(Optional, needs GPU/Kaggle)** Fine-tune the reranker on a train split of gold pairs. MUST split questions into train/test first (e.g. 300/197 by `cnr`) and report only on test. Mention clearly in README that this is fine-tuning (still AI-engineering scope, small). |

Deliverable: `results/eval_v2/retrieval_experiments.md` with one row per experiment:
metric deltas with CIs, latency delta, decision (adopt / reject / needs approval).

---

## Phase P3 — Real grounding & abstention (est. 3–5 days)

Current state: `routes.py` only drops cited chunk IDs not in top-5. It does not check that
claims are supported. No abstention.

| ID | Task | Details | Acceptance criteria |
|----|------|---------|---------------------|
| P3.1 | **Abstention threshold** | Use top reranker score. Calibrate threshold on eval_v2 data: choose threshold that maximizes (correct answers) subject to hallucination rate ≤ target (e.g. 5%) on answered questions. If below threshold → return fixed "insufficient evidence" answer with retrieved sources, skip LLM call. Add `status="abstained"` to `QueryResponse`. | Coverage vs hallucination curve plotted in `results/eval_v2/`. Unit tests for both branches. |
| P3.2 | **Claim-level citation verification** | New `src/legalrag/generation/verifier.py`: split answer into sentences, for each sentence with a `[Chunk ID: x]` check support (NLI model e.g. `cross-encoder/nli-deberta-v3-small`, or a cheap LLM call). Mark unsupported sentences; return `verified: bool` per citation in response. | New response field `citations[i].supported`. Re-evaluate hallucination rate "after verifier". |
| P3.3 | **Human-usable citations** | `Citation` schema already has `cnr`, `court_code`, `decision_date`, `title`. Add `quote` (the supporting span, ≤300 chars) and, if available in data, a source URL. Frontend `CitationCard.tsx` shows title + court + date + quote. | No raw chunk IDs shown to end users as the primary citation. |
| P3.4 | **Legal-safety layer** | Static disclaimer in API response + frontend ("Research aid, not legal advice; verify against the official judgment"). Simple classifier/keyword rule for personal-advice queries ("should I…", "can I sue…") → answer with disclaimer emphasis. | Visible on every result page. |

---

## Phase P4 — Production hardening (est. 2–4 days)

| ID | Task | Details |
|----|------|---------|
| P4.1 | **Lock down CORS** | `src/legalrag/api/main.py` currently `allow_origins=["*"]`. Make it a `Settings` field `cors_allowed_origins` (default: `https://legalrag-six.vercel.app`, `http://localhost:5173`). |
| P4.2 | **Rate limiting** | Per-IP limit on `POST /query` (e.g. `slowapi`, 10/min). Return 429 with sanitized message. Tests with TestClient. |
| P4.3 | **Optional API key** | `API_KEYS` setting; if set, require `X-API-Key` header. Off by default in `local_stub`. |
| P4.4 | **Fast BM25** | `rank_bm25.get_scores` over 538k docs ≈ 2.6 s. Evaluate `bm25s` (sparse matrix BM25). **Locked-architecture caveat**: k1=1.5, b=0.75 and tokenization must be identical; verify identical top-50 on the 497 questions (≥99% overlap) before proposing. This requires re-indexing BM25 into a new artifact — ask user first. |
| P4.5 | **Safer artifact loading** | Before `pickle.load` in `BM25Retriever.load`, verify SHA-256 against a pinned hash in config/manifest. Same for `dense.index`. Fail closed on mismatch. |
| P4.6 | **Concurrency** | Keep heavy CPU work in threadpool (current sync `def` is OK) but run uvicorn with an explicit worker/thread config; document memory implications (~10.5 GB per worker → 1 worker + threadpool). Add a semaphore limiting concurrent `/query` to N and return 503 when saturated. |
| P4.7 | **Observability** | `prometheus-fastapi-instrumentator` or manual counters: request count, status codes, p50/p95 of `retrieve_ms/rerank_ms/generate_ms`, abstention rate. Add `X-Request-ID` middleware; include in logs. |
| P4.8 | **Query/answer logging** | Append-only JSONL (question, retrieved ids, answer, status, latencies, request id) with a config flag; no PII beyond the query text; document retention. |

---

## Phase P5 — Feedback loop (est. 1–2 days)

| ID | Task |
|----|------|
| P5.1 | `POST /feedback {request_id, rating: up/down, comment?}` stored in JSONL/SQLite. |
| P5.2 | Thumbs up/down in `frontend/src/components/ResultsView.tsx`. |
| P5.3 | `scripts/feedback_to_eval.py`: converts thumbs-down items into candidate eval questions for human review (never auto-added to gold). |

---

## Phase P6 — Presentation (est. 1 day)

| ID | Task |
|----|------|
| P6.1 | README structure: Problem → Architecture (1 diagram) → Results with CIs → What I found (incl. negative results) → Limitations → Run it → Demo GIF/video link. |
| P6.2 | Record 60–90 s demo video (EC2 is usually stopped). Link at top of README. |
| P6.3 | Consolidate notebooks: mark `legalrag-100k-final.ipynb` canonical; move others to `Notebooks/archive/` with a one-line note. |
| P6.4 | Trim/merge AI-agent context files (`QWEN.md`, `LLMS.md`, `AGENTS.md`) — keep one (`CLAUDE.md`) or move them under `.agents/`. |
| P6.5 | Update `CLAUDE.md` "Current state" checklist after each phase. |

---

## Execution order & time estimate

Estimates assume one person, focused work days (~6 h). LLM-API waiting time included.

| Phase | Effort | Priority |
|-------|--------|----------|
| P0 Honesty & hygiene | 0.5–1 day | Must |
| P1 Credible evaluation | 4–6 days | Must |
| P3.1 + P3.3 + P3.4 Abstention, citations, disclaimer | 2–3 days | Must |
| P4.1–P4.3 Security basics | 1 day | Must (before sharing link publicly) |
| P6 Presentation | 1 day | Must |
| **Minimum credible version** | **≈ 9–12 days** | |
| P2 Retrieval experiments | 5–8 days | Should |
| P3.2 Claim verifier | 1–2 days | Should |
| P4.4–P4.8 Perf & observability | 2–3 days | Should |
| P5 Feedback loop | 1–2 days | Nice |
| **Full plan** | **≈ 19–28 days** | |

Calendar: ~2–3 weeks full-time for the minimum version, ~5–6 weeks full-time for everything;
roughly double that if part-time.

Dependencies: P0 → P1 → (P2, P3 in parallel) → P4 → P5 → P6 (README last, after numbers are final).

---

## Definition of done

- Every number in README has a confidence interval and was produced by a script in this repo.
- Generation metrics are for the model that is actually deployed, judged by a different model family, with reported human–judge agreement.
- The public endpoint cannot be used to drain the Groq key (CORS + rate limit).
- The system abstains instead of answering when evidence is weak, and shows human-readable citations with quotes.
- `pytest -q` passes on stubs; no frozen artifact was modified.
