# LegalRAG v2 Plan: Bombay High Court Pilot

Audience: the project owner and any AI coding agent working in this repo.

Goal: turn the 100k portfolio demo into a research tool that a practising Bombay High Court
lawyer can test on real work, and record results that show the working habits of an
experienced data scientist. The tool covers all practice areas.

Plain-language version: [WHAT_TO_FIX.md](WHAT_TO_FIX.md). Corpus numbers:
[DATA_PROFILE_BOMBAY.md](DATA_PROFILE_BOMBAY.md). Experiment history:
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

Status: P1 is done (peak 9.30 GiB, with BM25 the largest part at 4.04 GiB; see
`results/ram_baseline.json`). No other task is done. All RAM, chunk and cost figures marked
"estimate" must be replaced with measurements.

---

## 1. Decisions

| Decision | Choice |
|---|---|
| Court | Bombay High Court, all benches, code `27_1` |
| Practice areas | General, no filtering by subject |
| Supporting authority | Supreme Court config of the same dataset |
| v1 (100k, 538,079 chunks) | Kept unchanged as the historical baseline |
| v2 artifacts | New directory `artifacts/v2/`, never overwriting v1 |
| Search index | Reasoned judgments, starting at 5,000 characters or more (P3 tests this) |
| Case lookup | All records, searchable by title, CNR, date and judge |
| Experience claims | Show evidence of the work. Never claim years of experience you do not have |

Suggested v2 corpus, from the measured profile:

| Part | Judgments | Characters |
|---|---:|---:|
| Bombay HC, 5,000+ characters | 266,074 | 4.46 billion |
| Supreme Court, 5,000+ characters | 46,321 | 1.20 billion |
| **Total in search index** | **312,395** | **5.66 billion** |
| Lookup only: other Bombay HC records | 997,115 | not indexed for search |

Estimate: about 5.7 million chunks, roughly 10x v1, at the current chunk settings. A flat
FAISS index at that size would need about 17.5 GB, so approximate search is required.

**Hosting consequence:** the corpus grows while RAM must shrink, so t3.medium (4 GB) is likely
too small for the full v2 corpus. Expect t3.large or t4g.large (8 GB). If that is too costly,
start with Bombay HC 2015 onward (181,705 judgments) and grow later. P5 makes this decision
using measurements.

---

## 2. Ground rules

1. Never overwrite frozen v1 artifacts: `legal_judgments_clean.parquet`, `legal_chunks.parquet`,
   `bm25.pkl`, `dense.index`, `gold_eval.json`, `rag_results.json`, `rag_evaluation.json`.
2. Build each new retrieval component behind a config flag. It becomes the default only after
   beating the current version on the benchmark, or after an accepted and recorded trade-off.
3. Unit tests use stubs only: no network, no models. Keep `pytest -q` green.
4. One task, one conventional commit. Cloud work goes on a branch and is reviewed before merge.
5. Scripts that call a paid LLM print an estimated cost first, save after every item, and can
   resume after a failure.
6. Record every experiment in `docs/EXPERIMENT_LOG.md`, including the ones that fail.
7. A number goes into the README only if a script in this repo produced it.
8. The question generator, the answer model and the judge must not all be the same model.
9. Do not fabricate human labels. The owner and the lawyer provide them.

---

## 3. Cloud work (Claude Code on the web)

Cloud sessions write code and run stub tests. They do not need the real corpus. The repo must
be pushed to GitHub first.

| ID | Task | Output | Done when |
|---|---|---|---|
| C1 | Corpus builder: select court, keep reasoned judgments, clean repeated text, map bench names, parse case type from title, choose a date rule, remove duplicates | `scripts/build_corpus_v2.py`, tests | Fixtures cover each rule, and dropped rows are counted by reason |
| C2 | Data quality report generator | `scripts/data_quality_report.py` | Reports OCR noise, duplicates, missing dates and coverage by bench |
| C3 | Statistics and evaluation: bootstrap confidence intervals, paired comparison, independent judge, failure labels | `evaluation/stats.py`, `evaluation/judge.py`, `scripts/run_retrieval_eval.py`, `scripts/run_generation_eval.py` | The judge must differ from the generator, and every metric has an interval |
| C4 | Low-memory parts behind flags: `bm25s` with mmap, IVF-SQ8 and OPQ-PQ FAISS builders, SQLite chunk store, ONNX Runtime embedder and reranker without PyTorch | `retrieval/`, `requirements-serving.txt` | Stub tests and artifact contract tests pass |
| C5 | Trust layer: exact quote check, refusal status, readable citations, court, bench, year and judge filters, disclaimer | `generation/verifier.py`, API schemas, frontend | An unmatched quote never reaches the user as verified |
| C6 | IPC to BNS (also CrPC to BNSS, Evidence Act to BSA) query mapping from the official table | `retrieval/statute_map.py`, data file | Old and new section numbers each find both forms |
| C7 | Case lookup across all records | Lookup module and endpoint | Results say "not in our data", never "does not exist" |
| C8 | Security: restricted CORS, rate limit, login or API keys, no-logging mode, request IDs, basic metrics | `api/` | Tests cover 401, 429 and no-log behaviour |
| C9 | Pilot tools: accept, reject or edit buttons, time-saved prompt, comments, pilot report | `POST /feedback`, frontend, `scripts/pilot_report.py` | The report can be built from sample data |
| C10 | Reranker training code with a split by judgment, plus ONNX export | `scripts/train_reranker.py` | A judgment cannot appear in both train and test |
| C11 | Trade-off and refusal analysis tools | `scripts/tradeoff_table.py`, `scripts/refusal_curve.py` | They produce the tables and plots in section 5 |

---

## 4. PC work (your machine)

Machine: Core Ultra 7 265, about 31 GB RAM, RTX A1000 with 8 GB VRAM, about 338 GB free disk.

| ID | Task | Needs | Done when |
|---|---|---|---|
| P1 | Download v1 artifacts and measure RAM by component and at peak | `requirements.txt` | `results/ram_baseline.json` replaces the estimates |
| P2 | Get 20–30 real research questions from the lawyer, with expected answers or judgments | The lawyer | `data/eval_v2/lawyer_questions.json` (kept private) |
| P3 | Build the corpus. Read 100 random short orders and decide if the 5,000-character cut drops real reasoning | C1, C2 | Cut justified or changed, and the data quality report is written |
| P4 | Measure embedding speed on 10,000 chunks, then embed on PC or Kaggle, whichever is practical | P3 | Embeddings saved with a manifest |
| P5 | Build index options and compare recall, RAM, speed and monthly cost. Choose the instance | C4, P4 | Trade-off table written and the choice recorded |
| P6 | Create the v2 benchmark: lawyer questions plus about 500 generated questions, balanced by bench and year. Label 50 judge results yourself | C3, P2, P3 | Judge agreement (Cohen's kappa) reported |
| P7 | Run the baseline. Personally read failures and label their causes | C3, P5, P6 | Failure table by cause and question type |
| P8 | Fine-tune the reranker, using benchmark judgments only as test data | C10, P7 | Kept only if the paired test supports it |
| P9 | Set the refusal threshold on one data split and report it on another | C5, C11, P7 | Coverage against wrong-answer curve |
| P10 | Load test on the chosen instance with reranking on, then deploy with login | C8, P5 | p50 and p95 latency, RAM and CPU credit use recorded |
| P11 | Run a 2–4 week pilot with the lawyer, then write up results | C9, P10 | Pilot report with accepted answers and time saved |

---

## 5. Experience evidence

Each item must exist as a real output. These are what an interviewer can inspect.

| Habit | What to show | Output | Tasks |
|---|---|---|---|
| **Real users** | Run the pilot and report how many answers the lawyer accepted and how much time it saved | `results/pilot/pilot_report.md` | C9, P2, P11 |
| **Sound statistics** | Use error bars and paired tests, and claim an improvement only when the numbers support it | `results/eval_v2/metrics.json`, README tables | C3, P6, P7 |
| **Failure analysis** | Break failures down by cause and question type, then show what was fixed | `results/eval_v2/failure_analysis.md` | C3, P7 |
| **Failed experiments** | Keep a log that includes what did not work, such as the reranker dip | `docs/EXPERIMENT_LOG.md` | All |
| **Trade-offs** | Show quality against cost, RAM and speed, and explain the choice | `results/eval_v2/tradeoffs.md` | C11, P5 |
| **Data quality** | Report OCR errors, duplicates, missing dates and coverage by bench | `results/data_profile/data_quality_report.md` | C2, P3 |
| **Refusal tuning** | Show how the "not enough evidence" threshold trades answered questions against wrong answers | `results/eval_v2/refusal_curve.png` | C5, C11, P9 |

The owner should personally do the failure review, the human labels and the final write-up, and
be able to explain every number in them. Practise SQL, A/B testing and statistics separately,
because this project does not fully cover them.

---

## 6. Order and time

| Week | Cloud | PC |
|---|---|---|
| 1 | C1, C2, C3 | P1, P2, P3 |
| 2 | C4 | P4 |
| 3 | C6, C7 | P5, P6 |
| 4 | C5, C10, C11 | P7, P8, P9 |
| 5 | C8, C9 | P10 |
| 6–9 | Fixes from the pilot | P11 |

Allow five weeks of full-time build plus the pilot. Part-time work takes about twice as long.
Embedding (P4) is the largest schedule risk until its speed is measured.

Work flow: write code in the cloud on a branch, then pull to the PC and test against real data.

### Decision gates

| After | Question | If the answer is no |
|---|---|---|
| P3 | Does the 5,000-character cut keep the reasoning? | Lower it or add a rule, and re-size the corpus |
| P4 | Can embedding finish in a practical time? | Use Kaggle, or reduce to 2015 onward |
| P5 | Is recall close to the flat index within the budget? | Use more RAM or a smaller corpus. Record the trade-off |
| P7 | Is recall good enough on the lawyer's questions? | Fix retrieval before adding features |
| P9 | Is the wrong-answer rate acceptable at useful coverage? | Do not start the pilot yet |

---

## 7. Pitch rules

- Say what is covered and the date range. Bombay HC data ends in May 2026 and Supreme Court data
  in 2024, so recent judgments are missing until an update process exists.
- "Not found" means not in our data. It never means a case does not exist.
- Do not claim a judgment is still good law. The data has no overruled or treatment labels.
- Show the disclaimer: research aid, not legal advice, verify against the official copy.
- Warn users not to enter confidential client facts while queries go to an outside LLM provider.
- Follow the dataset's CC BY 4.0 attribution requirement. Have the lawyer or a legal adviser
  review the use of court data before any paid offering.

Why quote checking comes first: in *Pooja Ramesh Singh v. J&K Bank* (2026 INSC 668), the Supreme
Court found fake paragraphs attached to real citations and called citing unverified AI
precedents misconduct by the advocate.

---

## 8. Definition of done

- A lawyer used the system on real work, and the pilot report states the results, good or bad.
- Every README number has an interval and comes from a script in this repo.
- Generation metrics are for the deployed model, using an independent judge with human agreement
  reported.
- Quotes are checked against source text, and the system refuses when evidence is weak.
- Hosting costs are measured, and the service has login, rate limits and a no-logging mode.
- `pytest -q` passes on stubs, and no v1 artifact was changed.
