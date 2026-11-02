# Experiment Log

Every experiment and data check goes here, including the ones that did not work. Add new entries at
the top of the relevant section.

Results: **Adopted**, **Rejected**, **Mixed** or **Pending**.

## Entry template

```text
### Short title (date)
Question:
Change tested:
Data and metric:
Result (with confidence interval):
Decision:
What I learned:
```

## v2 experiments

### Bombay HC corpus profile (2 November 2026)
- **Question:** Is all of the Bombay HC data practical to search?
- **Data and metric:** All 1,263,189 records, counted with `scripts/profile_bombay_hc.py`.
- **Result:** 78.9% of records have under 5,000 characters, and these hold 32.6% of the text.
- **Decision:** Pending. Index reasoned judgments and keep all records in lookup, after plan task P3
  tests the cut by reading a sample.
- **What I learned:** My first assumption, that focusing on one court would shrink the corpus, was
  wrong. The full court is more than 12x the size of v1 by judgment count.

### Date field check (2 November 2026)
- **Question:** Is `decision_date` reliable?
- **Data and metric:** Compared `decision_date` with the date in the CNR across all Bombay records.
- **Result:** They differ on 24,933 records (2.0%). The first five rows of a filtered query all
  differed, which had suggested a much bigger problem.
- **Decision:** Pending. Choose a date rule in plan task P3.
- **What I learned:** The first rows of a query are not a sample.

## v1 experiments and incidents

These results are from the historical 100k system and were not re-tested with confidence intervals.

### Cross-encoder reranker on top of hybrid search
- **Question:** Does `ms-marco-MiniLM-L-6-v2` improve hybrid retrieval?
- **Data and metric:** Judgment-level recall on 497 gold questions.
- **Result:** Recall@1 rose from 0.322 to 0.356, but Recall@5 fell from 0.511 to 0.487, and Recall@10
  fell from 0.561 to 0.559.
- **Decision:** Mixed. It stayed in the deployed system, even though the context step uses the top 5,
  which is where recall fell. Re-test with paired statistics, and try a fine-tuned reranker.
- **What I learned:** A model trained on web search does not automatically suit Indian legal text. A
  better top result can come with a worse top 5.

### Dense search alone
- **Question:** Can semantic search replace keyword search?
- **Result:** Dense Recall@5 was 0.274, against 0.533 for BM25.
- **Decision:** Rejected as a standalone retriever. Kept inside hybrid search.
- **What I learned:** Legal questions depend on exact terms and section numbers. The benchmark may also
  favour BM25, because its questions come from the judgment text, so this needs a word-overlap check.

### Hybrid RRF fusion
- **Question:** Does combining BM25 and dense search help?
- **Result:** Recall@50 rose to 0.658, against 0.640 for BM25. Recall@5 fell to 0.511, against 0.533.
- **Decision:** Mixed. Adopted for its wider candidate pool, but BM25 alone did better in the top 5.
- **What I learned:** Report recall at the cut-off the system actually uses.

### Self-graded generation evaluation
- **Question:** Are the answers good?
- **Result:** Faithfulness was 3.40 out of 4, but the same model family wrote the questions, wrote the
  answers and judged them. The deployed Groq model was never evaluated.
- **Decision:** Rejected as evidence of quality. Replace with an independent judge and human labels.
- **What I learned:** An evaluation needs independence, not just more metrics.

### Failure statistic wording
- **Mistake:** Docs said 51.3% of failures came from retrieval.
- **Correction:** 255 of 497 questions (51.3% of all questions) did not have the right judgment in the
  top 5, which is about 84% of the 303 failures.
- **What I learned:** State the denominator.

### Court code labels
- **Mistake:** The v1 report labelled `10_8` as Bombay and made similar errors for other courts.
- **Correction:** CNR prefixes show `10_8` is Patna. See
  [DATA_PROFILE_BOMBAY.md](DATA_PROFILE_BOMBAY.md).
- **What I learned:** Check labels against an independent field in the data.

### BM25 artifact key mismatch (production incident)
- **Mistake:** The loader read `state["model"]`, but the production pickle used `"bm25"`, so the model was
  `None` and every query failed.
- **Fix:** The loader now accepts the known formats and fails closed, with regression tests.
- **What I learned:** Test against the format production really uses.

### Missing BGE query instruction (production incident)
- **Mistake:** Serving code left out the query instruction used in the experiment.
- **Fix:** `DenseRetriever.search()` applies it to queries only.
- **What I learned:** Record every setting needed to reproduce an index, and check it at serving time.

### 80k corpus run
- **Decision:** Rejected as a result. It was early scaffolding, replaced by the 100k run.
