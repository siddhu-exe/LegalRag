# LegalRAG Plan (simple version)

The goal has changed. This is no longer only a resume demo: a Bombay High Court lawyer should be
able to test it on real work, in any practice area. The detailed plan is in
[IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md).

## What we found in the data

- Bombay High Court has 1,263,189 records, more than 12x our current corpus. It covers Bombay, Nagpur,
  Aurangabad, Kolhapur and Goa.
- 78.9% of them are short orders under 5,000 characters.
- So we search about 312,000 fuller judgments (Bombay plus the Supreme Court), and keep every record
  available for case lookup.
- The corpus gets bigger while we reduce RAM, so we will probably need an 8 GB server. A 4 GB server
  only works if testing shows it can.
- Bombay data ends in May 2026 and Supreme Court data in 2024. The lawyer must be told this.
- Our old report labelled courts wrongly. Its "Bombay" numbers were really Patna.

Details: [DATA_PROFILE_BOMBAY.md](DATA_PROFILE_BOMBAY.md).

## Cloud work (code, in Claude Code on the web)

1. Code to build the new corpus and the data quality report.
2. Testing code: error bars, paired tests and an independent judge.
3. Low-RAM parts: `bm25s`, smaller FAISS index, SQLite for text, and ONNX instead of PyTorch.
4. Trust features: exact quote check, "not enough evidence" replies, real citations and filters.
5. IPC to BNS search and case lookup.
6. Security: login, rate limit and a no-logging mode.
7. Pilot tools: accept or reject buttons, time saved, and a report.
8. Reranker training code, trade-off tables and refusal charts.

## PC work (on your machine, with the real data)

1. Measure RAM on the current system.
2. Get 20–30 real questions from the lawyer.
3. Build the corpus, and read 100 short orders to check that the 5,000-character cut is safe.
4. Test embedding speed, then create the embeddings on your PC or Kaggle.
5. Compare index options, then choose the server.
6. Build the new test set, and grade 50 answers yourself.
7. Test the system, and read the failures yourself.
8. Train the reranker, and keep it only if it really helps.
9. Set the "not enough evidence" threshold.
10. Load test and deploy.
11. Run the pilot for 2–4 weeks, then write up the results.

## What makes this look like experienced work

- **Real users:** run the pilot and report results, such as how many answers the lawyer accepted and
  how much time it saved.
- **Sound statistics:** use error bars and paired tests, and only claim an improvement when the
  numbers support it.
- **Failure analysis:** break failures down by cause and question type, then show what you fixed.
- **Failed experiments:** keep a log that includes what didn't work. The reranker dip is a good example.
- **Trade-offs:** show quality against cost, RAM and speed, and explain why you chose one option.
- **Data quality:** report on OCR errors, duplicates, missing dates and coverage by bench.
- **Refusal tuning:** show how the "not enough evidence" threshold trades answered questions against
  wrong answers.

Do not claim years of experience you don't have. Let this evidence speak for itself. You should
also do the failure review and the 50 gradings yourself, so you can explain every number in an
interview.

## How long

About 5 weeks of full-time building, plus a 2–4 week pilot. Part-time takes about twice as long.
The biggest unknown is embedding speed, which we measure in week 2.
