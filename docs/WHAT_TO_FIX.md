# What to Fix in LegalRAG (simple version)

The project is good for a fresher resume. These are the things that would
make it trustworthy. The detailed step-by-step plan for an AI agent is in
`docs/IMPROVEMENT_PLAN.md`.

## 1. Fix wrong or overstated claims (about half a day)

- "51.3% of failures are retrieval failures" is wrong. It is 51.3% of **all questions**.
  Retrieval causes about **84% of the failures**.
- The README says citations are "verified". The code only checks that the chunk ID exists
  in the context. It does not check that the chunk supports the sentence.
- t3.xlarge vs t4.xlarge mix-up. The MIT badge but no LICENSE file. Committed junk
  (`frontend/.vite`). Too much marketing language.

## 2. Make the testing trustworthy (4–6 days). Most important.

- **Problem:** the same AI model wrote the questions, answered them, and graded the answers.
  It is grading its own homework.
- **Problem:** the model you deploy (Qwen on Groq) was never tested. The README numbers
  are for a different model.
- **Problem:** no error bars. With 497 questions, small differences (like the reranker
  dip at Recall@10) may just be luck.
- **Problem:** many questions are vague. "What did the court decide on bail?" matches
  thousands of cases, so "retrieval failure" is partly a question problem.
- **Fix:**
  - Re-test the deployed model.
  - Use a judge from a different AI family.
  - Grade 50 answers yourself and check that the judge agrees with you.
  - Add confidence intervals.
  - Report results separately on clear and vague questions.

## 3. Find the right judgment more often (5–8 days)

- Right now about 1 in 3 questions never gets the right case among the top 50 results.
- **Things to try** (measure each one, keep only what helps):
  - Group chunks by judgment.
  - Use a better reranker (bge-reranker).
  - Let the LLM rewrite the query.
  - Add court and year filters.

## 4. Stop confident wrong answers (2–5 days)

- If the search scores are low, say "I don't have enough evidence" instead of guessing.
- Show real citations (case title, court, date, quoted text), not chunk IDs.
- Check each cited sentence against its chunk.
- Add a "research aid, not legal advice" disclaimer.

## 5. Make the server safe to share (1–4 days)

- Anyone on the internet can call your API and use up your Groq key. Add a rate limit
  and allow only your website (CORS).
- BM25 search takes about 2.6 seconds. A faster library (`bm25s`) can make it much faster.
- Track errors and response times. Log queries so you can learn from them.

## 6. Feedback and presentation (2–3 days)

- Add a thumbs up/down button. Turn the bad answers into new test questions.
- Make the README short and honest, with error bars.
- Record a 1-minute demo video, because the EC2 server is usually off.

## How long will it take?

| Goal | Work days | Calendar (full-time) | Calendar (part-time) |
|------|-----------|----------------------|----------------------|
| Minimum: sections 1, 2, 4 (without sentence checking), 5 (security only), README | 9–12 days | 2–3 weeks | 4–6 weeks |
| Everything | 19–28 days | 5–6 weeks | 10–12 weeks |

**If you only have one week:** do sections 1 and 2, plus the rate limit and CORS.
That alone changes the interview story from "I built a RAG" to "I tested my RAG
properly, found where it fails, and fixed it".
