# LegalRAG

Ask a question about Indian case law and get an answer with citations, drawn from **100,000 High Court judgments** (538,079 chunks).

It combines keyword search (BM25), semantic search (BGE) and a cross-encoder reranker, then asks an LLM to answer using only the retrieved text. The main contribution is the **evaluation harness**: a 497-question gold set that measures retrieval recall and answer quality, including where the system fails.

**[Live demo](https://legalrag-six.vercel.app)** · React on Vercel, FastAPI on AWS EC2. The server is stopped when idle, so the demo may show a standby screen.

## How it works

```text
question
   ├── BM25 top-50 ──────┐
   └── Dense top-50 ─────┴─► RRF fusion (k=60) ─► cross-encoder rerank ─► top-5 chunks
                                                                              │
                         answer + [Chunk ID] citations ◄── Groq LLM (temp 0) ◄┘
```

Each response includes the answer, its citations, the retrieved chunk IDs, and per-stage latency (`retrieve_ms`, `rerank_ms`, `generate_ms`, `total_ms`).

## Results

Judgment-level Recall@k on 497 gold questions:

| Retriever | @1 | @5 | @10 | @50 |
| :--- | :---: | :---: | :---: | :---: |
| BM25 | **0.427** | **0.533** | **0.578** | 0.640 |
| Dense (bge-base-en-v1.5) | 0.197 | 0.274 | 0.334 | 0.447 |
| Hybrid (RRF) | 0.322 | 0.511 | 0.561 | **0.658** |
| Hybrid + cross-encoder | 0.356 | 0.487 | 0.559 | **0.658** |

What we learned:

- **BM25 beats dense search.** Legal queries lean on exact terms such as *Section 138 NI Act*.
- **Hybrid fusion gives the widest net**, with the best recall at 50 candidates.
- **The reranker helps @1 but hurts @5.** The MS-MARCO cross-encoder is not tuned for Indian legal language, so we report this as is.
- **Retrieval is the bottleneck.** In 255 of 497 questions (51.3%) the gold judgment was not in the top-5 context.

Answer quality (LLM judge, 1–4 scale): faithfulness 3.40, answer relevance 3.10, citation correctness 3.25. These scores come from an LLM judge, and the same model family wrote the questions, answered them and judged them, so treat them as indicative. Citations are checked only to confirm the chunk ID was in the context, not that the chunk supports the claim.

## Quickstart

Requires Python 3.10+.

```bash
git clone https://github.com/Siddharth23052005/LegalRAG.git
cd LegalRAG
pip install -r requirements.txt && pip install -e .
```

**Try it in seconds, with stubs and no models or API keys:**

```bash
ENVIRONMENT=local_stub uvicorn legalrag.api.main:app --port 7860
# open http://localhost:7860/docs
```

**Run the real system.** This needs about 10 GB of RAM and about 2.5 GB of artifacts, which are fetched from Hugging Face on startup:

```bash
export ENVIRONMENT=production
export GROQ_API_KEY=your-key
export HF_REPO_ID=siddhu23/LegalRag_Dataset
uvicorn legalrag.api.main:app --port 7860
```

Or use Docker:

```bash
docker build -t legalrag-api .
docker run -p 7860:7860 -e ENVIRONMENT=production -e GROQ_API_KEY=your-key \
  -e HF_REPO_ID=siddhu23/LegalRag_Dataset legalrag-api
```

Then query it:

```bash
curl -X POST localhost:7860/query -H 'Content-Type: application/json' \
  -d '{"question": "When can a court quash proceedings under Section 482 CrPC?"}'
```

`GET /health` reports liveness and `GET /ready` reports whether the models and indexes have loaded. A 502 from `/query` usually means `GROQ_MODEL_NAME` is not a model Groq currently serves. All settings are in `.env.example`.

**Tests** use stubs only (no models, no network): `pip install -r requirements-dev.txt && pytest -q`

## Project layout

```text
src/legalrag/   api/ · preprocessing/ · retrieval/ · generation/ · evaluation/
frontend/       React 19 + Vite + Tailwind UI
tests/          133 stub-based tests
scripts/        artifact download and data tooling
Notebooks/      Kaggle runs that built the indexes and benchmark
docs/           architecture, experiment report, reproducibility
```

## Docs

[Architecture](docs/ARCHITECTURE.md) · [Experiment report](docs/EXPERIMENT_REPORT.md) · [Reproducibility](docs/REPRODUCIBILITY.md) · [Project journey](docs/PROJECT_JOURNEY.md) · [Known issues and fixes](docs/WHAT_TO_FIX.md)

## Limitations

- Each query takes about 7–10 s on CPU, and BM25 over 538k chunks accounts for most of the retrieval time.
- There is no authentication and CORS is open. This is a demo, not a hardened service.
- This is not legal advice.

## Credits

Corpus: [overthelex/indian-court-decisions](https://huggingface.co/datasets/overthelex/indian-court-decisions) · Models: [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5), [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) · LLM: [Groq](https://groq.com)
