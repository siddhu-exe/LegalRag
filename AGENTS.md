# Repository Guidelines

Contributor guide for **LegalRAG**, a hybrid-retrieval, evidence-grounded QA system over 100,000 Indian High Court judgments.

## Project Structure & Module Organization
- `src/legalrag/` — installable package (import as `legalrag.*`):
  - `preprocessing/` (`cleaner.py`, `chunker.py`), `retrieval/` (`bm25.py`, `dense.py`, `fusion.py`, `reranker.py`), `generation/` (`client.py` Groq + `prompts.py`), `evaluation/` (`grounding.py`, `metrics.py`), `api/` (`main.py`, `routes.py`, `schemas.py`, `config.py`, `dependencies.py`, `provisioning.py`).
- `tests/` — 133 pytest tests (`test_*.py`): `test_api.py`, `test_api_endpoints.py`, `test_artifact_contracts.py`, `test_retrieval.py`, `test_retrieval_pipeline_integration.py`, `test_cleaner.py`, `test_chunker.py`, `test_prompts.py`, `test_generation_client.py`, `test_evaluation.py`, `test_download_artifacts.py`.
- `scripts/` — standalone helpers (`download_artifacts.py`, `collect_corpus.py`, `profile_data.py`, `analyze_courts.py`).
- `docs/` — architecture, experiment, journey, reproducibilty notes; `Notebooks/` — exploratory Kaggle notebooks; `artifacts/` — runtime indexes (gitignored; provisioned from the Hugging Face dataset `siddhu23/LegalRag_Dataset`).

## Build, Test, and Development Commands
- `python3 -m venv rag && source rag/bin/activate` — create and activate a virtualenv.
- `pip install -r requirements-dev.txt && pip install -e .` — install dev + package deps.
- `export ENVIRONMENT=local_stub` — required for local runs; `production` is the fail-closed default.
- `pytest -q` — run the suite (133 tests; stubs/mocks only). `pyproject.toml` sets `pythonpath = ["src"]`, so no install is needed; if you invoke Python directly, use `PYTHONPATH=src`.
- `uvicorn legalrag.api.main:app --host 0.0.0.0 --port 7860` — serve locally (`/docs`, `/health`, `/ready`).
- `docker build -t legalrag-api . && docker run -p 7860:7860 legalrag-api` — container run.
- `black src tests && isort src tests && flake8 src tests && mypy src` — format, lint, type-check.
- Real-artifact validation (only when `artifacts/` is populated): `PYTHONPATH=src python -c "from legalrag.retrieval.bm25 import BM25Retriever; r = BM25Retriever.load('artifacts/bm25.pkl'); r.validate(); print(type(r.model).__name__, len(r.chunk_ids))"`.

## Coding Style & Naming Conventions
- Python 3.10+; 4-space indentation; `black`/`isort` at line length 100; `flake8` and `mypy`.
- `snake_case` functions/variables, `PascalCase` classes, `UPPER_SNAKE_CASE` constants; modules short and lowercase.
- Add module docstrings; never leak raw API errors into answer fields.

## Testing Guidelines
- `pytest` (config in `pyproject.toml`); name files `test_<module>.py` and functions `test_<behavior>`.
- `tests/conftest.py` forces `ENVIRONMENT=local_stub`; tests must use stubs/mocks only — no model downloads, FAISS index, or network calls.
- Artifact serialization contracts (BM25 `{"bm25": ..., "chunk_ids": [...]}` and the dense FAISS/chunk-id contract) are covered by `tests/test_artifact_contracts.py`; extend these when changing any producer/consumer format.
- Prefer deterministic tests; add coverage for new retrieval/generation behavior.

## Commit & Pull Request Guidelines
- Follow Conventional Commits seen in history: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `build:` (e.g. `fix:changed dataset type`).
- Keep commits atomic and scoped to one change; write imperative summaries.
- PRs should describe the change and rationale, link related issues, and include test/benchmark evidence; add screenshots only for UI changes.

## Security & Configuration Tips
- Never commit secrets or `.env`; provide `GROQ_API_KEY`, `HF_REPO_ID`, `HF_TOKEN` via environment only.
- `GROQ_MODEL_NAME` must be a model ID currently served by Groq (check `GET https://api.groq.com/openai/v1/models`); a stale ID returns a provider 404 that surfaces as HTTP 502.
- Never commit or bake `artifacts/` into the image; they are large, gitignored, and provisioned at startup.
- Locked architecture (dataset scale, chunk sizes, retrieval cascade) — discuss before changing.
