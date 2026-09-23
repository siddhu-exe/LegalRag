# Repository Guidelines

Contributor guide for **LegalRAG**, a hybrid-retrieval, evidence-grounded QA system over 100,000 Indian High Court judgments.

## Project Structure & Module Organization
- `src/legalrag/` — installable package (import as `legalrag.*`):
  - `preprocessing/` cleaner + chunker, `retrieval/` BM25, dense, RRF fusion, reranker, `generation/` Groq client + prompt builders, `evaluation/` grounding + metrics, `api/` FastAPI app (routes, schemas, config, dependencies).
- `tests/` — unit suite (`test_*.py`) covering preprocessing, retrieval, generation, evaluation, and API.
- `scripts/` — standalone helpers (`download_artifacts.py`, `collect_corpus.py`, `profile_data.py`, `analyze_courts.py`).
- `docs/` — architecture, experiment, journey, reproducibilty notes; `Notebooks/` — exploratory Kaggle notebooks; `data/raw/` — corpus (gitignored artifacts).

## Build, Test, and Development Commands
- `python3 -m venv rag && source rag/bin/activate` — create and activate a virtualenv.
- `pip install -r requirements-dev.txt && pip install -e .` — install dev + package deps.
- `export ENVIRONMENT=local_stub` — required for local runs; `production` is the fail-closed default.
- `pytest tests/ -v` or `python -m unittest discover -s tests -v` — run the suite.
- `uvicorn legalrag.api.main:app --host 0.0.0.0 --port 7860` — serve locally (`/docs`, `/health`, `/ready`).
- `docker build -t legalrag-api . && docker run -p 7860:7860 legalrag-api` — container run.
- `black src tests && isort src tests && flake8 src tests && mypy src` — format, lint, type-check.

## Coding Style & Naming Conventions
- Python 3.10+; 4-space indentation; `black`/`isort` at line length 100; `flake8` and `mypy`.
- `snake_case` functions/variables, `PascalCase` classes, `UPPER_SNAKE_CASE` constants; modules short and lowercase.
- Add module docstrings; never leak raw API errors into answer fields.

## Testing Guidelines
- `pytest` (config in `pyproject.toml`); name files `test_<module>.py` and functions `test_<behavior>`.
- `tests/conftest.py` forces `ENVIRONMENT=local_stub`; tests must use stubs/mocks only — no model downloads, FAISS index, or network calls.
- Prefer deterministic tests; add coverage for new retrieval/generation behavior.

## Commit & Pull Request Guidelines
- Follow Conventional Commits seen in history: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `build:` (e.g. `fix:changed dataset type`).
- Keep commits atomic and scoped to one change; write imperative summaries.
- PRs should describe the change and rationale, link related issues, and include test/benchmark evidence; add screenshots only for UI changes.

## Security & Configuration Tips
- Never commit secrets or `.env`; provide `GROQ_API_KEY`, `HF_REPO_ID`, `HF_TOKEN` via environment only.
- Locked architecture (dataset scale, chunk sizes, retrieval cascade) — discuss before changing.
