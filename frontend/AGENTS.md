# Repository Guidelines

Contributor guide for the **LegalRAG frontend** — a React 19 + TypeScript + Vite + Tailwind CSS interface for querying and citing Indian High Court judgments. Development is scoped to `frontend/` and consumes the locked FastAPI backend.

## Project Structure & Module Organization
- `src/` — application source:
  - `main.tsx` (React root), `App.tsx` (state store + `ask` ⟷ `results` router), `index.css` (Tailwind layers/typography).
  - `api.ts` (fetch client, 120s timeout, abort + error mapping), `config.ts` (env helpers), `types.ts` (strict API contracts).
  - `components/` — `Header.tsx`, `AskView.tsx`, `ResultsView.tsx`, `GroundedAnswer.tsx`, `CitationCard.tsx`, `LatencyPill.tsx`, `BackendOfflineView.tsx`.
- `docs/` — `ARCHITECTURE.md`, `DEPLOYMENT.md`, `USER_GUIDE.md`.
- `stitch_design/` — design tokens and exported mockups; `ARCHITECTURE.md`, `CLAUDE.md`, `README.md` at the root.
- `dist/` — build output (gitignored).

## Build, Test, and Development Commands
- `npm install` — install dependencies.
- `npm run dev` — start the Vite dev server at `http://localhost:5173`.
- `npm run build` — typecheck via `tsc` then produce the production bundle.
- `npm run preview` — serve the built bundle locally.
- `npx tsc --noEmit` — typecheck only.
- `npx prettier --write .` — format (no ESLint config is committed).

## Coding Style & Naming Conventions
- TypeScript strict mode is on; avoid `any` and unused locals/params.
- `PascalCase` for components and files (`CitationCard.tsx`); `camelCase` for functions, hooks, and variables; `UPPER_SNAKE_CASE` for constants.
- Prefer named exports for components; keep API types centralized in `src/types.ts`.
- Style with Tailwind utility classes and the palette tokens in `tailwind.config.js` (`canvas`, `brass`, `slateSteel`, `vectorMint`); avoid raw hex in JSX.

## Testing Guidelines
No test runner is configured yet. Validate changes with `npx tsc --noEmit` and `npm run build`, and smoke-test flows (`ask` → `results`) via `npm run dev`. If adding tests, use Vitest + React Testing Library, naming files `*.test.tsx`.

## Commit & Pull Request Guidelines
- Use Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:` (scoped examples like `feat(frontend):` appear in history).
- Keep commits atomic and imperative; one logical change each.
- PRs should state the change and rationale, link issues, include build/typecheck evidence, and add screenshots for UI changes.

## Security & Configuration Tips
- Configure the backend via `.env` only: `VITE_API_URL` (no trailing slash) and `VITE_API_TIMEOUT_MS` (default `120000`). Never commit `.env`.
- Only `VITE_`-prefixed values reach the client bundle — never expose secrets like `GROQ_API_KEY` or `HF_TOKEN`.
- Null-guard `citation.cnr`, `court_code`, `decision_date`, and `title`; only `chunk_id` is guaranteed.
