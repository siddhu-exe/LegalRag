# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Frontend Scope & Identity

You are operating as a **Senior Frontend Engineer** dedicated to the user interface of **LegalRAG** — an autonomous judicial retrieval and grounded synthesis engine over 100,000 Indian High Court judgments.

All frontend development is scoped to `frontend/` and interfaces with the locked FastAPI backend (`src/legalrag/api/`).

---

## High-Level Architecture & Contracts

### 1. Backend Invariants (`docs/FRONTEND_INTEGRATION.md`)
The backend is a fixed, multi-stage retrieval cascade (BM25 + BGE Dense + RRF + Cross-Encoder + Groq LLM). The pipeline is **not configurable by the client**.

* **Base URL**: Default `http://localhost:7860` (configured via `VITE_API_BASE_URL`).
* **Endpoints**:
  - `POST /query`: Primary endpoint. Body **must contain only** `{ "question": string }` (3–4000 chars). Extra fields trigger HTTP `422`.
  - `GET /health`: Fast liveness check (`{ "status": "ok", "environment": string }`). Does not load models.
  - `GET /ready`: Pipeline readiness check (`200` ready, `503` starting/unusable).
* **Missing Endpoints (Do NOT stub or assume)**:
  - No `GET /chunks/{id}` or `GET /documents/{id}` (full document text cannot be fetched).
  - No streaming endpoints (no SSE / WebSocket / `/query/stream`).
  - No authentication, chat history, or multi-turn endpoints.
* **Latency Profile**:
  - Synchronous request: typical latency is ~7–10s on the real 538k chunk corpus (cold start up to ~30s).
  - Recommended client timeout: **120 seconds**. Never use a 5–10s timeout.
  - Implement single in-flight query locking (prevent duplicate submissions).

### 2. Data Shapes & Type Definitions

```ts
export interface QueryRequest {
  question: string; // 3 - 4000 characters
}

export interface Citation {
  chunk_id: string; // Opaque string, unique, React key (e.g., "WBCHCJ0003822020_1_2020-01-28_0")
  cnr: string | null;
  court_code: string | null;
  decision_date: string | null;
  title: string | null;
}

export interface QueryResponse {
  answer: string; // Grounded prose containing [Chunk ID: ...] citations
  citations: Citation[]; // Verified cited passages (may be empty)
  retrieved_chunk_ids: string[]; // Final top-5 chunks considered
  retrieve_ms: number; // BM25 + Dense + RRF duration
  rerank_ms: number; // Cross-encoder rerank duration
  generate_ms: number; // Groq LLM generation duration
  total_ms: number; // End-to-end server duration
  status: "ok" | "generation_error" | "retrieval_error";
}
```

* **Null Safety**: In `Citation`, only `chunk_id` is guaranteed non-null. Always null-guard `cnr`, `court_code`, `decision_date`, and `title`.
* **Error Mapping**:
  - `422`: Validation error (question < 3 or > 4000 chars, extra keys).
  - `500`: Retrieval or reranker failure ("Document retrieval failed.").
  - `502`: Generation failure / Groq provider error ("Answer generation failed.").
  - `503`: Pipeline warming / not ready ("Service is not ready.").

---

## Design System & Aesthetics (`frontend/stitch_design/`)

The design embodies **Editorial Minimalism** fused with **Technical Precision** (Tailwind CSS + Custom Design Tokens):

### 1. Color Palette
* **Canvas Foundation**: `#0B0E14` (surface-container-lowest), `#10131a` (surface), `#191c22` (surface-container-low), `#1d2026` (surface-container), `#272a31` (surface-container-high).
* **Primary Accent (Legal Brass/Amber)**: `#D4AF37` / `#f2ca50` — Evidentiary verification, citation badges, verified seals, focus states.
* **Secondary Accent (Slate Steel)**: `#7B96B2` / `#aec9e7` — Structural metadata, court tags, vector dimensions.
* **Tertiary Accent (Vector Mint)**: `#2DD4BF` / `#48e5d0` — Precision telemetry, latency badges (`*_ms`), high cosine similarity tags.
* **Hairline Borders**: `rgba(255, 255, 255, 0.08)` for 1px crisp structural framing.

### 2. Typography Hierarchy
* **Editorial Serif (`Newsreader`)**: Case titles, landmark headers, thesis summaries, statutory pull quotes (`font-headline-lg`, `font-headline-md`, `font-citation-quote`). Italic reserved for legal case citations.
* **System UI (`Inter`)**: Clean body prose, query inputs, navigation, analysis tabs (`font-body-lg`, `font-body-md`, `font-label-ui`).
* **Telemetry & Metadata (`JetBrains Mono`)**: Monospaced labels, chunk IDs, CNR numbers, query latencies, token counters (`font-label-mono`, `font-code-sm`).

### 3. Surface & Elevation
* **Discipline**: Crisp 1px hairline borders (`border-white/10` or `border-surface-container-highest/60`).
* **Corner Radius**: Sharp and minimal (2px-4px micro, 4px standard UI inputs/buttons, 8px structural containers). Over-rounded pills are strictly prohibited.
* **Evidentiary Glow**: Subtle amber radiance on verified cards (`shadow-[0_0_20px_-4px_rgba(212,175,55,0.15)]`).

---

## Frontend Architecture & Structure

When scaffolding or maintaining the frontend workspace (`frontend/`):

```
frontend/
├── stitch_design/            # UI/UX Reference specs & exported mockups
│   ├── legalrag_showcase/    # DESIGN.md token specifications
│   ├── legalrag_query_console/ # Query console HTML & screen mockups
│   └── legalrag_grounded_synthesis_citations/ # Grounded synthesis HTML & screen mockups
├── src/
│   ├── api/
│   │   ├── client.ts         # queryLegalRAG, fetchHealth, fetchReady
│   │   └── types.ts          # QueryRequest, QueryResponse, Citation
│   ├── components/
│   │   ├── layout/           # Header, Topbar, StatusPill
│   │   ├── query/            # QueryConsole, QueryTextarea, BenchmarkVectors
│   │   ├── synthesis/        # GroundedSynthesisView, CitationCard, GroundingHighlighter
│   │   └── telemetry/        # LatencyBreakdown, PipelineStageVisualizer
│   ├── hooks/
│   │   └── useLegalRag.ts    # State management for query lifecycle & errors
│   ├── App.tsx
│   └── main.tsx
├── tailwind.config.js        # Token mappings from stitch_design/DESIGN.md
├── package.json
└── vite.config.ts
```

---

## Common Development Commands (Vite + React)

* **Install dependencies**: `npm install` (within `frontend/`)
* **Start Dev Server**: `npm run dev` (runs on `http://localhost:5173` with proxy to `7860`)
* **Build Production Bundle**: `npm run build`
* **Lint / Format**: `npm run lint` / `npx prettier --write .`
* **Typecheck**: `npx tsc --noEmit`
