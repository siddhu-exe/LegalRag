# LegalRAG Web Interface

A production-grade, editorial legal search and precedent synthesis interface built with **React 19, TypeScript, Vite, and Tailwind CSS**.

The interface connects directly to the **LegalRAG** backend to perform hybrid retrieval (BM25 + BGE Dense + Cross-Encoder Reranking) and evidence-grounded judicial synthesis over **100,000 Indian High Court judgments** (538,079 chunks).

---

## 🏛️ System Overview

The frontend is structured around an automatic **AWS EC2 Readiness Gatekeeper** and a distraction-free **2-Page Architecture**:

```
                      ┌──────────────────────┐
                      │   App.tsx (Mount)    │
                      └──────────┬───────────┘
                                 │
                     [Probe GET /ready (AWS)]
                                 │
                 ┌───────────────┴───────────────┐
                 │                               │
       (If Ready: HTTP 200)             (If Offline / Paused)
                 │                               │
        ┌────────┴────────┐             ┌────────▼──────────────┐
        ▼                 ▼             │ BackendOfflineView    │
┌──────────────┐   ┌──────────────┐     │ (Cost-pause notice &  │
│   AskView    │   │ ResultsView  │     │  LinkedIn activation) │
│  (Page 1)    │   │  (Page 2)    │     └───────────────────────┘
└──────────────┘   └──────────────┘
```

1. **Readiness Gatekeeper (`BackendOfflineView.tsx`)**:
   - Automatically probes `GET /ready` on the backend cluster.
   - If the AWS EC2 (`t4.xlarge`) container instance is paused to reduce continuous cloud hosting costs for the 538k corpus index, the app presents an informative standby screen with technical infrastructure details.
   - Provides a direct **LinkedIn Contact CTA** to notify the admin to spin up the container instance on demand, alongside an instant **Re-check Status** button.

2. **Page 1: Ask View (`AskView.tsx`)**:
   - Editorial Hero & Corpus Status Badge (538k Chunks Active).
   - Judicial Inquiry Contextualizer with 3–4,000 character bounds & shortcut (`Cmd/Ctrl+Enter`).
   - Live Multi-Stage Visualizer (Hybrid Retrieval → Neural Rerank → Grounded LLM) with elapsed timer and cancellation.
   - 6 Curated Benchmark Test Vectors from the 497-question gold set.

3. **Page 2: Results View (`ResultsView.tsx`)**:
   - Inquiry Recap & Verified Precedent Counter.
   - Left Pane: Clean Judicial Precedent Synthesis with distraction-free inline citation badges (`[1]`, `[1 · CAL]`).
   - Right Pane: Interactive Authority Cards (CNR, Court, Date, Chunk ID with copy actions).
   - Base Strip: Segmented Pipeline Execution Telemetry (`retrieve_ms`, `rerank_ms`, `generate_ms`, `total_ms`).

---

## 🚀 Quickstart

### Prerequisites
- **Node.js**: v18.0.0 or later (Node 20+ recommended)
- **npm** / **yarn** / **pnpm**

### 1. Installation
```bash
cd frontend
npm install
```

### 2. Environment Configuration
Copy the example environment template:
```bash
cp .env.example .env.development
```

Configure your environment variables in `.env.development` (or `.env.production`):
```bash
# Backend API Base URL
VITE_API_URL=http://localhost:7860

# Request Timeout in Milliseconds (120 seconds for cold starts / 538k corpus search)
VITE_API_TIMEOUT_MS=120000
```

> **Note**: The client supports both `VITE_API_URL` and `VITE_API_BASE_URL`.

### 3. Running Locally
```bash
npm run dev
```
The application will launch on `http://localhost:5173`.

### 4. Building for Production
```bash
npm run build
```
Generates a zero-error, minified bundle in `frontend/dist/`.

### 5. Preview Production Build
```bash
npm run preview
```

---

## 🎨 Design System & Aesthetic Principles

Derived from `stitch_design/legalrag_showcase/DESIGN.md`, the design embodies **Editorial Minimalism** fused with **Technical Precision**:

| Token | Hex / Value | Semantic Role |
| :--- | :--- | :--- |
| `canvas-base` | `#0B0E14` | Deep obsidian backdrop |
| `canvas-surface` | `#10131a` | Base surface layer |
| `canvas-surfaceLow` | `#191c22` | Primary inquiry and reading cards |
| `canvas-surfaceHigh` | `#272a31` | Active, hover, and elevated surfaces |
| `brass` (Legal Brass) | `#D4AF37` / `#f2ca50` | Evidentiary verification, citation badges, primary CTA |
| `slateSteel` | `#7B96B2` / `#aec9e7` | Structural metadata, court tags, CNR numbers |
| `vectorMint` | `#2DD4BF` / `#48e5d0` | Precision latency telemetry (`*_ms`), high-confidence status |

### Typography
- **Editorial Serif (`Newsreader`)**: Case titles, judicial headers, pull quotes.
- **UI Body Sans (`Inter`)**: Query textarea, grounded synthesis prose, button controls.
- **Monospace (`JetBrains Mono`)**: Chunk IDs, CNR identifiers, latency badges, token metrics.

---

## 🔌 API Integration & Contract Invariants

The frontend communicates with the FastAPI endpoint via a single strict contract:

### Endpoint: `POST /query`
- **Request Body**: Must contain **only** `{ "question": string }` (3–4,000 chars).
- **Client Timeout**: 120,000 ms (120s) with single in-flight query locking and user-abort capability.

### Response Structure:
```typescript
export interface Citation {
  chunk_id: string;          // Guaranteed non-null unique key (e.g., "WBCHCJ0003822020_1_2020-01-28_0")
  cnr: string | null;        // Case Number Record
  court_code: string | null; // e.g., "BOM", "DEL", "CAL"
  decision_date: string | null;
  title: string | null;
}

export interface QueryResponse {
  answer: string;              // Grounded legal prose with [Chunk ID: ...] citations
  citations: Citation[];       // Verified cited authorities (may be empty)
  retrieved_chunk_ids: string[]; // Final top-5 chunks considered
  retrieve_ms: number;         // BM25 + BGE Dense + RRF latency
  rerank_ms: number;           // Cross-Encoder rerank latency
  generate_ms: number;         // Groq LLM generation latency
  total_ms: number;            // Server end-to-end duration
  status: "ok" | "generation_error" | "retrieval_error";
}
```

### Distraction-Free Citation Grounding
The `GroundedAnswer` component automatically parses raw `[Chunk ID: ...]` strings from the LLM response and transforms them into non-intrusive badges (`[1]`, `[1 · CAL]`):
- **Hover**: Displays a tooltip with the case title and High Court jurisdiction.
- **Click**: Highlights the badge and smoothly scrolls the corresponding `CitationCard` in the right pane into view.
- **Reverse Interaction**: Clicking an authority card highlights its badge in the text.

---

## 📁 Project Directory Structure

```
frontend/
├── dist/                     # Production build artifacts (generated on build)
├── docs/                     # Detailed developer & deployment guides
│   ├── ARCHITECTURE.md       # Deep dive into frontend architecture & state model
│   ├── DEPLOYMENT.md         # Deployment runbook (Vercel, Netlify, Cloudflare, Docker)
│   └── USER_GUIDE.md         # End-user & maintainer guide
├── src/
│   ├── components/
│   │   ├── Header.tsx        # Brand header, corpus badges, dynamic AWS status indicator
│   │   ├── BackendOfflineView.tsx # AWS EC2 standby screen, cost notice & LinkedIn activation CTA
│   │   ├── AskView.tsx       # Page 1: Query input, char counter, benchmark vectors, progress bar
│   │   ├── ResultsView.tsx   # Page 2: Dual-pane synthesis, citations sidebar, telemetry strip
│   │   ├── GroundedAnswer.tsx# Prose renderer with regex inline citation parsing
│   │   ├── CitationCard.tsx  # Authority card with copy CNR/Chunk ID actions
│   │   └── LatencyPill.tsx   # Segmented execution latency bar (Retrieve, Rerank, Groq)
│   ├── api.ts                # API client with 120s timeout, AbortController, error mapping
│   ├── config.ts             # Environment variable resolver
│   ├── types.ts              # Strict TypeScript interfaces
│   ├── App.tsx               # Root component & 2-page view state router
│   ├── main.tsx              # React DOM root mounting
│   └── index.css             # Tailwind base layers, fonts & global styles
├── stitch_design/            # UI/UX design tokens and exported HTML mockups
├── .env.example              # Environment variables template
├── .env.development          # Development configuration
├── .env.production           # Production configuration
├── index.html                # Main HTML with font preloads
├── package.json              # React 19, Lucide React, Tailwind
├── tailwind.config.js        # Design token extensions
├── tsconfig.json             # TypeScript compiler configuration
└── vite.config.ts            # Vite bundler configuration
```

---

## 🚢 Deployment Options

The frontend is a static Single Page Application (SPA) and can be hosted on any modern static hosting platform:

| Provider | Build Command | Output Directory | Environment Variable |
| :--- | :--- | :--- | :--- |
| **Vercel** | `npm run build` | `dist` | `VITE_API_URL` |
| **Netlify** | `npm run build` | `dist` | `VITE_API_URL` |
| **Cloudflare Pages** | `npm run build` | `dist` | `VITE_API_URL` |
| **AWS S3 + CloudFront** | `npm run build` | `dist` | Set in CI/CD pipeline |
| **Docker (Nginx)** | Multi-stage build | `/usr/share/nginx/html` | Baked via build-arg |

See the **[Deployment Runbook](docs/DEPLOYMENT.md)** for step-by-step setup guides, Nginx configs, and pre-deployment checklists.

---

## 🧪 Testing & Validation

```bash
# Typecheck with TypeScript compiler
npx tsc --noEmit

# Production build validation
npm run build
```

---

## 📚 Related Documentation

- **[Deployment Runbook](docs/DEPLOYMENT.md)** — Step-by-step guides for Vercel, Netlify, Docker, and AWS.
- **[Frontend Architecture](docs/ARCHITECTURE.md)** — In-depth component, state, and token architecture.
- **[User & Developer Guide](docs/USER_GUIDE.md)** — Testing queries, citation mechanics, and extension points.
- **[Backend Integration Specification](../docs/FRONTEND_INTEGRATION.md)** — Backend API contract & invariants.
- **[Main Project Readme](../README.md)** — Retrieval benchmarks, 100k corpus, and research findings.
