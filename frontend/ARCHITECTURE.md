# LegalRAG Frontend Architecture

## 1. Overview & Positioning
A streamlined, production-grade **React + TypeScript + Vite + Tailwind CSS** frontend for **LegalRAG** — an autonomous judicial retrieval and grounded synthesis engine over 100,000 Indian High Court judgments.

Built as a clean AI Engineering portfolio showcase with a 2-page flow (**Ask** and **Results**), interactive citation grounding, and compact pipeline telemetry.

---

## 2. Environment Configuration

The frontend interfaces with the backend API endpoint via environment configuration.

### `.env.example`
```bash
# Backend API URL (Production URL placeholder or FastAPI endpoint)
VITE_API_URL=https://your-deployed-backend-url.com

# Client Request Timeout (in milliseconds)
VITE_API_TIMEOUT_MS=120000
```

### URL Resolution Logic (`src/config.ts`)
- `API_BASE_URL`: `import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL || ""`
- `API_TIMEOUT_MS`: `Number(import.meta.env.VITE_API_TIMEOUT_MS) || 120000`
- `getApiBaseUrl()` returns the configured API base URL.

---

## 3. Directory & File Structure

```
frontend/
├── .env.development            # Development environment configuration
├── .env.production             # Production environment configuration
├── .env.example                # Configuration template
├── index.html                  # Root HTML with Google Fonts preloads
├── package.json                # React 19, Lucide React, Tailwind
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
├── ARCHITECTURE.md             # This architecture specification
└── src/
    ├── main.tsx                # Mount React root
    ├── App.tsx                 # State store + 2-page router ('ask' ⟷ 'results')
    ├── index.css               # Global reset, typography, Tailwind layers
    │
    ├── config.ts               # Environment configuration helper
    ├── types.ts                # Strict TypeScript contracts (QueryRequest, QueryResponse, Citation)
    │
    ├── api.ts                  # Fetch client (120s timeout, abort controller, error mapping)
    │
    └── components/
        ├── Header.tsx          # Brand header, corpus stats pill, pipeline architecture tag
        ├── LatencyPill.tsx     # Compact telemetry strip (Retrieve, Rerank, Generate, Total ms)
        ├── AskView.tsx         # Page 1: Query textarea, char counter (3-4000), sample questions, submit
        ├── ResultsView.tsx     # Page 2: Query recap, dual-pane answer + citation cards, latency bar
        ├── GroundedAnswer.tsx  # Markdown/prose renderer with interactive [Chunk ID: ...] badge tags
        └── CitationCard.tsx    # Source judgment card (Title, CNR, Court, Date, Chunk ID)
```

---

## 4. API Invariants & Error Handling (`src/api.ts`)

### Contract (`docs/FRONTEND_INTEGRATION.md`)
- **Endpoint:** `POST /query`
- **Request Body:** Must contain ONLY `{ "question": string }` (3–4000 characters). Extra keys return HTTP `422`.
- **Response Format:**
  ```ts
  export interface Citation {
    chunk_id: string; // Guaranteed non-null unique key
    cnr: string | null;
    court_code: string | null;
    decision_date: string | null;
    title: string | null;
  }

  export interface QueryResponse {
    answer: string; // Grounded prose with [Chunk ID: ...] tags
    citations: Citation[];
    retrieved_chunk_ids: string[];
    retrieve_ms: number;
    rerank_ms: number;
    generate_ms: number;
    total_ms: number;
    status: "ok" | "generation_error" | "retrieval_error";
  }
  ```

### Client Execution Rules
- Enforce a **120-second client timeout** via `AbortController` (cold starts and BM25 search on 538k chunks take time).
- Prevent duplicate/concurrent submissions with single in-flight query locking.
- Map HTTP error codes to user-friendly notifications:
  - `422`: Validation error ("Question must be between 3 and 4,000 characters.")
  - `500`: Retrieval pipeline error ("Document retrieval failed.")
  - `502`: Generation failure ("Answer generation failed.")
  - `503`: Backend warming up ("Service is warming up models. Please retry in ~15s.")
  - `AbortError`: "Request was cancelled or timed out after 120s."

---

## 5. View Flow & UX Behavior

### Page 1: `AskView`
- Large focused input card on the `#0B0E14` canvas.
- Real-time character count indicator with 3–4,000 bounds.
- Curated benchmark question cards covering criminal procedure, constitutional writs, bail jurisprudence, commercial law, and arbitration.
- Single primary CTA ("Synthesize Precedents") with keyboard submit support (`Ctrl+Enter` / `Cmd+Enter`).

### Loading State
- Inline progress indicator showing elapsed seconds with precision (`0.0s`).
- Multi-step status feedback ("Hybrid Retrieval Cascade", "Cross-Encoder Precision Rerank", "Grounded LLM Synthesis").
- Active "Cancel" button aborting the in-flight fetch.

### Page 2: `ResultsView`
- **Navigation:** "← New Inquiry" button returns cleanly to `AskView`.
- **Query Banner:** Shows the verbatim question queried.
- **Dual-Pane Layout:**
  - **Left / Main Pane:** `GroundedAnswer` parsing and converting `[Chunk ID: ...]` into amber clickable badge tags.
  - **Right Pane:** Scrollable list of `CitationCard`s with null-safe fields. Hovering/clicking a citation tag in the text highlights and scrolls to the card.
- **Bottom Telemetry:** Compact `LatencyPill` showing `retrieve_ms`, `rerank_ms`, `generate_ms`, and `total_ms`.

---

## 6. Design System & Tokens

Derived from `frontend/stitch_design/legalrag_showcase/DESIGN.md`:
- **Canvas:** `#0B0E14` (Deep obsidian base), `#191c22` (Card surface), `#272a31` (Elevated surface).
- **Hairline Borders:** `rgba(255, 255, 255, 0.08)` for 1px crisp structural frames.
- **Primary Accent (Legal Brass/Amber):** `#D4AF37` / `#f2ca50` for verified citation badges and active states.
- **Secondary Accent (Slate Steel):** `#7B96B2` / `#aec9e7` for court metadata and structure.
- **Tertiary Accent (Vector Mint):** `#2DD4BF` for latency badges (`*_ms`).
- **Typography:**
  - Serif: `Newsreader` (Case titles, editorial headings)
  - UI Sans: `Inter` (Body prose, buttons, controls)
  - Monospace: `JetBrains Mono` (Chunk IDs, CNR numbers, latency metrics)
