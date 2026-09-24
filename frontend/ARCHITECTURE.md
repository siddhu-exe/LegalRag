# LegalRAG Frontend Architecture

## 1. Overview & Positioning
A streamlined, production-grade **React + TypeScript + Vite + Tailwind CSS** frontend for **LegalRAG** — an autonomous judicial retrieval and grounded synthesis engine over 100,000 Indian High Court judgments.

Built as a clean AI Engineering portfolio showcase with a 2-page flow (**Ask** and **Results**), interactive citation grounding, compact pipeline telemetry, and a zero-reload **Local ⟷ Deployed backend switcher**.

---

## 2. Environment Configuration

The frontend dynamically switches between a local backend and a deployed cloud backend via runtime selection persisted in `localStorage`.

### `.env.example`
```bash
# Default active backend: "local" or "deployed"
VITE_DEFAULT_BACKEND=local

# Local FastAPI instance
VITE_API_LOCAL=http://localhost:7860

# Deployed Backend URL (Placeholder - set when production URL is available)
VITE_API_DEPLOYED=https://your-deployed-backend-url.com

# Client Request Timeout (in milliseconds)
VITE_API_TIMEOUT_MS=120000
```

### URL Resolution Logic (`src/config.ts`)
- `LOCAL_URL`: `import.meta.env.VITE_API_LOCAL || "http://localhost:7860"`
- `DEPLOYED_URL`: `import.meta.env.VITE_API_DEPLOYED || ""`
- `getActiveBaseUrl()`:
  - If `DEPLOYED_URL` is empty or placeholder, gracefully alert or disable deployed mode.
  - Resolves target based on user toggle (`localStorage.getItem('legalrag_backend_target') || import.meta.env.VITE_DEFAULT_BACKEND || 'local'`).

---

## 3. Directory & File Structure

```
frontend/
├── .env.development            # Local development environment
├── .env.production             # Production environment defaults
├── .env.example                # Template with placeholder for deployed URL
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
    ├── config.ts               # Env variables & runtime backend switcher
    ├── types.ts                # Strict TypeScript contracts (QueryRequest, QueryResponse, Citation)
    │
    ├── api.ts                  # Fetch client (120s timeout, abort controller, error mapping)
    │
    └── components/
        ├── Header.tsx          # Brand header, corpus stats pill, backend switcher
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
- 4 curated benchmark question cards:
  1. *Section 482 CrPC quashing of FIR in matrimonial disputes after amicable settlement*
  2. *Maintainability of writ petition under Article 226 when statutory alternative remedy exists*
  3. *Conditions and guidelines for grant of anticipatory bail in non-bailable offences*
  4. *Presumption of legal liability under Section 139 Negotiable Instruments Act in cheque dishonour cases*
- Single primary CTA ("Analyze Jurisprudence") with keyboard submit support (`Enter` / `Ctrl+Enter`).

### Loading State
- Inline progress indicator showing elapsed seconds.
- Multi-step status feedback ("Retrieving relevant High Court passages...", "Reranking candidates...", "Synthesizing grounded answer...").
- Active "Cancel Query" button aborting the in-flight fetch.

### Page 2: `ResultsView`
- **Navigation:** "← New Search" button returns cleanly to `AskView`.
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
