# LegalRAG Frontend — Architecture Specification

A technical deep dive into the design decisions, component hierarchy, state management, and design system of the **LegalRAG** web application.

---

## 1. System Positioning & Core Principles

The LegalRAG frontend is built as a flagship **AI Engineering Portfolio UI** designed to present complex multi-stage retrieval and LLM reasoning results with clarity, restraint, and academic rigor.

### Key Architectural Tenets:
1. **Editorial Restraint**: Prioritize reading clarity and legal legibility over aggressive SaaS animations or distracting UI elements.
2. **Strict Invariant Compliance**: Strictly adhere to the backend contracts defined in `docs/FRONTEND_INTEGRATION.md` (no fake endpoints, no stubbed full-text routes, no unsupported payload keys).
3. **Evidence-First Interactivity**: Make every assertion verifiable by linking inline synthesis citations directly to source judgment cards.
4. **Transparent Telemetry**: Expose exact pipeline latencies (Retrieval vs. Reranking vs. Generation) without cluttering the main reading surface.

---

## 2. Component Hierarchy & Data Flow

```
                                 ┌───────────────┐
                                 │   main.tsx    │
                                 └───────┬───────┘
                                         │
                                 ┌───────▼───────┐
                                 │    App.tsx    │
                         (Manages view & readiness state)
                                         │
             ┌───────────────────────────┼───────────────────────────┐
             │                           │                           │
     ┌───────▼───────┐           ┌───────▼───────┐           ┌───────▼───────────────┐
     │  Header.tsx   │           │  AskView.tsx  │           │BackendOfflineView.tsx │
     │(Corpus badges)│           │   (Page 1)    │           │ (AWS Standby / Status)│
     └───────────────┘           └───────┬───────┘           └───────────────────────┘
                                         │
                                 ┌───────▼───────┐
                                 │ResultsView.tsx│
                                 │   (Page 2)    │
                                 └───────┬───────┘
                                         │
                         ┌───────────────┴───────────────┐
                         │                               │
                 ┌───────▼───────────┐           ┌───────▼───────────┐
                 │GroundedAnswer.tsx │           │ CitationCard.tsx  │
                 │(Parsed badge tags)│           │(Authority cards)  │
                 └───────────────────┘           └───────────────────┘
```

---

## 3. State Model (`src/App.tsx`)

The application avoids heavy state-management libraries (Redux/Zustand) in favor of clean React state:

| State Variable | Type | Purpose |
| :--- | :--- | :--- |
| `backendStatus` | `'checking' \| 'ready' \| 'offline'` | Tracks AWS EC2 container readiness via `GET /ready`. |
| `view` | `'ask' \| 'results'` | Controls active page screen when backend is ready. |
| `question` | `string` | The active legal inquiry text. |
| `result` | `QueryResponse \| null` | The validated response object returned by `POST /query`. |
| `isLoading` | `boolean` | Indicates active backend query execution. |
| `error` | `string \| null` | User-facing error message mapped from HTTP status codes. |
| `abortController` | `AbortController \| null` | Allows immediate user cancellation of in-flight requests. |

---

## 4. API Client & Error Shielding (`src/api.ts`)

### Contract Details:
- **Endpoint**: `POST /query`
- **Payload**: `{ "question": string }` (3–4,000 characters). Extra keys strictly forbidden by FastAPI Pydantic schema.
- **Client Timeout**: 120 seconds (`AbortController` managed).

### Error Code Mapping:
- **`422 Unprocessable Entity`**: Mapped to *"Question must be between 3 and 4,000 characters."*
- **`500 Internal Server Error`**: Mapped to *"Document retrieval failed on server."*
- **`502 Bad Gateway`**: Mapped to *"Answer generation failed on server provider."*
- **`503 Service Unavailable`**: Mapped to *"Backend service is warming up. Please retry shortly."*
- **`AbortError / Timeout`**: Mapped to *"Request timed out or was cancelled."*

---

## 5. Regex Citation Parsing Engine (`src/components/GroundedAnswer.tsx`)

The backend LLM produces responses containing raw citation markers, e.g.:
```text
Under Section 482 CrPC, High Courts possess inherent jurisdiction [Chunk ID: WBCHCJ0003822020_1_2020-01-28_0]...
```

### Parsing Pipeline:
1. **Pre-computation**: A `Map<string, { index: number, citation: Citation }>` is constructed from `citations` for $O(1)$ lookups.
2. **Regex Scanning**: Matches `\[(?:Chunk(?:\s*ID)?[:\s]+)([^\]]+)\]`.
3. **Token Replacement**:
   - Replaces ugly raw UUIDs with sleek academic badges: `[1 · CAL]` or `[1]`.
   - Attaches `title` tooltip displaying the case law title and court name.
   - Binds `onClick` event to trigger smooth scrolling to `citation-${chunkId}` in the sidebar.
4. **Markdown Formatting**: Renders bold text (`**term**`), section headers (`### Section`), numbered lists, bullet points, and blockquotes with consistent typography.

---

## 6. Design System & CSS Token Architecture

Configured in `frontend/tailwind.config.js`:

```javascript
colors: {
  canvas: {
    base: "#0B0E14",             // Deep Midnight Obsidian
    surface: "#10131a",          // Root Surface
    surfaceLow: "#191c22",       // Content Containers & Query Card
    surfaceContainer: "#1d2026", // Mid-tier surface
    surfaceHigh: "#272a31",      // Active States & Elevated Cards
    surfaceHighest: "#32353c"
  },
  brass: {
    DEFAULT: "#D4AF37",          // Legal Brass / Amber Accent
    light: "#f2ca50",            // Active Highlight
    dim: "#e9c349",
    dark: "#554300"
  },
  slateSteel: {
    DEFAULT: "#7B96B2",          // Secondary Metallic Accent
    light: "#aec9e7",            // Court Tags & Structural Badges
    dark: "#2e4962"
  },
  vectorMint: {
    DEFAULT: "#2DD4BF",          // Precision Telemetry Mint
    light: "#48e5d0",            // Latency Counters (*_ms)
    dim: "#11c9b4",
    dark: "#004f46"
  }
}
```

### Typography Scale:
- **`Newsreader` (Serif)**: Editorial headings, case law titles, statutory citations.
- **`Inter` (Sans)**: Body paragraphs, form controls, button labels.
- **`JetBrains Mono` (Monospace)**: Chunk IDs, CNR numbers, query timers, telemetry badges.
