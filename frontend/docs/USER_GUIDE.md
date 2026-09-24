# LegalRAG Frontend — User & Developer Guide

A complete guide for users, developers, and maintainers interacting with or extending the **LegalRAG** web interface.

---

## 1. Application Flow & Architecture

The frontend is a dedicated Single Page Application (SPA) with automatic **AWS EC2 readiness verification** and a 2-page research flow:

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

### Initial State: AWS EC2 Readiness Gatekeeper (`BackendOfflineView.tsx`)
- **Automatic Probe**: On application load, the frontend checks `GET /ready` on the backend cluster.
- **Cost Transparency Notice**: If the backend is offline (AWS EC2 `t4.xlarge` instance paused to eliminate continuous hosting costs for the 538k corpus indexes), the app displays an informative standby screen.
- **On-Demand Activation CTA**: A direct LinkedIn button allows reviewers or recruiters to reach out to the administrator (`https://www.linkedin.com/in/siddharth-dongardive`) to spin up the container instance for a live demo.
- **Instant Re-Check**: A "Re-check Status" button allows immediate validation once the EC2 instance is powered on.

### Page 1: AskView (`src/components/AskView.tsx`)
- **Hero Banner**: Highlights the 538,079 chunk corpus scale and dual-index hybrid architecture.
- **Inquiry Contextualizer**: A spacious legal inquiry textarea with real-time character counting (`3 – 4,000` character bounds) and keyboard submit support (`Cmd+Enter` / `Ctrl+Enter`).
- **Live Multi-Stage Visualizer**: Activates during query execution, rendering an elapsed seconds counter and highlighting active pipeline stages (Hybrid Retrieval → Cross-Encoder Rerank → Grounded LLM Synthesis).
- **Cancel Button**: Employs browser `AbortController` to cancel long-running in-flight queries.
- **Curated Benchmark Test Vectors**: 6 one-click evaluation queries drawn from the 497-question gold benchmark set (Criminal Procedure, Writs, Anticipatory Bail, NI Act Presumptions, Arbitration, Service Law).

### Page 2: ResultsView (`src/components/ResultsView.tsx`)
- **Top Navigation Strip**:
  - `← New Inquiry`: Resets view to `AskView`.
  - `Grounded Synthesis` / `Notice` indicator: Reflects backend status (`ok`, `generation_error`, `retrieval_error`).
  - `Copy Synthesis`: Copies the full question, synthesis prose, and formatted citations to clipboard.
- **Inquiry Recap Banner**: Displays the exact verbatim question queried and authority counts.
- **Dual-Pane Layout**:
  - **Left Pane (`GroundedAnswer.tsx`)**: Renders the generated legal synthesis prose with academic citation badges (`[1]`, `[1 · CAL]`).
  - **Right Pane (`CitationCard.tsx`)**: Displays the cited judgment passages with High Court code, decision date, case title, CNR copy button, and Chunk ID copy button.
- **Bottom Telemetry Strip (`LatencyPill.tsx`)**: Displays execution time breakdown across:
  1. *Retrieval*: BM25 + BGE Dense + RRF (ms)
  2. *Rerank*: Cross-Encoder (ms)
  3. *Generation*: Groq LLM (ms)
  4. *Total*: End-to-end server duration (ms / seconds)

---

## 2. Interactive Citation Grounding

The citation grounding mechanism bridges the generated text with evidentiary source chunks:

```
LLM Output: "...pursuant to Section 482 [Chunk ID: WBCHCJ0003822020_1_2020-01-28_0]..."
                                       │
                                       ▼ (Regex Match in GroundedAnswer.tsx)
Rendered UI: "...pursuant to Section 482 [1 · CAL]..."
                                       │
                        ┌──────────────┴──────────────┐
                        ▼                             ▼
                 [On Hover]                      [On Click]
          Shows rich tooltip with         • Highlights badge in Brass
          case title & High Court         • Scrolls Right Pane to CitationCard #1
                                          • Highlights CitationCard #1
```

### Citation Numbering & Court Tags
- Citations are indexed `[1]`, `[2]`, `[3]` corresponding to their rank in the `citations` response array.
- If a court code exists (e.g. `BOM`, `DEL`, `CAL`, `KER`, `MAD`), the badge displays `[1 · CAL]`.
- If no court code is available, it gracefully defaults to `[1]`.

---

## 3. Key Files & Responsibilities

| File | Purpose | Key Details |
| :--- | :--- | :--- |
| `src/App.tsx` | View state & query orchestrator | Holds `backendStatus`, `view`, `question`, `result`, `isLoading`, `error`, `abortController`. |
| `src/api.ts` | Backend HTTP client | Handles `checkBackendReady()` (`GET /ready`), `POST /query`, 120s timeout, abort signals, and HTTP error code mapping (`422`, `500`, `502`, `503`). |
| `src/config.ts` | Configuration resolver | Reads `VITE_API_URL`, `VITE_API_BASE_URL`, and `VITE_API_TIMEOUT_MS`. |
| `src/types.ts` | Invariant TypeScript types | Defines `QueryRequest`, `QueryResponse`, `Citation`, `QueryStatus`, `BackendStatus`, `ActiveView`. |
| `src/components/BackendOfflineView.tsx` | AWS EC2 Standby Fallback | Explains cloud hosting pause, provides LinkedIn activation CTA and instant re-check button. |
| `src/components/AskView.tsx` | Inquiry page | Character validation, 6 benchmark cards, live execution progress bar. |
| `src/components/ResultsView.tsx` | Results page coordinator | Dual-pane layout, copy actions, status alerts, telemetry container. |
| `src/components/GroundedAnswer.tsx` | Distraction-free prose renderer | Custom regex parser converting `[Chunk ID: ...]` to interactive badges. |
| `src/components/CitationCard.tsx` | Source judgment authority card | Displays CNR, Court, Date, Case title in Newsreader font, Chunk ID. |
| `src/components/LatencyPill.tsx` | Performance breakdown badge | Segmented bar showing Retrieval %, Rerank %, and Generation %. |
| `src/components/Header.tsx` | Global top navigation bar | Brand logo, corpus size indicator, pipeline architecture tag. |

---

## 4. How to Customize Benchmark Test Queries

To modify or add test queries to the Ask page:
1. Open `src/components/AskView.tsx`.
2. Locate the `BENCHMARK_QUERIES` array:
   ```typescript
   const BENCHMARK_QUERIES: BenchmarkQuery[] = [
     {
       category: 'Criminal Procedure',
       query: 'Section 482 CrPC quashing of FIR in matrimonial disputes...',
       jurisdiction: 'Supreme Court & High Courts',
     },
     // Add your custom queries here
   ];
   ```
3. Save and Vite will hot-reload the changes instantly.

---

## 5. Troubleshooting & FAQ

### Q1: The search returns "Unable to connect to backend server" or "Failed to fetch".
- **Cause**: The frontend cannot reach the URL specified in `VITE_API_URL`.
- **Solution**:
  1. Check `.env.development` or `.env.production`. Ensure `VITE_API_URL` is set to the correct host (e.g. `http://54.80.219.70:7860`).
  2. Verify that the backend server is running and port 7860 is open.
  3. Verify that CORS is enabled on the FastAPI backend.

### Q2: Why does the request take 5–10 seconds?
- **Explanation**: The production backend searches over **538,079 chunks** across dual indexes (BM25 lexical + BGE dense FAISS), performs Reciprocal Rank Fusion, runs neural cross-encoder reranking on top-50 candidates, and then streams context to the Groq LLM for grounded synthesis.
- **Client Timeout**: The frontend client timeout is set to 120 seconds (`VITE_API_TIMEOUT_MS=120000`) to safely accommodate cold starts and heavy multi-statute queries.

### Q3: Why are there no citations listed for some answers?
- **Explanation**: LegalRAG enforces a strict anti-hallucination contract. If the LLM generates a claim that does not cite a valid chunk from the retrieved top-5 context, or if citations do not resolve to known chunks, the backend filters them out.

### Q4: How do I change the theme colors or typography?
- **Tailwind Config**: Open `tailwind.config.js` to modify color tokens (`canvas`, `brass`, `slateSteel`, `vectorMint`) or font families (`Newsreader`, `Inter`, `JetBrains Mono`).
- **Global CSS**: Open `src/index.css` for base layer styling and font imports.
