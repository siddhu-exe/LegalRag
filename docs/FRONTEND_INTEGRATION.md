# LegalRAG — Frontend Integration Guide

**Status:** single source of truth for the frontend developer.
**Derived from:** the current backend source (`src/legalrag/api/`), its live
OpenAPI schema, and responses captured from the running FastAPI app (2026-09-23).
**Scope:** how to call the backend correctly. Nothing here requires knowledge of the
retrieval internals, and nothing here may be changed by the frontend.

> Every field, status code, and header in this document was read from, or captured
> from, the actual backend implementation. If the backend changes, this document must
> be updated with it — do not "fix" a mismatch by guessing.

---

## 1. Backend Overview

The backend exposes one HTTP service (FastAPI) that runs a fixed, multi-stage
retrieval-augmented generation cascade. The cascade is **not configurable by the
client** — the frontend sends a question and receives a grounded answer plus citations.

```
Frontend (browser)
   │  POST /query  { "question": "..." }
   ▼
FastAPI  (src/legalrag/api)
   ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. BM25 lexical retrieval        → top-50 chunk IDs          │
│ 2. Dense (FAISS + BGE) retrieval → top-50 chunk IDs          │
│ 3. Reciprocal Rank Fusion (k=60) → merged top-50 candidates   │
│ 4. Cross-Encoder reranking        → final top-5 chunks        │
│ 5. Groq LLM generation           → grounded answer text      │
│ 6. Citation extraction + latency attribution                 │
└─────────────────────────────────────────────────────────────┘
   ▼
{ "answer", "citations": [...], "retrieved_chunk_ids": [...],
  "retrieve_ms", "rerank_ms", "generate_ms", "total_ms", "status" }
```

What each stage does, in one line each:

| Stage | What it does | Why the frontend cares |
| --- | --- | --- |
| BM25 (lexical) | Keyword/statute-term matching over 538,079 judgment chunks. | Dominates retrieval latency (~2–3 s). |
| Dense (semantic) | Sentence-embedding search (`BAAI/bge-base-en-v1.5`, 768-d) in a FAISS index. | Fast (~0.1 s), adds semantic recall BM25 misses. |
| RRF (k=60) | Merges the two ranked lists by reciprocal rank, no tuning needed. | Sub-millisecond; produces the candidate pool. |
| Cross-Encoder | Scores each `(question, chunk)` pair jointly and keeps the best 5. | ~1–1.5 s; determines which chunks can be cited. |
| Groq generation | Writes a grounded answer using only the top-5 chunks as context. | The slowest, most failure-prone stage (502 on failure). |
| Citation extraction | Parses `[Chunk ID: ...]` markers from the answer and attaches case metadata. | Determines the `citations` array contents. |

Two invariants the frontend must respect:

1. **Citations are grounded, not invented.** The backend drops any citation that does
   not match one of the retrieved top-5 chunks, so `citations` may legitimately be
   empty even for a successful answer.
2. **Production fails closed.** If the retrieval models/artifacts are unusable the
   backend returns an error — it never silently degrades to stub answers.

---

## 2. Base URL

| Environment | Base URL |
| --- | --- |
| Local development | `http://localhost:7860` |
| Production (AWS) | **Not yet defined.** No production hostname is committed in this repository. |

The backend listens on `0.0.0.0:7860` (`API_PORT` / `PORT`, default `7860`). In
production it will sit behind whatever hostname the AWS deployment assigns (load
balancer / container service / hosting platform). **The production base URL is a
deployment configuration item: obtain it from whoever deploys the backend, do not
invent one, and do not commit a guess.**

### Frontend configuration rules

- Read the base URL from a build-time environment variable — never hardcode
  `localhost` in application logic.
- Vite convention:

```bash
# .env.development  (local)
VITE_API_BASE_URL=http://localhost:7860

# .env.production   (set at build/deploy time)
VITE_API_BASE_URL=https://<api-host-assigned-at-deploy-time>
```

Usage:

```ts
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:7860";
```

Notes:

- Vite **inlines** `VITE_*` variables into the built JavaScript. They are public.
  Never place an API key, token, or server setting in a `VITE_*` variable.
- The fallback above is a convenience for local dev only; production builds must
  define `VITE_API_BASE_URL` explicitly.
- Ensure the value has no trailing slash if you concatenate paths manually.

---

## 3. API Endpoints

All frontend-relevant endpoints currently implemented:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness — is the process up? |
| `GET` | `/ready` | Readiness — is the inference pipeline usable? |
| `POST` | `/query` | Ask a legal question (the product endpoint). |
| `GET` | `/` | Service identity/metadata (small convenience endpoint). |
| `GET` | `/docs` | Swagger UI (interactive API explorer, dev/admin tooling). |
| `GET` | `/openapi.json` | Raw OpenAPI schema (codegen). |

There is **no authentication** on any endpoint, no API key header, and no other
resource endpoints (see §16).

### 3.1 `GET /health`

- **Purpose:** liveness. Confirms the HTTP process responds. It deliberately does
  **not** load models, indexes, or artifacts, so it stays fast even while the
  pipeline is initializing or broken.
- **Request headers:** none required. Body: none.
- **Success status:** `200`
- **Response body:**

```json
{ "status": "ok", "environment": "production" }
```

- `status`: `"ok"` (or `"degraded"` in the schema; the current implementation always
  returns `"ok"`).
- `environment`: `"production"` or `"local_stub"` — the active backend mode. Useful
  for diagnostics; there is no `degraded` path implemented today.
- **Error statuses:** none defined beyond generic infrastructure failures.

```bash
curl -s http://localhost:7860/health
```

```ts
const res = await fetch(`${API_BASE_URL}/health`);
const health: HealthResponse = await res.json();
```

### 3.2 `GET /ready`

- **Purpose:** readiness. Returns `200` only when the pipeline is initialized **and**
  every retrieval component is actually usable. Otherwise `503`.
- **Request headers:** none required. Body: none.
- **Success status:** `200`

```json
{ "status": "ready", "environment": "production" }
```

- **Error status:**

```json
{ "detail": "Service is not ready." }
```

- **Frontend guidance:** this is infrastructure/admin tooling, **not** a
  user-facing call. Do not poll it aggressively (see §11).

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:7860/ready
```

```ts
const res = await fetch(`${API_BASE_URL}/ready`);
if (res.status === 503) showBanner("Backend is starting up or unavailable.");
```

### 3.3 `POST /query`

The product endpoint: run the full cascade and return a grounded answer. Full
contract in §4 and §5.

- **Request headers:** `Content-Type: application/json`
- **Success status:** `200`
- **Error statuses:** `422` (invalid request), `500` (retrieval/reranking failure),
  `502` (generation failure), `503` (service not ready)

```bash
curl -s -X POST http://localhost:7860/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the essential elements required to establish negligence under Indian law?"}'
```

```ts
const res = await fetch(`${API_BASE_URL}/query`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question }),
});
```

### 3.4 `GET /` (convenience)

Returns service identity and the URL map. Not needed by the UI, but handy for
smoke-testing a deployment.

```json
{
  "name": "LegalRAG API",
  "version": "0.1.0",
  "environment": "production",
  "docs_url": "/docs",
  "health_url": "/health",
  "readiness_url": "/ready",
  "query_url": "/query"
}
```

---

## 4. `POST /query` — Complete Request Contract

**Method / path:** `POST /query`
**Required header:** `Content-Type: application/json`
**Body:** exactly one field.

```json
{
  "question": "What are the conditions for granting anticipatory bail under Section 438 CrPC?"
}
```

### Field specification

| Field | Type | Required | Rules |
| --- | --- | --- | --- |
| `question` | `string` | **Yes** | Must be a JSON string. Leading/trailing whitespace is stripped server-side. After trimming it must be **≥ 3** and **≤ 4000** characters. Must not be empty or whitespace-only. |

Validation summary (enforced by the backend, authoritative):

- minimum length **3** characters (after trimming),
- maximum length **4000** characters,
- non-string values (numbers, objects, arrays, `null`, booleans) → `422`,
- `{}` (missing `question`) → `422`,
- `""` or `"   "` → `422`,
- malformed JSON body → `422`,
- wrong/missing `Content-Type` → `422`,
- **extra fields are rejected** — the schema is `additionalProperties: false`
  ("extra inputs are not permitted"). There is no `top_k`, `model`, `temperature`,
  `session_id`, `stream`, or any other tunable.

### What must NOT be sent

The request body may contain **only** `question`. The following must never be sent by
the frontend, and the backend explicitly rejects them with `422`:

- `environment`, `groq_api_key`, `hf_token`, `hf_repo_id`
- `artifact_dir`, any file path
- `embedding_model_name`, `reranker_model_name`, `groq_model_name`
- `api_host`, `api_port`, provider credentials

**Server configuration is server-side only.** The backend exposes no configuration
input, and the frontend must never attempt to supply any. (There is a deliberate
regression test asserting this: `test_client_cannot_inject_server_settings`.)

### Valid request

```json
{ "question": "What are the essential elements required to establish negligence under Indian law?" }
```

### Invalid requests (all → `422`)

```json
{}
```

```json
{ "question": "" }
```

```json
{ "question": "   " }
```

```json
{ "question": 42 }
```

```json
{ "question": "ab" }
```

```json
{ "question": "valid question here", "environment": "production" }
```

```json
{ "question": "valid question here", "top_k": 5 }
```

---

## 5. `POST /query` Response (HTTP 200)

The response schema is `QueryResponse`. All fields below are always present in the
JSON body; none are conditional.

```json
{
  "answer": "Held, per the retrieved judgments [Chunk ID: CNR000001_chunk_0] [Chunk ID: CNR000231_chunk_0]",
  "citations": [
    {
      "chunk_id": "CNR000001_chunk_0",
      "cnr": "CNR000001",
      "court_code": "01",
      "decision_date": "2001-01-15",
      "title": "Case 1 v. State"
    },
    {
      "chunk_id": "CNR000231_chunk_0",
      "cnr": "CNR000231",
      "court_code": "06",
      "decision_date": "2011-01-15",
      "title": "Case 231 v. State"
    }
  ],
  "retrieved_chunk_ids": [
    "CNR000001_chunk_0",
    "CNR000231_chunk_0",
    "CNR000011_chunk_0",
    "CNR000221_chunk_0",
    "CNR000021_chunk_0"
  ],
  "retrieve_ms": 0.69,
  "rerank_ms": 1.32,
  "generate_ms": 0.16,
  "total_ms": 2.17,
  "status": "ok"
}
```

> The example above was captured end-to-end from the backend's scaled integration
> fixture (real BM25 + FAISS retrieval, deterministic stub embedding/reranker/
> generator). The shape is identical in production; only the values and the
> `chunk_id` format differ. **Real production chunk IDs look like**
> `WBCHCJ0003822020_1_2020-01-28_0` — `<CNR>_<seq>_<YYYY-MM-DD>_<chunk_index>`
> (verified against the shipped `legal_chunks.parquet`: 538,079 rows, 0 null and 0
> duplicate `chunk_id`). Treat `chunk_id` as an **opaque string**: use it as a key and
> as an identity for deep links added later, but do not parse it, split it, or assume
> the synthetic `CNR…_chunk_0` shape from the example above.

### Field-by-field

| Field | Type | Display in UI? | Meaning |
| --- | --- | --- | --- |
| `answer` | `string` | **Yes** | The generated grounded answer, with inline `[Chunk ID: …]` markers. |
| `citations` | `Citation[]` | **Yes** | Metadata for the sources the model actually cited. May be empty. |
| `retrieved_chunk_ids` | `string[]` | Optional / debug | The final top-5 chunk IDs that were considered (superset of citations). |
| `retrieve_ms` | `number` | Optional / debug | BM25 + Dense + RRF wall time, ms. |
| `rerank_ms` | `number` | Optional / debug | Cross-encoder wall time, ms. |
| `generate_ms` | `number` | Optional / debug | Groq generation wall time, ms. |
| `total_ms` | `number` | Optional / debug | Server-side end-to-end time, ms. |
| `status` | `"ok" \| "generation_error" \| "retrieval_error"` | Debug only | See the note below. |

**Three distinct groups — do not mix them in the UI:**

1. **Answer** (`answer`) — the primary content. Render as prose. The `[Chunk ID: …]`
   markers are part of the text; you may strip/hyperlink them, but the underlying
   citation data lives in `citations`.
2. **Citation metadata** (`citations`) — structured, renderable source cards
   (see §6).
3. **Debug/telemetry** (`retrieved_chunk_ids`, the four `*_ms` fields, `status`) —
   useful for a "sources considered" panel, a developer drawer, or latency
   monitoring. Never required for the core UX.

### `status` — important subtlety

The schema allows `"ok" | "generation_error" | "retrieval_error"`, but the **current
implementation returns HTTP 200 only with `"status": "ok"`**. Retrieval, reranking,
and generation failures are raised as HTTP `500` / `502` errors, not reported
in-body. Therefore:

- Treat `status !== "ok"` defensively (log it, show a generic error), but do not
  build a UI flow that depends on receiving `"generation_error"` in a 200 response.
- Correspondingly, `answer` is never an error string; the schema comment mentioning
  "error details if failed" is descriptive only, because failures never reach a 200.

### Array-element notes

- `citations[].chunk_id` is always a `string`.
- `citations[].cnr`, `.court_code`, `.decision_date`, `.title` are
  `string | null` — the backend coerces values to strings when present and returns
  `null` when metadata is missing. **Always null-guard before rendering.**
- `citations` is a subset of `retrieved_chunk_ids`; the reverse is not true.

---

## 6. Citations

A citation is a *pointer to a judgment chunk the model used*, plus the metadata needed
to render a readable source line. The backend builds the array with this logic:

1. Parse every `[Chunk ID: <id>]` marker from the generated answer
   (`extract_citations`), preserving order and de-duplicating. The accepted form is
   `[Chunk ID: <id>]` or `[Chunk ID: <id> | Court: … | Date: …]`, case-insensitive on
   `Chunk ID`; only the ID is captured. This is the marker you may highlight or link
   inside the answer text.
2. Discard any ID that is not among the retrieved top-5 chunks — this blocks
   hallucinated sources.
3. Attach metadata looked up from the chunk store.

### Citation fields

| Field | Type | Example | Meaning / rendering |
| --- | --- | --- | --- |
| `chunk_id` | `string` | `"WBCHCJ0003822020_1_2020-01-28_0"` | Stable identifier of the cited chunk. Always present, never null, unique. Use as the React key. Opaque — do not parse or reformat it for display. |
| `cnr` | `string \| null` | `"CNR000001"` | Case Record Number — the case identifier. Good secondary line / monospace badge. |
| `court_code` | `string \| null` | `"01"` | Code of the High Court. A short code, not a court name; map to a label if you want friendly names. |
| `decision_date` | `string \| null` | `"2001-01-15"` | Judgment date as a string (`YYYY-MM-DD` in practice). Format for display; do not assume a locale-specific format. |
| `title` | `string \| null` | `"Case 1 v. State"` | Case title / parties. Best primary label for a citation card. |

### Rendering guidance

- Render citations **below or beside the answer**, as an ordered list keyed by
  `chunk_id`.
- Show `title` first (fall back to `cnr`, then `chunk_id`), with `court_code` and
  `decision_date` as secondary metadata.
- If `citations` is empty, show a neutral note (e.g. "No sources cited for this
  answer") rather than an error — a valid grounded answer can cite nothing.
- Optionally make each citation scroll to / highlight the matching `[Chunk ID: …]`
  marker in the answer text.

Realistic rendering example:

```tsx
function Citations({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) return <p className="muted">No sources cited.</p>;
  return (
    <ol className="citations">
      {citations.map((c) => (
        <li key={c.chunk_id}>
          <strong>{c.title ?? c.cnr ?? c.chunk_id}</strong>
          <span className="meta">
            {[c.cnr, c.court_code ? `Court ${c.court_code}` : null, c.decision_date]
              .filter(Boolean)
              .join(" · ")}
          </span>
        </li>
      ))}
    </ol>
  );
}
```

### ⚠️ There is no way to fetch the full judgment text

The backend exposes **no** endpoint that returns a chunk's text or a document by ID.
`GET /chunks/{id}` and `GET /documents/{id}` **do not exist**. The frontend therefore
cannot display full judgment text, expand a citation to the source paragraph, or link
to the original PDF through this API. Citation cards must be built from the five
metadata fields above only. (If source-text display becomes a product requirement,
that requires a backend change — request it, do not fake it in the UI.)

---

## 7. Error Handling

### Error body shapes

There are exactly two shapes:

**Validation errors (`422`) — FastAPI/Pydantic format:**

```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["body", "question"],
      "msg": "Value error, Question must not be empty or whitespace-only.",
      "input": "",
      "ctx": { "error": {} }
    }
  ]
}
```

`detail` is an **array**; `loc` is an array of path segments; `type`, `msg`, `input`
(and sometimes `ctx`) are present. Observed `type` values: `missing`,
`value_error`, `string_too_short`, `string_too_long`, `extra_forbidden`,
`json_invalid`, `model_attributes_type`.

**Operational errors (`500` / `502` / `503`) — sanitized single string:**

```json
{ "detail": "Answer generation failed." }
```

`detail` is a **string**. The backend never returns stack traces, provider errors,
file paths, or credentials in any error body.

### Status code table

| Status | Meaning | Exact `detail` value(s) | Frontend behavior |
| --- | --- | --- | --- |
| `422` | Invalid request (validation failed before any retrieval work). This is a **frontend bug**, not a backend failure. | array of validation items (see above) | Do not retry. Validate the question client-side first (non-empty, 3–4000 chars). Log the `detail` in dev; show the user a field-level hint. |
| `500` | Retrieval or reranking stage failed (index/model/artifact problem). | `"Document retrieval failed."` or `"Candidate reranking failed."` | Do not auto-retry (retrying will not fix a broken index). Show "Something went wrong on our side." Offer manual retry. Alert ops if it persists. |
| `502` | Generation failed: LLM provider error, timeout, auth, or unusable output. | `"Answer generation failed."` | Safe to retry once after a short delay (transient provider issues are the common cause). Otherwise show "The answer service is temporarily unavailable." |
| `503` | Service not ready: pipeline still initializing, or artifacts/configuration unavailable. | `"Service is not ready."` | Treat as "backend starting up / unavailable". Show a banner and let the user retry manually; avoid tight polling loops. |

Body examples as returned by the backend:

```json
{ "detail": "Document retrieval failed." }
```

```json
{ "detail": "Candidate reranking failed." }
```

```json
{ "detail": "Answer generation failed." }
```

```json
{ "detail": "Service is not ready." }
```

### Recommended user-facing messages

> These are **frontend presentation guidance only**. The strings below are *not*
> backend responses — the backend never sends user-friendly copy. Never render the raw
> `detail` string to end users for 500/502/503; map it to your own copy.

| Status | Suggested copy (yours, not the API's) |
| --- | --- |
| `422` | "Please enter a question of at least 3 characters." |
| `500` | "We couldn't search the judgments right now. Please try again shortly." |
| `502` | "The answer service is temporarily unavailable. Please try again." |
| `503` | "The service is starting up. Please try again in a moment." |
| network error / timeout | "Couldn't reach the LegalRAG service. Check your connection and retry." |

Rules:

- Never surface `detail` raw for `500`/`502`/`503`, and never surface stack traces —
  the backend does not send them, and your error UI should not either.
- Treat unknown non-2xx statuses as a generic failure with a manual retry.

---

## 8. Loading States

`POST /query` is **synchronous and slow**: the backend runs real retrieval, a
cross-encoder, and an LLM call before responding. There is **no streaming and no
progress endpoint**.

Approximate timings measured during local validation on CPU with the real
538,079-chunk artifacts and the production model set:

| Stage | Measured | Notes |
| --- | --- | --- |
| Retrieval (BM25 + Dense + RRF) | ~2–4 s | BM25 over 538k chunks dominates (~2.5 s); dense ~0.15 s; RRF ~0 ms. |
| Reranking (Cross-Encoder, 50 pairs) | ~1–1.5 s | CPU inference over the fused candidate pool. |
| Generation (Groq) | several seconds | Network + provider queue dependent. |
| **Total** | **commonly ~7–10 s** | Warm pipeline, no cold-start. |

Also relevant: **cold start** (first ever request after container start, when models
and the 2.4 GB of artifacts are loaded) took ~30 s in local validation, and the
backend's own Groq client is configured with a 30 s timeout and up to 2 retries — so a
worst-case generation stage alone can last ~90 s.

Implications:

- **Do not** treat a few seconds of silence as a stalled request.
- **Do not hardcode these numbers** into UI logic — they are observations, not a
  contract. They will differ on production hardware and with provider latency.
- Show an immediate pending state: disable the submit button, show a spinner, and
  after a few seconds switch to a message such as "Searching 538,079 judgments… this
  can take up to ~30 seconds."
- Allow cancellation (AbortController) so a user can back out of a long request.
- Recommended client timeout: **120 s** (see §9). Do not use a 5–10 s timeout; it
  would abort perfectly healthy requests.
- Prevent duplicate submissions: one in-flight `/query` per user action. The backend
  is a single process with heavy models, so parallel duplicate requests just queue.

---

## 9. Frontend Fetch Example (TypeScript)

```ts
// src/api/legalrag.ts
import type {
  Citation,
  QueryResponse,
  ValidationErrorResponse,
} from "./types";

const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:7860";

/** Default client-side timeout: the cascade legitimately takes ~10 s, and the
 *  backend allows a 30 s provider timeout with up to 2 retries. */
const DEFAULT_TIMEOUT_MS = 120_000;

export class LegalRagApiError extends Error {
  readonly status: number;
  readonly kind: "validation" | "retrieval" | "generation" | "not_ready" | "network";

  constructor(kind: LegalRagApiError["kind"], status: number, message: string) {
    super(message);
    this.name = "LegalRagApiError";
    this.kind = kind;
    this.status = status;
  }
}

function kindForStatus(status: number): LegalRagApiError["kind"] {
  switch (status) {
    case 422:
      return "validation";
    case 500:
      return "retrieval";
    case 502:
      return "generation";
    case 503:
      return "not_ready";
    default:
      return "network";
  }
}

/** Asks the LegalRAG backend a question. Throws LegalRagApiError on failure. */
export async function queryLegalRAG(
  question: string,
  options: { signal?: AbortSignal; timeoutMs?: number } = {},
): Promise<QueryResponse> {
  const trimmed = question.trim();

  // Supplementary client-side validation only; the backend is authoritative.
  if (trimmed.length < 3) {
    throw new LegalRagApiError(
      "validation",
      422,
      "Please enter a question of at least 3 characters.",
    );
  }
  if (trimmed.length > 4000) {
    throw new LegalRagApiError(
      "validation",
      422,
      "Please shorten your question to 4000 characters or fewer.",
    );
  }

  const controller = new AbortController();
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  // Allow the caller to cancel as well.
  const externalSignal = options.signal;
  const onExternalAbort = () => controller.abort();
  externalSignal?.addEventListener("abort", onExternalAbort, { once: true });

  try {
    const response = await fetch(`${API_BASE_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: trimmed }), // ONLY this field
      signal: controller.signal,
    });

    if (!response.ok) {
      let detail: unknown;
      try {
        detail = (await response.json())?.detail;
      } catch {
        detail = undefined;
      }
      if (import.meta.env.DEV) {
        console.debug("[legalrag] error response", response.status, detail);
      }
      throw new LegalRagApiError(
        kindForStatus(response.status),
        response.status,
        messageForStatus(response.status),
      );
    }

    return (await response.json()) as QueryResponse;
  } catch (error) {
    if (error instanceof LegalRagApiError) throw error;
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new LegalRagApiError("network", 0, "The request was cancelled or timed out.");
    }
    throw new LegalRagApiError("network", 0, "Couldn't reach the LegalRAG service.");
  } finally {
    clearTimeout(timeoutId);
    externalSignal?.removeEventListener("abort", onExternalAbort);
  }
}

/** Safe, user-facing copy. Never echo raw backend `detail` for 5xx responses. */
function messageForStatus(status: number): string {
  switch (status) {
    case 422:
      return "Please check your question and try again.";
    case 500:
      return "We couldn't search the judgments right now. Please try again shortly.";
    case 502:
      return "The answer service is temporarily unavailable. Please try again.";
    case 503:
      return "The service is starting up. Please try again in a moment.";
    default:
      return "Something went wrong. Please try again.";
  }
}

/** Optional: type guard for the 422 shape (useful in dev tooling only). */
export function isValidationError(value: unknown): value is ValidationErrorResponse {
  const detail = (value as { detail?: unknown })?.detail;
  return Array.isArray(detail);
}
```

Usage with loading state:

```ts
async function onSubmit(question: string) {
  setStatus("loading");
  setError(null);
  try {
    const data = await queryLegalRAG(question);
    setAnswer(data.answer);
    setCitations(data.citations);
    setMeta({ total_ms: data.total_ms, retrieved: data.retrieved_chunk_ids.length });
    setStatus("done");
  } catch (err) {
    setStatus("error");
    setError(err instanceof LegalRagApiError ? err.message : "Unexpected error.");
    // Retry is only worth offering for generation / not-ready / network.
    setCanRetry(err instanceof LegalRagApiError && err.kind !== "validation");
  }
}
```

Security rules for this file:

- No `GROQ_API_KEY`, no `HF_TOKEN`, no Hugging Face credentials, no artifact paths —
  ever. The frontend talks to one public HTTP endpoint and needs nothing else.
- Do not send server configuration in the body (it would be rejected with `422`).
- Do not log `detail` arrays to end-user-visible surfaces; keep them in the console.

---

## 10. TypeScript Types

These match the backend's Pydantic schemas exactly. `src/api/types.ts`:

```ts
/** POST /query request body — exactly one field, nothing else. */
export interface QueryRequest {
  question: string;
}

/** Pipeline execution status. Currently always "ok" on a 200 response. */
export type QueryStatus = "ok" | "generation_error" | "retrieval_error";

/** One cited source passage. Only `chunk_id` is guaranteed non-null. */
export interface Citation {
  chunk_id: string;
  cnr: string | null;
  court_code: string | null;
  decision_date: string | null;
  title: string | null;
}

/** POST /query success response (HTTP 200). */
export interface QueryResponse {
  answer: string;
  citations: Citation[];
  retrieved_chunk_ids: string[];
  retrieve_ms: number;
  rerank_ms: number;
  generate_ms: number;
  total_ms: number;
  status: QueryStatus;
}

/** 500 / 502 / 503 — sanitized single-string detail. */
export interface ErrorResponse {
  detail: string;
}

/** One entry in a 422 validation error. */
export interface ValidationErrorItem {
  type: string;
  loc: Array<string | number>;
  msg: string;
  input?: unknown;
  ctx?: Record<string, unknown>;
}

/** 422 — FastAPI/Pydantic validation error. `detail` is an ARRAY here. */
export interface ValidationErrorResponse {
  detail: ValidationErrorItem[];
}

/** GET /health */
export interface HealthResponse {
  status: "ok" | "degraded";
  environment: string;
}

/** GET /ready (200) */
export interface ReadinessResponse {
  status: "ready";
  environment: string;
}
```

Caveats when generating types from `/openapi.json` instead of copying the above:

- `citations` and `retrieved_chunk_ids` are **not** in the schema's `required` list
  (they have server-side defaults), so generated types may mark them optional. The
  server always serializes both as arrays; treat a missing value defensively
  (`data.citations ?? []`) but do not treat them as optionally-absent by design.
- `Citation.chunk_id` **is** required; the other citation fields are nullable.
- The OpenAPI document only declares `200` and `422` responses for `/query`; the
  `500` / `502` / `503` bodies are produced by explicit exception handlers and by
  route-level raises, so add those types by hand (as above).

---

## 11. Health vs Readiness

| | `GET /health` | `GET /ready` |
| --- | --- | --- |
| Question answered | "Is the process alive?" | "Can it actually serve queries?" |
| Loads models/artifacts | No (deliberately lightweight) | Yes — validates the initialized pipeline |
| Status | `200` (always, if the process is up) | `200` ready, `503` not ready |
| Body | `{"status":"ok","environment":"…"}` | `{"status":"ready","environment":"…"}` or `{"detail":"Service is not ready."}` |
| Use for | liveness probes, "is the backend reachable" | load-balancer/orchestrator readiness gates, admin status UIs |

Guidance:

- Use `/health` for a cheap connectivity check in an admin/status page.
- Use `/ready` **once** when you need to decide whether to enable a "Ask" button
  (e.g. a single check on app load, or after a `503`), and only on explicit user
  action. **Do not poll `/ready` on a timer** — a readiness check validates real
  components and is not free, and aggressive polling in production adds load for no
  benefit.
- Never gate user-visible features on `/ready` at request time; just call `/query`
  and handle `503` if it happens.
- `environment` in both responses tells you which mode the backend is in
  (`production` / `local_stub`). Useful when pointing the frontend at an unknown
  deployment. It is not a secret.

---

## 12. CORS

Current backend configuration (`src/legalrag/api/main.py`):

| Setting | Value |
| --- | --- |
| `allow_origins` | `["*"]` (all origins) |
| `allow_credentials` | `true` |
| `allow_methods` | `["*"]` → `DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT` |
| `allow_headers` | `["*"]` (mirrors requested headers) |
| `max_age` | `600` (preflight cache, seconds) |

Observed behavior on the running app (verified with a preflight and a simple request
from `Origin: https://frontend.example.com`):

```
OPTIONS /query  -> 200
  access-control-allow-origin: https://frontend.example.com
  access-control-allow-credentials: true
  access-control-allow-methods: DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT
  access-control-allow-headers: content-type
  access-control-max-age: 600
  vary: Origin

GET /health (with Origin)
  access-control-allow-origin: https://frontend.example.com
  access-control-allow-credentials: true
  vary: Origin
```

Notes:

- Because credentials are enabled, the middleware echoes the **requesting origin**
  rather than a literal `*`, so browser requests from any origin succeed today.
- CORS is **hardcoded**, not environment-driven: there is no `CORS_ORIGINS` setting
  in `config.py`. **Restricting allowed origins to the real frontend domain is an
  open deployment-configuration item** — it requires a (small) backend change before
  or during AWS deployment, and the frontend must then be served from an allowed
  origin.
- The frontend does not need cookies or credentials; the API is unauthenticated. If
  you use `fetch` without `credentials: "include"`, CORS behaves the same.

---

## 13. Security

**Never put these in frontend code, a `VITE_*` variable, localStorage, or a bundle:**

- `GROQ_API_KEY` (the LLM provider key — billed, and a full account takeover risk)
- `HF_TOKEN`, `HF_REPO_ID` (Hugging Face credentials/repo identifiers)
- any artifact paths (`artifacts/bm25.pkl`, `artifacts/dense.index`,
  `artifacts/legal_chunks.parquet`) or server filesystem layout
- server configuration: `ENVIRONMENT`, `EMBEDDING_MODEL_NAME`, `RERANKER_MODEL_NAME`,
  `GROQ_MODEL_NAME`, `GROQ_REQUEST_TIMEOUT`, `GROQ_MAX_RETRIES`, `API_HOST`,
  `API_PORT`, `MODEL_WARMUP_ENABLED`

The only backend-related configuration the frontend needs is the public API base URL
(`VITE_API_BASE_URL`), which is itself public.

Additional rules:

- Frontend validation is **supplementary**; the backend is authoritative. Client-side
  checks (min/max length, non-empty) exist to give fast feedback, not to enforce the
  contract.
- Expect validation errors (`422`) to include the rejected input in `detail[].input`.
  Do not render that at scale or log it to analytics.
- The backend deliberately sanitizes all operational errors — do not add UI that
  tries to display provider errors; you will never receive them.
- Since the API is unauthenticated, do not treat a successful response as proof of
  user identity or entitlement.

---

## 14. Query UX Guidance

Tied directly to the backend contract:

| Situation | Backend reality | Frontend guidance |
| --- | --- | --- |
| **Empty question** | `422` (`value_error`, min length 3 after trimming) | Disable submit when the trimmed input is < 3 chars. Never send whitespace-only. Show an inline hint, not an error toast. |
| **Submitting** | Request may take ~7–10 s (worst case longer) | Immediately show a pending state; disable the button; prevent duplicate submits; keep the question visible. |
| **Long-running** | No streaming, no progress; generation alone can take ~90 s worst case | After ~5 s, reassure: "Searching 538,079 judgments…". Allow cancel via AbortController. Do not time out before 120 s. |
| **Successful answer** | `200` with `answer`, `citations`, latencies | Render the answer as prose. Optionally show `total_ms` in a developer panel only. |
| **Insufficient evidence** | The prompt instructs the model to reply with a fixed sentence: "The provided judgment context does not contain sufficient information to answer this inquiry." | Render it as a normal (non-error) answer. It is a valid grounded outcome, not a failure — do not show an error state or a retry prompt. |
| **Citations** | `citations` may be empty even on success | Render source cards below the answer; show a neutral "no sources cited" note when empty. Null-guard every field except `chunk_id`. |
| **Generation failure** | `502` (`"Answer generation failed."`) | Show "answer service temporarily unavailable"; offer one manual retry. Do not auto-retry in a loop. |
| **Retrieval/reranking failure** | `500` (`"Document retrieval failed."` / `"Candidate reranking failed."`) | Generic apology + retry. Do not auto-retry; a broken index will not heal. |
| **Backend not ready** | `503` (`"Service is not ready."`) — common right after a deploy/cold start (~30 s) | Show "starting up" and allow a manual retry. Avoid tight polling. |
| **Backend unreachable** | `fetch` rejects (network error) | "Couldn't reach the service." Offer retry. |
| **Invalid request** | `422` | Indicates a frontend bug — log to console in dev, fix the payload. |
| **Retry** | Idempotent: `POST /query` only reads, and costs one LLM call | Manual retry is fine and safe. Auto-retry only for `502`/network, at most once, with a short delay. Never auto-retry `422`. |
| **Duplicate/parallel queries** | Single backend process with heavy models | Serialize queries client-side: one in-flight request at a time. |

---

## 15. Example Complete Flow

```
User types a question
        │
        ▼
Frontend validates (trim, 3–4000 chars)          → invalid: inline hint, no request
        │
        ▼
POST {VITE_API_BASE_URL}/query
     Content-Type: application/json
     { "question": "..." }                        ← exactly one field
        │
        ▼
Backend: BM25 top-50  +  Dense top-50
        │
        ▼
Backend: RRF (k=60) → merged candidate pool (top-50)
        │
        ▼
Backend: CrossEncoder → top-5 chunks
        │
        ▼
Backend: Groq generation → grounded answer + [Chunk ID: …] markers
        │
        ▼
HTTP 200 JSON
        │
        ▼
Frontend renders:
        • Answer                (from `answer`)
        • Citations             (from `citations`)
        • Optional metadata     (from `retrieved_chunk_ids`, `*_ms`, `status`)
```

### Complete request

```http
POST /query HTTP/1.1
Host: localhost:7860
Content-Type: application/json

{"question": "dishonour of cheque under section 138 negotiable instruments act statutory notice"}
```

### Complete response (captured from the backend)

```json
{
  "answer": "Held, per the retrieved judgments [Chunk ID: CNR000001_chunk_0] [Chunk ID: CNR000231_chunk_0]",
  "citations": [
    {
      "chunk_id": "CNR000001_chunk_0",
      "cnr": "CNR000001",
      "court_code": "01",
      "decision_date": "2001-01-15",
      "title": "Case 1 v. State"
    },
    {
      "chunk_id": "CNR000231_chunk_0",
      "cnr": "CNR000231",
      "court_code": "06",
      "decision_date": "2011-01-15",
      "title": "Case 231 v. State"
    }
  ],
  "retrieved_chunk_ids": [
    "CNR000001_chunk_0",
    "CNR000231_chunk_0",
    "CNR000011_chunk_0",
    "CNR000221_chunk_0",
    "CNR000021_chunk_0"
  ],
  "retrieve_ms": 0.69,
  "rerank_ms": 1.32,
  "generate_ms": 0.16,
  "total_ms": 2.17,
  "status": "ok"
}
```

Stage counts observed for this request: BM25 `50`, Dense `50`, RRF `50`, reranker
`5` (all five IDs resolved against the chunk store).

> Captured from the repository's scaled integration fixture (real BM25 + real FAISS
> retrieval over a synthetic corpus, deterministic stub embedding/reranker/generator,
> no network). The response *shape* is identical in production. In production the
> chunk IDs follow `<CNR>_<seq>_<YYYY-MM-DD>_<chunk_index>` (e.g.
> `WBCHCJ0003822020_1_2020-01-28_0`), the corpus is 538,079 chunks, and latencies are
> ~7–10 s total rather than the sub-millisecond figures seen here.

---

## 16. What the Frontend Must NOT Assume

The following **do not exist** in the backend and must not be used, stubbed, or
planned against:

- ❌ `GET /documents/{id}` — no document lookup.
- ❌ `GET /chunks/{id}` — no chunk/full-text lookup. Citation text cannot be fetched.
- ❌ Any streaming endpoint (`/query/stream`, SSE, WebSocket) — responses are
  single-shot JSON.
- ❌ Any authentication endpoint (login/register/token) — the API is unauthenticated.
- ❌ Chat history / conversation endpoints — the API is stateless and single-turn.
- ❌ Feedback / rating endpoints.
- ❌ Pagination parameters — there is no paginated collection endpoint.
- ❌ Job/async endpoints (`/query/async`, task IDs, polling URLs).
- ❌ Any request field other than `question` (no `top_k`, `model`, `stream`,
  `session_id`, `filters`, `court`, `date_range`, …). Extra fields are rejected `422`.
- ❌ Any response field not listed in §5 — do not invent `id`, `sources`, `confidence`,
  `chunks`, `session_id`, or `request_id`.
- ❌ Admin/config endpoints exposing backend settings.

Only `/health`, `/ready`, `/query` (plus the convenience `/` and the docs endpoints)
exist.

---

## 17. API Quick Reference

| Method | Endpoint | Purpose | Success | Request | Response |
| --- | --- | --- | --- | --- | --- |
| `GET` | `/health` | Liveness (no models loaded) | `200` | — | `{ status, environment }` |
| `GET` | `/ready` | Readiness (validates pipeline) | `200` / `503` | — | `{ status: "ready", environment }` / `{ detail }` |
| `POST` | `/query` | Legal RAG query (full cascade) | `200` | `{ "question": string }` | `QueryResponse` (see §5) |
| `GET` | `/` | Service identity | `200` | — | `{ name, version, environment, *_url }` |
| `GET` | `/docs` | Swagger UI | `200` | — | HTML |
| `GET` | `/openapi.json` | OpenAPI schema | `200` | — | JSON |

Error codes for `POST /query`: `422` invalid request · `500` retrieval/reranking
failure · `502` generation failure · `503` not ready.

Latency expectation: ~7–10 s typical total (retrieval ~2–4 s, rerank ~1–1.5 s,
generation several seconds), with ~30 s cold start after a deploy. Use a 120 s client
timeout and a visible loading state.

---

## Appendix — Fields the frontend may safely rely on

- Always present in a `200` response: `answer`, `citations`,
  `retrieved_chunk_ids`, `retrieve_ms`, `rerank_ms`, `generate_ms`, `total_ms`,
  `status`.
- Always present in every citation: `chunk_id` (non-null, unique, opaque string of
  the form `<CNR>_<seq>_<YYYY-MM-DD>_<chunk_index>` in production).
- Always `null`-able: `cnr`, `court_code`, `decision_date`, `title`.
- `status` is `"ok"` for every current `200` response.
- `citations` ⊆ `retrieved_chunk_ids`, and `retrieved_chunk_ids` has at most 5
  entries on a normal request.
- All `*_ms` values are non-negative numbers in milliseconds.
