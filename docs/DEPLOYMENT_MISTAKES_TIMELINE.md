# LegalRAG Deployment Mistakes & Engineering Timeline

Chronological engineering postmortem of the LegalRAG backend production/deployment effort — from the
start of the backend API work through the current Azure Container Apps ephemeral-storage limitation on
20 September 2026.

This document records **what we attempted**, **what failed**, **why it failed**, **what we discovered**,
**what we changed**, **what was validated**, and **what remains unresolved**. It exists as a debugging
and deployment reference, so technical details (sizes, digests, status codes, config) are preserved
deliberately.

This is **not** a description of project-wide failure. It is a mix of real code bugs, a real
deployment/runtime bug, and an infrastructure capacity limitation that is still open.

---

## Current Status

**As of 20 September 2026**

- Backend API is implemented on branch `feat/backend-api`, with the full target architecture wired:
  `Frontend → FastAPI → BM25 + Dense retrieval → RRF → CrossEncoder reranking → Groq generation → answer + citations + latency`.
- P1 production-hardening work is implemented and documented.
- Local test suite: **66 tests, OK** — all stubs/mocks, no production models, no artifact downloads, no Groq calls.
- Docker image builds and is pushed to Azure Container Registry `legalragregistry` as
  `legalragregistry.azurecr.io/legalrag:latest` (deployed by immutable digest).
- Azure Container App `legalrag-api` is deployed (region `centralindia`, Consumption workload profile,
  4 CPU / 8 GiB memory / 8 GiB ephemeral storage, external ingress on port `7860`).
- `/health` returns **200** (liveness).
- Automatic startup artifact provisioning now **works** from the Hugging Face **Dataset** repository
  `siddhu23/LegalRag_Dataset`.
- **Blocker:** `/ready` remains **503** because `dense.index` (~1652.98 MB) cannot be written — the
  container runs out of ephemeral storage during download.
- `/query` has **not** been exercised against the full production pipeline yet.

**Bottom line:** the code bugs are fixed and validated locally; the remaining blocker is an
infrastructure/storage-capacity problem, not a retrieval, Groq, Docker, or provisioning-code problem.

---

## Project Context

- LegalRAG is the production-oriented RAG system built around **100,000 Indian High Court judgments**.
- Final retrieval artifacts:
  - `bm25.pkl`
  - `dense.index`
  - `legal_chunks.parquet`
- Approximate artifact sizes:
  - `bm25.pkl`: ~552 MB
  - `dense.index`: ~1.65 GB
  - `legal_chunks.parquet`: ~249 MB
  - **total: ~2.45 GB**
- Retrieval architecture: **BM25 top-50 + Dense top-50 → RRF (k=60) → CrossEncoder reranker → top-5 context → LLM**.
- Production LLM provider: **Groq**; currently configured model: **`openai/gpt-oss-120b`**.
- Production artifact repository: **`siddhu23/LegalRag_Dataset`** — this is a Hugging Face **DATASET** repository.
- Production artifacts are intentionally **NOT baked into the Docker image**; they are intended to be
  provisioned at runtime from Hugging Face.
- The local machine has roughly **6 GB RAM**, so production models/artifacts were deliberately never
  loaded locally. Local development uses `local_stub`/mocks.

---

## Timeline

### 19 September 2026 — Phase 1: Backend API development

**What happened**

Backend API work started on branch `feat/backend-api` against the target architecture
`Frontend → FastAPI → BM25 + Dense retrieval → RRF → CrossEncoder reranking → Groq generation →
answer + citations + latency`.

Production artifacts were deliberately kept out of the Docker image and left to runtime provisioning.
Because the local machine has ~6 GB RAM, production models/artifacts were never loaded locally; local
development relied on `local_stub`/mocks.

**Result**

A working backend API existed locally, but it had not yet been hardened for production.

**Lesson**

Designing for "artifacts provisioned at runtime, never baked in" is correct for a ~2.45 GB artifact set —
but it makes startup provisioning and readiness part of the critical path, so they must be treated as
first-class production behaviour rather than an afterthought.

---

### 19 September 2026 — Phase 2: Senior backend review identifies production issues

**What happened**

A senior backend review flagged a set of production-blocking issues, including:

1. Docker defaulted to `ENVIRONMENT=local_stub`.
2. `/health` reported healthy even when the production pipeline was unavailable.
3. No automatic production artifact provisioning existed.
4. `/query` exposed server settings in the request schema.
5. Production could silently return stub responses.
6. Groq needed timeout/retry hardening.
7. Citation parsing had a mismatch with the context headers.
8. `court_name` did not exist in `legal_chunks.parquet`.
9. Error responses used HTTP 200 for failures.
10. Pipeline singleton initialization was not thread-safe.
11. Docker had an unquoted `faiss-cpu>=...` shell/redirection issue.
12. `.env.example` still referenced obsolete Gemini configuration.

**Root cause**

The backend was built for correctness on a local stub pipeline and had not yet been reviewed through a
production lens (fail-closed startup, honest readiness, input hardening, error semantics, concurrency).

**Result**

These became the P1 hardening backlog. They are documented here as the issues that **led to** the P1
work — not as still-open problems.

**Lesson**

A stub-friendly default is dangerous in production. Any default that lets a container boot into a
non-production code path will eventually ship that path to production.

---

### 19 September 2026 — Phase 3: P1 hardening

**What happened**

P1 hardening was implemented, tightly scoped to the review findings.

**What we changed**

- Docker default changed to `ENVIRONMENT=production`.
- Production **fail-closed** behaviour implemented.
- `local_stub` only works when explicitly selected.
- `/health` became a **liveness** check.
- `/ready` added for production **readiness**.
- Production artifact provisioning added.
- `/query` now accepts **only** `{"question": "..."}`; extra server-configuration fields are rejected.
- Production cannot downgrade to stub.
- Proper HTTP semantics introduced:
  - `422` invalid request
  - `500` retrieval/reranking failure
  - `502` generation failure
  - `503` not ready
- Groq client received explicit timeout and bounded retry/backoff.
- Pipeline singleton initialization became thread-safe.
- Citation parser fixed to extract **only** the chunk ID.
- `court_code` replaced the incorrect `court_name` API usage.
- Docker `faiss-cpu` dependency shell/redirection issue fixed.
- Gemini configuration removed from the current backend configuration.
- Documentation updated.

**What was validated**

- P1 suite initially reached **63 tests**.
- After the artifact-downloader changes, the suite reached **66 tests**.
- Final reported local result: **66 tests, OK**.
- Tests used stubs/mocks; they did **not** load production models or download production artifacts.

**Lesson**

"Fail closed" plus "honest readiness" plus "no stub in production" are the three controls that turn a
demo API into a deployable service. Without them, a broken production pipeline can silently serve
fabricated stub answers.

---

### 20 September 2026 — Phase 4: Docker build and layer caching

**What happened**

Docker dependency installation initially took a very long time because Docker builds in its own isolated
environment and cannot reuse the host virtual environment.

**What we did (and did not do)**

- The local venv was **NOT** copied into the Docker image.
- No host Python packages, symlinks, or mechanisms that make the image depend on the local machine.

**What we changed**

Reorganized the Dockerfile so dependency installation sits in a cacheable layer:

```text
COPY requirements.txt
→ RUN pip install ...
→ COPY source code
```

This lets Docker cache the dependency-install layer whenever `requirements.txt` is unchanged, so
editing files under `src/` does not reinstall Python dependencies.

**Result**

- The **first** full dependency build still had to download dependencies such as PyTorch — that cost is
  unavoidable on a cold cache.
- After the successful build, the image was pushed to Azure Container Registry.
  - ACR: `legalragregistry`
  - Image: `legalragregistry.azurecr.io/legalrag:latest`

**Lesson**

Copy dependency manifests before source, and never try to shortcut a container build by reusing the host
venv. The cache boundary is `requirements.txt`, not the source tree.

---

### 20 September 2026 — Phase 5: Azure deployment

**What happened**

Azure resources were provisioned:

| Resource | Value |
| --- | --- |
| Resource group | `legalrag-rg` |
| ACR | `legalragregistry` |
| Container Apps environment | `legalrag-env` |
| Container App | `legalrag-api` |
| Region | `centralindia` |
| Workload profile | Consumption |
| CPU | 4 |
| Memory | 8 GiB |
| Ephemeral storage | 8 GiB |
| GPU | OFF |
| Ingress | External HTTP |
| Port | 7860 |

Public domain: `redsmoke-65171c89.centralindia.azurecontainerapps.io`

**What failed**

The container environment initially failed to start because `ENVIRONMENT=production` had a **trailing space**.

**Root cause**

An invisible whitespace character in the environment value — the platform received a value that did not
exactly match `production`.

**What we changed**

Corrected the env var value.

**Result**

- After correction, `/health` returned **200**.

**Lesson**

Environment values are exact strings. A trailing space is enough to break a fail-closed startup path, and
it is invisible in most dashboards.

---

### 20 September 2026 — Phase 6: Old image / revision issue (`/ready` returned 404)

**What happened**

After the first P1 deployment, `/health` worked but `/ready` returned:

```text
404 Not Found
```

**Root cause**

The running revision was still serving the **older application image**. The deployment referenced the
mutable tag `legalragregistry.azurecr.io/legalrag:latest`, which can point at a stale or unexpected image.

**What we changed**

Deployed by **immutable image digest** instead of a mutable tag, to guarantee the exact intended image.

Earlier image digest deployed:

```text
sha256:c6134756f022b0337ca7ff964c225724e8312854f8f48d8e3889b0adfc1fc7fc
```

**Result**

The updated deployment then exposed `/ready`.

**Lesson**

`latest` is a moving target. Deploy by digest when you need to be certain which image is running.

---

### 20 September 2026 — Phase 7: Artifact provisioning bug (missing artifacts)

**What happened**

The first production `/ready` attempt reported:

- `bm25.pkl` missing
- `dense.index` missing
- `legal_chunks.parquet` missing

Before changing code, the startup execution path was traced. Investigation confirmed that **startup
provisioning WAS being called before pipeline initialization** — so the ordering was correct and the
failure was elsewhere.

**Root cause**

`scripts/download_artifacts.py` called `hf_hub_download()` **without `repo_type`**.

The repository `siddhu23/LegalRag_Dataset` is a Hugging Face **DATASET** repository, but the default for
`hf_hub_download()` is a **model** repository. Hugging Face therefore queried the model-repository API,
where the artifact files do not exist — so every artifact appeared "missing".

**What we changed**

Added to the existing `hf_hub_download()` call:

```python
repo_type="dataset"
```

Nothing else changed: artifact filenames, target-directory behaviour, `HF_REPO_ID`, `HF_TOKEN` handling,
provisioning architecture, production pipeline, retrieval logic, Docker architecture, and Groq
configuration were all left intact.

**What was validated**

- Tests increased from **63 to 66**; all passed.
- New tests assert that `hf_hub_download()` is called with `repo_type="dataset"` for every artifact.
- Token handling confirmed secure: the token is passed as a keyword argument (never on the command
  line/argv) and is never logged.
- No production artifacts were downloaded locally, no Groq calls were made, and no real
  embedding/reranker models were loaded.
- Documentation review found no place describing the artifact repository as a *model* repository, so no
  wording correction was required.

A new image was built and pushed.

New image digest:

```text
sha256:6281c9acb3d38f7aa7a36a80e7046387fa61a7434c78d82c1d1c10ee00906be2
```

Azure was updated to this immutable digest.

**Result**

The dataset-repository fix resolved the provisioning bug.

**Lesson**

Hugging Face hosts models, datasets, and spaces behind different APIs. A dataset repo needs an explicit
`repo_type="dataset"`; without it, you get "file not found" errors that look like a provisioning or
authentication failure but are actually an API-targeting mistake.

---

### 20 September 2026 — Phase 8: Current Azure storage limitation

**What happened**

After the dataset-repository fix, automatic provisioning finally worked. Azure successfully downloaded:

| Artifact | Result |
| --- | --- |
| `bm25.pkl` | downloaded, ~551.97 MB |
| `legal_chunks.parquet` | downloaded, ~248.58 MB |
| `dense.index` | **FAILED** — ~1652.98 MB |

`dense.index` failed with:

```text
No space left on device
```

Hugging Face reported only **~296.69 MB free** at the download location.

**Root cause**

The Container App has:

- CPU: 4
- Memory: 8 GiB
- Ephemeral storage: **8 GiB**
- Workload profile: Consumption

The artifact files together are approximately **2.45 GB**, and Hugging Face/Xet downloading requires
**additional temporary disk space during reconstruction**. With the current ephemeral filesystem, the
large `dense.index` cannot complete.

The current blocker is therefore **NOT**:

- Hugging Face authentication
- HF repository type
- provisioning code
- Docker image
- Groq
- retrieval logic

It is Azure Container Apps **ephemeral storage capacity**.

**Result / current state**

- Automatic provisioning works.
- Two of three artifacts downloaded successfully.
- `dense.index` cannot fit with the available temporary filesystem space.
- `/ready` therefore remains **503**.
- `/query` has **NOT** been tested with the production pipeline yet.

**Lesson**

"8 GiB ephemeral storage" is not 8 GiB of usable artifact space — download staging/reconstruction
overhead for a ~1.65 GB index can exhaust the remaining headroom. Artifact size plus download overhead,
not artifact size alone, must fit.

---

### 20 September 2026 — Phase 9: Current Azure storage investigation

**What happened**

Azure checks were run to characterize the capacity problem.

Container resources returned:

```json
{
  "cpu": 4,
  "ephemeralStorage": "8Gi",
  "memory": "8Gi"
}
```

Environment workload profile: **Consumption**.

`az storage account list --resource-group legalrag-rg` returned **no rows** — there is currently **no
storage account** in resource group `legalrag-rg`.

Container Apps environment domain: `redsmoke-65171c89.centralindia.azurecontainerapps.io`.

**Result**

The capacity problem is confirmed as an infrastructure/provisioning gap: there is no persistent storage
attached to the environment.

**Lesson**

Multi-gigabyte retrieval artifacts are a poor fit for ephemeral container storage. Persistent mounted
storage (e.g. Azure Files) is the expected direction — but note that this has **not** been implemented.

---

## Root-Cause Lessons

- **Defaults decide production behaviour.** `ENVIRONMENT=local_stub` as a Docker default meant production
  could silently serve stub answers. Fail-closed defaults fixed that.
- **Honest health/readiness is non-optional.** `/health` reporting healthy while the production pipeline
  was unavailable was actively misleading. Splitting liveness (`/health`) from readiness (`/ready`) fixed it.
- **Never trust a mutable tag.** `:latest` caused a stale revision to be served; deploying by immutable
  digest revealed the truth.
- **Invisible characters break fail-closed paths.** A trailing space in `ENVIRONMENT=production` caused an
  initial container startup failure.
- **Target the right Hugging Face API.** A dataset repo without `repo_type="dataset"` produced
  "missing artifacts" that looked like a provisioning bug but were an API-targeting bug.
- **Cache the dependency layer, not the source layer.** `COPY requirements.txt` → `pip install` →
  `COPY src/` makes `src/` edits cheap; the cold-cache first build is still expensive.
- **Validate deployment boundaries with mocks.** 66 local tests with stubs/mocks caught code bugs without
  ever loading production models or calling Groq.
- **Distinguish code bugs from infrastructure limits.** Phases 2–7 were code/config/deployment bugs and are
  fixed; Phase 8–9 is a storage-capacity limitation and is still open.

---

## Current Blocker

**Azure Container Apps ephemeral storage is too small for `dense.index` provisioning.**

- `bm25.pkl` (~551.97 MB) and `legal_chunks.parquet` (~248.58 MB) download successfully.
- `dense.index` (~1652.98 MB) fails with `No space left on device` (only ~296.69 MB free reported at the
  download location).
- Total artifacts (~2.45 GB) plus Hugging Face/Xet reconstruction overhead exceed the container's 8 GiB
  ephemeral filesystem.
- Consequence: `/ready` stays **503**, and `/query` has not yet been validated against the production pipeline.

---

## Next Planned Step

Investigate **Azure Files / persistent storage** for the large retrieval artifacts.

**Status:** not implemented. No storage account currently exists in `legalrag-rg`. The intended production
behaviour remains that the container **automatically downloads** the three artifacts from the Hugging Face
Dataset repository at startup — a manual Azure artifact-upload workaround is **not** the plan.

