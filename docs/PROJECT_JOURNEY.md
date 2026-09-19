# Project Journey: Building LegalRAG

This document outlines the chronological engineering and research journey of **LegalRAG**—from initial problem formulation and dataset discovery to building the final 100k-document hybrid retrieval evaluation pipeline over Indian High Court judgments.

---

## 1. Problem Definition

Legal document retrieval and question answering present unique challenges that general-purpose RAG systems fail to address effectively:

1. **Domain-Specific Lexical Precision vs. Semantic Abstraction**: Legal inquiries often hinge on specific statutory provisions (e.g., Section 482 of CrPC, Section 138 of NI Act), procedural terms, Latin maxims, and exact case identifiers (CNR numbers, party names). Standard dense retrievers frequently map conceptually similar legal arguments to nearby embedding spaces while missing the exact statutory section or procedural threshold upon which the case turned.
2. **Document Length and Structural Density**: Indian court decisions are lengthy, unstructured, narrative texts comprising procedural histories, evidence summaries, arguments from both petitioners and respondents, statutory citations, and judicial rationales. Naive chunking breaks semantic cohesion and severs legal tests from their outcomes.
3. **Severe Cost of Hallucinations and Misattribution**: In legal applications, generating an authoritative-sounding legal argument grounded in the wrong judgment or fabricating a statutory interpretation carries severe operational risk. Evaluation cannot simply measure surface semantic similarity; it must measure evidence grounding, citation accuracy, and unsupported claim rates.

The objective of LegalRAG was defined: **Build an end-to-end, reproducible, evidence-grounded legal RAG pipeline over Indian High Court decisions and rigorously benchmark lexical, dense, hybrid, and reranked retrieval strategies coupled with multi-faceted RAG evaluation.**

---

## 2. Dataset Selection

To reflect realistic legal research across Indian jurisdictions, the project selected the Hugging Face dataset:
- **Repository**: `overthelex/indian-court-decisions`
- **Configuration**: `high_courts`
- **Split**: `train`

The source corpus spans millions of decisions across state High Courts. Because the complete dataset exceeds local memory and disk constraints, the project established a streaming ingestion strategy using Hugging Face datasets' `streaming=True` mode with a deterministic shuffle buffer (`buffer_size=10,000`, `seed=42`).

Key metadata attributes tracked per decision:
- `cnr`: Unique Case Number Record across Indian courts.
- `court_code` / `court_name`: High Court identifier (e.g., Punjab & Haryana, Delhi, Bombay, Calcutta).
- `decision_date` / `year`: Temporal judgment metadata.
- `full_text`: Raw unsegmented judgment transcript.
- `case_type`, `disposal_nature`, `judge`, `petitioner`, `respondent`.

---

## 3. Corpus Construction

### Exploratory Ingestion
Early experiments explored collection scripts (`scripts/download_subset.py` for 5k records and `scripts/collect_corpus.py` for 20k balanced records). An intermediate 80k experiment provided preliminary baseline indicators.

### Final 100k Corpus Ingestion
For the definitive experiment, the corpus was collected from scratch to assemble **100,000 unique High Court judgments**:
- **Deduplication**: Exact text deduplication maintained via an in-memory SHA/text hash set during streaming.
- **Court Representation**: Ingestion sampled across 24 High Court jurisdictions.
- **Persistence**: Saved in 20 chunked Parquet shards (`judgments_000.parquet` to `judgments_019.parquet`) totaling ~485 MB uncompressed.

### Quality Profiling & Normalization
Data profiling revealed that **762 documents (0.76%)** contained corrupted control character sequences (including null bytes `\x00-\x08`, form feeds, and vertical tabs). Document text was sanitized:
- Documents with severe corruption (>1% control characters) were flagged.
- Normalization stripped non-printable control characters `[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]` while preserving critical legal formatting and punctuation, consolidating irregular whitespace.
- Cleaned corpus saved as `legal_judgments_clean.parquet`.

---

## 4. Chunking Strategy

Legal texts require chunking that balances context granularity for dense retrieval against preserving sufficient procedural context for lexical matching and downstream LLM comprehension.

### Baseline Hyperparameters
The pipeline locked the chunking strategy using LangChain's `RecursiveCharacterTextSplitter`:
```python
CHUNK_SIZE = 1200          # Characters (~200–250 tokens)
CHUNK_OVERLAP = 200        # Characters (~35–40 tokens overlap)
MIN_CHUNK_LENGTH = 100     # Minimum character threshold to discard trivial fragments
```

### Resulting Chunk Corpus
- **Input**: 100,000 cleaned judgments
- **Output Chunks**: **538,079 chunks**
- **Storage**: `legal_chunks.parquet` (260.7 MB)
- **Metadata**: Each chunk carries `chunk_id` (format: `{cnr}_chunk_{index}`), `cnr`, `chunk_index`, and `text`.

---

## 5. Benchmark Construction

A major pitfall of RAG evaluation is using synthetic questions disconnected from verifiable source passages. LegalRAG implemented an evidence-grounded benchmark pipeline:

### Document Selection
- **500 evaluation documents** selected from the 100k corpus using length bounds ($500 \le \text{length} \le \text{MAX\_CHARS}$) and round-robin sampling across courts to guarantee jurisdiction diversity.
- Persisted to `evaluation_documents.parquet`.

### Question Taxonomy
The benchmark was designed to target five critical legal inquiry types:
1. **Fact (100 target)**: Specific factual allegations, dates, monetary amounts, procedural events.
2. **Reasoning (150 target)**: Judicial rationales, ratio decidendi, statutory interpretation logic.
3. **Legal Provision (80 target)**: Specific sections, statutory acts, procedural codes, constitutional articles.
4. **Outcome (120 target)**: Final decree, bail grant/dismissal, sentence reduction, remand directions.
5. **Multi-Hop (50 target)**: Inquiries requiring synthesized facts across multiple parts of the judgment.

### Candidate Generation
- A structured generation pipeline via OmniRoute created 500 candidate evaluation records (`eval_candidates.json`), each outputting `{question, reference_answer, question_type, supporting_text}`.

---

## 6. BM25 Retrieval

The first baseline established lexical retrieval capability across the 538,079 chunks:
- **Implementation**: `rank-bm25` (`BM25Okapi`).
- **Tokenization**: Regex-based alphanumeric tokenization with lowercasing.
- **Index**: Serialized to `bm25.pkl` (578.8 MB).
- **Candidate Pool**: Top 50 chunks retrieved per query.
- **100k Benchmark Performance**:
  - **Judgment Recall@1**: 0.4266 (42.66%)
  - **Judgment Recall@5**: 0.5332 (53.32%)
  - **Judgment Recall@10**: 0.5775 (57.75%)
  - **Judgment Recall@50**: 0.6398 (63.98%)

*Key observation*: BM25 proved exceptionally strong at finding specific CNR numbers, named statutes, and exact legal terminology.

---

## 7. Dense Retrieval

To evaluate semantic matching beyond literal keyword overlap, the project integrated dense vector retrieval:
- **Model**: `BAAI/bge-base-en-v1.5` (768-dimensional embeddings).
- **Multi-GPU Sharded Inference**: Checkpointed inference across 2 × NVIDIA Tesla T4 GPUs with batch size 64 and FP16 precision.
- **Shard Structure**: 108 embedding shards (`emb_0000.npy` to `emb_0107.npy`, 5,000 vectors per shard) totaling ~1.65 GB.
- **FAISS Indexing**: `faiss.IndexFlatIP` over L2-normalized vectors (equivalent to cosine similarity), serialized to `dense.index` (1.65 GB).
- **100k Benchmark Performance**:
  - **Judgment Recall@1**: 0.1972 (19.72%)
  - **Judgment Recall@5**: 0.2736 (27.36%)
  - **Judgment Recall@10**: 0.3340 (33.40%)
  - **Judgment Recall@50**: 0.4467 (44.67%)

*Key observation*: Out-of-the-box dense retrieval struggled on raw Indian legal texts compared to BM25 due to dense models smoothing over exact statutory citations and section numbers.

---

## 8. Hybrid RRF

To combine the keyword precision of BM25 with the semantic generalization of dense embeddings, the pipeline implemented **Reciprocal Rank Fusion (RRF)**:

$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{k + \text{rank}_m(d)}$$

where $k = 60$.

- **Candidate Pool**: Union of Top-50 BM25 and Top-50 Dense candidates merged and re-ranked into a combined Top-50 list (`hybrid_top50.json`).
- **100k Benchmark Performance**:
  - **Judgment Recall@1**: 0.3219 (32.19%)
  - **Judgment Recall@5**: 0.5111 (51.11%)
  - **Judgment Recall@10**: 0.5614 (56.14%)
  - **Judgment Recall@50**: 0.6579 (65.79%)

*Key observation*: Hybrid RRF achieved the highest broad recall across the entire pipeline at cutoff 50 (65.79%), successfully expanding the candidate pool.

---

## 9. Cross-Encoder Reranking

To refine candidate ordering within the top-50 hybrid pool before feeding context to the LLM:
- **Reranker Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Inference**: Evaluated query-passage pairs dynamically with max length 512.
- **Output**: Re-ordered top-50 results saved to `reranked_top50.json`.
- **100k Benchmark Performance**:
  - **Judgment Recall@1**: 0.3561 (35.61% — **+3.42% over Hybrid-RRF**)
  - **Judgment Recall@5**: 0.4869 (48.69%)
  - **Judgment Recall@10**: 0.5594 (55.94%)
  - **Judgment Recall@50**: 0.6579 (65.79%)

*Key observation*: Cross-encoder reranking boosted top-1 precision significantly (+3.42 percentage points), though full cross-encoder scoring on MS-MARCO weights slightly penalized cutoff@5 compared to pure lexical BM25 due to domain-shift on legal terminology.

---

## 10. RAG Context Construction

For each evaluation question, the **Top 5 reranked chunks** were formatted into structured context blocks:

```text
[Chunk ID: {chunk_id} | Court: {court_name} | Date: {decision_date}]
{chunk_text}
```

The system prompt strictly enforced:
- Synthesize answers solely from the provided text blocks.
- Explicitly cite source chunks using `[Chunk ID: ...]` syntax.
- If information is insufficient or contradictory, explicitly state that rather than extrapolating.

---

## 11. RAG Generation

RAG answer generation was executed over all benchmark questions:
- **Routing Infrastructure**: Routed through OmniRoute via an OpenAI-compatible API layer to the configured generation model at temperature 0.
- **Checkpointing**: Every generated response was immediately saved to disk (`rag_results.json`).
- **Inspection**: Sample responses verified grounding, formatting, and citation compliance before batch execution.

---

## 12. Evidence Validation

Before finalizing evaluation scores, the benchmark went through evidence validation:
- **Algorithm**: `find_gold_chunks` verified that the `supporting_text` extracted during question generation was verifiably present in the actual chunk text via exact substring or sliding-window phrase matching.
- **Candidate Filtering**: Out of 500 generated candidate questions, **3 candidates** could not be reliably anchored to their source chunks:
  - `PHHC010839062016_1_2016-01-07` (outcome)
  - `RJHC020411592007_1_2008-01-10` (reasoning)
  - `PHHC010373551998_1_2014-03-04` (outcome)
- **Final Locked Benchmark**: Exactly **497 validated gold evaluation questions** were finalized in `gold_eval.json`.

---

## 13. RAG Evaluation

A dedicated, isolated LLM judge evaluated all 497 generated answers across six core dimensions:
1. **Answer Relevance (1–4)**: Directness and completeness in addressing the legal question.
2. **Faithfulness (1–4)**: Adherence to provided context without ungrounded extrapolations.
3. **Citation Correctness (1–4)**: Accuracy of chunk and CNR citations.
4. **Unsupported Claim Rate (0.0–1.0)**: Proportion of factual claims lacking support in the retrieved chunks.
5. **Hallucination (Boolean)**: Presence of fabricated facts or contradictory conclusions.
6. **Overall Score (1–4)**: Composite generation quality.

Results were checkpointed to `rag_evaluation.json` and aggregated into `rag_metrics.json`.

---

## 14. Final 100k Experiment Results Summary

### Retrieval Scoreboard (497 Benchmark Questions)
| Strategy | Recall@1 | Recall@5 | Recall@10 | Recall@50 |
| :--- | :---: | :---: | :---: | :---: |
| **BM25** | **0.4266** | **0.5332** | **0.5775** | 0.6398 |
| **Dense (BGE-Base)** | 0.1972 | 0.2736 | 0.3340 | 0.4467 |
| **Hybrid-RRF** | 0.3219 | 0.5111 | 0.5614 | **0.6579** |
| **Hybrid-RRF + Cross-Encoder** | 0.3561 | 0.4869 | 0.5594 | **0.6579** |

### Downstream RAG Generation Metrics (497 Evaluations)
| Metric | 100k Final Result | Baseline Reference (Old 80k) |
| :--- | :---: | :---: |
| **Answer Relevance** | **3.0966 / 4** | 2.57 / 4 |
| **Faithfulness** | **3.4004 / 4** | 3.09 / 4 |
| **Citation Correctness** | **3.2515 / 4** | 2.97 / 4 |
| **Unsupported Claim Rate** | **0.1489 (14.89%)** | 0.259 (25.9%) |
| **Hallucination Rate** | **0.2274 (22.74%)** | 0.350 (35.0%) |
| **Overall Score** | **3.0926 / 4** | 2.56 / 4 |

---

## 15. Current Project State

The research, corpus engineering, retrieval modeling, and RAG evaluation phases are complete and verified against frozen artifacts.

### Completed:
- [x] 100k High Court corpus ingestion, cleaning, and Parquet serialization.
- [x] Baseline chunking into 538,079 chunks.
- [x] 497-question evidence-grounded evaluation benchmark.
- [x] Full retrieval experimentation (BM25, BGE-Base, Hybrid RRF, Cross-Encoder).
- [x] Checkpointed RAG generation and judge evaluation.
- [x] Comprehensive failure analysis and error taxonomy.

### Completed Engineering Milestones (Phase 2):
- [x] **FastAPI Backend (`src/legalrag/api/`)**: Built dual-mode (`local_stub` / `production`) service exposing `/health` and `/query` endpoints with singleton dependency injection and granular latency attribution (`retrieve_ms`, `rerank_ms`, `generate_ms`, `total_ms`).
- [x] **Production Exception Shielding & LLMOps**: Trapped all generation, rate-limit (429), and network errors to prevent raw exception leakage into answers.
- [x] **Citation Grounding Defense**: Enforced strict cross-referencing between extracted `[Chunk ID: ...]` citations and retrieved top-5 context chunks.
- [x] **Groq Generation Client Migration**: Implemented official `groq` SDK client integration with typed `GenerationResult` structures, token usage tracking, and graceful error shielding.
- [x] **Docker Containerization for Cloud Deployment**: Created non-root (UID 1000) Dockerfile adhering to port 7860 binding standards with automated Hub artifact download scripts for Azure Container Apps and Hugging Face Spaces.

### Next Roadmap (Phase 3 UI & Optimization):
- [ ] **Streamlit / Web UI**: Modern legal search, citation inspection, and court jurisdiction filtering dashboard.
- [ ] **Vector Quantization (IVFPQ / HNSW)**: Sub-50ms vector search for scale beyond 1M judgments.
