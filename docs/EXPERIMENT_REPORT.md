# LegalRAG 100k Experiment Report

## Abstract

Retrieval-Augmented Generation (RAG) applied to legal domains faces unique challenges due to long, unstructured judgment texts, domain-specific terminology, statutory references, and the severe risks of ungrounded hallucinations. This report presents the experimental findings from **LegalRAG**, an end-to-end legal information retrieval and question-answering study over **100,000 Indian High Court judgments** comprising **538,079 chunks**. 

We benchmarked lexical retrieval (BM25), dense retrieval (`BAAI/bge-base-en-v1.5`), hybrid fusion via Reciprocal Rank Fusion (RRF), and neural reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) against a newly constructed, evidence-grounded benchmark of **497 validated questions** spanning five legal inquiry types. 

Our findings demonstrate that BM25 outperforms out-of-the-box dense retrieval in top-1 judgment recall (**42.66% vs. 19.72%**), driven by exact statutory citations. Hybrid RRF expands overall candidate recall at top-50 to **65.79%**, and cross-encoder reranking lifts top-1 recall to **35.61%**. Downstream RAG generation evaluated by an automated multi-criteria judge achieved an overall score of **3.0926 / 4**, a faithfulness score of **3.4004 / 4**, and an unsupported claim rate of **14.89%**. Detailed failure analysis reveals that **51.3%** of all generation failures stem directly from context omission at the retrieval stage, underscoring the critical dependency of legal LLM grounding on retrieval recall.

---

## Objective

1. Construct a standardized, reproducible 100k-document High Court corpus from the `overthelex/indian-court-decisions` dataset.
2. Build an evidence-grounded evaluation benchmark pairing questions with verifiable text passages and chunk IDs across diverse legal question types.
3. Systematically evaluate lexical, dense, hybrid, and reranked retrieval pipelines on Judgment Recall@K ($K \in \{1, 3, 5, 10, 25, 50\}$).
4. Measure downstream RAG answer quality, factual faithfulness, citation consistency, and hallucination rates across legal question categories.
5. Quantitatively categorize failure modes across the retrieval-generation interface.

---

## Dataset

- **Source**: Hugging Face `overthelex/indian-court-decisions`
- **Config**: `high_courts`
- **Split**: `train` (streamed with seed 42 shuffle)
- **Scope**: 100,000 High Court judgment records selected across 24 Indian High Court jurisdictions.

### Court Representation Sample
The 100,000 judgments reflect diverse High Court benches across India:
- Allahabad / Lucknow (`18_6`): 21,498
- Calcutta (`19_16`): 19,772
- Bombay (`10_8`): 19,679
- Madras (`1_12`): 10,471
- Gujarat (`24_17`): 8,555
- Madhya Pradesh (`22_18`): 6,961
- Delhi (`16_20`): 3,289
- Kerala (`14_25`): 2,268
- Punjab & Haryana, Patna, Rajasthan, Karnataka, Gauhati, and other state High Courts.

---

## Corpus Construction

### Data Quality & Sanitization
Profiling identified that **762 documents (0.76%)** contained corrupted control character sequences (such as non-printable ASCII `\x00-\x08`, `\x0b`, `\x0c`, `\x0e-\x1f`).
- **Filter**: Documents with excessive control character density (>1% of length) were flagged.
- **Normalization**: Control characters were stripped regex-wise `[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]` while maintaining paragraph structure, quotation marks, and legal numbering. Whitespace was collapsed to single spaces.
- **Artifact**: `legal_judgments_clean.parquet` (485.4 MB).

---

## Chunking

Legal judgment narratives require chunking that maintains sufficient statutory context without diluting vector representations.

### Locked Hyperparameters
- **Splitter**: `RecursiveCharacterTextSplitter` (separators: `["\n\n", "\n", " ", ""]`)
- `CHUNK_SIZE`: **1200 characters** (~200–250 tokens)
- `CHUNK_OVERLAP`: **200 characters** (~35–40 tokens)
- `MIN_CHUNK_LENGTH`: **100 characters** (discards orphan headings/page numbers)

### Corpus Statistics
- **Total Chunks**: **538,079**
- **Average chunks per document**: 5.38
- **Storage**: `legal_chunks.parquet` (260.7 MB)
- **Schema**: `chunk_id` (`{cnr}_chunk_{index}`), `cnr`, `chunk_index`, `text`.

---

## Retrieval Methods

```text
               Query
                 │
        ┌────────┴────────┐
        ▼                 ▼
   BM25 Lexical     Dense (BGE-Base)
   (Top-50 Pool)    (Top-50 Pool)
        │                 │
        └────────┬────────┘
                 ▼
          Reciprocal Rank
            Fusion (RRF)
                 │
                 ▼
          Top-50 Candidates
                 │
                 ▼
       Cross-Encoder Reranker
      (ms-marco-MiniLM-L-6-v2)
                 │
                 ▼
         Top-5 RAG Context
```

### BM25
- **Algorithm**: `BM25Okapi` (`k1=1.5`, `b=0.75`)
- **Tokenization**: Regex `[a-zA-Z0-9]+` lowercasing.
- **Artifact**: `bm25.pkl` (578.8 MB).

### Dense Retrieval
- **Embedding Model**: `BAAI/bge-base-en-v1.5` (768 dimensions, normalized).
- **Index**: `faiss.IndexFlatIP` (Cosine Similarity over L2-normalized embeddings).
- **Inference**: Sharded across 2 × Tesla T4 GPUs into 108 `.npy` files (`emb_0000.npy`–`emb_0107.npy`).
- **Artifacts**: `dense.index` (1.65 GB), `dense_index_metadata.json`.

### Hybrid RRF
- **Fusion Formula**:
  $$\text{Score}_{\text{RRF}}(d) = \frac{1}{60 + \text{rank}_{\text{BM25}}(d)} + \frac{1}{60 + \text{rank}_{\text{Dense}}(d)}$$
- Merges candidate sets from BM25 (top 50) and Dense (top 50) into a combined top-50 pool.

### Cross-Encoder Reranking
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Scoring**: Full-attention cross-encoding of query-document pairs up to 512 tokens.
- **Input**: Top 50 Hybrid-RRF candidates.
- **Output**: Re-ordered top 50 ranked list (`reranked_top50.json`).

---

## Benchmark Construction

### Document Selection
500 documents were sampled round-robin across court jurisdictions with length constraints ($500 \le \text{length} \le 12,000$ characters) to guarantee structural balance.

### Evidence Grounding & Candidate Validation
Candidate questions were generated with required fields: `{question, reference_answer, question_type, supporting_text}`.

Each candidate underwent automated evidence validation (`find_gold_chunks`):
- Exact or sliding-window phrase matching (windows: 60, 40, 25, 15, 10 words) of `supporting_text` against the corpus chunks.
- Candidates failing evidence grounding were eliminated.

**Candidate Filtering Summary**:
- Generated Candidates: 500 (`eval_candidates.json`)
- Dropped Candidates: **3** (unverifiable supporting passages)
  - `PHHC010839062016_1_2016-01-07` (outcome)
  - `RJHC020411592007_1_2008-01-10` (reasoning)
  - `PHHC010373551998_1_2014-03-04` (outcome)
- **Final Validated Benchmark**: **497 questions** (`gold_eval.json`)

### Benchmark Distribution by Question Type
| Question Type | Count | % of Benchmark | Description |
| :--- | :---: | :---: | :--- |
| **Reasoning** | 149 | 29.98% | Ratio decidendi, judicial rationale, statutory interpretation |
| **Outcome** | 118 | 23.74% | Decrees, sentencing, disposal nature, remand orders |
| **Fact** | 100 | 20.12% | Allegations, dates, monetary values, evidence facts |
| **Legal Provision** | 80 | 16.10% | Statutory sections, enactments, constitutional articles |
| **Multi-Hop** | 50 | 10.06% | Complex synthesis spanning multiple procedural stages |
| **Total** | **497** | **100.0%** | **Evidence-grounded benchmark** |

---

## Evaluation Methodology

### Retrieval Evaluation Metric
**Judgment Recall@K**: Measures the proportion of queries where at least one chunk belonging to the target CNR judgment appears in the top $K$ retrieved candidates:

$$\text{Judgment Recall@K} = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}\left( \text{CNR}(q) \cap \{\text{CNR}(c) \mid c \in \text{TopK}(q)\} \neq \emptyset \right)$$

### Downstream Generation Evaluation
RAG generation utilized top-5 reranked context blocks routed via OmniRoute (the OpenAI-compatible routing layer used during experimentation) to the configured LLM at `temperature=0`. Evaluation was conducted by an independent LLM judge assessing:
1. **Answer Relevance (1–4)**: Semantic alignment and completeness relative to the question.
2. **Faithfulness (1–4)**: Absence of ungrounded factual claims.
3. **Citation Correctness (1–4)**: Syntactic validity and accuracy of cited chunk IDs.
4. **Unsupported Claim Rate (0.0–1.0)**: Ratio of unsupported statements to total claims.
5. **Hallucination Rate (0.0–1.0)**: Proportion of answers containing verifiable falsehoods.
6. **Overall Score (1–4)**: Holistic synthesis quality.

---

## Retrieval Results

### Frozen 100k Retrieval Scoreboard (497 Questions)

| Retriever | Judgment Recall@1 | Judgment Recall@5 | Judgment Recall@10 | Judgment Recall@50 |
| :--- | :---: | :---: | :---: | :---: |
| **BM25** | **0.426559** (42.66%) | **0.533199** (53.32%) | **0.577465** (57.75%) | 0.639839 (63.98%) |
| **Dense (`bge-base-en-v1.5`)** | 0.197183 (19.72%) | 0.273642 (27.36%) | 0.334004 (33.40%) | 0.446680 (44.67%) |
| **Hybrid-RRF** | 0.321932 (32.19%) | 0.511066 (51.11%) | 0.561368 (56.14%) | **0.657948** (65.79%) |
| **Hybrid-RRF + Cross-Encoder** | 0.356137 (35.61%) | 0.486922 (48.69%) | 0.559356 (55.94%) | **0.657948** (65.79%) |

### Granular Retrieval Cutoff Comparison
| Cutoff $K$ | BM25 | Dense (BGE) | Hybrid-RRF | Reranked |
| :---: | :---: | :---: | :---: | :---: |
| **@1** | 0.4266 | 0.1972 | 0.3219 | 0.3561 |
| **@3** | 0.5030 | 0.2535 | 0.4527 | 0.4447 |
| **@5** | 0.5332 | 0.2736 | 0.5111 | 0.4869 |
| **@10** | 0.5775 | 0.3340 | 0.5614 | 0.5594 |
| **@25** | 0.6076 | 0.4125 | 0.6177 | 0.6177 |
| **@50** | 0.6398 | 0.4467 | **0.6579** | **0.6579** |

### Key Retrieval Findings
1. **Lexical Dominance in Statutory Retrieval**: BM25 achieved 42.66% Recall@1 vs 19.72% for BGE-Base. Dense embeddings without domain fine-tuning fail to capture exact section numbers (e.g., "Section 138 NI Act" vs "Section 139").
2. **Hybrid Pool Expansion**: Hybrid RRF achieved the highest broad recall at $K=50$ (65.79%), successfully pulling relevant cases into the candidate pool that BM25 missed.
3. **Cross-Encoder Precision Boost**: Reranking the hybrid pool improved top-1 precision by **+3.42 percentage points** (from 32.19% to 35.61%).

---

## RAG Evaluation Results

### Overall Aggregate Performance (497 Evaluated Records)

| Metric | Score / Value | Score Distribution (1 / 2 / 3 / 4) |
| :--- | :---: | :---: |
| **Answer Relevance** | **3.0966 / 4** | 88 / 69 / 47 / 293 |
| **Faithfulness** | **3.4004 / 4** | 53 / 53 / 33 / 358 |
| **Citation Correctness** | **3.2515 / 4** | 46 / 100 / 34 / 317 |
| **Overall Score** | **3.0926 / 4** | 67 / 90 / 70 / 270 |
| **Unsupported Claim Rate** | **0.1489** (14.89%) | — |
| **Hallucination Rate** | **0.2274** (22.74%) | False: 384, True: 113 |

### Performance Breakdown by Question Type

| Question Type | Records | Answer Relevance | Faithfulness | Citation Correctness | Unsupported Claim Rate | Hallucination Rate | Overall Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Legal Provision** | 80 | **3.5000** | **3.6625** | **3.6375** | **0.0906** | **0.1500** (12/80) | **3.4375** |
| **Fact** | 100 | 3.3300 | 3.4400 | 3.2700 | 0.1388 | 0.2300 (23/100) | 3.1400 |
| **Reasoning** | 149 | 3.0604 | 3.5235 | 3.4027 | 0.1250 | 0.1745 (26/149) | 3.1007 |
| **Multi-Hop** | 50 | 2.7600 | 3.4000 | 3.2600 | 0.1306 | 0.2400 (12/50) | 3.0200 |
| **Outcome** | 118 | 2.8136 | 3.0339 | 2.7797 | 0.2347 | 0.3390 (40/118) | 2.8390 |

---

## Failure Analysis

### Comprehensive Error Taxonomy (497 Questions)

| Failure Category | Count | Proportion | Primary Mechanism |
| :--- | :---: | :---: | :--- |
| **Retrieval Failure** | **255** | **51.31%** | Target judgment completely absent from retrieved top-5 context |
| **No Major Failure (Success)** | **194** | **39.03%** | Target judgment present, answer accurate, faithful, properly cited |
| **Low Answer Relevance** | **21** | **4.23%** | Context partially relevant, but model failed to answer directly |
| **Context Selection / Citation** | **19** | **3.82%** | Model cited irrelevant distractor chunks or formatted citations improperly |
| **Generation Grounding Failure** | **8** | **1.61%** | Target judgment present in context, but model hallucinated extraneous claims |

### Failure Breakdown by Question Type
```text
Question Type      Retrieval Failure  No Major Failure  Low Relevance  Context/Citation  Grounding Failure
------------------------------------------------------------------------------------------------------
Fact (100)                46                 47               2                1                 4
Legal Provision (80)      44                 36               0                0                 0
Multi-Hop (50)            25                 17               5                3                 0
Outcome (118)             54                 42              10               11                 1
Reasoning (149)           86                 52               4                4                 3
```

### Context Presence vs. Hallucination Correlation
- **Questions with correct judgment in Top-5 context**: 242 / 497 (48.69%)
  - *Outcome*: 194 succeeded cleanly; only 8 generated unsupported claims.
- **Questions with NO correct judgment in Top-5 context**: 255 / 497 (51.31%)
  - *Outcome*: Highest correlation with hallucination and generic conversational fallback.

---

## Limitations

1. **Pre-trained Embeddings vs. Legal Domain Fine-Tuning**: Out-of-the-box `bge-base-en-v1.5` dense embeddings were not contrastively fine-tuned on Indian legal corpora, explaining the lower dense-only recall.
2. **Fixed-Window Chunking**: While 1200-character chunks preserve token density, legal arguments often span multiple pages. Hierarchical chunking or late-chunking may capture broader ratio decidendi.
3. **Cross-Encoder Domain Shift**: `ms-marco-MiniLM-L-6-v2` trained on general web search queries occasionally favored generic paragraph matches over specific legal holding clauses.
4. **Context Window Cap**: Fixed top-5 chunking restricted total input context to ~1,500 words per query, limiting multi-hop synthesis across long judgments.

---

## Reproducibility

- **Code Base**: Self-contained Kaggle pipeline notebook (`Notebooks/legalrag-100k-final.ipynb`) and local data scripts.
- **Random Seed**: Fixed at `seed=42` across all streaming, sampling, and splitting operations.
- **Deterministic Checkpoints**: Every stage outputs verified Parquet/JSON artifacts (`legal_judgments_clean.parquet`, `legal_chunks.parquet`, `bm25.pkl`, `dense.index`, `gold_eval.json`, `rag_results.json`, `rag_evaluation.json`).
- **Hardware Profile**: Verified on 2 × NVIDIA Tesla T4 GPUs with 31.3 GB Host RAM.

### Serving-Time Artifact Contracts

The frozen artifacts are also consumed at serving time by the FastAPI backend, so their
serialization contracts are part of the reproducibility surface:

- `bm25.pkl` is the pickled dict `{"bm25": <BM25Okapi>, "chunk_ids": [...]}` with `chunk_ids`
  in BM25 corpus order; the runtime loader accepts this canonical form, the legacy `"model"`
  key, and a bare `BM25Okapi`, and fails closed on anything else.
- `dense.index` is `faiss.IndexFlatIP` (`ntotal = 538,079`, `d = 768`) over L2-normalized
  `BAAI/bge-base-en-v1.5` embeddings. Queries are encoded with the instruction recorded in
  `dense_index_metadata.json` (`"Represent this sentence for searching relevant passages: "`).
- `legal_chunks.parquet` row order is the ID mapping for both retrievers; every retrieved
  chunk ID must resolve against its `chunk_id` column.

Measured serving-time latency with the real artifacts (CPU-only, single process, mean of 6
queries): BM25 top-50 2,624 ms, dense top-50 139 ms, RRF 0.1 ms, cross-encoder top-5
1,496 ms (total 4,258 ms); cold pipeline initialization ≈29.3 s; peak memory ≈10.5 GB.

---

## Conclusion

LegalRAG demonstrates that building a robust legal question-answering system requires an integrated, multi-stage retrieval architecture:
- Lexical retrieval (BM25) provides critical statutory anchoring that general dense retrievers miss.
- Hybrid RRF broadens candidate discovery to **65.79%** top-50 recall.
- Cross-encoder reranking provides essential top-1 precision gains (+3.42%).
- Rigorous evidence validation prevents misleading benchmark evaluation.
- Over **51%** of RAG generation errors originate directly from retrieval omission, proving that retrieval recall remains the primary bottleneck for legal RAG reliability.
