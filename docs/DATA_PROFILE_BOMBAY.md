# Bombay High Court Data Profile

Measured on 2 November 2026 from `overthelex/indian-court-decisions`, using all three splits.
Reproduce with `python scripts/profile_bombay_hc.py`. Raw output:
`results/data_profile/bombay_profile.json`.

The chunk and memory figures in the "Sizing" section are estimates. Everything else is measured.

## Coverage by bench

Bombay HC is `court_code = 27_1` with CNR prefix `HCBM`. The dataset's `bench` values are internal
codes, so they need readable names.

| `bench` | Bench | CNR prefix | Records | 5,000+ chars | Median chars |
|---|---|---|---:|---:|---:|
| `newas` | Bombay, Appellate Side (inferred) | HCBM01 | 446,709 | 89,004 | 2,266 |
| `hcaurdb` | Aurangabad | HCBM03 | 354,428 | 88,423 | 2,576 |
| `testcase` | Nagpur (424 of 424 sampled headers name Nagpur) | HCBM04 | 182,278 | 21,846 | 1,803 |
| `newos` | Bombay, Original Side (inferred) | HCBM02 | 162,260 | 35,836 | 2,143 |
| `kolhcdb` | Kolhapur | HCBM07 | 80,710 | 16,947 | 2,404 |
| `hcbgoa` | Goa | HCBM05 | 36,804 | 14,018 | 3,211 |
| **Total** | | | **1,263,189** | **266,074** | |

Most records are short orders:

- 997,115 records (78.9%) have under 5,000 characters, but hold only 32.6% of all text.
- 556,587 records (44.1%) have under 2,000 characters.

A short order can still matter, so the 5,000-character cut must be checked by reading a sample
(plan task P3). All records stay available through case lookup.

## Coverage by year

Periods use the date in the CNR.

| Period | Records | 5,000+ chars | Characters in 5,000+ records |
|---|---:|---:|---:|
| Before 2010 | 209,937 | 29,270 | 0.41 billion |
| 2010–14 | 312,430 | 55,099 | 0.74 billion |
| 2015–19 | 367,988 | 83,779 | 1.62 billion |
| 2020–23 | 249,825 | 63,752 | 1.19 billion |
| 2024 onward | 123,009 | 34,174 | 0.50 billion |

The latest Bombay `decision_date` is 2026-05-05.

## Supreme Court

| Records | 5,000+ chars | Characters in 5,000+ records | Years |
|---:|---:|---:|---|
| 50,053 | 46,321 | 1.20 billion | 1950–2024 |

There is no Supreme Court data after 2024, including the 2026 ruling on AI citations. A lawyer must
be told about this gap.

## Data quality findings

| Finding | Measured | Action |
|---|---|---|
| CNR values | 1,263,189 unique, no duplicate IDs | Safe as primary key |
| `court_name` | Blank on all Bombay records | Add our own court and bench names |
| `case_type` | Blank on 640,240 (50.7%) | Parse from title, for example `WP/1519/2000` |
| `judge` | Blank on 1,208 (0.1%) | Show "not recorded" |
| Outcome | Blank or `other` on 206,031 (16.3%) | Do not present outcome statistics as complete |
| Dates | `decision_date` differs from the CNR date on 24,933 (2.0%), and the year differs on 8,911 (0.7%) | Check these against the judgment text, and choose a rule |
| Impossible dates | The latest date in a CNR is the year 3021 | Reject impossible dates |
| Repeated text | 54.1% of sampled texts repeat a 4-word phrase 3+ times in the first 3,000 characters | Remove extraction repeats such as `CORAM ... CORAM ... CORAM` |
| Near duplicates | 0.3% of sampled texts | Remove them, keeping all linked CNRs |
| Non-English text | 0.0% of sampled texts above 20% non-ASCII | Measure OCR-garbled English separately |

Sample: 3,121 Bombay rows from the test split, selected by CNR hash. The repeated text figure is an
upper limit, because ordinary legal phrases can also repeat.

Sampling lesson: the first five rows of a filtered query all showed date mismatches, which suggested
a large problem. The full count showed 2.0%. Use full counts or random samples, not the first rows.

## Fields the data does not have

- No law report citations (SCC, AIR). These need extracting from judgment text.
- No link to the official judgment. We can show stored text and the CNR.
- No overruled or treatment labels, so the system cannot say a case is still good law.

## Correction to the v1 report

The v1 report labelled court codes incorrectly. Checking codes against CNR prefixes shows:

| Code | v1 report said | CNR prefix | Actual court |
|---|---|---|---|
| `10_8` | Bombay | BRHC | Patna |
| `18_6` | Allahabad / Lucknow | GAHC | Gauhati |
| `1_12` | Madras | JKHC | Jammu & Kashmir |
| `22_18` | Madhya Pradesh | CGHC | Chhattisgarh |
| `16_20` | Delhi | TRHC | Tripura |
| `14_25` | Kerala | MNHC | Manipur |
| `19_16` | Calcutta | WBCH | Calcutta (correct) |
| `24_17` | Gujarat | GJHC | Gujarat (correct) |

So the "19,679 Bombay judgments" in the v1 report were Patna judgments. Counting
`legal_chunks.parquet` by CNR prefix shows v1 holds only **1,037 Bombay judgments** (6,030 chunks),
about 1% of v1 and under 0.1% of all Bombay records. Delhi has 63, Karnataka 7 and Allahabad 2. v1's
search quality therefore says little about Bombay law. See `results/ram_baseline.json`.

## Sizing (estimates)

These assume about 1,000 new characters per chunk with the current 1,200 size and 200 overlap.

| Corpus | Judgments | Estimated chunks | Flat FAISS | SQ8 | PQ96 |
|---|---:|---:|---:|---:|---:|
| Bombay 5,000+ and Supreme Court 5,000+ | 312,395 | 5.7M | 17.5 GB | 4.4 GB | 0.6 GB |
| Bombay 2015 onward 5,000+ and Supreme Court 5,000+ | 228,026 | 4.5M | 13.8 GB | 3.5 GB | 0.5 GB |

PQ recall loss on this data is unknown until measured in plan task P5.
