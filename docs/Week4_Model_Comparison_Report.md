# Week 4 — Model Comparison Report (CORRECTED, re-measured in Week 5)

> **Status.** The original Week 4 report quoted round numbers (HitRate@3 72 % / 84 % / 96 %, 12.5 / 22.3 / 45.6 ms, Citation accuracy 95 %) that cannot be
> reproduced from the code and data in the repository: the index used then did not contain the Placement Policy, Internship Policy, Academic Calendars or IPT
> guidelines (scanned PDFs, never OCR-ed), Citation accuracy 95 % is impossible with 25 queries (steps of 4 pp), and no run log exists. The original text is kept as
> `Week4_Model_Comparison_Report_ORIGINAL.md`. The numbers below were produced by `python src/compare_models.py` on the rebuilt corpus and are stored in
> `evaluation/model_comparison_results.json` together with all per-query records.

## 1. Setup
* Benchmark `evaluation/qa_benchmark.json` (25 queries, one ground-truth *file name* per query). 2 queries were dropped because their ground-truth document
  (course tutorial / formula sheet) is no longer in the corpus (study material is excluded as non-policy) → **n = 23**.
* Queries 1-15 are hand-written questions; queries 16-25 are 100-character snippets copied from the target chunk of a randomly chosen document
  (`create_benchmark.py`). Many of those snippets are boilerplate headers shared by dozens of course descriptions ("School of Technology Management and Engineering …"),
  so a single ground-truth file is not identifiable from them: they leak the source text *and* are ambiguous.
* Only retrieval is compared here (no LLM calls, no API quota); Citation accuracy / fallback rate / end-to-end latency of the LLM stage are measured in the Week 5
  evaluation (`docs/Week5_Evaluation_and_Error_Analysis_Report.md`, §2.2), not here.
* Metric definitions are those of Week 5; Precision@3 is the share of the 3 retrieved chunks that belong to the ground-truth file (the original script used HitRate@3 / 3).

## 2. Results — all 23 queries
| Metric | BM25 baseline | Dense V1 (Week 3) | Hybrid RRF + cross-encoder (Week 4) |
|---|---|---|---|
| HitRate@1 | 34.8% | 39.1% | 43.5% |
| HitRate@3 | 60.9% | 52.2% | 60.9% |
| MRR@3 | 0.449 | 0.457 | 0.522 |
| Precision@3 | 0.377 | 0.391 | 0.406 |
| Recall@3 | 60.9% | 52.2% | 60.9% |
| Median retrieval latency (ms) | 4.7 | 8.8 | 559.3 |

## 3. Results — the 15 hand-written queries only
| Metric | BM25 baseline | Dense V1 (Week 3) | Hybrid RRF + cross-encoder (Week 4) |
|---|---|---|---|
| HitRate@1 | 40.0% | 53.3% | 53.3% |
| HitRate@3 | 80.0% | 73.3% | 73.3% |
| MRR@3 | 0.556 | 0.633 | 0.633 |
| Precision@3 | 0.489 | 0.556 | 0.511 |
| Recall@3 | 80.0% | 73.3% | 73.3% |
| Median retrieval latency (ms) | 4.1 | 8.6 | 502.3 |

## 4. Reading the numbers
* The ordering reported in Week 4 (hybrid > dense > BM25) is only partly confirmed: on the hand-written queries BM25 reaches the highest HitRate@3 (80.0%)
  while dense and hybrid are better at rank 1 (53.3% vs 40.0%); with 15 queries one query is 6.7 pp, so the three systems are
  statistically indistinguishable here. The 50-query Week 5 benchmark (with document-level multi-gold labels) is the better yardstick: HitRate@1 0.725 (BM25), 0.675 (dense),
  0.800 (RRF), 0.900 (RRF + cross-encoder).
* Misses among the hand-written queries for the hybrid model: What is the penalty for using unfair means during an exam?; What items are strictly prohibited inside the examination hall?; When does the academic year start for first-year students?; How many leaves are permitted for medical reasons?. Several are *label* problems rather than retrieval errors:
  the Student Resource Book reproduces the UFM penalty table and the attendance rule, so a correct chunk from a second document counts as a miss when only one file is accepted;
  the first-year academic calendar is a scanned table whose OCR text is scrambled.
* The cross-encoder costs about 559 ms per query on a CPU (median) versus 5 ms for BM25 and
  9 ms for dense retrieval; mean latencies are not reported because a laptop sleep event occasionally inflates a single query by minutes.

## 5. What this changes for the project
The Week 3/4 claims "dense improved retrieval accuracy from 73 % to 93 %" and "hybrid reached 96 % HitRate@3" should not be cited. Use the Week 5 figures instead.
