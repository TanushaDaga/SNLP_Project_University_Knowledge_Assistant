# University Knowledge Assistant — SNLP Group 4 (E020, E024, E025, E026, E028)

Grounded question answering over the official MPSTME / NMIMS policy documents: hybrid retrieval
(BM25 + FAISS + Reciprocal Rank Fusion), cross-encoder re-ranking, a calibrated confidence gate, a citation-verified
LLM answer (Groq `openai/gpt-oss-120b`) and a production REST API.

## Quick start (Windows / Linux, CPU only)

```bash
pip install -r requirements.txt            # Python 3.10+; tested on 3.14
# .env must contain  GROQ_API_KEY=...      (optional: without it answers are extractive, everything still runs)

uvicorn src.api.app:app --host 0.0.0.0 --port 8000      # REST API  -> http://localhost:8000/docs
python src/app.py                                       # Gradio chat UI (unchanged since Week 4)
pytest -q                                               # 62 tests, uses no LLM quota
```

`Data/processed/` already contains the rebuilt corpus (`corpus.json`), the FAISS index and the confidence
calibration. Everything below is only needed to *regenerate* artefacts.

| Step | Command | Output |
|---|---|---|
| 1 Build corpus (PDF + OCR + DOCX) | `python src/preprocess.py` | `Data/processed/corpus.json`, `corpus_manifest.json` |
| 2 Fit confidence calibrator | `python src/calibrate_confidence.py` | `Data/processed/calibration.json` |
| 3 Verify benchmark ground truth | `python src/validate_test_suite.py` | pass/fail list |
| 4 Week 5 evaluation (50 queries) | `python src/comprehensive_evaluator.py` (`--no-llm` = zero API usage) | `evaluation/week5_quantitative_metrics.json` |
| 5 Error analysis + report | `python src/error_analyzer.py` | `docs/Week5_Evaluation_and_Error_Analysis_Report.md` |
| 6 Week 4 comparison (retrieval only) | `python src/compare_models.py` | `evaluation/model_comparison_results.json` |

## Layout

```
src/knowledge_assistant/   reusable SDK (client.py = UniversityKnowledgeEngine, retrieval.py, generation.py, guardrails.py, config.py, schemas.py)
src/api/app.py             FastAPI service (/health /ready /api/v1/search /api/v1/ask)
src/preprocess.py          corpus builder with OCR fallback        src/app.py  Gradio UI (original)
src/baseline_retrieval.py  src/dense_indexer.py  src/improved_rag.py  src/rag_pipeline.py   Week 1-4 entry points (thin wrappers)
src/comprehensive_evaluator.py  src/error_analyzer.py  src/validate_test_suite.py  src/calibrate_confidence.py
evaluation/                benchmark (50 q), calibration set, metrics, failure cases
tests/test_verification.py automated checks      docs/   reports + API.md      examples/  curl + Python client
Data/Raw/…                 source documents       Data/processed_legacy_week4/  the old (Week 4) index, kept for reference
```

## Proving the LLM is really used
`python scripts/verify_llm_live.py` (about 3 calls) - random-word test, style test and Groq's own request counters. In every API response `generation_mode` is `llm` (live), `llm_cached` (identical prompt answered before), `extractive` (no LLM) or `none` (refused), and `llm_tokens` is the token count billed by Groq. Send `"use_cache": false` to force a live call. Answer length adapts to the question (`ANSWER_STYLE=adaptive`: short question -> one or two sentences, "explain ..." -> structured Markdown; `detailed` = always structured); the benchmark in the Week 5 report used `ANSWER_STYLE=concise`.

## Free-tier LLM policy
The Groq key is read from `.env` only to answer questions through this pipeline. Calls are spaced, 429-retried with
`Retry-After`, capped at ~1.7k tokens each and **cached on disk** (`Data/cache/llm_cache.json`); `pytest` never calls the API
(`USE_LLM=0`), `/api/v1/search` never calls it, and `--no-llm` runs the full evaluation without it.

See `docs/` for the Week 5 evaluation & error analysis, the Week 6 asset-engineering report and the API reference.
