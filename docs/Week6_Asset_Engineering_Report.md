# Week 6 — Asset Engineering Report (SDK + REST API)

**Project:** University Knowledge Assistant — Group 4 (E020, E024, E025, E026, E028)

## 1. What was delivered

| Deliverable | Location |
|---|---|
| Reusable Python package `knowledge_assistant` (`pip install -e .`) | `src/knowledge_assistant/` · `pyproject.toml` |
| `UniversityKnowledgeEngine.search()` / `.ask()` | `src/knowledge_assistant/client.py` |
| FastAPI service, OpenAPI docs at `/docs` | `src/api/app.py` |
| Automated tests (unit, integrity, engine logic, API) | `tests/test_verification.py` |
| API reference, curl + Python samples, peer-group integration guide | `docs/API.md` · `examples/` |
| Week 5 evaluation + error analysis (feeds the design decisions below) | `docs/Week5_Evaluation_and_Error_Analysis_Report.md` |

Run: `uvicorn src.api.app:app --host 0.0.0.0 --port 8000` → Swagger UI at `http://localhost:8000/docs`.

## 2. Architecture and decoupling

```
            ┌──────────────────────────── knowledge_assistant (importable SDK) ─────────────────────────────┐
 query ───► │ guardrails.detect_injection ─► retrieval.HybridRetriever ─► Calibrator ─► generation.LLMGenerator │
            │                               BM25 ┐                         (confidence     (Groq, cached,       │
            │                               FAISS ┴► RRF(k=60) ► cross-enc  gate 0.35)     rate-limit aware)   │
            │                                                      ▼                           ▼               │
            │                                         neighbour-chunk window     guardrails.verify_quote + ungrounded_numbers │
            └──────────────────────────────────────────────┬───────────────────────────────────────────────────┘
                                                           ▼
                client.UniversityKnowledgeEngine.ask() ─► AssistantResponse  ──►  api/app.py (Pydantic v2 models, middleware)
                                                                              └─►  src/app.py Gradio UI via improved_rag.py wrapper
```

* **One code path.** The REST API, the Gradio UI (`src/app.py`, unchanged), the evaluator and the Week 1-4 entry points
  (`baseline_retrieval.py`, `dense_indexer.py`, `improved_rag.py`, `rag_pipeline.py`) all call the same package code; the old
  modules are thin wrappers that keep their original signatures.
* **Separation of concerns.** `retrieval` has no LLM knowledge; `generation` has no retrieval knowledge; `guardrails` is pure
  functions (unit-tested without models); `client` orchestrates; `api` only translates HTTP ⇄ dataclasses.
* **Dependency injection.** `UniversityKnowledgeEngine(retriever=…, generator=…, calibrator=…)` lets tests (and other groups) swap parts.
* **Configuration** via `EngineConfig` and environment variables (`GROQ_API_KEY`, `GROQ_MODEL`, `USE_LLM`, `LLM_CACHE`, `RATE_LIMIT_PER_MIN`, `MAX_BODY_BYTES`).

## 3. Input / output schemas

Pydantic v2 models in `src/api/app.py` (rendered in `/openapi.json`):

* `QueryRequest { query: str (3-1000), top_k: int = 3 (1-10), strict_mode: bool = true }`
* `AssistantResponse { query, answer, supporting_document|null, source_section|null, confidence_score∈[0,1], retrieval_metadata{retrieval_latency_ms, rerank_latency_ms, generation_latency_ms, total_latency_ms, method="Hybrid_BM25_Dense_RRF_CrossEncoder", n_candidates, generation_mode, model, llm_live_latency_ms, llm_wait_ms}, is_fallback, refusal_reason, evidence_verified, ungrounded_numbers }`
* `SearchResponse`, `HealthResponse`, `ErrorResponse` — full tables in [`docs/API.md`](API.md).

## 4. Reliability, guardrails and instrumentation

| Concern | Mechanism |
|---|---|
| Input validation | Pydantic constraints → HTTP 422 with `{"error_code":"validation_error"}`; 16 KB body cap → 413 |
| Rate limiting | in-memory sliding window, 60 POST/min/IP → 429 + `Retry-After` |
| Latency instrumentation | per-stage timers (retrieval / rerank / generation / total) in every response; `X-Process-Time-ms`, `X-Request-ID` headers; one structured log line per request |
| Health | `/health` (always 200, reports `degraded:` reason), `/ready` (503 until loaded) |
| Fallback / exception guards | startup failure keeps the process alive (reported by `/health`); LLM error, quota exhaustion or malformed JSON → extractive answer; unexpected exception → generic 500 without leaking internals (details only in the log) |
| Hallucination control | (1) generic injection / privacy / credential guard; (2) calibrated confidence gate at 0.35; (3) LLM instructed to return `answerable:false` and numbered passages; (4) cited quote must occur **verbatim** in the retrieved text, otherwise it is replaced by the most relevant verbatim sentence and confidence × 0.8; (5) numbers in the answer absent from the evidence → confidence × 0.7 and listed in `ungrounded_numbers` |
| Free-tier LLM hygiene | ≥2 s spacing, `Retry-After`-aware retries, daily-quota circuit breaker, ~1.7 k tokens/request, on-disk response cache; tests and `/search` use no quota |

Confidence is **not** the LLM's self-assessment (observed ≈0.95-0.98 for nearly every query in Weeks 3-4). It is
`sigmoid(a·z + b)` of the top cross-encoder logit `z`, with `a, b` fitted by logistic regression on a held-out calibration set
(63 queries in the shipped re-fit, 46 in the initial fit; both disjoint from the benchmark; AUC and operating point are stored in `Data/processed/calibration.json` / `calibration_v1_initial.json`).

## 4b. Answer style, query understanding and LLM transparency (added after user testing)
* **Adaptive answers.** The default prompt (`ANSWER_STYLE=adaptive`) matches length to the question: single facts, yes/no and "one word" requests get one or two plain sentences; "explain / rules / process" questions get a bold direct answer, `###` sections with bullet points and a source line. `detailed` always uses the structured format. It follows requested styles ("explain like I'm 5", "in a table", "step by step"). The Week 5 numbers were measured with the concise prompt (`ANSWER_STYLE=concise`, pinned in the evaluator together with `QUERY_EXPANSION=0`), so they remain exactly reproducible; the detailed style was checked on live examples, not re-benchmarked.
* **Relative dates.** "this month", "next month", "today", "this year" are resolved from the system clock for retrieval and the LLM is told today's date (`interpreted_query` in the response shows the rewrite). Small glossary of spelling variants (Dussehra → Dushera, end-sem → term end examination).
* **Style words are not search terms.** "explain like I'm 5 / in simple words / end with X" is stripped from the retrieval query and kept for the LLM.
* **Proof of live LLM use.** `generation_mode`, `model`, `llm_tokens`, the `use_cache` switch and `scripts/verify_llm_live.py` (random-word test, style test, provider request counters) - see `docs/API.md` §1.6.

## 5. Test coverage (`pytest -q`)

`pytest -q` → **56 passed** (1 optional live-LLM test deselected; no test uses API quota; runtime ≈ 15-35 s on CPU).

| Group | Tests | What is checked |
|---|---|---|
| `TestMetricMath` | 5 | HitRate / MRR / Precision / Recall formulas, fact matching, Wilson interval |
| `TestGuardrails` | 13 | 5 jailbreak / probe patterns flagged, 4 benign look-alikes ("act as a team leader", "hackathon") pass, verbatim-quote check, ungrounded numbers (incl. U+202F regression), calibrator monotone |
| `TestFusion` | 2 | RRF formula `Σ 1/(k+rank)`, chunk-size cap |
| `TestBenchmark` | 5 | 50 queries = 5 × 10, unique ids, required fields, **every key fact literally present in the gold document**, calibration set disjoint from benchmark |
| `TestMetricsFile` | 7 | no None / NaN / inf anywhere, HitRate@k ∈ [0,1] and monotone, all 50 queries executed, **aggregates equal recomputation from per-query records**, latencies measured (not constant), confidence ∈ [0,1] |
| `TestEngineLogic` | 8 | stub retriever / LLM: verified citation, low-confidence refusal, `strict_mode=false`, injection refused before retrieval, LLM decline → refusal, fabricated number lowers confidence, unverifiable quote repaired, empty query rejected |
| `TestAPI` | 10 | `/health` 200 + schema, `/ready`, `/api/v1/ask` 200 + schema + refusal contract, `/api/v1/search` ranking, 422 validation, 413 payload, JSON 404, OpenAPI paths, 429 rate limit |
| `TestCorpusAndCompat` | 6 | corpus schema and size cap, scanned policies indexed via OCR, study material / old SRB editions excluded, no empty documents, FAISS size = corpus size, the original Gradio UI call signature still works |

Manual verification performed in addition: live `uvicorn` server driven with `examples/curl_examples.sh` and `examples/api_client.py` (200 / 422 / refusal paths, real Groq answer with verified citation), and `src/app.py` (original Gradio UI, unmodified) answering a query through the compatibility wrapper.

## 6. Deployment

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt                          # or: pip install -e .[api]
# put GROQ_API_KEY=... in .env  (optional)
uvicorn src.api.app:app --host 0.0.0.0 --port 8000       # add --workers 2 for more throughput (≈1.2 GB RAM per worker)
```

Smoke test: `curl http://localhost:8000/health` → `{"status":"healthy","vector_index_size":2175,…}`.
Docker is not required (CPU-only, no GPU, no external services other than the optional Groq API); a `python:3.12-slim` image with
`pip install -r requirements.txt` and the command above is sufficient.

## 7. Known limitations (carried over honestly from Week 5)

* **Answer quality (Week 5, 50 queries, single model `openai/gpt-oss-120b`):** citation accuracy 80.0 %, answer correctness 55.0 %, false-refusal 15.0 %, unanswerable answered 10.0 % (1 of 10, a soft leak); retrieval HitRate@1 0.900 / HitRate@3 0.975. n is small — see the confidence intervals in the Week 5 report.
* **Weak spots with root causes** (Week 5 §4): colloquial paraphrases are over-refused by the cross-encoder-based gate (a re-fitted gate halves it but does not remove it); comparative / multi-document questions often retrieve only one of the named documents; scanned calendars and flattened tables are read wrongly with high confidence; the Student Resource Book and the standalone UFM table disagree on one sanction and the pipeline has no document-priority rule.
* **Single citation:** the response cites one document although an answer may rest on several (planned: list of citations).
* **Degraded mode:** with no LLM the service returns extractive sentences and leaks 3 of 10 unanswerable queries; monitor `generation_mode` and `llm_available` in production.
* **Free-tier LLM:** 8 000 tokens/minute and a daily token cap; requests are spaced and retried, answers cached, and the engine fails over from `openai/gpt-oss-120b` to `openai/gpt-oss-20b` when the daily quota of the first is exhausted (`generation_mode`/`model` in every response say which model answered).
* **Concurrency:** inference is serialised by a lock; throughput ≈ 1 request / 2 s per worker. Rate limiting is per process (in-memory); use a gateway for multi-worker deployments.
* **Data hygiene recommended to the corpus owners:** redact the default portal password in `FAQ_s_.docx`, remove duplicate files, tag the press-clipping compilation as non-policy, reconcile the SRB / UFM-table discrepancy.
