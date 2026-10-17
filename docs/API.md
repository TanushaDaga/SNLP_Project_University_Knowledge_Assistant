# University Knowledge Assistant — REST API & SDK (v1.0.0)

Grounded question answering over the official NMIMS MPSTME policy documents (attendance, examinations / UFM,
placement, internship / IPT, capstone, community service, academic calendar, open electives, Student Resource Book …).

* **Pipeline:** injection guard → BM25 + FAISS → Reciprocal Rank Fusion → cross-encoder re-rank → calibrated confidence gate →
  grounded LLM answer (Groq `openai/gpt-oss-120b`) → citation verification → numeric-grounding check.
* **Base URL (local):** `http://localhost:8000` · **Interactive docs:** `/docs` (Swagger UI) · `/redoc` · `/openapi.json`
* **Start the server:** `uvicorn src.api.app:app --host 0.0.0.0 --port 8000` (from the project root; first start loads models, ~10 s).
* **Auth:** none (campus-internal service). Put it behind the university reverse proxy / API gateway before exposing it outside the LAN.
* **Content type:** `application/json`, UTF-8.

---

## 1. Endpoints

| Method | Path | Purpose | Uses LLM quota |
|---|---|---|---|
| GET | `/health` | liveness, index size, models, uptime (always HTTP 200; `status` says `healthy` or `degraded: …`) | no |
| GET | `/ready` | readiness probe — HTTP 503 until the engine has loaded | no |
| POST | `/api/v1/search` | ranked evidence chunks, no generation | **no** |
| POST | `/api/v1/ask` | grounded answer + `supporting_document` + `source_section` + `confidence_score` | yes (1 call, cached) |

### 1.1 Request body (`QueryRequest`) — both POST endpoints

| Field | Type | Default | Constraints | Meaning |
|---|---|---|---|---|
| `query` | string | — | 3–1000 chars after trimming | natural-language question |
| `top_k` | int | 3 | 1–10 | chunks retrieved / used as evidence |
| `strict_mode` | bool | true | | `true`: refuse when calibrated confidence < 0.35. `false`: skip the confidence gate (injection guard stays on) |
| `use_cache` | bool | true | | `false`: never reuse a stored LLM answer, always make a live LLM call (use it to prove the LLM is really called) |

### 1.2 `POST /api/v1/ask` → `AssistantResponse`

```jsonc
{
  "query": "...",
  "answer": "...",                       // grounded answer, or the standard refusal text
  "supporting_document": "Folder/file.pdf" | null,
  "source_section": "verbatim passage"   | null,
  "confidence_score": 0.0-1.0,
  "retrieval_metadata": {
    "retrieval_latency_ms": 96.3,        // BM25 + FAISS + RRF
    "rerank_latency_ms": 461.4,          // cross-encoder
    "generation_latency_ms": 1052.3,     // LLM wall-clock (0 for refusals / extractive)
    "total_latency_ms": 1610.6,
    "method": "Hybrid_BM25_Dense_RRF_CrossEncoder",
    "n_candidates": 20,
    "generation_mode": "llm | llm_cached | extractive | none",
    "model": "openai/gpt-oss-120b" | null,
    "llm_live_latency_ms": 1049.5 | null,
    "llm_wait_ms": 0.0,                  // free-tier back-off slept during the request
    "llm_tokens": 2157,                  // tokens billed by Groq for this answer (null if no LLM call)
    "interpreted_query": "... (October 2026 Oct-26)"   // retrieval query after "this month" / spelling-variant rewriting, null if unchanged
  },
  "is_fallback": false,                  // true => refusal
  "refusal_reason": null,                // injection:<type> | low_confidence | llm_declined | no_results
  "evidence_verified": true,             // quote found verbatim in the retrieved text
  "ungrounded_numbers": []               // numbers in the answer absent from the evidence (lowers confidence)
}
```

**Guardrail contract**

| Situation | HTTP | `is_fallback` | `supporting_document` | `confidence_score` |
|---|---|---|---|---|
| Answer found and cited | 200 | false | `Folder/file.pdf` | ≥ 0.35 |
| Calibrated confidence < 0.35 (`strict_mode=true`) | 200 | **true** | `null` | the (low) confidence |
| Prompt-injection / jailbreak / credential / student-record probe | 200 | **true** | `null` | 0.0 |
| LLM judges the passages do not contain the answer | 200 | **true** | `null` | retrieval confidence |

Refusal text: *"The official university documentation does not contain sufficient details to answer this query. Please contact the Academic Office or the relevant department for assistance."*
Integrators should branch on `is_fallback` / `supporting_document is null`, never on the wording of `answer`.

### 1.3 `POST /api/v1/search` → `SearchResponse`

```jsonc
{ "query": "...", "top_k": 3, "retrieval_latency_ms": 540.2, "method": "Hybrid_BM25_Dense_RRF_CrossEncoder",
  "results": [ { "rank": 1, "document": "Folder/file.pdf", "text_snippet": "first 300 chars", "similarity_score": 6.26,
                 "chunk_id": 7, "page_number": 1, "category": "Attendance Rules", "rrf_score": 0.0322 } ] }
```
`similarity_score` is the cross-encoder logit (unbounded, higher = more relevant); it is **not** a probability — use `confidence_score` from `/ask` for that.

### 1.4 `GET /health` → `HealthResponse`
`{"status": "healthy", "vector_index_size": 2175, "embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "llm_available": true, "confidence_calibrated": true, "uptime_seconds": 12.5, "version": "1.0.0"}`

### 1.5 Errors (all JSON: `{"error_code", "detail", "request_id"}`)

| HTTP | `error_code` | When | Client action |
|---|---|---|---|
| 413 | `payload_too_large` | body > 16 KB | shorten the query |
| 422 | `validation_error` | blank / too short / too long query, `top_k` out of range, wrong types | fix the request |
| 429 | `rate_limited` | > 60 POST requests / minute / client IP (`RATE_LIMIT_PER_MIN`) | wait `Retry-After` seconds |
| 503 | `http_error` | engine not loaded (missing corpus / models) — see `/health` | retry later, alert ops |
| 500 | `internal_error` | unexpected bug; details only in server log, correlate with `request_id` | retry once, then report |

Every response carries `X-Request-ID` and `X-Process-Time-ms` headers.
If the LLM provider is rate-limited or down, `/ask` does **not** fail: it returns an *extractive* answer (best verbatim sentences, `generation_mode: "extractive"`).

---

### 1.6 How to verify that an answer really comes from the LLM
* `retrieval_metadata.generation_mode`: **`llm`** = a live Groq call was made for this request; `llm_cached` = a stored answer to the *identical prompt* (same question, same retrieved passages) from an earlier live call is reused; `extractive` = no LLM (quota/key/network problem), the answer is verbatim sentences; `none` = refused before generation.
* `model` and `llm_tokens` are returned by Groq itself (a canned answer has no token count). Send `"use_cache": false` to force a live call.
* Style test: ask `... explain like I'm 5 and end your answer with the word BANANA-<random number>` - only a live model can obey.
* `python scripts/verify_llm_live.py` runs the nonce test, a style test and reads Groq's own request counters before/after.
* Answer style: the default `ANSWER_STYLE=adaptive` matches length to the question (single fact / yes-no / "one word" -> one or two plain sentences; "explain / rules / process" -> structured Markdown with headings and bold; explicit requests such as "briefly" or "in a table" override); `detailed` always uses the structured format; the Week 5 benchmark numbers were measured with `ANSWER_STYLE=concise` (1-4 sentences).

## 2. Code samples

### 2.1 cURL
```bash
curl -s -X POST http://localhost:8000/api/v1/ask -H "Content-Type: application/json" \
     -d '{"query": "What is the minimum attendance percentage required to be eligible for semester-end examinations?"}'
```
More in [`examples/curl_examples.sh`](../examples/curl_examples.sh).

### 2.2 Python (`requests`) — see [`examples/api_client.py`](../examples/api_client.py)
```python
from examples.api_client import KnowledgeAssistantClient   # or copy the 40-line class
c = KnowledgeAssistantClient("http://localhost:8000")
r = c.ask("Can I reject a job offer I received through campus placement?")
if r["is_fallback"]:
    print("No grounded answer:", r["refusal_reason"])
else:
    print(r["answer"]); print("source:", r["supporting_document"], "| conf", r["confidence_score"])
```

### 2.3 In-process SDK (no HTTP)
```python
from knowledge_assistant import UniversityKnowledgeEngine
engine = UniversityKnowledgeEngine()                 # loads models + index once
hits = engine.search("railway concession", top_k=3)  # -> List[SearchResult]
resp = engine.ask("What is the UFM penalty for a smart watch?", enforce_grounding=True)   # -> AssistantResponse
print(resp.answer, resp.supporting_document, resp.confidence_score)
```
Install as a package: `pip install -e .` (see `pyproject.toml`); configuration through `EngineConfig` or the environment variables below.

---

## 3. Integration guide for peer groups

### 3.1 Group 10 — Admin Copilot
Use **`POST /api/v1/ask`** as the policy-lookup tool of the copilot.

* Treat `supporting_document` + `source_section` as the **citation to show the staff member**; never display an answer without them.
* If `is_fallback` is true, the copilot must say it found no official rule and route the case to a human — do not let the copilot's own LLM fill the gap.
* For paperwork workflows (leave, defaulter undertaking, bonafide) use `/api/v1/search` with `top_k=5` and filter on `category == "Important Forms"`.
* The service intentionally refuses requests to *change* records, to disclose individual students' data and credentials (guard categories `record_tampering`, `student_record_probe`, `credential_probe`) — such actions belong to the copilot's own authenticated back-end, not to this read-only knowledge service.

```python
def policy_lookup(question: str) -> dict:
    r = client.ask(question, top_k=3)
    return {"found": not r["is_fallback"], "text": r["answer"], "citation": r["supporting_document"],
            "evidence": r["source_section"], "confidence": r["confidence_score"]}
```

### 3.2 Group 8 — Placement Intelligence
Use **`POST /api/v1/search`** (no LLM quota, ≈0.5 s) to pull the policy clauses behind a placement decision, and **`/ask`** for human-readable summaries.

* Placement rules live in `Placement Policy/Final Placement Policy - Batch 2027.pdf` (phases, dream-offer 1.5× rule, ten skips, absence rules) and `Placement Policy/Internship Policy_Batch 27.pdf` (TIP). The two documents differ on offer rejection — query both (`top_k=5`) and show both citations.
* Eligibility questions that depend on individual data (a particular student's CGPA, backlogs) cannot be answered by this service: the placement policy contains **no CGPA or backlog thresholds** (they come from each company's job description). Combine the policy text with your own student data.
* Cache answers by `(query, top_k)` for 24 h; the corpus changes once per academic year.

### 3.3 Operational notes
* One engine instance per process; inference is CPU bound, requests are serialised internally (typical `/search` ≈ 0.5 s, `/ask` ≈ 1.5–2.5 s with a warm LLM). Run several uvicorn workers (`--workers 2`) only if each fits in memory (~1.2 GB).
* Free-tier LLM: calls are spaced (`GROQ_MIN_INTERVAL_S`, default 2 s), HTTP 429 is retried with `Retry-After`, answers are cached in `Data/cache/llm_cache.json`; on quota exhaustion the service degrades to extractive answers.
* Environment variables: `GROQ_API_KEY` (in `.env`, never committed), `GROQ_MODEL`, `USE_LLM=0` (disable LLM), `LLM_CACHE=0`, `RATE_LIMIT_PER_MIN`, `MAX_BODY_BYTES`.

## 4. Versioning
`/api/v1/*` is stable; additive fields may appear in responses (clients must ignore unknown fields). Breaking changes will ship under `/api/v2`.
