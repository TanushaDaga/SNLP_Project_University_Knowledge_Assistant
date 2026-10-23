# Integration Architecture: the Knowledge Assistant as a Shared NLP Asset

**Project:** University Knowledge Assistant, Group 4
**Status of this page:** design document for the Cross-Group Integration Challenge (project brief, section 22). The brief names two integration points for Group 4: **Group 9 → Group 4** (curriculum documents → knowledge assistant) and **Group 11 → Group 4** (university circulars → knowledge assistant). We describe both here. Nothing has been connected to another group's live system. The Group 4 side (the API) exists and is tested. The Group 9 and Group 11 feeds are **proposed designs**. We have not seen their real output files, so the field names we use for them are assumptions taken from the brief, and they would need to be confirmed with each group.

## 1. Role in the AI University Platform

Group 4 owns one job: answer a policy question from the official documents, with a source, the exact text and a confidence score, or refuse. Other components should not each build their own copy of that, so we expose it as a service (the "Shared NLP Asset").

| Documents in (inbound) | Group 4 service | Questions in |
|---|---|---|
| Group 9, Curriculum Intelligence Engine: curriculum documents, syllabi, course outcomes | Ingestion, then the Group 4 index and API (FastAPI) | Students and faculty through a UI |
| Group 11, University Communications Intelligence: circulars and notices | `POST /api/v1/ask` is the single entry point for questions | Any other platform component that calls the API |

- **Inbound:** Groups 9 and 11 give us documents. We index them, so the assistant's knowledge grows without any change to the API.
- **Hub:** `POST /api/v1/ask` is the central query endpoint for all of this material. `POST /api/v1/search` gives the ranked evidence without the LLM. `GET /health` and `GET /ready` are for monitoring. The full schema is in API.md.

## 2. Inbound: how documents get into the index

Today the index is built by `src/preprocess.py` from files under `Data/Raw/<category>/`, then `dense_indexer.py` builds the FAISS index. The index stores a digest of the corpus and rebuilds itself when the corpus changes. The simplest way to take in another group's output is to reuse that path rather than add a new one:

1. The other group places files in a new category folder (or sends them to a shared folder we pick up).
2. We run the corpus build. Each file becomes chunks with the existing fields: `chunk_id`, `source_document`, `doc_path`, `category`, `page`, `extraction`, `text`.
3. The index rebuilds, and the new documents are searchable through the same `/ask` and `/search` endpoints. No API change is needed.

Rules we would apply to every inbound feed:

- **Original text, not summaries only.** The assistant must quote verbatim, so the quote has to come from a real document.
- **One source name per document.** `supporting_document` is shown to students, so it must be a name a person can look up.
- **Recalibrate after a big addition.** The 0.35 threshold was fitted on the current corpus. A large new set of documents can shift the cross-encoder scores, so we should re-run the calibration script and the benchmark before turning on a new feed.
- **Do not re-run the Week 5 evaluation with different settings.** The saved results stay as they are. A new feed would get its own test questions.

### 2.1 Group 9 → Group 4: curriculum documents

*What the brief says Group 9 works with:* course descriptions, syllabi and learning outcomes. Its analysis outputs include topic overlap between courses, common and unique topics, and outcome-to-topic mapping.

*What it adds for us:* questions about course content would be answered from the syllabus itself. Our index has only a few course-level documents today (for example the diploma course plans). Examples of questions we could then answer: "What topics are in Mathematics II for Diploma Sem II?" or "Which course outcomes does this subject list?".

*How:*
- The original syllabus and course files go into a `Curriculum` category folder and are indexed as described in section 2. The `category` field is returned by `/search`, so a client can ask for curriculum content only.
- Group 9's own analysis results (for example "72% overlap between Course A and Course B") are not official documents. If we index them at all, they should be marked as generated analysis and kept apart from the official syllabus, and the assistant should not cite them as university rules.

*Open points to agree with Group 9:* how the academic year and programme are labelled (syllabus text changes each year), and whether they pass the original files or only parsed text. If they send only parsed text, we need page numbers to keep citations useful.

### 2.2 Group 11 → Group 4: university circulars

*What the brief says Group 11 produces:* a short structured version of each circular with a title, who needs to act, the action, the deadline, important documents and a contact, in versions for students, faculty and administrators. Their brief also says the summary must not add information that is not in the source.

*What it adds for us:* the policy PDFs change about once a year, but circulars change things during the year (a holiday moved, an exam date shifted, a registration window opened). Feeding circulars in lets the assistant see them. Example question: "What is the last date to register for the end-semester examination?"

*How:*

| Group 11 output (from the brief) | How we would use it |
|---|---|
| Original circular or notice | indexed in a `Circulars` category and cited as `supporting_document` |
| Title | used as the source name shown to the student |
| Deadline / date of the circular | stored as a date for each document (a new field, not built yet) |
| Who needs to act, action, documents, contact | already inside the circular text, so retrieval finds them; the structured summary can be indexed too |

The answer should always cite the **original circular**. Group 11's summary alone should not be the source section. This matches Group 11's own rule that a summary must not go beyond its source, and our rule that a quote must appear word for word.

*Important limitation:* we have **no rule yet for which document wins** when a circular and a policy document disagree. Our own corpus already has this problem (the Student Resource Book and the UFM table disagree on one row). Before Group 11's feed goes live we would need a date or authority field, for example "a newer circular overrides the handbook", used when ranking and when building the answer. Until then, the assistant could return conflicting passages without saying which is current. This is a real gap.

## 3. `/api/v1/ask` as the central query hub

Anything that asks the assistant a question uses the same endpoint. A calling program should do three things: call `/ask`, check `is_fallback`, and show `supporting_document` and `source_section` with the answer. It should not decide from the wording of `answer`.

### 3.1 Request and response

```http
POST http://<host>:8000/api/v1/ask
Content-Type: application/json

{"query": "What is the minimum attendance required to be eligible for exams?", "top_k": 3}
```

A grounded answer (shape only; the values below are illustrative, not a recorded run):

```json
{
  "query": "What is the minimum attendance required to be eligible for exams?",
  "answer": "<answer text>",
  "supporting_document": "Attendance Rules/Attendance Rules from 2025-26 for all students.pdf",
  "source_section": "<verbatim passage from that document>",
  "confidence_score": 0.9,
  "retrieval_metadata": { "total_latency_ms": 1600, "generation_mode": "llm", "model": "openai/gpt-oss-120b" },
  "is_fallback": false,
  "refusal_reason": null,
  "evidence_verified": true,
  "ungrounded_numbers": []
}
```

A refusal comes back with HTTP 200:

```json
{
  "answer": "The official university documentation does not contain sufficient details to answer this query. ...",
  "supporting_document": null,
  "source_section": null,
  "confidence_score": 0.12,
  "is_fallback": true,
  "refusal_reason": "low_confidence"
}
```

`refusal_reason` is one of `injection:<type>`, `low_confidence`, `llm_declined`, `no_results`.

### 3.2 How a calling program uses the response

```python
import requests

def policy_lookup(question: str) -> dict:
    r = requests.post("http://localhost:8000/api/v1/ask",
                      json={"query": question, "top_k": 3}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data["is_fallback"]:
        return {"found": False, "reason": data["refusal_reason"]}
    return {
        "found": True,
        "text": data["answer"],
        "citation": data["supporting_document"],
        "evidence": data["source_section"],
        "confidence": data["confidence_score"],
    }
```

- Show `citation` and `evidence` to the reader every time.
- When `found` is false, say that no official rule was found and point the reader to the Academic Office. Do not fill the gap with another model.
- Cache answers by `(query, top_k)`. The policy documents change slowly, but a circular feed would need a shorter cache time.
- `similarity_score` from `/search` is a cross-encoder score, not a probability. Use `confidence_score` from `/ask` for that.

Our own REST demo page (`src/rest_demo_ui.py`) works this way. Its "API inspector" tab shows the raw request, the JSON response, and what a calling program would do with it.

## 4. Operations and platform concerns

| Topic | Current state | Needed for the platform |
|---|---|---|
| Auth | none (campus-internal) | put the service behind the university gateway before use outside the LAN |
| Rate limit | 60 POST per minute per IP, in memory | shared limit if several instances run |
| Concurrency | one process, inference is serialised | load testing has not been done; add workers after measuring |
| LLM quota | Groq free tier, cached, falls back to extractive answers | a paid key or in-house model if several components call `/ask` |
| Versioning | `/api/v1`, new fields may be added | clients must ignore unknown fields |
| Monitoring | `/health`, `/ready`, `X-Request-ID` per response | platform health checks can poll these |

If several components call `/ask` at once, the free-tier LLM limit (about 8,000 tokens per minute) is the first thing that will break. The service will keep responding but with extractive answers, which are less accurate (25% answer correctness without the LLM in our Week 5 run, against 55% with it).

## 5. What is done and what is not

| Item | Status |
|---|---|
| `/ask`, `/search`, `/health`, `/ready`, schemas, tests | built and tested in Weeks 5 and 6 |
| REST demo page that calls the API over HTTP | built and tried in Week 7 |
| Ingestion of Group 9 and Group 11 documents | design only |
| Group 11 circular dates and a "newer overrides older" rule | not built |
| Live test with Group 9 or Group 11 output | not done |
| Auth, shared rate limiting, deployment | not built |
