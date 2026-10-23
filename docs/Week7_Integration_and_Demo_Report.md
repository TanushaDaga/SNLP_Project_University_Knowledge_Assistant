# Week 7: Integration and Demo Report

**Project:** University Knowledge Assistant, Group 4 (E020, E024, E025, E026, E028)

## 1. Overview

This week we did not change the retrieval pipeline, the API, the evaluation scripts or the original UI. We worked on four things around them: a second demo page that uses the API over HTTP, a Responsible AI statement, a design for how our API fits into the AI University Platform, and this report.

| Deliverable | Location |
|---|---|
| Responsible AI Statement (1 page) | `docs/Responsible_AI_Statement.md` |
| REST demo page (new, talks to the API over HTTP) | `src/rest_demo_ui.py` |
| Integration architecture (Group 9 and Group 11 into our index) | `docs/Integration_Architecture.md` |
| This report | `docs/Week7_Integration_and_Demo_Report.md` |

Before writing, we re-ran the test suite with the LLM switched off (`USE_LLM=0`). All 62 tests pass, and one live-LLM test is skipped by default. We did not re-run any evaluation, so all numbers in our reports are still the Week 5 results.

## 2. UI and API integration

We now have two demo pages and one API, all on the same code:

- **Original Gradio demo** (`python src/app.py`, port 7860). It calls the assistant through the `improved_rag.py` wrapper, in the same Python process as the engine. It does not send HTTP requests. We left this file unchanged.
- **FastAPI service** (`uvicorn src.api.app:app --port 8000`). It is for other programs: `/api/v1/ask`, `/api/v1/search`, `/health`, `/ready`. Swagger docs are at `/docs`. Example clients are in `examples/api_client.py` and `examples/curl_examples.sh`.
- **REST demo page** (`python src/rest_demo_ui.py`, port 7861, new this week). It has the same look as the original but only talks to the API over HTTP. The "Ask" tab is a chat that shows the answer, source document, verbatim quote, confidence and mode. The "API inspector" tab shows the raw request and JSON response for `/ask` or `/search`, and what a program calling the API would do with it (use the answer, or hand over to a person when `is_fallback` is true). It shows the API status at the top and gives a clear message if the API is not running.

We tested the new page against the running API. A normal question (minimum attendance for exams) came back with the Attendance Rules PDF as source, the verbatim sentence, confidence 0.97, evidence verified and `generation_mode: llm`. An answer-key request was refused with reason `injection:integrity_violation`. The inspector sent a `/search` request and showed the raw response. We did not run an automated test for this page, and we only tried it in one browser.

## 3. Responsible AI principles

The full statement is in `docs/Responsible_AI_Statement.md`. In short:

| Area | What we say |
|---|---|
| Privacy | The index holds only university policy documents, with no student records. A pattern scan of all 2,175 chunks found no 11-digit ID numbers, but it found 117 distinct email addresses and 21 phone numbers that the university published itself (staff, and committee members in the Student Resource Book, including a few students). They are still in the index and should be redacted before wider use. `FAQ_s_.docx` also holds a default portal password. User questions are sent to Groq. |
| Bias | The assistant repeats what the documents say, including any bias in them. The prompt and the quote check keep the model from adding its own claims, but the LLM still writes the wording. No bias test was run. |
| Hallucination | Hybrid retrieval, a confidence gate at 0.35, a verbatim quote check and a check on numbers. These reduce wrong answers but do not stop them: on our benchmark 1 of 10 unanswerable questions was answered, and answer correctness was 55%. |
| Human oversight | `confidence_score` below 0.35 gives a refusal with no source, and downstream tools should send `is_fallback` cases to a person. A high score is not proof: one wrong answer had 0.98. |
| Limitations | No rule for conflicting documents, weak on comparison questions and scanned tables, some casual wording is still refused, no load testing, local runs only. |

## 4. Cross-group integration readiness

The project brief (section 22) names two integration points for our group: Group 9 → Group 4 (curriculum documents) and Group 11 → Group 4 (university circulars). The design for both is in `docs/Integration_Architecture.md`. We have not seen the real output files of either group, and nothing is connected to another group's live system, so this section is a plan and not a tested integration. The "API inspector" tab of the REST demo page shows the call that any other component would make to `/api/v1/ask`, and how the response would be used.

| Group | Direction | Idea | Status |
|---|---|---|---|
| Group 9, Curriculum Intelligence Engine | into our index | syllabi and course documents added as a new category, so `/ask` can answer questions about course content; their generated analysis would be kept apart from official documents | design only |
| Group 11, University Communications Intelligence | into our index | original circulars and notices added (cited as the source, with their title and date), so answers reflect changes during the year | design only; needs a date field and a rule for newer circulars versus older policy |

Problems we found while writing this:

- Our system has no way to choose between two documents that disagree, and circulars from Group 11 will make that more common. This is the main thing to fix before that feed is used.
- The Groq free tier (about 8,000 tokens per minute) cannot serve several components at once. When it runs out the API gives extractive answers, which were less accurate in our Week 5 run (25% answer correctness without the LLM, 55% with it).
- The confidence threshold was fitted on the current documents, so after adding a large new set we would need to recalibrate.

## 5. Not done yet

- An automated test for the REST demo page.
- Removing contact details and the default password from the index.
- A document date or authority rule.
- Auth, shared rate limiting and deployment beyond local runs.
- Testing any of the integrations with the other groups.
