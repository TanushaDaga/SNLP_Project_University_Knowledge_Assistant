# Responsible AI Statement

**Project:** University Knowledge Assistant, Group 4 (E020, E024, E025, E026, E028) · 21 October 2026

The assistant answers questions about NMIMS MPSTME rules using only the official documents we indexed. Students may act on what it says, so we tried to make it cautious and easy to check. This is what we did and where it falls short.

**Privacy.** The index holds 111 university policy documents, forms and calendars. No student marksheets, attendance data or application records were uploaded, and there is no link to any student database. The guard refuses requests for individual records or credentials. A pattern scan of all 2,175 chunks found no student ID numbers, but it did find contact details that the university itself published: 117 distinct email addresses (mostly staff) and 21 phone numbers, mainly committee members in the Student Resource Book, a few of them students. They are still in the index and should be redacted before wider use. `FAQ_s_.docx` also contains a default portal password, which should be removed from the index. Questions typed by users are sent to an external LLM provider (Groq), so users should not enter personal details.

**Bias.** Policy text can favour one kind of student (for example full-time over diploma) and the assistant repeats it as written. What we controlled is the model adding opinions of its own: the prompt limits it to the retrieved passages and the cited quote must appear word for word in them. This lowers the risk but does not remove it, because the LLM still words the answer. We did not run a bias test across student groups.

**Hallucination.** Hybrid retrieval (BM25 + dense, RRF, cross-encoder) puts the best passages in front of the model. A confidence gate refuses below 0.35. Every answer carries `supporting_document` and the verbatim `source_section`, which is checked against the retrieved text, and numbers not found in the evidence lower the confidence. These checks reduce invented rules but do not guarantee against them. On our 50-question benchmark, 1 of 10 unanswerable questions was answered anyway, and answer correctness was 55%. The citation is there so a person can catch the rest.

**Human oversight.** `confidence_score` comes from the cross-encoder score fitted on a calibration set, not from the LLM's own estimate. Below 0.35 the response has `is_fallback = true`, no source, and tells the student to contact the Academic Office. Any tool built on this API should send `is_fallback` cases to staff. A high score is not proof: one wrong answer on the scanned calendar had 0.98. The assistant only gives information; it cannot approve or change anything.

**Limitations.**
- No rule for conflicting documents (the Student Resource Book and the UFM table disagree on one row), so it may pick either.
- Comparison questions can retrieve passages from only one document.
- OCR can scramble dates in scanned tables.
- Casual wording can still cause a refusal, and a politely worded request can slip past the guard (B5_09).
- The index only knows the documents loaded when it was built.
- No load testing, local runs only, and with 40 answerable questions, differences under about 10 points are noise.

For anything that affects a student's record, the official office decides.
