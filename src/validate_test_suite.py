"""
src/validate_test_suite.py
Mechanical verification of the ground truth in evaluation/expanded_test_suite.json (and
calibration_set.json): every expected document must exist in the corpus and every key fact /
evidence phrase of an answerable query must literally occur in that document's text.

    python src/validate_test_suite.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from knowledge_assistant.guardrails import _squash  # noqa: E402

CORPUS = ROOT / "Data" / "processed" / "corpus.json"
SUITE = ROOT / "evaluation" / "expanded_test_suite.json"
REQUIRED = ["query_id", "category", "bucket", "query", "expected_document", "key_facts", "is_answerable"]


def doc_texts() -> dict:
    texts = defaultdict(list)
    for c in json.loads(CORPUS.read_text(encoding="utf-8")):
        texts[c.get("doc_path") or c["source_document"]].append(c["text"])
    return {k: _squash(" ".join(v)) for k, v in texts.items()}


def present(fact: str, text_squashed: str) -> bool:
    return any(_squash(a) in text_squashed for a in fact.split("||") if _squash(a))


def validate(path: Path = SUITE, verbose: bool = True) -> list:
    problems = []
    suite = json.loads(path.read_text(encoding="utf-8"))
    texts = doc_texts()
    ids = set()
    for tc in suite:
        qid = tc.get("query_id", "?")
        for k in REQUIRED:
            if k not in tc:
                problems.append(f"{qid}: missing field {k}")
        if qid in ids:
            problems.append(f"{qid}: duplicate id")
        ids.add(qid)
        if tc.get("is_answerable"):
            docs = list(tc.get("acceptable_documents") or []) + [tc.get("expected_document")]
            docs += list(tc.get("gold_documents") or [])
            joined = ""
            for d in filter(None, docs):
                if d not in texts:
                    problems.append(f"{qid}: document not in corpus: {d}")
                else:
                    joined += texts[d]
            for f in tc.get("key_facts", []) + tc.get("evidence_phrases", []):
                if not present(f, joined):
                    problems.append(f"{qid}: fact/phrase not found in gold document(s): {f!r}")
            if not tc.get("key_facts"):
                problems.append(f"{qid}: answerable query without key_facts")
        else:
            if tc.get("expected_document"):
                problems.append(f"{qid}: unanswerable query must have expected_document = null")
            whole = "".join(texts.values())
            for term in tc.get("absence_terms", []):
                if _squash(term) in whole:
                    problems.append(f"{qid}: corpus DOES mention {term!r} - query may be answerable")
    if verbose:
        n_buckets = {b: sum(1 for t in suite if t["bucket"] == b) for b in range(1, 6)}
        print(f"{path.name}: {len(suite)} queries, per bucket {n_buckets}")
        print("OK - all ground truth verified against corpus text" if not problems else
              f"{len(problems)} PROBLEM(S):\n  " + "\n  ".join(problems))
    return problems


def validate_calibration(path: Path = ROOT / "evaluation" / "calibration_set.json", verbose: bool = True) -> list:
    problems, texts = [], doc_texts()
    whole = "".join(texts.values())
    for it in json.loads(path.read_text(encoding="utf-8")):
        if it["is_answerable"]:
            doc = it.get("expected_document")
            if doc not in texts:
                problems.append(f"{it['query_id']}: document not in corpus: {doc}")
            else:
                for f in it.get("key_facts", []):
                    if not present(f, texts[doc]):
                        problems.append(f"{it['query_id']}: fact not found in {doc}: {f!r}")
        else:
            for term in it.get("absence_terms", []):
                if _squash(term) in whole:
                    problems.append(f"{it['query_id']}: corpus DOES mention {term!r}")
    if verbose:
        sep = "\n  "
        print("calibration_set.json: " + ("OK" if not problems else
                                          f"{len(problems)} PROBLEM(S):{sep}" + sep.join(problems)))
    return problems


if __name__ == "__main__":
    bad = validate()
    cal = ROOT / "evaluation" / "calibration_set.json"
    if cal.exists():
        bad += validate_calibration(cal)
    sys.exit(1 if bad else 0)
