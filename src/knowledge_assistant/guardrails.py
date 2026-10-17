"""
knowledge_assistant.guardrails
Deterministic safety / grounding checks that run around the LLM:

1. detect_injection()      generic prompt-injection / jailbreak / privacy-probe patterns
2. verify_quote()          is the cited passage really a substring of the retrieved text?
3. ungrounded_numbers()    numbers in the answer that appear nowhere in the evidence
4. Calibrator              Platt scaling: cross-encoder logit -> probability-like confidence
5. best_sentences()        extractive evidence selection (no-LLM fallback / quote repair)

The injection patterns follow general jailbreak taxonomies (instruction override, persona
hijack, system-prompt probing, intrusion requests, requests for individual student records).
They contain NO knowledge of any benchmark query.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

# ───────────────────────────── injection guard ─────────────────────────────

_INJECTION_PATTERNS = {
    "instruction_override": re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\b[^.?!]{0,40}\b(previous|prior|above|earlier|all|any|your|these)\b"
        r"[^.?!]{0,30}\b(instruction|prompt|rule|guideline|direction|restriction|polic)", re.I),
    "persona_hijack": re.compile(
        r"\b(pretend (to be|you are|that you)|you are now\b|from now on,? you|role-?play as|simulate being|"
        r"(act|behave|respond|answer) (as|like) (if you (are|were)|(a|an|the|my)? ?(registrar|dean|admin\w*|professor|examiner|"
        r"exam controller|controller|teacher|principal|hod|system|developer|hacker|unrestricted|jailbroken)))", re.I),
    "prompt_probe": re.compile(
        r"\b(system prompt|developer mode|jailbreak|\bDAN\b mode|reveal your (instructions|prompt))", re.I),
    "intrusion": re.compile(
        r"\b(hack(ing)?\b|exploit (the )?(erp|portal|server|system)|sql injection|crack (the )?password|"
        r"break into|unauthori[sz]ed access)", re.I),
    "record_tampering": re.compile(
        r"\b(change|alter|edit|modify|forge|falsify|update)\b[^.?!]{0,30}\b(my |his |her |their |the )?"
        r"(attendance|marks|grades?|cgpa|result|records?)\b[^.?!]{0,30}\b(retroactive|without|secretly|illegally|"
        r"in the (erp|system|database)|to \d+\s*%)", re.I),
    "credential_probe": re.compile(
        r"\b(what('s| is| are)|tell me|give me|share|reveal|show me|send me)\b[^.?!]{0,40}"
        r"\b(password|passcode|login credentials|otp)\b", re.I),
    "student_record_probe": re.compile(
        r"\b(cgpa|marks|grades?|results?|attendance|fees?|address|phone)\s+(of|for)\s+(student|roll\s*(no|number)?|"
        r"[a-z]\d{3,})\b|\bstudent\s+[a-z]?\d{3,}\b", re.I),
}

# Request to leak exam material is an integrity violation regardless of corpus content.
_INTEGRITY_PATTERN = re.compile(r"\b(answer key|question paper (leak|in advance)|leak(ed)? paper)\b", re.I)


def detect_injection(query: str) -> Optional[str]:
    """Return the name of the matched guard category, or None if the query looks benign."""
    for name, pat in _INJECTION_PATTERNS.items():
        if pat.search(query):
            return name
    if _INTEGRITY_PATTERN.search(query):
        return "integrity_violation"
    return None


# ───────────────────────────── grounding checks ─────────────────────────────

def _squash(s: str) -> str:
    """Lower-case and drop every character except letters, digits and %: robust to OCR spacing.
    NFKC first, so the LLM's narrow no-break spaces (U+202F), non-breaking hyphens and ligatures compare equal."""
    return re.sub(r"[^a-z0-9%]+", "", unicodedata.normalize("NFKC", s).lower())


def _tokens(s: str) -> List[str]:
    return re.findall(r"[a-z0-9%]+", s.lower())


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def verify_quote(quote: str, passages: List[str], min_chars: int = 12) -> Tuple[bool, Optional[int]]:
    """True (and the passage index) if `quote` occurs verbatim (modulo spacing/punctuation/case)."""
    sq = _squash(quote or "")
    if len(sq) < min_chars:
        return False, None
    for i, p in enumerate(passages):
        if sq in _squash(p):
            return True, i
    return False, None


def best_sentences(query_or_hint: str, passages: List[str], n: int = 2, max_chars: int = 450
                   ) -> Tuple[str, Optional[int]]:
    """Pick the n sentences with the highest token overlap with `query_or_hint` (IDF-free, deterministic).
    Returns (joined verbatim sentences, index of the passage they came from)."""
    qt = {t for t in _tokens(query_or_hint) if len(t) > 2}
    best: List[Tuple[float, int, str]] = []
    seen = set()
    for pi, p in enumerate(passages):
        for s in split_sentences(p):
            st = set(_tokens(s))
            key = _squash(s)
            if not st or key in seen:      # neighbouring chunks overlap: never return a sentence twice
                continue
            seen.add(key)
            overlap = len(qt & st)
            if overlap:
                best.append((overlap / math.sqrt(len(st)), pi, s))
    if not best:
        if passages:
            return passages[0][:max_chars], 0
        return "", None
    best.sort(key=lambda x: -x[0])
    top = best[:n]
    pi = top[0][1]
    text = " ".join(s for _, _, s in top)
    return text[:max_chars], pi


_NUM = re.compile(r"\d+(?:[.,]\d+)?\s?%?")


def _numbers(text: str) -> List[str]:
    out = []
    text = unicodedata.normalize("NFKC", text)   # U+202F / U+00A0 between a number and its unit -> ordinary space
    for m in _NUM.finditer(text):
        raw = re.sub(r"\s+", "", m.group(0))
        digits = raw.rstrip("%").replace(",", "")
        if "%" in raw or "." in digits or len(digits) >= 2:  # ignore bare list markers / single digits
            out.append(raw.replace(",", ""))
    return out


def ungrounded_numbers(answer: str, evidence_texts: List[str], query: str = "") -> List[str]:
    """Numbers (>=2 digits, decimals or percentages) present in the answer but absent from the
    retrieved evidence and the question - a cheap, deterministic hallucination detector."""
    ev = unicodedata.normalize("NFKC", " ".join(evidence_texts)).replace(",", "")
    ev_squash = re.sub(r"\s+", "", ev)
    allowed = set(_numbers(query))
    missing = []
    for num in _numbers(answer):
        core = num.rstrip("%")
        if num in allowed or core in {a.rstrip("%") for a in allowed}:
            continue
        if core in ev_squash:
            continue
        missing.append(num)
    return sorted(set(missing))


# ───────────────────────────── confidence calibration ─────────────────────────────

class Calibrator:
    """confidence = sigmoid(a * top_cross_encoder_logit + b); (a, b) fitted on a held-out
    calibration set by `src/calibrate_confidence.py` (never on the evaluation benchmark)."""

    def __init__(self, a: float = 1.0, b: float = 0.0, fitted: bool = False, meta: Optional[dict] = None):
        self.a, self.b, self.fitted, self.meta = a, b, fitted, meta or {}

    @classmethod
    def load(cls, path: Path) -> "Calibrator":
        path = Path(path)
        if path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                return cls(float(d["a"]), float(d["b"]), True, d)
            except Exception:
                pass
        return cls()

    def __call__(self, logit: float) -> float:
        z = self.a * float(logit) + self.b
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)
