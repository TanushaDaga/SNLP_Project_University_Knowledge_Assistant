"""
knowledge_assistant.query_rewrite
Two small, deterministic query-understanding steps that run BEFORE retrieval:

1. resolve_relative_dates()  "this month", "next month", "today", "tomorrow", "this year" ... -> concrete month / date
   strings taken from the system clock, so "When is my holiday this month?" retrieves the October-2026 rows of the
   holiday list and the LLM is told today's date (a policy document cannot know what "this month" means).
2. expand_terms()            a tiny glossary of spelling / wording variants between how students write and how the
   documents write ("Dussehra" -> "Dushera", "end-sem" -> "term end examination").

The original question is always kept for the answer; only the retrieval query is rewritten.
Both steps are no-ops for queries that contain none of these expressions, so they do not affect the benchmark.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Optional

_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]

_GLOSSARY = {
    r"\bd[ua]s+e?h?e?ra\b|\bdasara\b|\bvijayadashami\b": "Dushera",
    r"\bend[- ]?sems?\b|\bend semester exams?\b|\bsem(ester)? end exams?\b": "term end examination",
    r"\bmid[- ]?sems?\b|\bmid semester exams?\b": "mid term test",
    r"\bcut[- ]?off (for )?attendance\b|\battendance cut[- ]?off\b": "minimum attendance requirement",
    r"\bbunk(ing)?\b": "absent from",
    r"\bbacklogs?\b|\bkt\b": "F grade re-examination",
    r"\bcheating\b": "unfair means",
    r"\bid cards?\b": "identity card",
    r"\bleaves\b|\bdays? off\b": "absence attendance relaxation",
}


@dataclass
class RewriteResult:
    original: str
    retrieval_query: str
    date_context: Optional[str] = None          # e.g. "Today is Saturday, 17 October 2026"
    notes: List[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.retrieval_query != self.original


def _shift_month(d: date, delta: int) -> date:
    m = d.month - 1 + delta
    return date(d.year + m // 12, m % 12 + 1, 1)


def _month_text(d: date) -> str:
    """'October 2026 Oct-26' - documents write both the full month and the 'dd-Mon-yy' form."""
    return f"{_MONTHS[d.month - 1]} {d.year} {_MONTHS[d.month - 1][:3]}-{str(d.year)[2:]}"


def resolve_relative_dates(query: str, today: Optional[date] = None) -> RewriteResult:
    today = today or date.today()
    q = query
    notes: List[str] = []
    extra: List[str] = []

    def add(label: str, text: str):
        notes.append(f"'{label}' -> {text}")
        extra.append(text)

    low = query.lower()
    if re.search(r"\bthis month\b|\bcurrent month\b|\bin the current month\b", low):
        add("this month", _month_text(today))
    if re.search(r"\bnext month\b", low):
        d = _shift_month(today, 1)
        add("next month", _month_text(d))
    if re.search(r"\blast month\b|\bprevious month\b", low):
        d = _shift_month(today, -1)
        add("last month", _month_text(d))
    if re.search(r"\btoday\b|\bnow\b(?! that)", low) and re.search(r"holiday|exam|lecture|class|deadline|due|open|closed|schedule|date", low):
        add("today", today.strftime("%d-%b-%y") + f" {_MONTHS[today.month - 1]} {today.year}")
    if re.search(r"\btomorrow\b", low):
        t = today + timedelta(days=1)
        add("tomorrow", t.strftime("%d-%b-%y") + f" {_MONTHS[t.month - 1]} {t.year}")
    if re.search(r"\bthis (calendar |academic )?year\b|\bcurrent year\b", low):
        add("this year", str(today.year))
    if re.search(r"\bnext year\b", low):
        add("next year", str(today.year + 1))
    if re.search(r"\b(next|upcoming) (holiday|exam|test|vacation)\b", low):
        notes.append("'next/upcoming' -> compared against today's date by the LLM")

    ctx = today.strftime("Today is %A, %d %B %Y.") if notes else None
    rq = query if not extra else f"{query} ({'; '.join(extra)})"
    return RewriteResult(original=query, retrieval_query=rq, date_context=ctx, notes=notes)


def expand_terms(query: str) -> tuple[str, List[str]]:
    added: List[str] = []
    for pattern, canonical in _GLOSSARY.items():
        if re.search(pattern, query, flags=re.I) and canonical.lower() not in query.lower():
            added.append(canonical)
    if not added:
        return query, []
    return f"{query} ({'; '.join(added)})", [f"glossary: + {', '.join(added)}"]


# Style / format instructions ("explain like I'm 5", "in simple words", "end your answer with X") say HOW to answer, not WHAT to
# look up. They are removed from the retrieval query (they only add noise for BM25 / the cross-encoder) but are kept in the
# question that is sent to the LLM, which follows them.
_STYLE_PATTERNS = [
    r"\b(and\s+)?(explain|tell|describe|answer|say)\s+(it|this|that|me)?\s*(to me\s+)?(like|as if|as though|as)\s+(i\s*am|i'?m|im|i\s+was|i\s+were|you('re| are)\s+explaining\s+to)\s+(a|an)?\s*[\w-]+(\s+year[- ]old)?(\s+\w+)?",
    r"\b(like|as if|as though|as)\s+(i\s*am|i'?m|im)\s+(a|an|5|five)\s*[\w-]*(\s+year[- ]old)?(\s+child|\s+kid)?",
    r"^\s*(please\s+)?(explain|tell|describe|show|give)(\s+(me|us|to me))?(\s+about)?\b",   # leading request verb
    r"\beli5\b", r"\bfor (a )?(dummies|beginners|kids?|children)\b",
    r"\bin (very |really )?(simple|easy|plain|layman'?s?|basic)( english| hindi)?( words| terms| language| way)?\b",
    r"\b(simply|simplified)\b", r"\bstep[- ]by[- ]step\b", r"\bin (short|brief|detail|depth|points|bullets?|bullet points|a table|table form|hindi|marathi)\b",
    r"\b(briefly|in detail|with analogy)\b",
    r",?\s*\bwith (a |an |some )?([\w-]+ ){0,3}(examples?|analogy|analogies)\b",
    r"\b(end|finish|start|begin)\s+(your\s+)?(answer|reply|response)\s+with\b.*$",
    r"\b(please|pls|kindly)\b",
]


def strip_style_instructions(query: str) -> tuple[str, List[str]]:
    core, removed = query, []
    for pat in _STYLE_PATTERNS:
        m = re.search(pat, core, flags=re.I)
        if m:
            removed.append(m.group(0).strip())
            core = (core[:m.start()] + " " + core[m.end():])
    core = re.sub(r"\s+", " ", core)
    core = re.sub(r"\s+([,.?;:])", r"\1", core)
    core = re.sub(r"([?.])\s*[,;:]+", r"\1", core).strip(" ,.;:-")
    core = re.sub(r"^(and|also)\s+", "", core, flags=re.I)
    if len(re.findall(r"\w+", core)) < 3:      # nothing meaningful left -> keep the original query
        return query, []
    return core, removed


def rewrite_query(query: str, today: Optional[date] = None, expand: bool = True) -> RewriteResult:
    if not expand:
        return RewriteResult(original=query, retrieval_query=query)
    core, removed = strip_style_instructions(query)
    res = resolve_relative_dates(core, today)
    res.original = query
    res.notes += [f"style instruction removed from retrieval: '{r}'" for r in removed]
    expanded, notes = expand_terms(res.retrieval_query)
    res.retrieval_query = expanded
    res.notes += notes
    return res
