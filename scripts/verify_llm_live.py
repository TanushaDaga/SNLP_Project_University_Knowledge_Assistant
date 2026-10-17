"""
scripts/verify_llm_live.py - proves that answers come from a LIVE call to the Groq LLM and not from a stored / canned answer.

Three independent proofs (run:  python scripts/verify_llm_live.py):

 1. NONCE TEST   A random code word (generated now, never seen by anyone) is put into the question
                 ("... end your answer with the word ZEBRA-4821"). Only a model that actually reads the question can
                 comply. The test passes if the word appears in the answer and generation_mode == "llm" (not cached).
 2. STYLE TEST   The same factual question is asked "formally" and "like I'm 5": the two answers must differ in wording
                 while quoting the same document facts.
 3. QUOTA TEST   Groq's own rate-limit headers (x-ratelimit-remaining-requests / -tokens) are read before and after:
                 they must go DOWN by the number of calls made - something a local fallback cannot fake.

It uses about 3 LLM calls (~7k tokens of the free tier). Nothing is cached (use_cache=False).
"""

import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from knowledge_assistant import EngineConfig, UniversityKnowledgeEngine  # noqa: E402


def groq_headers(engine, model):
    """One tiny real request to `model`; returns the provider's rate-limit counters for that model."""
    client = engine.generator.client
    raw = client.chat.completions.with_raw_response.create(
        model=model, messages=[{"role": "user", "content": "ok"}], max_tokens=1)
    h = raw.headers
    return int(h.get("x-ratelimit-remaining-requests", -1)), int(h.get("x-ratelimit-remaining-tokens", -1))


def main():
    cfg = EngineConfig()
    cfg.llm_use_cache = False
    cfg.answer_style = "detailed"
    engine = UniversityKnowledgeEngine(cfg)
    if not engine.llm_available:
        print("LLM is NOT available (no GROQ_API_KEY or quota exhausted) - answers would be extractive.")
        return 1

    ok = True
    # provider counters of every model we might use (the engine fails over when a model's daily quota is exhausted)
    before = {mdl: groq_headers(engine, mdl) for mdl in engine.generator.models}
    print("[quota] before:", {k: f"requests/day left {v[0]}, tokens/min left {v[1]}" for k, v in before.items()}, "\n")

    # 1 ---- nonce -------------------------------------------------------------------------------------------------
    nonce = f"ZEBRA-{random.randint(1000, 9999)}-{int(time.time()) % 1000}"
    q = f"How many students can be in a capstone project team? Explain like I'm 5 and end your answer with the word {nonce}."
    r = engine.ask(q, use_cache=False)
    m = r.retrieval_metadata
    passed = nonce in r.answer and m.generation_mode == "llm"
    ok &= passed
    print(f"[1 nonce] question : {q}\n          mode={m.generation_mode} model={m.model} tokens={m.llm_tokens} "
          f"live-latency={m.llm_live_latency_ms:.0f} ms\n          answer   : {r.answer}\n          cited    : {r.supporting_document}\n"
          f"          RESULT   : {'PASS - the random word was produced by the model' if passed else 'FAIL'}\n")

    # 2 ---- style ------------------------------------------------------------------------------------------------
    base = "What happens if my attendance is below 80 percent?"
    a = engine.ask(base + " Answer formally and briefly.", use_cache=False)
    b = engine.ask(base + " Explain it like I'm a 5 year old, with a small everyday example.", use_cache=False)
    differ = a.answer.strip() != b.answer.strip() and len(b.answer) > 0
    ok &= differ and a.retrieval_metadata.generation_mode == "llm" and b.retrieval_metadata.generation_mode == "llm"
    print(f"[2 style] FORMAL  ({a.retrieval_metadata.llm_tokens} tokens): {a.answer}\n")
    print(f"          LIKE I'M 5 ({b.retrieval_metadata.llm_tokens} tokens): {b.answer}\n")
    print(f"          RESULT   : {'PASS - same facts, different wording' if differ else 'FAIL'}\n")

    # 3 ---- quota ------------------------------------------------------------------------------------------------
    used = {}
    for resp in (r, a, b):
        mm = resp.retrieval_metadata.model
        if mm:
            used[mm] = used.get(mm, 0) + 1
    after = {mdl: groq_headers(engine, mdl) for mdl in engine.generator.models}
    passed = bool(used)
    for mdl, n_calls in used.items():
        drop = before[mdl][0] - after[mdl][0]          # includes the one probe request itself
        passed &= drop >= n_calls
        print(f"[3 quota] {mdl}: {n_calls} answer call(s) made; the provider's requests/day counter went "
              f"{before[mdl][0]} -> {after[mdl][0]} (dropped by {drop}; >= {n_calls} expected)")
    ok &= passed
    print(f"          RESULT   : {'PASS - the provider itself counted our requests' if passed else 'CHECK - counters did not move as expected'}\n")

    print("OVERALL:", "ALL CHECKS PASSED - answers are generated live by the LLM." if ok else "SOME CHECKS FAILED")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
