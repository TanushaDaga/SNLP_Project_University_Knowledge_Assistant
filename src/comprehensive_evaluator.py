"""
src/comprehensive_evaluator.py - Week 5 automated evaluation harness.

Runs every query of evaluation/expanded_test_suite.json through four systems

    BM25                  sparse baseline (Weeks 1-2)           retrieval metrics
    Dense                 MiniLM + FAISS (Week 3)               retrieval metrics
    Hybrid_RRF            BM25 + Dense fused with RRF only      retrieval metrics (ablation)
    Hybrid_RRF_CE         + cross-encoder re-rank (Week 4/6)    retrieval metrics AND full RAG metrics
                                                                (engine.ask: guardrails + LLM + citation check)

and writes the *measured* numbers to evaluation/week5_quantitative_metrics.json.
Nothing in this file is hard-coded: every figure is an aggregate of per-query records that are saved
next to it.  LLM answers are cached on disk (Data/cache/llm_cache.json), so re-running costs no quota.

    python src/comprehensive_evaluator.py                 # full run (uses the Groq key for <= ~45 calls)
    python src/comprehensive_evaluator.py --no-llm        # extractive answers only, zero API usage
    python src/comprehensive_evaluator.py --limit 5       # smoke test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from knowledge_assistant import EngineConfig, UniversityKnowledgeEngine  # noqa: E402
from knowledge_assistant.guardrails import _squash  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("evaluator")

SUITE_PATH = ROOT / "evaluation" / "expanded_test_suite.json"
OUTPUT_PATH = ROOT / "evaluation" / "week5_quantitative_metrics.json"
BUCKETS = {1: "Lexical / Specific Terminology", 2: "Semantic / Conversational Paraphrase",
           3: "Multi-Constraint / Conditional", 4: "Cross-Document / Semantic Overlap",
           5: "Adversarial / Out-of-Domain / Negative"}


# ───────────────────────────── metric primitives ─────────────────────────────

def first_relevant_rank(docs: List[str], acceptable: List[str]) -> Optional[int]:
    for i, d in enumerate(docs, 1):
        if d in acceptable:
            return i
    return None


def hit_at_k(docs, acceptable, k) -> bool:
    r = first_relevant_rank(docs[:k], acceptable)
    return r is not None


def mrr_at_k(docs, acceptable, k) -> float:
    r = first_relevant_rank(docs[:k], acceptable)
    return 1.0 / r if r else 0.0


def precision_at_k(docs, acceptable, k) -> float:
    """Share of the k slots occupied by chunks of a relevant document (missing results count as misses)."""
    return sum(1 for d in docs[:k] if d in acceptable) / k


def recall_at_k(docs, gold: List[str], k) -> float:
    """Share of gold documents that appear in the top-k (cross-document queries have 2 gold docs)."""
    if not gold:
        return 0.0
    return len(set(gold) & set(docs[:k])) / len(gold)


def _norm(text: str) -> str:
    return _squash(text.replace("’", "'"))


def fact_present(fact: str, text: str) -> bool:
    """A key fact may list alternatives separated by ' || '; any one suffices. Matching ignores case,
    spacing and punctuation (so '75 %' == '75%')."""
    t = _norm(text)
    return any(_norm(alt) in t for alt in fact.split("||") if _norm(alt))


def fact_coverage(facts: List[str], text: str) -> float:
    if not facts:
        return 1.0
    return sum(fact_present(f, text) for f in facts) / len(facts)


def evidence_coverage(phrases: List[str], texts: List[str]) -> float:
    if not phrases:
        return 1.0
    joined = " ".join(texts)
    return sum(fact_present(p, joined) for p in phrases) / len(phrases)


def wilson(k: int, n: int, z: float = 1.96) -> Optional[List[float]]:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def pct(vals: List[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def mean(vals) -> float:
    vals = list(vals)
    return sum(vals) / len(vals) if vals else 0.0


# ───────────────────────────── per-query records ─────────────────────────────

def acceptable_docs(tc: dict) -> List[str]:
    docs = list(tc.get("acceptable_documents") or [])
    if tc.get("expected_document") and tc["expected_document"] not in docs:
        docs.append(tc["expected_document"])
    return docs


def retrieval_record(tc: dict, chunks: List[dict], latency_ms: float) -> dict:
    acc = acceptable_docs(tc)
    gold = list(tc.get("gold_documents") or ([tc["expected_document"]] if tc.get("expected_document") else []))
    docs = [c.get("doc_path") or c.get("source_document") for c in chunks]
    ev = tc.get("evidence_phrases", [])
    rec = {"retrieved_docs": docs[:5], "retrieved_chunk_ids": [c["chunk_id"] for c in chunks[:5]],
           "retrieval_latency_ms": latency_ms}
    if tc["is_answerable"]:
        rec.update({
            "first_relevant_rank": first_relevant_rank(docs, acc),
            "hit@1": hit_at_k(docs, acc, 1), "hit@3": hit_at_k(docs, acc, 3), "hit@5": hit_at_k(docs, acc, 5),
            "mrr@3": mrr_at_k(docs, acc, 3), "precision@3": precision_at_k(docs, acc, 3),
            "recall@3": recall_at_k(docs, gold, 3),
            "evidence_coverage@3": evidence_coverage(ev, [c["text"] for c in chunks[:3]]),
            "evidence_hit@3": evidence_coverage(ev, [c["text"] for c in chunks[:3]]) == 1.0 if ev else None,
        })
    return rec


def run_retrieval_system(name: str, fn, suite: List[dict]) -> List[dict]:
    out = []
    for tc in suite:
        t0 = time.perf_counter()
        try:
            chunks = fn(tc["query"])
        except Exception as exc:
            log.warning("[%s] %s failed: %s", name, tc["query_id"], exc)
            chunks = []
        ms = (time.perf_counter() - t0) * 1e3
        rec = {"query_id": tc["query_id"], "bucket": tc["bucket"], "is_answerable": tc["is_answerable"]}
        rec.update(retrieval_record(tc, chunks, ms))
        out.append(rec)
    log.info("%s: retrieval pass done (%d queries)", name, len(out))
    return out


def rag_record(tc: dict, resp) -> dict:
    meta = resp.retrieval_metadata
    acc = acceptable_docs(tc)
    ev = tc.get("evidence_phrases", [])
    facts = tc.get("key_facts", [])
    refused = bool(resp.is_fallback)
    rec = {
        "query_id": tc["query_id"], "bucket": tc["bucket"], "is_answerable": tc["is_answerable"],
        "query": tc["query"], "answer": resp.answer, "supporting_document": resp.supporting_document,
        "source_section": resp.source_section, "confidence_score": resp.confidence_score,
        "is_refusal": refused, "refusal_reason": resp.refusal_reason,
        "evidence_verified": resp.evidence_verified, "ungrounded_numbers": resp.ungrounded_numbers,
        "retrieved_docs": [r.document for r in resp.search_results],
        "retrieved_chunk_ids": [r.chunk_id for r in resp.search_results],
        "retrieved_scores": [round(r.similarity_score, 4) for r in resp.search_results],
        "retrieval_latency_ms": meta.retrieval_latency_ms, "rerank_latency_ms": meta.rerank_latency_ms,
        "generation_latency_ms": meta.generation_latency_ms, "total_latency_ms": meta.total_latency_ms,
        "llm_live_latency_ms": meta.llm_live_latency_ms, "llm_wait_ms": meta.llm_wait_ms,
        "generation_mode": meta.generation_mode, "model": meta.model,
        # latency a deployment without free-tier throttling / caching would see
        "service_latency_ms": meta.retrieval_latency_ms + meta.rerank_latency_ms + (
            meta.llm_live_latency_ms if meta.llm_live_latency_ms is not None else meta.generation_latency_ms),
    }
    if tc["is_answerable"]:
        cited = resp.supporting_document
        rec.update({
            "false_refusal": refused,
            "citation_correct": (not refused) and cited in acc,
            "citation_specific": (not refused) and bool(ev) and any(
                fact_present(p, resp.source_section or "") for p in ev),
            "fact_coverage": 0.0 if refused else fact_coverage(facts, resp.answer),
            "answer_correct": (not refused) and fact_coverage(facts, resp.answer) == 1.0,
            "hallucinated_numbers": bool(resp.ungrounded_numbers) and not refused,
        })
    else:
        rec["hallucination"] = not refused          # answered something instead of refusing
    return rec


# ───────────────────────────── aggregation ─────────────────────────────

def aggregate_retrieval(recs: List[dict]) -> dict:
    ans = [r for r in recs if r["is_answerable"]]
    n = len(ans)
    out = {"n_queries": len(recs), "n_answerable": n}
    if n:
        for m in ("hit@1", "hit@3", "hit@5", "mrr@3", "precision@3", "recall@3"):
            out[m] = mean(r[m] for r in ans)
        ev = [r["evidence_hit@3"] for r in ans if r.get("evidence_hit@3") is not None]
        out["evidence_hit@3"] = mean(ev) if ev else None
        out["hit@1_ci95"] = wilson(sum(r["hit@1"] for r in ans), n)
        out["hit@3_ci95"] = wilson(sum(r["hit@3"] for r in ans), n)
    out["avg_retrieval_latency_ms"] = mean(r["retrieval_latency_ms"] for r in recs)
    out["p95_retrieval_latency_ms"] = pct([r["retrieval_latency_ms"] for r in recs], 0.95)
    return out


def aggregate_rag(recs: List[dict]) -> dict:
    ans = [r for r in recs if r["is_answerable"]]
    un = [r for r in recs if not r["is_answerable"]]
    out = {"n_queries": len(recs), "n_answerable": len(ans), "n_unanswerable": len(un)}
    if ans:
        out.update({
            "citation_accuracy": mean(r["citation_correct"] for r in ans),
            "citation_accuracy_ci95": wilson(sum(r["citation_correct"] for r in ans), len(ans)),
            "citation_specificity": mean(r["citation_specific"] for r in ans),
            "fact_coverage": mean(r["fact_coverage"] for r in ans),
            "answer_correctness": mean(r["answer_correct"] for r in ans),
            "answer_correctness_ci95": wilson(sum(r["answer_correct"] for r in ans), len(ans)),
            "false_refusal_rate": mean(r["false_refusal"] for r in ans),
            "ungrounded_number_rate": mean(r["hallucinated_numbers"] for r in ans),
        })
    if un:
        out["hallucination_rate"] = mean(r["hallucination"] for r in un)
        out["hallucination_rate_ci95"] = wilson(sum(r["hallucination"] for r in un), len(un))
        out["correct_refusal_rate"] = 1.0 - out["hallucination_rate"]
    for stage in ("retrieval", "rerank", "generation", "total"):
        vals = [r[f"{stage}_latency_ms"] for r in recs]
        out[f"avg_{stage}_latency_ms"] = mean(vals)
        out[f"p50_{stage}_latency_ms"] = pct(vals, 0.5)
        out[f"p95_{stage}_latency_ms"] = pct(vals, 0.95)
    out["avg_service_latency_ms"] = mean(r["service_latency_ms"] for r in recs)
    out["p50_service_latency_ms"] = pct([r["service_latency_ms"] for r in recs], 0.5)
    out["p95_service_latency_ms"] = pct([r["service_latency_ms"] for r in recs], 0.95)
    out["total_llm_wait_ms"] = sum(r.get("llm_wait_ms", 0.0) for r in recs)
    live = [r["llm_live_latency_ms"] for r in recs if r.get("llm_live_latency_ms") is not None]
    out["avg_llm_live_latency_ms"] = mean(live) if live else None
    modes: Dict[str, int] = {}
    for r in recs:
        modes[r["generation_mode"]] = modes.get(r["generation_mode"], 0) + 1
    out["generation_modes"] = modes
    # per-LLM breakdown (the free-tier daily quota of the primary model can force a fail-over to the fallback model)
    by_model: Dict[str, dict] = {}
    for r in recs:
        m = r.get("model") or ("extractive" if r["generation_mode"] == "extractive" else None)
        if m is None:
            continue
        d = by_model.setdefault(m, {"n": 0, "answerable": 0, "answer_correct": 0, "citation_correct": 0})
        d["n"] += 1
        if r["is_answerable"]:
            d["answerable"] += 1
            d["answer_correct"] += int(bool(r.get("answer_correct")))
            d["citation_correct"] += int(bool(r.get("citation_correct")))
    out["by_generation_model"] = by_model
    return out


def by_bucket(recs: List[dict], fn) -> dict:
    return {str(b): fn([r for r in recs if r["bucket"] == b]) for b in BUCKETS if any(r["bucket"] == b for r in recs)}


# ───────────────────────────── main ─────────────────────────────

def integrity_check(output: dict) -> None:
    """Fail loudly if a metric is missing / NaN / out of range (anti-fabrication guard)."""
    def walk(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}", v)
        elif isinstance(obj, float):
            assert not math.isnan(obj) and not math.isinf(obj), f"INTEGRITY: {prefix} is NaN/inf"
    walk("metrics", output["systems"])
    for name, s in output["systems"].items():
        g = s["retrieval_global"]
        for m in ("hit@1", "hit@3", "hit@5", "mrr@3", "precision@3", "recall@3"):
            assert 0.0 <= g[m] <= 1.0, f"INTEGRITY: {name}.{m}={g[m]} outside [0,1]"
        assert g["hit@1"] <= g["hit@3"] <= g["hit@5"], f"INTEGRITY: {name} hit@k not monotone"


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM (extractive answers, zero API usage)")
    ap.add_argument("--limit", type=int, default=0, help="only run the first N queries (smoke test)")
    ap.add_argument("--output", default=str(OUTPUT_PATH))
    ap.add_argument("--calibration", default=None,
                    help="confidence-gate calibration JSON (default Data/processed/calibration.json)")
    args = ap.parse_args(argv)

    suite = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    if args.limit:
        suite = suite[: args.limit]
    elif len(suite) < 50:
        raise SystemExit(f"Benchmark must contain >= 50 queries, found {len(suite)}")
    log.info("Loaded %d test cases", len(suite))

    cfg = EngineConfig()
    cfg.answer_style = "concise"        # the style of the reported Week 5 numbers (and of the cached answers)
    cfg.query_expansion = False       # benchmark measured without the post-hoc query rewriting
    if args.calibration:
        cfg.calibration_path = Path(args.calibration)
    if args.no_llm:
        cfg.llm_enabled = False
    engine = UniversityKnowledgeEngine(cfg)
    rt = engine.retriever

    systems = {
        "BM25": lambda q: rt.bm25.retrieve(q, top_k=5),
        "Dense_FAISS": lambda q: rt.dense.retrieve(q, top_k=5),
        "Hybrid_RRF": lambda q: rt.retrieve(q, top_k=5, use_rerank=False, expand=False)[0],
        "Hybrid_RRF_CrossEncoder": lambda q: rt.retrieve(q, top_k=5, expand=False)[0],
    }
    retrieval: Dict[str, List[dict]] = {}
    for name, fn in systems.items():
        fn(suite[0]["query"])  # warm-up (model/tokeniser caches) so the first query is not an outlier
        retrieval[name] = run_retrieval_system(name, fn, suite)

    # ── full RAG pass (Hybrid + cross-encoder + guardrails + LLM) ──
    rag: List[dict] = []
    for i, tc in enumerate(suite, 1):
        try:
            resp = engine.ask(tc["query"], enforce_grounding=True, top_k=3)
            rec = rag_record(tc, resp)
        except Exception as exc:  # never let one query abort the run; record it honestly
            log.error("ask() crashed on %s: %s", tc["query_id"], exc)
            rec = {"query_id": tc["query_id"], "bucket": tc["bucket"], "is_answerable": tc["is_answerable"],
                   "query": tc["query"], "answer": "", "is_refusal": True, "error": f"{type(exc).__name__}: {exc}",
                   "retrieval_latency_ms": 0.0, "rerank_latency_ms": 0.0, "generation_latency_ms": 0.0,
                   "total_latency_ms": 0.0, "service_latency_ms": 0.0, "llm_wait_ms": 0.0, "generation_mode": "error", "retrieved_docs": [],
                   "llm_live_latency_ms": None, "confidence_score": 0.0, "supporting_document": None,
                   "source_section": None, "refusal_reason": "crash", "evidence_verified": False,
                   "ungrounded_numbers": []}
            if tc["is_answerable"]:
                rec.update(false_refusal=True, citation_correct=False, citation_specific=False,
                           fact_coverage=0.0, answer_correct=False, hallucinated_numbers=False)
            else:
                rec["hallucination"] = False
        rag.append(rec)
        log.info("[%d/%d] %s refusal=%s conf=%.2f doc=%s mode=%s total=%.0fms", i, len(suite), tc["query_id"],
                 rec["is_refusal"], rec.get("confidence_score", 0.0), rec.get("supporting_document"),
                 rec["generation_mode"], rec["total_latency_ms"])

    corpus_digest = hashlib.sha1(Path(cfg.corpus_path).read_bytes()).hexdigest()[:12]
    llm = getattr(engine.generator, "stats", {})
    output = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_total_queries": len(suite),
        "n_answerable": sum(t["is_answerable"] for t in suite),
        "n_unanswerable": sum(not t["is_answerable"] for t in suite),
        "benchmark_sha1": hashlib.sha1(SUITE_PATH.read_bytes()).hexdigest()[:12],
        "corpus_sha1": corpus_digest, "corpus_chunks": len(rt.corpus),
        "configuration": {"embedding_model": cfg.embedding_model, "cross_encoder": cfg.cross_encoder_model,
                          "rrf_k": cfg.rrf_k, "candidates_per_retriever": cfg.candidates_per_retriever,
                          "rerank_depth": cfg.rerank_depth, "confidence_threshold": cfg.confidence_threshold,
                          "llm_model": cfg.llm_model if engine.llm_available or llm.get("cache_hits") else None,
                          "llm_enabled": not args.no_llm, "calibration": engine.calibrator.meta or None,
                          "calibration_file": Path(cfg.calibration_path).name,
                          "answer_style": cfg.answer_style, "query_expansion": cfg.query_expansion},
        "llm_usage_this_run": llm,
        "systems": {},
    }
    for name, recs in retrieval.items():
        output["systems"][name] = {"retrieval_global": aggregate_retrieval(recs),
                                   "retrieval_per_bucket": by_bucket(recs, aggregate_retrieval),
                                   "per_query_retrieval": recs}
    output["systems"]["Hybrid_RRF_CrossEncoder"].update({
        "rag_global": aggregate_rag(rag), "rag_per_bucket": by_bucket(rag, aggregate_rag), "per_query_rag": rag})
    integrity_check(output)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    print_summary(output)
    log.info("Saved %s", args.output)
    return output


def print_summary(o: dict) -> None:
    from tabulate import tabulate
    names = list(o["systems"])
    rows = []
    for m in ("hit@1", "hit@3", "hit@5", "mrr@3", "precision@3", "recall@3", "evidence_hit@3"):
        rows.append([m] + [f"{o['systems'][n]['retrieval_global'].get(m) or 0:.3f}" for n in names])
    rows.append(["avg retrieval ms"] + [f"{o['systems'][n]['retrieval_global']['avg_retrieval_latency_ms']:.1f}"
                                        for n in names])
    print("\n" + "=" * 100 + "\nRETRIEVAL (answerable queries)\n" + "=" * 100)
    print(tabulate(rows, headers=["metric"] + names, tablefmt="github"))
    g = o["systems"]["Hybrid_RRF_CrossEncoder"]["rag_global"]
    print("\nFULL RAG (Hybrid_RRF_CrossEncoder)")
    keys = ["citation_accuracy", "citation_specificity", "fact_coverage", "answer_correctness",
            "false_refusal_rate", "hallucination_rate", "ungrounded_number_rate", "avg_retrieval_latency_ms",
            "avg_rerank_latency_ms", "avg_generation_latency_ms", "avg_total_latency_ms", "generation_modes"]
    print(tabulate([[k, g.get(k)] for k in keys], tablefmt="github"))


if __name__ == "__main__":
    main()
