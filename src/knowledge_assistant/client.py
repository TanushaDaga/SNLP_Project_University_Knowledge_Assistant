"""
knowledge_assistant.client
`UniversityKnowledgeEngine` - the reusable SDK entry point (Week 6).

    from knowledge_assistant import UniversityKnowledgeEngine
    engine = UniversityKnowledgeEngine()
    hits = engine.search("attendance condonation", top_k=3)      # -> List[SearchResult]
    resp = engine.ask("What is the minimum attendance?")          # -> AssistantResponse

Pipeline of `ask()`:
    injection guard -> hybrid retrieval (BM25 + FAISS -> RRF -> cross-encoder) -> calibrated
    confidence gate -> grounded generation (Groq LLM, extractive fallback) -> citation
    verification (quote must be found verbatim in the retrieved text) -> numeric-grounding check
    -> final confidence gate.
"""

from __future__ import annotations

import logging
import re
import time
from typing import List, Optional

from .config import FALLBACK_SENTENCE, REFUSAL_MESSAGE, RETRIEVAL_METHOD, EngineConfig
from .generation import LLMGenerator, extractive_answer
from .guardrails import (Calibrator, best_sentences, detect_injection, ungrounded_numbers,
                         verify_quote)
from .query_rewrite import rewrite_query
from .retrieval import HybridRetriever
from .schemas import AssistantResponse, RetrievalMetadata, SearchResult

log = logging.getLogger(__name__)


class EngineNotReady(RuntimeError):
    """Raised when a required model / index could not be loaded."""


class UniversityKnowledgeEngine:
    def __init__(self, config: Optional[EngineConfig] = None, retriever: Optional[HybridRetriever] = None,
                 generator: Optional[object] = None, calibrator: Optional[Calibrator] = None):
        self.cfg = config or EngineConfig()
        self._start = time.time()
        try:
            self.retriever = retriever or HybridRetriever(self.cfg)
        except Exception as exc:
            raise EngineNotReady(f"Retriever could not be initialised: {exc}") from exc
        if self.retriever.cross_encoder is None:
            raise EngineNotReady("Cross-encoder is not loaded; confidence cannot be computed.")
        self.generator = generator if generator is not None else LLMGenerator(self.cfg)
        self.calibrator = calibrator or Calibrator.load(self.cfg.calibration_path)
        if not self.calibrator.fitted:
            log.warning("No calibration file at %s - using the raw sigmoid of the cross-encoder logit. "
                        "Run `python src/calibrate_confidence.py`.", self.cfg.calibration_path)

    # ───────────────────────── properties used by /health ─────────────────────────

    @property
    def index_size(self) -> int:
        return self.retriever.index_size

    @property
    def embedding_model_name(self) -> str:
        return self.cfg.embedding_model

    @property
    def llm_available(self) -> bool:
        return bool(getattr(self.generator, "available", False))

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start

    # ───────────────────────── search ─────────────────────────

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        chunks, _ = self.retriever.retrieve(query, top_k=top_k)
        return [self._to_result(i, c) for i, c in enumerate(chunks, 1)]

    @staticmethod
    def _to_result(rank: int, c: dict) -> SearchResult:
        return SearchResult(rank=rank, document=c.get("doc_path") or c.get("source_document", ""),
                            text_snippet=c["text"][:300], similarity_score=float(c.get("score", 0.0)),
                            chunk_id=c.get("chunk_id"), page_number=c.get("page"),
                            category=c.get("category"), rrf_score=c.get("rrf_score"))

    # ───────────────────────── ask ─────────────────────────

    def _refuse(self, query: str, reason: str, confidence: float, t_start: float,
                meta: Optional[RetrievalMetadata] = None, results: Optional[List[SearchResult]] = None
                ) -> AssistantResponse:
        meta = meta or RetrievalMetadata()
        meta.total_latency_ms = (time.perf_counter() - t_start) * 1e3
        return AssistantResponse(query=query, answer=REFUSAL_MESSAGE, supporting_document=None,
                                 source_section=None, confidence_score=round(float(confidence), 4),
                                 retrieval_metadata=meta, is_fallback=True, refusal_reason=reason,
                                 search_results=results or [])

    def ask(self, query: str, enforce_grounding: bool = True, top_k: int = 3, use_cache: bool = True
            ) -> AssistantResponse:
        t_start = time.perf_counter()
        query = (query or "").strip()
        if not query:
            raise ValueError("query must not be empty")

        # 1. security guard (always on)
        guard = detect_injection(query)
        if guard:
            return self._refuse(query, f"injection:{guard}", 0.0, t_start)

        # 2. query understanding (relative dates / spelling variants) and retrieval
        rw = rewrite_query(query, expand=self.cfg.query_expansion)
        chunks, tm = self.retriever.retrieve(rw.retrieval_query, top_k=top_k)
        meta = RetrievalMetadata(retrieval_latency_ms=tm["retrieval_ms"], rerank_latency_ms=tm["rerank_ms"],
                                 n_candidates=tm["n_candidates"],
                                 interpreted_query=rw.retrieval_query if rw.changed else None)
        if not chunks:
            return self._refuse(query, "no_results", 0.0, t_start, meta)
        results = [self._to_result(i, c) for i, c in enumerate(chunks, 1)]

        # 3. retrieval-confidence gate (calibrated cross-encoder score of the best chunk)
        base_conf = self.calibrator(chunks[0]["cross_encoder_score"])
        if enforce_grounding and base_conf < self.cfg.confidence_threshold:
            return self._refuse(query, "low_confidence", base_conf, t_start, meta, results)

        # 4. generation
        t_gen = time.perf_counter()
        gen = self._generate(query, chunks, rw.date_context, use_cache)
        meta.generation_latency_ms = (time.perf_counter() - t_gen) * 1e3
        meta.generation_mode = gen["mode"]
        meta.model = gen.get("model")
        meta.llm_tokens = gen.get("tokens") if gen["mode"].startswith("llm") else None
        meta.llm_live_latency_ms = gen.get("latency_ms") if gen["mode"].startswith("llm") else None
        meta.llm_wait_ms = gen.get("wait_ms", 0.0) if gen["mode"] == "llm" else 0.0
        passages: List[str] = gen["passages"]

        gen["answer"] = re.sub(r"\s*【[^】]*】", "", gen["answer"]).strip()   # drop the model's own 【3】 markers

        if not gen["answerable"] or FALLBACK_SENTENCE.lower()[:60] in gen["answer"].lower():
            return self._refuse(query, "llm_declined", base_conf, t_start, meta, results)

        # 5. citation verification (verbatim quote -> document)
        cited_idx, quote, verified = None, "", False
        for ev in gen["evidence"]:
            q = str(ev.get("quote") or "")
            claimed = ev.get("passage")
            ordered = ([claimed - 1] if isinstance(claimed, int) and 1 <= claimed <= len(passages) else [])
            ok, idx = verify_quote(q, [passages[i] for i in ordered]) if ordered else (False, None)
            if ok:
                idx = ordered[0]
            else:
                ok, idx = verify_quote(q, passages)
            if ok:
                cited_idx, quote, verified = idx, q.strip(), True
                break
        if not verified:  # repair: take the most relevant verbatim sentence instead of an unverified quote
            quote, cited_idx = best_sentences(f"{gen['answer']} {query}", passages, n=2)
            cited_idx = cited_idx if cited_idx is not None else 0
        chunk = chunks[cited_idx]

        # 6. numeric grounding
        bad_numbers = ungrounded_numbers(gen["answer"], passages, query)

        confidence = base_conf * (1.0 if verified else 0.8) * (0.7 if bad_numbers else 1.0)
        meta.total_latency_ms = (time.perf_counter() - t_start) * 1e3
        if enforce_grounding and confidence < self.cfg.confidence_threshold:
            return self._refuse(query, "low_confidence", confidence, t_start, meta, results)

        return AssistantResponse(
            query=query, answer=gen["answer"],
            supporting_document=chunk.get("doc_path") or chunk.get("source_document"),
            source_section=quote[:600], confidence_score=round(float(confidence), 4),
            retrieval_metadata=meta, is_fallback=False, evidence_verified=verified,
            ungrounded_numbers=bad_numbers, search_results=results)

    def _generate(self, query: str, chunks: List[dict], date_context: Optional[str] = None,
                  use_cache: bool = True) -> dict:
        if getattr(self.generator, "available", False):
            try:
                extra = {}
                if date_context:
                    extra["date_context"] = date_context
                if not use_cache:
                    extra["use_cache"] = False
                return self.generator.generate(query, chunks, **extra)
            except Exception as exc:  # quota exhausted, network, malformed JSON ...
                log.warning("LLM generation failed (%s) - using extractive fallback", type(exc).__name__)
                out = extractive_answer(query, chunks, self.cfg.max_context_chars)
                out["mode"] = "extractive"
                out["llm_error"] = f"{type(exc).__name__}: {exc}"[:200]
                return out
        return extractive_answer(query, chunks, self.cfg.max_context_chars)
