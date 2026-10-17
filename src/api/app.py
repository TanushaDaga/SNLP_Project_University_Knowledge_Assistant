"""
src/api/app.py - University Knowledge Assistant REST API (Week 6)

    uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health          liveness + index size + embedding model + uptime (always 200)
    GET  /ready           readiness (503 until the engine has loaded)
    POST /api/v1/search   ranked chunks with document path, page and cross-encoder score
    POST /api/v1/ask      grounded answer + supporting_document + source_section + confidence
Interactive docs: /docs (Swagger UI), /redoc, /openapi.json
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Deque, Dict, List, Optional

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from knowledge_assistant import EngineNotReady, UniversityKnowledgeEngine  # noqa: E402
from knowledge_assistant.config import RETRIEVAL_METHOD  # noqa: E402

log = logging.getLogger("knowledge_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

APP_VERSION = "1.0.0"
MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", 16 * 1024))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", 60))


# ───────────────────────────── schemas (Pydantic v2) ─────────────────────────────

class QueryRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "query": "What is the minimum attendance required to appear for end-semester exams?",
        "top_k": 3, "strict_mode": True}})

    query: str = Field(..., min_length=3, max_length=1000, description="Natural-language question")
    top_k: int = Field(3, ge=1, le=10, description="Number of chunks to retrieve / use as evidence")
    strict_mode: bool = Field(True, description="Apply the confidence gate (refuse below 0.35)")
    use_cache: bool = Field(True, description="false = never reuse a stored LLM answer; always make a live LLM call (for demos / verification)")

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("query must contain at least 3 non-blank characters")
        return v


class SearchResultItem(BaseModel):
    rank: int
    document: str = Field(..., description="Folder/file.pdf")
    text_snippet: str
    similarity_score: float = Field(..., description="Cross-encoder relevance logit (higher = better)")
    chunk_id: Optional[int] = None
    page_number: Optional[int] = None
    category: Optional[str] = None
    rrf_score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: List[SearchResultItem]
    retrieval_latency_ms: float
    method: str = RETRIEVAL_METHOD


class RetrievalMetadataModel(BaseModel):
    retrieval_latency_ms: float
    rerank_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float
    method: str = RETRIEVAL_METHOD
    n_candidates: int = 0
    generation_mode: str = "none"
    model: Optional[str] = None
    llm_live_latency_ms: Optional[float] = None
    llm_wait_ms: float = 0.0
    llm_tokens: Optional[int] = None
    interpreted_query: Optional[str] = None


class AssistantResponseModel(BaseModel):
    query: str
    answer: str
    supporting_document: Optional[str] = Field(None, description="Folder/file.pdf, null on refusal")
    source_section: Optional[str] = Field(None, description="Verbatim evidence passage, null on refusal")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    retrieval_metadata: RetrievalMetadataModel
    is_fallback: bool = False
    refusal_reason: Optional[str] = None
    evidence_verified: bool = False
    ungrounded_numbers: List[str] = []


class HealthResponse(BaseModel):
    status: str
    vector_index_size: int
    embedding_model: str
    cross_encoder_model: str
    llm_available: bool
    confidence_calibrated: bool
    uptime_seconds: float
    version: str


class ErrorResponse(BaseModel):
    error_code: str
    detail: str
    request_id: Optional[str] = None


# ───────────────────────────── state & lifecycle ─────────────────────────────

_START = time.time()
_engine_lock = threading.Lock()   # model inference is CPU bound -> serialise requests


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = None
    app.state.engine_error = None
    try:
        app.state.engine = UniversityKnowledgeEngine()
        log.info("Engine ready: %d vectors, LLM available=%s", app.state.engine.index_size,
                 app.state.engine.llm_available)
    except Exception as exc:  # keep the process alive so /health can report the problem
        app.state.engine_error = f"{type(exc).__name__}: {exc}"
        log.error("Engine failed to initialise: %s", app.state.engine_error)
    yield


app = FastAPI(
    title="University Knowledge Assistant API",
    version=APP_VERSION,
    description=(
        "Grounded question answering over official NMIMS MPSTME policy documents "
        "(Hybrid BM25 + FAISS -> RRF -> cross-encoder -> citation-verified LLM answer).\n\n"
        "Peer integrations: Group 10 (Admin Copilot) and Group 8 (Placement Intelligence) - "
        "see docs/API.md."),
    lifespan=lifespan,
    responses={422: {"model": ErrorResponse}, 429: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

_hits: Dict[str, Deque[float]] = defaultdict(deque)


def _err(status: int, code: str, detail: str, request: Optional[Request] = None, headers: Optional[dict] = None):
    rid = getattr(getattr(request, "state", None), "request_id", None) if request else None
    return JSONResponse(status_code=status, content={"error_code": code, "detail": detail, "request_id": rid},
                        headers=headers)


@app.middleware("http")
async def instrumentation(request: Request, call_next):
    """Request id, latency header, payload-size guard and per-client rate limit."""
    request.state.request_id = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()

    if request.method == "POST":
        try:
            length = int(request.headers.get("content-length", 0))
        except ValueError:
            length = 0
        if length > MAX_BODY_BYTES:
            return _err(413, "payload_too_large", f"Body exceeds {MAX_BODY_BYTES} bytes.", request)
        client = request.client.host if request.client else "unknown"
        now = time.time()
        q = _hits[client]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT_PER_MIN:
            retry = max(1, int(60 - (now - q[0])))
            return _err(429, "rate_limited", f"Limit is {RATE_LIMIT_PER_MIN} requests/minute.", request,
                        {"Retry-After": str(retry)})
        q.append(now)

    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1e3
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Process-Time-ms"] = f"{elapsed:.2f}"
    log.info("%s %s -> %s in %.1f ms [%s]", request.method, request.url.path, response.status_code, elapsed,
             request.state.request_id)
    return response


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    msgs = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors())
    return _err(422, "validation_error", msgs, request)


@app.exception_handler(StarletteHTTPException)
async def _http_handler(request: Request, exc: StarletteHTTPException):
    return _err(exc.status_code, "http_error", str(exc.detail), request, getattr(exc, "headers", None))


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    log.exception("Unhandled error on %s", request.url.path)
    return _err(500, "internal_error", "Internal server error. The incident has been logged.", request)


def _engine(request: Request) -> UniversityKnowledgeEngine:
    eng = getattr(request.app.state, "engine", None)
    if eng is None:
        raise HTTPException(503, f"Knowledge engine unavailable: {request.app.state.engine_error}")
    return eng


# ───────────────────────────── endpoints ─────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Operations"], summary="Liveness / service info")
def health(request: Request):
    eng = getattr(request.app.state, "engine", None)
    if eng is None:
        return HealthResponse(status=f"degraded: {request.app.state.engine_error}", vector_index_size=0,
                              embedding_model="unavailable", cross_encoder_model="unavailable",
                              llm_available=False, confidence_calibrated=False,
                              uptime_seconds=round(time.time() - _START, 2), version=APP_VERSION)
    return HealthResponse(status="healthy", vector_index_size=eng.index_size,
                          embedding_model=eng.embedding_model_name,
                          cross_encoder_model=eng.cfg.cross_encoder_model, llm_available=eng.llm_available,
                          confidence_calibrated=eng.calibrator.fitted,
                          uptime_seconds=round(time.time() - _START, 2), version=APP_VERSION)


@app.get("/ready", tags=["Operations"], summary="Readiness probe (503 until models are loaded)")
def ready(request: Request):
    _engine(request)
    return {"ready": True}


@app.post("/api/v1/search", response_model=SearchResponse, tags=["Retrieval"],
          summary="Hybrid retrieval (no generation, no LLM quota used)")
def search(body: QueryRequest, request: Request):
    eng = _engine(request)
    t0 = time.perf_counter()
    with _engine_lock:
        results = eng.search(body.query, top_k=body.top_k)
    return SearchResponse(
        query=body.query, top_k=body.top_k,
        results=[SearchResultItem(**{**r.to_dict(), "similarity_score": round(r.similarity_score, 5)})
                 for r in results],
        retrieval_latency_ms=round((time.perf_counter() - t0) * 1e3, 2))


@app.post("/api/v1/ask", response_model=AssistantResponseModel, tags=["Question Answering"],
          summary="Grounded answer with verified citation")
def ask(body: QueryRequest, request: Request):
    """Pipeline: injection guard -> BM25+FAISS -> RRF -> cross-encoder -> confidence gate -> LLM ->
    citation verification. On refusal (`is_fallback=true`) `supporting_document` and `source_section`
    are `null` and `confidence_score` is the calibrated retrieval confidence (< 0.35 for low-confidence refusals)."""
    eng = _engine(request)
    with _engine_lock:
        resp = eng.ask(body.query, enforce_grounding=body.strict_mode, top_k=body.top_k, use_cache=body.use_cache)
    m = resp.retrieval_metadata
    return AssistantResponseModel(
        query=resp.query, answer=resp.answer, supporting_document=resp.supporting_document,
        source_section=resp.source_section, confidence_score=min(1.0, max(0.0, resp.confidence_score)),
        retrieval_metadata=RetrievalMetadataModel(
            retrieval_latency_ms=round(m.retrieval_latency_ms, 2), rerank_latency_ms=round(m.rerank_latency_ms, 2),
            generation_latency_ms=round(m.generation_latency_ms, 2), total_latency_ms=round(m.total_latency_ms, 2),
            method=m.method, n_candidates=m.n_candidates, generation_mode=m.generation_mode, model=m.model,
            llm_live_latency_ms=None if m.llm_live_latency_ms is None else round(m.llm_live_latency_ms, 2),
            llm_wait_ms=round(m.llm_wait_ms, 2), llm_tokens=m.llm_tokens, interpreted_query=m.interpreted_query),
        is_fallback=resp.is_fallback, refusal_reason=resp.refusal_reason,
        evidence_verified=resp.evidence_verified, ungrounded_numbers=resp.ungrounded_numbers)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000)
