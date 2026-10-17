"""
knowledge_assistant.schemas
Plain dataclasses returned by the SDK (the REST layer mirrors them with Pydantic models).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional

from .config import RETRIEVAL_METHOD


@dataclass
class SearchResult:
    """One ranked chunk."""
    rank: int
    document: str                    # "Folder/file.pdf"
    text_snippet: str
    similarity_score: float          # cross-encoder logit (higher = more relevant)
    chunk_id: Optional[int] = None
    page_number: Optional[int] = None
    category: Optional[str] = None
    rrf_score: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalMetadata:
    retrieval_latency_ms: float = 0.0     # BM25 + dense + RRF fusion
    rerank_latency_ms: float = 0.0        # cross-encoder
    generation_latency_ms: float = 0.0    # LLM / extractive generation
    total_latency_ms: float = 0.0
    method: str = RETRIEVAL_METHOD
    n_candidates: int = 0
    generation_mode: str = "none"         # llm | llm_cached | extractive | none
    model: Optional[str] = None
    llm_live_latency_ms: Optional[float] = None   # pure request time of the real API call (also when served from cache)
    llm_tokens: Optional[int] = None              # total tokens billed by the provider for this answer
    interpreted_query: Optional[str] = None       # retrieval query after date resolution / glossary expansion (None = unchanged)
    llm_wait_ms: float = 0.0                      # free-tier spacing / 429 back-off slept during this request

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AssistantResponse:
    query: str
    answer: str
    supporting_document: Optional[str]
    source_section: Optional[str]
    confidence_score: float
    retrieval_metadata: RetrievalMetadata = field(default_factory=RetrievalMetadata)
    is_fallback: bool = False
    refusal_reason: Optional[str] = None       # injection | low_confidence | llm_declined | no_results
    evidence_verified: bool = False            # cited passage was found verbatim in the retrieved text
    ungrounded_numbers: List[str] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["retrieval_metadata"] = self.retrieval_metadata.to_dict()
        d["search_results"] = [r.to_dict() for r in self.search_results]
        return d
