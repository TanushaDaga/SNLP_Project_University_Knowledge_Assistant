"""
knowledge_assistant.config
Central configuration: paths, model names, thresholds. Everything can be overridden
with environment variables so the package can be deployed without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")  # never prints or logs the key

RETRIEVAL_METHOD = "Hybrid_BM25_Dense_RRF_CrossEncoder"
FALLBACK_SENTENCE = (
    "The official documentation does not contain sufficient details to answer this query."
)
REFUSAL_MESSAGE = (
    "The official university documentation does not contain sufficient details to answer this "
    "query. Please contact the Academic Office or the relevant department for assistance."
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class EngineConfig:
    corpus_path: Path = PROJECT_ROOT / "Data" / "processed" / "corpus.json"
    index_dir: Path = PROJECT_ROOT / "Data" / "processed"
    calibration_path: Path = PROJECT_ROOT / "Data" / "processed" / "calibration.json"
    llm_cache_path: Path = PROJECT_ROOT / "Data" / "cache" / "llm_cache.json"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # retrieval
    candidates_per_retriever: int = 25     # top-N from BM25 and from FAISS
    rrf_k: int = 60                        # Reciprocal Rank Fusion constant
    rerank_depth: int = 20                 # fused candidates scored by the cross-encoder
    expand_neighbours: bool = True         # contextual window expansion (prev/next chunk)
    max_context_chars: int = 9000          # budget for the LLM context (free-tier friendly)

    # guardrails
    confidence_threshold: float = 0.35     # below this the engine refuses

    # generation (Groq free tier -> conservative defaults)
    llm_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    # Free-tier quotas are per model: when the primary model's daily quota is used up the engine fails over to this
    # one (set GROQ_FALLBACK_MODEL="" to disable). Every response records which model produced it.
    llm_fallback_model: str = field(default_factory=lambda: os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b"))
    llm_enabled: bool = field(default_factory=lambda: os.getenv("USE_LLM", "1") not in ("0", "false", "False"))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("GROQ_MAX_TOKENS", 1600))
    # "adaptive" (default): short question -> short answer, explanation question -> structured answer;
    # "detailed": always the structured bullet-point format;
    # "concise": 1-4 sentences - the style that was used for the Week 5 benchmark numbers (cached answers).
    answer_style: str = field(default_factory=lambda: os.getenv("ANSWER_STYLE", "adaptive"))
    # resolve "this month / today / next year" against the system clock and expand a few spelling variants
    query_expansion: bool = field(default_factory=lambda: os.getenv("QUERY_EXPANSION", "1") not in ("0", "false", "False"))
    llm_min_interval_s: float = field(default_factory=lambda: _env_float("GROQ_MIN_INTERVAL_S", 2.0))
    llm_max_retries: int = field(default_factory=lambda: _env_int("GROQ_MAX_RETRIES", 3))
    llm_use_cache: bool = field(default_factory=lambda: os.getenv("LLM_CACHE", "1") not in ("0", "false", "False"))

    def api_key(self) -> str | None:
        return os.getenv("GROQ_API_KEY") or os.getenv("API_KEY")
