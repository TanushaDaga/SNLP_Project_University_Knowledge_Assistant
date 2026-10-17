"""
knowledge_assistant.retrieval
Single implementation of the three retrieval stages used from Week 1 to Week 6:

    BM25Retriever   - sparse lexical baseline (Weeks 1-2)
    DenseRetriever  - bi-encoder + FAISS IndexFlatIP (Week 3)
    HybridRetriever - BM25 + dense  ->  Reciprocal Rank Fusion  ->  cross-encoder (Week 4)

Audit note: the Week 4 `improved_rag.py` documented RRF but actually merged the two candidate
lists with a dict union (no fusion). Fusion is now really performed and its contribution
can be measured separately (`use_rerank=False`).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

if not hasattr(inspect, "formatargspec"):  # nltk shim for Python 3.14
    inspect.formatargspec = lambda *a, **k: ""  # type: ignore[attr-defined]

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from .config import EngineConfig

log = logging.getLogger(__name__)
for _noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers", "urllib3", "faiss"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


def _load_local_first(factory, name: str):
    """Load a Hugging Face model from the local cache without network round trips; download only if absent."""
    try:
        return factory(name, local_files_only=True)
    except Exception:
        log.info("Model %s not in local cache - downloading", name)
        return factory(name)


def _ensure_nltk() -> None:
    for res, path in (("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab"),
                      ("wordnet", "corpora/wordnet.zip")):
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(res, quiet=True)
            except Exception as exc:  # pragma: no cover - offline
                log.warning("NLTK resource %s unavailable: %s", res, exc)


def load_corpus(corpus_path: Path) -> List[dict]:
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found at {corpus_path}. Build it with: python src/preprocess.py")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    # Backward compatibility with the Week 4 corpus (no doc_path/category/page).
    for i, c in enumerate(corpus):
        c.setdefault("chunk_id", i)
        c.setdefault("doc_path", c.get("source_document", ""))
        c.setdefault("category", None)
        c.setdefault("page", None)
    return corpus


# ───────────────────────────── BM25 ─────────────────────────────

class BM25Retriever:
    """Okapi BM25 over lemmatised tokens."""

    def __init__(self, corpus: List[dict]):
        from rank_bm25 import BM25Okapi
        _ensure_nltk()
        self.corpus = corpus
        self._lemmatizer = WordNetLemmatizer()
        self._lemma_cache: Dict[str, str] = {}
        tokenized = [self._tokens(c["text"]) for c in corpus]
        self._bm25 = BM25Okapi(tokenized)

    def _lemma(self, w: str) -> str:
        v = self._lemma_cache.get(w)
        if v is None:
            v = self._lemmatizer.lemmatize(w)
            self._lemma_cache[w] = v
        return v

    def _tokens(self, text: str) -> List[str]:
        return [self._lemma(w) for w in word_tokenize(text.lower()) if w.isalnum()]

    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        q = self._tokens(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        order = np.argsort(scores)[::-1][:top_k]
        out = []
        for rank, idx in enumerate(order, 1):
            if scores[idx] <= 0:  # no lexical overlap at all -> not a retrieval hit
                break
            r = dict(self.corpus[idx])
            r["score"] = float(scores[idx])
            r["bm25_rank"] = rank
            out.append(r)
        return out


# ───────────────────────────── dense / FAISS ─────────────────────────────

class DenseRetriever:
    """all-MiniLM-L6-v2 embeddings in a FAISS IndexFlatIP (cosine on L2-normalised vectors)."""

    def __init__(self, corpus: List[dict], index_dir: Path, model_name: str, corpus_path: Optional[Path] = None):
        self.corpus = corpus
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self.index_path = self.index_dir / "dense_index.faiss"
        self.meta_path = self.index_dir / "dense_index_meta.json"
        self._corpus_digest = self._digest(corpus)
        self.model = None
        self.index = None

    @staticmethod
    def _digest(corpus: List[dict]) -> str:
        h = hashlib.sha1()
        for c in corpus:
            h.update(c["text"].encode("utf-8", "ignore"))
            h.update(b"\x00")
        return h.hexdigest()

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            log.info("Loading embedding model %s", self.model_name)
            self.model = _load_local_first(SentenceTransformer, self.model_name)
        return self.model

    def load_or_build(self) -> "DenseRetriever":
        import faiss
        if self.index_path.exists() and self.meta_path.exists():
            try:
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                if (meta.get("corpus_digest") == self._corpus_digest and
                        meta.get("model") == self.model_name):
                    self.index = faiss.read_index(str(self.index_path))
                    if self.index.ntotal == len(self.corpus):
                        log.info("Loaded FAISS index (%d vectors)", self.index.ntotal)
                        return self
            except Exception as exc:
                log.warning("Could not reuse FAISS index (%s) - rebuilding", exc)
        log.warning("FAISS index missing or stale for this corpus - rebuilding")
        self.build()
        return self

    def build(self) -> None:
        import faiss
        model = self._load_model()
        texts = [c["text"] for c in self.corpus]
        emb = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(emb)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps({
            "corpus_digest": self._corpus_digest, "model": self.model_name,
            "n_vectors": int(self.index.ntotal), "dim": int(emb.shape[1])}), encoding="utf-8")
        log.info("Built FAISS index with %d vectors", self.index.ntotal)

    @property
    def metadata(self) -> List[dict]:
        return self.corpus

    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        import faiss
        if self.index is None:
            self.load_or_build()
        q = self._load_model().encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, top_k)
        out = []
        for rank, (s, i) in enumerate(zip(scores[0], ids[0]), 1):
            if i < 0:
                continue
            r = dict(self.corpus[int(i)])
            r["score"] = float(s)
            r["dense_rank"] = rank
            out.append(r)
        return out


# ───────────────────────────── hybrid ─────────────────────────────

def reciprocal_rank_fusion(rankings: List[List[dict]], k: int = 60, top_n: Optional[int] = None) -> List[dict]:
    """Standard RRF: score(d) = sum_r 1 / (k + rank_r(d)). Ranks are 1-based list positions."""
    scores: Dict[int, float] = {}
    best: Dict[int, dict] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in best:
                best[cid] = dict(item)
            else:  # keep per-retriever rank annotations from both lists
                for key in ("bm25_rank", "dense_rank"):
                    if key in item:
                        best[cid][key] = item[key]
    fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if top_n:
        fused = fused[:top_n]
    out = []
    for cid, s in fused:
        d = best[cid]
        d["rrf_score"] = s
        out.append(d)
    return out


class HybridRetriever:
    """BM25 + dense -> RRF -> cross-encoder re-ranking -> neighbour-window expansion."""

    def __init__(self, config: Optional[EngineConfig] = None, load_cross_encoder: bool = True):
        self.cfg = config or EngineConfig()
        self.corpus = load_corpus(self.cfg.corpus_path)
        self.bm25 = BM25Retriever(self.corpus)
        self.dense = DenseRetriever(self.corpus, self.cfg.index_dir, self.cfg.embedding_model).load_or_build()
        self.cross_encoder = None
        if load_cross_encoder:
            from sentence_transformers import CrossEncoder
            log.info("Loading cross-encoder %s", self.cfg.cross_encoder_model)
            self.cross_encoder = _load_local_first(CrossEncoder, self.cfg.cross_encoder_model)

    # -- helpers ------------------------------------------------------------

    @property
    def index_size(self) -> int:
        return int(self.dense.index.ntotal) if self.dense.index is not None else 0

    def _neighbours(self, chunk: dict) -> tuple[str, str]:
        """Previous / next chunk of the same document (fixes tables and lists cut across chunks)."""
        cid = chunk["chunk_id"]
        doc = chunk.get("doc_path")
        prev = self.corpus[cid - 1]["text"] if cid > 0 and self.corpus[cid - 1].get("doc_path") == doc else ""
        nxt = (self.corpus[cid + 1]["text"]
               if cid + 1 < len(self.corpus) and self.corpus[cid + 1].get("doc_path") == doc else "")
        return prev, nxt

    # -- public API -----------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3, use_rerank: bool = True, expand: Optional[bool] = None
                 ) -> tuple[List[dict], dict]:
        """Return (ranked chunks, timings_ms). Each chunk keeps its original `text`; the stitched
        neighbour window is stored separately in `expanded_text`."""
        cfg = self.cfg
        timings = {"bm25_ms": 0.0, "dense_ms": 0.0, "fusion_ms": 0.0, "retrieval_ms": 0.0,
                   "rerank_ms": 0.0, "n_candidates": 0}

        t0 = time.perf_counter()
        bm25_hits = self.bm25.retrieve(query, top_k=cfg.candidates_per_retriever)
        t1 = time.perf_counter()
        dense_hits = self.dense.retrieve(query, top_k=cfg.candidates_per_retriever)
        t2 = time.perf_counter()
        fused = reciprocal_rank_fusion([bm25_hits, dense_hits], k=cfg.rrf_k,
                                       top_n=cfg.rerank_depth if use_rerank else None)
        t3 = time.perf_counter()
        timings.update(bm25_ms=(t1 - t0) * 1e3, dense_ms=(t2 - t1) * 1e3, fusion_ms=(t3 - t2) * 1e3,
                       retrieval_ms=(t3 - t0) * 1e3, n_candidates=len(fused))
        if not fused:
            return [], timings

        if use_rerank and self.cross_encoder is not None:
            t4 = time.perf_counter()
            scores = self.cross_encoder.predict([[query, c["text"]] for c in fused], show_progress_bar=False)
            for c, s in zip(fused, scores):
                c["cross_encoder_score"] = float(s)
            fused.sort(key=lambda c: c["cross_encoder_score"], reverse=True)
            timings["rerank_ms"] = (time.perf_counter() - t4) * 1e3

        top = fused[:top_k]
        do_expand = cfg.expand_neighbours if expand is None else expand
        for c in top:
            c["score"] = c.get("cross_encoder_score", c.get("rrf_score", 0.0))
            if do_expand:
                c["prev_text"], c["next_text"] = self._neighbours(c)
            else:
                c["prev_text"] = c["next_text"] = ""
            c["expanded_text"] = " ".join(x for x in (c["prev_text"], c["text"], c["next_text"]) if x)
        return top, timings
