"""
Week 4 improved RAG (BM25 + FAISS -> RRF -> cross-encoder -> grounded LLM answer).

This module is now a thin, backward-compatible wrapper around the reusable package
`knowledge_assistant` so the Gradio UI (src/app.py), compare_models.py and the REST API all
run the *same* code path.

    python src/improved_rag.py --query "What is the minimum attendance required for exams?"
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from knowledge_assistant import EngineConfig, UniversityKnowledgeEngine


class ImprovedUniversityAssistant:
    def __init__(self, index_dir=None, corpus_path=None, engine=None):
        if engine is None:
            cfg = EngineConfig()
            if index_dir:
                cfg.index_dir = Path(index_dir)
            if corpus_path:
                cfg.corpus_path = Path(corpus_path)
            engine = UniversityKnowledgeEngine(cfg)
        self.engine = engine
        self.dense_indexer = engine.retriever.dense      # kept for older scripts
        self.bm25_retriever = engine.retriever.bm25
        self.cross_encoder = engine.retriever.cross_encoder
        self.llm_initialized = engine.llm_available

    def retrieve(self, query, top_k=3):
        chunks, _ = self.engine.retriever.retrieve(query, top_k=top_k)
        return chunks

    def generate_answer(self, query, top_k=3):
        """Legacy dict output (query/answer/sources/confidence_score/...) plus the retrieved chunks."""
        resp = self.engine.ask(query, enforce_grounding=True, top_k=min(top_k, 5))
        chunks = self.retrieve(query, top_k=top_k) if top_k > 5 else []
        return {
            "query": query,
            "answer": resp.answer,
            "supporting_document": resp.supporting_document,
            "source_section": resp.source_section,
            "sources": ([{"document": resp.supporting_document, "snippet": resp.source_section}]
                        if resp.supporting_document else []),
            "confidence_score": resp.confidence_score,
            "retrieval_method": resp.retrieval_metadata.method,
            "retrieval_time_ms": resp.retrieval_metadata.retrieval_latency_ms
                                 + resp.retrieval_metadata.rerank_latency_ms,
            "is_fallback": resp.is_fallback,
            "generation_mode": resp.retrieval_metadata.generation_mode,
            "retrieved_chunks": chunks or [
                {"source_document": r.document, "text": r.text_snippet, "cross_encoder_score": r.similarity_score}
                for r in resp.search_results],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved RAG pipeline (hybrid + re-rank)")
    parser.add_argument("--query", type=str, help="Natural language query")
    args = parser.parse_args()
    assistant = ImprovedUniversityAssistant()
    if args.query:
        out = assistant.generate_answer(args.query)
        out.pop("retrieved_chunks", None)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print("Run with --query '<your question>' to test.")
