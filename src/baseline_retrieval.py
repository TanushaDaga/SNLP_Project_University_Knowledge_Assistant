"""
Week 1-2 BM25 baseline (thin wrapper around knowledge_assistant.retrieval.BM25Retriever).

    python src/baseline_retrieval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from knowledge_assistant.config import EngineConfig
from knowledge_assistant.retrieval import BM25Retriever, load_corpus


class BaselineRetriever(BM25Retriever):
    """BM25Okapi over lemmatised sentence-window chunks. API unchanged since Week 2."""

    def __init__(self, corpus_path=None):
        corpus_path = Path(corpus_path) if corpus_path else EngineConfig().corpus_path
        self.corpus_path = corpus_path
        super().__init__(load_corpus(corpus_path))


if __name__ == "__main__":
    retriever = BaselineRetriever()
    q = "What is the minimum attendance required for exams?"
    print(f"\nQuery: {q}\n")
    for i, res in enumerate(retriever.retrieve(q, top_k=3), 1):
        print(f"Rank {i} (BM25 {res['score']:.3f})  {res['doc_path']}")
        print(f"    {res['text'][:200]}...\n")
