"""
Week 3 dense retriever (thin wrapper around knowledge_assistant.retrieval.DenseRetriever).

The index is validated against a digest of the corpus and rebuilt automatically when stale
(the old version silently kept using an index built from a different corpus).

    python src/dense_indexer.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from knowledge_assistant.config import EngineConfig
from knowledge_assistant.retrieval import DenseRetriever, load_corpus


class DenseIndexer(DenseRetriever):
    def __init__(self, corpus_path=None, index_dir=None):
        cfg = EngineConfig()
        self.corpus_path = Path(corpus_path) if corpus_path else cfg.corpus_path
        super().__init__(load_corpus(self.corpus_path), Path(index_dir) if index_dir else cfg.index_dir,
                         cfg.embedding_model)

    def load_or_build_index(self):
        return self.load_or_build()

    def build_index(self):
        return self.build()


if __name__ == "__main__":
    idx = DenseIndexer()
    idx.load_or_build_index()
    q = "What is the minimum attendance required for exams?"
    print(f"\nQuery: {q}\n")
    for i, res in enumerate(idx.retrieve(q, top_k=3), 1):
        print(f"Rank {i} (cosine {res['score']:.3f})  {res['doc_path']}")
        print(f"    {res['text'][:200]}...\n")
