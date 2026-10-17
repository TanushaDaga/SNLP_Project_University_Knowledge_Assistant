import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import json
from pathlib import Path
from baseline_retrieval import BaselineRetriever
from dense_indexer import DenseIndexer

def rrf_fusion(bm25_results, dense_results, top_n=10, k=60):
    rrf_scores = {}
    chunks_map = {}
    
    for rank, res in enumerate(bm25_results, 1):
        chunk_id = res['chunk_id']
        chunks_map[chunk_id] = res
        if chunk_id not in rrf_scores: rrf_scores[chunk_id] = 0.0
        rrf_scores[chunk_id] += 1.0 / (k + rank)
        
    for rank, res in enumerate(dense_results, 1):
        chunk_id = res['chunk_id']
        if chunk_id not in chunks_map: chunks_map[chunk_id] = res
        if chunk_id not in rrf_scores: rrf_scores[chunk_id] = 0.0
        rrf_scores[chunk_id] += 1.0 / (k + rank)
        
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunks_map[cid] for cid, score in sorted_chunks[:top_n]]

def test_rrf():
    with open('evaluation/qa_benchmark.json', 'r') as f:
        queries = json.load(f)
        
    bm25 = BaselineRetriever(corpus_path="Data/processed/corpus.json")
    dense = DenseIndexer(corpus_path="Data/processed/corpus.json", index_dir="Data/processed")
    dense.load_or_build_index()
    
    hits_1 = 0
    hits_3 = 0
    
    for item in queries:
        q = item['question']
        gt = item['source_document']
        
        b_res = bm25.retrieve(q, top_k=40)
        d_res = dense.retrieve(q, top_k=40)
        
        fused = rrf_fusion(b_res, d_res, top_n=3)
        docs = [x['source_document'] for x in fused]
        
        if docs and docs[0] == gt:
            hits_1 += 1
        if gt in docs:
            hits_3 += 1
            
    print(f"RRF HitRate@1: {hits_1/len(queries):.2%}")
    print(f"RRF HitRate@3: {hits_3/len(queries):.2%}")

if __name__ == '__main__':
    test_rrf()
