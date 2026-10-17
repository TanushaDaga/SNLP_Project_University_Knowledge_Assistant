import json
import time
import logging
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent))
from baseline_retrieval import BaselineRetriever
from rag_pipeline import UniversityKnowledgeAssistant

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate():
    benchmark_path = Path("evaluation/qa_benchmark.json")
    results_path = Path("evaluation/week3_results.json")
    
    if not benchmark_path.exists():
        logging.error(f"Benchmark file not found: {benchmark_path}")
        return
        
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)
        
    logging.info(f"Loaded {len(queries)} test queries.")
    
    logging.info("Initializing BM25 Baseline...")
    bm25_retriever = BaselineRetriever()
    
    logging.info("Initializing Dense RAG Pipeline...")
    rag_assistant = UniversityKnowledgeAssistant()
    
    results = []
    
    bm25_correct_retrieval = 0
    rag_correct_retrieval = 0
    rag_citation_accuracy = 0
    
    bm25_total_time = 0
    rag_total_time = 0
    
    print("\nRunning Evaluation...\n")
    
    for i, item in enumerate(queries):
        query = item['question']
        ground_truth_doc = item['source_document']
        
        # 1. Run BM25
        t0 = time.time()
        bm25_results = bm25_retriever.retrieve(query, top_k=3)
        t1 = time.time()
        
        bm25_time = t1 - t0
        bm25_total_time += bm25_time
        
        bm25_retrieved_docs = [res['source_document'] for res in bm25_results]
        bm25_hit = ground_truth_doc in bm25_retrieved_docs
        if bm25_hit:
            bm25_correct_retrieval += 1
            
        # 2. Run Dense RAG
        t0 = time.time()
        rag_result = rag_assistant.generate_answer(query, top_k=3)
        t1 = time.time()
        
        rag_time = t1 - t0
        rag_total_time += rag_time
        
        # We need to get the retrieval results for Dense Indexer to compute retrieval accuracy
        dense_results = rag_assistant.indexer.retrieve(query, top_k=3)
        dense_retrieved_docs = [res['source_document'] for res in dense_results]
        rag_hit = ground_truth_doc in dense_retrieved_docs
        if rag_hit:
            rag_correct_retrieval += 1
            
        # Check Citation Accuracy
        citation_hit = rag_result.get('supporting_document') == ground_truth_doc
        if citation_hit:
            rag_citation_accuracy += 1
            
        results.append({
            "query": query,
            "ground_truth": ground_truth_doc,
            "bm25_hit": bm25_hit,
            "bm25_latency": bm25_time,
            "rag_hit": rag_hit,
            "rag_citation_hit": citation_hit,
            "rag_latency": rag_time,
            "rag_output": rag_result
        })
        
        if (i+1) % 5 == 0:
            logging.info(f"Processed {i+1}/{len(queries)} queries...")
            
    num_queries = len(queries)
    
    metrics = {
        "bm25_retrieval_accuracy": bm25_correct_retrieval / num_queries,
        "rag_retrieval_accuracy": rag_correct_retrieval / num_queries,
        "rag_citation_accuracy": rag_citation_accuracy / num_queries,
        "bm25_avg_latency": bm25_total_time / num_queries,
        "rag_avg_latency": rag_total_time / num_queries,
    }
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({"metrics": metrics, "detailed_results": results}, f, indent=2)
        
    print("\n=======================================================")
    print("               WEEK 3 EVALUATION RESULTS               ")
    print("=======================================================")
    print(f"| Metric                   | BM25 Baseline | Dense RAG |")
    print(f"|--------------------------|---------------|-----------|")
    print(f"| Retrieval Acc (Top-3)    | {metrics['bm25_retrieval_accuracy']:.1%}        | {metrics['rag_retrieval_accuracy']:.1%}     |")
    print(f"| Citation Accuracy        | N/A           | {metrics['rag_citation_accuracy']:.1%}     |")
    print(f"| Avg Latency (seconds)    | {metrics['bm25_avg_latency']:.4f}s       | {metrics['rag_avg_latency']:.4f}s   |")
    print("=======================================================\n")
    print(f"Detailed results saved to {results_path}")

if __name__ == "__main__":
    evaluate()
