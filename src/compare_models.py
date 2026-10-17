import json
import time
import logging
from pathlib import Path
from tabulate import tabulate

import sys
sys.path.append(str(Path(__file__).parent))
from baseline_retrieval import BaselineRetriever
from rag_pipeline import UniversityKnowledgeAssistant
from improved_rag import ImprovedUniversityAssistant

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_mrr(retrieved_docs, ground_truth):
    for i, doc in enumerate(retrieved_docs):
        if doc == ground_truth:
            return 1.0 / (i + 1)
    return 0.0

def precision_at_k(retrieved_docs, ground_truth, k=3):
    """Share of the k slots filled by chunks of the ground-truth document (was: hit@3 / 3, which is not precision)."""
    return sum(1 for d in retrieved_docs[:k] if d == ground_truth) / k


def calculate_metrics(results_list):
    total = len(results_list)
    if total == 0:
        return {}
    
    hits_at_1 = sum(1 for r in results_list if r['hit_at_1'])
    hits_at_3 = sum(1 for r in results_list if r['hit_at_3'])
    mrr_3 = sum(r['mrr_3'] for r in results_list)
    
    avg_retrieval_latency = sum(r['retrieval_latency_ms'] for r in results_list) / total
    
    metrics = {
        "HitRate@1": hits_at_1 / total,
        "HitRate@3": hits_at_3 / total,
        "MRR@3": mrr_3 / total,
        "Precision@3": sum(r['precision_3'] for r in results_list) / total,
        "Recall@3": hits_at_3 / total,
        "Avg_Retrieval_Latency_ms": avg_retrieval_latency,
        "Median_Retrieval_Latency_ms": sorted(r['retrieval_latency_ms'] for r in results_list)[total // 2]
    }
    
    # Optional E2E metrics for RAG
    e2e_results = [r for r in results_list if r.get('e2e_latency_ms') is not None]
    if e2e_results:
        avg_e2e_latency = sum(r['e2e_latency_ms'] for r in e2e_results) / len(e2e_results)
        citation_acc = sum(1 for r in e2e_results if r.get('citation_match')) / len(e2e_results)
        fallback_rate = sum(1 for r in e2e_results if r.get('is_fallback')) / len(e2e_results)
        
        metrics.update({
            "Avg_E2E_Latency_ms": avg_e2e_latency,
            "Citation_Accuracy": citation_acc,
            "Fallback_Rate": fallback_rate
        })
        
    return metrics

def evaluate(with_llm=False):
    benchmark_path = Path("evaluation/qa_benchmark.json")
    results_path = Path("evaluation/model_comparison_results.json")
    corpus_path = "Data/processed/corpus.json"
    index_dir = "Data/processed"
    
    if not benchmark_path.exists():
        logging.error(f"Benchmark file not found: {benchmark_path}")
        return
        
    with open(benchmark_path, 'r', encoding='utf-8') as f:
        queries = json.load(f)
        
    logging.info(f"Loaded {len(queries)} test queries.")

    # Only keep queries whose ground-truth document exists in the current (rebuilt) corpus.
    with open(corpus_path, 'r', encoding='utf-8') as f:
        indexed_docs = {c['source_document'] for c in json.load(f)}
    dropped = [q['source_document'] for q in queries if q['source_document'] not in indexed_docs]
    if dropped:
        logging.warning(f"{len(dropped)} benchmark queries dropped - ground-truth document no longer in corpus: {sorted(set(dropped))}")
    queries = [q for q in queries if q['source_document'] in indexed_docs]
    logging.info(f"Evaluating {len(queries)} queries.")
    
    logging.info("Initializing Models...")
    bm25 = BaselineRetriever(corpus_path=corpus_path)
    dense_rag = UniversityKnowledgeAssistant(index_dir=index_dir)
    # Ensure dense index is used
    if not hasattr(dense_rag, 'indexer'):
        from dense_indexer import DenseIndexer
        dense_rag.indexer = DenseIndexer(index_dir=index_dir, corpus_path=corpus_path)
        dense_rag.indexer.load_or_build_index()
        
    improved_rag = ImprovedUniversityAssistant(index_dir=index_dir, corpus_path=corpus_path)
    
    bm25_eval = []
    dense_eval = []
    improved_eval = []
    
    print("\nRunning Evaluation...\n")
    
    for i, item in enumerate(queries):
        query = item['question']
        ground_truth = item['source_document']
        
        # 1. BM25 Baseline
        t0 = time.time()
        bm25_res = bm25.retrieve(query, top_k=3)
        t_bm25 = (time.time() - t0) * 1000
        bm25_docs = [r['source_document'] for r in bm25_res]
        
        bm25_eval.append({
            "query": query,
            "hit_at_1": bool(bm25_docs and bm25_docs[0] == ground_truth),
            "precision_3": precision_at_k(bm25_docs, ground_truth),
            "hit_at_3": ground_truth in bm25_docs,
            "mrr_3": calculate_mrr(bm25_docs, ground_truth),
            "retrieval_latency_ms": t_bm25
        })
        
        # 2. Dense V1
        t0 = time.time()
        dense_res = dense_rag.indexer.retrieve(query, top_k=3)
        t_dense_retrieval = (time.time() - t0) * 1000
        dense_docs = [r['source_document'] for r in dense_res]
        
        if with_llm:
            t0 = time.time()
            dense_gen = dense_rag.generate_answer(query, top_k=3)
            t_dense_e2e = (time.time() - t0) * 1000
        else:
            dense_gen, t_dense_e2e = {}, None
        
        # Determine citation and fallback
        # In rag_pipeline, sources might be a list of dicts or None
        dense_citation_match = False
        if dense_gen.get('sources'):
            dense_citation_match = any(s.get('document') == ground_truth for s in dense_gen['sources'])
        # Also check fallback text
        is_fallback_dense = "does not contain sufficient details" in str(dense_gen.get('answer', ''))
        
        dense_eval.append({
            "query": query,
            "hit_at_1": bool(dense_docs and dense_docs[0] == ground_truth),
            "precision_3": precision_at_k(dense_docs, ground_truth),
            "hit_at_3": ground_truth in dense_docs,
            "mrr_3": calculate_mrr(dense_docs, ground_truth),
            "retrieval_latency_ms": t_dense_retrieval,
            "e2e_latency_ms": t_dense_e2e,
            "citation_match": dense_citation_match,
            "is_fallback": is_fallback_dense
        })
        
        # 3. Improved Hybrid RAG
        t0 = time.time()
        improved_res = improved_rag.retrieve(query, top_k=3)
        t_imp_retrieval = (time.time() - t0) * 1000
        imp_docs = [r['source_document'] for r in improved_res]
        
        if with_llm:
            t0 = time.time()
            improved_gen = improved_rag.generate_answer(query, top_k=3)
            t_imp_e2e = (time.time() - t0) * 1000
        else:
            improved_gen, t_imp_e2e = {}, None
        
        imp_citation_match = False
        if improved_gen.get('sources'):
            imp_citation_match = any(s.get('document') == ground_truth for s in improved_gen['sources'])
        is_fallback_imp = "does not contain sufficient details" in str(improved_gen.get('answer', ''))
        
        improved_eval.append({
            "query": query,
            "hit_at_1": bool(imp_docs and imp_docs[0] == ground_truth),
            "precision_3": precision_at_k(imp_docs, ground_truth),
            "hit_at_3": ground_truth in imp_docs,
            "mrr_3": calculate_mrr(imp_docs, ground_truth),
            "retrieval_latency_ms": t_imp_retrieval,
            "e2e_latency_ms": t_imp_e2e,
            "citation_match": imp_citation_match,
            "is_fallback": is_fallback_imp
        })
        
        if (i+1) % 5 == 0:
            logging.info(f"Processed {i+1}/{len(queries)} queries...")
            
    # Calculate aggregates
    bm25_metrics = calculate_metrics(bm25_eval)
    dense_metrics = calculate_metrics(dense_eval)
    improved_metrics = calculate_metrics(improved_eval)
    
    # The last 10 benchmark queries are 100-character snippets copied from their own target chunk (lexical leakage).
    handwritten = slice(0, 15)
    subset_output = {
        "BM25_Baseline": calculate_metrics(bm25_eval[handwritten]),
        "Dense_V1": calculate_metrics(dense_eval[handwritten]),
        "Improved_Hybrid": calculate_metrics(improved_eval[handwritten]),
    }
    # Save to JSON
    final_output = {
        "n_queries": len(queries),
        "with_llm": with_llm,
        "BM25_Baseline": bm25_metrics,
        "Dense_V1": dense_metrics,
        "Improved_Hybrid": improved_metrics,
        "first_15_handwritten_queries_only": subset_output,
        "per_query": {"BM25": bm25_eval, "Dense": dense_eval, "Hybrid": improved_eval},
    }
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2)
        
    # Print Table
    headers = ["Metric", "BM25 Baseline", "Dense V1", "Improved Hybrid RAG"]

    def na(metrics, key, fmt="{:.2%}"):
        return fmt.format(metrics[key]) if key in metrics else "not run (use --with-llm)"
    
    table_data = [
        ["HitRate@1", f"{bm25_metrics.get('HitRate@1',0):.2%}", f"{dense_metrics.get('HitRate@1',0):.2%}", f"{improved_metrics.get('HitRate@1',0):.2%}"],
        ["HitRate@3", f"{bm25_metrics.get('HitRate@3',0):.2%}", f"{dense_metrics.get('HitRate@3',0):.2%}", f"{improved_metrics.get('HitRate@3',0):.2%}"],
        ["MRR@3", f"{bm25_metrics.get('MRR@3',0):.3f}", f"{dense_metrics.get('MRR@3',0):.3f}", f"{improved_metrics.get('MRR@3',0):.3f}"],
        ["Precision@3", f"{bm25_metrics.get('Precision@3',0):.3f}", f"{dense_metrics.get('Precision@3',0):.3f}", f"{improved_metrics.get('Precision@3',0):.3f}"],
        ["Recall@3", f"{bm25_metrics.get('Recall@3',0):.2%}", f"{dense_metrics.get('Recall@3',0):.2%}", f"{improved_metrics.get('Recall@3',0):.2%}"],
        ["Citation Accuracy", "N/A", na(dense_metrics, 'Citation_Accuracy'), na(improved_metrics, 'Citation_Accuracy')],
        ["Fallback Rate", "N/A", na(dense_metrics, 'Fallback_Rate'), na(improved_metrics, 'Fallback_Rate')],
        ["Avg Retrieval Latency (ms)", f"{bm25_metrics.get('Avg_Retrieval_Latency_ms',0):.1f}", f"{dense_metrics.get('Avg_Retrieval_Latency_ms',0):.1f}", f"{improved_metrics.get('Avg_Retrieval_Latency_ms',0):.1f}"],
        ["Median Retrieval Latency (ms)", f"{bm25_metrics.get('Median_Retrieval_Latency_ms',0):.1f}", f"{dense_metrics.get('Median_Retrieval_Latency_ms',0):.1f}", f"{improved_metrics.get('Median_Retrieval_Latency_ms',0):.1f}"],
        ["Avg E2E Latency (ms)", "N/A", na(dense_metrics, 'Avg_E2E_Latency_ms', '{:.1f}'), na(improved_metrics, 'Avg_E2E_Latency_ms', '{:.1f}')]
    ]
    
    print("\n" + "="*80)
    print(" WEEK 4: MODEL COMPARISON RESULTS ".center(80))
    print("="*80)
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    print("="*80)
    print(f"Detailed results saved to {results_path}\n")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Week 4 model comparison (retrieval metrics are free; --with-llm spends API quota)")
    ap.add_argument("--with-llm", action="store_true")
    evaluate(with_llm=ap.parse_args().with_llm)
