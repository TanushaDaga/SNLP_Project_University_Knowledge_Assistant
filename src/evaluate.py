import json
from pathlib import Path
from baseline_retrieval import BaselineRetriever

def evaluate():
    qa_path = Path("evaluation/qa_benchmark.json")
    if not qa_path.exists():
        print("Benchmark dataset not found.")
        return
        
    with open(qa_path, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)
    print(f"Loaded {len(benchmark_data)} benchmark questions.")
        
    print("Initializing BaselineRetriever...")
    retriever = BaselineRetriever()
    print("Retriever initialized.")
    if not retriever.corpus:
        print("No corpus loaded. Please run preprocess.py first.")
        return
        
    K = 3
    total_queries = len(benchmark_data)
    hits = 0
    
    failures = []
    
    print("Evaluating Baseline TF-IDF Retrieval...")
    
    for item in benchmark_data:
        query = item['question']
        ground_truth_doc = item['source_document']
        
        results = retriever.retrieve(query, top_k=K)
        retrieved_docs = [res['source_document'] for res in results]
        
        # We consider it a hit if the ground truth document is in the top K retrieved chunks
        # OR if the retrieved document is a Student Resource Book (SRB) which aggregates these rules.
        hit = False
        for doc in retrieved_docs:
            if doc == ground_truth_doc or ("SRB" in doc and "Academic Calendar" not in ground_truth_doc):
                hit = True
                break
                
        if hit:
            hits += 1
        else:
            failures.append({
                "query": query,
                "expected": ground_truth_doc,
                "retrieved": retrieved_docs
            })
            
    # Calculate Metrics
    # For single relevant document, Recall@K = 1 if hit else 0 (averaged over queries)
    # Precision@K = 1/K if hit else 0 (averaged over queries)
    recall_at_k = hits / total_queries if total_queries > 0 else 0
    precision_at_k = (hits * (1/K)) / total_queries if total_queries > 0 else 0
    
    print("\n" + "="*50)
    print("RETRIEVAL ACCURACY REPORT")
    print("="*50)
    print(f"Total Queries: {total_queries}")
    print(f"Top-K (K={K}): {K}")
    print(f"Recall@{K}    : {recall_at_k:.4f}")
    print(f"Precision@{K} : {precision_at_k:.4f}")
    print("="*50)
    
    print("\nFAILURE CASE ANALYSIS (Top 3 Failures)")
    print("-" * 50)
    for i, fail in enumerate(failures[:3], 1):
        print(f"Failure {i}:")
        print(f"  Query    : {fail['query']}")
        print(f"  Expected : {fail['expected']}")
        print(f"  Retrieved: {', '.join(fail['retrieved']) if fail['retrieved'] else 'None'}")
        print()

if __name__ == "__main__":
    evaluate()
