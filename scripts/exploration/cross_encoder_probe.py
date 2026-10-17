import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from improved_rag import ImprovedUniversityAssistant

def test():
    assistant = ImprovedUniversityAssistant()
    print("Testing if Cross-Encoder can find UFM when fed 100 chunks...")
    query = "What is the penalty for using unfair means during an exam?"
    
    # Force feed 100 chunks to ensure UFM is included
    bm25 = assistant.bm25_retriever.retrieve(query, top_k=50)
    dense = assistant.dense_indexer.retrieve(query, top_k=50)
    
    candidates_map = {}
    for res in bm25 + dense:
        candidates_map[res['chunk_id']] = res
        
    fused = list(candidates_map.values())
    texts = [c['text'] for c in fused]
    pairs = [[query, txt] for txt in texts]
    
    scores = assistant.cross_encoder.predict(pairs)
    for c, s in zip(fused, scores):
        c['cross_encoder_score'] = float(s)
        
    fused = sorted(fused, key=lambda x: x['cross_encoder_score'], reverse=True)
    
    print("\n--- TOP 10 RERANKED CHUNKS ---")
    for i, c in enumerate(fused[:10]):
        is_ufm = "UFM" in c['source_document']
        marker = "*** UFM ***" if is_ufm else ""
        print(f"{i+1}. {c['source_document']} | Score: {c.get('cross_encoder_score', 0):.2f} {marker}")

if __name__ == "__main__":
    test()
