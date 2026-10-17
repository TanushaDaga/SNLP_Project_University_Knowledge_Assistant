import json
import os
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Ensure the dense indexer is importable
import sys
sys.path.append(str(Path(__file__).parent))
from dense_indexer import DenseIndexer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class UniversityKnowledgeAssistant:
    def __init__(self, index_dir="Data/processed", corpus_path="Data/processed/corpus.json"):
        self.indexer = DenseIndexer(corpus_path=corpus_path, index_dir=index_dir)
        self.indexer.load_or_build_index()

        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY") or os.getenv("API_KEY")
        
        self.llm_initialized = False
        if self.api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                self.llm_initialized = True
                logging.info("Groq LLM initialized successfully.")
            except ImportError:
                logging.warning("groq not installed. Falling back to mock LLM.")
            except Exception as e:
                logging.warning(f"Error initializing Groq: {e}. Falling back to mock LLM.")
        else:
            logging.warning("No API_KEY found in .env. Falling back to mock LLM.")

    def _generate_mock_response(self, query, top_chunk):
        """Extractive fallback when no LLM is reachable (clearly labelled, uses the real cosine score)."""
        return {
            "query": query,
            "answer": f"[extractive fallback - LLM unavailable] {top_chunk['text']}",
            "sources": [{"document": top_chunk['source_document'], "snippet": top_chunk['text']}],
            "confidence_score": max(0.0, min(1.0, float(top_chunk['score']))),  # FAISS cosine in [0, 1]
            "retrieved_chunks": [top_chunk]
        }

    def generate_answer(self, query, top_k=5):
        # 1. Retrieval Step
        retrieved_chunks = self.indexer.retrieve(query, top_k=top_k)
        
        if not retrieved_chunks:
            return {
                "query": query,
                "answer": "The official documentation does not contain sufficient details to answer this query.",
                "supporting_document": None,
                "source_section": None,
                "confidence_score": 0.0
            }

        top_chunk = retrieved_chunks[0]

        # If LLM is not initialized, return mock
        if not self.llm_initialized:
            return self._generate_mock_response(query, retrieved_chunks[0])
            
        # 2. Prompt Engineering Step
        context_text = "\n\n".join([f"Document: {chunk['source_document']}\nText: {chunk['text']}" for chunk in retrieved_chunks])
        
        system_prompt = f"""You are an authoritative university knowledge assistant.
Your task is to answer the user's query STRICTLY based on the provided context below.
If the context does not contain the answer, you MUST output exactly: "The official documentation does not contain sufficient details to answer this query."

Context:
{context_text}

Provide your response strictly as a JSON object with the following keys:
- "query": The user's original query.
- "answer": A comprehensive and complete factual answer synthesized from multiple sources in the context. Ensure you do not truncate the answer. Format your answer with distinct paragraphs, and use bullet points or numbered lists where appropriate for readability. Do NOT use markdown code blocks for the JSON itself.
- "sources": A list of dictionaries, each containing "document" (the document name) and "snippet" (a verbatim text snippet from the retrieved chunk used to answer).
- "confidence_score": A float between 0.0 and 1.0 indicating your confidence in the answer based on the context.

Example JSON output format:
{{
  "query": "...",
  "answer": "...",
  "sources": [
    {{"document": "...", "snippet": "..."}}
  ],
  "confidence_score": 0.95
}}

Do not include markdown code block formatting (like ```json), just output the raw JSON object.
"""
        
        # 3. Generation Step
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                model=os.getenv("GROQ_MODEL_V1", "qwen/qwen3.8-27b"),  # Week 3 model (override via env if decommissioned)
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            response_text = chat_completion.choices[0].message.content.strip()
            # Try to parse the JSON response
            result = json.loads(response_text)
            
            # Ensure all required keys exist
            required_keys = ["query", "answer", "sources", "confidence_score"]
            for key in required_keys:
                if key not in result:
                    result[key] = None
            
            # Pass all retrieved chunks to the UI
            result["retrieved_chunks"] = retrieved_chunks
                    
            return result
            
        except Exception as e:
            logging.error(f"Error during LLM generation: {e}")
            # Fallback to mock response on API failure
            mock = self._generate_mock_response(query, top_chunk)
            mock["retrieved_chunks"] = retrieved_chunks
            return mock

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="University Knowledge Assistant RAG Pipeline")
    parser.add_argument("--query", type=str, help="Natural language query to ask the assistant")
    args = parser.parse_args()

    assistant = UniversityKnowledgeAssistant()

    if args.query:
        print(f"\nProcessing Query: '{args.query}'...\n")
        result = assistant.generate_answer(args.query)
        print(json.dumps(result, indent=2))
    else:
        print("Interactive Mode. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                user_query = input("\nEnter your query: ")
                if user_query.lower() in ['exit', 'quit']:
                    break
                if not user_query.strip():
                    continue
                    
                result = assistant.generate_answer(user_query)
                print("\nResponse:")
                print(json.dumps(result, indent=2))
            except KeyboardInterrupt:
                break
