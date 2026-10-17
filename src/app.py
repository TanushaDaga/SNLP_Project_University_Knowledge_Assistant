import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import gradio as gr
import logging
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from improved_rag import ImprovedUniversityAssistant

logging.basicConfig(level=logging.INFO)

print("Starting UI server. Loading Improved Hybrid RAG pipeline...")
assistant = ImprovedUniversityAssistant(index_dir="Data/processed", corpus_path="Data/processed/corpus.json")
print("RAG Pipeline loaded and ready.")

# Custom CSS for a premium, glassmorphism look
custom_css = """
body {
    background: linear-gradient(135deg, #1e1e2f, #2a2a4a);
    color: #ffffff;
    font-family: 'Inter', sans-serif;
}
.gradio-container {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(15px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    padding: 30px;
    margin-top: 40px;
}
.gr-button {
    background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%) !important;
    border: none !important;
    transition: transform 0.2s, box-shadow 0.2s;
}
.gr-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(75, 108, 183, 0.4) !important;
}
.bot, .user {
    border-radius: 15px !important;
}
"""

def chat_interface(message, history):
    start_time = time.time()
    
    # Generate the RAG answer (restoring top_k=25, token limits handled in pipeline)
    result = assistant.generate_answer(message, top_k=25)
    
    execution_time = time.time() - start_time
    
    answer = result.get('answer') or "I couldn't find an answer."
    sources = result.get('sources') or []
    score = result.get('confidence_score')
    if score is None: score = 0.0
    retrieved_chunks = result.get('retrieved_chunks') or []
    
    # Format a beautiful markdown response
    response = f"{answer}\n\n"
    
    if sources:
        response += f"---\n### AI Synthesized Sources (Top {len(sources)})\n"
        for i, src in enumerate(sources, 1):
            response += f"**{i}. `{src.get('document', 'Unknown')}`**\n"
            response += f"> *\"{src.get('snippet', '')}\"*\n\n"
            
    if retrieved_chunks:
        response += f"---\n### All Retrieved Document Contexts (Top {len(retrieved_chunks)})\n"
        for i, chunk in enumerate(retrieved_chunks, 1):
            chunk_score = chunk.get('cross_encoder_score')
            if chunk_score is None: chunk_score = chunk.get('score', 0.0)
            response += f"{i}. `{chunk.get('source_document', 'Unknown')}` (Relevance Score: {chunk_score:.2f})\n"
            
    response += f"\n**Overall Confidence:** {float(score):.2f} | **Response time:** {execution_time:.2f}s"
    
    return response

# Build the Gradio UI
with gr.Blocks() as app:
    with gr.Row():
        gr.Markdown(
            """
            # University Knowledge Assistant
            ### AI-Powered RAG Search System
            Ask any question regarding university policies, attendance rules, capstone projects, or placements!
            """
        )
    
    gr.ChatInterface(
        fn=chat_interface,
        chatbot=gr.Chatbot(height=500, avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/4712/4712010.png")),
        textbox=gr.Textbox(placeholder="E.g., What is the minimum attendance required for exams?", container=False, scale=7),
        title=None,
        description=None,
        examples=[
            "What is the minimum attendance required to be eligible for exams?",
            "How many leaves are permitted for medical reasons?",
            "What are the rules and formatting guidelines for the capstone project report?",
            "What is the penalty for using unfair means during an exam?"
        ],
    )

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", show_error=True, css=custom_css, theme=gr.themes.Monochrome())
