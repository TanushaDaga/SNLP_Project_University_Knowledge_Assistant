"""
rest_demo_ui.py - Gradio demo that talks to the Knowledge Assistant over HTTP (REST).

Unlike src/app.py (which runs the engine in the same process), this page only sends requests to the
FastAPI service, the same way any other program would. Start the API first:

    uvicorn src.api.app:app --host 0.0.0.0 --port 8000
    python src/rest_demo_ui.py                      # opens on http://127.0.0.1:7861

Set API_URL to point at another host, e.g.  API_URL=http://192.168.1.20:8000
"""

import json
import os
import time

import gradio as gr
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 90

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
}
.bot, .user {
    border-radius: 15px !important;
}
"""


def call_api(path, payload):
    """POST to the API. Returns (json_or_None, error_message_or_None, http_status)."""
    try:
        r = requests.post(f"{API_URL}{path}", json=payload, timeout=TIMEOUT)
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach the API at {API_URL}. Start it with: uvicorn src.api.app:app --port 8000", None
    except requests.exceptions.Timeout:
        return None, "The API took too long to answer.", None
    try:
        body = r.json()
    except ValueError:
        return None, f"The API returned HTTP {r.status_code} with a non-JSON body.", r.status_code
    if r.status_code != 200:
        detail = body.get("detail", "") if isinstance(body, dict) else ""
        if r.status_code == 429:
            detail = f"Too many requests. Retry in {r.headers.get('Retry-After', '?')} s."
        return None, f"API error {r.status_code}: {detail}", r.status_code
    return body, None, r.status_code


def api_status():
    try:
        h = requests.get(f"{API_URL}/health", timeout=5).json()
    except Exception:
        return f"**API status:** not reachable at `{API_URL}`"
    return (f"**API status:** {h.get('status')} · index size {h.get('vector_index_size')} · "
            f"LLM available: {h.get('llm_available')} · `{API_URL}`")


def format_answer(resp, elapsed):
    meta = resp.get("retrieval_metadata") or {}
    conf = float(resp.get("confidence_score") or 0.0)
    out = f"{resp.get('answer', '')}\n\n"
    if resp.get("is_fallback"):
        out += f"---\n*No grounded answer.* Reason: `{resp.get('refusal_reason')}`\n"
    else:
        out += f"---\n**Source document:** `{resp.get('supporting_document')}`\n\n"
        if resp.get("source_section"):
            out += f"> {resp['source_section']}\n\n"
        if resp.get("ungrounded_numbers"):
            out += f"Numbers not found in the evidence: {resp['ungrounded_numbers']}\n\n"
    out += (f"**Confidence:** {conf:.2f} | **Evidence verified:** {resp.get('evidence_verified')} | "
            f"**Mode:** {meta.get('generation_mode')} | **API time:** {meta.get('total_latency_ms', 0) / 1000:.2f}s "
            f"(round trip {elapsed:.2f}s)")
    return out


def chat_interface(message, history):
    start = time.time()
    resp, err, _ = call_api("/api/v1/ask", {"query": message, "top_k": 3})
    if err:
        return err
    return format_answer(resp, time.time() - start)


def inspect_call(question, endpoint):
    """Show the raw request and response, and what a downstream program would do with it."""
    path = "/api/v1/ask" if endpoint == "ask" else "/api/v1/search"
    payload = {"query": question, "top_k": 3}
    resp, err, status = call_api(path, payload)
    request_text = f"POST {API_URL}{path}\n{json.dumps(payload, indent=2)}"
    if err:
        return request_text, err, ""
    raw = f"HTTP {status}\n{json.dumps(resp, indent=2, ensure_ascii=False)}"
    if endpoint == "ask":
        if resp["is_fallback"]:
            consumer = (f"found: False\nroute_to_human: True\nreason: {resp['refusal_reason']}")
        else:
            consumer = (f"found: True\ncitation: {resp['supporting_document']}\n"
                        f"confidence: {resp['confidence_score']:.2f}\n"
                        f"evidence_verified: {resp['evidence_verified']}")
    else:
        consumer = "\n".join(
            f"{h['rank']}. {h['document']} (p.{h['page_number']}, {h['category']})" for h in resp["results"])
    return request_text, raw, consumer


with gr.Blocks() as app:
    gr.Markdown(
        """
        # University Knowledge Assistant
        ### REST client demo
        This page talks to the assistant only through its REST API (`/api/v1/ask`, `/api/v1/search`).
        """
    )
    status_md = gr.Markdown(api_status())
    gr.Button("Refresh API status", size="sm").click(api_status, outputs=status_md)

    with gr.Tab("Ask"):
        gr.ChatInterface(
            fn=chat_interface,
            chatbot=gr.Chatbot(height=500),
            textbox=gr.Textbox(placeholder="E.g., What is the minimum attendance required for exams?",
                               container=False, scale=7),
            title=None,
            description=None,
            examples=[
                "What is the minimum attendance required to be eligible for exams?",
                "How many leaves are permitted for medical reasons?",
                "What is the penalty for using unfair means during an exam?",
                "Can I get my exam answer key?",
            ],
        )

    with gr.Tab("API inspector"):
        gr.Markdown("See the raw HTTP request and JSON response, and how a program calling the API "
                    "could use it (an example of a downstream consumer).")
        q = gr.Textbox(label="Question", value="What is the minimum attendance required to appear for exams?")
        ep = gr.Radio(["ask", "search"], value="search", label="Endpoint (search uses no LLM quota)")
        go = gr.Button("Send request")
        req_box = gr.Code(label="Request", interactive=False)
        raw_box = gr.Code(label="Response", language="json", interactive=False)
        cons_box = gr.Textbox(label="What a calling program would do with it", lines=5, interactive=False)
        go.click(inspect_call, inputs=[q, ep], outputs=[req_box, raw_box, cons_box])

if __name__ == "__main__":
    app.launch(server_name="127.0.0.1", server_port=7861, show_error=True, css=custom_css,
               theme=gr.themes.Monochrome())
