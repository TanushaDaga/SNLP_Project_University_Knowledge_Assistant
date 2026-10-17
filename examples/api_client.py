"""
examples/api_client.py - minimal Python client for the University Knowledge Assistant REST API.

    uvicorn src.api.app:app --host 0.0.0.0 --port 8000        # in another terminal
    python examples/api_client.py "What is the minimum attendance for exams?"

Only `requests` is needed. The same class can be copied into Group 10 (Admin Copilot) or Group 8
(Placement Intelligence) code bases.
"""

import json
import sys
import time
from typing import Optional

import requests


class KnowledgeAssistantClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 60.0, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    def _post(self, path: str, payload: dict) -> dict:
        for attempt in range(self.retries + 1):
            r = requests.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
            if r.status_code == 429:                      # rate limited -> honour Retry-After
                time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
                continue
            if r.status_code == 503 and attempt < self.retries:   # engine still warming up
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("API unavailable after retries")

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def search(self, query: str, top_k: int = 3) -> dict:
        return self._post("/api/v1/search", {"query": query, "top_k": top_k})

    def ask(self, query: str, top_k: int = 3, strict_mode: bool = True) -> dict:
        return self._post("/api/v1/ask", {"query": query, "top_k": top_k, "strict_mode": strict_mode})


def answer_or_none(client: KnowledgeAssistantClient, question: str) -> Optional[str]:
    """Typical integration pattern: only surface an answer that is grounded; otherwise fall back to a human."""
    resp = client.ask(question)
    if resp["is_fallback"] or resp["supporting_document"] is None:
        return None
    return f'{resp["answer"]}  [source: {resp["supporting_document"]}, confidence {resp["confidence_score"]:.2f}]'


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is the minimum attendance required to appear for exams?"
    c = KnowledgeAssistantClient()
    print("health:", json.dumps(c.health()))
    print("\n--- /api/v1/ask ---")
    print(json.dumps(c.ask(q), indent=2, ensure_ascii=False))
    print("\n--- /api/v1/search ---")
    for hit in c.search(q, top_k=3)["results"]:
        print(f'{hit["rank"]}. {hit["document"]} (score {hit["similarity_score"]:.2f}) p.{hit["page_number"]}')
