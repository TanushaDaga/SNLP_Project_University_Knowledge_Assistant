#!/usr/bin/env bash
# examples/curl_examples.sh - quick manual test of the REST API (server must be running on :8000)
#   uvicorn src.api.app:app --host 0.0.0.0 --port 8000
BASE="${BASE:-http://localhost:8000}"

echo "== health =="
curl -s "$BASE/health" | python -m json.tool

echo "== ask (grounded answer) =="
curl -s -X POST "$BASE/api/v1/ask" -H "Content-Type: application/json" \
     -d '{"query": "What is the minimum attendance percentage required to be eligible for semester-end examinations?", "top_k": 3, "strict_mode": true}' \
     | python -m json.tool

echo "== ask (out-of-domain -> standardised refusal, supporting_document = null) =="
curl -s -X POST "$BASE/api/v1/ask" -H "Content-Type: application/json" \
     -d '{"query": "Who is the current Prime Minister of India?"}' | python -m json.tool

echo "== search (retrieval only, no LLM quota) =="
curl -s -X POST "$BASE/api/v1/search" -H "Content-Type: application/json" \
     -d '{"query": "placement dress code", "top_k": 3}' | python -m json.tool

echo "== validation error (HTTP 422) =="
curl -s -i -X POST "$BASE/api/v1/ask" -H "Content-Type: application/json" -d '{"query": " "}' | head -n 12
