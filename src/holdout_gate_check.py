"""
src/holdout_gate_check.py
Measures how the ORIGINAL confidence gate (calibration fitted on direct questions, Data/processed/calibration_v1_initial.json)
and the RE-FITTED gate (colloquial queries added, Data/processed/calibration.json) behave on a fresh hold-out of 16
colloquial queries (evaluation/holdout_colloquial_set.json) that were used for neither fit and are not in the benchmark.
No LLM is involved.  Output: evaluation/holdout_gate_comparison.json
"""

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from knowledge_assistant.config import EngineConfig  # noqa: E402
from knowledge_assistant.guardrails import Calibrator  # noqa: E402
from knowledge_assistant.retrieval import HybridRetriever  # noqa: E402


def main() -> dict:
    cfg = EngineConfig()
    retr = HybridRetriever(cfg)
    hold = json.loads((ROOT / "evaluation" / "holdout_colloquial_set.json").read_text(encoding="utf-8"))
    gates = {"initial_gate": Calibrator.load(ROOT / "Data" / "processed" / "calibration_v1_initial.json"),
             "refitted_gate": Calibrator.load(cfg.calibration_path)}
    rows = []
    for h in hold:
        chunks, _ = retr.retrieve(h["query"], top_k=1)
        z = float(chunks[0]["cross_encoder_score"]) if chunks else -12.0
        rows.append({"query_id": h["query_id"], "query": h["query"], "is_answerable": h["is_answerable"], "top_logit": z,
                     **{name: g(z) for name, g in gates.items()}})
    out = {"n": len(rows), "threshold": cfg.confidence_threshold, "per_query": rows}
    for name in gates:
        a = [r[name] >= cfg.confidence_threshold for r in rows if r["is_answerable"]]
        u = [r[name] >= cfg.confidence_threshold for r in rows if not r["is_answerable"]]
        out[name] = {"answerable_accepted": sum(a), "n_answerable": len(a), "unanswerable_accepted": sum(u),
                     "n_unanswerable": len(u)}
    (ROOT / "evaluation" / "holdout_gate_comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name in gates:
        o = out[name]
        print(f"{name}: answerable accepted {o['answerable_accepted']}/{o['n_answerable']}, "
              f"unanswerable accepted {o['unanswerable_accepted']}/{o['n_unanswerable']}")
    return out


if __name__ == "__main__":
    main()
