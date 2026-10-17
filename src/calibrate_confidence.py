"""
src/calibrate_confidence.py
Fit the confidence calibrator (Platt scaling) on a HELD-OUT calibration set.

Why: raw cross-encoder logits are unbounded (about -11 ... +10) and the previous pipeline
reported the LLM's self-assessed "confidence_score" (observed: 0.95-0.98 for almost every query,
including unanswerable ones). Here the confidence is P(answerable | top cross-encoder logit),
estimated with logistic regression on evaluation/calibration_set.json - a set of queries that is
DISJOINT from the 50-query benchmark so the benchmark stays an unbiased test set.

Output: Data/processed/calibration.json  {a, b, n, auc, ...}
        confidence = sigmoid(a * logit + b)
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from knowledge_assistant.config import EngineConfig  # noqa: E402
from knowledge_assistant.guardrails import detect_injection  # noqa: E402
from knowledge_assistant.retrieval import HybridRetriever  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("calibrate")

CAL_SET = SRC.parent / "evaluation" / "calibration_set.json"


HOLDOUT = SRC.parent / "evaluation" / "holdout_colloquial_set.json"


def top_logit(retriever, query: str) -> float:
    chunks, _ = retriever.retrieve(query, top_k=1)
    return float(chunks[0]["cross_encoder_score"]) if chunks else -12.0


def main(argv=None) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-version", type=int, default=2,
                    help="1 = original calibration set only, 2 = plus the colloquial additions made after the first benchmark run")
    ap.add_argument("--output", default=None, help="where to write the calibration JSON (default: Data/processed/calibration.json)")
    args = ap.parse_args(argv)

    cfg = EngineConfig()
    if args.output:
        cfg.calibration_path = Path(args.output)
    items = [i for i in json.loads(CAL_SET.read_text(encoding="utf-8")) if i.get("version", 1) <= args.max_version]
    bench = SRC.parent / "evaluation" / "expanded_test_suite.json"
    if bench.exists():  # leakage guard: calibration queries must not appear in the benchmark
        test_q = {t["query"].strip().lower() for t in json.loads(bench.read_text(encoding="utf-8"))}
        overlap = [i["query"] for i in items if i["query"].strip().lower() in test_q]
        assert not overlap, f"Calibration set overlaps with the benchmark: {overlap}"

    retriever = HybridRetriever(cfg)
    xs, ys, skipped = [], [], 0
    for it in items:
        if detect_injection(it["query"]):
            skipped += 1  # handled by the deterministic guard, never reaches the confidence gate
            continue
        xs.append(top_logit(retriever, it["query"]))
        ys.append(1 if it["is_answerable"] else 0)
    x = np.array(xs).reshape(-1, 1)
    y = np.array(ys)
    log.info("Calibration points: %d (answerable=%d, unanswerable=%d, guard-skipped=%d)",
             len(y), int(y.sum()), int((1 - y).sum()), skipped)

    clf = LogisticRegression(C=10.0, max_iter=1000).fit(x, y)
    a, b = float(clf.coef_[0][0]), float(clf.intercept_[0])
    probs = 1.0 / (1.0 + np.exp(-(a * x[:, 0] + b)))
    auc = float(roc_auc_score(y, probs))
    thr = cfg.confidence_threshold
    pred = probs >= thr
    acc = float((pred == y.astype(bool)).mean())
    tpr = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    fpr = float(pred[y == 0].mean()) if (y == 0).any() else float("nan")
    logit_at_thr = (math.log(thr / (1 - thr)) - b) / a if a else float("nan")

    hold = None
    if HOLDOUT.exists():   # fresh colloquial queries never used for fitting
        hq = json.loads(HOLDOUT.read_text(encoding="utf-8"))
        hx = np.array([top_logit(retriever, h["query"]) for h in hq])
        hy = np.array([1 if h["is_answerable"] else 0 for h in hq])
        hp = 1.0 / (1.0 + np.exp(-(a * hx + b))) >= thr
        hold = {"n": int(len(hy)), "answerable_accept_rate": float(hp[hy == 1].mean()),
                "unanswerable_accept_rate": float(hp[hy == 0].mean()),
                "accuracy": float((hp == hy.astype(bool)).mean())}
        log.info("HOLD-OUT (%d fresh colloquial queries): accept(answerable)=%.3f accept(unanswerable)=%.3f acc=%.3f",
                 hold["n"], hold["answerable_accept_rate"], hold["unanswerable_accept_rate"], hold["accuracy"])

    out = {"a": a, "b": b, "n": int(len(y)), "max_version": args.max_version, "holdout": hold, "n_answerable": int(y.sum()),
           "n_unanswerable": int((1 - y).sum()), "auc": auc, "threshold": thr,
           "accuracy_at_threshold": acc, "answerable_accept_rate": tpr,
           "unanswerable_accept_rate": fpr, "cross_encoder_logit_at_threshold": logit_at_thr,
           "feature": "top1_cross_encoder_logit", "fitted_on": "evaluation/calibration_set.json"}
    cfg.calibration_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.calibration_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Saved %s", cfg.calibration_path)
    log.info("AUC=%.3f  acc@%.2f=%.3f  accept(answerable)=%.3f  accept(unanswerable)=%.3f  logit@thr=%.2f",
             auc, thr, acc, tpr, fpr, logit_at_thr)
    return out


if __name__ == "__main__":
    main()
