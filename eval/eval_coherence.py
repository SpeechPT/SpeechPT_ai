from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import numpy as np

try:
    from eval.common import bootstrap_ci, ensure_parent, load_json, safe_pearsonr
except ImportError:  # pragma: no cover
    from common import bootstrap_ci, ensure_parent, load_json, safe_pearsonr

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def keypoint_metrics(data: List[dict]) -> dict:
    gold_cov = np.array([d["gold_coverage"] for d in data], dtype=float)
    pred_cov = np.array([d["pred_coverage"] for d in data], dtype=float)

    tp = fp = fn = 0
    for item in data:
        gold_missed = set(item.get("gold_missed", []))
        pred_missed = set(item.get("pred_missed", []))
        tp += len(gold_missed & pred_missed)
        fp += len(pred_missed - gold_missed)
        fn += len(gold_missed - pred_missed)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    accuracy = float(np.mean((pred_cov >= 0.7) == (gold_cov >= 0.7)))

    return {
        "corr": safe_pearsonr(gold_cov, pred_cov),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": accuracy,
    }


def pr_curve(data: List[dict], out_path: Path) -> List[dict]:
    gold_cov = np.array([d["gold_coverage"] for d in data], dtype=float)
    pred_cov = np.array([d["pred_coverage"] for d in data], dtype=float)

    points = []
    for threshold in np.linspace(0, 1, 21):
        gold_bin = gold_cov >= threshold
        pred_bin = pred_cov >= threshold
        tp = np.sum(gold_bin & pred_bin)
        fp = np.sum((~gold_bin) & pred_bin)
        fn = np.sum(gold_bin & (~pred_bin))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        points.append({"threshold": float(threshold), "precision": float(precision), "recall": float(recall)})

    ensure_parent(out_path)
    out_path.write_text(json.dumps(points, ensure_ascii=False, indent=2))
    return points


def main():
    parser = argparse.ArgumentParser(description="Evaluate CE metrics from gold and prediction JSON files")
    parser.add_argument("--gold", default="eval/data/coherence_gold.json", help="Path to CE gold json")
    parser.add_argument("--pred", default="examples/example_ce_output.json", help="Path to CE prediction json")
    parser.add_argument("--out", default="eval/results/coherence_pr_curve.json", help="Output path for PR curve json")
    parser.add_argument("--summary-out", default="eval/results/coherence_summary.json", help="Output path for summary json")
    args = parser.parse_args()

    gold = load_json(Path(args.gold))
    pred = load_json(Path(args.pred))

    pred_map = {item["slide_id"]: item for item in pred}
    merged = []
    for gold_item in gold:
        pred_item = pred_map.get(gold_item["slide_id"], {})
        merged.append(
            {
                "slide_id": gold_item["slide_id"],
                "gold_coverage": gold_item["gold_coverage"],
                "pred_coverage": pred_item.get("coverage", pred_item.get("pred_coverage", 0.0)),
                "gold_missed": gold_item.get("gold_missed", []),
                "pred_missed": pred_item.get("missed_keypoints", pred_item.get("pred_missed", [])),
            }
        )

    metrics = keypoint_metrics(merged)
    coverage_values = np.array([item["pred_coverage"] for item in merged], dtype=float)
    ci_low, ci_high = bootstrap_ci(coverage_values, lambda x: float(np.mean(x)))
    points = pr_curve(merged, Path(args.out))

    summary = {"metrics": metrics, "coverage_ci": [ci_low, ci_high], "pr_curve_points": len(points)}
    ensure_parent(Path(args.summary_out))
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    logger.info("coverage_corr=%.3f precision=%.3f recall=%.3f accuracy=%.3f", metrics["corr"], metrics["precision"], metrics["recall"], metrics["accuracy"])
    logger.info("coverage 95%% CI: [%.3f, %.3f]", ci_low, ci_high)
    logger.info("saved summary=%s pr_curve=%s", args.summary_out, args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
