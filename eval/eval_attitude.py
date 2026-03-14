from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import numpy as np

try:
    from eval.common import ensure_parent, load_json, safe_pearsonr
except ImportError:  # pragma: no cover
    from common import ensure_parent, load_json, safe_pearsonr

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def cp_f1(pred: List[float], gold: List[float], tolerance_sec: float) -> float:
    matched = 0
    used = set()
    for pred_time in pred:
        for idx, gold_time in enumerate(gold):
            if idx in used:
                continue
            if abs(pred_time - gold_time) <= tolerance_sec:
                matched += 1
                used.add(idx)
                break
    precision = matched / (len(pred) + 1e-8)
    recall = matched / (len(gold) + 1e-8)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def silence_f1(pred: List[int], gold: List[int]) -> float:
    pred_arr = np.array(pred, dtype=int)
    gold_arr = np.array(gold, dtype=int)
    tp = np.sum((pred_arr == 1) & (gold_arr == 1))
    fp = np.sum((pred_arr == 1) & (gold_arr == 0))
    fn = np.sum((pred_arr == 0) & (gold_arr == 1))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def main():
    parser = argparse.ArgumentParser(description="Evaluate AE metrics from gold and prediction JSON files")
    parser.add_argument("--gold", default="eval/data/attitude_gold.json", help="Path to AE gold json")
    parser.add_argument("--pred", default="examples/example_ae_output.json", help="Path to AE prediction json")
    parser.add_argument("--out", default="eval/results/attitude_summary.json", help="Output path for summary json")
    args = parser.parse_args()

    gold = load_json(Path(args.gold))
    pred = load_json(Path(args.pred))
    pred_map = {item["slide_id"]: item for item in pred}

    f1_by_tolerance = {1: [], 3: [], 5: []}
    speech_rate_mae = []
    silence_scores = []
    energy_corr = []

    for gold_item in gold:
        pred_item = pred_map.get(gold_item["slide_id"], gold_item)

        pred_cp = pred_item.get("change_points_pred", [cp["time_sec"] for cp in pred_item.get("change_points", [])])
        gold_cp = gold_item.get("change_points_gold", [])
        for tol in f1_by_tolerance:
            f1_by_tolerance[tol].append(cp_f1(pred_cp, gold_cp, tol))

        pred_rate = pred_item.get("speech_rate_pred", pred_item.get("features", {}).get("avg_speech_rate", 0.0))
        gold_rate = gold_item.get("speech_rate_gold", 0.0)
        speech_rate_mae.append(abs(pred_rate - gold_rate))

        pred_silence = pred_item.get("silence_pred", gold_item.get("silence_gold", []))
        gold_silence = gold_item.get("silence_gold", [])
        silence_scores.append(silence_f1(pred_silence, gold_silence))

        if "feature_energy" in gold_item and "perception_energy" in gold_item:
            energy_corr.append(
                safe_pearsonr(np.array(gold_item["feature_energy"], dtype=float), np.array(gold_item["perception_energy"], dtype=float))
            )

    result = {
        "f1_tol_1s": float(np.mean(f1_by_tolerance[1])) if f1_by_tolerance[1] else 0.0,
        "f1_tol_3s": float(np.mean(f1_by_tolerance[3])) if f1_by_tolerance[3] else 0.0,
        "f1_tol_5s": float(np.mean(f1_by_tolerance[5])) if f1_by_tolerance[5] else 0.0,
        "speech_rate_mae": float(np.mean(speech_rate_mae)) if speech_rate_mae else 0.0,
        "silence_f1": float(np.mean(silence_scores)) if silence_scores else 0.0,
        "energy_perception_corr": float(np.mean(energy_corr)) if energy_corr else 0.0,
    }

    ensure_parent(Path(args.out))
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    logger.info("saved attitude summary to %s", args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
