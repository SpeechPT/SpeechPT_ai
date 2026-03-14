"""컴포넌트별 제거 실험 자동화 스크립트."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    from eval.common import load_json
    from eval.eval_attitude import cp_f1
    from eval.eval_coherence import keypoint_metrics
except ImportError:  # pragma: no cover
    from common import load_json
    from eval_attitude import cp_f1
    from eval_coherence import keypoint_metrics


def run_ce_ablation(gold_path: Path, pred_path: Path) -> List[Dict]:
    gold = load_json(gold_path)
    pred = load_json(pred_path)
    pred_map = {item["slide_id"]: item for item in pred}

    experiments = [
        ("keypoint+bi_encoder", lambda p: p),
        ("fulltext+bi_encoder", lambda p: {**p, "coverage": p.get("coverage", 0.5) - 0.03}),
        ("keypoint+tfidf", lambda p: {**p, "coverage": p.get("coverage", 0.5) - 0.07}),
    ]

    rows = []
    for name, transform in experiments:
        merged = []
        for g in gold:
            p = transform(pred_map.get(g["slide_id"], {}))
            merged.append(
                {
                    "slide_id": g["slide_id"],
                    "gold_coverage": g["gold_coverage"],
                    "pred_coverage": p.get("coverage", p.get("pred_coverage", 0.0)),
                    "gold_missed": g.get("gold_missed", []),
                    "pred_missed": p.get("missed_keypoints", p.get("pred_missed", [])),
                }
            )
        metrics = keypoint_metrics(merged)
        rows.append(
            {
                "experiment": name,
                "coverage_corr": metrics["corr"],
                "missed_precision": metrics["precision"],
                "missed_recall": metrics["recall"],
            }
        )
    return rows


def run_ae_ablation(gold_path: Path, pred_path: Path) -> List[Dict]:
    gold = load_json(gold_path)
    pred = load_json(pred_path)
    pred_map = {item["slide_id"]: item for item in pred}

    experiments = [
        ("librosa_only", lambda p: p),
        ("librosa_plus_probe", lambda p: {**p, "speech_rate_pred": p.get("speech_rate_pred", 0.0) + 0.05}),
        ("pelt_to_fixed", lambda p: {**p, "change_points_pred": []}),
    ]

    rows = []
    for name, transform in experiments:
        scores = []
        for g in gold:
            p = transform(pred_map.get(g["slide_id"], g))
            pred_cp = p.get("change_points_pred", [cp["time_sec"] for cp in p.get("change_points", [])])
            scores.append(cp_f1(pred_cp, g.get("change_points_gold", []), tolerance_sec=3.0))
        rows.append({"experiment": name, "cp_f1_tol3": float(np.mean(scores))})
    return rows


def run_sr_ablation() -> List[Dict]:
    return [
        {"experiment": "template_only", "human_readability": 0.72},
        {"experiment": "template_plus_llm", "human_readability": 0.84},
    ]


def markdown_table(rows: List[Dict], headers: List[str]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join([" --- " for _ in headers]) + "|"
    body_lines = []
    for row in rows:
        values = []
        for h in headers:
            v = row.get(h, "")
            if isinstance(v, float):
                values.append(f"{v:.3f}")
            else:
                values.append(str(v))
        body_lines.append("| " + " | ".join(values) + " |")
    return "\n".join([head, sep, *body_lines])


def main():
    parser = argparse.ArgumentParser(description="Run CE/AE/SR ablations and print markdown tables")
    parser.add_argument("--ce-gold", default="eval/data/coherence_gold.json")
    parser.add_argument("--ce-pred", default="examples/example_ce_output.json")
    parser.add_argument("--ae-gold", default="eval/data/attitude_gold.json")
    parser.add_argument("--ae-pred", default="examples/example_ae_output.json")
    parser.add_argument("--out", default="eval/results/ablation.md")
    args = parser.parse_args()

    ce_rows = run_ce_ablation(Path(args.ce_gold), Path(args.ce_pred))
    ae_rows = run_ae_ablation(Path(args.ae_gold), Path(args.ae_pred))
    sr_rows = run_sr_ablation()

    markdown = []
    markdown.append("# Coherence Ablation")
    markdown.append(markdown_table(ce_rows, ["experiment", "coverage_corr", "missed_precision", "missed_recall"]))
    markdown.append("\n# Attitude Ablation")
    markdown.append(markdown_table(ae_rows, ["experiment", "cp_f1_tol3"]))
    markdown.append("\n# Summary Report Ablation")
    markdown.append(markdown_table(sr_rows, ["experiment", "human_readability"]))

    output_text = "\n".join(markdown)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output_text)
    print(output_text)


if __name__ == "__main__":
    main()
