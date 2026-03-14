"""프로소디 피처 시계열의 변화점을 검출한다."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import ruptures as rpt

logger = logging.getLogger(__name__)


@dataclass
class ChangePoint:
    time_sec: float
    type: str  # "speed_drop", "energy_drop", "pitch_shift", "general"
    magnitude: float


def _zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    sigma = np.nanstd(x) + 1e-8
    return (x - mu) / sigma


def detect_change_points(features: Dict[str, np.ndarray], frame_times: np.ndarray, config: Dict | None = None) -> List[ChangePoint]:
    cfg = config or {}
    cp_cfg = cfg.get("change_point", {})
    penalty = float(cp_cfg.get("penalty", 3.0))
    model = cp_cfg.get("model", "rbf")
    merge_tol = float(cp_cfg.get("merge_tolerance_sec", 2.0))

    # 선택 피처 구성
    series_list: List[np.ndarray] = []
    for key in ["energy", "pitch", "speech_rate_per_sec"]:
        if key in features:
            series_list.append(_zscore(np.nan_to_num(features[key], nan=0.0)))
    if not series_list:
        return []
    mat = np.vstack(series_list).T  # shape (T, F)

    algo = rpt.Pelt(model=model).fit(mat)
    change_indices = algo.predict(pen=penalty)
    change_indices = [idx for idx in change_indices if idx < len(frame_times)]  # ignore end marker

    cps: List[ChangePoint] = []
    for idx in change_indices:
        t = float(frame_times[idx])
        before = mat[max(0, idx - 3) : idx].mean(axis=0)
        after = mat[idx : min(len(mat), idx + 3)].mean(axis=0)
        diff = after - before
        dominant = int(np.argmax(np.abs(diff)))
        mag = float(np.abs(diff[dominant]))
        key = ["energy", "pitch", "speech_rate_per_sec"][dominant]
        if key == "energy":
            ctype = "energy_drop" if diff[dominant] < 0 else "energy_rise"
        elif key == "speech_rate_per_sec":
            ctype = "speed_drop" if diff[dominant] < 0 else "speed_rise"
        else:
            ctype = "pitch_shift"
        cps.append(ChangePoint(time_sec=t, type=ctype, magnitude=mag))

    # 병합: 시간 차이가 merge_tol 이하이면 하나로 합치기
    merged: List[ChangePoint] = []
    for cp in sorted(cps, key=lambda c: c.time_sec):
        if merged and (cp.time_sec - merged[-1].time_sec) <= merge_tol:
            # keep the larger magnitude
            if cp.magnitude > merged[-1].magnitude:
                merged[-1] = cp
        else:
            merged.append(cp)
    return merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dummy_times = np.linspace(0, 10, 50)
    dummy_energy = np.concatenate([np.ones(25), np.ones(25) * -2])
    feats = {"energy": dummy_energy, "pitch": np.zeros_like(dummy_energy), "speech_rate_per_sec": np.zeros_like(dummy_energy)}
    cps = detect_change_points(feats, dummy_times)
    for cp in cps:
        logger.info("CP @ %.2fs type=%s mag=%.2f", cp.time_sec, cp.type, cp.magnitude)
