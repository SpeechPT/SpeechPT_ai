from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import stats


def load_json(path: Path):
    return json.loads(path.read_text())


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    corr, _ = stats.pearsonr(x, y)
    return float(corr)


def bootstrap_ci(data: np.ndarray, fn: Callable[[np.ndarray], float], iters: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    if len(data) == 0:
        return 0.0, 0.0
    stats_list = []
    n = len(data)
    for _ in range(iters):
        idx = np.random.randint(0, n, n)
        stats_list.append(fn(data[idx]))
    low = float(np.percentile(stats_list, 100 * (alpha / 2)))
    high = float(np.percentile(stats_list, 100 * (1 - alpha / 2)))
    return low, high


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
