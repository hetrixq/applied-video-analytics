from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .utils import compute_metrics


@dataclass
class Threshold1D:
    threshold: float

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return (scores >= self.threshold).astype(int)


def fit_threshold(
    scores: np.ndarray, y_true: np.ndarray, n_thresholds: int, metric: str
) -> Threshold1D:
    scores = scores.astype(float)
    y_true = y_true.astype(int)

    if len(scores) == 0:
        return Threshold1D(0.0)

    mn, mx = float(scores.min()), float(scores.max())
    if abs(mx - mn) < 1e-12:
        return Threshold1D(mn)

    best_t, best_m = mn, -1.0
    for t in np.linspace(mn, mx, int(n_thresholds)):
        y_pred = (scores >= t).astype(int)
        m = compute_metrics(y_true, y_pred)[metric]
        if m > best_m:
            best_m = m
            best_t = float(t)

    return Threshold1D(best_t)
