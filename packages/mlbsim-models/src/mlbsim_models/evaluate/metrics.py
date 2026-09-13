"""Proper scoring rules and calibration metrics (pure NumPy, no sklearn).

Binary-outcome helpers take ``y_true`` in {0, 1} and ``p`` the predicted
probability of the positive class. Count helpers take observed and predicted
means.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

_EPS = 1e-15


def _prep(y_true: Any, p: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    y = np.asarray(y_true, dtype=np.float64).ravel()
    q = np.clip(np.asarray(p, dtype=np.float64).ravel(), _EPS, 1.0 - _EPS)
    if y.shape != q.shape:
        raise ValueError(f"shape mismatch: y_true {y.shape} vs p {q.shape}")
    return y, q


def log_loss(y_true: Any, p: Any) -> float:
    """Mean binary cross-entropy (natural log)."""
    y, q = _prep(y_true, p)
    return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))


def brier_score(y_true: Any, p: Any) -> float:
    y, q = _prep(y_true, p)
    return float(np.mean((q - y) ** 2))


def accuracy(y_true: Any, p: Any, *, threshold: float = 0.5) -> float:
    y, q = _prep(y_true, p)
    return float(np.mean((q >= threshold).astype(np.float64) == y))


def roc_auc(y_true: Any, p: Any) -> float:
    """AUC via the Mann-Whitney U statistic (handles ties with average ranks)."""
    y, q = _prep(y_true, p)
    n_pos = float(np.sum(y == 1.0))
    n_neg = float(np.sum(y == 0.0))
    if n_pos == 0.0 or n_neg == 0.0:
        return float("nan")
    order = np.argsort(q, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_q = q[order]
    i = 0
    while i < len(sorted_q):
        j = i
        while j < len(sorted_q) and sorted_q[j] == sorted_q[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    rank_sum_pos = float(np.sum(ranks[y == 1.0]))
    return (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def calibration_table(y_true: Any, p: Any, *, n_bins: int = 10) -> list[dict[str, float]]:
    """Equal-width reliability bins on [0, 1]."""
    y, q = _prep(y_true, p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(q, edges[1:-1], right=False), 0, n_bins - 1)
    rows: list[dict[str, float]] = []
    for b in range(n_bins):
        mask = idx == b
        n = int(np.sum(mask))
        if n == 0:
            continue
        rows.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "n": float(n),
                "mean_pred": float(np.mean(q[mask])),
                "mean_actual": float(np.mean(y[mask])),
            }
        )
    return rows


def expected_calibration_error(y_true: Any, p: Any, *, n_bins: int = 10) -> float:
    """Sample-weighted mean |mean_pred - mean_actual| across reliability bins."""
    y, _ = _prep(y_true, p)
    table = calibration_table(y_true, p, n_bins=n_bins)
    total = float(len(y))
    return float(sum(r["n"] / total * abs(r["mean_pred"] - r["mean_actual"]) for r in table))


def mae(y_true: Any, y_pred: Any) -> float:
    a = np.asarray(y_true, dtype=np.float64).ravel()
    b = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean(np.abs(a - b)))


def rmse(y_true: Any, y_pred: Any) -> float:
    a = np.asarray(y_true, dtype=np.float64).ravel()
    b = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((a - b) ** 2)))


def poisson_deviance(y_true: Any, mu: Any) -> float:
    """Mean Poisson deviance: 2*Σ[y*log(y/μ) - (y - μ)] / n."""
    y = np.asarray(y_true, dtype=np.float64).ravel()
    m = np.clip(np.asarray(mu, dtype=np.float64).ravel(), _EPS, None)
    term = np.where(y > 0.0, y * np.log(y / m), 0.0) - (y - m)
    return float(2.0 * np.mean(term))


def classification_report(y_true: Any, p: Any, *, n_bins: int = 10) -> dict[str, Any]:
    """Bundle the common binary metrics into one dict."""
    return {
        "n": int(np.asarray(y_true).size),
        "accuracy": accuracy(y_true, p),
        "log_loss": log_loss(y_true, p),
        "brier": brier_score(y_true, p),
        "roc_auc": roc_auc(y_true, p),
        "ece": expected_calibration_error(y_true, p, n_bins=n_bins),
        "base_rate": float(np.mean(np.asarray(y_true, dtype=np.float64))),
        "mean_pred": float(np.mean(np.asarray(p, dtype=np.float64))),
    }
