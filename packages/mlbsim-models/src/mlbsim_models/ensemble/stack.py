"""Stacked meta-learner over base win-probability models.

Base probabilities are mapped to the logit scale and a small L2 logistic
regression learns per-model weights plus an intercept correction. Trained on a
held-out time slice (never the base models' own training data).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

Array = NDArray[np.float64]
_EPS = 1e-6


def _logit(p: Array) -> Array:
    q = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(q / (1.0 - q))


class StackedWinModel:
    """Fit / predict P(home win) from a matrix of base-model probabilities."""

    def __init__(self, base_names: list[str], *, c: float = 1.0) -> None:
        self.base_names = list(base_names)
        self.c = c
        self.clf = LogisticRegression(C=c, max_iter=1000)
        self._fitted = False

    def fit(self, base_probs: Any, y: Any) -> StackedWinModel:
        p = np.asarray(base_probs, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != len(self.base_names):
            raise ValueError(f"expected (n, {len(self.base_names)}) base_probs, got {p.shape}")
        self.clf.fit(_logit(p), np.asarray(y, dtype=np.int64).ravel())
        self._fitted = True
        return self

    def predict_proba(self, base_probs: Any) -> Array:
        if not self._fitted:
            raise RuntimeError("model is not fitted")
        p = np.asarray(base_probs, dtype=np.float64)
        return np.asarray(self.clf.predict_proba(_logit(p))[:, 1], dtype=np.float64)

    def weights(self) -> dict[str, float]:
        return {
            "intercept": float(self.clf.intercept_[0]),
            **dict(zip(self.base_names, self.clf.coef_.ravel().tolist(), strict=True)),
        }

    def save(self, path: str | Path) -> None:
        joblib.dump({"base_names": self.base_names, "c": self.c, "clf": self.clf}, path)

    @classmethod
    def load(cls, path: str | Path) -> StackedWinModel:
        blob = joblib.load(path)
        obj = cls(blob["base_names"], c=blob["c"])
        obj.clf = blob["clf"]
        obj._fitted = True
        return obj
