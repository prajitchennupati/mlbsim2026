"""L2-regularised logistic regression on home-away differenced features.

A deliberately simple, well-calibrated baseline for game win probability. The
feature order is fixed (``mlbsim_features.DIFF_FEATURE_ORDER``) so a stored model
and a live prediction always line up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Array = NDArray[np.float64]


class WinLogit:
    """Fit / predict P(home team wins) from a diff-feature matrix."""

    def __init__(self, feature_names: list[str], *, c: float = 1.0) -> None:
        self.feature_names = list(feature_names)
        self.c = c
        self.pipeline: Pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=c, max_iter=1000)),
            ]
        )
        self._fitted = False

    def fit(self, x: Any, y: Any) -> WinLogit:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64).ravel()
        if x_arr.shape[1] != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, got {x_arr.shape[1]}")
        self.pipeline.fit(x_arr, y_arr)
        self._fitted = True
        return self

    def predict_proba(self, x: Any) -> Array:
        """P(home win) for each row."""
        if not self._fitted:
            raise RuntimeError("model is not fitted")
        x_arr = np.asarray(x, dtype=np.float64)
        proba = np.asarray(self.pipeline.predict_proba(x_arr), dtype=np.float64)
        return proba[:, 1]

    def coefficients(self) -> dict[str, float]:
        clf: LogisticRegression = self.pipeline.named_steps["clf"]
        return dict(zip(self.feature_names, clf.coef_.ravel().tolist(), strict=True))

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {"feature_names": self.feature_names, "c": self.c, "pipeline": self.pipeline},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> WinLogit:
        blob = joblib.load(path)
        obj = cls(blob["feature_names"], c=blob["c"])
        obj.pipeline = blob["pipeline"]
        obj._fitted = True
        return obj
