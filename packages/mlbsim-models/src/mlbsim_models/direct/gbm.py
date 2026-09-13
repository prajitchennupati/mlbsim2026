"""Gradient-boosted trees for game win probability.

``HistGradientBoostingClassifier`` (scikit-learn's histogram-based GBM, the same
family as LightGBM) over the fixed ``DIFF_FEATURE_ORDER`` diff features. It is a
non-linear counterpart to :class:`WinLogit` and a third base model for the
ensemble stack. Kept lightly regularised — the feature set is small (13) and one
season of training data is easy to overfit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingClassifier

Array = NDArray[np.float64]


class GBMWinModel:
    """Fit / predict P(home team wins) from a diff-feature matrix."""

    def __init__(
        self,
        feature_names: list[str],
        *,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        max_iter: int = 300,
        l2_regularization: float = 1.0,
        min_samples_leaf: int = 40,
        random_state: int = 0,
    ) -> None:
        self.feature_names = list(feature_names)
        self.params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "l2_regularization": l2_regularization,
            "min_samples_leaf": min_samples_leaf,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "random_state": random_state,
        }
        self.clf = HistGradientBoostingClassifier(**self.params)
        self._fitted = False

    def fit(self, x: Any, y: Any) -> GBMWinModel:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64).ravel()
        if x_arr.shape[1] != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, got {x_arr.shape[1]}")
        self.clf.fit(x_arr, y_arr)
        self._fitted = True
        return self

    def predict_proba(self, x: Any) -> Array:
        """P(home win) for each row."""
        if not self._fitted:
            raise RuntimeError("model is not fitted")
        x_arr = np.asarray(x, dtype=np.float64)
        proba = np.asarray(self.clf.predict_proba(x_arr), dtype=np.float64)
        return proba[:, 1]

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {"feature_names": self.feature_names, "params": self.params, "clf": self.clf}, path
        )

    @classmethod
    def load(cls, path: str | Path) -> GBMWinModel:
        blob = joblib.load(path)
        obj = cls(blob["feature_names"])
        obj.params = blob["params"]
        obj.clf = blob["clf"]
        obj._fitted = True
        return obj
