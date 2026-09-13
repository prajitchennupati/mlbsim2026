"""A small feed-forward neural net (multi-layer perceptron) for game win
probability -- same fit/predict_proba/save/load interface as WinLogit and
GBMWinModel, so it drops into the same training/ensemble plumbing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Array = NDArray[np.float64]


class MLPWinModel:
    """Fit / predict P(home team wins) from a diff-feature matrix."""

    def __init__(
        self,
        feature_names: list[str],
        *,
        hidden_layer_sizes: tuple[int, ...] = (16, 8),
        alpha: float = 1e-2,
        max_iter: int = 2000,
        random_state: int = 0,
    ) -> None:
        self.feature_names = list(feature_names)
        self.params = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "alpha": alpha,
            "max_iter": max_iter,
            "random_state": random_state,
            "early_stopping": True,
            "validation_fraction": 0.15,
        }
        self.pipeline: Pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", MLPClassifier(**self.params)),
            ]
        )
        self._fitted = False

    def fit(self, x: Any, y: Any) -> MLPWinModel:
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

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {"feature_names": self.feature_names, "params": self.params, "pipeline": self.pipeline},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> MLPWinModel:
        blob = joblib.load(path)
        obj = cls(blob["feature_names"])
        obj.params = blob["params"]
        obj.pipeline = blob["pipeline"]
        obj._fitted = True
        return obj
