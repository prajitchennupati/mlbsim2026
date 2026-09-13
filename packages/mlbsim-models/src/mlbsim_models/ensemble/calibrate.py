"""Probability calibration: isotonic (default) and Platt (for comparison)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

Array = NDArray[np.float64]
_EPS = 1e-6


class IsotonicCalibrator:
    """Monotone non-parametric mapping from raw p to calibrated p."""

    def __init__(self) -> None:
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._fitted = False

    def fit(self, p: Any, y: Any) -> IsotonicCalibrator:
        self.iso.fit(
            np.asarray(p, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()
        )
        self._fitted = True
        return self

    def transform(self, p: Any) -> Array:
        if not self._fitted:
            raise RuntimeError("calibrator is not fitted")
        return np.asarray(
            self.iso.predict(np.asarray(p, dtype=np.float64).ravel()), dtype=np.float64
        )

    def save(self, path: str | Path) -> None:
        joblib.dump({"kind": "isotonic", "model": self.iso}, path)

    @classmethod
    def load(cls, path: str | Path) -> IsotonicCalibrator:
        obj = cls()
        obj.iso = joblib.load(path)["model"]
        obj._fitted = True
        return obj


class PlattCalibrator:
    """One-parameter logistic recalibration on the logit of the raw probability."""

    def __init__(self) -> None:
        self.clf = LogisticRegression(C=1e6, max_iter=1000)
        self._fitted = False

    def fit(self, p: Any, y: Any) -> PlattCalibrator:
        q = np.clip(np.asarray(p, dtype=np.float64).ravel(), _EPS, 1.0 - _EPS)
        x = np.log(q / (1.0 - q)).reshape(-1, 1)
        self.clf.fit(x, np.asarray(y, dtype=np.int64).ravel())
        self._fitted = True
        return self

    def transform(self, p: Any) -> Array:
        if not self._fitted:
            raise RuntimeError("calibrator is not fitted")
        q = np.clip(np.asarray(p, dtype=np.float64).ravel(), _EPS, 1.0 - _EPS)
        x = np.log(q / (1.0 - q)).reshape(-1, 1)
        return np.asarray(self.clf.predict_proba(x)[:, 1], dtype=np.float64)

    def save(self, path: str | Path) -> None:
        joblib.dump({"kind": "platt", "model": self.clf}, path)

    @classmethod
    def load(cls, path: str | Path) -> PlattCalibrator:
        obj = cls()
        obj.clf = joblib.load(path)["model"]
        obj._fitted = True
        return obj


Calibrator = IsotonicCalibrator | PlattCalibrator

# Isotonic regression needs a lot of points before its step function stops
# overfitting the holdout; below this, Platt's single parameter is safer.
_MIN_ISOTONIC = 1_000


def fit_calibrator(p: Any, y: Any, *, min_isotonic: int = _MIN_ISOTONIC) -> Calibrator:
    """Isotonic when the calibration set is large, else Platt."""
    n = int(np.asarray(y).size)
    cal: Calibrator = IsotonicCalibrator() if n >= min_isotonic else PlattCalibrator()
    return cal.fit(p, y)


def load_calibrator(path: str | Path) -> Calibrator:
    """Load a calibrator saved by either class, dispatching on its stored ``kind``."""
    kind = joblib.load(path).get("kind", "isotonic")
    return PlattCalibrator.load(path) if kind == "platt" else IsotonicCalibrator.load(path)
