"""Poisson GLM for a team's runs, plus a score-distribution grid.

For M2 the two sides are modelled independently (a bivariate-Poisson coupling is
a documented later refinement). The grid and its derived probabilities are pure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import statsmodels.api as sm
from numpy.typing import NDArray

Array = NDArray[np.float64]
_MAX_RUNS = 25
_MU_CLIP = (1e-6, 40.0)  # a team's expected runs in a 9-inning game; guards GLM divergence


class RunsPoisson:
    """Fit / predict a team's expected runs (μ) from a feature matrix."""

    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = list(feature_names)
        self._result: Any = None

    def fit(self, x: Any, y_runs: Any) -> RunsPoisson:
        x_arr = sm.add_constant(np.asarray(x, dtype=np.float64), has_constant="add")
        y_arr = np.asarray(y_runs, dtype=np.float64).ravel()
        self._result = sm.GLM(y_arr, x_arr, family=sm.families.Poisson()).fit()
        return self

    def predict_mu(self, x: Any) -> Array:
        if self._result is None:
            raise RuntimeError("model is not fitted")
        x_arr = sm.add_constant(np.asarray(x, dtype=np.float64), has_constant="add")
        mu = np.asarray(self._result.predict(x_arr), dtype=np.float64)
        # A diverged GLM (separable / collinear training data) can return inf/NaN;
        # clamp to a physically sane range so downstream grids stay finite.
        return np.clip(np.nan_to_num(mu, nan=_MU_CLIP[0]), *_MU_CLIP)

    def save(self, path: str | Path) -> None:
        joblib.dump({"feature_names": self.feature_names, "result": self._result}, path)

    @classmethod
    def load(cls, path: str | Path) -> RunsPoisson:
        blob = joblib.load(path)
        obj = cls(blob["feature_names"])
        obj._result = blob["result"]
        return obj


def _pois_pmf(mu: float, k: int) -> Array:
    """Poisson pmf over 0..k-1 via the P(i) = P(i-1)*mu/i recurrence."""
    mu = min(max(float(mu), 1e-9), _MU_CLIP[1])
    pmf = np.empty(k, dtype=np.float64)
    pmf[0] = np.exp(-mu)
    for i in range(1, k):
        pmf[i] = pmf[i - 1] * mu / i
    return pmf


def score_grid(mu_home: float, mu_away: float, *, max_runs: int = _MAX_RUNS) -> Array:
    """Joint pmf ``grid[h, a] = P(home scores h, away scores a)`` (independent Poisson)."""
    ph = _pois_pmf(mu_home, max_runs)
    pa = _pois_pmf(mu_away, max_runs)
    grid = np.outer(ph, pa)
    total = grid.sum()
    return grid / total if total else grid


def _dist_from_axis(pmf: Array, *, top: int = 15) -> dict[str, float]:
    return {str(i): round(float(p), 6) for i, p in enumerate(pmf[:top])}


def derive_game_probs(grid: Array) -> dict[str, Any]:
    """Read game-level probabilities off a normalised score grid."""
    n = grid.shape[0]
    h_idx, a_idx = np.indices((n, n))

    p_home_win = float(grid[h_idx > a_idx].sum())
    p_tie = float(grid[h_idx == a_idx].sum())
    p_away_win = float(grid[h_idx < a_idx].sum())
    # Ties in 9 go to extra innings; split the residual mass evenly.
    home_win_prob = p_home_win + 0.5 * p_tie

    home_runs_pmf = grid.sum(axis=1)
    away_runs_pmf = grid.sum(axis=0)
    exp_home = float((np.arange(n) * home_runs_pmf).sum())
    exp_away = float((np.arange(n) * away_runs_pmf).sum())

    diff = h_idx - a_idx
    total = h_idx + a_idx
    flat = grid.ravel()
    order = np.argsort(flat)[::-1][:5]
    most_likely = [
        {
            "home": int(h_idx.ravel()[i]),
            "away": int(a_idx.ravel()[i]),
            "p": round(float(flat[i]), 5),
        }
        for i in order
    ]

    return {
        "home_win_prob": round(home_win_prob, 5),
        "away_win_prob": round(1.0 - home_win_prob, 5),
        "p_extra_innings": round(p_tie, 5),
        "p_home_win_reg": round(p_home_win, 5),
        "p_away_win_reg": round(p_away_win, 5),
        "exp_home_runs": round(exp_home, 3),
        "exp_away_runs": round(exp_away, 3),
        "exp_total_runs": round(exp_home + exp_away, 3),
        "exp_run_diff": round(exp_home - exp_away, 3),
        "p_shutout_home": round(float(grid[0, 1:].sum()), 5),
        "p_shutout_away": round(float(grid[1:, 0].sum()), 5),
        "p_one_run_game": round(float(grid[np.abs(diff) == 1].sum()), 5),
        "total_runs_dist": {str(t): round(float(grid[total == t].sum()), 6) for t in range(0, 25)},
        "home_score_dist": _dist_from_axis(home_runs_pmf),
        "away_score_dist": _dist_from_axis(away_runs_pmf),
        "most_likely_scores": most_likely,
    }
