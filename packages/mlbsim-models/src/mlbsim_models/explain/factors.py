"""Exact SHAP factors for the logistic win model.

For a linear model on standardised features the SHAP value of feature ``j`` is
``coef_j * z_j`` (with ``E[z_j] = 0``), and ``logit(p) = intercept + Sum_j phi_j``
exactly — no ``shap`` dependency needed. The GBM path (later) uses
``shap.TreeExplainer``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

# Human-readable, direction-aware labels for the fs_v1 diff features.
_LABELS: dict[str, str] = {
    "d_win_pct": "season win rate",
    "d_win_pct_l10": "win rate over the last 10 games",
    "d_rs_pg": "runs scored per game",
    "d_ra_pg": "runs allowed per game",
    "d_run_diff_pg": "run differential per game",
    "d_run_diff_pg_l10": "run differential per game (last 10)",
    "d_rest_days": "days of rest",
    "d_sp_era": "starting-pitcher ERA",
    "d_sp_fip": "starting-pitcher FIP",
    "d_sp_k_pct": "starting-pitcher strikeout rate",
    "d_sp_bb_pct": "starting-pitcher walk rate",
    "d_sp_ip_per_start": "starting-pitcher innings per start",
    "d_sp_era_l3": "starting-pitcher ERA over the last 3 starts",
}


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def linear_shap(win_logit: Any, x_row: Any) -> tuple[dict[str, float], float]:
    """Return ``(feature -> logit contribution, base_logit)`` for one game.

    ``win_logit`` is a fitted :class:`mlbsim_models.direct.WinLogit`.
    """
    pipe = win_logit.pipeline
    scaler = pipe.named_steps["scale"]
    clf = pipe.named_steps["clf"]
    x = np.asarray(x_row, dtype=np.float64).ravel()
    z = (x - scaler.mean_) / scaler.scale_
    phi = clf.coef_.ravel() * z
    base = float(clf.intercept_[0])
    return {name: float(p) for name, p in zip(win_logit.feature_names, phi, strict=True)}, base


def top_factors(
    contribs: dict[str, float],
    base_logit: float,
    *,
    k: int = 4,
    home_name: str = "home",
    away_name: str = "away",
) -> list[dict[str, Any]]:
    """Rank factors by absolute effect and describe each in words."""
    total = base_logit + sum(contribs.values())
    p_full = _sigmoid(total)
    ranked = sorted(contribs.items(), key=lambda kv: -abs(kv[1]))[:k]
    out: list[dict[str, Any]] = []
    for feat, phi in ranked:
        favours = home_name if phi > 0 else away_name
        p_without = _sigmoid(total - phi)
        out.append(
            {
                "feature": feat,
                "label": _LABELS.get(feat, feat),
                "logit_contribution": round(phi, 4),
                "favours": favours,
                "prob_shift": round(p_full - p_without, 4),
            }
        )
    return out
