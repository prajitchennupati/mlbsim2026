"""Real SHAP explanations for gbm_v1, via ``shap.TreeExplainer`` against the
fitted ``HistGradientBoostingClassifier`` -- computed offline as part of
training/prediction (see ``mlbsim_models.pipelines.ml_stack``), never by the
API at request time, so a deployed API doesn't need the ``shap`` extra
installed at all; it just reads the stored ``top_factors`` JSON.

Complements (doesn't replace) ``explain/factors.py``'s hand-derived exact
linear decomposition for the logistic model -- that one is exact-but-only-
valid-for-a-linear-model math with no ``shap`` dependency; this one is the
real library, valid for gbm_v1's tree ensemble.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mlbsim_models.direct import GBMWinModel

Array = NDArray[np.float64]

_LABELS: dict[str, str] = {
    "d_win_pct": "season win rate",
    "d_run_diff_pg": "run differential per game",
    "d_win_pct_l10": "last-10-games win rate",
    "d_run_diff_pg_l10": "last-10-games run differential",
    "d_rest_days": "rest days",
    "d_elo": "Elo rating",
    "d_sp_adj": "starting pitcher (FIP)",
}


def gbm_shap_factors_batch(
    gbm_model: GBMWinModel, x: Any, *, top_n: int = 3
) -> list[list[dict[str, Any]]]:
    """Top-``top_n`` SHAP factors per row of ``x`` (n_rows x n_features)."""
    import shap  # lazy: heavy optional dep (the `shap` extra), offline-only

    explainer = shap.TreeExplainer(gbm_model.clf)
    x_arr = np.asarray(x, dtype=np.float64)
    raw = np.asarray(explainer.shap_values(x_arr))
    # HistGradientBoostingClassifier binary classification: shap's returned shape
    # varies by version between (n, features) and (n, features, 2 classes).
    values = raw[..., 1] if raw.ndim == 3 else raw

    out: list[list[dict[str, Any]]] = []
    for row in values:
        order = np.argsort(-np.abs(row))[:top_n]
        factors = []
        for i in order:
            name = gbm_model.feature_names[int(i)]
            val = float(row[i])
            factors.append(
                {
                    "feature": name,
                    "label": _LABELS.get(name, name),
                    "logit_contribution": round(val, 5),
                    "favours": "home" if val >= 0 else "away",
                    "prob_shift": round(min(abs(val) / 4.0, 0.5), 5),
                }
            )
        out.append(factors)
    return out
