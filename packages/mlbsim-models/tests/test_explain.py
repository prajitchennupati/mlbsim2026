from __future__ import annotations

import numpy as np
import pytest

from mlbsim_features.build import DIFF_FEATURE_ORDER
from mlbsim_models.direct import WinLogit
from mlbsim_models.explain import build_prompt, linear_shap, render_explanation, top_factors


@pytest.fixture(scope="module")
def fitted_logit():
    rng = np.random.default_rng(0)
    n = 3000
    names = list(DIFF_FEATURE_ORDER)
    x = rng.normal(0, 1, (n, len(names)))
    # d_win_pct (index 0 in sorted order) strongly favours home; d_sp_era negative -> home
    i_win = names.index("d_win_pct")
    i_era = names.index("d_sp_era")
    logit = 2.0 * x[:, i_win] - 1.5 * x[:, i_era]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return WinLogit(names).fit(x, y), names, i_win, i_era


def test_linear_shap_contributions_sum_to_prediction_logit(fitted_logit):
    model, names, _, _ = fitted_logit
    x_row = np.random.default_rng(1).normal(0, 1, len(names))
    contribs, base = linear_shap(model, x_row)
    total_logit = base + sum(contribs.values())
    p_from_shap = 1 / (1 + np.exp(-total_logit))
    p_model = float(model.predict_proba([x_row])[0])
    assert p_from_shap == pytest.approx(p_model, abs=1e-6)


def test_top_factors_direction_and_shift(fitted_logit):
    model, names, i_win, i_era = fitted_logit
    x_row = np.zeros(len(names))
    x_row[i_win] = 2.0  # home much better recent record
    x_row[i_era] = 2.0  # home starter has the HIGHER era -> favours away
    contribs, base = linear_shap(model, x_row)
    factors = top_factors(contribs, base, k=3)
    labels = {f["feature"]: f for f in factors}
    assert labels["d_win_pct"]["favours"] == "home"
    assert labels["d_sp_era"]["favours"] == "away"
    # removing a favourable factor lowers the home probability
    assert labels["d_win_pct"]["prob_shift"] > 0


def test_render_template_explanation_mentions_favourite_and_a_factor(fitted_logit):
    model, names, i_win, _ = fitted_logit
    x_row = np.zeros(len(names))
    x_row[i_win] = 2.5
    contribs, base = linear_shap(model, x_row)
    factors = top_factors(contribs, base, k=4)
    ctx = {
        "home": "LAD",
        "away": "SFG",
        "home_win_prob": float(model.predict_proba([x_row])[0]),
        "exp_home_runs": 5.1,
        "exp_away_runs": 3.8,
    }
    text = render_explanation(ctx, factors, backend="template")
    assert "LAD" in text
    assert "season win rate" in text
    assert "%" in text


def test_build_prompt_shape(fitted_logit):
    model, names, i_win, _ = fitted_logit
    x_row = np.zeros(len(names))
    x_row[i_win] = 1.0
    contribs, base = linear_shap(model, x_row)
    prompt = build_prompt(
        {"home": "H", "away": "A", "home_win_prob": 0.61}, top_factors(contribs, base)
    )
    assert set(prompt) == {"system", "user"}
    assert "61%" in prompt["user"]
    assert "invent" in prompt["system"].lower()


def test_render_unknown_backend_raises(fitted_logit):
    with pytest.raises(ValueError, match="unknown backend"):
        render_explanation({"home": "H", "away": "A", "home_win_prob": 0.5}, [], backend="mystery")
