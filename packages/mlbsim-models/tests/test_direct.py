from __future__ import annotations

import numpy as np
import pytest

from mlbsim_models.direct import (
    GBMWinModel,
    RunsPoisson,
    WinLogit,
    derive_game_probs,
    score_grid,
)


def test_win_logit_learns_separable_signal():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=(n, 3))
    logit = 1.5 * x[:, 0] - 1.0 * x[:, 1]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    model = WinLogit(["a", "b", "c"]).fit(x, y)
    p = model.predict_proba(x)
    assert p.shape == (n,)
    assert ((p > 0.5).astype(int) == y).mean() > 0.7
    coef = model.coefficients()
    assert coef["a"] > 0 > coef["b"]
    assert abs(coef["c"]) < abs(coef["a"])  # noise feature shrinks


def test_win_logit_roundtrip(tmp_path):
    x = np.random.default_rng(1).normal(size=(300, 2))
    y = (x[:, 0] > 0).astype(int)
    m = WinLogit(["a", "b"]).fit(x, y)
    path = tmp_path / "m.joblib"
    m.save(path)
    loaded = WinLogit.load(path)
    np.testing.assert_allclose(m.predict_proba(x), loaded.predict_proba(x))


def test_gbm_learns_nonlinear_signal_and_roundtrips(tmp_path):
    rng = np.random.default_rng(2)
    n = 3000
    x = rng.normal(size=(n, 3))
    # an interaction the logistic model can't see: sign(x0 * x1)
    logit = 2.0 * np.sign(x[:, 0] * x[:, 1])
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    m = GBMWinModel(["a", "b", "c"]).fit(x, y)
    p = m.predict_proba(x)
    assert p.shape == (n,)
    assert ((p > 0.5).astype(int) == y).mean() > 0.75  # beats the linear model here

    with pytest.raises(ValueError, match="expected 3 features"):
        m.fit(x[:, :2], y)

    path = tmp_path / "g.joblib"
    m.save(path)
    np.testing.assert_allclose(m.predict_proba(x), GBMWinModel.load(path).predict_proba(x))


def test_score_grid_is_normalised_and_matches_poisson_means():
    grid = score_grid(4.8, 4.1)
    assert grid.sum() == pytest.approx(1.0, abs=1e-6)
    n = grid.shape[0]
    exp_home = (np.arange(n) * grid.sum(axis=1)).sum()
    exp_away = (np.arange(n) * grid.sum(axis=0)).sum()
    assert exp_home == pytest.approx(4.8, abs=0.05)
    assert exp_away == pytest.approx(4.1, abs=0.05)


def test_derive_game_probs_consistency():
    d = derive_game_probs(score_grid(5.0, 4.0))
    assert d["home_win_prob"] + d["away_win_prob"] == pytest.approx(1.0, abs=1e-6)
    assert d["home_win_prob"] > 0.5  # home scores more on average
    assert 0.0 < d["p_extra_innings"] < 0.15
    assert d["p_shutout_away"] > d["p_shutout_home"]  # weaker away offense
    assert d["exp_total_runs"] == pytest.approx(9.0, abs=0.1)
    probs = [float(v) for v in d["total_runs_dist"].values()]
    assert sum(probs) == pytest.approx(1.0, abs=1e-3)
    assert d["most_likely_scores"][0]["p"] >= d["most_likely_scores"][-1]["p"]


def test_runs_poisson_recovers_rate():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3000, 2))
    mu = np.exp(1.5 + 0.3 * x[:, 0])
    y = rng.poisson(mu)
    model = RunsPoisson(["a", "b"]).fit(x, y)
    pred = model.predict_mu(x)
    assert np.corrcoef(pred, mu)[0, 1] > 0.9
