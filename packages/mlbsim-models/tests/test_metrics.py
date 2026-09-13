from __future__ import annotations

import math

import numpy as np
import pytest

from mlbsim_models.evaluate import metrics as me


def test_log_loss_matches_hand_value():
    y = [1, 0, 1, 0]
    p = [0.9, 0.1, 0.8, 0.3]
    expected = -np.mean([math.log(0.9), math.log(0.9), math.log(0.8), math.log(0.7)])
    assert me.log_loss(y, p) == pytest.approx(expected, abs=1e-9)


def test_perfect_and_worst_predictions():
    y = [1, 0, 1, 0]
    assert me.brier_score(y, y) == pytest.approx(0.0, abs=1e-12)  # ~0 up to prob clipping
    assert me.accuracy(y, y) == 1.0
    assert me.log_loss(y, [1 - v for v in y]) > 10  # clipped, very large


def test_roc_auc_known_cases():
    assert me.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert me.roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)
    assert me.roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)  # all ties
    assert math.isnan(me.roc_auc([1, 1, 1], [0.4, 0.5, 0.6]))  # one class


def test_calibration_and_ece():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)  # calibrated by construction
    ece = me.expected_calibration_error(y, p, n_bins=10)
    assert ece < 0.02
    table = me.calibration_table(y, p, n_bins=10)
    assert len(table) == 10
    for row in table:
        assert abs(row["mean_pred"] - row["mean_actual"]) < 0.05


def test_regression_metrics():
    y = [1, 2, 3, 4]
    yhat = [1, 2, 3, 5]
    assert me.mae(y, yhat) == pytest.approx(0.25)
    assert me.rmse(y, yhat) == pytest.approx(0.5)
    assert me.poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-9)
    assert me.poisson_deviance(y, yhat) > 0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        me.log_loss([1, 0], [0.5])


def test_classification_report_bundle():
    y = [1, 0, 1, 1, 0, 0, 1, 0]
    p = [0.7, 0.2, 0.6, 0.9, 0.3, 0.4, 0.8, 0.1]
    rep = me.classification_report(y, p)
    assert rep["n"] == 8
    assert rep["base_rate"] == 0.5
    assert 0.0 <= rep["roc_auc"] <= 1.0
    assert rep["log_loss"] < me.log_loss(y, [0.5] * 8)  # better than coin flip
