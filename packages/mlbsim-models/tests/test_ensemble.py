from __future__ import annotations

import numpy as np
import pytest

from mlbsim_models.ensemble import (
    IsotonicCalibrator,
    PlattCalibrator,
    StackedWinModel,
    fit_calibrator,
    load_calibrator,
)
from mlbsim_models.evaluate.metrics import expected_calibration_error, log_loss


def _synth(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.15, 0.85, n)
    y = (rng.uniform(size=n) < truth).astype(int)
    good = np.clip(truth + rng.normal(0, 0.05, n), 0.01, 0.99)  # sharp, ~unbiased
    ok = np.clip(0.5 + 0.6 * (truth - 0.5) + rng.normal(0, 0.08, n), 0.01, 0.99)  # shrunk
    biased = np.clip(truth + 0.12, 0.01, 0.99)  # systematically high
    return np.column_stack([good, ok, biased]), y, truth


def test_stack_tracks_the_best_base_and_beats_the_worst():
    x, y, _ = _synth(n=12000, seed=1)
    cut = 8000
    m = StackedWinModel(["good", "ok", "biased"]).fit(x[:cut], y[:cut])
    base_ll = [log_loss(y[cut:], x[cut:, j]) for j in range(3)]
    stack_ll = log_loss(y[cut:], m.predict_proba(x[cut:]))
    assert stack_ll <= min(base_ll) + 0.01  # never much worse than the best base
    assert stack_ll < max(base_ll)  # clearly better than the worst base
    w = m.weights()
    assert set(w) == {"intercept", "good", "ok", "biased"}
    # the meta-learner de-biases the systematically-high base via a negative intercept
    assert w["intercept"] < 0.0


def test_stack_roundtrip(tmp_path):
    x, y, _ = _synth(seed=2)
    m = StackedWinModel(["a", "b", "c"]).fit(x, y)
    p = tmp_path / "s.joblib"
    m.save(p)
    np.testing.assert_allclose(m.predict_proba(x), StackedWinModel.load(p).predict_proba(x))


def test_isotonic_reduces_calibration_error_and_is_monotone():
    x, y, _ = _synth(seed=3)
    raw = x[:, 2]  # the biased-high base model
    cut = 4000
    cal = IsotonicCalibrator().fit(raw[:cut], y[:cut])
    out = cal.transform(raw[cut:])
    assert expected_calibration_error(y[cut:], out) < expected_calibration_error(y[cut:], raw[cut:])
    grid = np.linspace(0.02, 0.98, 50)
    mono = cal.transform(grid)
    assert np.all(np.diff(mono) >= -1e-9)


def test_platt_calibrator_shifts_toward_truth():
    x, y, _ = _synth(seed=4)
    raw = x[:, 2]
    cut = 4000
    out = PlattCalibrator().fit(raw[:cut], y[:cut]).transform(raw[cut:])
    assert abs(out.mean() - y[cut:].mean()) < abs(raw[cut:].mean() - y[cut:].mean())


def test_stack_shape_validation():
    m = StackedWinModel(["a", "b"])
    with pytest.raises(ValueError, match="expected"):
        m.fit(np.zeros((10, 3)), np.zeros(10))


def test_fit_calibrator_picks_platt_when_small_isotonic_when_large():
    x, y, _ = _synth(n=6000, seed=7)
    raw = x[:, 2]
    small = fit_calibrator(raw[:300], y[:300], min_isotonic=1000)
    large = fit_calibrator(raw, y, min_isotonic=1000)
    assert isinstance(small, PlattCalibrator)
    assert isinstance(large, IsotonicCalibrator)


def test_load_calibrator_dispatches_on_stored_kind(tmp_path):
    x, y, _ = _synth(seed=8)
    for cal in (PlattCalibrator().fit(x[:, 0], y), IsotonicCalibrator().fit(x[:, 0], y)):
        p = tmp_path / "c.joblib"
        cal.save(p)
        back = load_calibrator(p)
        assert type(back) is type(cal)
        np.testing.assert_allclose(back.transform(x[:20, 0]), cal.transform(x[:20, 0]))
