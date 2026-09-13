from __future__ import annotations

import numpy as np
import pytest

from mlbsim_engine.outcomes import LEAGUE_RATES
from mlbsim_engine.rates import rates_from_stat_json
from mlbsim_models.evaluate import projection_error
from mlbsim_models.projections.marcel import project_marcel


def _line(pa, *, h, hr, so):
    return {"pa": pa, "h": h, "hr": hr, "d2b": 25, "t3b": 2, "bb": 55, "hbp": 4, "so": so}


def test_perfect_projection_has_zero_error():
    actuals = [_line(600, h=150, hr=25, so=130) for _ in range(10)]
    projected = [rates_from_stat_json(a) for a in actuals]
    err = projection_error(projected, actuals)
    assert err["overall_mae"] == pytest.approx(0.0, abs=1e-9)
    assert err["per_stat"]["hr"]["rmse"] == pytest.approx(0.0, abs=1e-9)


def test_league_projection_is_worse_than_marcel_on_a_stable_population():
    rng = np.random.default_rng(0)
    actuals, marcel_proj, league_proj = [], [], []
    for _ in range(60):
        pa = 600
        hr = int(rng.integers(10, 40))
        so = int(rng.integers(90, 190))
        h = int(rng.integers(120, 185))
        actual = _line(pa, h=h, hr=hr, so=so)
        prior = _line(
            pa,
            h=h + int(rng.integers(-12, 12)),
            hr=hr + int(rng.integers(-4, 4)),
            so=so + int(rng.integers(-15, 15)),
        )
        actuals.append(actual)
        marcel_proj.append(project_marcel([prior, prior, prior], age=28).rates)
        league_proj.append(LEAGUE_RATES)

    marcel_err = projection_error(marcel_proj, actuals)["overall_rmse"]
    league_err = projection_error(league_proj, actuals)["overall_rmse"]
    assert marcel_err < league_err  # Marcel beats "everyone is league-average"


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        projection_error([LEAGUE_RATES], [])
