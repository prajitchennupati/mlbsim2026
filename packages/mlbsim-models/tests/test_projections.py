from __future__ import annotations

import numpy as np
import pytest

from mlbsim_engine.outcomes import HR, LEAGUE_RATES, K
from mlbsim_models.projections.marcel import age_factor, project_marcel
from mlbsim_models.projections.shrinkage import shrink_rates


def _season(pa: int, *, hr: int, so: int, h: int) -> dict:
    return {"pa": pa, "h": h, "hr": hr, "d2b": 28, "t3b": 2, "bb": 60, "hbp": 5, "so": so}


def test_marcel_reproduces_a_stable_hitter():
    line = _season(650, hr=32, so=140, h=170)
    proj = project_marcel([line, line, line], age=28)
    # projected HR rate close to observed 32/650, lightly regressed
    assert proj.rates[HR] == pytest.approx(32 / 650, rel=0.20)
    assert proj.rates.sum() == pytest.approx(1.0)
    assert 500 < proj.proj_pa < 620  # 200 + 0.5*650 + 0.1*650


def test_marcel_regresses_small_samples_toward_league():
    tiny = _season(40, hr=6, so=5, h=18)  # absurd 60-HR pace on 40 PA
    proj = project_marcel([tiny, None, None], age=27)
    assert proj.rates[HR] < 6 / 40  # pulled way down
    assert proj.rates[HR] > LEAGUE_RATES[HR]  # but still above league


def test_marcel_age_curve():
    assert age_factor(24) > 1.0
    assert age_factor(29) == pytest.approx(1.0)
    assert age_factor(36) < 1.0
    line = _season(600, hr=25, so=130, h=160)
    young = project_marcel([line, line, line], age=24)
    old = project_marcel([line, line, line], age=36)
    assert young.rates[HR] > old.rates[HR]
    assert young.rates[K] < old.rates[K]  # Ks drift up with age


def test_marcel_empty_history_is_league():
    proj = project_marcel([None, None, None])
    np.testing.assert_allclose(proj.rates, LEAGUE_RATES, atol=1e-9)
    assert proj.proj_pa == pytest.approx(200.0)


def test_shrinkage_pulls_toward_population_and_normalises():
    rng = np.random.default_rng(0)
    population = rng.dirichlet(np.ones(8) * 40, size=200)  # tight cluster
    extreme = population.mean(axis=0).copy()
    extreme[HR] *= 4
    extreme /= extreme.sum()

    barely = shrink_rates(extreme, player_n=20, population=population)
    lots = shrink_rates(extreme, player_n=2000, population=population)
    assert barely.sum() == pytest.approx(1.0)
    # small sample -> closer to population mean than the raw extreme
    assert abs(barely[HR] - population.mean(axis=0)[HR]) < abs(
        extreme[HR] - population.mean(axis=0)[HR]
    )
    # large sample -> stays near the player's own rate
    assert abs(lots[HR] - extreme[HR]) < abs(barely[HR] - extreme[HR])


def test_shrinkage_zero_sample_is_population_mean():
    population = np.random.default_rng(1).dirichlet(np.ones(8) * 30, size=100)
    out = shrink_rates(population[0], player_n=0.0, population=population)
    np.testing.assert_allclose(out, population.mean(axis=0), atol=1e-9)
