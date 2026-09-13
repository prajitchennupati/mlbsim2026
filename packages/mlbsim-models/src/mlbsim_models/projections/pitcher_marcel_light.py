"""A Marcel-style starting-pitcher projection: weighted recent seasons plus
regression to the league mean -- the same idea as the classic Marcel
projection system (and this project's own ``projections/marcel.py``), just
season-level rather than requiring per-game logs, since a starter's
multi-year history is one cheap Stats API call (``yearByYear``), not a
box-score backfill.

Deliberately scoped to starting pitchers, not full batting lineups: this
project doesn't ingest actual starting lineups yet (that needs a pre-game
lineup pull it doesn't do), but it does know who's on the mound
(``GameProbable``) for every game with a probable pitcher announced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_WEIGHTS = (5.0, 4.0, 3.0)  # most-recent-season first
_REGRESS_BF = 200.0  # battersFaced worth of league-average blended in
_TYPICAL_START_BF = 23.0  # ~5.2 IP at a typical BF/out rate

# Modern-era rough league averages, per batter faced.
_LEAGUE_RATE = {
    "so_rate": 0.225,
    "bb_rate": 0.082,
    "hbp_rate": 0.010,
    "hr_rate": 0.028,
    "outs_per_bf": 0.735,
}


@dataclass(frozen=True, slots=True)
class SeasonLine:
    season: int
    outs: int
    home_runs: int
    walks: int
    hit_by_pitch: int
    strikeouts: int
    batters_faced: int


@dataclass(frozen=True, slots=True)
class PitcherProjection:
    proj_ip_per_start: float
    proj_so_per_start: float
    proj_bb_per_start: float
    proj_hr_per_start: float
    proj_era_equiv: float  # FIP-based ERA-equivalent, not observed ER
    p_6plus_k: float
    p_6plus_ip: float
    n_seasons_used: int


def project_pitcher(seasons: list[SeasonLine]) -> PitcherProjection:
    """``seasons`` newest-last. Weighted-sum Marcel with a fixed regression
    pool of league-average battersFaced, then scaled to one typical start."""
    recent = sorted(seasons, key=lambda s: s.season)[-len(_WEIGHTS) :]
    # `_WEIGHTS[-len(recent):]` would be `_WEIGHTS[-0:]` == the *whole* tuple
    # when `recent` is empty (Python's -0 == 0) -- slice from the front instead.
    weights = _WEIGHTS[len(_WEIGHTS) - len(recent) :]

    def wsum(attr: str) -> float:
        return sum(w * float(getattr(s, attr)) for w, s in zip(weights, recent, strict=True))

    w_bf = wsum("batters_faced")
    w_so = wsum("strikeouts")
    w_bb = wsum("walks")
    w_hbp = wsum("hit_by_pitch")
    w_hr = wsum("home_runs")
    w_outs = wsum("outs")

    total_bf = w_bf + _REGRESS_BF

    def blended_rate(weighted_sum: float, rate_key: str) -> float:
        if not total_bf:
            return _LEAGUE_RATE[rate_key]
        return (weighted_sum + _REGRESS_BF * _LEAGUE_RATE[rate_key]) / total_bf

    so_rate = blended_rate(w_so, "so_rate")
    bb_rate = blended_rate(w_bb, "bb_rate")
    hr_rate = blended_rate(w_hr, "hr_rate")
    hbp_rate = blended_rate(w_hbp, "hbp_rate")
    outs_per_bf = blended_rate(w_outs, "outs_per_bf")

    bf_per_start = _TYPICAL_START_BF
    proj_ip = (bf_per_start * outs_per_bf) / 3
    proj_so = bf_per_start * so_rate
    proj_bb = bf_per_start * bb_rate
    proj_hr = bf_per_start * hr_rate
    proj_hbp = bf_per_start * hbp_rate

    fip = (13 * proj_hr + 3 * (proj_bb + proj_hbp) - 2 * proj_so) / max(proj_ip, 1e-9) + 3.10

    # P(6+ K) / P(6+ IP) this start, via a Poisson approx on the projected rate.
    def p_at_least(mu: float, k: int) -> float:
        pmf = math.exp(-mu)
        cdf = pmf
        for i in range(1, k):
            pmf *= mu / i
            cdf += pmf
        return max(0.0, 1.0 - cdf)

    return PitcherProjection(
        proj_ip_per_start=round(proj_ip, 2),
        proj_so_per_start=round(proj_so, 2),
        proj_bb_per_start=round(proj_bb, 2),
        proj_hr_per_start=round(proj_hr, 2),
        proj_era_equiv=round(max(fip, 0.0), 2),
        p_6plus_k=round(p_at_least(proj_so, 6), 4),
        p_6plus_ip=round(p_at_least(proj_ip, 6), 4),
        n_seasons_used=len(recent),
    )
