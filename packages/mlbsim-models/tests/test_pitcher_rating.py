from __future__ import annotations

from mlbsim_models.ratings.pitcher import (
    LEAGUE_AVG_FIP,
    MAX_ADJUSTMENT,
    MIN_BATTERS_FACED,
    PitcherLine,
    fip,
    rating_adjustment,
)


def _line(*, outs=90, hr=3, bb=12, hbp=1, so=30, bf=95) -> PitcherLine:
    return PitcherLine(
        outs=outs, home_runs=hr, walks=bb, hit_by_pitch=hbp, strikeouts=so, batters_faced=bf
    )


def test_fip_matches_hand_computation():
    # 30 IP, 3 HR, 12 BB, 1 HBP, 30 K, constant 3.10
    line = _line()
    expected = (13 * 3 + 3 * (12 + 1) - 2 * 30) / 30.0 + 3.10
    assert abs(fip(line) - expected) < 1e-9


def test_ace_gets_a_positive_adjustment():
    # Lots of strikeouts, few walks/homers -> well below league-average FIP.
    ace = _line(outs=90, hr=1, bb=6, hbp=0, so=45, bf=95)
    assert fip(ace) < LEAGUE_AVG_FIP
    assert rating_adjustment(ace, points_per_fip_run=12.0) > 0


def test_bad_starter_gets_a_negative_adjustment():
    scrub = _line(outs=90, hr=8, bb=20, hbp=2, so=15, bf=95)
    assert fip(scrub) > LEAGUE_AVG_FIP
    assert rating_adjustment(scrub, points_per_fip_run=12.0) < 0


def test_no_data_is_neutral():
    assert rating_adjustment(None, points_per_fip_run=12.0) == 0.0


def test_too_few_batters_faced_is_neutral():
    thin = _line(outs=6, hr=0, bb=0, hbp=0, so=3, bf=MIN_BATTERS_FACED - 1)
    assert rating_adjustment(thin, points_per_fip_run=12.0) == 0.0


def test_adjustment_is_capped():
    extreme = _line(outs=180, hr=0, bb=0, hbp=0, so=90, bf=190)
    assert rating_adjustment(extreme, points_per_fip_run=1000.0) == MAX_ADJUSTMENT
