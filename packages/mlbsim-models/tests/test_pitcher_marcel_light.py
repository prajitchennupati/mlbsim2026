from __future__ import annotations

from mlbsim_models.projections.pitcher_marcel_light import SeasonLine, project_pitcher


def _line(season: int, *, outs=300, hr=15, bb=40, hbp=5, so=150, bf=500) -> SeasonLine:
    return SeasonLine(
        season=season,
        outs=outs,
        home_runs=hr,
        walks=bb,
        hit_by_pitch=hbp,
        strikeouts=so,
        batters_faced=bf,
    )


def test_ace_projects_better_than_league_average():
    ace = [_line(2024, so=220, hr=8, bb=25), _line(2025, so=230, hr=7, bb=22)]
    proj = project_pitcher(ace)
    league_neutral = project_pitcher([])
    assert proj.proj_era_equiv < league_neutral.proj_era_equiv
    assert proj.proj_so_per_start > league_neutral.proj_so_per_start


def test_empty_history_is_neutral_not_a_crash():
    proj = project_pitcher([])
    assert proj.n_seasons_used == 0
    assert proj.proj_ip_per_start > 0
    assert 0 <= proj.p_6plus_k <= 1


def test_uses_at_most_three_most_recent_seasons():
    seasons = [_line(y) for y in range(2018, 2026)]
    proj = project_pitcher(seasons)
    assert proj.n_seasons_used == 3


def test_more_history_gives_a_less_extreme_projection_than_one_hot_season():
    hot_start = [_line(2026, outs=30, hr=0, bb=2, so=20, bf=45)]  # tiny, unsustainable sample
    proj_thin = project_pitcher(hot_start)
    proj_none = project_pitcher([])
    # a 20-K half-game shouldn't swing the full-start projection all the way there
    assert proj_thin.proj_so_per_start < 15
    assert proj_thin.proj_so_per_start > proj_none.proj_so_per_start
