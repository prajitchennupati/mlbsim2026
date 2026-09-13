from __future__ import annotations

from mlbsim_features.build import (
    DIFF_FEATURE_ORDER,
    assemble_game_features,
    feature_vector,
    pitcher_form,
    team_form,
)


def _team_games():
    # 6 prior games: 4 wins, RS totals 30, RA totals 18
    return [
        {"game_date": "2024-04-01", "runs_for": 6, "runs_against": 2, "won": True},
        {"game_date": "2024-04-02", "runs_for": 3, "runs_against": 4, "won": False},
        {"game_date": "2024-04-03", "runs_for": 7, "runs_against": 1, "won": True},
        {"game_date": "2024-04-05", "runs_for": 4, "runs_against": 5, "won": False},
        {"game_date": "2024-04-06", "runs_for": 5, "runs_against": 3, "won": True},
        {"game_date": "2024-04-07", "runs_for": 5, "runs_against": 3, "won": True},
    ]


def test_team_form_basic_rates():
    f = team_form(_team_games(), "2024-04-09")
    assert f["games_played"] == 6
    assert f["win_pct"] == 4 / 6
    assert f["rs_pg"] == 30 / 6
    assert f["ra_pg"] == 18 / 6
    assert f["run_diff_pg"] == (30 - 18) / 6
    assert f["rest_days"] == 2  # 04-07 -> 04-09


def test_team_form_excludes_as_of_day_and_future():
    all_but_last = team_form(_team_games(), "2024-04-07")  # excludes 04-07
    assert all_but_last["games_played"] == 5


def test_team_form_neutral_when_empty():
    f = team_form([], "2024-04-01")
    assert f["win_pct"] == 0.5 and f["games_played"] == 0.0


def test_pitcher_form_era_and_k_pct():
    starts = [
        {
            "game_date": "2024-04-02",
            "outs": 18,
            "er": 2,
            "so": 7,
            "bb": 1,
            "hbp": 0,
            "hr": 1,
            "bf": 24,
            "is_start": True,
        },
        {
            "game_date": "2024-04-08",
            "outs": 15,
            "er": 4,
            "so": 5,
            "bb": 3,
            "hbp": 1,
            "hr": 2,
            "bf": 22,
            "is_start": True,
        },
        {
            "game_date": "2024-04-10",
            "outs": 3,
            "er": 1,
            "so": 1,
            "bb": 0,
            "hbp": 0,
            "hr": 0,
            "bf": 5,
            "is_start": False,
        },  # relief appearance -> ignored
    ]
    f = pitcher_form(starts, "2024-04-14")
    assert f["sp_starts"] == 2.0
    ip = 33 / 3
    assert f["sp_era"] == 9.0 * 6 / ip
    assert f["sp_k_pct"] == 12 / 46


def test_assemble_and_vector_alignment():
    tf = team_form(_team_games(), "2024-04-09")
    weak = team_form([], "2024-04-09")
    pf = pitcher_form([], "2024-04-09")
    feats = assemble_game_features(tf, weak, pf, pf)
    assert set(feats) == {"home", "away", "diff"}
    assert feats["diff"]["d_win_pct"] == round(tf["win_pct"] - 0.5, 5)
    vec = feature_vector(feats["diff"])
    assert len(vec) == len(DIFF_FEATURE_ORDER)
    assert all(isinstance(v, float) for v in vec)
