from __future__ import annotations

from mlbsim_models.ml.team_form import build_tracker, run_team_form


def _games() -> list[dict]:
    # team 1 always beats team 2, 5-1, on consecutive days.
    return [
        {
            "game_pk": 100 + i,
            "game_date": f"2024-04-{i + 1:02d}",
            "home_team_id": 1 if i % 2 == 0 else 2,
            "away_team_id": 2 if i % 2 == 0 else 1,
            "home_score": 5 if i % 2 == 0 else 1,
            "away_score": 1 if i % 2 == 0 else 5,
        }
        for i in range(12)
    ]


def test_first_game_is_neutral():
    rows = run_team_form(_games())
    assert rows[0].home_form["win_pct"] == 0.5
    assert rows[0].home_form["games_played"] == 0.0


def test_form_never_sees_its_own_game_result():
    rows = run_team_form(_games())
    # Both teams play every game here, so by the 5th game (index 4) each side
    # has exactly 4 strictly-prior games counted -- never this one.
    row = rows[4]
    assert row.home_form["games_played"] == 4.0
    assert row.away_form["games_played"] == 4.0


def test_dominant_team_accumulates_a_winning_form():
    rows = run_team_form(_games())
    last = rows[-1]
    # by the last game, team 1's form (whichever side it's on) should show a
    # strong positive run differential and a team 2 side should be negative.
    forms = [last.home_form, last.away_form]
    assert any(f["run_diff_pg"] > 2.0 for f in forms)
    assert any(f["run_diff_pg"] < -2.0 for f in forms)


def test_build_tracker_reflects_final_state():
    tracker = build_tracker(_games())
    snap1 = tracker.snapshot(1, __import__("datetime").date(2024, 5, 1))
    assert snap1["games_played"] == 12
    assert snap1["win_pct"] == 1.0


def test_missing_score_games_are_skipped():
    games = [
        *_games(),
        {
            "game_pk": 999,
            "game_date": "2024-04-13",
            "home_team_id": 1,
            "away_team_id": 2,
            "home_score": None,
            "away_score": None,
        },
    ]
    rows = run_team_form(games)
    assert len(rows) == 12  # the unplayed game is skipped, not crashed on
