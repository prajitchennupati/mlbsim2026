from __future__ import annotations

from mlbsim_data.parse import parse_boxscore
from mlbsim_data.validate import BATTING_GAME_LOG_SCHEMA, validate_rows


def test_lineups_and_starters(game_boxscore):
    bp = parse_boxscore(game_boxscore, game_pk=744914, season=2024)
    assert bp.starters == {"home": 684320, "away": 686613}
    assert len(bp.lineups) == 18  # 9 per side
    slots = sorted(r["batting_order"] for r in bp.lineups if r["team_id"] == 141)
    assert slots == list(range(1, 10))


def test_batting_logs_validate_and_totals(game_boxscore):
    bp = parse_boxscore(game_boxscore, game_pk=744914, season=2024)
    validate_rows(bp.batting_logs, BATTING_GAME_LOG_SCHEMA)
    bichette = next(r for r in bp.batting_logs if r["player_id"] == 666182)
    assert bichette["ab"] == 3
    assert bichette["bb"] == 1
    assert bichette["so"] == 1
    assert bichette["pa"] == 4


def test_pitching_logs(game_boxscore):
    bp = parse_boxscore(game_boxscore, game_pk=744914, season=2024)
    starter = next(r for r in bp.pitching_logs if r["player_id"] == 684320)
    assert starter["is_start"] is True
    assert starter["outs"] == 20  # 6.2 IP
    assert starter["so"] == 6
    assert starter["er"] == 1
    assert starter["got_loss"] is True
    assert starter["quality_start"] is True


def test_team_logs(game_boxscore):
    bp = parse_boxscore(game_boxscore, game_pk=744914, season=2024)
    tor = next(r for r in bp.team_logs if r["team_id"] == 141)
    hou = next(r for r in bp.team_logs if r["team_id"] == 117)
    assert tor["runs_for"] == 1 and tor["runs_against"] == 3
    assert tor["won"] is False
    assert hou["won"] is True
    assert hou["is_home"] is False
