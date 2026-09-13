from __future__ import annotations

import datetime as dt

from mlbsim_data.parse import parse_schedule


def test_schedule_shapes(schedule_games):
    p = parse_schedule(schedule_games, source_ts=dt.datetime(2024, 6, 30, tzinfo=dt.UTC))
    assert len(p.games) == 3
    assert len(p.teams) == 6  # 3 games, distinct teams
    assert {g["game_pk"] for g in p.games} == {744914, 744840, 746535}


def test_game_row_fields(schedule_games):
    p = parse_schedule(schedule_games)
    g = next(g for g in p.games if g["game_pk"] == 744914)
    assert g["season"] == 2024
    assert g["game_date"] == dt.date(2024, 7, 1)
    assert g["game_type"] == "R"
    assert g["home_team_id"] == 141
    assert g["away_team_id"] == 117
    assert g["is_doubleheader"] is False
    assert g["dh_game_num"] == 1


def test_probables_and_weather(schedule_games):
    p = parse_schedule(schedule_games)
    probs = {(r["game_pk"], r["team_id"]): r["probable_pitcher_id"] for r in p.probables}
    assert probs[(744914, 117)] == 686613  # Hunter Brown
    assert probs[(744914, 141)] == 684320  # Yariel Rodríguez
    wx = next(w for w in p.weather if w["game_pk"] == 744914)
    assert wx["temp_f"] == 69
    assert wx["wind_mph"] == 8
    assert wx["condition"] == "Sunny"
