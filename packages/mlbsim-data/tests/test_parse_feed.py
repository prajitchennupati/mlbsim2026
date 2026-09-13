from __future__ import annotations

from mlbsim_data.parse import parse_boxscore, parse_game_feed
from mlbsim_data.validate import PLATE_APPEARANCE_SCHEMA, validate_rows


def _parse(feed, box):
    bp = parse_boxscore(box, game_pk=744914, season=2024)
    return parse_game_feed(feed, batting_order_by_player=bp.batting_order_by_player())


def test_game_row(game_feed, game_boxscore):
    fp = _parse(game_feed, game_boxscore)
    g = fp.game
    assert g["game_pk"] == 744914
    assert g["season"] == 2024
    assert (g["away_score"], g["home_score"]) == (3, 1)
    assert g["n_innings"] == 9
    assert g["winning_pitcher_id"] == 686613
    assert g["losing_pitcher_id"] == 684320
    assert g["save_pitcher_id"] == 623352


def test_park_and_teams(game_feed, game_boxscore):
    fp = _parse(game_feed, game_boxscore)
    assert fp.park["park_id"] == 14
    assert fp.park["roof_type"] == "retractable"
    assert fp.park["altitude_ft"] == 270
    leagues = {t["team_id"]: (t["league"], t["division"]) for t in fp.teams}
    assert leagues[117] == ("AL", "W")  # Astros
    assert leagues[141] == ("AL", "E")  # Blue Jays


def test_plate_appearances_validate(game_feed, game_boxscore):
    fp = _parse(game_feed, game_boxscore)
    validate_rows(fp.plate_appearances, PLATE_APPEARANCE_SCHEMA)
    assert len(fp.plate_appearances) >= 30


def test_base_out_state_reconstruction(game_feed, game_boxscore):
    """Contiguous 2nd-inning bottom half: single, walk, groundout, walk."""
    fp = _parse(game_feed, game_boxscore)
    by_idx = {pa["at_bat_index"]: pa for pa in fp.plate_appearances}

    single = by_idx[9]
    assert (single["event_type"], single["outs_start"]) == ("single", 0)
    assert single["base_state_start"] == 0 and single["base_state_end"] == 1  # runner to 1B

    walk = by_idx[10]
    assert walk["base_state_start"] == 1 and walk["base_state_end"] == 3  # 1B+2B

    loaded_walk = by_idx[12]
    assert loaded_walk["base_state_end"] == 7  # bases loaded

    # first batter of the game: nobody on, 0 out -> 1 out
    lead = by_idx[0]
    assert (lead["outs_start"], lead["outs_end"]) == (0, 1)
    assert lead["base_state_start"] == 0


def test_home_run_row(game_feed, game_boxscore):
    fp = _parse(game_feed, game_boxscore)
    hr = next(pa for pa in fp.plate_appearances if pa["event_type"] == "home_run")
    assert hr["rbi"] == 1
    assert hr["runs_on_play"] == 1
    assert hr["away_score_after"] == 1
    assert hr["base_state_end"] == 0


def test_pitch_and_batted_ball_rows(game_feed, game_boxscore):
    fp = _parse(game_feed, game_boxscore)
    assert len(fp.pitches) > 100
    p0 = fp.pitches[0]
    assert p0["pitch_id"] == p0["pa_id"] * 100 + p0["seq"]
    assert p0["pitch_type"] == "FF"
    assert p0["release_speed"] == 95.9
    assert all(
        bb["pa_id"] in {pa["pa_id"] for pa in fp.plate_appearances} for bb in fp.batted_balls
    )


def test_times_through_order_increments(game_feed, game_boxscore):
    fp = _parse(game_feed, game_boxscore)
    tto_by_idx = {pa["at_bat_index"]: pa["times_through_order"] for pa in fp.plate_appearances}
    assert tto_by_idx[0] == 1  # leadoff, first time
    # by the top of the 6th the Astros leadoff man faces the pitcher a 2nd time
    second_times = [pa for pa in fp.plate_appearances if pa["times_through_order"] == 2]
    assert second_times, "expected at least one 2nd-time-through PA"
