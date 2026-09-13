from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from mlbsim_models.pipelines.live import state_from_feed

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "mlbsim-data"
    / "tests"
    / "fixtures"
    / "statsapi"
    / "game_744914_feed.json"
)


@pytest.fixture
def final_feed() -> dict:
    return json.loads(_FIXTURE.read_text())


def test_final_game_yields_no_live_state(final_feed: dict):
    assert state_from_feed(final_feed) is None


def test_live_feed_is_parsed_into_a_game_state(final_feed: dict):
    feed = copy.deepcopy(final_feed)
    feed["gameData"]["status"]["abstractGameState"] = "Live"
    ls = feed["liveData"]["linescore"]
    ls["currentInning"] = 7
    ls["isTopInning"] = True
    ls["outs"] = 2
    ls["offense"] = {"first": {"id": 1}, "third": {"id": 2}, "battingOrder": 4}
    ls["teams"] = {"home": {"runs": 2}, "away": {"runs": 5}}

    st = state_from_feed(feed)
    assert st is not None
    assert (st.inning, st.half, st.outs) == (7, "top", 2)
    assert st.base_state == 0b101  # first + third
    assert (st.home_score, st.away_score) == (2, 5)
    assert st.bo_away == 3  # battingOrder 4 -> index 3, away bats in the top half
    assert st.bo_home == 0
    # inning 7 >= 6 -> both starters are assumed out
    assert st.home_sp_out and st.away_sp_out


def test_early_inning_keeps_the_listed_starter(final_feed: dict):
    feed = copy.deepcopy(final_feed)
    feed["gameData"]["status"]["abstractGameState"] = "Live"
    feed["gameData"]["probablePitchers"] = {"home": {"id": 100}, "away": {"id": 200}}
    ls = feed["liveData"]["linescore"]
    ls.update(currentInning=3, isTopInning=False, outs=1)
    ls["offense"] = {"battingOrder": 1}
    ls["teams"] = {"home": {"runs": 0}, "away": {"runs": 1}}
    # current pitcher is still the listed away starter
    feed["liveData"]["plays"]["currentPlay"]["matchup"]["pitcher"] = {"id": 200}

    st = state_from_feed(feed)
    assert st is not None
    assert st.away_sp_out is False  # current pitcher == probable away starter, inning < 6
