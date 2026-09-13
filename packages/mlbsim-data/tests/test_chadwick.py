from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from mlbsim_data.sources.chadwick import parse_people

_FIXTURE = Path(__file__).parent / "fixtures" / "chadwick_people_sample.csv"


@pytest.fixture(scope="module")
def records():
    return parse_people(_FIXTURE.read_text(encoding="utf-8"))


def test_keeps_only_mlb_players_with_mlbam_id(records):
    ids = {r.player_id for r in records}
    assert ids == {430911, 605141, 555555}  # drops the no-MLBAM and never-played rows


def test_full_record_is_reconciled(records):
    betts = next(r for r in records if r.player_id == 605141)
    assert betts.full_name == "Mookie Betts"
    assert betts.retro_id == "bettm001"
    assert betts.bbref_id == "bettsmo01"
    assert betts.fg_id == "13611"
    assert betts.birth_date == dt.date(1992, 10, 7)
    assert betts.mlb_debut_year == 2014
    assert betts.mlb_final_year == 2024


def test_missing_optional_ids_and_birthdate_become_none(records):
    partial = next(r for r in records if r.player_id == 555555)
    assert partial.bbref_id is None
    assert partial.fg_id is None
    assert partial.birth_date is None
    assert partial.retro_id == "part001"
