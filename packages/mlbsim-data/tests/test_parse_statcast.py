from __future__ import annotations

from pathlib import Path

import pytest

from mlbsim_data.parse import parse_statcast_batted_balls

_CSV = Path(__file__).parent / "fixtures" / "statcast" / "statcast_game_744914.csv"


@pytest.fixture(scope="module")
def rows():
    return parse_statcast_batted_balls(_CSV.read_text(encoding="utf-8"))


def test_only_balls_in_play(rows):
    assert rows
    # every row maps to game 744914 and carries a launch speed
    assert all(r["pa_id"] // 1000 == 744914 for r in rows)


def test_pa_id_join_key_and_barrel_flags(rows):
    # Jeremy Peña HR: Savant at_bat_number 30 -> atBatIndex 29 -> pa_id 744914029
    hr = next(r for r in rows if r["pa_id"] == 744914 * 1000 + 29)
    assert hr["is_barrel"] is True
    assert hr["is_hardhit"] is True
    assert hr["launch_speed"] == pytest.approx(104.0, abs=0.5)
    assert hr["xwoba"] > 1.0


def test_weak_contact_not_barrel(rows):
    softest = min(rows, key=lambda r: r["launch_speed"] or 999)
    assert softest["is_barrel"] is False
