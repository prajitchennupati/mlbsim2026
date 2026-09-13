from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

_FIXTURES = Path(__file__).parent / "fixtures" / "statsapi"
_GAME_PK = 744914


def _load(name: str) -> Any:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def game_pk() -> int:
    return _GAME_PK


@pytest.fixture(scope="session")
def schedule_games() -> list[dict[str, Any]]:
    payload = _load("schedule_2024-07-01.json")
    return [g for day in payload["dates"] for g in day["games"]]


@pytest.fixture(scope="session")
def game_feed() -> dict[str, Any]:
    return _load(f"game_{_GAME_PK}_feed.json")


@pytest.fixture(scope="session")
def game_boxscore() -> dict[str, Any]:
    return _load(f"game_{_GAME_PK}_boxscore.json")
