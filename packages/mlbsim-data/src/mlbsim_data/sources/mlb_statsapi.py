"""Client for the MLB Stats API (``statsapi.mlb.com``).

Covers the endpoints M1 needs: the schedule, the per-game live feed (full
play-by-play), and the boxscore. Completed ("Final") games are immutable, so
their landing-zone copies never expire; anything else gets a short TTL.
"""

from __future__ import annotations

from typing import Any, cast

from mlbsim_core import get_logger, get_settings
from mlbsim_data.sources.http import fetch_json

_log = get_logger(__name__)

_LIVE_TTL = 60.0  # seconds, for in-progress games
_SCHEDULE_HYDRATE = "probablePitcher,linescore,venue,weather,team"


def _base() -> str:
    return get_settings().mlb_statsapi_base.rstrip("/")


def schedule(
    start_date: str,
    end_date: str | None = None,
    *,
    sport_id: int = 1,
    game_types: str = "R,F,D,L,W",
    max_age_seconds: float | None = _LIVE_TTL,
) -> list[dict[str, Any]]:
    """Return the flat list of game objects for the date range (inclusive, ``YYYY-MM-DD``)."""
    end_date = end_date or start_date
    payload = fetch_json(
        f"{_base()}/v1/schedule",
        namespace="statsapi/schedule",
        key=f"{start_date}_{end_date}",
        params={
            "sportId": str(sport_id),
            "startDate": start_date,
            "endDate": end_date,
            "gameType": game_types,
            "hydrate": _SCHEDULE_HYDRATE,
        },
        max_age_seconds=max_age_seconds,
    )
    games: list[dict[str, Any]] = []
    for day in payload.get("dates", []):
        games.extend(day.get("games", []))
    _log.info("statsapi.schedule", start=start_date, end=end_date, n_games=len(games))
    return games


def _game_is_final(obj: dict[str, Any]) -> bool:
    state = (
        obj.get("gameData", {}).get("status", {}).get("abstractGameState")
        or obj.get("status", {}).get("abstractGameState")
        or ""
    )
    return state == "Final"


def game_feed(game_pk: int, *, assume_final: bool = False) -> dict[str, Any]:
    """Return the full ``feed/live`` document for one game."""
    # First fetch with a short TTL; if it turns out the game is Final, re-request
    # with no expiry so the immutable copy is retained.
    obj = fetch_json(
        f"{_base()}/v1.1/game/{game_pk}/feed/live",
        namespace="statsapi/feed",
        key=str(game_pk),
        max_age_seconds=None if assume_final else _LIVE_TTL,
    )
    if not assume_final and _game_is_final(obj):
        obj = fetch_json(
            f"{_base()}/v1.1/game/{game_pk}/feed/live",
            namespace="statsapi/feed",
            key=str(game_pk),
            max_age_seconds=None,
        )
    return cast("dict[str, Any]", obj)


def boxscore(game_pk: int, *, assume_final: bool = False) -> dict[str, Any]:
    """Return the boxscore document for one game."""
    return cast(
        "dict[str, Any]",
        fetch_json(
            f"{_base()}/v1/game/{game_pk}/boxscore",
            namespace="statsapi/boxscore",
            key=str(game_pk),
            max_age_seconds=None if assume_final else _LIVE_TTL,
        ),
    )
