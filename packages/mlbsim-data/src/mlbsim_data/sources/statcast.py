"""Client for the Baseball Savant Statcast CSV export.

Savant caps a single query at ~25k pitches, so callers should chunk long date
ranges (a week is safe). A ``game_pk`` filter is available for per-game
enrichment during ``ingest_game``.
"""

from __future__ import annotations

from mlbsim_core import get_logger
from mlbsim_data.sources.http import fetch_text

_log = get_logger(__name__)

_CSV_URL = "https://baseballsavant.mlb.com/statcast_search/csv"
_WEEK = 7 * 24 * 3600


def fetch_statcast_csv(
    *,
    start_date: str,
    end_date: str | None = None,
    game_pk: int | None = None,
    max_age_seconds: float | None = None,
) -> str:
    """Return the raw Statcast ``details`` CSV for a date range (and optional game)."""
    end_date = end_date or start_date
    params = {
        "all": "true",
        "type": "details",
        "player_type": "batter",
        "game_date_gt": start_date,
        "game_date_lt": end_date,
        "min_pitches": "0",
        "min_results": "0",
        "sort_col": "pitches",
        "sort_order": "desc",
    }
    key = f"{start_date}_{end_date}"
    if game_pk is not None:
        params["game_pk"] = str(game_pk)
        key = f"game_{game_pk}"
    # Completed data is immutable; default to permanent landing-zone retention.
    text = fetch_text(
        _CSV_URL,
        namespace="statcast",
        key=key,
        suffix=".csv",
        params=params,
        max_age_seconds=max_age_seconds,
    )
    _log.info("statcast.fetch", start=start_date, end=end_date, game_pk=game_pk, bytes=len(text))
    return text
