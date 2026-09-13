"""Ingestion orchestration: fetch -> parse -> validate -> load (FK-safe order)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from mlbsim_core import get_logger, session_scope
from mlbsim_data.loaders import warehouse as wh
from mlbsim_data.parse import (
    parse_boxscore,
    parse_game_feed,
    parse_schedule,
    parse_statcast_batted_balls,
)
from mlbsim_data.sources import mlb_statsapi, statcast
from mlbsim_data.validate import (
    BATTING_GAME_LOG_SCHEMA,
    GAME_SCHEMA,
    PLATE_APPEARANCE_SCHEMA,
    validate_rows,
)

_log = get_logger(__name__)

_FINAL_STATES = {"Final", "Game Over", "Completed Early"}


@dataclass(slots=True)
class GameIngestSummary:
    game_pk: int
    status: str
    plate_appearances: int = 0
    pitches: int = 0
    batting_logs: int = 0
    pitching_logs: int = 0
    skipped: bool = False


@dataclass(slots=True)
class RangeIngestSummary:
    start: str
    end: str
    games_seen: int = 0
    games_loaded: int = 0
    game_pks: list[int] = field(default_factory=list)


def ingest_schedule_range(start_date: str, end_date: str | None = None) -> RangeIngestSummary:
    """Load ``parks`` / ``teams`` / ``games`` / ``game_weather`` / ``game_probables``."""
    end_date = end_date or start_date
    games = mlb_statsapi.schedule(start_date, end_date)
    parsed = parse_schedule(games)
    validate_rows(parsed.games, GAME_SCHEMA)

    with session_scope() as s:
        wh.load_parks(s, parsed.parks)  # teams + games FK-reference parks
        wh.load_teams(s, parsed.teams)
        wh.load_players(s, parsed.players)  # game_probables FK-references players
        loaded = wh.load_games(s, parsed.games)
        wh.load_game_weather(s, parsed.weather)
        wh.load_game_probables(s, parsed.probables)

    summary = RangeIngestSummary(
        start=start_date,
        end=end_date,
        games_seen=len(games),
        games_loaded=loaded,
        game_pks=[g["game_pk"] for g in parsed.games],
    )
    _log.info(
        "ingest.schedule", **{k: getattr(summary, k) for k in ("start", "end", "games_loaded")}
    )
    return summary


def ingest_game(
    game_pk: int, *, assume_final: bool = False, require_final: bool = False
) -> GameIngestSummary:
    """Fetch one game's feed + boxscore, parse, validate, and load everything."""
    feed = mlb_statsapi.game_feed(game_pk, assume_final=assume_final)
    status = feed["gameData"].get("status", {}).get("detailedState", "Unknown")
    season = int(feed["gameData"].get("game", {}).get("season") or dt.date.today().year)

    if require_final and status not in _FINAL_STATES:
        _log.info("ingest.game.skip_not_final", game_pk=game_pk, status=status)
        return GameIngestSummary(game_pk=game_pk, status=status, skipped=True)

    box = mlb_statsapi.boxscore(game_pk, assume_final=assume_final)
    box_parsed = parse_boxscore(box, game_pk=game_pk, season=season)
    feed_parsed = parse_game_feed(
        feed, batting_order_by_player=box_parsed.batting_order_by_player()
    )

    game_row = feed_parsed.game
    game_row["home_sp_id"] = box_parsed.starters.get("home")
    game_row["away_sp_id"] = box_parsed.starters.get("away")

    validate_rows([game_row], GAME_SCHEMA)
    validate_rows(feed_parsed.plate_appearances, PLATE_APPEARANCE_SCHEMA)
    validate_rows(box_parsed.batting_logs, BATTING_GAME_LOG_SCHEMA)

    with session_scope() as s:
        if feed_parsed.parks:
            wh.load_parks(s, feed_parsed.parks)
        wh.load_teams(s, feed_parsed.teams)
        wh.load_players(s, feed_parsed.players)
        wh.load_games(s, [game_row])
        if feed_parsed.weather:
            wh.load_game_weather(s, [feed_parsed.weather])
        wh.load_lineups(s, box_parsed.lineups)
        wh.load_plate_appearances(s, feed_parsed.plate_appearances)
        wh.load_pitches(s, feed_parsed.pitches)
        wh.load_batted_balls(s, feed_parsed.batted_balls)
        wh.load_batting_logs(s, box_parsed.batting_logs)
        wh.load_pitching_logs(s, box_parsed.pitching_logs)
        wh.load_team_logs(s, box_parsed.team_logs)

    summary = GameIngestSummary(
        game_pk=game_pk,
        status=status,
        plate_appearances=len(feed_parsed.plate_appearances),
        pitches=len(feed_parsed.pitches),
        batting_logs=len(box_parsed.batting_logs),
        pitching_logs=len(box_parsed.pitching_logs),
    )
    _log.info(
        "ingest.game",
        game_pk=game_pk,
        status=status,
        pa=summary.plate_appearances,
        pitches=summary.pitches,
    )
    return summary


def ingest_statcast_range(start_date: str, end_date: str, *, chunk_days: int = 7) -> int:
    """Enrich ``batted_balls`` with Statcast expected stats over a date range."""
    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)
    total = 0
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + dt.timedelta(days=chunk_days - 1), end)
        csv_text = statcast.fetch_statcast_csv(
            start_date=cursor.isoformat(), end_date=chunk_end.isoformat()
        )
        rows = parse_statcast_batted_balls(csv_text)
        if rows:
            with session_scope() as s:
                total += wh.load_batted_balls(s, rows)
        cursor = chunk_end + dt.timedelta(days=1)
    _log.info("ingest.statcast", start=start_date, end=end_date, batted_balls=total)
    return total


def ingest_season(
    year: int, *, only_final: bool = True, with_statcast: bool = True
) -> RangeIngestSummary:
    """Ingest an entire regular + postseason: schedule, each game's detail, then Statcast."""
    summary = ingest_schedule_range(f"{year}-01-01", f"{year}-12-31")
    for pk in summary.game_pks:
        try:
            ingest_game(pk, assume_final=only_final, require_final=only_final)
        except Exception:
            _log.exception("ingest.season.game_failed", game_pk=pk)
    if with_statcast:
        try:
            ingest_statcast_range(f"{year}-03-01", f"{year}-11-30")
        except Exception:
            _log.exception("ingest.season.statcast_failed", year=year)
    return summary
