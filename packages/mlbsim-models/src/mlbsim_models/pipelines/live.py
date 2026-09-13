"""Live in-game win probability: resume the PA simulator from the current state."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy import insert

from mlbsim_core import get_logger, session_scope
from mlbsim_data.loaders.upsert import upsert
from mlbsim_data.models import Game, GamePrediction, ModelVersion
from mlbsim_data.sources import mlb_statsapi
from mlbsim_engine import ENGINE_VERSION, GameState, live_win_probability
from mlbsim_engine.pipeline import offense_rates, starter_rates

_log = get_logger(__name__)

MODEL_ID = "live_v1"
_LIVE_STATES = {"Live", "In Progress", "Manager challenge"}


@dataclass(slots=True)
class LiveUpdateSummary:
    date: str
    live_games: int
    written: int


def _to_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def state_from_feed(feed: dict[str, Any]) -> GameState | None:
    """Extract a :class:`GameState` from a GUMBO ``feed/live`` document, or None if not live."""
    gd, ld = feed.get("gameData", {}), feed.get("liveData", {})
    if gd.get("status", {}).get("abstractGameState") != "Live":
        return None
    ls = ld.get("linescore", {}) or {}
    teams = ls.get("teams", {}) or {}
    off = ls.get("offense", {}) or {}
    base = (
        (1 if off.get("first") else 0)
        | (2 if off.get("second") else 0)
        | (4 if off.get("third") else 0)
    )

    cur = (ld.get("plays", {}) or {}).get("currentPlay", {}) or {}
    raw_outs = ls.get("outs") if ls.get("outs") is not None else cur.get("count", {}).get("outs")
    outs = _to_int(raw_outs)
    is_top = bool(ls.get("isTopInning", True))
    batting_order = _to_int(off.get("battingOrder"))

    probable = gd.get("probablePitchers", {}) or {}
    cur_pitcher = _to_int((cur.get("matchup", {}) or {}).get("pitcher", {}).get("id"))
    home_sp = _to_int((probable.get("home") or {}).get("id"))
    away_sp = _to_int((probable.get("away") or {}).get("id"))
    inning = max(_to_int(ls.get("currentInning")) or 1, 1)

    return GameState(
        inning=inning,
        half="top" if is_top else "bottom",
        outs=min(max(outs, 0), 2),
        base_state=base,
        home_score=_to_int((teams.get("home") or {}).get("runs")),
        away_score=_to_int((teams.get("away") or {}).get("runs")),
        bo_away=(batting_order - 1) % 9 if is_top and batting_order else 0,
        bo_home=(batting_order - 1) % 9 if not is_top and batting_order else 0,
        home_sp_out=inning >= 6 or (cur_pitcher != 0 and is_top and cur_pitcher != home_sp),
        away_sp_out=inning >= 6 or (cur_pitcher != 0 and not is_top and cur_pitcher != away_sp),
    )


def _register_model() -> None:
    with session_scope() as s:
        upsert(
            s,
            ModelVersion,
            [
                {
                    "model_id": MODEL_ID,
                    "name": "Live win probability (resumed PA simulation)",
                    "kind": "sim",
                    "version": ENGINE_VERSION,
                    "trained_at": dt.datetime.now(dt.UTC),
                    "hyperparams_json": {"engine_version": ENGINE_VERSION},
                    "notes": "Resumes the plate-appearance simulator from the live game state.",
                }
            ],
            index_elements=["model_id"],
        )


def live_win_prob_for_game(
    game_pk: int, *, n_sims: int = 6_000, seed: int = 0
) -> dict[str, Any] | None:
    """Fetch a game's live feed, resume-simulate, and return the win-prob payload."""
    feed = mlb_statsapi.game_feed(game_pk)
    state = state_from_feed(feed)
    if state is None:
        return None
    with session_scope() as s:
        game = s.get(Game, game_pk)
        if game is None:
            return None
        home_id, away_id, game_date = game.home_team_id, game.away_team_id, game.game_date
        home_sp_id, away_sp_id = game.home_sp_id, game.away_sp_id

    p = live_win_probability(
        state,
        home_lineup=offense_rates(game_pk, home_id, game_date),
        away_lineup=offense_rates(game_pk, away_id, game_date),
        home_sp=starter_rates(home_sp_id, game_date),
        away_sp=starter_rates(away_sp_id, game_date),
        n_sims=n_sims,
        seed=seed,
    )
    return {
        "game_pk": game_pk,
        "home_win_prob": round(p, 5),
        "state": {
            "inning": state.inning,
            "half": state.half,
            "outs": state.outs,
            "base_state": state.base_state,
            "home_score": state.home_score,
            "away_score": state.away_score,
        },
    }


def update_live_games(date: str | None = None, *, n_sims: int = 6_000) -> LiveUpdateSummary:
    """Write an ``is_live`` prediction row for every in-progress game on ``date``."""
    day = date or dt.date.today().isoformat()
    schedule = mlb_statsapi.schedule(day, day)
    live_pks = [
        _to_int(g.get("gamePk"))
        for g in schedule
        if g.get("status", {}).get("abstractGameState") == "Live"
        or g.get("status", {}).get("detailedState") in _LIVE_STATES
    ]
    if live_pks:
        _register_model()

    now = dt.datetime.now(dt.UTC)
    rows: list[dict[str, Any]] = []
    for pk in live_pks:
        payload = live_win_prob_for_game(pk, n_sims=n_sims)
        if payload is None:
            continue
        rows.append(
            {
                "game_pk": pk,
                "model_id": MODEL_ID,
                "created_at": now,
                "is_live": True,
                "game_state_json": payload["state"],
                "home_win_prob": payload["home_win_prob"],
                "away_win_prob": round(1.0 - payload["home_win_prob"], 5),
            }
        )
    with session_scope() as s:
        if rows:
            s.execute(insert(GamePrediction), rows)

    summary = LiveUpdateSummary(date=day, live_games=len(live_pks), written=len(rows))
    _log.info("live.update", **asdict(summary))
    return summary
