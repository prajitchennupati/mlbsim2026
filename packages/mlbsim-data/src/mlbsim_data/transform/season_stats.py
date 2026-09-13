"""Roll up game logs / play-by-play into ``player_season_stats`` and
``team_season_stats``.

``through_date`` (ISO ``YYYY-MM-DD``) restricts the aggregation to games on or
before that date, so the same code produces both full-season lines and the
point-in-time lines the M2 feature store needs.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import and_, case, func, select
from sqlalchemy.orm import Session

from mlbsim_core import get_logger, session_scope
from mlbsim_data.loaders.upsert import upsert
from mlbsim_data.models import (
    BattingGameLog,
    Game,
    PitchingGameLog,
    PlateAppearance,
    PlayerSeasonStat,
    TeamGameLog,
    TeamSeasonStat,
)
from mlbsim_data.transform.constants import FIP_CONSTANT, WOBA_WEIGHTS

_log = get_logger(__name__)

_HIT_EVENTS = ("single", "double", "triple", "home_run")


def _safe(numer: float, denom: float) -> float | None:
    return round(numer / denom, 4) if denom else None


# ---------------------------------------------------------------------------
# batting
# ---------------------------------------------------------------------------
def _batting_rates(c: dict[str, int]) -> dict[str, float | None]:
    singles = c["h"] - c["d2b"] - c["t3b"] - c["hr"]
    ubb = c["bb"] - c.get("ibb", 0)
    obp_denom = c["ab"] + c["bb"] + c["hbp"] + c["sf"]
    woba_num = (
        WOBA_WEIGHTS["bb"] * ubb
        + WOBA_WEIGHTS["hbp"] * c["hbp"]
        + WOBA_WEIGHTS["b1"] * singles
        + WOBA_WEIGHTS["b2"] * c["d2b"]
        + WOBA_WEIGHTS["b3"] * c["t3b"]
        + WOBA_WEIGHTS["hr"] * c["hr"]
    )
    woba_denom = c["ab"] + ubb + c["sf"] + c["hbp"]
    avg = _safe(c["h"], c["ab"])
    obp = _safe(c["h"] + c["bb"] + c["hbp"], obp_denom)
    slg = _safe(c["tb"], c["ab"])
    return {
        "avg": avg,
        "obp": obp,
        "slg": slg,
        "ops": None if obp is None or slg is None else round(obp + slg, 4),
        "iso": None if avg is None or slg is None else round(slg - avg, 4),
        "babip": _safe(c["h"] - c["hr"], c["ab"] - c["so"] - c["hr"] + c["sf"]),
        "bb_pct": _safe(c["bb"], c["pa"]),
        "k_pct": _safe(c["so"], c["pa"]),
        "woba": _safe(woba_num, woba_denom),
    }


def _batting_all(session: Session, season: int, through_date: str | None) -> list[dict[str, Any]]:
    g = BattingGameLog
    stmt = (
        select(
            g.player_id,
            func.count().label("games"),
            *[func.coalesce(func.sum(getattr(g, col)), 0).label(col) for col in _BAT_COLS],
        )
        .join(Game, Game.game_pk == g.game_pk)
        .where(g.season == season)
        .group_by(g.player_id)
    )
    if through_date:
        stmt = stmt.where(Game.game_date <= dt.date.fromisoformat(through_date))

    rows: list[dict[str, Any]] = []
    for rec in session.execute(stmt).mappings():
        counts = {col: int(rec[col]) for col in _BAT_COLS}
        rows.append(
            _season_row(
                rec["player_id"],
                season,
                "batting",
                "all",
                through_date=through_date,
                games=rec["games"],
                stat={**counts, **_batting_rates(counts)},
            )
        )
    return rows


_BAT_COLS = (
    "pa",
    "ab",
    "r",
    "h",
    "d2b",
    "t3b",
    "hr",
    "rbi",
    "bb",
    "ibb",
    "hbp",
    "so",
    "sb",
    "cs",
    "sac",
    "sf",
    "gidp",
    "tb",
)


def _batting_split(session: Session, season: int, through_date: str | None) -> list[dict[str, Any]]:
    pa = PlateAppearance
    is_hit = pa.event_type.in_(_HIT_EVENTS)
    stmt = (
        select(
            pa.batter_id.label("player_id"),
            pa.pitch_hand,
            func.count().label("pa"),
            func.sum(case((is_hit, 1), else_=0)).label("h"),
            func.sum(case((pa.event_type == "double", 1), else_=0)).label("d2b"),
            func.sum(case((pa.event_type == "triple", 1), else_=0)).label("t3b"),
            func.sum(case((pa.event_type == "home_run", 1), else_=0)).label("hr"),
            func.sum(case((pa.event_type.in_(("walk", "intent_walk")), 1), else_=0)).label("bb"),
            func.sum(case((pa.event_type == "hit_by_pitch", 1), else_=0)).label("hbp"),
            func.sum(case((pa.event_type.like("strikeout%"), 1), else_=0)).label("so"),
            func.sum(case((pa.is_sac, 1), else_=0)).label("sac_sf"),
        )
        .join(Game, Game.game_pk == pa.game_pk)
        .where(and_(pa.season == season, pa.pitch_hand.in_(("L", "R"))))
        .group_by(pa.batter_id, pa.pitch_hand)
    )
    if through_date:
        stmt = stmt.where(Game.game_date <= dt.date.fromisoformat(through_date))

    rows: list[dict[str, Any]] = []
    for rec in session.execute(stmt).mappings():
        pa_n = int(rec["pa"])
        ab = pa_n - int(rec["bb"]) - int(rec["hbp"]) - int(rec["sac_sf"])
        h, d2b, t3b, hr = (int(rec[k]) for k in ("h", "d2b", "t3b", "hr"))
        tb = (h - d2b - t3b - hr) + 2 * d2b + 3 * t3b + 4 * hr
        counts = {
            "pa": pa_n,
            "ab": max(ab, 0),
            "h": h,
            "d2b": d2b,
            "t3b": t3b,
            "hr": hr,
            "bb": int(rec["bb"]),
            "ibb": 0,
            "hbp": int(rec["hbp"]),
            "so": int(rec["so"]),
            "sf": 0,
            "tb": tb,
        }
        split = "vs_L" if rec["pitch_hand"] == "L" else "vs_R"
        rows.append(
            _season_row(
                rec["player_id"],
                season,
                "batting",
                split,
                through_date=through_date,
                games=None,
                stat={**counts, **_batting_rates(counts)},
            )
        )
    return rows


# ---------------------------------------------------------------------------
# pitching
# ---------------------------------------------------------------------------
_PIT_COLS = ("bf", "outs", "h", "r", "er", "bb", "ibb", "hbp", "so", "hr", "pitches", "strikes")


def _pitching_all(session: Session, season: int, through_date: str | None) -> list[dict[str, Any]]:
    g = PitchingGameLog
    stmt = (
        select(
            g.player_id,
            func.count().label("games"),
            func.sum(case((g.is_start, 1), else_=0)).label("gs"),
            func.sum(case((g.got_win, 1), else_=0)).label("w"),
            func.sum(case((g.got_loss, 1), else_=0)).label("l"),
            func.sum(case((g.got_save, 1), else_=0)).label("sv"),
            func.sum(case((g.quality_start, 1), else_=0)).label("qs"),
            *[func.coalesce(func.sum(getattr(g, col)), 0).label(col) for col in _PIT_COLS],
        )
        .join(Game, Game.game_pk == g.game_pk)
        .where(g.season == season)
        .group_by(g.player_id)
    )
    if through_date:
        stmt = stmt.where(Game.game_date <= dt.date.fromisoformat(through_date))

    rows: list[dict[str, Any]] = []
    for rec in session.execute(stmt).mappings():
        c = {col: int(rec[col]) for col in _PIT_COLS}
        ip = c["outs"] / 3
        fip = _safe(13 * c["hr"] + 3 * (c["bb"] + c["hbp"]) - 2 * c["so"], ip) if ip else None
        stat = {
            **c,
            "gs": int(rec["gs"]),
            "w": int(rec["w"]),
            "l": int(rec["l"]),
            "sv": int(rec["sv"]),
            "qs": int(rec["qs"]),
            "ip": round(ip, 1),
            "era": _safe(9 * c["er"], ip),
            "whip": _safe(c["h"] + c["bb"], ip),
            "k9": _safe(9 * c["so"], ip),
            "bb9": _safe(9 * c["bb"], ip),
            "hr9": _safe(9 * c["hr"], ip),
            "k_pct": _safe(c["so"], c["bf"]),
            "bb_pct": _safe(c["bb"], c["bf"]),
            "fip": None if fip is None else round(fip + FIP_CONSTANT, 3),
        }
        rows.append(
            _season_row(
                rec["player_id"],
                season,
                "pitching",
                "all",
                through_date=through_date,
                games=rec["games"],
                stat=stat,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# team
# ---------------------------------------------------------------------------
def rebuild_team_season_stats(season: int, *, through_date: str | None = None) -> int:
    g = TeamGameLog
    stmt = (
        select(
            g.team_id,
            func.count().label("g"),
            func.coalesce(func.sum(g.runs_for), 0).label("rf"),
            func.coalesce(func.sum(g.runs_against), 0).label("ra"),
            func.sum(case((g.won.is_(True), 1), else_=0)).label("w"),
            func.sum(case((g.won.is_(False), 1), else_=0)).label("losses"),
            func.sum(case((g.is_home, 1), else_=0)).label("home_g"),
        )
        .join(Game, Game.game_pk == g.game_pk)
        .where(g.season == season)
        .group_by(g.team_id)
    )
    if through_date:
        stmt = stmt.where(Game.game_date <= dt.date.fromisoformat(through_date))

    rows: list[dict[str, Any]] = []
    with session_scope() as s:
        for rec in s.execute(stmt).mappings():
            games, rf, ra = int(rec["g"]), int(rec["rf"]), int(rec["ra"])
            stat = {
                "g": games,
                "w": int(rec["w"]),
                "l": int(rec["losses"]),
                "home_g": int(rec["home_g"]),
                "rf": rf,
                "ra": ra,
                "run_diff": rf - ra,
                "rs_per_g": _safe(rf, games),
                "ra_per_g": _safe(ra, games),
                "win_pct": _safe(int(rec["w"]), games),
                "pythag_win_pct": _safe(rf**2, rf**2 + ra**2),
            }
            rows.append(
                {
                    "team_id": rec["team_id"],
                    "season": season,
                    "split": "all",
                    "through_date": through_date,
                    "stat_json": stat,
                }
            )
        written = upsert(s, TeamSeasonStat, rows, index_elements=["team_id", "season", "split"])
    _log.info("season_stats.team", season=season, through_date=through_date, rows=written)
    return written


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------
def _season_row(
    player_id: int,
    season: int,
    group: str,
    split: str,
    *,
    through_date: str | None,
    games: int | None,
    stat: dict[str, Any],
) -> dict[str, Any]:
    return {
        "player_id": player_id,
        "season": season,
        "group": group,
        "split": split,
        "through_date": through_date,
        "games": games or 0,
        "stat_json": stat,
    }


def rebuild_season_stats(season: int, *, through_date: str | None = None) -> int:
    """Rebuild all ``player_season_stats`` rows for a season. Returns rows written."""
    with session_scope() as s:
        rows = (
            _batting_all(s, season, through_date)
            + _batting_split(s, season, through_date)
            + _pitching_all(s, season, through_date)
        )
        written = upsert(
            s,
            PlayerSeasonStat,
            rows,
            index_elements=["player_id", "season", "group", "split"],
        )
    _log.info("season_stats.player", season=season, through_date=through_date, rows=written)
    return written
