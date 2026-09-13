"""Build and persist ``warehouse.game_features`` from warehouse facts."""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from typing import Any

from sqlalchemy import select

from mlbsim_core import get_logger, session_scope
from mlbsim_data.loaders.upsert import upsert
from mlbsim_data.models import (
    FeatureSet,
    Game,
    GameFeature,
    PitchingGameLog,
    TeamGameLog,
)
from mlbsim_features.asof import snapshot_hash
from mlbsim_features.build import (
    DIFF_FEATURE_ORDER,
    FEATURE_SET_ID,
    assemble_game_features,
    pitcher_form,
    team_form,
)

_log = get_logger(__name__)

_SPEC = {
    "diff_features": list(DIFF_FEATURE_ORDER),
    "sources": ["team_game_logs", "pitching_game_logs"],
    "as_of": "game_date 00:00 (pre-game); strict '<' history bound",
}


def _team_logs_by_team(season: int) -> dict[int, list[dict[str, Any]]]:
    stmt = (
        select(
            TeamGameLog.team_id,
            Game.game_pk,
            Game.game_date,
            TeamGameLog.runs_for,
            TeamGameLog.runs_against,
            TeamGameLog.won,
        )
        .join(Game, Game.game_pk == TeamGameLog.game_pk)
        .where(TeamGameLog.season == season)
    )
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with session_scope() as s:
        for r in s.execute(stmt):
            out[r.team_id].append(
                {
                    "game_pk": r.game_pk,
                    "game_date": r.game_date,
                    "runs_for": r.runs_for,
                    "runs_against": r.runs_against,
                    "won": r.won,
                }
            )
    return out


def _pitch_logs_by_pitcher(season: int) -> dict[int, list[dict[str, Any]]]:
    cols = ("outs", "er", "so", "bb", "hbp", "hr", "bf", "is_start")
    stmt = (
        select(
            PitchingGameLog.player_id,
            Game.game_date,
            *[getattr(PitchingGameLog, c) for c in cols],
        )
        .join(Game, Game.game_pk == PitchingGameLog.game_pk)
        .where(PitchingGameLog.season == season)
    )
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with session_scope() as s:
        for r in s.execute(stmt):
            out[r.player_id].append({"game_date": r.game_date, **{c: getattr(r, c) for c in cols}})
    return out


def _target_games(seasons: list[int] | None, game_pks: list[int] | None) -> list[dict[str, Any]]:
    stmt = select(
        Game.game_pk,
        Game.game_date,
        Game.season,
        Game.home_team_id,
        Game.away_team_id,
        Game.home_sp_id,
        Game.away_sp_id,
    ).where(Game.game_type == "R")
    if seasons:
        stmt = stmt.where(Game.season.in_(seasons))
    if game_pks:
        stmt = stmt.where(Game.game_pk.in_(game_pks))
    with session_scope() as s:
        return [
            {
                "game_pk": r.game_pk,
                "game_date": r.game_date,
                "season": r.season,
                "home_team_id": r.home_team_id,
                "away_team_id": r.away_team_id,
                "home_sp_id": r.home_sp_id,
                "away_sp_id": r.away_sp_id,
            }
            for r in s.execute(stmt)
        ]


def build_game_features(
    *,
    seasons: list[int] | None = None,
    game_pks: list[int] | None = None,
    feature_set_id: str = FEATURE_SET_ID,
) -> int:
    """Compute + upsert home/away/diff feature rows for the target games."""
    games = _target_games(seasons, game_pks)
    by_season: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for g in games:
        by_season[g["season"]].append(g)

    now = dt.datetime.now(dt.UTC)
    rows: list[dict[str, Any]] = []
    for season, season_games in by_season.items():
        team_logs = _team_logs_by_team(season)
        pitch_logs = _pitch_logs_by_pitcher(season)
        for g in season_games:
            as_of = g["game_date"]
            home_hist = team_logs.get(g["home_team_id"], [])
            away_hist = team_logs.get(g["away_team_id"], [])
            feats = assemble_game_features(
                team_form(home_hist, as_of),
                team_form(away_hist, as_of),
                pitcher_form(pitch_logs.get(g["home_sp_id"] or -1, []), as_of),
                pitcher_form(pitch_logs.get(g["away_sp_id"] or -1, []), as_of),
            )
            src_ids = [r["game_pk"] for r in home_hist + away_hist if r["game_date"] < as_of]
            digest = snapshot_hash(src_ids, max_source_ts=as_of.isoformat())
            for side in ("home", "away", "diff"):
                rows.append(
                    {
                        "game_pk": g["game_pk"],
                        "feature_set_id": feature_set_id,
                        "side": side,
                        "as_of_ts": dt.datetime.combine(as_of, dt.time(), dt.UTC),
                        "features": feats[side],
                        "data_snapshot_hash": digest,
                        "computed_at": now,
                    }
                )

    with session_scope() as s:
        upsert(
            s,
            FeatureSet,
            [
                {
                    "feature_set_id": feature_set_id,
                    "name": "Team form + starter form (home-away diff)",
                    "version": "1",
                    "spec_json": _SPEC,
                }
            ],
            index_elements=["feature_set_id"],
        )
        written = upsert(
            s,
            GameFeature,
            rows,
            index_elements=["game_pk", "feature_set_id", "side"],
        )
    _log.info("features.build", games=len(games), rows=written, feature_set=feature_set_id)
    return written
