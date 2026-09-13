"""Build Elo ratings from the warehouse and persist ratings + pre-game predictions."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy import delete, insert, select

from mlbsim_core import get_logger, session_scope
from mlbsim_data.loaders.upsert import upsert
from mlbsim_data.models import EloRating, Game, GamePrediction, ModelVersion
from mlbsim_models.ratings.elo import EloConfig, run_elo

_log = get_logger(__name__)

MODEL_ID = "elo_v1"
_REG_SEASON_TYPES = ("R",)


@dataclass(slots=True)
class EloBuildSummary:
    model_id: str
    games_processed: int
    rating_rows: int
    prediction_rows: int


def _load_games(seasons: list[int] | None) -> list[dict[str, Any]]:
    stmt = select(
        Game.game_pk,
        Game.game_date,
        Game.season,
        Game.home_team_id,
        Game.away_team_id,
        Game.home_score,
        Game.away_score,
    ).where(
        Game.game_type.in_(_REG_SEASON_TYPES),
        Game.home_score.is_not(None),
        Game.away_score.is_not(None),
    )
    if seasons:
        stmt = stmt.where(Game.season.in_(seasons))
    with session_scope() as s:
        return [
            {
                "game_pk": r.game_pk,
                "game_date": r.game_date.isoformat(),
                "season": r.season,
                "home_team_id": r.home_team_id,
                "away_team_id": r.away_team_id,
                "home_score": r.home_score,
                "away_score": r.away_score,
            }
            for r in s.execute(stmt)
        ]


def build_elo(
    *, seasons: list[int] | None = None, cfg: EloConfig | None = None, rebuild: bool = True
) -> EloBuildSummary:
    """Run Elo over finished regular-season games; persist ratings and predictions."""
    cfg = cfg or EloConfig()
    games = _load_games(seasons)
    results = run_elo(games, cfg)
    now = dt.datetime.now(dt.UTC)

    # One rating row per (team, date): keep the last game of a doubleheader, since
    # results are already chronological. This also satisfies the ON CONFLICT
    # constraint (no duplicate PKs within one upsert batch).
    rating_by_key: dict[tuple[int, dt.date], dict[str, Any]] = {}
    for res in results:
        as_of = dt.date.fromisoformat(res.game_date)
        for tid, rating in (
            (res.home_team_id, res.home_rating_post),
            (res.away_team_id, res.away_rating_post),
        ):
            rating_by_key[(tid, as_of)] = {
                "entity_type": "team",
                "entity_id": tid,
                "as_of_date": as_of,
                "rating": round(rating, 2),
                "rating_sp_adj": None,
                "games_played": 0,
            }
    rating_rows: list[dict[str, Any]] = list(rating_by_key.values())
    prediction_rows = [
        {
            "game_pk": res.game_pk,
            "model_id": MODEL_ID,
            "created_at": now,
            "is_live": False,
            "home_win_prob": round(res.home_win_prob, 5),
            "away_win_prob": round(1.0 - res.home_win_prob, 5),
            "factors": {
                "home_rating_pre": round(res.home_rating_pre, 1),
                "away_rating_pre": round(res.away_rating_pre, 1),
                "home_field": cfg.home_field,
            },
        }
        for res in results
    ]

    with session_scope() as s:
        upsert(
            s,
            ModelVersion,
            [
                {
                    "model_id": MODEL_ID,
                    "name": "Elo (team, MOV, season revert)",
                    "kind": "elo",
                    "version": "1",
                    "trained_at": now,
                    "hyperparams_json": {
                        "k": cfg.k,
                        "home_field": cfg.home_field,
                        "scale": cfg.scale,
                        "mov_enabled": cfg.mov_enabled,
                        "season_revert": cfg.season_revert,
                    },
                    "notes": "Leakage-proof W/L baseline; chronological single pass.",
                }
            ],
            index_elements=["model_id"],
        )
        if rebuild:
            s.execute(delete(GamePrediction).where(GamePrediction.model_id == MODEL_ID))
        upsert(
            s,
            EloRating,
            rating_rows,
            index_elements=["entity_type", "entity_id", "as_of_date"],
        )
        if prediction_rows:
            s.execute(insert(GamePrediction), prediction_rows)

    summary = EloBuildSummary(
        model_id=MODEL_ID,
        games_processed=len(results),
        rating_rows=len(rating_rows),
        prediction_rows=len(prediction_rows),
    )
    _log.info("elo.build", **asdict(summary))
    return summary
