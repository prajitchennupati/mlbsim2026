"""Build Elo ratings from the warehouse and persist ratings + pre-game predictions."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy import delete, insert, select

from mlbsim_core import get_logger, session_scope
from mlbsim_data.loaders.upsert import upsert
from mlbsim_data.models import (
    FINAL_STATUSES,
    VOID_STATUSES,
    EloRating,
    Game,
    GamePrediction,
    ModelVersion,
)
from mlbsim_models.ratings.elo import DEFAULT_RATING, EloConfig, expected_home_win, run_elo

_log = get_logger(__name__)

MODEL_ID = "elo_v1"
_REG_SEASON_TYPES = ("R",)


def _factors(
    home_rating: float, away_rating: float, home_field: float, p_home: float
) -> dict[str, Any]:
    """Elo isn't a sum-of-features model, so there's exactly one "factor" —
    still shaped as `top_factors` so the frontend's explanation panel and the
    /predictions/{id}/explanation endpoint have something to render."""
    favours = "home" if p_home >= 0.5 else "away"
    return {
        "home_rating": round(home_rating, 1),
        "away_rating": round(away_rating, 1),
        "home_field": home_field,
        "top_factors": [
            {
                "feature": "elo_rating",
                "label": f"Elo rating ({round(home_rating)} vs {round(away_rating)}, "
                f"+{home_field:g} home field)",
                "logit_contribution": 0.0,
                "favours": favours,
                "prob_shift": round((p_home if favours == "home" else 1.0 - p_home) - 0.5, 5),
            }
        ],
    }


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
        Game.status.in_(FINAL_STATUSES),
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
            "factors": _factors(
                res.home_rating_pre, res.away_rating_pre, cfg.home_field, res.home_win_prob
            ),
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


def predict_elo_date(
    start: str, end: str | None = None, *, cfg: EloConfig | None = None
) -> EloBuildSummary:
    """Pre-game elo_v1 predictions for not-yet-decided regular-season games in
    ``[start, end]`` (default: just ``start``) — "not yet decided" means
    ``status`` isn't in ``FINAL_STATUSES``/``VOID_STATUSES``, *not* a null
    score: the Stats API reports 0-0 for plenty of games that haven't started.
    Uses each team's most recent Elo rating from *before* the window (falling
    back to the default rating for a team with no finished games yet).
    ``build_elo`` only ever scores games with a real final result, so a
    freshly-ingested slate has no elo_v1 prediction until this runs; call it
    after ``build_elo`` for the same season.

    Ratings are snapshotted once at the start of the window rather than
    re-propagated day by day — future results aren't known yet to update on,
    and a multi-day slate's ratings don't move enough for that to matter.
    Idempotent: existing elo_v1 rows for the affected games are replaced.
    """
    cfg = cfg or EloConfig()
    end = end or start
    start_d, end_d = dt.date.fromisoformat(start), dt.date.fromisoformat(end)

    with session_scope() as s:
        games = list(
            s.execute(
                select(
                    Game.game_pk,
                    Game.home_team_id,
                    Game.away_team_id,
                )
                .where(
                    Game.game_type.in_(_REG_SEASON_TYPES),
                    Game.game_date >= start_d,
                    Game.game_date <= end_d,
                    Game.status.not_in(FINAL_STATUSES | VOID_STATUSES),
                )
                .order_by(Game.game_date, Game.game_pk)
            )
        )
        if not games:
            return EloBuildSummary(
                model_id=MODEL_ID, games_processed=0, rating_rows=0, prediction_rows=0
            )

        team_ids = {g.home_team_id for g in games} | {g.away_team_id for g in games}
        latest_rating: dict[int, float] = {}
        for tid, rating in s.execute(
            select(EloRating.entity_id, EloRating.rating)
            .where(
                EloRating.entity_type == "team",
                EloRating.entity_id.in_(team_ids),
                EloRating.as_of_date < start_d,
            )
            .order_by(EloRating.entity_id, EloRating.as_of_date)
        ):
            latest_rating[tid] = float(rating)  # ascending as_of_date -> last write is latest

        now = dt.datetime.now(dt.UTC)
        game_pks = [g.game_pk for g in games]
        prediction_rows = []
        for g in games:
            rh = latest_rating.get(g.home_team_id, DEFAULT_RATING)
            ra = latest_rating.get(g.away_team_id, DEFAULT_RATING)
            p = expected_home_win(rh, ra, cfg)
            prediction_rows.append(
                {
                    "game_pk": g.game_pk,
                    "model_id": MODEL_ID,
                    "created_at": now,
                    "is_live": False,
                    "home_win_prob": round(p, 5),
                    "away_win_prob": round(1.0 - p, 5),
                    "factors": _factors(rh, ra, cfg.home_field, p),
                }
            )

        s.execute(
            delete(GamePrediction).where(
                GamePrediction.model_id == MODEL_ID,
                GamePrediction.game_pk.in_(game_pks),
            )
        )
        s.execute(insert(GamePrediction), prediction_rows)

    summary = EloBuildSummary(
        model_id=MODEL_ID,
        games_processed=len(games),
        rating_rows=0,
        prediction_rows=len(prediction_rows),
    )
    _log.info("elo.predict", **asdict(summary))
    return summary
