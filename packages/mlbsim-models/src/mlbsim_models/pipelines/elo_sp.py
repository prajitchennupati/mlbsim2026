"""elo_sp_v1: team Elo + a starting-pitcher FIP adjustment.

Validated against the real 2026 season (scripts/backtest_pitcher_adjustment.py,
1326 held-out games from 2026-06-01 on): a FIP-based adjustment scaled at
12-20 rating points per FIP-run consistently beat plain elo_v1 on accuracy,
Brier, and log-loss, peaking around scale 16-20. SCALE below is a deliberately
slightly conservative pick inside that range, not the single best grid point
(picking the literal best point on the same sample it's evaluated on would be
mild overfitting to that one backtest).

Two entry points:
  predict_elo_sp_date  - live/upcoming slate. Cutoff is always "yesterday",
                         which is naturally leakage-free for today-or-later games.
  backfill_elo_sp      - historical games, for the scorecard. Uses coarse
                         monthly-ish period snapshots (same discipline as the
                         backtest) rather than a snapshot per game, since a
                         season of games only needs a handful of distinct
                         pitcher/cutoff pairs that way.
"""

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
    GameProbable,
    ModelVersion,
)
from mlbsim_data.sources.mlb_statsapi import player_pitching_stats_range
from mlbsim_models.ratings.elo import EloConfig, expected_home_win
from mlbsim_models.ratings.pitcher import PitcherLine, rating_adjustment

_log = get_logger(__name__)

MODEL_ID = "elo_sp_v1"
SCALE = 15.0  # rating points per FIP-run better than league average; see module docstring
_REG_SEASON_TYPES = ("R",)


@dataclass(slots=True)
class EloSpSummary:
    prediction_rows: int
    pitcher_fetches: int


def _ip_to_outs(ip: str | None) -> int:
    if not ip:
        return 0
    whole, _, frac = ip.partition(".")
    return int(whole or 0) * 3 + int(frac or 0)


def _fetch_line(pitcher_id: int, season: int, season_start: str, cutoff: str) -> PitcherLine | None:
    stat = player_pitching_stats_range(pitcher_id, season_start, cutoff, season=season)
    if not stat:
        return None
    return PitcherLine(
        outs=_ip_to_outs(stat.get("inningsPitched")),
        home_runs=int(stat.get("homeRuns", 0)),
        walks=int(stat.get("baseOnBalls", 0)),
        hit_by_pitch=int(stat.get("hitBatsmen", 0)),
        strikeouts=int(stat.get("strikeOuts", 0)),
        batters_faced=int(stat.get("battersFaced", 0)),
    )


def _ensure_registered(now: dt.datetime) -> None:
    with session_scope() as s:
        upsert(
            s,
            ModelVersion,
            [
                {
                    "model_id": MODEL_ID,
                    "name": "Elo + starting pitcher (FIP-adjusted)",
                    "kind": "elo",
                    "version": "1",
                    "trained_at": now,
                    "hyperparams_json": {"points_per_fip_run": SCALE},
                    "notes": (
                        "Team Elo with a bounded per-game adjustment from each "
                        "starter's point-in-time FIP. Validated to beat plain "
                        "elo_v1 on the 2026 season; see backtest_pitcher_adjustment.py."
                    ),
                }
            ],
            index_elements=["model_id"],
        )


def _team_ratings_before(team_ids: set[int], before: dt.date) -> dict[int, float]:
    with session_scope() as s:
        rows = s.execute(
            select(EloRating.entity_id, EloRating.rating)
            .where(
                EloRating.entity_type == "team",
                EloRating.entity_id.in_(team_ids),
                EloRating.as_of_date < before,
            )
            .order_by(EloRating.entity_id, EloRating.as_of_date)
        )
        out: dict[int, float] = {}
        for tid, rating in rows:
            out[tid] = float(rating)  # ascending as_of_date -> last write is latest
        return out


def _write_predictions(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    game_pks = [r["game_pk"] for r in rows]
    with session_scope() as s:
        s.execute(
            delete(GamePrediction).where(
                GamePrediction.model_id == MODEL_ID, GamePrediction.game_pk.in_(game_pks)
            )
        )
        s.execute(insert(GamePrediction), rows)


def predict_elo_sp_date(
    start: str, end: str | None = None, *, cfg: EloConfig | None = None
) -> EloSpSummary:
    """Pre-game elo_sp_v1 picks for not-yet-decided games in [start, end]."""
    cfg = cfg or EloConfig()
    end = end or start
    start_d, end_d = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    now = dt.datetime.now(dt.UTC)
    _ensure_registered(now)

    with session_scope() as s:
        games = list(
            s.execute(
                select(Game.game_pk, Game.season, Game.home_team_id, Game.away_team_id)
                .where(
                    Game.game_type.in_(_REG_SEASON_TYPES),
                    Game.game_date >= start_d,
                    Game.game_date <= end_d,
                    Game.status.not_in(FINAL_STATUSES | VOID_STATUSES),
                )
            )
        )
        if not games:
            return EloSpSummary(prediction_rows=0, pitcher_fetches=0)
        probable_rows = s.execute(
            select(GameProbable.game_pk, GameProbable.team_id, GameProbable.probable_pitcher_id)
        )
        probables = {(r.game_pk, r.team_id): r.probable_pitcher_id for r in probable_rows}

    team_ids = {g.home_team_id for g in games} | {g.away_team_id for g in games}
    ratings = _team_ratings_before(team_ids, start_d)
    season = games[0].season
    cutoff = (start_d - dt.timedelta(days=1)).isoformat()
    season_start = f"{season}-03-01"

    line_cache: dict[int, PitcherLine | None] = {}
    fetches = 0

    def line_for(pid: int | None) -> PitcherLine | None:
        nonlocal fetches
        if pid is None:
            return None
        if pid not in line_cache:
            line_cache[pid] = _fetch_line(pid, season, season_start, cutoff)
            fetches += 1
        return line_cache[pid]

    rows = []
    for g in games:
        rh = ratings.get(g.home_team_id, 1500.0)
        ra = ratings.get(g.away_team_id, 1500.0)
        home_adj = rating_adjustment(
            line_for(probables.get((g.game_pk, g.home_team_id))), points_per_fip_run=SCALE
        )
        away_adj = rating_adjustment(
            line_for(probables.get((g.game_pk, g.away_team_id))), points_per_fip_run=SCALE
        )
        p = expected_home_win(rh + home_adj, ra + away_adj, cfg)
        rows.append(
            {
                "game_pk": g.game_pk,
                "model_id": MODEL_ID,
                "created_at": now,
                "is_live": False,
                "home_win_prob": round(p, 5),
                "away_win_prob": round(1.0 - p, 5),
                "factors": {
                    "home_rating": round(rh, 1),
                    "away_rating": round(ra, 1),
                    "home_sp_adjustment": round(home_adj, 1),
                    "away_sp_adjustment": round(away_adj, 1),
                    "home_field": cfg.home_field,
                    "top_factors": [
                        {
                            "feature": "starting_pitcher",
                            "label": (
                                f"Starting pitcher edge ({home_adj:+.0f} home / "
                                f"{away_adj:+.0f} away)"
                            ),
                            "logit_contribution": 0.0,
                            "favours": "home" if home_adj >= away_adj else "away",
                            "prob_shift": round(abs(home_adj - away_adj) / 400.0, 5),
                        }
                    ],
                },
            }
        )
    _write_predictions(rows)
    summary = EloSpSummary(prediction_rows=len(rows), pitcher_fetches=fetches)
    _log.info("elo_sp.predict", **asdict(summary))
    return summary


def backfill_elo_sp(
    season: int,
    period_bounds: list[str],
    *,
    season_start: str,
    cfg: EloConfig | None = None,
) -> EloSpSummary:
    """elo_sp_v1 predictions for already-finished games, using coarse
    (monthly-ish) point-in-time pitcher snapshots — see module docstring."""
    cfg = cfg or EloConfig()
    now = dt.datetime.now(dt.UTC)
    _ensure_registered(now)

    with session_scope() as s:
        games = list(
            s.execute(
                select(
                    Game.game_pk,
                    Game.game_date,
                    Game.home_team_id,
                    Game.away_team_id,
                )
                .where(
                    Game.season == season,
                    Game.game_type.in_(_REG_SEASON_TYPES),
                    Game.status.in_(FINAL_STATUSES),
                    Game.home_score.is_not(None),
                    Game.game_date >= dt.date.fromisoformat(period_bounds[0]),
                )
            )
        )
        probable_rows = s.execute(
            select(GameProbable.game_pk, GameProbable.team_id, GameProbable.probable_pitcher_id)
        )
        probables = {(r.game_pk, r.team_id): r.probable_pitcher_id for r in probable_rows}

    def period_of(game_date: dt.date) -> int:
        for i, bound in enumerate(period_bounds):
            if game_date.isoformat() < bound:
                return i
        return len(period_bounds)

    fetches = 0
    line_cache: dict[tuple[int, int], PitcherLine | None] = {}

    def line_for(pid: int | None, period: int) -> PitcherLine | None:
        nonlocal fetches
        if pid is None or period == 0:
            return None
        key = (period, pid)
        if key not in line_cache:
            boundary = dt.date.fromisoformat(period_bounds[period - 1])
            cutoff = (boundary - dt.timedelta(days=1)).isoformat()
            line_cache[key] = _fetch_line(pid, season, season_start, cutoff)
            fetches += 1
        return line_cache[key]

    rows = []
    for g in games:
        period = period_of(g.game_date)
        ratings = _team_ratings_before({g.home_team_id, g.away_team_id}, g.game_date)
        rh = ratings.get(g.home_team_id, 1500.0)
        ra = ratings.get(g.away_team_id, 1500.0)
        home_adj = rating_adjustment(
            line_for(probables.get((g.game_pk, g.home_team_id)), period), points_per_fip_run=SCALE
        )
        away_adj = rating_adjustment(
            line_for(probables.get((g.game_pk, g.away_team_id)), period), points_per_fip_run=SCALE
        )
        p = expected_home_win(rh + home_adj, ra + away_adj, cfg)
        rows.append(
            {
                "game_pk": g.game_pk,
                "model_id": MODEL_ID,
                "created_at": now,
                "is_live": False,
                "home_win_prob": round(p, 5),
                "away_win_prob": round(1.0 - p, 5),
                "factors": {
                    "home_rating": round(rh, 1),
                    "away_rating": round(ra, 1),
                    "home_sp_adjustment": round(home_adj, 1),
                    "away_sp_adjustment": round(away_adj, 1),
                    "home_field": cfg.home_field,
                },
            }
        )
    _write_predictions(rows)
    summary = EloSpSummary(prediction_rows=len(rows), pitcher_fetches=fetches)
    _log.info("elo_sp.backfill", **asdict(summary))
    return summary
