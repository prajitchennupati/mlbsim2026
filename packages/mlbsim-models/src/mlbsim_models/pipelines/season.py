"""Assemble a :class:`SeasonSetup` from the warehouse and run the season Monte Carlo."""

from __future__ import annotations

import datetime as dt
import time
from typing import Any

import numpy as np
from sqlalchemy import insert, select

from mlbsim_core import get_logger, session_scope
from mlbsim_data.models import (
    EloRating,
    Game,
    GamePrediction,
    SeasonSimTeamResult,
    SeriesPrediction,
    SimulationRun,
    Team,
)
from mlbsim_data.standings import team_records
from mlbsim_engine import ENGINE_VERSION
from mlbsim_engine.season import SeasonSetup, simulate_season

_log = get_logger(__name__)

_LEAGUE = {"AL": 0, "American League": 0, "NL": 1, "National League": 1}
_DIV = {"E": 0, "C": 1, "W": 2}


def _div_code(value: str | None) -> int:
    if not value:
        return 0
    last = value.strip().rsplit(" ", 1)[-1]
    return {"East": 0, "Central": 1, "West": 2, "E": 0, "C": 1, "W": 2}.get(last, 0)


def build_season_setup(
    season: int, *, model_id: str = "direct_v1", as_of: str | None = None
) -> tuple[SeasonSetup, dict[str, Any]]:
    cutoff = dt.date.fromisoformat(as_of) if as_of else dt.date.today()

    with session_scope() as s:
        teams = list(
            s.execute(select(Team.team_id, Team.league, Team.division).order_by(Team.team_id))
        )
        if len(teams) != 30:
            raise ValueError(f"expected 30 teams in warehouse.teams, found {len(teams)}")
        team_ids = [t.team_id for t in teams]
        idx = {tid: i for i, tid in enumerate(team_ids)}
        league = np.array([_LEAGUE.get((t.league or "").strip(), 0) for t in teams], np.int64)
        division = np.array([_div_code(t.division) for t in teams], np.int64)

        w2d = np.zeros(30, np.int64)
        l2d = np.zeros(30, np.int64)
        for tid, rec in team_records(s, season, through=cutoff).items():
            if tid in idx:
                w2d[idx[tid]] = rec.wins
                l2d[idx[tid]] = rec.losses

        elo = {
            eid: float(rating)
            for eid, rating in s.execute(
                select(EloRating.entity_id, EloRating.rating)
                .where(EloRating.entity_type == "team")
                .order_by(EloRating.as_of_date)
            )
        }

        rem = list(
            s.execute(
                select(Game.game_pk, Game.home_team_id, Game.away_team_id)
                .where(
                    Game.season == season,
                    Game.game_type == "R",
                    Game.game_date >= cutoff,
                )
                .order_by(Game.game_pk)
            )
        )
        preds = {
            gpk: float(p)
            for gpk, p in s.execute(
                select(GamePrediction.game_pk, GamePrediction.home_win_prob)
                .where(GamePrediction.model_id == model_id)
                .order_by(GamePrediction.created_at)
            )
        }

    strength = np.array([(elo.get(tid, 1500.0) - 1500.0) / 400.0 for tid in team_ids], np.float64)
    rem_home = np.array([idx[g.home_team_id] for g in rem], np.int64)
    rem_away = np.array([idx[g.away_team_id] for g in rem], np.int64)
    rem_p_home = np.array(
        [
            preds.get(
                g.game_pk,
                float(
                    1.0
                    / (
                        1.0
                        + np.exp(
                            -(strength[idx[g.home_team_id]] - strength[idx[g.away_team_id]] + 0.15)
                        )
                    )
                ),
            )
            for g in rem
        ],
        np.float64,
    )

    setup = SeasonSetup(
        team_ids=team_ids,
        league=league,
        division=division,
        wins_to_date=w2d,
        losses_to_date=l2d,
        strength=strength,
        rem_home=rem_home,
        rem_away=rem_away,
        rem_p_home=rem_p_home,
    )
    meta = {
        "season": season,
        "as_of": cutoff.isoformat(),
        "model_id": model_id,
        "remaining_games": len(rem),
        "games_with_prediction": sum(1 for g in rem if g.game_pk in preds),
    }
    return setup, meta


def simulate_season_from_db(
    season: int,
    *,
    n_sims: int = 100_000,
    seed: int = 0,
    model_id: str = "direct_v1",
    as_of: str | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    setup, meta = build_season_setup(season, model_id=model_id, as_of=as_of)
    t0 = time.perf_counter()
    result = simulate_season(setup, n_sims=n_sims, seed=seed)
    runtime_ms = int((time.perf_counter() - t0) * 1000)

    team_rows = result.summary()
    series_rows = result.series_length_summary()

    if persist:
        now = dt.datetime.now(dt.UTC)
        with session_scope() as s:
            sim_run_id = s.execute(
                insert(SimulationRun)
                .values(
                    scope="season",
                    target_id=season,
                    model_id=model_id,
                    engine_version=ENGINE_VERSION,
                    n_sims=n_sims,
                    seed=seed,
                    runtime_ms=runtime_ms,
                    params_json=meta,
                    created_at=now,
                )
                .returning(SimulationRun.sim_run_id)
            ).scalar_one()
            s.execute(
                insert(SeasonSimTeamResult),
                [
                    {
                        "sim_run_id": sim_run_id,
                        "team_id": r["team_id"],
                        "p_playoffs": r["p_playoffs"],
                        "p_division": r["p_division"],
                        "p_wildcard": r["p_wildcard"],
                        "p_bye": r["p_bye"],
                        "p_pennant": r["p_pennant"],
                        "p_world_series": r["p_world_series"],
                        "exp_wins": r["exp_wins"],
                        "exp_losses": r["exp_losses"],
                        "exp_seed": r["exp_seed"],
                        "exp_postseason_game_wins": r["exp_postseason_game_wins"],
                        "win_dist": r["win_dist"],
                    }
                    for r in team_rows
                ],
            )
            if series_rows:
                s.execute(
                    insert(SeriesPrediction),
                    [
                        {
                            "sim_run_id": sim_run_id,
                            "round": rnd,
                            "best_of": d["best_of"],
                            "p_sweep": d["p_sweep"],
                            "exp_games": d["exp_games"],
                            "game_count_dist": d["game_count_dist"],
                        }
                        for rnd, d in series_rows.items()
                    ],
                )
            meta["sim_run_id"] = sim_run_id

    meta["runtime_ms"] = runtime_ms
    top = sorted(team_rows, key=lambda r: -r["p_world_series"])[:8]
    _log.info(
        "season.simulate",
        season=season,
        n_sims=n_sims,
        runtime_ms=runtime_ms,
        remaining_games=meta["remaining_games"],
    )
    return {"meta": meta, "top_world_series": top, "series": series_rows}
