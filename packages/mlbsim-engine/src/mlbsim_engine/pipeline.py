"""Run the game simulator for a warehouse game and persist the result."""

from __future__ import annotations

import datetime as dt
import time
from typing import Any

import numpy as np
from sqlalchemy import insert, select

from mlbsim_core import get_logger, session_scope
from mlbsim_data.models import (
    Game,
    GameSimResultRow,
    Lineup,
    PlayerSeasonStat,
    SimulationRun,
)
from mlbsim_engine import ENGINE_VERSION
from mlbsim_engine.game import simulate_game
from mlbsim_engine.rates import neutral_lineup, neutral_pitcher, rates_from_stat_json

_log = get_logger(__name__)


def _season_stat(player_id: int, group: str, game_date: dt.date) -> dict[str, Any] | None:
    """Latest point-in-time ``player_season_stats`` line for a player as of a date."""
    stmt = (
        select(PlayerSeasonStat.stat_json, PlayerSeasonStat.through_date)
        .where(
            PlayerSeasonStat.player_id == player_id,
            PlayerSeasonStat.group == group,
            PlayerSeasonStat.split == "all",
        )
        .order_by(PlayerSeasonStat.through_date.desc().nulls_last())
    )
    with session_scope() as s:
        for stat_json, through in s.execute(stmt):
            if through is None or dt.date.fromisoformat(through) <= game_date:
                return dict(stat_json)
    return None


def offense_rates(game_pk: int, team_id: int, game_date: dt.date) -> np.ndarray:
    with session_scope() as s:
        slots = list(
            s.execute(
                select(Lineup.batting_order, Lineup.player_id)
                .where(Lineup.game_pk == game_pk, Lineup.team_id == team_id)
                .order_by(Lineup.batting_order)
            )
        )
    if not slots:
        return neutral_lineup()
    rows = []
    for _, player_id in slots[:9]:
        stat = _season_stat(player_id, "batting", game_date)
        rows.append(rates_from_stat_json(stat) if stat else neutral_lineup()[0])
    while len(rows) < 9:
        rows.append(neutral_lineup()[0])
    return np.vstack(rows)


def starter_rates(player_id: int | None, game_date: dt.date) -> np.ndarray:
    if player_id is None:
        return neutral_pitcher()
    stat = _season_stat(player_id, "pitching", game_date)
    return rates_from_stat_json(stat, is_pitcher=True) if stat else neutral_pitcher()


def simulate_game_from_db(
    game_pk: int, *, n_sims: int = 10_000, seed: int = 0, persist: bool = True
) -> dict[str, Any]:
    """Load lineups + rates for ``game_pk``, simulate, and (optionally) store."""
    with session_scope() as s:
        game = s.get(Game, game_pk)
        if game is None:
            raise LookupError(f"game {game_pk} not ingested")
        home_team_id, away_team_id = game.home_team_id, game.away_team_id
        game_date = game.game_date
        home_sp_id, away_sp_id = game.home_sp_id, game.away_sp_id

    home_off = offense_rates(game_pk, home_team_id, game_date)
    away_off = offense_rates(game_pk, away_team_id, game_date)
    home_sp = starter_rates(home_sp_id, game_date)
    away_sp = starter_rates(away_sp_id, game_date)

    t0 = time.perf_counter()
    result = simulate_game(
        home_lineup=home_off,
        away_lineup=away_off,
        home_sp=home_sp,
        away_sp=away_sp,
        n_sims=n_sims,
        seed=seed,
    )
    runtime_ms = int((time.perf_counter() - t0) * 1000)
    summary = result.summary()

    if persist:
        now = dt.datetime.now(dt.UTC)
        with session_scope() as s:
            sim_run_id = s.execute(
                insert(SimulationRun)
                .values(
                    scope="game",
                    target_id=game_pk,
                    model_id=None,
                    engine_version=ENGINE_VERSION,
                    n_sims=n_sims,
                    seed=seed,
                    runtime_ms=runtime_ms,
                    params_json={"hard_hook_inning": 8},
                    created_at=now,
                )
                .returning(SimulationRun.sim_run_id)
            ).scalar_one()
            s.execute(
                insert(GameSimResultRow).values(
                    sim_run_id=sim_run_id,
                    game_pk=game_pk,
                    home_win_prob=summary["home_win_prob"],
                    exp_home_runs=summary["exp_home_runs"],
                    exp_away_runs=summary["exp_away_runs"],
                    summary_json=summary,
                )
            )
            summary["sim_run_id"] = sim_run_id

    summary["runtime_ms"] = runtime_ms
    _log.info(
        "engine.simulate_game",
        game_pk=game_pk,
        n_sims=n_sims,
        runtime_ms=runtime_ms,
        home_win_prob=summary["home_win_prob"],
    )
    return summary
