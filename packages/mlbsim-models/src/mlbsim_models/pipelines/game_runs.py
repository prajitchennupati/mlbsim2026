"""runs_v1: team-level Poisson score + inning-level run forecasts.

Reuses ``direct.runs_glm``'s ``score_grid``/``derive_game_probs`` -- the same
independent-Poisson machinery already in the codebase for ``direct_v1`` -- for
the score-distribution math. What's new is estimating each game's expected
runs from Game-derived team offense/defense factors (no box-score ingest
needed: `Game.home_score`/`away_score` is enough) and a league-wide per-inning
scoring shape parsed straight out of the schedule endpoint's ``linescore``
(real per-inning history already sitting in the cached landing-zone JSON from
the season's schedule ingest -- no extra network calls).
"""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sqlalchemy import insert, select, update

from mlbsim_core import get_logger, session_scope
from mlbsim_data.models import FINAL_STATUSES, Game, GamePrediction, GameSimResultRow, SimulationRun
from mlbsim_data.sources.mlb_statsapi import schedule
from mlbsim_models.direct.runs_glm import derive_game_probs, score_grid

_log = get_logger(__name__)

MODEL_ID = "runs_v1"
ENGINE_VERSION = "runs_v1"
_N_INNINGS = 9
_MC_SAMPLES = 10_000


def build_team_run_factors(season: int) -> tuple[dict[int, float], dict[int, float], float, float]:
    """(offense_factor, defense_factor, league_avg_runs_per_team_game, home_boost)
    -- offense/defense are each team's runs-scored/-allowed per game relative
    to the league average; home_boost is the empirical home/away scoring ratio."""
    with session_scope() as s:
        stmt = select(
            Game.home_team_id, Game.away_team_id, Game.home_score, Game.away_score
        ).where(
            Game.season == season,
            Game.game_type == "R",
            Game.status.in_(FINAL_STATUSES),
            Game.home_score.is_not(None),
        )
        rows = list(s.execute(stmt))
    scored: dict[int, list[int]] = defaultdict(list)
    allowed: dict[int, list[int]] = defaultdict(list)
    total_runs = 0
    home_runs_sum = away_runs_sum = 0
    for h, a, hs, as_ in rows:
        scored[h].append(hs)
        allowed[h].append(as_)
        scored[a].append(as_)
        allowed[a].append(hs)
        total_runs += hs + as_
        home_runs_sum += hs
        away_runs_sum += as_

    n_team_games = 2 * len(rows) or 1
    league_avg = total_runs / n_team_games if total_runs else 4.3
    offense = {tid: (sum(v) / len(v)) / league_avg for tid, v in scored.items()}
    defense = {tid: (sum(v) / len(v)) / league_avg for tid, v in allowed.items()}
    home_boost = 1.05
    if rows and away_runs_sum:
        home_boost = (home_runs_sum / len(rows)) / (away_runs_sum / len(rows))
    return offense, defense, league_avg, home_boost


def league_inning_shares(start: str, end: str) -> tuple[list[float], list[float]]:
    """(home_share[9], away_share[9]): average fraction of a team's game total
    that arrives in each inning, from real linescore data (schedule endpoint,
    already cached from the season's schedule ingest)."""
    games = schedule(start, end)
    home_by_inning = [0.0] * _N_INNINGS
    away_by_inning = [0.0] * _N_INNINGS
    for g in games:
        if g.get("status", {}).get("detailedState") not in FINAL_STATUSES:
            continue
        for inn in g.get("linescore", {}).get("innings", []):
            num = inn.get("num")
            if not num or num > _N_INNINGS:
                continue
            hr = inn.get("home", {}).get("runs")
            ar = inn.get("away", {}).get("runs")
            if hr is not None:
                home_by_inning[num - 1] += hr
            if ar is not None:
                away_by_inning[num - 1] += ar
    home_total = sum(home_by_inning) or 1.0
    away_total = sum(away_by_inning) or 1.0
    home_share = [x / home_total for x in home_by_inning]
    away_share = [x / away_total for x in away_by_inning]
    return home_share, away_share


def _poisson_bucket_probs(mu: float) -> dict[str, float]:
    mu = max(mu, 1e-9)
    p0 = math.exp(-mu)
    p1 = p0 * mu
    p2 = p1 * mu / 2
    return {
        "p_scoreless": round(p0, 5),
        "p_score_1plus": round(1 - p0, 5),
        "p_score_2plus": round(max(1 - p0 - p1, 0.0), 5),
        "p_score_3plus": round(max(1 - p0 - p1 - p2, 0.0), 5),
    }


def _inning_row(inning: int, mu: float) -> dict[str, Any]:
    return {"inning": inning, "exp_runs": round(mu, 3), **_poisson_bucket_probs(mu)}


def _inning_probs(
    mu_home: float, mu_away: float, home_share: list[float], away_share: list[float]
) -> dict[str, Any]:
    home_rows = [_inning_row(i + 1, mu_home * home_share[i]) for i in range(_N_INNINGS)]
    away_rows = [_inning_row(i + 1, mu_away * away_share[i]) for i in range(_N_INNINGS)]

    # p_home_lead_after: cheap Monte Carlo over the per-inning Poisson rates
    # (correlating the cumulative score requires sampling, not a closed form).
    rng = np.random.default_rng(0)
    home_draws = np.zeros(_MC_SAMPLES)
    away_draws = np.zeros(_MC_SAMPLES)
    p_lead = []
    for i in range(_N_INNINGS):
        home_draws += rng.poisson(mu_home * home_share[i], _MC_SAMPLES)
        away_draws += rng.poisson(mu_away * away_share[i], _MC_SAMPLES)
        p_lead.append(round(float(np.mean(home_draws > away_draws)), 4))

    return {"home": home_rows, "away": away_rows, "p_home_lead_after": p_lead}


@dataclass(slots=True)
class RunsSummary:
    prediction_rows: int
    sim_rows: int


def predict_runs_date(
    season: int, start: str, end: str | None = None, *, season_start: str = "2026-03-01"
) -> RunsSummary:
    """Score distributions + inning-level forecasts for not-yet-decided games
    in [start, end]. Requires an existing win-prob prediction (any model) for
    the same games -- this only adds the run/score/inning layer on top."""
    end = end or start
    start_d, end_d = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    offense, defense, league_avg, home_boost = build_team_run_factors(season)
    home_share, away_share = league_inning_shares(season_start, end)

    with session_scope() as s:
        games = list(
            s.execute(
                select(Game.game_pk, Game.game_date, Game.home_team_id, Game.away_team_id).where(
                    Game.season == season,
                    Game.game_type == "R",
                    Game.game_date >= start_d,
                    Game.game_date <= end_d,
                    Game.home_score.is_(None),
                )
            )
        )
        preferred = ("ensemble_v1", "elo_sp_v1", "elo_v1")
        win_preds: dict[int, GamePrediction] = {}
        for model_id in preferred:
            for pk, pred in s.execute(
                select(GamePrediction.game_pk, GamePrediction)
                .where(
                    GamePrediction.model_id == model_id,
                    GamePrediction.game_pk.in_([g.game_pk for g in games]),
                    GamePrediction.is_live.is_(False),
                )
                .order_by(GamePrediction.created_at)
            ):
                win_preds.setdefault(pk, pred)

    now = dt.datetime.now(dt.UTC)
    pred_rows: list[dict[str, Any]] = []
    sim_run_ids: list[int] = []
    with session_scope() as s:
        for g in games:
            off_h, off_a = offense.get(g.home_team_id, 1.0), offense.get(g.away_team_id, 1.0)
            def_h, def_a = defense.get(g.home_team_id, 1.0), defense.get(g.away_team_id, 1.0)
            mu_home = league_avg * off_h * def_a * home_boost
            mu_away = league_avg * off_a * def_h
            grid = score_grid(mu_home, mu_away)
            probs = derive_game_probs(grid)

            win = win_preds.get(g.game_pk)
            if win is not None:
                s.execute(
                    update(GamePrediction)
                    .where(GamePrediction.pred_id == win.pred_id)
                    .values(
                        exp_home_runs=probs["exp_home_runs"],
                        exp_away_runs=probs["exp_away_runs"],
                        home_score_dist=probs["home_score_dist"],
                        away_score_dist=probs["away_score_dist"],
                        total_runs_dist=probs["total_runs_dist"],
                        most_likely_scores=probs["most_likely_scores"],
                        p_extra_innings=probs["p_extra_innings"],
                        p_shutout_home=probs["p_shutout_home"],
                        p_shutout_away=probs["p_shutout_away"],
                        p_one_run_game=probs["p_one_run_game"],
                    )
                )
                pred_rows.append({"game_pk": g.game_pk})

            inning_probs = _inning_probs(mu_home, mu_away, home_share, away_share)
            sim_run_id = s.execute(
                insert(SimulationRun)
                .values(
                    scope="game",
                    target_id=g.game_pk,
                    model_id=MODEL_ID,
                    engine_version=ENGINE_VERSION,
                    n_sims=_MC_SAMPLES,
                    seed=0,
                    runtime_ms=None,
                    params_json={"mu_home": mu_home, "mu_away": mu_away},
                    created_at=now,
                )
                .returning(SimulationRun.sim_run_id)
            ).scalar_one()
            s.execute(
                insert(GameSimResultRow).values(
                    sim_run_id=sim_run_id,
                    game_pk=g.game_pk,
                    home_win_prob=probs["home_win_prob"],
                    exp_home_runs=probs["exp_home_runs"],
                    exp_away_runs=probs["exp_away_runs"],
                    summary_json={"inning_probs": inning_probs, **probs},
                )
            )
            sim_run_ids.append(sim_run_id)

    summary = RunsSummary(prediction_rows=len(pred_rows), sim_rows=len(sim_run_ids))
    _log.info("runs.predict", **asdict(summary))
    return summary
