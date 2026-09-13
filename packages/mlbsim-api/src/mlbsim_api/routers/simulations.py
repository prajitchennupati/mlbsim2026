"""Season Monte Carlo results."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from mlbsim_api._common import team_abbr_map
from mlbsim_api.deps import get_session
from mlbsim_api.schemas import PlayoffsOut, TeamSummary
from mlbsim_data.models import SeasonSimTeamResult, SeriesPrediction, SimulationRun, Team

router = APIRouter(prefix="/simulations", tags=["simulations"])


def latest_season_run(s: Session) -> SimulationRun | None:
    return s.scalar(
        select(SimulationRun)
        .where(SimulationRun.scope == "season")
        .order_by(desc(SimulationRun.created_at))
        .limit(1)
    )


def season_payload(s: Session, run: SimulationRun) -> PlayoffsOut:
    abbr = team_abbr_map(s)
    names = {t.team_id: t.name for t in s.scalars(select(Team))}
    leagues = {t.team_id: (t.league, t.division) for t in s.scalars(select(Team))}
    teams = [
        TeamSummary(
            team_id=r.team_id,
            abbr=abbr.get(r.team_id),
            name=names.get(r.team_id),
            league=leagues.get(r.team_id, (None, None))[0],
            division=leagues.get(r.team_id, (None, None))[1],
            p_playoffs=float(r.p_playoffs),
            p_division=float(r.p_division),
            p_world_series=float(r.p_world_series),
            exp_wins=float(r.exp_wins),
            exp_seed=float(r.exp_seed) if r.exp_seed is not None else None,
        )
        for r in s.scalars(
            select(SeasonSimTeamResult)
            .where(SeasonSimTeamResult.sim_run_id == run.sim_run_id)
            .order_by(desc(SeasonSimTeamResult.p_world_series))
        )
    ]
    series = [
        {
            "round": sp.round,
            "best_of": sp.best_of,
            "p_sweep": float(sp.p_sweep),
            "exp_games": float(sp.exp_games),
            "game_count_dist": sp.game_count_dist,
        }
        for sp in s.scalars(
            select(SeriesPrediction).where(SeriesPrediction.sim_run_id == run.sim_run_id)
        )
    ]
    meta = run.params_json or {}
    return PlayoffsOut(
        sim_run_id=run.sim_run_id,
        season=meta.get("season") or run.target_id,
        as_of=meta.get("as_of"),
        teams=teams,
        series=series,
    )


@router.get("/season/latest", response_model=PlayoffsOut)
def season_latest(s: Session = Depends(get_session)) -> PlayoffsOut:
    run = latest_season_run(s)
    if run is None:
        raise HTTPException(404, "no season simulation has been run yet")
    return season_payload(s, run)
