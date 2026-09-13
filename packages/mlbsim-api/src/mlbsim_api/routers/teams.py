"""Team list + detail + schedule."""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from mlbsim_api._common import (
    build_game_summary,
    latest_prediction,
    outcomes_by_pred_id,
    team_abbr_map,
    team_by_abbr,
)
from mlbsim_api.deps import get_session
from mlbsim_api.schemas import GameSummary, TeamSummary
from mlbsim_data.models import EloRating, Game, SeasonSimTeamResult, SimulationRun, Team
from mlbsim_data.standings import TeamRecord, team_records

router = APIRouter(prefix="/teams", tags=["teams"])


def _latest_season_sim(s: Session) -> dict[int, SeasonSimTeamResult]:
    run = s.scalar(
        select(SimulationRun)
        .where(SimulationRun.scope == "season")
        .order_by(desc(SimulationRun.created_at))
        .limit(1)
    )
    if run is None:
        return {}
    return {
        r.team_id: r
        for r in s.scalars(
            select(SeasonSimTeamResult).where(SeasonSimTeamResult.sim_run_id == run.sim_run_id)
        )
    }


def _elo(s: Session) -> dict[int, float]:
    return {
        eid: float(rating)
        for eid, rating in s.execute(
            select(EloRating.entity_id, EloRating.rating)
            .where(EloRating.entity_type == "team")
            .order_by(EloRating.as_of_date)
        )
    }


def _summary(
    t: Team,
    records: dict[int, TeamRecord],
    elo: dict[int, float],
    sims: dict[int, SeasonSimTeamResult],
) -> TeamSummary:
    rec = records.get(t.team_id)
    sim = sims.get(t.team_id)
    return TeamSummary(
        team_id=t.team_id,
        abbr=t.abbr,
        name=t.name,
        league=t.league,
        division=t.division,
        wins=rec.wins if rec else None,
        losses=rec.losses if rec else None,
        elo=elo.get(t.team_id),
        p_playoffs=float(sim.p_playoffs) if sim else None,
        p_division=float(sim.p_division) if sim else None,
        p_world_series=float(sim.p_world_series) if sim else None,
        exp_wins=float(sim.exp_wins) if sim else None,
        exp_seed=float(sim.exp_seed) if sim and sim.exp_seed is not None else None,
    )


@router.get("", response_model=list[TeamSummary])
def list_teams(season: int | None = None, s: Session = Depends(get_session)) -> list[TeamSummary]:
    yr = season or dt.date.today().year
    records, elo, sims = team_records(s, yr), _elo(s), _latest_season_sim(s)
    return [_summary(t, records, elo, sims) for t in s.scalars(select(Team).order_by(Team.abbr))]


@router.get("/{abbr}", response_model=TeamSummary)
def team_detail(
    abbr: str, season: int | None = None, s: Session = Depends(get_session)
) -> TeamSummary:
    t = team_by_abbr(s, abbr)
    if t is None:
        raise HTTPException(404, f"unknown team {abbr!r}")
    yr = season or dt.date.today().year
    return _summary(t, team_records(s, yr), _elo(s), _latest_season_sim(s))


@router.get("/{abbr}/schedule", response_model=list[GameSummary])
def team_schedule(abbr: str, s: Session = Depends(get_session)) -> list[GameSummary]:
    t = team_by_abbr(s, abbr)
    if t is None:
        raise HTTPException(404, f"unknown team {abbr!r}")
    abbr_map = team_abbr_map(s)
    games = list(
        s.scalars(
            select(Game)
            .where((Game.home_team_id == t.team_id) | (Game.away_team_id == t.team_id))
            .order_by(Game.game_date, Game.game_pk)
        )
    )
    preds = {g.game_pk: latest_prediction(s, g.game_pk) for g in games}
    outcomes = outcomes_by_pred_id(s, [p.pred_id for p in preds.values() if p is not None])

    out: list[GameSummary] = []
    for g in games:
        p = preds[g.game_pk]
        oc = outcomes.get(p.pred_id) if p else None
        out.append(build_game_summary(g, abbr_map, p, oc))
    return out
