"""Team list + detail + schedule."""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import case, desc, func, select
from sqlalchemy.orm import Session

from mlbsim_api._common import latest_prediction, team_abbr_map, team_by_abbr
from mlbsim_api.deps import get_session
from mlbsim_api.schemas import GameSummary, TeamSummary
from mlbsim_data.models import (
    EloRating,
    Game,
    SeasonSimTeamResult,
    SimulationRun,
    Team,
    TeamGameLog,
)

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


def _records(s: Session) -> dict[int, tuple[int, int]]:
    rows = s.execute(
        select(
            TeamGameLog.team_id,
            func.sum(case((TeamGameLog.won.is_(True), 1), else_=0)),
            func.count(),
        ).group_by(TeamGameLog.team_id)
    )
    out: dict[int, tuple[int, int]] = {}
    for tid, won, n in rows:
        w = int(won or 0)
        out[tid] = (w, int(n) - w)
    return out


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
    records: Mapping[int, tuple[int | None, int | None]],
    elo: dict[int, float],
    sims: dict[int, SeasonSimTeamResult],
) -> TeamSummary:
    rec = records.get(t.team_id, (None, None))
    sim = sims.get(t.team_id)
    return TeamSummary(
        team_id=t.team_id,
        abbr=t.abbr,
        name=t.name,
        league=t.league,
        division=t.division,
        wins=rec[0],
        losses=rec[1],
        elo=elo.get(t.team_id),
        p_playoffs=float(sim.p_playoffs) if sim else None,
        p_division=float(sim.p_division) if sim else None,
        p_world_series=float(sim.p_world_series) if sim else None,
        exp_wins=float(sim.exp_wins) if sim else None,
        exp_seed=float(sim.exp_seed) if sim and sim.exp_seed is not None else None,
    )


@router.get("", response_model=list[TeamSummary])
def list_teams(s: Session = Depends(get_session)) -> list[TeamSummary]:
    records, elo, sims = _records(s), _elo(s), _latest_season_sim(s)
    return [_summary(t, records, elo, sims) for t in s.scalars(select(Team).order_by(Team.abbr))]


@router.get("/{abbr}", response_model=TeamSummary)
def team_detail(abbr: str, s: Session = Depends(get_session)) -> TeamSummary:
    t = team_by_abbr(s, abbr)
    if t is None:
        raise HTTPException(404, f"unknown team {abbr!r}")
    return _summary(t, _records(s), _elo(s), _latest_season_sim(s))


@router.get("/{abbr}/schedule", response_model=list[GameSummary])
def team_schedule(abbr: str, s: Session = Depends(get_session)) -> list[GameSummary]:
    t = team_by_abbr(s, abbr)
    if t is None:
        raise HTTPException(404, f"unknown team {abbr!r}")
    names = team_abbr_map(s)
    games = s.scalars(
        select(Game)
        .where((Game.home_team_id == t.team_id) | (Game.away_team_id == t.team_id))
        .order_by(Game.game_date, Game.game_pk)
    )
    out: list[GameSummary] = []
    for g in games:
        p = latest_prediction(s, g.game_pk)
        out.append(
            GameSummary(
                game_pk=g.game_pk,
                season=g.season,
                game_date=g.game_date,
                status=g.status,
                home_team_id=g.home_team_id,
                away_team_id=g.away_team_id,
                home_abbr=names.get(g.home_team_id),
                away_abbr=names.get(g.away_team_id),
                home_score=g.home_score,
                away_score=g.away_score,
                home_win_prob=float(p.home_win_prob) if p else None,
                away_win_prob=float(p.away_win_prob) if p else None,
                model_id=p.model_id if p else None,
            )
        )
    return out
