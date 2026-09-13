"""Computed standings from team game logs."""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, Query
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from mlbsim_api.deps import get_session
from mlbsim_api.schemas import StandingRow
from mlbsim_data.models import Game, Team, TeamGameLog

router = APIRouter(prefix="/standings", tags=["standings"])


@router.get("", response_model=list[StandingRow])
def standings(
    season: int | None = Query(None),
    date: dt.date | None = Query(None, description="only games on/before this date"),
    s: Session = Depends(get_session),
) -> list[StandingRow]:
    yr = season or (date.year if date else dt.date.today().year)
    stmt = (
        select(
            TeamGameLog.team_id,
            func.sum(case((TeamGameLog.won.is_(True), 1), else_=0)).label("w"),
            func.count().label("g"),
            func.coalesce(func.sum(TeamGameLog.runs_for), 0).label("rf"),
            func.coalesce(func.sum(TeamGameLog.runs_against), 0).label("ra"),
        )
        .join(Game, Game.game_pk == TeamGameLog.game_pk)
        .where(TeamGameLog.season == yr)
        .group_by(TeamGameLog.team_id)
    )
    if date:
        stmt = stmt.where(Game.game_date <= date)

    teams = {t.team_id: t for t in s.scalars(select(Team))}
    out: list[StandingRow] = []
    for tid, w_raw, g_raw, rf, ra in s.execute(stmt):
        t = teams.get(tid)
        w, g = int(w_raw), int(g_raw)
        out.append(
            StandingRow(
                team_id=tid,
                abbr=t.abbr if t else None,
                league=t.league if t else None,
                division=t.division if t else None,
                wins=w,
                losses=g - w,
                pct=round(w / g, 4) if g else 0.0,
                run_diff=int(rf) - int(ra),
            )
        )
    out.sort(key=lambda r: (r.league or "", r.division or "", -r.pct))
    return out
