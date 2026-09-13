"""Computed standings, straight from Game results."""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from mlbsim_api.deps import get_session
from mlbsim_api.schemas import StandingRow
from mlbsim_data.models import Team
from mlbsim_data.standings import team_records

router = APIRouter(prefix="/standings", tags=["standings"])


@router.get("", response_model=list[StandingRow])
def standings(
    season: int | None = Query(None),
    date: dt.date | None = Query(None, description="only games strictly before this date"),
    s: Session = Depends(get_session),
) -> list[StandingRow]:
    yr = season or (date.year if date else dt.date.today().year)
    records = team_records(s, yr, through=date)
    teams = {t.team_id: t for t in s.scalars(select(Team))}

    out: list[StandingRow] = []
    for tid, rec in records.items():
        t = teams.get(tid)
        out.append(
            StandingRow(
                team_id=tid,
                abbr=t.abbr if t else None,
                league=t.league if t else None,
                division=t.division if t else None,
                wins=rec.wins,
                losses=rec.losses,
                pct=rec.pct,
                run_diff=rec.run_diff,
            )
        )
    out.sort(key=lambda r: (r.league or "", r.division or "", -r.pct))
    return out
