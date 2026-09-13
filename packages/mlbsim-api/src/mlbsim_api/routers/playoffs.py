"""Playoff picture — the latest season Monte Carlo, framed as odds + series."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from mlbsim_api.deps import get_session
from mlbsim_api.routers.simulations import latest_season_run, season_payload
from mlbsim_api.schemas import PlayoffsOut

router = APIRouter(prefix="/playoffs", tags=["playoffs"])


@router.get("", response_model=PlayoffsOut)
def playoffs(s: Session = Depends(get_session)) -> PlayoffsOut:
    run = latest_season_run(s)
    if run is None:
        raise HTTPException(404, "no season simulation has been run yet")
    payload = season_payload(s, run)
    payload.teams = [t for t in payload.teams if (t.p_playoffs or 0) > 0.001]
    return payload
