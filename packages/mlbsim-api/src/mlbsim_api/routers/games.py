"""Game list + detail, inning table, player projections, simulation summary."""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from mlbsim_api._common import (
    latest_prediction,
    outcomes_by_pred_id,
    predicted_winner,
    team_abbr_map,
    team_by_abbr,
)
from mlbsim_api.deps import get_session
from mlbsim_api.schemas import (
    GameDetail,
    GamePlayers,
    GameSummary,
    InningRow,
    InningTable,
    PlayerPredictionOut,
    SimulationOut,
)
from mlbsim_data.models import (
    Game,
    GameSimResultRow,
    PlayerPrediction,
    SimulationRun,
    is_game_final,
)

router = APIRouter(prefix="/games", tags=["games"])


@router.get("", response_model=list[GameSummary])
def list_games(
    date: dt.date | None = Query(None, description="default: today"),
    team: str | None = Query(None, description="team abbreviation"),
    status: str | None = None,
    s: Session = Depends(get_session),
) -> list[GameSummary]:
    day = date or dt.date.today()
    abbr = team_abbr_map(s)
    stmt = (
        select(Game).where(Game.game_date == day).order_by(Game.scheduled_start_utc, Game.game_pk)
    )
    if team:
        t = team_by_abbr(s, team)
        if t is None:
            raise HTTPException(404, f"unknown team {team!r}")
        stmt = stmt.where((Game.home_team_id == t.team_id) | (Game.away_team_id == t.team_id))
    if status:
        stmt = stmt.where(Game.status.ilike(f"%{status}%"))

    games = list(s.scalars(stmt))
    preds = {g.game_pk: latest_prediction(s, g.game_pk) for g in games}
    outcomes = outcomes_by_pred_id(s, [p.pred_id for p in preds.values() if p is not None])

    out: list[GameSummary] = []
    for g in games:
        p = preds[g.game_pk]
        oc = outcomes.get(p.pred_id) if p else None
        out.append(
            GameSummary(
                game_pk=g.game_pk,
                season=g.season,
                game_date=g.game_date,
                start_time_utc=g.scheduled_start_utc,
                status=g.status,
                is_final=is_game_final(g.status),
                home_team_id=g.home_team_id,
                away_team_id=g.away_team_id,
                home_abbr=abbr.get(g.home_team_id),
                away_abbr=abbr.get(g.away_team_id),
                home_score=g.home_score,
                away_score=g.away_score,
                home_win_prob=float(p.home_win_prob) if p else None,
                away_win_prob=float(p.away_win_prob) if p else None,
                exp_home_runs=float(p.exp_home_runs) if p and p.exp_home_runs is not None else None,
                exp_away_runs=float(p.exp_away_runs) if p and p.exp_away_runs is not None else None,
                model_id=p.model_id if p else None,
                pred_id=p.pred_id if p else None,
                predicted_winner=predicted_winner(p) if p else None,
                actual_winner=oc.actual_winner if oc else None,
                correct=oc.correct if oc else None,
                brier=float(oc.brier) if oc and oc.brier is not None else None,
            )
        )
    return out


@router.get("/{game_pk}", response_model=GameDetail)
def game_detail(game_pk: int, s: Session = Depends(get_session)) -> GameDetail:
    g = s.get(Game, game_pk)
    if g is None:
        raise HTTPException(404, f"game {game_pk} not found")
    abbr = team_abbr_map(s)
    p = latest_prediction(s, game_pk)
    oc = outcomes_by_pred_id(s, [p.pred_id]).get(p.pred_id) if p else None
    return GameDetail(
        game_pk=g.game_pk,
        season=g.season,
        game_date=g.game_date,
        start_time_utc=g.scheduled_start_utc,
        status=g.status,
        is_final=is_game_final(g.status),
        home_team_id=g.home_team_id,
        away_team_id=g.away_team_id,
        home_abbr=abbr.get(g.home_team_id),
        away_abbr=abbr.get(g.away_team_id),
        home_score=g.home_score,
        away_score=g.away_score,
        park_id=g.park_id,
        scheduled_start_utc=g.scheduled_start_utc,
        home_sp_id=g.home_sp_id,
        away_sp_id=g.away_sp_id,
        model_id=p.model_id if p else None,
        pred_id=p.pred_id if p else None,
        predicted_winner=predicted_winner(p) if p else None,
        actual_winner=oc.actual_winner if oc else None,
        correct=oc.correct if oc else None,
        brier=float(oc.brier) if oc and oc.brier is not None else None,
        home_win_prob=float(p.home_win_prob) if p else None,
        away_win_prob=float(p.away_win_prob) if p else None,
        exp_home_runs=float(p.exp_home_runs) if p and p.exp_home_runs is not None else None,
        exp_away_runs=float(p.exp_away_runs) if p and p.exp_away_runs is not None else None,
        p_extra_innings=float(p.p_extra_innings) if p and p.p_extra_innings is not None else None,
        p_shutout_home=float(p.p_shutout_home) if p and p.p_shutout_home is not None else None,
        p_shutout_away=float(p.p_shutout_away) if p and p.p_shutout_away is not None else None,
        p_one_run_game=float(p.p_one_run_game) if p and p.p_one_run_game is not None else None,
        home_score_dist=p.home_score_dist if p else None,
        away_score_dist=p.away_score_dist if p else None,
        total_runs_dist=p.total_runs_dist if p else None,
        most_likely_scores=p.most_likely_scores if p else None,
        factors=p.factors if p else None,
    )


def _latest_sim(s: Session, game_pk: int) -> tuple[SimulationRun, GameSimResultRow] | None:
    row = s.execute(
        select(SimulationRun, GameSimResultRow)
        .join(GameSimResultRow, GameSimResultRow.sim_run_id == SimulationRun.sim_run_id)
        .where(GameSimResultRow.game_pk == game_pk)
        .order_by(desc(SimulationRun.created_at))
        .limit(1)
    ).first()
    return (row[0], row[1]) if row else None


@router.get("/{game_pk}/simulation", response_model=SimulationOut)
def game_simulation(game_pk: int, s: Session = Depends(get_session)) -> SimulationOut:
    got = _latest_sim(s, game_pk)
    if got is None:
        raise HTTPException(404, f"no simulation for game {game_pk}")
    run, res = got
    return SimulationOut(
        game_pk=game_pk,
        sim_run_id=run.sim_run_id,
        n_sims=run.n_sims,
        engine_version=run.engine_version,
        runtime_ms=run.runtime_ms,
        summary=res.summary_json,
    )


@router.get("/{game_pk}/innings", response_model=InningTable)
def game_innings(game_pk: int, s: Session = Depends(get_session)) -> InningTable:
    got = _latest_sim(s, game_pk)
    if got is None:
        return InningTable(game_pk=game_pk, source="none")
    inning_probs = (got[1].summary_json or {}).get("inning_probs", {})
    rows: list[InningRow] = []
    for side in ("home", "away"):
        for i, r in enumerate(inning_probs.get(side, []), start=1):
            rows.append(
                InningRow(
                    inning=i,
                    side=side,
                    exp_runs=r["exp_runs"],
                    p_score_1plus=r["p_score_1plus"],
                    p_score_2plus=r["p_score_2plus"],
                    p_score_3plus=r["p_score_3plus"],
                    p_scoreless=r["p_scoreless"],
                )
            )
    return InningTable(
        game_pk=game_pk,
        source="simulation",
        rows=rows,
        p_home_lead_after=inning_probs.get("p_home_lead_after", []),
    )


@router.get("/{game_pk}/players", response_model=GamePlayers)
def game_players(game_pk: int, s: Session = Depends(get_session)) -> GamePlayers:
    preds = list(
        s.scalars(
            select(PlayerPrediction)
            .where(PlayerPrediction.game_pk == game_pk)
            .order_by(desc(PlayerPrediction.created_at))
        )
    )
    seen: set[tuple[int, str]] = set()
    batters: list[PlayerPredictionOut] = []
    pitchers: list[PlayerPredictionOut] = []
    for pr in preds:
        key = (pr.player_id, pr.role)
        if key in seen:
            continue
        seen.add(key)
        item = PlayerPredictionOut(
            player_id=pr.player_id,
            role=pr.role,
            model_id=pr.model_id,
            proj=pr.proj_json,
            prob=pr.prob_json,
        )
        (batters if pr.role == "bat" else pitchers).append(item)
    return GamePlayers(game_pk=game_pk, batters=batters, pitchers=pitchers)


@router.get("/{game_pk}/live", status_code=404)
def game_live(game_pk: int) -> dict[str, str]:
    raise HTTPException(404, "live predictions arrive in M9")
