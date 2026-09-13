"""Player card: identity, season stat lines, and recent per-game predictions."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from mlbsim_api.deps import get_session
from mlbsim_api.schemas import (
    PlayerCard,
    PlayerPredictionHistoryRow,
    PlayerSeasonLine,
)
from mlbsim_data.models import Game, Player, PlayerPrediction, PlayerSeasonStat

router = APIRouter(prefix="/players", tags=["players"])


@router.get("/{player_id}", response_model=PlayerCard)
def player_card(
    player_id: int,
    seasons: int = Query(4, ge=1, le=15, description="How many recent season lines to include."),
    limit: int = Query(20, ge=1, le=200, description="How many recent predictions to include."),
    s: Session = Depends(get_session),
) -> PlayerCard:
    player = s.get(Player, player_id)
    if player is None:
        raise HTTPException(404, f"unknown player {player_id}")

    lines = [
        PlayerSeasonLine(
            season=row.season,
            group=row.group,
            split=row.split,
            games=row.games,
            through_date=row.through_date,
            stats=row.stat_json,
        )
        for row in s.scalars(
            select(PlayerSeasonStat)
            .where(
                PlayerSeasonStat.player_id == player_id,
                PlayerSeasonStat.split == "all",
                PlayerSeasonStat.through_date.is_(None),
            )
            .order_by(desc(PlayerSeasonStat.season), PlayerSeasonStat.group)
            .limit(seasons * 2)  # batting + pitching per season
        )
    ]

    recent = [
        PlayerPredictionHistoryRow(
            pred_id=pred.pred_id,
            game_pk=pred.game_pk,
            game_date=game.game_date if game else None,
            model_id=pred.model_id,
            role=pred.role,
            created_at=pred.created_at,
            proj=pred.proj_json,
            prob=pred.prob_json,
        )
        for pred, game in s.execute(
            select(PlayerPrediction, Game)
            .outerjoin(Game, Game.game_pk == PlayerPrediction.game_pk)
            .where(PlayerPrediction.player_id == player_id)
            .order_by(desc(PlayerPrediction.created_at))
            .limit(limit)
        )
    ]

    return PlayerCard(
        player_id=player.player_id,
        full_name=player.full_name,
        bats=player.bats,
        throws=player.throws,
        primary_pos=player.primary_pos,
        birth_date=player.birth_date,
        season_lines=lines,
        recent_predictions=recent,
    )
