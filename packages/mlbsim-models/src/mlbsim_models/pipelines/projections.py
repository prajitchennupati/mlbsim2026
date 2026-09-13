"""Build Marcel projections from ``player_season_stats`` and store them back."""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import select

from mlbsim_core import get_logger, session_scope
from mlbsim_data.loaders.upsert import upsert
from mlbsim_data.models import Player, PlayerSeasonStat
from mlbsim_models.projections.marcel import MarcelProjection, project_marcel

_log = get_logger(__name__)

MARCEL_SPLIT = "marcel"


def _age_on(birth_date: dt.date | None, season: int) -> float | None:
    if birth_date is None:
        return None
    return (dt.date(season, 7, 1) - birth_date).days / 365.25


def _prior_lines(player_id: int, group: str, target_season: int) -> list[dict[str, Any] | None]:
    stmt = select(PlayerSeasonStat.season, PlayerSeasonStat.stat_json).where(
        PlayerSeasonStat.player_id == player_id,
        PlayerSeasonStat.group == group,
        PlayerSeasonStat.split == "all",
        PlayerSeasonStat.season.in_([target_season - 1, target_season - 2, target_season - 3]),
    )
    with session_scope() as s:
        by_season = {row.season: dict(row.stat_json) for row in s.execute(stmt)}
    return [by_season.get(target_season - k) for k in (1, 2, 3)]


def marcel_for_player(
    player_id: int, target_season: int, *, is_pitcher: bool = False
) -> MarcelProjection:
    group = "pitching" if is_pitcher else "batting"
    with session_scope() as s:
        player = s.get(Player, player_id)
        birth = player.birth_date if player else None
    priors = _prior_lines(player_id, group, target_season)
    return project_marcel(priors, age=_age_on(birth, target_season), is_pitcher=is_pitcher)


def store_marcel_projections(target_season: int) -> int:
    """Project every player with a prior-season line; store as split='marcel' rows."""
    prior = target_season - 1
    with session_scope() as s:
        rows = list(
            s.execute(
                select(PlayerSeasonStat.player_id, PlayerSeasonStat.group).where(
                    PlayerSeasonStat.season == prior,
                    PlayerSeasonStat.split == "all",
                )
            )
        )

    out: list[dict[str, Any]] = []
    for player_id, group in rows:
        is_p = group == "pitching"
        proj = marcel_for_player(player_id, target_season, is_pitcher=is_p)
        out.append(
            {
                "player_id": player_id,
                "season": target_season,
                "group": group,
                "split": MARCEL_SPLIT,
                "through_date": f"{target_season}-01-01",
                "games": 0,
                "stat_json": {
                    "proj_pa": proj.proj_pa,
                    "age_factor": proj.age_factor,
                    "rates": proj.rates.round(6).tolist(),
                    **proj.components,
                },
            }
        )

    with session_scope() as s:
        written = upsert(
            s,
            PlayerSeasonStat,
            out,
            index_elements=["player_id", "season", "group", "split"],
        )
    _log.info("projections.marcel", season=target_season, players=written)
    return written
