"""Regular-season W/L + run differential, computed straight from ``Game``.

``warehouse.team_game_log`` (per-team, per-game aggregates) is only populated
by a full box-score ingest (``ingest game`` / ``ingest season``), which is a
lot heavier than a plain schedule ingest. Every ``Game`` row already carries
the final score once a game is over, so standings, team records, and the
season simulation's win-to-date bootstrap don't need to wait on that —
this reads the same source `list_games` already trusts for scores.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from mlbsim_data.models import FINAL_STATUSES, Game


@dataclass(slots=True)
class TeamRecord:
    wins: int = 0
    losses: int = 0
    runs_for: int = 0
    runs_against: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses

    @property
    def pct(self) -> float:
        return round(self.wins / self.games, 4) if self.games else 0.0

    @property
    def run_diff(self) -> int:
        return self.runs_for - self.runs_against


def team_records(
    session: Session,
    season: int,
    *,
    through: dt.date | None = None,
    game_types: tuple[str, ...] = ("R",),
) -> dict[int, TeamRecord]:
    """Per-team record for finished games in ``season`` (optionally only games
    strictly before ``through``, for a point-in-time cutoff)."""
    stmt = select(
        Game.home_team_id,
        Game.away_team_id,
        Game.home_score,
        Game.away_score,
    ).where(
        Game.season == season,
        Game.game_type.in_(game_types),
        Game.status.in_(FINAL_STATUSES),
        Game.home_score.is_not(None),
        Game.away_score.is_not(None),
    )
    if through is not None:
        stmt = stmt.where(Game.game_date < through)

    out: dict[int, TeamRecord] = {}

    def bump(team_id: int, won: bool, runs_for: int, runs_against: int) -> None:
        rec = out.setdefault(team_id, TeamRecord())
        if won:
            rec.wins += 1
        else:
            rec.losses += 1
        rec.runs_for += runs_for
        rec.runs_against += runs_against

    for home_id, away_id, raw_hs, raw_as in session.execute(stmt):
        hs, as_ = int(raw_hs), int(raw_as)
        bump(home_id, hs > as_, hs, as_)
        bump(away_id, as_ > hs, as_, hs)
    return out
