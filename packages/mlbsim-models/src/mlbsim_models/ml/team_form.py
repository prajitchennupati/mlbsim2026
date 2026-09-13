"""Point-in-time team form, computed straight from ``Game`` in one chronological
pass -- the same "process games in order, snapshot before updating" discipline
as ``mlbsim_models.ratings.elo.run_elo``, so a team's form for game N never
sees game N's own result (or any later one).

This deliberately doesn't need ``warehouse.team_game_log`` (a full box-score
ingest) -- win/loss and runs for/against are already on ``Game`` once a game
is Final.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

_NEUTRAL = {
    "win_pct": 0.5,
    "run_diff_pg": 0.0,
    "win_pct_l10": 0.5,
    "run_diff_pg_l10": 0.0,
    "rest_days": 3.0,
    "games_played": 0.0,
}
_L10 = 10


@dataclass(frozen=True, slots=True)
class TeamFormRow:
    game_pk: int
    home_form: dict[str, float]
    away_form: dict[str, float]


class TeamFormTracker:
    """Mutable per-team game log, updated one game at a time."""

    def __init__(self) -> None:
        # per team: list of (runs_for, runs_against, won)
        self._history: dict[int, list[tuple[int, int, bool]]] = defaultdict(list)
        self._last_date: dict[int, dt.date] = {}

    def snapshot(self, team_id: int, as_of: dt.date) -> dict[str, float]:
        hist = self._history.get(team_id)
        if not hist:
            return dict(_NEUTRAL)
        n = len(hist)
        rf = [h[0] for h in hist]
        ra = [h[1] for h in hist]
        won = [1.0 if h[2] else 0.0 for h in hist]
        last10 = hist[-_L10:]
        rest = (as_of - self._last_date[team_id]).days if team_id in self._last_date else 3
        return {
            "win_pct": sum(won) / n,
            "run_diff_pg": (sum(rf) - sum(ra)) / n,
            "win_pct_l10": sum(1.0 for h in last10 if h[2]) / len(last10),
            "run_diff_pg_l10": (
                (sum(h[0] for h in last10) - sum(h[1] for h in last10)) / len(last10)
            ),
            "rest_days": float(min(rest, 10)),
            "games_played": float(n),
        }

    def process_game(
        self,
        *,
        game_pk: int,
        game_date: dt.date,
        home_team_id: int,
        away_team_id: int,
        home_score: int,
        away_score: int,
    ) -> TeamFormRow:
        row = TeamFormRow(
            game_pk=game_pk,
            home_form=self.snapshot(home_team_id, game_date),
            away_form=self.snapshot(away_team_id, game_date),
        )
        self._history[home_team_id].append((home_score, away_score, home_score > away_score))
        self._history[away_team_id].append((away_score, home_score, away_score > home_score))
        self._last_date[home_team_id] = game_date
        self._last_date[away_team_id] = game_date
        return row


def run_team_form(games: list[dict[str, Any]]) -> list[TeamFormRow]:
    """Process an iterable of game dicts in chronological order.

    Each dict needs: ``game_pk``, ``game_date`` (date or ISO str), ``home_team_id``,
    ``away_team_id``, ``home_score``, ``away_score``. Games missing a score are
    skipped (not yet played) -- same convention as ``run_elo``.
    """
    tracker = TeamFormTracker()
    ordered = sorted(
        (g for g in games if g.get("home_score") is not None and g.get("away_score") is not None),
        key=lambda g: (str(g["game_date"]), int(g["game_pk"])),
    )
    out = []
    for g in ordered:
        out.append(tracker.process_game(**_coerce(g)))
    return out


def build_tracker(games: list[dict[str, Any]]) -> TeamFormTracker:
    """Like `run_team_form`, but hands back the fully-updated tracker instead
    of per-game snapshots -- for a live slate, where the caller wants each
    team's *current* form (`tracker.snapshot(team_id, today)`), not a replay."""
    tracker = TeamFormTracker()
    ordered = sorted(
        (g for g in games if g.get("home_score") is not None and g.get("away_score") is not None),
        key=lambda g: (str(g["game_date"]), int(g["game_pk"])),
    )
    for g in ordered:
        tracker.process_game(**_coerce(g))
    return tracker


def _coerce(g: dict[str, Any]) -> dict[str, Any]:
    gd = g["game_date"]
    game_date = gd if isinstance(gd, dt.date) else dt.date.fromisoformat(str(gd))
    return {
        "game_pk": int(g["game_pk"]),
        "game_date": game_date,
        "home_team_id": int(g["home_team_id"]),
        "away_team_id": int(g["away_team_id"]),
        "home_score": int(g["home_score"]),
        "away_score": int(g["away_score"]),
    }
