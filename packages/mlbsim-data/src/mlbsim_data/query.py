"""Read-side helpers for inspecting ingested data (backs ``mlbsim query``)."""

from __future__ import annotations

from collections import Counter
from typing import Any

from sqlalchemy import func, select

from mlbsim_core import session_scope
from mlbsim_data.models import Game, Pitch, PlateAppearance, Team


def describe_game(game_pk: int) -> dict[str, Any]:
    """Return a compact summary of one ingested game and its plate appearances."""
    with session_scope() as s:
        game = s.get(Game, game_pk)
        if game is None:
            raise LookupError(f"game {game_pk} not ingested")

        team_name: dict[int, str] = dict(
            s.execute(select(Team.team_id, Team.name).order_by(Team.team_id)).tuples().all()
        )
        pas = list(
            s.scalars(
                select(PlateAppearance)
                .where(PlateAppearance.game_pk == game_pk)
                .order_by(PlateAppearance.at_bat_index)
            )
        )
        n_pitches = s.scalar(
            select(func.count()).select_from(Pitch).where(Pitch.game_pk == game_pk)
        )
        events = Counter(pa.event_type for pa in pas)

        return {
            "game_pk": game_pk,
            "date": game.game_date.isoformat() if game.game_date else None,
            "status": game.status,
            "matchup": (
                f"{team_name.get(game.away_team_id, game.away_team_id)} "
                f"({game.away_score}) @ "
                f"{team_name.get(game.home_team_id, game.home_team_id)} "
                f"({game.home_score})"
            ),
            "innings": game.n_innings,
            "plate_appearances": len(pas),
            "pitches": int(n_pitches or 0),
            "event_breakdown": dict(events.most_common()),
            "first_pa": _pa_line(pas[0], team_name) if pas else None,
            "last_pa": _pa_line(pas[-1], team_name) if pas else None,
        }


def _pa_line(pa: PlateAppearance, _team_name: dict[int, str]) -> str:
    return (
        f"inn {pa.inning} {pa.half} | outs {pa.outs_start}->{pa.outs_end} | "
        f"bases {pa.base_state_start:03b}->{pa.base_state_end:03b} | "
        f"{pa.event_type} (rbi {pa.rbi}, runs {pa.runs_on_play})"
    )


def render_game(game_pk: int) -> str:
    d = describe_game(game_pk)
    lines = [
        f"game {d['game_pk']}  {d['date']}  [{d['status']}]",
        f"  {d['matchup']}   innings={d['innings']}",
        f"  plate_appearances={d['plate_appearances']}  pitches={d['pitches']}",
        f"  events: {d['event_breakdown']}",
        f"  first: {d['first_pa']}",
        f"  last:  {d['last_pa']}",
    ]
    return "\n".join(lines)
