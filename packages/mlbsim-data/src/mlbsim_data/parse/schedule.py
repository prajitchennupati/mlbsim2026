"""Parse Stats API ``/v1/schedule`` game objects."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from mlbsim_data.parse.common import (
    division_code,
    league_code,
    parse_wind,
    to_date,
    to_datetime,
    to_int,
)


@dataclass(slots=True)
class ScheduleParse:
    games: list[dict[str, Any]] = field(default_factory=list)
    teams: list[dict[str, Any]] = field(default_factory=list)
    parks: list[dict[str, Any]] = field(default_factory=list)
    players: list[dict[str, Any]] = field(default_factory=list)  # probable-pitcher stubs
    weather: list[dict[str, Any]] = field(default_factory=list)
    probables: list[dict[str, Any]] = field(default_factory=list)


def _team_row(node: dict[str, Any]) -> dict[str, Any] | None:
    team = node.get("team", {})
    tid = to_int(team.get("id"))
    if tid is None:
        return None
    return {
        "team_id": tid,
        # postseason placeholder rows ("NL 4/5 Winner" etc.) can exceed the column widths
        "abbr": (team.get("abbreviation") or team.get("teamCode") or str(tid))[:4],
        "name": (team.get("name") or team.get("teamName") or "")[:80],
        "league": league_code((team.get("league") or {}).get("name")),
        "division": division_code((team.get("division") or {}).get("name")),
        "park_id": to_int((team.get("venue") or {}).get("id")),
    }


def parse_schedule(
    games: list[dict[str, Any]], *, source_ts: dt.datetime | None = None
) -> ScheduleParse:
    """Convert raw schedule game objects into warehouse-ready row dicts."""
    now = source_ts or dt.datetime.now(dt.UTC)
    out = ScheduleParse()
    seen_teams: set[int] = set()
    seen_parks: set[int] = set()
    seen_games: set[int] = set()
    seen_players: set[int] = set()

    def _note_park(venue: dict[str, Any] | None) -> None:
        pid = to_int((venue or {}).get("id"))
        if pid is not None and pid not in seen_parks:
            seen_parks.add(pid)
            out.parks.append({"park_id": pid, "name": (venue or {}).get("name") or ""})

    for g in games:
        game_pk = to_int(g.get("gamePk"))
        if game_pk is None or game_pk in seen_games:
            continue  # the API can list a suspended/resumed game under two dates
        seen_games.add(game_pk)
        home, away = g["teams"]["home"], g["teams"]["away"]
        home_id, away_id = to_int(home["team"]["id"]), to_int(away["team"]["id"])
        _note_park(g.get("venue"))
        status = g.get("status", {})
        dh = g.get("doubleHeader", "N")
        game_date = to_date(g.get("officialDate") or g.get("gameDate"))
        season = to_int(g.get("season")) or (game_date.year if game_date else None)

        out.games.append(
            {
                "game_pk": game_pk,
                "season": season,
                "game_date": game_date,
                "game_type": g.get("gameType", "R"),
                "status": status.get("detailedState", "Unknown")[:24],
                "home_team_id": home_id,
                "away_team_id": away_id,
                "park_id": to_int((g.get("venue") or {}).get("id")),
                "scheduled_start_utc": to_datetime(g.get("gameDate")),
                "day_night": g.get("dayNight"),
                "is_doubleheader": dh != "N",
                "dh_game_num": to_int(g.get("gameNumber")) or 1,
                "scheduled_innings": to_int(g.get("scheduledInnings")) or 9,
                "home_score": to_int(home.get("score")),
                "away_score": to_int(away.get("score")),
            }
        )

        for node, tid in ((home, home_id), (away, away_id)):
            row = _team_row(node)
            if row and tid is not None and tid not in seen_teams:
                seen_teams.add(tid)
                out.teams.append(row)
                _note_park((node.get("team") or {}).get("venue"))
            prob = node.get("probablePitcher") or {}
            prob_id = to_int(prob.get("id"))
            if prob_id is not None:
                out.probables.append(
                    {
                        "game_pk": game_pk,
                        "team_id": tid,
                        "probable_pitcher_id": prob_id,
                        "source_ts": now,
                    }
                )
                if prob_id not in seen_players:
                    seen_players.add(prob_id)
                    out.players.append(
                        {"player_id": prob_id, "full_name": prob.get("fullName") or ""}
                    )

        w = g.get("weather") or {}
        if w:
            speed, direction = parse_wind(w.get("wind"))
            out.weather.append(
                {
                    "game_pk": game_pk,
                    "forecast_ts": now,
                    "source": "statsapi",
                    "temp_f": to_int(w.get("temp")),
                    "wind_mph": speed,
                    "wind_dir": direction,
                    "condition": w.get("condition"),
                }
            )

    return out
