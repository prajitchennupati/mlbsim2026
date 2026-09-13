"""Parse the Stats API ``/v1.1/game/{pk}/feed/live`` document.

Produces warehouse-ready rows for ``games``, ``parks``, ``teams``, ``players``,
``game_weather``, ``plate_appearances``, ``pitches``, and ``batted_balls``.

Base-out state is reconstructed by replaying every play (including non-atBat
actions such as steals and pickoffs) in order; PA rows are emitted only for
``result.type == 'atBat'``.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from mlbsim_data.parse.common import (
    base_mask,
    division_code,
    league_code,
    parse_wind,
    to_date,
    to_datetime,
    to_float,
    to_int,
)

_SAC_EVENTS = {"sac_fly", "sac_bunt", "sac_fly_double_play", "sac_bunt_double_play"}

# Baserunning / administrative events that GUMBO can tag ``result.type == "atBat"``
# but which are not plate-appearance outcomes (they don't end the batter's PA).
_NON_PA_EVENTS = {
    "stolen_base_2b",
    "stolen_base_3b",
    "stolen_base_home",
    "stolen_base",
    "caught_stealing_2b",
    "caught_stealing_3b",
    "caught_stealing_home",
    "caught_stealing",
    "pickoff_1b",
    "pickoff_2b",
    "pickoff_3b",
    "pickoff",
    "pickoff_caught_stealing_2b",
    "pickoff_caught_stealing_3b",
    "pickoff_caught_stealing_home",
    "pickoff_error_1b",
    "pickoff_error_2b",
    "pickoff_error_3b",
    "wild_pitch",
    "passed_ball",
    "balk",
    "defensive_indiff",
    "other_advance",
    "runner_double_play",
    "cs_double_play",
    "stolen_base_double_play",
    "defensive_switch",
    "runner_placed",
    "pitching_substitution",
    "offensive_substitution",
    "defensive_substitution",
    "pitcher_switch",
    "game_advisory",
    "ejection",
}


@dataclass(slots=True)
class FeedParse:
    game: dict[str, Any]
    park: dict[str, Any] | None = None  # detailed row for the game venue
    parks: list[dict[str, Any]] = field(default_factory=list)  # every referenced venue (FK targets)
    teams: list[dict[str, Any]] = field(default_factory=list)
    players: list[dict[str, Any]] = field(default_factory=list)
    weather: dict[str, Any] | None = None
    plate_appearances: list[dict[str, Any]] = field(default_factory=list)
    pitches: list[dict[str, Any]] = field(default_factory=list)
    batted_balls: list[dict[str, Any]] = field(default_factory=list)


def _all_park_rows(game_data: dict[str, Any]) -> list[dict[str, Any]]:
    """One uniform row per venue the game/teams reference, so team+game FKs resolve.

    The game venue carries full detail; each team's home venue is id+name only
    (rendered through the same shape with ``None`` for the missing fields).
    """
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for venue in (
        game_data.get("venue", {}) or {},
        *((node.get("venue") or {}) for node in game_data.get("teams", {}).values()),
    ):
        row = _park_row(venue)
        if row and row["park_id"] not in seen:
            seen.add(row["park_id"])
            rows.append(row)
    return rows


def _team_rows(game_data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for node in game_data.get("teams", {}).values():
        tid = to_int(node.get("id"))
        if tid is None:
            continue
        rows.append(
            {
                "team_id": tid,
                "abbr": node.get("abbreviation") or str(tid),
                "name": node.get("name") or "",
                "league": league_code((node.get("league") or {}).get("name")),
                "division": division_code((node.get("division") or {}).get("name")),
                "park_id": to_int((node.get("venue") or {}).get("id")),
            }
        )
    return rows


def _park_row(venue: dict[str, Any]) -> dict[str, Any] | None:
    pid = to_int(venue.get("id"))
    if pid is None:
        return None
    loc = venue.get("location", {}) or {}
    coords = loc.get("defaultCoordinates", {}) or {}
    fi = venue.get("fieldInfo", {}) or {}
    return {
        "park_id": pid,
        "name": venue.get("name") or "",
        "lat": to_float(coords.get("latitude")),
        "lon": to_float(coords.get("longitude")),
        "altitude_ft": to_int(loc.get("elevation")),
        "roof_type": (fi.get("roofType") or "").lower() or None,
        "orientation_deg": to_float(loc.get("azimuthAngle")),
        "lf_dist": to_int(fi.get("leftLine")),
        "cf_dist": to_int(fi.get("center")),
        "rf_dist": to_int(fi.get("rightLine")),
    }


def _player_rows(game_data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for node in game_data.get("players", {}).values():
        pid = to_int(node.get("id"))
        if pid is None:
            continue
        rows.append(
            {
                "player_id": pid,
                "full_name": node.get("fullName") or "",
                "bats": (node.get("batSide") or {}).get("code"),
                "throws": (node.get("pitchHand") or {}).get("code"),
                "birth_date": to_date(node.get("birthDate")),
                "primary_pos": (node.get("primaryPosition") or {}).get("abbreviation"),
                "debut_date": to_date(node.get("mlbDebutDate")),
            }
        )
    return rows


def _game_row(feed: dict[str, Any]) -> dict[str, Any]:
    gd = feed["gameData"]
    ls = feed["liveData"].get("linescore", {}) or {}
    ls_teams = ls.get("teams", {}) or {}
    decisions = feed["liveData"].get("decisions", {}) or {}
    game = gd.get("game", {})
    dtinfo = gd.get("datetime", {})
    home_id = to_int(gd["teams"]["home"]["id"])
    away_id = to_int(gd["teams"]["away"]["id"])
    game_date = to_date(dtinfo.get("officialDate") or dtinfo.get("originalDate"))
    return {
        "game_pk": to_int(feed.get("gamePk") or game.get("pk")),
        "season": to_int(game.get("season")) or (game_date.year if game_date else None),
        "game_date": game_date,
        "game_type": game.get("type", "R"),
        "status": gd.get("status", {}).get("detailedState", "Unknown")[:24],
        "home_team_id": home_id,
        "away_team_id": away_id,
        "park_id": to_int(gd.get("venue", {}).get("id")),
        "scheduled_start_utc": to_datetime(dtinfo.get("dateTime")),
        "day_night": dtinfo.get("dayNight"),
        "is_doubleheader": game.get("doubleHeader", "N") != "N",
        "dh_game_num": to_int(game.get("gameNumber")) or 1,
        "scheduled_innings": to_int(ls.get("scheduledInnings")) or 9,
        "home_score": to_int((ls_teams.get("home") or {}).get("runs")),
        "away_score": to_int((ls_teams.get("away") or {}).get("runs")),
        "n_innings": to_int(ls.get("currentInning")),
        "winning_pitcher_id": to_int((decisions.get("winner") or {}).get("id")),
        "losing_pitcher_id": to_int((decisions.get("loser") or {}).get("id")),
        "save_pitcher_id": to_int((decisions.get("save") or {}).get("id")),
    }


def _weather_row(
    game_pk: int | None, gd: dict[str, Any], now: dt.datetime
) -> dict[str, Any] | None:
    w = gd.get("weather") or {}
    if not w or game_pk is None:
        return None
    speed, direction = parse_wind(w.get("wind"))
    return {
        "game_pk": game_pk,
        "forecast_ts": now,
        "source": "statsapi",
        "temp_f": to_int(w.get("temp")),
        "wind_mph": speed,
        "wind_dir": direction,
        "condition": w.get("condition"),
    }


def _count_pitches(play: dict[str, Any]) -> int:
    return sum(
        1 for e in play.get("playEvents", []) if e.get("isPitch") or e.get("type") == "pitch"
    )


def _pitch_rows(
    play: dict[str, Any], pa_id: int, game_pk: int, season: int | None
) -> list[dict[str, Any]]:
    rows = []
    seq = 0
    for e in play.get("playEvents", []):
        if not (e.get("isPitch") or e.get("type") == "pitch"):
            continue
        seq += 1
        d = e.get("details", {}) or {}
        pd = e.get("pitchData", {}) or {}
        coords = pd.get("coordinates", {}) or {}
        breaks = pd.get("breaks", {}) or {}
        cnt = e.get("count", {}) or {}
        rows.append(
            {
                "pitch_id": pa_id * 100 + seq,
                "pa_id": pa_id,
                "game_pk": game_pk,
                "season": season,
                "seq": seq,
                "balls_after": to_int(cnt.get("balls")),
                "strikes_after": to_int(cnt.get("strikes")),
                "pitch_type": (d.get("type") or {}).get("code"),
                "description": d.get("description"),
                "call_code": (d.get("call") or {}).get("code"),
                "is_in_play": bool(d.get("isInPlay")),
                "release_speed": to_float(pd.get("startSpeed")),
                "end_speed": to_float(pd.get("endSpeed")),
                "spin_rate": to_int(breaks.get("spinRate")),
                "spin_direction": to_int(breaks.get("spinDirection")),
                "extension": to_float(pd.get("extension")),
                "plate_x": to_float(coords.get("pX")),
                "plate_z": to_float(coords.get("pZ")),
                "zone": to_int(pd.get("zone")),
            }
        )
    return rows


def _batted_ball_row(play: dict[str, Any], pa_id: int) -> dict[str, Any] | None:
    for e in play.get("playEvents", []):
        hd = e.get("hitData")
        if hd:
            coords = hd.get("coordinates", {}) or {}
            return {
                "pa_id": pa_id,
                "launch_speed": to_float(hd.get("launchSpeed")),
                "launch_angle": to_float(hd.get("launchAngle")),
                "total_distance": to_int(hd.get("totalDistance")),
                "trajectory": hd.get("trajectory"),
                "hardness": hd.get("hardness"),
                "hit_x": to_float(coords.get("coordX")),
                "hit_y": to_float(coords.get("coordY")),
            }
    return None


def parse_game_feed(
    feed: dict[str, Any],
    *,
    source_ts: dt.datetime | None = None,
    batting_order_by_player: dict[int, int] | None = None,
) -> FeedParse:
    now = source_ts or dt.datetime.now(dt.UTC)
    gd = feed["gameData"]
    game = _game_row(feed)
    game_pk = game["game_pk"]
    season = game["season"]
    home_id, away_id = game["home_team_id"], game["away_team_id"]
    order_map = batting_order_by_player or {}

    result = FeedParse(
        game=game,
        park=_park_row(gd.get("venue", {}) or {}),
        parks=_all_park_rows(gd),
        teams=_team_rows(gd),
        players=_player_rows(gd),
        weather=_weather_row(game_pk, gd, now),
    )

    outs_running: dict[tuple[int, str], int] = {}
    base_end_running: dict[tuple[int, str], int] = {}
    tto: dict[tuple[int, int], int] = {}
    pitch_count: dict[int, int] = {}

    for play in feed["liveData"]["plays"].get("allPlays", []):
        about = play.get("about", {})
        matchup = play.get("matchup", {})
        runners = play.get("runners", [])
        res = play.get("result", {})
        inning = to_int(about.get("inning")) or 0
        half = about.get("halfInning") or "top"
        key = (inning, half)

        start_outs = outs_running.get(key, 0)
        start_bases = base_end_running.get(key, 0)
        outs_made = sum(1 for r in runners if (r.get("movement") or {}).get("isOut"))
        if res.get("isOut") and outs_made == 0:
            outs_made = 1
        end_bases = base_mask(
            on_first=matchup.get("postOnFirst"),
            on_second=matchup.get("postOnSecond"),
            on_third=matchup.get("postOnThird"),
        )
        end_outs = min(3, start_outs + outs_made)

        if res.get("type") == "atBat" and (res.get("eventType") or "") not in _NON_PA_EVENTS:
            at_bat_index = to_int(about.get("atBatIndex")) or 0
            pa_id = game_pk * 1000 + at_bat_index
            batter_id = to_int((matchup.get("batter") or {}).get("id"))
            pitcher_id = to_int((matchup.get("pitcher") or {}).get("id"))
            n_pitches = _count_pitches(play)
            if pitcher_id is not None:
                pitch_count[pitcher_id] = pitch_count.get(pitcher_id, 0) + n_pitches
            tto_key = (pitcher_id or -1, batter_id or -1)
            tto[tto_key] = tto.get(tto_key, 0) + 1
            event_type = res.get("eventType") or "unknown"

            result.plate_appearances.append(
                {
                    "pa_id": pa_id,
                    "game_pk": game_pk,
                    "season": season,
                    "at_bat_index": at_bat_index,
                    "inning": inning,
                    "half": half,
                    "batter_id": batter_id,
                    "pitcher_id": pitcher_id,
                    "batting_team_id": away_id if half == "top" else home_id,
                    "bat_side": (matchup.get("batSide") or {}).get("code"),
                    "pitch_hand": (matchup.get("pitchHand") or {}).get("code"),
                    "batting_order": order_map.get(batter_id) if batter_id else None,
                    "outs_start": start_outs,
                    "outs_end": end_outs,
                    "base_state_start": start_bases,
                    "base_state_end": end_bases,
                    "event_type": event_type,
                    "event": res.get("event"),
                    "description": (res.get("description") or "")[:400] or None,
                    "is_out": bool(res.get("isOut")),
                    "is_sac": event_type in _SAC_EVENTS,
                    "rbi": to_int(res.get("rbi")) or 0,
                    "runs_on_play": sum(
                        1 for r in runners if (r.get("movement") or {}).get("end") == "score"
                    ),
                    "away_score_after": to_int(res.get("awayScore")) or 0,
                    "home_score_after": to_int(res.get("homeScore")) or 0,
                    "times_through_order": tto[tto_key],
                    "pitcher_pitch_no": pitch_count.get(pitcher_id) if pitcher_id else None,
                    "leverage_index": None,
                    "wpa": None,
                }
            )
            result.pitches.extend(_pitch_rows(play, pa_id, game_pk, season))
            bb = _batted_ball_row(play, pa_id)
            if bb is not None:
                result.batted_balls.append(bb)

        base_end_running[key] = end_bases
        outs_running[key] = end_outs

    return result
