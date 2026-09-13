"""Parse the Stats API ``/v1/game/{pk}/boxscore`` document into game logs + lineups."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from mlbsim_data.parse.common import innings_to_outs, to_int

_BAT = "batting"
_PIT = "pitching"


@dataclass(slots=True)
class BoxscoreParse:
    batting_logs: list[dict[str, Any]] = field(default_factory=list)
    pitching_logs: list[dict[str, Any]] = field(default_factory=list)
    team_logs: list[dict[str, Any]] = field(default_factory=list)
    lineups: list[dict[str, Any]] = field(default_factory=list)
    starters: dict[str, int | None] = field(default_factory=dict)

    def batting_order_by_player(self) -> dict[int, int]:
        """Map ``player_id -> batting slot (1..9)`` across both lineups."""
        return {r["player_id"]: r["batting_order"] for r in self.lineups}


def _batting_log(
    pid: int, node: dict[str, Any], *, game_pk: int, season: int, team_id: int
) -> dict[str, Any] | None:
    b = (node.get("stats") or {}).get(_BAT) or {}
    if "atBats" not in b and "plateAppearances" not in b:
        return None
    bo = to_int(node.get("battingOrder"))
    return {
        "player_id": pid,
        "game_pk": game_pk,
        "team_id": team_id,
        "season": season,
        "batting_order": bo // 100 if bo else None,
        "pa": to_int(b.get("plateAppearances")) or 0,
        "ab": to_int(b.get("atBats")) or 0,
        "r": to_int(b.get("runs")) or 0,
        "h": to_int(b.get("hits")) or 0,
        "d2b": to_int(b.get("doubles")) or 0,
        "t3b": to_int(b.get("triples")) or 0,
        "hr": to_int(b.get("homeRuns")) or 0,
        "rbi": to_int(b.get("rbi")) or 0,
        "bb": to_int(b.get("baseOnBalls")) or 0,
        "ibb": to_int(b.get("intentionalWalks")) or 0,
        "hbp": to_int(b.get("hitByPitch")) or 0,
        "so": to_int(b.get("strikeOuts")) or 0,
        "sb": to_int(b.get("stolenBases")) or 0,
        "cs": to_int(b.get("caughtStealing")) or 0,
        "sac": to_int(b.get("sacBunts")) or 0,
        "sf": to_int(b.get("sacFlies")) or 0,
        "gidp": to_int(b.get("groundIntoDoublePlay")) or 0,
        "tb": to_int(b.get("totalBases")) or 0,
    }


def _pitching_log(
    pid: int, node: dict[str, Any], *, game_pk: int, season: int, team_id: int, is_starter: bool
) -> dict[str, Any] | None:
    p = (node.get("stats") or {}).get(_PIT) or {}
    if "battersFaced" not in p and "outs" not in p:
        return None
    outs = to_int(p.get("outs"))
    if outs is None:
        outs = innings_to_outs(p.get("inningsPitched"))
    er = to_int(p.get("earnedRuns")) or 0
    is_start = to_int(p.get("gamesStarted")) == 1 or is_starter
    return {
        "player_id": pid,
        "game_pk": game_pk,
        "team_id": team_id,
        "season": season,
        "is_start": is_start,
        "bf": to_int(p.get("battersFaced")) or 0,
        "outs": outs or 0,
        "h": to_int(p.get("hits")) or 0,
        "r": to_int(p.get("runs")) or 0,
        "er": er,
        "bb": to_int(p.get("baseOnBalls")) or 0,
        "ibb": to_int(p.get("intentionalWalks")) or 0,
        "hbp": to_int(p.get("hitBatsmen")) or 0,
        "so": to_int(p.get("strikeOuts")) or 0,
        "hr": to_int(p.get("homeRuns")) or 0,
        "pitches": to_int(p.get("numberOfPitches") or p.get("pitchesThrown")) or 0,
        "strikes": to_int(p.get("strikes")) or 0,
        "got_win": (to_int(p.get("wins")) or 0) >= 1,
        "got_loss": (to_int(p.get("losses")) or 0) >= 1,
        "got_save": (to_int(p.get("saves")) or 0) >= 1,
        "quality_start": bool(is_start and (outs or 0) >= 18 and er <= 3),
    }


def parse_boxscore(
    box: dict[str, Any],
    *,
    game_pk: int,
    season: int,
    source_ts: dt.datetime | None = None,
) -> BoxscoreParse:
    now = source_ts or dt.datetime.now(dt.UTC)
    out = BoxscoreParse()
    sides = box.get("teams", {})

    side_team = {s: to_int(sides.get(s, {}).get("team", {}).get("id")) for s in ("home", "away")}

    for side in ("home", "away"):
        node = sides.get(side, {})
        team_id = side_team[side]
        opp_id = side_team["away" if side == "home" else "home"]
        if team_id is None:
            continue
        players = node.get("players", {}) or {}
        pitcher_ids = [to_int(x) for x in node.get("pitchers", []) or []]
        starter = pitcher_ids[0] if pitcher_ids else None
        out.starters[side] = starter

        for slot, raw_pid in enumerate(node.get("battingOrder", []) or [], start=1):
            pid = to_int(raw_pid)
            if pid is None:
                continue
            pnode = players.get(f"ID{pid}", {})
            out.lineups.append(
                {
                    "game_pk": game_pk,
                    "team_id": team_id,
                    "batting_order": slot,
                    "slot_sequence": 0,
                    "player_id": pid,
                    "position": (pnode.get("position") or {}).get("abbreviation"),
                    "source": "actual",
                    "source_ts": now,
                }
            )

        for pkey, pnode in players.items():
            pid = to_int(pnode.get("person", {}).get("id")) or to_int(pkey.removeprefix("ID"))
            if pid is None:
                continue
            blog = _batting_log(pid, pnode, game_pk=game_pk, season=season, team_id=team_id)
            if blog:
                out.batting_logs.append(blog)
            plog = _pitching_log(
                pid,
                pnode,
                game_pk=game_pk,
                season=season,
                team_id=team_id,
                is_starter=(pid == starter),
            )
            if plog:
                out.pitching_logs.append(plog)

        ts = node.get("teamStats", {}) or {}
        bat, pit = ts.get(_BAT, {}) or {}, ts.get(_PIT, {}) or {}
        runs_for = to_int(bat.get("runs")) or 0
        runs_against = to_int(pit.get("runs")) or 0
        out.team_logs.append(
            {
                "team_id": team_id,
                "game_pk": game_pk,
                "season": season,
                "opponent_id": opp_id,
                "is_home": side == "home",
                "runs_for": runs_for,
                "runs_against": runs_against,
                "hits_for": to_int(bat.get("hits")) or 0,
                "hits_against": to_int(pit.get("hits")) or 0,
                "won": None if runs_for == runs_against else runs_for > runs_against,
            }
        )

    # A person id can appear under both sides' ``players`` maps (stale roster
    # entries, DFA'd players); keep the row with real activity so the warehouse
    # upsert doesn't hit ON CONFLICT twice for one (game_pk, player_id).
    out.batting_logs = _dedupe_by_player(out.batting_logs, "pa")
    out.pitching_logs = _dedupe_by_player(out.pitching_logs, "bf")
    return out


def _dedupe_by_player(rows: list[dict[str, Any]], activity_key: str) -> list[dict[str, Any]]:
    best: dict[int, dict[str, Any]] = {}
    for r in rows:
        pid = r["player_id"]
        cur = best.get(pid)
        if cur is None or (r.get(activity_key) or 0) > (cur.get(activity_key) or 0):
            best[pid] = r
    return list(best.values())
