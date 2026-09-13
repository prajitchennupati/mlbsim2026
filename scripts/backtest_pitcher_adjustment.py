"""Backtest: does a starting-pitcher FIP adjustment beat plain elo_v1?

Point-in-time discipline: each pitcher's stat snapshot is cumulative through
the day BEFORE their period starts -- never including games in or after that
period -- so a game's prediction never sees data from its own or a later
period. The season's first period has no prior data at all and gets
adjustment=0 for every game (identical to plain Elo).

Usage:
    .venv/bin/python scripts/backtest_pitcher_adjustment.py
"""

from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from mlbsim_core import get_logger, session_scope
from mlbsim_data.models import FINAL_STATUSES, Game, GamePrediction, GameProbable
from mlbsim_data.sources.mlb_statsapi import player_pitching_stats_range
from mlbsim_models.evaluate.metrics import accuracy, brier_score, log_loss
from mlbsim_models.ratings.elo import EloConfig, expected_home_win
from mlbsim_models.ratings.pitcher import PitcherLine, rating_adjustment

_log = get_logger(__name__)

SEASON = 2026
SEASON_START = "2026-03-15"
# Splits the season into 3 periods; period 0 (before the first bound) has no
# prior in-season data and always gets adjustment=0.
PERIOD_BOUNDS = ["2026-06-01", "2026-08-01"]
CANDIDATE_SCALES = [6.0, 10.0, 12.0, 16.0, 20.0, 25.0]  # points per FIP-run

Game_ = dict[str, Any]


def _period_of(game_date: dt.date) -> int:
    for i, bound in enumerate(PERIOD_BOUNDS):
        if game_date.isoformat() < bound:
            return i
    return len(PERIOD_BOUNDS)


def _ip_to_outs(ip: str | None) -> int:
    """MLB's innings-pitched string ("34.1") -> outs (34*3 + 1 = 103)."""
    if not ip:
        return 0
    whole, _, frac = ip.partition(".")
    return int(whole or 0) * 3 + int(frac or 0)


def _fetch_line(pitcher_id: int, end_date: str) -> PitcherLine | None:
    stat = player_pitching_stats_range(pitcher_id, SEASON_START, end_date, season=SEASON)
    if not stat:
        return None
    return PitcherLine(
        outs=_ip_to_outs(stat.get("inningsPitched")),
        home_runs=int(stat.get("homeRuns", 0)),
        walks=int(stat.get("baseOnBalls", 0)),
        hit_by_pitch=int(stat.get("hitBatsmen", 0)),
        strikeouts=int(stat.get("strikeOuts", 0)),
        batters_faced=int(stat.get("battersFaced", 0)),
    )


def _load_games() -> list[Game_]:
    with session_scope() as s:
        rows = list(
            s.execute(
                select(
                    Game.game_pk,
                    Game.game_date,
                    Game.home_team_id,
                    Game.away_team_id,
                    Game.home_score,
                    Game.away_score,
                    GamePrediction.home_win_prob,
                    GamePrediction.factors,
                )
                .join(GamePrediction, GamePrediction.game_pk == Game.game_pk)
                .where(
                    Game.season == SEASON,
                    Game.game_type == "R",
                    Game.status.in_(FINAL_STATUSES),
                    Game.home_score.is_not(None),
                    GamePrediction.model_id == "elo_v1",
                    GamePrediction.is_live.is_(False),
                )
                .order_by(Game.game_date, Game.game_pk)
            )
        )
        probable_stmt = select(
            GameProbable.game_pk, GameProbable.team_id, GameProbable.probable_pitcher_id
        )
        probables = {
            (r.game_pk, r.team_id): r.probable_pitcher_id for r in s.execute(probable_stmt)
        }

    print(f"{len(rows)} finished {SEASON} regular-season games with an elo_v1 prediction")
    games: list[Game_] = []
    for r in rows:
        factors = r.factors or {}
        if factors.get("home_rating") is None or factors.get("away_rating") is None:
            continue
        games.append(
            {
                "period": _period_of(r.game_date),
                "home_rating": factors["home_rating"],
                "away_rating": factors["away_rating"],
                "home_sp": probables.get((r.game_pk, r.home_team_id)),
                "away_sp": probables.get((r.game_pk, r.away_team_id)),
                "y": 1 if r.home_score > r.away_score else 0,
                "baseline_p": float(r.home_win_prob),
            }
        )
    return games


def _fetch_all_lines(games: list[Game_]) -> dict[tuple[int, int], PitcherLine | None]:
    needed: dict[int, set[int]] = {1: set(), 2: set()}
    for g in games:
        period = g["period"]
        if period not in needed:
            continue
        for side in ("home_sp", "away_sp"):
            if g[side]:
                needed[period].add(g[side])

    total = sum(len(v) for v in needed.values())
    print(f"fetching point-in-time pitcher snapshots: {total} calls across {len(needed)} periods")

    lines: dict[tuple[int, int], PitcherLine | None] = {}
    done = 0
    t0 = time.perf_counter()
    for period, pitcher_ids in needed.items():
        cutoff = dt.date.fromisoformat(PERIOD_BOUNDS[period - 1]) - dt.timedelta(days=1)
        for pid in sorted(pitcher_ids):
            try:
                lines[(period, pid)] = _fetch_line(pid, cutoff.isoformat())
            except Exception as exc:
                _log.warning("pitcher_fetch_failed", pitcher_id=pid, error=str(exc))
                lines[(period, pid)] = None
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{total} fetched ({time.perf_counter() - t0:.0f}s)")
            time.sleep(0.12)
    return lines


def _score(preds: list[float], ys: list[int], label: str) -> None:
    print(
        f"{label:28s} n={len(ys):4d}  acc={accuracy(ys, preds):.4f}  "
        f"brier={brier_score(ys, preds):.5f}  log_loss={log_loss(ys, preds):.5f}"
    )


def _adjusted_preds(
    games: list[Game_],
    lines: dict[tuple[int, int], PitcherLine | None],
    scale: float,
    cfg: EloConfig,
) -> list[float]:
    preds = []
    for g in games:
        home_line = lines.get((g["period"], g["home_sp"])) if g["home_sp"] else None
        away_line = lines.get((g["period"], g["away_sp"])) if g["away_sp"] else None
        home_adj = rating_adjustment(home_line, points_per_fip_run=scale)
        away_adj = rating_adjustment(away_line, points_per_fip_run=scale)
        p = expected_home_win(g["home_rating"] + home_adj, g["away_rating"] + away_adj, cfg)
        preds.append(p)
    return preds


def main() -> None:
    games = _load_games()
    lines = _fetch_all_lines(games)
    cfg = EloConfig()

    ys_all = [g["y"] for g in games]
    print("\n--- full season ---")
    _score([g["baseline_p"] for g in games], ys_all, "baseline (elo_v1, stored)")

    adj_games = [g for g in games if g["period"] > 0]
    ys_adj = [g["y"] for g in adj_games]
    print(f"\n--- periods with pitcher data only (after {PERIOD_BOUNDS[0]}) ---")
    _score([g["baseline_p"] for g in adj_games], ys_adj, "baseline (elo_v1, stored)")

    print()
    for scale in CANDIDATE_SCALES:
        preds = _adjusted_preds(adj_games, lines, scale, cfg)
        _score(preds, ys_adj, f"pitcher-adjusted (scale={scale:g})")


if __name__ == "__main__":
    main()
