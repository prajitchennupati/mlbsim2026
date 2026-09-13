"""Starting-pitcher quality as a bounded nudge on top of team Elo.

Team Elo alone has no idea who's on the mound — an ace and a fifth starter
get the exact same team rating. A starter accounts for roughly 55-65% of a
game's outs, so this isn't a second model, just a small adjustment to the
two teams' effective ratings for one game.

FIP (fielding-independent pitching) isolates what a pitcher directly
controls — strikeouts, walks, HBP, home runs — instead of balls in play,
which is the same reasoning `mlbsim_features.build.pitcher_form` already
uses for its point-in-time per-start feature. The formula and constant here
match that one for consistency.
"""

from __future__ import annotations

from dataclasses import dataclass

from mlbsim_data.sources.mlb_statsapi import player_pitching_stats_range

FIP_CONSTANT = 3.10  # matches mlbsim_features.build.pitcher_form
LEAGUE_AVG_FIP = 4.20  # fixed reference point, not fit to any one sample
MIN_BATTERS_FACED = 30  # below this a cumulative FIP is too noisy to trust
MAX_ADJUSTMENT = 40.0  # rating points; caps an extreme small-sample FIP


@dataclass(frozen=True, slots=True)
class PitcherLine:
    outs: int
    home_runs: int
    walks: int
    hit_by_pitch: int
    strikeouts: int
    batters_faced: int


def fip(line: PitcherLine, *, constant: float = FIP_CONSTANT) -> float:
    ip = line.outs / 3 or 1e-9
    walks_hbp = line.walks + line.hit_by_pitch
    return (13 * line.home_runs + 3 * walks_hbp - 2 * line.strikeouts) / ip + constant


def rating_adjustment(
    line: PitcherLine | None,
    *,
    points_per_fip_run: float,
    league_avg_fip: float = LEAGUE_AVG_FIP,
) -> float:
    """Elo-scale points to add to a team's rating for a game this pitcher starts.

    Positive means better than league average. Returns 0 for a pitcher with no
    (or too little) prior-to-this-game data — the neutral, no-information case.
    """
    if line is None or line.batters_faced < MIN_BATTERS_FACED:
        return 0.0
    delta_runs = league_avg_fip - fip(line)
    points = delta_runs * points_per_fip_run
    return max(-MAX_ADJUSTMENT, min(MAX_ADJUSTMENT, points))


def _ip_to_outs(ip: str | None) -> int:
    """MLB's innings-pitched string ("34.1") -> outs (34*3 + 1 = 103)."""
    if not ip:
        return 0
    whole, _, frac = ip.partition(".")
    return int(whole or 0) * 3 + int(frac or 0)


def fetch_pitcher_line(
    pitcher_id: int, season: int, season_start: str, cutoff: str
) -> PitcherLine | None:
    """A starter's cumulative line over [season_start, cutoff] — a single cheap
    Stats API call (MLB aggregates it server-side), not a box-score ingest.
    Shared by elo_sp and the GBM/MLP feature builder so both use the exact
    same point-in-time pitcher signal."""
    stat = player_pitching_stats_range(pitcher_id, season_start, cutoff, season=season)
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
