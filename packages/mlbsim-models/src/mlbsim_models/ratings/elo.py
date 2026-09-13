"""Elo team ratings — the leakage-proof W/L baseline.

Design follows FiveThirtyEight's MLB Elo: a low base ``K``, a home-field bump in
rating points, a margin-of-victory multiplier that is dampened for lopsided
favorites (so blowouts by strong teams do not over-inflate), and season-to-season
reversion toward the mean.

Everything here is pure and processes games in strict chronological order, so a
rating "as of" a game only ever reflects earlier games.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

DEFAULT_RATING = 1500.0


@dataclass(frozen=True, slots=True)
class EloConfig:
    k: float = 4.0
    home_field: float = 24.0  # ~ .545 win prob for two equal teams
    scale: float = 400.0
    mov_enabled: bool = True
    season_revert: float = 0.33  # fraction pulled back to the mean between seasons


@dataclass(frozen=True, slots=True)
class EloResult:
    """Pre-game snapshot for one game plus the post-game ratings."""

    game_pk: int
    game_date: str
    season: int
    home_team_id: int
    away_team_id: int
    home_rating_pre: float
    away_rating_pre: float
    home_win_prob: float
    home_rating_post: float
    away_rating_post: float


def expected_home_win(home_rating: float, away_rating: float, cfg: EloConfig) -> float:
    """Logistic expectation that the home team wins, including home-field advantage."""
    diff = (home_rating + cfg.home_field) - away_rating
    return float(1.0 / (1.0 + 10.0 ** (-diff / cfg.scale)))


def mov_multiplier(run_diff: int, winner_rating_edge: float) -> float:
    """Margin-of-victory multiplier, dampened when the winner was already favored.

    ``winner_rating_edge`` is the winner's pre-game rating advantage (including
    home field), which shrinks the multiplier for expected blowouts.
    """
    margin = abs(run_diff)
    if margin == 0:
        return 1.0
    return math.log(margin + 1.0) * (2.2 / (0.001 * winner_rating_edge + 2.2))


class EloRatings:
    """Mutable table of team ratings, updated one game at a time."""

    def __init__(self, cfg: EloConfig | None = None, *, initial: float = DEFAULT_RATING) -> None:
        self.cfg = cfg or EloConfig()
        self._initial = initial
        self._r: dict[int, float] = {}

    def rating(self, team_id: int) -> float:
        return self._r.get(team_id, self._initial)

    def revert_to_mean(self, mean: float = DEFAULT_RATING) -> None:
        f = self.cfg.season_revert
        for tid, r in self._r.items():
            self._r[tid] = r + f * (mean - r)

    def process_game(
        self,
        *,
        game_pk: int,
        game_date: str,
        season: int,
        home_team_id: int,
        away_team_id: int,
        home_score: int,
        away_score: int,
    ) -> EloResult:
        cfg = self.cfg
        rh, ra = self.rating(home_team_id), self.rating(away_team_id)
        exp_home = expected_home_win(rh, ra, cfg)
        home_won = home_score > away_score
        s_home = 1.0 if home_won else 0.0

        mult = 1.0
        if cfg.mov_enabled and home_score != away_score:
            edge = (rh + cfg.home_field) - ra if home_won else ra - (rh + cfg.home_field)
            mult = mov_multiplier(home_score - away_score, edge)

        delta = cfg.k * mult * (s_home - exp_home)
        rh_post, ra_post = rh + delta, ra - delta
        self._r[home_team_id] = rh_post
        self._r[away_team_id] = ra_post

        return EloResult(
            game_pk=game_pk,
            game_date=game_date,
            season=season,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_rating_pre=rh,
            away_rating_pre=ra,
            home_win_prob=exp_home,
            home_rating_post=rh_post,
            away_rating_post=ra_post,
        )


def run_elo(games: Iterable[dict[str, Any]], cfg: EloConfig | None = None) -> list[EloResult]:
    """Process an iterable of game dicts in chronological order.

    Each game dict needs: ``game_pk``, ``game_date`` (ISO str), ``season``,
    ``home_team_id``, ``away_team_id``, ``home_score``, ``away_score``. Games with
    a missing score are skipped (not yet played).
    """
    cfg = cfg or EloConfig()
    table = EloRatings(cfg)
    ordered = sorted(
        (g for g in games if g.get("home_score") is not None and g.get("away_score") is not None),
        key=lambda g: (str(g["game_date"]), int(g["game_pk"])),
    )
    results: list[EloResult] = []
    prev_season: int | None = None
    for g in ordered:
        season = int(g["season"])
        if prev_season is not None and season != prev_season:
            table.revert_to_mean()
        prev_season = season
        results.append(
            table.process_game(
                game_pk=int(g["game_pk"]),
                game_date=str(g["game_date"]),
                season=season,
                home_team_id=int(g["home_team_id"]),
                away_team_id=int(g["away_team_id"]),
                home_score=int(g["home_score"]),
                away_score=int(g["away_score"]),
            )
        )
    return results
