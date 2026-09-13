"""Vectorised remaining-season Monte Carlo.

Given current standings and a home-win probability for every remaining game, draw
each game's winner, resolve seeding per league, run the postseason bracket, and
tally per-team probabilities. Chunked over ``n_sims`` so 100k seasons fit in
memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mlbsim_engine.playoffs import seed_league, simulate_bracket

FloatArr = NDArray[np.float64]
IntArr = NDArray[np.int64]

_N_TEAMS = 30
_MAX_WINS = 163
_CHUNK = 5_000
_ROUND_BEST_OF = {"WC": 3, "DS": 5, "LCS": 7, "WS": 7}


@dataclass(slots=True)
class SeasonSetup:
    team_ids: list[int]  # length 30, defines global index order
    league: IntArr  # (30,) 0 = AL, 1 = NL
    division: IntArr  # (30,) 0..2 within league
    wins_to_date: IntArr  # (30,)
    losses_to_date: IntArr  # (30,)
    strength: FloatArr  # (30,) rating (e.g. (elo - 1500) / 400)
    rem_home: IntArr  # (G,) global home-team index per remaining game
    rem_away: IntArr  # (G,)
    rem_p_home: FloatArr  # (G,) P(home team wins)

    def validate(self) -> None:
        if len(self.team_ids) != _N_TEAMS:
            raise ValueError(f"expected {_N_TEAMS} teams, got {len(self.team_ids)}")
        for name in ("league", "division", "wins_to_date", "losses_to_date", "strength"):
            if getattr(self, name).shape != (_N_TEAMS,):
                raise ValueError(f"{name} must have shape ({_N_TEAMS},)")
        g = self.rem_home.shape[0]
        if not (self.rem_away.shape[0] == self.rem_p_home.shape[0] == g):
            raise ValueError("remaining-game arrays must be the same length")


@dataclass(slots=True)
class SeasonSimResult:
    n_sims: int
    team_ids: list[int]
    made: IntArr  # per-team counts, all (30,)
    division: IntArr
    wildcard: IntArr
    bye: IntArr
    pennant: IntArr
    world_series: IntArr
    wins_sum: FloatArr
    seed_sum: FloatArr  # sum of playoff seed over sims where the team made it
    post_game_wins: FloatArr
    win_hist: IntArr  # (30, _MAX_WINS + 1)
    round_length_hist: dict[str, IntArr] = field(default_factory=dict)  # round -> counts by game #

    def summary(self) -> list[dict[str, Any]]:
        n = self.n_sims
        out: list[dict[str, Any]] = []
        for i, tid in enumerate(self.team_ids):
            made = int(self.made[i])
            hist = self.win_hist[i]
            out.append(
                {
                    "team_id": tid,
                    "p_playoffs": round(made / n, 5),
                    "p_division": round(int(self.division[i]) / n, 5),
                    "p_wildcard": round(int(self.wildcard[i]) / n, 5),
                    "p_bye": round(int(self.bye[i]) / n, 5),
                    "p_pennant": round(int(self.pennant[i]) / n, 5),
                    "p_world_series": round(int(self.world_series[i]) / n, 5),
                    "exp_wins": round(float(self.wins_sum[i]) / n, 2),
                    "exp_losses": round(162.0 - float(self.wins_sum[i]) / n, 2),
                    "exp_seed": round(float(self.seed_sum[i]) / made, 2) if made else None,
                    "exp_postseason_game_wins": round(float(self.post_game_wins[i]) / n, 3),
                    "win_dist": {str(w): int(c) for w, c in enumerate(hist) if c > 0},
                }
            )
        return out

    def series_length_summary(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for rnd, hist in self.round_length_hist.items():
            total = int(hist.sum())
            if total == 0:
                continue
            best_of = _ROUND_BEST_OF[rnd]
            clinch = best_of // 2 + 1
            games = np.arange(len(hist))
            exp_games = float((games * hist).sum() / total)
            out[rnd] = {
                "best_of": best_of,
                "p_sweep": round(float(hist[clinch] / total), 5),
                "exp_games": round(exp_games, 3),
                "game_count_dist": {
                    str(g): round(float(hist[g] / total), 5)
                    for g in range(clinch, best_of + 1)
                    if hist[g] > 0
                },
            }
        return out


def _simulate_chunk(
    setup: SeasonSetup, n: int, rng: np.random.Generator, acc: dict[str, Any]
) -> None:
    wins = np.tile(setup.wins_to_date.astype(np.int64), (n, 1))
    home_win = rng.random((n, setup.rem_home.shape[0])) < setup.rem_p_home
    for t in range(_N_TEAMS):
        hm = setup.rem_home == t
        am = setup.rem_away == t
        hw_t = home_win[:, hm]
        aw_t = home_win[:, am]
        wins[:, t] += hw_t.sum(axis=1) + (~aw_t).sum(axis=1)

    al = np.where(setup.league == 0)[0]
    nl = np.where(setup.league == 1)[0]
    seeds_global = {}
    seed_no_full = np.zeros((n, _N_TEAMS), dtype=np.int64)
    made_full = np.zeros((n, _N_TEAMS), dtype=bool)
    div_full = np.zeros((n, _N_TEAMS), dtype=bool)
    wc_full = np.zeros((n, _N_TEAMS), dtype=bool)
    for tag, gidx in (("al", al), ("nl", nl)):
        res = seed_league(wins[:, gidx], setup.strength[gidx], setup.division[gidx], rng)
        seeds_global[tag] = gidx[res["seeds_local"]]
        seed_no_full[:, gidx] = res["seed_no"]
        made_full[:, gidx] = res["made"]
        div_full[:, gidx] = res["division_winner"]
        wc_full[:, gidx] = res["wild_card"]

    bracket = simulate_bracket(seeds_global["al"], seeds_global["nl"], setup.strength, wins, rng)

    acc["made"] += made_full.sum(axis=0)
    acc["division"] += div_full.sum(axis=0)
    acc["wildcard"] += wc_full.sum(axis=0)
    acc["bye"] += ((seed_no_full == 1) | (seed_no_full == 2)).sum(axis=0)
    acc["wins_sum"] += wins.sum(axis=0)
    acc["seed_sum"] += np.where(made_full, seed_no_full, 0).sum(axis=0)
    for pen in (bracket["pennant_al"], bracket["pennant_nl"]):
        np.add.at(acc["pennant"], pen, 1)
    np.add.at(acc["world_series"], bracket["ws_winner"], 1)
    for t in range(_N_TEAMS):
        np.add.at(acc["win_hist"][t], np.clip(wins[:, t], 0, _MAX_WINS), 1)
    acc["post_game_wins"] += bracket["post_game_wins"].sum(axis=0)
    for rnd, g in bracket["round_lengths"].items():
        np.add.at(acc["round_length_hist"][rnd], g, 1)


def simulate_season(setup: SeasonSetup, *, n_sims: int = 100_000, seed: int = 0) -> SeasonSimResult:
    setup.validate()
    rng = np.random.default_rng(seed)
    acc: dict[str, Any] = {
        "made": np.zeros(_N_TEAMS, np.int64),
        "division": np.zeros(_N_TEAMS, np.int64),
        "wildcard": np.zeros(_N_TEAMS, np.int64),
        "bye": np.zeros(_N_TEAMS, np.int64),
        "pennant": np.zeros(_N_TEAMS, np.int64),
        "world_series": np.zeros(_N_TEAMS, np.int64),
        "wins_sum": np.zeros(_N_TEAMS, np.float64),
        "seed_sum": np.zeros(_N_TEAMS, np.float64),
        "post_game_wins": np.zeros(_N_TEAMS, np.float64),
        "win_hist": np.zeros((_N_TEAMS, _MAX_WINS + 1), np.int64),
        "round_length_hist": {r: np.zeros(9, np.int64) for r in _ROUND_BEST_OF},
    }
    done = 0
    while done < n_sims:
        n = min(_CHUNK, n_sims - done)
        _simulate_chunk(setup, n, rng, acc)
        done += n

    return SeasonSimResult(
        n_sims=n_sims,
        team_ids=setup.team_ids,
        made=acc["made"],
        division=acc["division"],
        wildcard=acc["wildcard"],
        bye=acc["bye"],
        pennant=acc["pennant"],
        world_series=acc["world_series"],
        wins_sum=acc["wins_sum"],
        seed_sum=acc["seed_sum"],
        post_game_wins=acc["post_game_wins"],
        win_hist=acc["win_hist"],
        round_length_hist=acc["round_length_hist"],
    )
