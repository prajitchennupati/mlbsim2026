"""Vectorised postseason: seeding, tiebreak, best-of-K series, the bracket.

The 2022+ format: per league, 3 division winners (seeds 1-3 by record) plus 3
wild cards (seeds 4-6). Seeds 1-2 get a bye; seed 3 hosts 6 and seed 4 hosts 5
in a best-of-3 Wild Card round; Division Series best-of-5 (2-2-1), Championship
Series and World Series best-of-7 (2-3-2). Higher seed hosts the odd games.

Regular-season ties are broken here by team strength plus a small per-sim jitter
(a proxy for the head-to-head cascade the real rules use); exact tiebreakers are
a documented later refinement.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArr = NDArray[np.float64]
IntArr = NDArray[np.int64]
BoolArr = NDArray[np.bool_]

# ln(10): with strength expressed in Elo/400 units this recovers standard Elo,
# p = 1 / (1 + 10 ** (-(elo_a - elo_b) / 400)).
_STRENGTH_SCALE = 2.302585
_HFA_LOGIT = 0.15  # ~.537 home win prob for two equal teams
_TIE_STRENGTH_W = 1e-3
_TIE_NOISE_W = 1e-6

# home-team-is-A pattern per series game (A = higher seed / better record)
HOME_BO3 = (True, True, True)  # all at the higher seed
HOME_BO5 = (True, True, False, False, True)  # 2-2-1
HOME_BO7 = (True, True, False, False, False, True, True)  # 2-3-2


def _logistic(x: FloatArr) -> FloatArr:
    return 1.0 / (1.0 + np.exp(-x))


def game_prob_a(s_a: FloatArr, s_b: FloatArr, *, home_a: bool) -> FloatArr:
    """P(team A beats team B) in one game, given strength ratings."""
    logit = _STRENGTH_SCALE * (s_a - s_b) + (_HFA_LOGIT if home_a else -_HFA_LOGIT)
    return _logistic(logit)


def simulate_series(
    a_idx: IntArr,
    b_idx: IntArr,
    strength: FloatArr,
    *,
    best_of: int,
    home_pattern: tuple[bool, ...],
    rng: np.random.Generator,
) -> tuple[BoolArr, IntArr]:
    """Return ``(a_wins_series, n_games)`` for a cohort of series.

    ``a_idx`` / ``b_idx`` are global team indices per sim; A is the home-pattern
    team (higher seed / better record).
    """
    n = a_idx.shape[0]
    clinch = best_of // 2 + 1
    s_a, s_b = strength[a_idx], strength[b_idx]
    p_home = game_prob_a(s_a, s_b, home_a=True)
    p_road = game_prob_a(s_a, s_b, home_a=False)

    a_ws = np.zeros(n, dtype=np.int64)
    b_ws = np.zeros(n, dtype=np.int64)
    games = np.zeros(n, dtype=np.int64)
    for g in range(best_of):
        still = (a_ws < clinch) & (b_ws < clinch)
        if not still.any():
            break
        p_a = p_home if home_pattern[g] else p_road
        a_win = still & (rng.random(n) < p_a)
        a_ws += a_win
        b_ws += still & ~a_win
        games += still
    return a_ws >= clinch, games


def _rank_key(wins: IntArr, strength: FloatArr, rng: np.random.Generator) -> FloatArr:
    """Sortable key: wins dominate, strength breaks ties, tiny noise breaks the rest."""
    n, k = wins.shape
    s = strength.reshape(1, k)
    s_norm = (s - s.min()) / (s.max() - s.min() + 1e-9)
    noise = rng.random((n, k)) * _TIE_NOISE_W
    return wins.astype(np.float64) + _TIE_STRENGTH_W * s_norm + noise


def seed_league(
    wins: IntArr,
    strength: FloatArr,
    division: IntArr,
    rng: np.random.Generator,
) -> dict[str, NDArray[Any]]:
    """Seed one league.

    ``wins`` is ``(n_sims, T)`` for the league's teams (global indices in
    ``team_global``); ``division`` is ``(T,)`` in {0,1,2}. Returns global-index
    arrays: ``seeds`` ``(n_sims, 6)``, ``division_winner`` / ``wild_card`` /
    ``made`` ``(n_sims, T)`` bool, ``seed_no`` ``(n_sims, T)``.
    """
    n, t = wins.shape
    key = _rank_key(wins, strength, rng)
    rows = np.arange(n)

    dw_local = np.stack(
        [np.where(division == d, key, -np.inf).argmax(axis=1) for d in (0, 1, 2)], axis=1
    )  # (n, 3) local indices of the three division winners
    dw_keys = key[rows[:, None], dw_local]
    dw_order = np.argsort(-dw_keys, axis=1)  # rank the 3 winners
    seeds_local = np.empty((n, 6), dtype=np.int64)
    for r in range(3):
        seeds_local[:, r] = dw_local[rows, dw_order[:, r]]

    key_nw = key.copy()
    np.put_along_axis(key_nw, dw_local, -np.inf, axis=1)
    wc_local = np.argsort(-key_nw, axis=1)[:, :3]
    seeds_local[:, 3:] = wc_local

    seed_no = np.zeros((n, t), dtype=np.int64)
    for r in range(6):
        seed_no[rows, seeds_local[:, r]] = r + 1
    division_winner = (seed_no >= 1) & (seed_no <= 3)
    wild_card = (seed_no >= 4) & (seed_no <= 6)
    made = seed_no > 0
    return {
        "seeds_local": seeds_local,
        "seed_no": seed_no,
        "division_winner": division_winner,
        "wild_card": wild_card,
        "made": made,
    }


def simulate_bracket(
    seeds_al: IntArr,
    seeds_nl: IntArr,
    strength: FloatArr,
    wins_global: IntArr,
    rng: np.random.Generator,
    *,
    n_teams: int = 30,
) -> dict[str, Any]:
    """Run both league brackets + the World Series.

    ``seeds_*`` are ``(n_sims, 6)`` GLOBAL team indices (seed 1..6). Returns
    per-sim arrays: ``pennant_al`` / ``pennant_nl`` (global idx), ``ws_winner``
    (global idx), ``post_game_wins`` ``(n_sims, n_teams)``, plus round-length
    arrays for WC/DS/LCS/WS.
    """
    n = seeds_al.shape[0]
    rows = np.arange(n)
    pgw = np.zeros((n, n_teams), dtype=np.int64)
    lengths: dict[str, list[IntArr]] = {"WC": [], "DS": [], "LCS": [], "WS": []}
    pennants: list[IntArr] = []

    def run(
        a_idx: IntArr, b_idx: IntArr, best_of: int, home: tuple[bool, ...]
    ) -> tuple[BoolArr, IntArr]:
        a_wins, games = simulate_series(
            a_idx, b_idx, strength, best_of=best_of, home_pattern=home, rng=rng
        )
        clinch = best_of // 2 + 1
        a_gw = np.where(a_wins, clinch, games - clinch)
        np.add.at(pgw, (rows, a_idx), a_gw)
        np.add.at(pgw, (rows, b_idx), games - a_gw)
        return a_wins, games

    for seeds in (seeds_al, seeds_nl):
        s1, s2, s3, s4, s5, s6 = (seeds[:, i] for i in range(6))
        w36, g = run(s3, s6, 3, HOME_BO3)
        lengths["WC"].append(g)
        w45, g = run(s4, s5, 3, HOME_BO3)
        lengths["WC"].append(g)

        ds1_b = np.where(w36, s3, s6)
        ds2_b = np.where(w45, s4, s5)
        a1, g = run(s1, ds1_b, 5, HOME_BO5)
        lengths["DS"].append(g)
        a2, g = run(s2, ds2_b, 5, HOME_BO5)
        lengths["DS"].append(g)

        lcs_1 = np.where(a1, s1, ds1_b)
        lcs_2 = np.where(a2, s2, ds2_b)
        one_hosts = wins_global[rows, lcs_1] >= wins_global[rows, lcs_2]
        lcs_a = np.where(one_hosts, lcs_1, lcs_2)
        lcs_b = np.where(one_hosts, lcs_2, lcs_1)
        a_wins, g = run(lcs_a, lcs_b, 7, HOME_BO7)
        lengths["LCS"].append(g)
        pennants.append(np.where(a_wins, lcs_a, lcs_b))

    pen_al, pen_nl = pennants
    al_hosts = wins_global[rows, pen_al] >= wins_global[rows, pen_nl]
    ws_a = np.where(al_hosts, pen_al, pen_nl)
    ws_b = np.where(al_hosts, pen_nl, pen_al)
    a_wins, g = run(ws_a, ws_b, 7, HOME_BO7)
    lengths["WS"].append(g)
    ws_winner = np.where(a_wins, ws_a, ws_b)

    return {
        "pennant_al": pen_al,
        "pennant_nl": pen_nl,
        "ws_winner": ws_winner,
        "post_game_wins": pgw,
        "round_lengths": {k: np.concatenate(v) for k, v in lengths.items()},
    }
