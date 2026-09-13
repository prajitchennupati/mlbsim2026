"""Vectorised plate-appearance game simulator.

All ``n`` simulated games advance together, one plate appearance per iteration
for the sims currently batting. Everything is NumPy array ops — no per-game
Python loop — so 10k sims/game runs in well under a second. A Numba path is an
optional M10 optimisation, not a dependency here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mlbsim_engine.baserunning import advance_batch
from mlbsim_engine.outcomes import BB, HBP, HR, N_OUTCOMES, K
from mlbsim_engine.rates import neutral_lineup, neutral_pitcher, odds_ratio_matchup

Array = NDArray[np.float64]
IntArr = NDArray[np.int64]
_INN_COLS = 12  # innings 1..9 in cols 0..8; 10, 11, 12+ in cols 9..11
_SAFETY_INNINGS = 30
_MAX_STEPS = _SAFETY_INNINGS * 20
_CHUNK = 25_000  # per-batch cohort size; bounds working set + per-step allocation
_TB = np.array([0, 0, 0, 0, 1, 2, 3, 4], dtype=np.int64)  # total bases per outcome
_HIT = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
_BAT_STATS = ("pa", "ab", "h", "hr", "bb", "k", "tb", "rbi")
_SP_STATS = ("bf", "outs", "k", "h", "bb", "hr", "r")


def _empty_bat(n: int) -> dict[str, IntArr]:
    return {s: np.zeros((9, n), dtype=np.int64) for s in _BAT_STATS}


def _empty_sp(n: int) -> dict[str, IntArr]:
    return {s: np.zeros(n, dtype=np.int64) for s in _SP_STATS}


@dataclass(slots=True)
class GameSimResult:
    n_sims: int
    seed: int
    home_score: IntArr
    away_score: IntArr
    inn_home: IntArr  # (n, 12)
    inn_away: IntArr
    went_extras: NDArray[np.bool_]
    bat_home: dict[str, IntArr]  # each (9, n)
    bat_away: dict[str, IntArr]
    sp_home: dict[str, IntArr]  # each (n,)
    sp_away: dict[str, IntArr]

    # -- game level -----------------------------------------------------------
    @property
    def home_win_prob(self) -> float:
        wins = (self.home_score > self.away_score).sum()
        ties = (self.home_score == self.away_score).sum()  # only via safety cap
        return float((wins + 0.5 * ties) / self.n_sims)

    def score_grid(self, max_runs: int = 15) -> Array:
        g = np.zeros((max_runs + 1, max_runs + 1), dtype=np.float64)
        h = np.clip(self.home_score, 0, max_runs)
        a = np.clip(self.away_score, 0, max_runs)
        np.add.at(g, (h, a), 1.0)
        return g / self.n_sims

    # -- players -----------------------------------------------------------
    def _bat(self, side: str) -> dict[str, IntArr]:
        return self.bat_home if side == "home" else self.bat_away

    def batting_lines(self, side: str) -> dict[int, dict[str, float]]:
        b = self._bat(side)
        return {
            slot: {s: round(float(b[s][slot].mean()), 3) for s in _BAT_STATS} for slot in range(9)
        }

    def batting_props(self, side: str) -> dict[int, dict[str, float]]:
        b = self._bat(side)
        out: dict[int, dict[str, float]] = {}
        for slot in range(9):
            h = b["h"][slot]
            out[slot] = {
                "pa": round(float(b["pa"][slot].mean()), 3),
                "mean_h": round(float(h.mean()), 3),
                "mean_hr": round(float(b["hr"][slot].mean()), 3),
                "mean_tb": round(float(b["tb"][slot].mean()), 3),
                "mean_rbi": round(float(b["rbi"][slot].mean()), 3),
                "mean_bb": round(float(b["bb"][slot].mean()), 3),
                "mean_k": round(float(b["k"][slot].mean()), 3),
                "p_1plus_h": round(float((h >= 1).mean()), 5),
                "p_2plus_h": round(float((h >= 2).mean()), 5),
                "p_3plus_h": round(float((h >= 3).mean()), 5),
                "p_hr": round(float((b["hr"][slot] >= 1).mean()), 5),
                "p_rbi": round(float((b["rbi"][slot] >= 1).mean()), 5),
                "p_bb": round(float((b["bb"][slot] >= 1).mean()), 5),
                "p_k": round(float((b["k"][slot] >= 1).mean()), 5),
            }
        return out

    def pitcher_props(self, side: str) -> dict[str, float]:
        sp = self.sp_home if side == "home" else self.sp_away
        ip = sp["outs"] / 3.0
        k = sp["k"]
        return {
            "mean_ip": round(float(ip.mean()), 3),
            "mean_k": round(float(k.mean()), 3),
            "mean_bf": round(float(sp["bf"].mean()), 3),
            "mean_h": round(float(sp["h"].mean()), 3),
            "mean_bb": round(float(sp["bb"].mean()), 3),
            "mean_hr": round(float(sp["hr"].mean()), 3),
            "mean_runs": round(float(sp["r"].mean()), 3),
            "p_5plus_k": round(float((k >= 5).mean()), 5),
            "p_6plus_k": round(float((k >= 6).mean()), 5),
            "p_7plus_k": round(float((k >= 7).mean()), 5),
            "p_8plus_k": round(float((k >= 8).mean()), 5),
            "p_10plus_k": round(float((k >= 10).mean()), 5),
            "p_5plus_ip": round(float((sp["outs"] >= 15).mean()), 5),
            "p_6plus_ip": round(float((sp["outs"] >= 18).mean()), 5),
            "p_7plus_ip": round(float((sp["outs"] >= 21).mean()), 5),
            "p_hr_allowed": round(float((sp["hr"] >= 1).mean()), 5),
            "p_run_allowed": round(float((sp["r"] >= 1).mean()), 5),
        }

    # -- innings ---------------------------------------------------------------
    def inning_probs(self) -> dict[str, Any]:
        def per_team(arr: IntArr) -> list[dict[str, float]]:
            return [
                {
                    "exp_runs": round(float(arr[:, c].mean()), 4),
                    "p_score_1plus": round(float((arr[:, c] >= 1).mean()), 5),
                    "p_score_2plus": round(float((arr[:, c] >= 2).mean()), 5),
                    "p_score_3plus": round(float((arr[:, c] >= 3).mean()), 5),
                    "p_scoreless": round(float((arr[:, c] == 0).mean()), 5),
                }
                for c in range(_INN_COLS)
            ]

        cum_h = np.cumsum(self.inn_home, axis=1)
        cum_a = np.cumsum(self.inn_away, axis=1)
        p_home_lead = [
            round(
                float(((cum_h[:, c] > cum_a[:, c]) + 0.5 * (cum_h[:, c] == cum_a[:, c])).mean()),
                5,
            )
            for c in range(_INN_COLS)
        ]
        return {
            "home": per_team(self.inn_home),
            "away": per_team(self.inn_away),
            "p_home_lead_after": p_home_lead,
        }

    # -- bundle -------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        h, a, n = self.home_score, self.away_score, self.n_sims
        grid = self.score_grid()
        flat_order = np.argsort(grid, axis=None)[::-1][:5]
        most_likely = [
            {
                "home": int(i // grid.shape[1]),
                "away": int(i % grid.shape[1]),
                "p": round(float(grid.flat[i]), 5),
            }
            for i in flat_order
        ]
        return {
            "n_sims": n,
            "home_win_prob": round(self.home_win_prob, 5),
            "away_win_prob": round(1.0 - self.home_win_prob, 5),
            "exp_home_runs": round(float(h.mean()), 3),
            "exp_away_runs": round(float(a.mean()), 3),
            "exp_total_runs": round(float((h + a).mean()), 3),
            "p_extra_innings": round(float(self.went_extras.mean()), 5),
            "p_shutout_home": round(float((a == 0).mean()), 5),
            "p_shutout_away": round(float((h == 0).mean()), 5),
            "p_one_run_game": round(float((np.abs(h - a) == 1).mean()), 5),
            "most_likely_scores": most_likely,
            "home_score_dist": _pmf(h, n),
            "away_score_dist": _pmf(a, n),
            "total_runs_dist": _pmf(h + a, n, cap=25),
            "inning_probs": self.inning_probs(),
            "home_batting": self.batting_lines("home"),
            "away_batting": self.batting_lines("away"),
            "home_batting_props": self.batting_props("home"),
            "away_batting_props": self.batting_props("away"),
            "home_pitcher_props": self.pitcher_props("home"),
            "away_pitcher_props": self.pitcher_props("away"),
        }


@dataclass(slots=True)
class GameState:
    """A mid-game situation to resume simulation from (all scalars)."""

    inning: int = 1
    half: str = "top"  # "top" | "bottom"
    outs: int = 0
    base_state: int = 0  # 3-bit mask, 1B=1 2B=2 3B=4
    home_score: int = 0
    away_score: int = 0
    bo_home: int = 0  # next home batter, lineup index 0..8
    bo_away: int = 0
    home_sp_out: bool = False  # starter already pulled -> use the home bullpen
    away_sp_out: bool = False


def _pmf(x: IntArr, n: int, *, cap: int = 15) -> dict[str, float]:
    counts = np.bincount(np.clip(x, 0, cap), minlength=cap + 1)[: cap + 1]
    return {str(i): round(float(c) / n, 6) for i, c in enumerate(counts)}


def _draw_events(probs: Array, rng: np.random.Generator) -> IntArr:
    """Inverse-CDF multinomial draw per row of ``probs`` (n, 8) -> (n,) codes."""
    cdf = np.cumsum(probs, axis=1)
    u = rng.random((probs.shape[0], 1))
    codes = (u > cdf).sum(axis=1)
    return np.asarray(np.minimum(codes, N_OUTCOMES - 1), dtype=np.int64)


# ``sidx`` is always a subset of ``arange(n)`` (one active sim per iteration), so the
# index pairs are unique and plain ``arr[idx] += v`` is correct and ~3x faster than
# ``np.add.at`` (which pays for unbuffered/duplicate-safe scatter it doesn't need).
def _accum_bat(b: dict[str, IntArr], slots: IntArr, sidx: IntArr, e: IntArr, r: IntArr) -> None:
    at = (slots, sidx)
    b["pa"][at] += 1
    b["ab"][at] += ((e != BB) & (e != HBP)).astype(np.int64)
    b["h"][at] += _HIT[e]
    b["hr"][at] += (e == HR).astype(np.int64)
    b["bb"][at] += (e == BB).astype(np.int64)
    b["k"][at] += (e == K).astype(np.int64)
    b["tb"][at] += _TB[e]
    b["rbi"][at] += r


def _accum_sp(sp: dict[str, IntArr], sidx: IntArr, e: IntArr, r: IntArr, oa: IntArr) -> None:
    sp["bf"][sidx] += 1
    sp["outs"][sidx] += oa
    sp["k"][sidx] += (e == K).astype(np.int64)
    sp["h"][sidx] += _HIT[e]
    sp["bb"][sidx] += ((e == BB) | (e == HBP)).astype(np.int64)
    sp["hr"][sidx] += (e == HR).astype(np.int64)
    sp["r"][sidx] += r


def simulate_game(
    *,
    home_lineup: Array | None = None,
    away_lineup: Array | None = None,
    home_sp: Array | None = None,
    away_sp: Array | None = None,
    home_bp: Array | None = None,
    away_bp: Array | None = None,
    n_sims: int = 10_000,
    seed: int = 0,
    hard_hook_inning: int = 8,
    track_batting: bool = True,
    start_state: GameState | None = None,
) -> GameSimResult:
    """Simulate ``n_sims`` games between the two rate sets and summarise.

    The starter is pulled once its batters-faced count reaches a per-sim draw
    (~19-28 BF, i.e. roughly 4.5-8 IP) or the ``hard_hook_inning`` is reached,
    whichever comes first; a single reliever rate vector covers the rest.

    Pass ``start_state`` to resume from a mid-game situation (live win probability):
    inning-runs and per-player tallies then only cover the remainder of the game.

    Large ``n_sims`` is processed in cohorts of :data:`_CHUNK` so the working set and
    per-step allocation stay bounded (throughput is then flat in ``n_sims``).
    """
    st = start_state or GameState()
    home_off = neutral_lineup() if home_lineup is None else np.asarray(home_lineup, np.float64)
    away_off = neutral_lineup() if away_lineup is None else np.asarray(away_lineup, np.float64)
    hsp = neutral_pitcher() if home_sp is None else np.asarray(home_sp, np.float64)
    asp = neutral_pitcher() if away_sp is None else np.asarray(away_sp, np.float64)
    hbp = hsp if home_bp is None else np.asarray(home_bp, np.float64)
    abp = asp if away_bp is None else np.asarray(away_bp, np.float64)
    rates = (home_off, away_off, hsp, asp, hbp, abp)

    if n_sims <= _CHUNK:
        return _simulate_cohort(
            n_sims, seed, np.random.default_rng(seed), rates, st, hard_hook_inning, track_batting
        )

    parts: list[GameSimResult] = []
    done = 0
    ci = 0
    while done < n_sims:
        m = min(_CHUNK, n_sims - done)
        parts.append(
            _simulate_cohort(
                m,
                seed,
                np.random.default_rng([seed, ci]),
                rates,
                st,
                hard_hook_inning,
                track_batting,
            )
        )
        done += m
        ci += 1
    return _concat_results(parts, seed)


def _concat_results(parts: list[GameSimResult], seed: int) -> GameSimResult:
    def cat_bat(key: str) -> dict[str, IntArr]:
        dicts = [getattr(p, key) for p in parts]
        return {s: np.concatenate([d[s] for d in dicts], axis=1) for s in _BAT_STATS}

    def cat_sp(key: str) -> dict[str, IntArr]:
        dicts = [getattr(p, key) for p in parts]
        return {s: np.concatenate([d[s] for d in dicts]) for s in _SP_STATS}

    return GameSimResult(
        n_sims=sum(p.n_sims for p in parts),
        seed=seed,
        home_score=np.concatenate([p.home_score for p in parts]),
        away_score=np.concatenate([p.away_score for p in parts]),
        inn_home=np.concatenate([p.inn_home for p in parts]),
        inn_away=np.concatenate([p.inn_away for p in parts]),
        went_extras=np.concatenate([p.went_extras for p in parts]),
        bat_home=cat_bat("bat_home"),
        bat_away=cat_bat("bat_away"),
        sp_home=cat_sp("sp_home"),
        sp_away=cat_sp("sp_away"),
    )


def _simulate_cohort(  # noqa: PLR0915, PLR0917
    n: int,
    seed: int,
    rng: np.random.Generator,
    rates: tuple[Array, Array, Array, Array, Array, Array],
    st: GameState,
    hard_hook_inning: int,
    track_batting: bool,
) -> GameSimResult:
    home_off, away_off, hsp, asp, hbp, abp = rates
    hook_bf_home = rng.integers(19, 29, n)
    hook_bf_away = rng.integers(19, 29, n)

    idx = np.arange(n)
    home_score = np.full(n, st.home_score, np.int64)
    away_score = np.full(n, st.away_score, np.int64)
    inn_home = np.zeros((n, _INN_COLS), np.int64)
    inn_away = np.zeros((n, _INN_COLS), np.int64)
    inning = np.full(n, max(st.inning, 1), np.int64)
    top = np.full(n, st.half == "top", dtype=bool)
    outs = np.full(n, min(max(st.outs, 0), 2), np.int64)
    base = np.full(n, st.base_state & 0b111, np.int64)
    bo_home = np.full(n, st.bo_home % 9, np.int64)
    bo_away = np.full(n, st.bo_away % 9, np.int64)
    live = np.ones(n, dtype=bool)
    went_extras = np.full(n, st.inning >= 10, dtype=bool)
    bat_home, bat_away = _empty_bat(n), _empty_bat(n)
    sp_home, sp_away = _empty_sp(n), _empty_sp(n)
    if st.home_sp_out:
        hook_bf_home = np.zeros(n, np.int64)
    if st.away_sp_out:
        hook_bf_away = np.zeros(n, np.int64)

    for _ in range(_MAX_STEPS):
        if not live.any():
            break
        active = live & (outs < 3)

        if not active.any():
            flip = live & (outs >= 3)
            end_top = flip & top & (inning >= 9) & (home_score > away_score)
            end_bot = flip & ~top & (inning >= 9) & (home_score != away_score)
            live[end_top | end_bot] = False
            still = live & (outs >= 3)
            inning = np.where(still & ~top, inning + 1, inning)
            top = np.where(still, ~top, top)
            outs = np.where(still, 0, outs)
            ghost = still & (inning >= 10)
            base = np.where(still, np.where(ghost, 2, 0), base)
            went_extras |= ghost
            if (inning > _SAFETY_INNINGS).any():
                cap = live & (inning > _SAFETY_INNINGS)
                home_score[cap] += (away_score[cap] >= home_score[cap]).astype(np.int64)
                live[cap] = False
            continue

        bo = np.where(top, bo_away, bo_home)
        batter_rates = np.where(top[:, None], away_off[bo], home_off[bo])
        bp_home = (sp_home["bf"] >= hook_bf_home) | (inning > hard_hook_inning)
        bp_away = (sp_away["bf"] >= hook_bf_away) | (inning > hard_hook_inning)
        home_pitch = np.where(bp_home[:, None], hbp, hsp)  # home pitches in the top half
        away_pitch = np.where(bp_away[:, None], abp, asp)
        pitcher_rates = np.where(top[:, None], home_pitch, away_pitch)

        probs = odds_ratio_matchup(batter_rates, pitcher_rates)
        ev = _draw_events(probs, rng)
        new_base, runs, outs_add = advance_batch(base, outs, ev, rng.random((n, 3)))

        ap_home = active & ~top
        ap_away = active & top
        home_score[ap_home] += runs[ap_home]
        away_score[ap_away] += runs[ap_away]
        col = np.clip(inning - 1, 0, _INN_COLS - 1)
        inn_home[idx[ap_home], col[ap_home]] += runs[ap_home]  # ap_* indices are unique
        inn_away[idx[ap_away], col[ap_away]] += runs[ap_away]

        base = np.where(active, new_base, base)
        outs = np.where(active, outs + outs_add, outs)
        bo_away = np.where(ap_away, (bo_away + 1) % 9, bo_away)
        bo_home = np.where(ap_home, (bo_home + 1) % 9, bo_home)

        walkoff = ap_home & (inning >= 9) & (home_score > away_score)
        live[walkoff] = False

        # starter lines: accumulate while the starter is still in
        sp_h_m = ap_away & ~bp_home  # home SP faces the away lineup (top half)
        sp_a_m = ap_home & ~bp_away
        if sp_h_m.any():
            _accum_sp(sp_home, idx[sp_h_m], ev[sp_h_m], runs[sp_h_m], outs_add[sp_h_m])
        if sp_a_m.any():
            _accum_sp(sp_away, idx[sp_a_m], ev[sp_a_m], runs[sp_a_m], outs_add[sp_a_m])

        if track_batting:
            if ap_home.any():
                _accum_bat(bat_home, bo[ap_home], idx[ap_home], ev[ap_home], runs[ap_home])
            if ap_away.any():
                _accum_bat(bat_away, bo[ap_away], idx[ap_away], ev[ap_away], runs[ap_away])

    return GameSimResult(
        n_sims=n,
        seed=seed,
        home_score=home_score,
        away_score=away_score,
        inn_home=inn_home,
        inn_away=inn_away,
        went_extras=went_extras,
        bat_home=bat_home,
        bat_away=bat_away,
        sp_home=sp_home,
        sp_away=sp_away,
    )


def live_win_probability(
    state: GameState,
    *,
    home_lineup: Array | None = None,
    away_lineup: Array | None = None,
    home_sp: Array | None = None,
    away_sp: Array | None = None,
    home_bp: Array | None = None,
    away_bp: Array | None = None,
    n_sims: int = 6_000,
    seed: int = 0,
) -> float:
    """Resume the plate-appearance simulator from ``state`` and read off P(home win)."""
    res = simulate_game(
        home_lineup=home_lineup,
        away_lineup=away_lineup,
        home_sp=home_sp,
        away_sp=away_sp,
        home_bp=home_bp,
        away_bp=away_bp,
        n_sims=n_sims,
        seed=seed,
        track_batting=False,
        start_state=state,
    )
    return res.home_win_prob
