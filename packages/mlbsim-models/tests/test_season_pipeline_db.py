"""simulate_season_from_db against a live database (schema must be migrated)."""

from __future__ import annotations

import datetime as dt
import random

import pytest
from sqlalchemy import func, select, text

from mlbsim_core import session_scope
from mlbsim_data.models import (
    EloRating,
    Game,
    SeasonSimTeamResult,
    SeriesPrediction,
    SimulationRun,
    TeamGameLog,
)
from mlbsim_models.pipelines import build_season_setup, simulate_season_from_db

pytestmark = pytest.mark.integration

_SEASON = 2026
_AS_OF = "2026-08-01"
_TABLES = ["team_game_logs", "games", "elo_ratings", "players", "teams"]


@pytest.fixture
def league_ready():
    rng = random.Random(7)
    teams = list(range(200, 230))  # 30 team ids
    strength = {t: rng.gauss(0, 70) for t in teams}  # Elo points above 1500

    with session_scope() as s:
        s.execute(text("TRUNCATE " + ", ".join(f"warehouse.{t}" for t in _TABLES) + " CASCADE"))
        s.execute(text("TRUNCATE serving.simulation_runs, serving.model_versions CASCADE"))
        team_rows = []
        for i, tid in enumerate(teams):
            league = "AL" if i < 15 else "NL"  # warehouse.teams.league is varchar(2)
            division = ("E", "C", "W")[(i // 5) % 3]  # ... .division is varchar(4)
            team_rows.append(f"({tid},'T{tid}','T{tid}','{league}','{division}')")
        s.execute(
            text(
                "INSERT INTO warehouse.teams (team_id, abbr, name, league, division) VALUES "
                + ",".join(team_rows)
            )
        )
        s.execute(
            EloRating.__table__.insert(),
            [
                {
                    "entity_type": "team",
                    "entity_id": tid,
                    "as_of_date": dt.date(2026, 7, 31),
                    "rating": 1500 + strength[tid],
                    "games_played": 100,
                }
                for tid in teams
            ],
        )

        games, tgl = [], []
        pk = 950000
        # finished games (before AS_OF) -> team_game_logs
        for _ in range(900):
            h, a = rng.sample(teams, 2)
            ph = 1 / (1 + 10 ** (-((strength[h] - strength[a] + 25) / 400)))
            home_won = rng.random() < ph
            hs, as_ = (5, 3) if home_won else (3, 5)
            games.append(
                {
                    "game_pk": pk,
                    "season": _SEASON,
                    "game_date": dt.date(2026, 6, 15),
                    "game_type": "R",
                    "status": "Final",
                    "home_team_id": h,
                    "away_team_id": a,
                    "home_score": hs,
                    "away_score": as_,
                    "dh_game_num": 1,
                    "is_doubleheader": False,
                    "scheduled_innings": 9,
                }
            )
            for tid, won, home in ((h, home_won, True), (a, not home_won, False)):
                tgl.append(
                    {
                        "team_id": tid,
                        "game_pk": pk,
                        "season": _SEASON,
                        "opponent_id": a if tid == h else h,
                        "is_home": home,
                        "runs_for": hs if tid == h else as_,
                        "runs_against": as_ if tid == h else hs,
                        "hits_for": 8,
                        "hits_against": 8,
                        "won": won,
                    }
                )
            pk += 1
        # remaining games (after AS_OF), unplayed
        for _ in range(700):
            h, a = rng.sample(teams, 2)
            games.append(
                {
                    "game_pk": pk,
                    "season": _SEASON,
                    "game_date": dt.date(2026, 9, 10),
                    "game_type": "R",
                    "status": "Scheduled",
                    "home_team_id": h,
                    "away_team_id": a,
                    "home_score": None,
                    "away_score": None,
                    "dh_game_num": 1,
                    "is_doubleheader": False,
                    "scheduled_innings": 9,
                }
            )
            pk += 1
        s.execute(Game.__table__.insert(), games)
        s.execute(TeamGameLog.__table__.insert(), tgl)
    yield teams


def test_build_setup_shapes(league_ready):
    setup, meta = build_season_setup(_SEASON, as_of=_AS_OF)
    setup.validate()  # raises if not 30 teams / consistent
    assert meta["remaining_games"] == 700
    assert setup.wins_to_date.sum() == 900  # one win per finished game
    assert set(setup.league.tolist()) == {0, 1}
    assert set(setup.division.tolist()) == {0, 1, 2}


def test_simulate_season_persists_and_totals(league_ready):
    out = simulate_season_from_db(_SEASON, n_sims=8000, seed=1, as_of=_AS_OF)
    rows = out["top_world_series"]
    assert rows and 0.0 < rows[0]["p_world_series"] <= 1.0

    with session_scope() as s:
        run = s.scalar(select(SimulationRun).where(SimulationRun.scope == "season"))
        assert run is not None and run.n_sims == 8000
        team_res = list(
            s.scalars(
                select(SeasonSimTeamResult).where(SeasonSimTeamResult.sim_run_id == run.sim_run_id)
            )
        )
        assert len(team_res) == 30
        assert sum(float(r.p_playoffs) for r in team_res) == pytest.approx(12.0, abs=0.05)
        assert sum(float(r.p_world_series) for r in team_res) == pytest.approx(1.0, abs=0.05)
        assert (
            s.scalar(
                select(func.count())
                .select_from(SeriesPrediction)
                .where(SeriesPrediction.sim_run_id == run.sim_run_id)
            )
            == 4
        )


def test_stronger_teams_more_likely_to_make_the_playoffs(league_ready):
    import numpy as np

    from mlbsim_engine.season import simulate_season

    setup, _ = build_season_setup(_SEASON, as_of=_AS_OF)
    res = simulate_season(setup, n_sims=8000, seed=2)
    p = np.array([r["p_playoffs"] for r in res.summary()])
    assert np.corrcoef(p, setup.strength)[0, 1] > 0.5
