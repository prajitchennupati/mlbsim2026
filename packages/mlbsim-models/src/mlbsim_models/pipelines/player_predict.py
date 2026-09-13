"""Per-game player predictions read off the plate-appearance simulator."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy import delete, insert, select

from mlbsim_core import get_logger, session_scope
from mlbsim_data.loaders.upsert import upsert
from mlbsim_data.models import Game, Lineup, ModelVersion, PlayerPrediction
from mlbsim_engine import ENGINE_VERSION
from mlbsim_engine.pipeline import simulate_game_from_db

_log = get_logger(__name__)

MODEL_ID = "sim_v1"


@dataclass(slots=True)
class PlayerPredictSummary:
    game_pk: int
    model_id: str
    sim_run_id: int | None
    batters: int
    pitchers: int


def _register_model() -> None:
    with session_scope() as s:
        upsert(
            s,
            ModelVersion,
            [
                {
                    "model_id": MODEL_ID,
                    "name": "Plate-appearance simulator (player lines)",
                    "kind": "sim",
                    "version": ENGINE_VERSION,
                    "trained_at": dt.datetime.now(dt.UTC),
                    "hyperparams_json": {"engine_version": ENGINE_VERSION},
                    "notes": "Per-game batter/pitcher distributions from mlbsim-engine.",
                }
            ],
            index_elements=["model_id"],
        )


def _lineup_players(game_pk: int) -> dict[str, list[int]]:
    with session_scope() as s:
        game = s.get(Game, game_pk)
        if game is None:
            raise LookupError(f"game {game_pk} not ingested")
        home_id, away_id, home_sp, away_sp = (
            game.home_team_id,
            game.away_team_id,
            game.home_sp_id,
            game.away_sp_id,
        )
        rows = list(
            s.execute(
                select(Lineup.team_id, Lineup.batting_order, Lineup.player_id)
                .where(Lineup.game_pk == game_pk)
                .order_by(Lineup.team_id, Lineup.batting_order)
            )
        )
    home = [pid for tid, _, pid in rows if tid == home_id][:9]
    away = [pid for tid, _, pid in rows if tid == away_id][:9]
    return {"home": home, "away": away, "home_sp": [home_sp], "away_sp": [away_sp]}


def predict_players_for_game(
    game_pk: int, *, n_sims: int = 10_000, seed: int = 0, rebuild: bool = True
) -> PlayerPredictSummary:
    """Simulate the game, then persist one player_predictions row per batter + starter."""
    players = _lineup_players(game_pk)
    summary = simulate_game_from_db(game_pk, n_sims=n_sims, seed=seed, persist=True)
    _register_model()
    now = dt.datetime.now(dt.UTC)

    rows: list[dict[str, Any]] = []
    for side in ("home", "away"):
        bat_lines = summary[f"{side}_batting"]
        bat_props = summary[f"{side}_batting_props"]
        for slot, player_id in enumerate(players[side]):
            rows.append(
                {
                    "game_pk": game_pk,
                    "player_id": player_id,
                    "model_id": MODEL_ID,
                    "role": "bat",
                    "created_at": now,
                    "proj_json": bat_lines.get(slot, {}),
                    "prob_json": bat_props.get(slot, {}),
                    "dist_json": None,
                }
            )
        sp_id = players[f"{side}_sp"][0]
        if sp_id is not None:
            props = summary[f"{side}_pitcher_props"]
            rows.append(
                {
                    "game_pk": game_pk,
                    "player_id": sp_id,
                    "model_id": MODEL_ID,
                    "role": "pitch",
                    "created_at": now,
                    "proj_json": {k: v for k, v in props.items() if k.startswith("mean_")},
                    "prob_json": {k: v for k, v in props.items() if k.startswith("p_")},
                    "dist_json": None,
                }
            )

    with session_scope() as s:
        if rebuild:
            s.execute(
                delete(PlayerPrediction).where(
                    PlayerPrediction.game_pk == game_pk, PlayerPrediction.model_id == MODEL_ID
                )
            )
        if rows:
            s.execute(insert(PlayerPrediction), rows)

    n_pitch = sum(1 for r in rows if r["role"] == "pitch")
    result = PlayerPredictSummary(
        game_pk=game_pk,
        model_id=MODEL_ID,
        sim_run_id=summary.get("sim_run_id"),
        batters=len(rows) - n_pitch,
        pitchers=n_pitch,
    )
    _log.info("player_predict", **asdict(result))
    return result
