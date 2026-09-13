"""Starting-pitcher performance projections for real, upcoming games -- see
``projections/pitcher_marcel_light.py`` for the projection math. Writes
``player_season_stats`` (so ``GET /players/{id}`` has something to show) and
``player_predictions`` (so each game page's existing pitcher table has real
data instead of an empty state) for every probable starter in a date range.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass

from sqlalchemy import delete, insert, select

from mlbsim_core import get_logger, session_scope
from mlbsim_data.loaders.upsert import upsert
from mlbsim_data.models import (
    FINAL_STATUSES,
    VOID_STATUSES,
    Game,
    GameProbable,
    ModelVersion,
    PlayerPrediction,
    PlayerSeasonStat,
)
from mlbsim_data.sources.mlb_statsapi import player_year_by_year_pitching
from mlbsim_models.projections.pitcher_marcel_light import SeasonLine, project_pitcher

_log = get_logger(__name__)

MODEL_ID = "marcel_light_v1"


def _season_lines(player_id: int) -> list[SeasonLine]:
    lines = []
    for sp in player_year_by_year_pitching(player_id):
        stat = sp.get("stat", {})
        season = sp.get("season")
        ip = stat.get("inningsPitched")
        if not season or not ip:
            continue
        whole, _, frac = str(ip).partition(".")
        outs = int(whole or 0) * 3 + int(frac or 0)
        lines.append(
            SeasonLine(
                season=int(season),
                outs=outs,
                home_runs=int(stat.get("homeRuns", 0)),
                walks=int(stat.get("baseOnBalls", 0)),
                hit_by_pitch=int(stat.get("hitBatsmen", 0)),
                strikeouts=int(stat.get("strikeOuts", 0)),
                batters_faced=int(stat.get("battersFaced", 0)),
            )
        )
    return lines


@dataclass(slots=True)
class PitcherProjectSummary:
    players: int
    prediction_rows: int


def project_starters_for_slate(
    season: int, start: str, end: str | None = None
) -> PitcherProjectSummary:
    """Probable starters for not-yet-decided games in [start, end]."""
    end = end or start
    start_d, end_d = dt.date.fromisoformat(start), dt.date.fromisoformat(end)

    with session_scope() as s:
        games = list(
            s.execute(
                select(Game.game_pk, Game.home_team_id, Game.away_team_id).where(
                    Game.season == season,
                    Game.game_date >= start_d,
                    Game.game_date <= end_d,
                    Game.status.not_in(FINAL_STATUSES | VOID_STATUSES),
                )
            )
        )
        game_pks = [g.game_pk for g in games]
        prob_stmt = select(GameProbable.game_pk, GameProbable.probable_pitcher_id).where(
            GameProbable.game_pk.in_(game_pks), GameProbable.probable_pitcher_id.is_not(None)
        )
        probables = list(s.execute(prob_stmt))

    pitcher_games: dict[int, list[int]] = {}
    for game_pk, pitcher_id in probables:
        pitcher_games.setdefault(pitcher_id, []).append(game_pk)

    now = dt.datetime.now(dt.UTC)
    season_stat_rows = []
    pred_rows = []
    for pitcher_id, pks in pitcher_games.items():
        lines = _season_lines(pitcher_id)
        proj = project_pitcher(lines)
        proj_dict = asdict(proj)
        season_stat_rows.append(
            {
                "player_id": pitcher_id,
                "season": season,
                "group": "pitching",
                "split": "all",
                "through_date": None,
                "games": proj.n_seasons_used,
                "stat_json": proj_dict,
            }
        )
        for game_pk in pks:
            pred_rows.append(
                {
                    "game_pk": game_pk,
                    "player_id": pitcher_id,
                    "model_id": MODEL_ID,
                    "role": "pitch",
                    "created_at": now,
                    "proj_json": {
                        "mean_ip": proj.proj_ip_per_start,
                        "mean_k": proj.proj_so_per_start,
                        "mean_bb": proj.proj_bb_per_start,
                        "mean_hr": proj.proj_hr_per_start,
                        "era_equiv": proj.proj_era_equiv,
                    },
                    "prob_json": {"p_6plus_k": proj.p_6plus_k, "p_6plus_ip": proj.p_6plus_ip},
                    "dist_json": None,
                }
            )

    with session_scope() as s:
        upsert(
            s,
            ModelVersion,
            [
                {
                    "model_id": MODEL_ID,
                    "name": "Marcel-style starter projection (season-level)",
                    "kind": "marcel",
                    "version": "1",
                    "trained_at": now,
                    "hyperparams_json": {"weights": [5, 4, 3], "regress_bf": 200},
                    "notes": (
                        "Weighted recent seasons + regression to the league mean, "
                        "from Stats API yearByYear pitching -- no box-score backfill."
                    ),
                }
            ],
            index_elements=["model_id"],
        )
        if season_stat_rows:
            upsert(
                s,
                PlayerSeasonStat,
                season_stat_rows,
                index_elements=["player_id", "season", "group", "split"],
            )
        if pred_rows:
            s.execute(
                delete(PlayerPrediction).where(
                    PlayerPrediction.model_id == MODEL_ID,
                    PlayerPrediction.game_pk.in_([r["game_pk"] for r in pred_rows]),
                )
            )
            s.execute(insert(PlayerPrediction), pred_rows)

    summary = PitcherProjectSummary(players=len(pitcher_games), prediction_rows=len(pred_rows))
    _log.info("pitcher_projection.done", **asdict(summary))
    return summary
