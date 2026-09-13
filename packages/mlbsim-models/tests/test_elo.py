from __future__ import annotations

from mlbsim_models.ratings.elo import (
    DEFAULT_RATING,
    EloConfig,
    EloRatings,
    expected_home_win,
    mov_multiplier,
    run_elo,
)


def test_equal_teams_home_edge():
    cfg = EloConfig()
    p = expected_home_win(1500, 1500, cfg)
    assert 0.50 < p < 0.60  # home-field only


def test_win_prob_monotone_in_rating_gap():
    cfg = EloConfig()
    gaps = [expected_home_win(1500 + d, 1500, cfg) for d in (-200, -100, 0, 100, 200)]
    assert gaps == sorted(gaps)
    assert gaps[0] < 0.5 < gaps[-1]


def test_mov_multiplier_dampens_expected_blowouts():
    # same margin, but the second winner was a much bigger favorite -> smaller multiplier
    underdog = mov_multiplier(6, winner_rating_edge=-50)
    favorite = mov_multiplier(6, winner_rating_edge=300)
    assert underdog > favorite > 0
    assert mov_multiplier(0, 0) == 1.0


def test_winning_raises_rating_losing_lowers():
    t = EloRatings(EloConfig(mov_enabled=False))
    res = t.process_game(
        game_pk=1,
        game_date="2024-04-01",
        season=2024,
        home_team_id=10,
        away_team_id=20,
        home_score=5,
        away_score=2,
    )
    assert res.home_rating_post > res.home_rating_pre
    assert res.away_rating_post < res.away_rating_pre
    # zero-sum
    assert (
        round(
            (res.home_rating_post - res.home_rating_pre)
            + (res.away_rating_post - res.away_rating_pre),
            6,
        )
        == 0.0
    )


def test_run_elo_is_chronological_and_dominant_team_climbs():
    games = []
    for i in range(40):
        # team 1 always beats team 2
        games.append(
            {
                "game_pk": 100 - i,  # deliberately out of order
                "game_date": f"2024-04-{i + 1:02d}",
                "season": 2024,
                "home_team_id": 1 if i % 2 == 0 else 2,
                "away_team_id": 2 if i % 2 == 0 else 1,
                "home_score": 4 if i % 2 == 0 else 1,
                "away_score": 1 if i % 2 == 0 else 4,
            }
        )
    res = run_elo(games)
    assert [r.game_date for r in res] == sorted(r.game_date for r in res)
    final = EloRatings()
    for r in res:
        final._r[r.home_team_id] = r.home_rating_post
        final._r[r.away_team_id] = r.away_rating_post
    assert final.rating(1) > DEFAULT_RATING > final.rating(2)


def test_run_elo_skips_unplayed_games():
    games = [
        {
            "game_pk": 1,
            "game_date": "2024-04-01",
            "season": 2024,
            "home_team_id": 1,
            "away_team_id": 2,
            "home_score": 3,
            "away_score": 2,
        },
        {
            "game_pk": 2,
            "game_date": "2024-04-02",
            "season": 2024,
            "home_team_id": 1,
            "away_team_id": 2,
            "home_score": None,
            "away_score": None,
        },
    ]
    assert len(run_elo(games)) == 1


def test_season_reversion_pulls_toward_mean():
    t = EloRatings(EloConfig(season_revert=0.5))
    t._r[7] = 1700.0
    t.revert_to_mean()
    assert t.rating(7) == 1600.0
