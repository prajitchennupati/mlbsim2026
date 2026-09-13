from __future__ import annotations

import datetime as dt

from mlbsim_models.evaluate.backtest import (
    month_starts,
    split_rows_by_date,
    walk_forward_splits,
)


def test_month_starts_inclusive():
    ms = month_starts(dt.date(2023, 11, 15), dt.date(2024, 2, 1))
    assert ms == [
        dt.date(2023, 11, 1),
        dt.date(2023, 12, 1),
        dt.date(2024, 1, 1),
        dt.date(2024, 2, 1),
    ]


def test_walk_forward_folds_are_disjoint_and_ordered():
    folds = list(
        walk_forward_splits(
            first_test_month=dt.date(2024, 4, 1), last_test_month=dt.date(2024, 6, 1)
        )
    )
    assert [f.fold for f in folds] == [0, 1, 2]
    assert folds[0].train_end == folds[0].test_start == dt.date(2024, 4, 1)
    assert folds[0].test_end == dt.date(2024, 4, 30)
    assert folds[1].test_start == dt.date(2024, 5, 1)
    # training window for a later fold ends later (expanding)
    assert folds[2].train_end > folds[0].train_end


def test_split_rows_uses_strict_lower_bound():
    rows = [
        {"game_date": "2024-03-31", "x": "train"},
        {"game_date": "2024-04-01", "x": "test"},  # == train_end -> not training, is test
        {"game_date": "2024-04-15", "x": "test"},
        {"game_date": "2024-05-02", "x": "future"},
    ]
    split = next(
        walk_forward_splits(
            first_test_month=dt.date(2024, 4, 1), last_test_month=dt.date(2024, 4, 1)
        )
    )
    train, test = split_rows_by_date(rows, split)
    assert [r["x"] for r in train] == ["train"]
    assert [r["x"] for r in test] == ["test", "test"]
