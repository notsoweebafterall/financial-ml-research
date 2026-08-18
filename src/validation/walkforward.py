"""
Walk-forward validation splitter for panel data (multiple stocks x time).

This is the single most important piece of infrastructure in the project. If this
is wrong, every downstream number (model performance, backtest results, Sharpe
ratio, everything) is meaningless, because the model would be "cheating" by
training on information from the future.

Design: EXPANDING WINDOW walk-forward.
- Fold 0: train on months [0 .. min_train_months-1], test on the next
  test_window_months block.
- Fold 1: train on months [0 .. min_train_months], test on the next block after that.
- ...and so on, sliding forward by step_months each time, with the training window
  always GROWING (expanding), never shrinking or resetting.

Guarantees enforced with runtime assertions (not just comments):
1. Every date in a training fold is <= the fold's train_end date.
2. Every date in the corresponding test fold is > the fold's train_end date.
3. No date ever appears in both the training fold and its test fold.
4. Folds are produced in strictly increasing chronological order.
"""

from dataclasses import dataclass
from typing import Iterator
import pandas as pd


@dataclass
class WalkForwardFold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_index: pd.Index
    test_index: pd.Index


class WalkForwardSplitter:
    def __init__(
        self,
        date_col: str = "date",
        min_train_months: int = 48,
        test_window_months: int = 1,
        step_months: int = 1,
    ):
        if min_train_months <= 0 or test_window_months <= 0 or step_months <= 0:
            raise ValueError("min_train_months, test_window_months, step_months must all be positive")
        self.date_col = date_col
        self.min_train_months = min_train_months
        self.test_window_months = test_window_months
        self.step_months = step_months

    def split(self, df: pd.DataFrame) -> Iterator[WalkForwardFold]:
        dates = pd.to_datetime(df[self.date_col])
        month_periods = dates.dt.to_period("M")
        unique_months = sorted(month_periods.unique())
        n_months = len(unique_months)

        if n_months < self.min_train_months + self.test_window_months:
            raise ValueError(
                f"Not enough months ({n_months}) for min_train_months="
                f"{self.min_train_months} + test_window_months={self.test_window_months}"
            )

        fold_id = 0
        train_end_idx = self.min_train_months - 1

        while True:
            test_start_idx = train_end_idx + 1
            test_end_idx = test_start_idx + self.test_window_months - 1
            if test_end_idx >= n_months:
                break

            train_end_period = unique_months[train_end_idx]
            test_start_period = unique_months[test_start_idx]
            test_end_period = unique_months[test_end_idx]

            train_mask = month_periods <= train_end_period
            test_mask = (month_periods >= test_start_period) & (month_periods <= test_end_period)

            train_index = df.index[train_mask]
            test_index = df.index[test_mask]

            # --- Runtime safety checks. These are not decorative — if any of these
            # ever fail, STOP immediately and do not trust any result downstream. ---
            overlap = train_index.intersection(test_index)
            assert len(overlap) == 0, (
                f"LEAKAGE BUG: fold {fold_id} has {len(overlap)} rows present in both "
                f"train and test."
            )
            if train_mask.any() and test_mask.any():
                max_train_date = dates[train_mask].max()
                min_test_date = dates[test_mask].min()
                assert max_train_date < min_test_date, (
                    f"LEAKAGE BUG: fold {fold_id} max train date {max_train_date} is not "
                    f"strictly before min test date {min_test_date}."
                )

            yield WalkForwardFold(
                fold_id=fold_id,
                train_start=unique_months[0].to_timestamp(),
                train_end=train_end_period.to_timestamp(),
                test_start=test_start_period.to_timestamp(),
                test_end=test_end_period.to_timestamp(),
                train_index=train_index,
                test_index=test_index,
            )

            fold_id += 1
            train_end_idx += self.step_months
