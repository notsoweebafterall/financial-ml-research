import pandas as pd
import numpy as np
from src.validation.walkforward import WalkForwardSplitter


def make_synthetic_panel(n_tickers=60, n_months=84, start="2019-09-01"):
    """Mimics the real characteristics_panel.parquet shape: 60 tickers x ~84 usable months."""
    dates = pd.date_range(start=start, periods=n_months, freq="MS")
    rows = []
    for t in range(n_tickers):
        ticker = f"TICK{t:03d}"
        for d in dates:
            rows.append({"ticker": ticker, "date": d, "some_feature": np.random.randn()})
    return pd.DataFrame(rows)


def test_no_overlap_and_correct_ordering():
    df = make_synthetic_panel()
    splitter = WalkForwardSplitter(min_train_months=48, test_window_months=1, step_months=1)
    folds = list(splitter.split(df))

    assert len(folds) > 0, "No folds produced"

    for fold in folds:
        train_dates = pd.to_datetime(df.loc[fold.train_index, "date"])
        test_dates = pd.to_datetime(df.loc[fold.test_index, "date"])

        # No shared row indices
        assert set(fold.train_index).isdisjoint(set(fold.test_index))

        # Every train date strictly before every test date
        assert train_dates.max() < test_dates.min(), (
            f"Fold {fold.fold_id}: train max {train_dates.max()} not before test min {test_dates.min()}"
        )

        # Test window is exactly 1 month as configured
        assert test_dates.dt.to_period("M").nunique() == 1

    # Folds should be in increasing chronological order
    train_ends = [f.train_end for f in folds]
    assert train_ends == sorted(train_ends)
    print(f"PASS: {len(folds)} folds produced, all leak-free, correctly ordered.")


def test_expanding_window_actually_expands():
    df = make_synthetic_panel()
    splitter = WalkForwardSplitter(min_train_months=48, test_window_months=1, step_months=1)
    folds = list(splitter.split(df))

    train_sizes = [len(f.train_index) for f in folds]
    # Training set size should strictly grow fold over fold (expanding window)
    assert all(train_sizes[i] < train_sizes[i + 1] for i in range(len(train_sizes) - 1)), (
        "Training window did not strictly expand across folds"
    )
    print(f"PASS: training window expands monotonically across {len(folds)} folds "
          f"(from {train_sizes[0]} to {train_sizes[-1]} rows).")


def test_deliberately_broken_splitter_gets_caught():
    """
    Sanity-check the sanity-check: prove the assertions actually fire when there
    IS a leak, so we know they're not just silently passing no matter what.
    """
    df = make_synthetic_panel()
    dates = pd.to_datetime(df["date"])
    month_periods = dates.dt.to_period("M")
    unique_months = sorted(month_periods.unique())

    # Deliberately construct a broken fold: test set includes one row that's
    # ALSO in the train set (classic off-by-one leakage bug)
    train_end_period = unique_months[47]
    test_start_period = unique_months[48]

    train_mask = month_periods <= train_end_period
    test_mask = month_periods == test_start_period

    # Inject a leak: add one training-period row into the test index too
    leaked_row = df.index[train_mask][0]
    broken_test_index = df.index[test_mask].union(pd.Index([leaked_row]))
    broken_train_index = df.index[train_mask]

    overlap = broken_train_index.intersection(broken_test_index)
    try:
        assert len(overlap) == 0, "LEAKAGE BUG: overlap detected (this is SUPPOSED to fail)"
        raise RuntimeError("Test failed: the leak was NOT caught, assertion did not fire!")
    except AssertionError:
        print("PASS: deliberately broken split was correctly caught by the leakage assertion.")


if __name__ == "__main__":
    test_no_overlap_and_correct_ordering()
    test_expanding_window_actually_expands()
    test_deliberately_broken_splitter_gets_caught()
    print("\nAll walk-forward validation tests passed.")