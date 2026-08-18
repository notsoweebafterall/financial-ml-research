"""
Trains the neural network across all 36 real walk-forward folds and appends its
predictions to the existing data/processed/oos_predictions.parquet (alongside
Ridge/RandomForest/XGBoost from Phase 4), so it plugs directly into Phase 6's
evaluation without needing to rebuild anything there.

Usage (from project root, with .venv active):
    python -m src.models.run_nn
"""

import time
import pandas as pd
from pathlib import Path

from src.validation.walkforward import WalkForwardSplitter
from src.models.nn_trainer import train_nn_one_fold

PANEL_PATH = Path("data/processed/characteristics_panel.parquet")
PREDICTIONS_PATH = Path("data/processed/oos_predictions.parquet")

LABEL_COL = "next_month_return"
DATE_COL = "date"
NON_FEATURE_COLS = {"ticker", "date", LABEL_COL}


def main():
    panel = pd.read_parquet(PANEL_PATH)
    panel[DATE_COL] = pd.to_datetime(panel[DATE_COL])

    # Match the exact trim applied in Phase 4 — the raw panel on disk still has all
    # 120 months (including the 36-month warmup period). Phase 4 trimmed this in
    # memory but never saved the trimmed version back to disk, so we must reapply
    # the same trim here or we'll silently reproduce the "72 folds instead of 36" bug.
    TRIM_START = "2019-09-01"
    before_trim = len(panel)
    panel = panel[panel[DATE_COL] >= TRIM_START].reset_index(drop=True)
    print(f"Trimmed panel to date >= {TRIM_START}: {before_trim} -> {len(panel)} rows "
          f"(matches the fix applied in Phase 4)")

    feature_cols = [c for c in panel.columns if c not in NON_FEATURE_COLS]
    print(f"Using {len(feature_cols)} feature columns.")

    splitter = WalkForwardSplitter(
        date_col=DATE_COL, min_train_months=48, test_window_months=1, step_months=1
    )

    all_predictions = []
    start = time.time()

    for fold in splitter.split(panel):
        train_df = panel.loc[fold.train_index]
        test_df = panel.loc[fold.test_index]

        # Same label-NaN boundary case handled in Phase 5 — skip a test month with
        # no valid label rather than training toward garbage or crashing
        if test_df[LABEL_COL].isna().all():
            print(f"Fold {fold.fold_id}: skipping test month {fold.test_start.date()} "
                  f"(no valid label — boundary month)")
            continue

        result = train_nn_one_fold(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            label_col=LABEL_COL,
            date_col=DATE_COL,
            fold_id=fold.fold_id,
        )
        all_predictions.append(result.predictions)

        elapsed = time.time() - start
        print(f"Fold {fold.fold_id:2d}/{35} | test month {fold.test_start.date()} | "
              f"best_n_epochs={result.best_n_epochs:3d} | "
              f"inner_val_loss={result.inner_val_loss:.6f} | "
              f"elapsed={elapsed:.0f}s")

    nn_predictions = pd.concat(all_predictions, ignore_index=True)
    nn_predictions["date"] = pd.to_datetime(nn_predictions["date"])

    # Append to the existing predictions file from Phase 4, rather than overwriting.
    # Force both frames to a consistent datetime dtype before concatenating — mixing
    # a string-typed date column (however it was saved before) with Timestamp objects
    # produces an 'object' dtype column that pyarrow cannot write, causing a crash.
    existing = pd.read_parquet(PREDICTIONS_PATH)
    existing["date"] = pd.to_datetime(existing["date"])
    combined = pd.concat([existing, nn_predictions], ignore_index=True)
    combined.to_parquet(PREDICTIONS_PATH, index=False)

    print(f"\nDone. Added {len(nn_predictions)} NeuralNet predictions.")
    print(f"Total rows in {PREDICTIONS_PATH}: {len(combined)}")
    print(f"Models now present: {sorted(combined['model_name'].unique())}")


if __name__ == "__main__":
    main()
