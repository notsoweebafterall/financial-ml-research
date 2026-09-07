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


def main(market: str = "us"):
    market = market.lower()
    print(f"=== RUNNING NEURAL NETWORK ({market.upper()}) ===", flush=True)

    if market == "us":
        panel_path = Path("data/processed/characteristics_panel.parquet")
        predictions_path = Path("data/processed/oos_predictions.parquet")
    elif market == "india":
        panel_path = Path("data/processed/india_characteristics_panel.parquet")
        predictions_path = Path("data/processed/india_oos_predictions.parquet")
    else:
        raise ValueError(f"Unknown market '{market}'. Expected 'us' or 'india'.")

    panel = pd.read_parquet(panel_path)
    panel[DATE_COL] = pd.to_datetime(panel[DATE_COL])

    # Dynamic 36-month warmup date trim
    all_dates = sorted(panel[DATE_COL].unique())
    trim_start = all_dates[36] if len(all_dates) > 36 else all_dates[0]
    before_trim = len(panel)
    panel = panel[panel[DATE_COL] >= trim_start].reset_index(drop=True)
    print(f"Trimmed panel to date >= {trim_start.date()}: {before_trim} -> {len(panel)} rows "
          f"(after 36-month ltr_reversal warmup)")

    feature_cols = [c for c in panel.columns if c not in NON_FEATURE_COLS]
    print(f"Using {len(feature_cols)} feature columns.")

    splitter = WalkForwardSplitter(
        date_col=DATE_COL, min_train_months=48, test_window_months=1, step_months=1
    )

    folds = list(splitter.split(panel))
    total_folds = len(folds)
    print(f"Generated {total_folds} expanding walk-forward folds.")

    all_predictions = []
    start = time.time()

    for fold in folds:
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
        print(f"Fold {fold.fold_id:2d}/{total_folds-1} | test month {fold.test_start.date()} | "
              f"best_n_epochs={result.best_n_epochs:3d} | "
              f"inner_val_loss={result.inner_val_loss:.6f} | "
              f"elapsed={elapsed:.0f}s")

    nn_predictions = pd.concat(all_predictions, ignore_index=True)
    nn_predictions["date"] = pd.to_datetime(nn_predictions["date"])

    # Append NeuralNet predictions to existing oos_predictions
    existing = pd.read_parquet(predictions_path)
    existing["date"] = pd.to_datetime(existing["date"])
    # Remove any existing NeuralNet predictions if re-running to avoid duplicates
    existing = existing[existing["model_name"] != "NeuralNet"]
    combined = pd.concat([existing, nn_predictions], ignore_index=True)
    combined.to_parquet(predictions_path, index=False)

    print(f"\nDone. Added {len(nn_predictions)} NeuralNet predictions.")
    print(f"Total rows in {predictions_path}: {len(combined)}")
    print(f"Models now present: {sorted(combined['model_name'].unique())}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Neural Network model training for US or India market.")
    parser.add_argument("--market", type=str, default="us", choices=["us", "india"], help="Target market (default: us)")
    args = parser.parse_args()

    main(market=args.market)

