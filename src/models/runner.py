"""
Shared runner script for Phase 4: Model Training and Out-of-Sample Prediction.
Iterates over expanding walk-forward folds for Ridge, Random Forest, and XGBoost models,
saves predictions, logs hyperparameters, and verifies leak safety.
"""

import os
import json
import time
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from src.validation.walkforward import WalkForwardSplitter
from src.models.ridge_model import RidgeModelTrainer
from src.models.random_forest_model import RandomForestModelTrainer
from src.models.xgboost_model import XGBoostModelTrainer
from collections import defaultdict
import matplotlib.pyplot as plt


def run_phase4(market: str = "us"):
    market = market.lower()
    print(f"=== STARTING PHASE 4 ({market.upper()}): MODEL TRAINING & OUT-OF-SAMPLE PREDICTIONS ===", flush=True)
    t_start = time.time()

    if market == "us":
        data_path = "data/processed/characteristics_panel.parquet"
        output_pred_path = "data/processed/oos_predictions.parquet"
        output_params_path = "results/phase4_selected_hyperparameters.csv"
        feature_importance_path = "results/phase6_feature_importance.csv"
    elif market == "india":
        data_path = "data/processed/india_characteristics_panel.parquet"
        output_pred_path = "data/processed/india_oos_predictions.parquet"
        output_params_path = "results/india_phase4_selected_hyperparameters.csv"
        feature_importance_path = "results/india_phase6_feature_importance.csv"
    else:
        raise ValueError(f"Unknown market '{market}'. Expected 'us' or 'india'.")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input file not found: {data_path}")

    raw_df = pd.read_parquet(data_path)
    print(f"Loaded raw characteristics panel dataset: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns.", flush=True)

    # Dynamically determine min_usable_date after 36-month warmup for ltr_reversal
    all_dates = sorted(raw_df["date"].unique())
    min_usable_date = all_dates[36] if len(all_dates) > 36 else all_dates[0]
    df = raw_df[raw_df["date"] >= min_usable_date].reset_index(drop=True)
    print(
        f"Trimmed dataset to dates >= '{min_usable_date}' (first fully-usable month after 36-month ltr_reversal warmup): "
        f"{df.shape[0]} rows, {df['date'].nunique()} total months.",
        flush=True,
    )

    feature_cols = [c for c in df.columns if c not in ["ticker", "date", "next_month_return"]]
    target_col = "next_month_return"
    print(f"Extracted {len(feature_cols)} characteristic features.", flush=True)

    splitter = WalkForwardSplitter(
        date_col="date",
        min_train_months=48,
        test_window_months=1,
        step_months=1,
    )

    folds = list(splitter.split(df))
    print(f"Generated {len(folds)} expanding walk-forward folds.", flush=True)

    all_predictions: List[pd.DataFrame] = []
    hyperparam_logs: List[Dict[str, Any]] = []
    feature_importance_records: List[Dict[str, Any]] = []

    # Verification tracking for spot checks
    spot_check_results = []

    for fold in folds:
        train_df = df.iloc[fold.train_index]
        test_df = df.iloc[fold.test_index]

        # 1. Ridge Model
        ridge_trainer = RidgeModelTrainer(feature_cols=feature_cols, target_col=target_col)
        ridge_preds, ridge_params = ridge_trainer.train_and_predict(
            fold_id=fold.fold_id, train_df=train_df, test_df=test_df
        )
        all_predictions.append(ridge_preds)
        if ridge_trainer.feature_importance_ is not None:
            for feature, importance in ridge_trainer.feature_importance_.items():
                feature_importance_records.append(
                    {
                        "fold_id": fold.fold_id,
                        "model_name": "Ridge",
                        "feature": feature,
                        "importance": float(importance),
                    }
                )
        hyperparam_logs.append(
            {
                "fold_id": fold.fold_id,
                "model_name": "Ridge",
                "selected_params": json.dumps(ridge_params),
            }
        )

        # 2. Random Forest Model
        rf_trainer = RandomForestModelTrainer(feature_cols=feature_cols, target_col=target_col)
        rf_preds, rf_params = rf_trainer.train_and_predict(
            fold_id=fold.fold_id, train_df=train_df, test_df=test_df
        )
        all_predictions.append(rf_preds)
        if rf_trainer.feature_importance_ is not None:
            for feature, importance in rf_trainer.feature_importance_.items():
                feature_importance_records.append(
                    {
                        "fold_id": fold.fold_id,
                        "model_name": "RandomForest",
                        "feature": feature,
                        "importance": float(importance),
                    }
                )
        hyperparam_logs.append(
            {
                "fold_id": fold.fold_id,
                "model_name": "RandomForest",
                "selected_params": json.dumps(rf_params),
            }
        )

        # 3. XGBoost Model
        xgb_trainer = XGBoostModelTrainer(feature_cols=feature_cols, target_col=target_col)
        xgb_preds, xgb_params = xgb_trainer.train_and_predict(
            fold_id=fold.fold_id, train_df=train_df, test_df=test_df
        )
        all_predictions.append(xgb_preds)
        if xgb_trainer.feature_importance_ is not None:
            for feature, importance in xgb_trainer.feature_importance_.items():
                feature_importance_records.append(
                    {
                        "fold_id": fold.fold_id,
                        "model_name": "XGBoost",
                        "feature": feature,
                        "importance": float(importance),
                    }
                )
        hyperparam_logs.append(
            {
                "fold_id": fold.fold_id,
                "model_name": "XGBoost",
                "selected_params": json.dumps(xgb_params),
            }
        )

        # Spot check imputer on specific folds (e.g. fold 0 and fold 10)
        if fold.fold_id in [0, 10]:
            clean_train = train_df.dropna(subset=[target_col])
            manual_medians = clean_train[feature_cols].median(numeric_only=True).values
            fitted_medians = ridge_trainer.fitted_imputer.statistics_
            max_diff = np.max(np.abs(manual_medians - fitted_medians))

            spot_check_results.append(
                {
                    "fold_id": fold.fold_id,
                    "train_start": str(fold.train_start.date()),
                    "train_end": str(fold.train_end.date()),
                    "train_row_count": len(clean_train),
                    "max_median_diff": max_diff,
                    "is_leak_safe": max_diff < 1e-12,
                }
            )

        if (fold.fold_id + 1) % 5 == 0 or (fold.fold_id + 1) == len(folds):
            print(f"Processed fold {fold.fold_id + 1}/{len(folds)}...", flush=True)

    # Combine all predictions
    oos_preds_df = pd.concat(all_predictions, ignore_index=True)

    # Save out-of-sample predictions
    os.makedirs(os.path.dirname(output_pred_path), exist_ok=True)
    oos_preds_df.to_parquet(output_pred_path, index=False)
    print(f"Saved out-of-sample predictions to '{output_pred_path}' ({len(oos_preds_df)} total rows).", flush=True)

    # Save hyperparameter logs
    hyperparam_df = pd.DataFrame(hyperparam_logs)
    os.makedirs(os.path.dirname(output_params_path), exist_ok=True)
    hyperparam_df.to_csv(output_params_path, index=False)

    # Aggregate feature importance across walk-forward folds.
    feature_importance_df = pd.DataFrame(feature_importance_records)

    if not feature_importance_df.empty:
        feature_importance_summary = (
            feature_importance_df
            .groupby(["model_name", "feature"])["importance"]
            .agg(
                mean_importance="mean",
                std_importance="std",
            )
            .reset_index()
            .sort_values(
                ["model_name", "mean_importance"],
                ascending=[True, False],
            )
        )

        feature_importance_summary.to_csv(
            feature_importance_path,
            index=False,
        )

        print(
            f"Saved feature importance summary to '{feature_importance_path}'.",
            flush=True,
        )

    print(f"Saved hyperparameter log to '{output_params_path}'.", flush=True)

    t_end = time.time()
    print(f"Phase 4 pipeline completed in {t_end - t_start:.2f} seconds.", flush=True)

    # Run verification metrics and display summary
    run_verifications(oos_preds_df, hyperparam_df, spot_check_results)


def run_verifications(
    oos_df: pd.DataFrame, hyperparam_df: pd.DataFrame, spot_check_results: List[Dict[str, Any]]
):
    print("\n" + "=" * 60, flush=True)
    print("PHASE 4 VERIFICATION RESULTS", flush=True)
    print("=" * 60, flush=True)

    # 1. Imputer Leak-Safety Spot Check
    print("\n--- 1. IMPUTER LEAK-SAFETY SPOT CHECK ---", flush=True)
    for check in spot_check_results:
        status = "PASSED" if check["is_leak_safe"] else "FAILED"
        print(
            f"Fold {check['fold_id']} (train dates {check['train_start']} to {check['train_end']}, {check['train_row_count']} rows):",
            flush=True,
        )
        print(f"  Fitted Imputer Statistics vs Manual Train Median Max Diff: {check['max_median_diff']:.2e} -> [{status}]", flush=True)

    # 2. Total Row Counts
    print("\n--- 2. TOTAL PREDICTION ROW COUNTS ---", flush=True)
    total_rows = len(oos_df)
    print(f"Total OOS Prediction Rows: {total_rows}", flush=True)
    for model_name, sub in oos_df.groupby("model_name"):
        print(f"  - {model_name}: {len(sub)} rows across {sub['fold_id'].nunique()} folds", flush=True)

    # 3. Out-of-Sample R2 Metrics & Cross-Sectional Prediction Dispersion
    print("\n--- 3. OUT-OF-SAMPLE R^2 & MONTHLY CROSS-SECTIONAL DISPERSION ---", flush=True)
    valid_preds = oos_df.dropna(subset=["actual_return"]).copy()

    for model_name, sub in oos_df.groupby("model_name"):
        # Filter for valid test target rows for R2 calculation
        sub_valid = sub.dropna(subset=["actual_return"])
        y_true = sub_valid["actual_return"].values
        y_pred = sub_valid["predicted_return"].values

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot_zero = np.sum(y_true**2)
        ss_tot_mean = np.sum((y_true - np.mean(y_true)) ** 2)

        r2_oos_zero = 1.0 - (ss_res / ss_tot_zero)
        r2_sample_mean = 1.0 - (ss_res / ss_tot_mean)

        # Compute monthly cross-sectional standard deviation across stocks
        monthly_cs_std = sub.groupby("date")["predicted_return"].std().mean()

        print(
            f"Model: {model_name:15s} | R^2 (Zero/Gu-Kelly-Xiu): {r2_oos_zero * 100:.3f}% | R^2 (Sample Mean): {r2_sample_mean * 100:.3f}% | Avg Monthly CS Std: {monthly_cs_std:.6f} ({monthly_cs_std*100:.4f}%)",
            flush=True,
        )

    # 4. Hyperparameter Stability Log Preview
    print("\n--- 4. HYPERPARAMETER STABILITY LOG PREVIEW ---", flush=True)
    print("First 9 rows of hyperparameter log:", flush=True)
    print(hyperparam_df.head(9).to_string(index=False), flush=True)

    print("\nSelected Hyperparameter Distribution Summary across folds:", flush=True)
    for model_name, sub in hyperparam_df.groupby("model_name"):
        print(f"\n{model_name} Selected Params Frequency:", flush=True)
        print(sub["selected_params"].value_counts().to_string(), flush=True)

    print("\n" + "=" * 60, flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Phase 4 model training for US or India market.")
    parser.add_argument("--market", type=str, default="us", choices=["us", "india"], help="Target market (default: us)")
    args = parser.parse_args()

    run_phase4(market=args.market)

