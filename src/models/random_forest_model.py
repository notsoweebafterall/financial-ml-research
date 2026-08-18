"""
Random Forest Regressor trainer with leak-safe imputation and time-respecting inner validation tuning.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


class RandomForestModelTrainer:
    def __init__(self, feature_cols: List[str], target_col: str = "next_month_return"):
        self.feature_cols = feature_cols
        self.target_col = target_col
        # Expanded grid with lower/shallower depth options
        self.grid_n_estimators = [50, 100, 200]
        self.grid_max_depth = [3, 5, 10]
        self.fitted_imputer: SimpleImputer | None = None

    def train_and_predict(
        self, fold_id: int, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes two-stage fit for Random Forest Regressor:
        1. Inner validation tuning on train_df (first 80% dates inner train, last 20% dates inner val)
        2. Final refit on full train_df and prediction on test_df
        """
        clean_train = train_df.dropna(subset=[self.target_col]).copy()

        # --- Stage 1: Inner Validation Tuning ---
        unique_train_dates = sorted(clean_train["date"].unique())
        n_dates = len(unique_train_dates)
        val_cutoff_idx = int(n_dates * 0.8)
        inner_train_dates = set(unique_train_dates[:val_cutoff_idx])
        inner_val_dates = set(unique_train_dates[val_cutoff_idx:])

        inner_train_df = clean_train[clean_train["date"].isin(inner_train_dates)]
        inner_val_df = clean_train[clean_train["date"].isin(inner_val_dates)]

        # Fit Imputer ONLY on inner_train
        tuning_imputer = SimpleImputer(strategy="median")
        X_it_imp = tuning_imputer.fit_transform(inner_train_df[self.feature_cols])
        y_it = inner_train_df[self.target_col].values

        # Transform inner_val using inner_train fitted imputer
        X_iv_imp = tuning_imputer.transform(inner_val_df[self.feature_cols])
        y_iv = inner_val_df[self.target_col].values

        best_params = {"n_estimators": 100, "max_depth": 5}
        best_mse = float("inf")

        for n_est in self.grid_n_estimators:
            for depth in self.grid_max_depth:
                model = RandomForestRegressor(
                    n_estimators=n_est,
                    max_depth=depth,
                    random_state=42,
                    n_jobs=-1,
                )
                model.fit(X_it_imp, y_it)
                preds = model.predict(X_iv_imp)
                mse = np.mean((y_iv - preds) ** 2)
                if mse < best_mse:
                    best_mse = mse
                    best_params = {"n_estimators": n_est, "max_depth": depth}

        # --- Stage 2: Final Model Refit on Full Training Window ---
        self.fitted_imputer = SimpleImputer(strategy="median")
        X_train_imp = self.fitted_imputer.fit_transform(clean_train[self.feature_cols])
        y_train = clean_train[self.target_col].values

        final_model = RandomForestRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            random_state=42,
            n_jobs=-1,
        )
        final_model.fit(X_train_imp, y_train)

        # Transform test set using full-window fitted imputer
        X_test_imp = self.fitted_imputer.transform(test_df[self.feature_cols])
        test_preds = final_model.predict(X_test_imp)

        pred_df = pd.DataFrame(
            {
                "fold_id": fold_id,
                "ticker": test_df["ticker"].values,
                "date": test_df["date"].values,
                "actual_return": test_df[self.target_col].values,
                "predicted_return": test_preds,
                "model_name": "RandomForest",
            }
        )

        return pred_df, best_params
