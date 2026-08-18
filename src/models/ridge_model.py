"""
Ridge Regression trainer with leak-safe imputation, standard scaling, and time-respecting inner validation tuning.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class RidgeModelTrainer:
    def __init__(self, feature_cols: List[str], target_col: str = "next_month_return"):
        self.feature_cols = feature_cols
        self.target_col = target_col
        # Expanded alpha grid including strong regularization values
        self.grid_alphas = [
            0.1,
            1.0,
            10.0,
            100.0,
            500.0,
            1000.0,
            2500.0,
            5000.0,
            10000.0,
            25000.0,
            50000.0,
            100000.0,
        ]
        self.fitted_imputer: SimpleImputer | None = None
        self.fitted_scaler: StandardScaler | None = None

    def train_and_predict(
        self, fold_id: int, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes two-stage fit for Ridge regression:
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

        # Fit Imputer and Scaler ONLY on inner_train
        tuning_imputer = SimpleImputer(strategy="median")
        X_it_imp = tuning_imputer.fit_transform(inner_train_df[self.feature_cols])

        tuning_scaler = StandardScaler()
        X_it_scaled = tuning_scaler.fit_transform(X_it_imp)
        y_it = inner_train_df[self.target_col].values

        # Transform inner_val using inner_train fitted objects
        X_iv_imp = tuning_imputer.transform(inner_val_df[self.feature_cols])
        X_iv_scaled = tuning_scaler.transform(X_iv_imp)
        y_iv = inner_val_df[self.target_col].values

        best_alpha = self.grid_alphas[0]
        best_mse = float("inf")

        for alpha in self.grid_alphas:
            model = Ridge(alpha=alpha)
            model.fit(X_it_scaled, y_it)
            preds = model.predict(X_iv_scaled)
            mse = np.mean((y_iv - preds) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_alpha = alpha

        # --- Stage 2: Final Model Refit on Full Training Window ---
        self.fitted_imputer = SimpleImputer(strategy="median")
        X_train_imp = self.fitted_imputer.fit_transform(clean_train[self.feature_cols])

        self.fitted_scaler = StandardScaler()
        X_train_scaled = self.fitted_scaler.fit_transform(X_train_imp)
        y_train = clean_train[self.target_col].values

        final_model = Ridge(alpha=best_alpha)
        final_model.fit(X_train_scaled, y_train)

        # Transform test set using full-window fitted objects
        X_test_imp = self.fitted_imputer.transform(test_df[self.feature_cols])
        X_test_scaled = self.fitted_scaler.transform(X_test_imp)

        test_preds = final_model.predict(X_test_scaled)

        pred_df = pd.DataFrame(
            {
                "fold_id": fold_id,
                "ticker": test_df["ticker"].values,
                "date": test_df["date"].values,
                "actual_return": test_df[self.target_col].values,
                "predicted_return": test_preds,
                "model_name": "Ridge",
            }
        )

        selected_params = {"alpha": best_alpha}
        return pred_df, selected_params
