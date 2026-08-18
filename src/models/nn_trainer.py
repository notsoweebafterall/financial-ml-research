"""
Neural network trainer for the walk-forward panel.

Uses sklearn's MLPRegressor with a SMALL architecture (appropriate for ~3,000-5,000
rows / 27 features — this is not deep-learning-scale data, and an oversized network
would just overfit noise).

Key design decisions, and why:

1. Architecture: (32, 16) hidden layers, ReLU. Small on purpose.

2. Early stopping is done MANUALLY, not via sklearn's built-in
   `early_stopping=True` — because that built-in option does a RANDOM split of the
   training data to create its internal validation set, which would violate time
   ordering (the same mistake we've been avoiding throughout this project). Instead:
   the LAST 20% of each fold's training window (by date) is held out as an inner
   validation set, exactly like the tuning approach used for Ridge/RF/XGBoost in
   Phase 4.

3. Two-stage fit, same pattern as Phase 4:
   Stage A (tuning): fit imputer+scaler on inner-train only (first 80% of the
   training window), train epoch-by-epoch, monitor loss on inner-val (last 20%),
   stop when validation loss hasn't improved for `patience` epochs, remember how
   many epochs that took (best_n_epochs).
   Stage B (final): fit a FRESH imputer+scaler on the FULL training window (now
   that we know how many epochs to use), train for exactly best_n_epochs, then
   predict on the actual held-out test month.

4. Random seed is fixed for reproducibility — NN weight initialization is
   stochastic by default, and we want the same result if this is re-run.
"""

from dataclasses import dataclass
from typing import Optional
import warnings
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

# Expected and harmless: each .fit() call is deliberately exactly 1 epoch
# (max_iter=1, warm_start=True), so sklearn's "hasn't converged in max_iter"
# warning fires every single epoch by design. Silencing it here so real
# warnings aren't buried under ~hundreds of expected ones.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


RANDOM_STATE = 42
HIDDEN_LAYERS = (32, 16)
MAX_EPOCHS = 300
PATIENCE = 15


@dataclass
class NNFoldResult:
    fold_id: int
    best_n_epochs: int
    inner_val_loss: float
    predictions: pd.DataFrame  # columns: ticker, date, actual_return, predicted_return


def _time_split_inner(train_df: pd.DataFrame, date_col: str, inner_val_frac: float = 0.2):
    """Split a training window into inner-train (first 80%) / inner-val (last 20%)
    strictly by date, not randomly."""
    dates_sorted = train_df[date_col].sort_values().unique()
    cutoff_idx = int(len(dates_sorted) * (1 - inner_val_frac))
    cutoff_date = dates_sorted[cutoff_idx]
    inner_train = train_df[train_df[date_col] < cutoff_date]
    inner_val = train_df[train_df[date_col] >= cutoff_date]
    return inner_train, inner_val


def _make_model(random_state=RANDOM_STATE):
    return MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        alpha=1e-3,          # L2 regularization, modest default given small data
        learning_rate_init=1e-3,
        warm_start=True,     # allows repeated .fit() calls to continue training
        max_iter=1,          # each .fit() call = exactly 1 epoch, when warm_start=True
        random_state=random_state,
    )


def train_nn_one_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    label_col: str,
    date_col: str,
    fold_id: int,
) -> NNFoldResult:
    # --- Stage A: tuning (find best_n_epochs via time-respecting inner validation) ---
    inner_train, inner_val = _time_split_inner(train_df, date_col)

    imputer_tune = SimpleImputer(strategy="median").fit(inner_train[feature_cols])
    X_inner_train = imputer_tune.transform(inner_train[feature_cols])
    X_inner_val = imputer_tune.transform(inner_val[feature_cols])

    scaler_tune = StandardScaler().fit(X_inner_train)
    X_inner_train = scaler_tune.transform(X_inner_train)
    X_inner_val = scaler_tune.transform(X_inner_val)

    y_inner_train = inner_train[label_col].values
    y_inner_val = inner_val[label_col].values

    model = _make_model()
    best_val_loss = np.inf
    best_n_epochs = 1
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.fit(X_inner_train, y_inner_train)  # one more epoch (warm_start=True)
        val_pred = model.predict(X_inner_val)
        val_loss = float(np.mean((val_pred - y_inner_val) ** 2))

        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_n_epochs = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            break

    # --- Stage B: final fit on the FULL training window using best_n_epochs ---
    imputer_final = SimpleImputer(strategy="median").fit(train_df[feature_cols])
    X_train_full = imputer_final.transform(train_df[feature_cols])
    scaler_final = StandardScaler().fit(X_train_full)
    X_train_full = scaler_final.transform(X_train_full)
    y_train_full = train_df[label_col].values

    final_model = _make_model()
    for _ in range(best_n_epochs):
        final_model.fit(X_train_full, y_train_full)

    X_test = imputer_final.transform(test_df[feature_cols])
    X_test = scaler_final.transform(X_test)
    test_predictions = final_model.predict(X_test)

    pred_df = pd.DataFrame({
        "fold_id": fold_id,
        "ticker": test_df["ticker"].values,
        "date": test_df[date_col].values,
        "actual_return": test_df[label_col].values,
        "predicted_return": test_predictions,
        "model_name": "NeuralNet",
    })

    return NNFoldResult(
        fold_id=fold_id,
        best_n_epochs=best_n_epochs,
        inner_val_loss=best_val_loss,
        predictions=pred_df,
    )
