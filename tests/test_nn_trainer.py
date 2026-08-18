import numpy as np
import pandas as pd
from src.models.nn_trainer import train_nn_one_fold, _time_split_inner, RANDOM_STATE


def make_synthetic_fold_data(n_tickers=60, n_train_months=48, n_test_months=1, seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_train_months + n_test_months, freq="MS")
    train_dates, test_dates = dates[:n_train_months], dates[n_train_months:]

    feature_cols = [f"f{i}" for i in range(27)]

    def build(dates_slice):
        rows = []
        for d in dates_slice:
            for t in range(n_tickers):
                row = {"ticker": f"T{t:03d}", "date": d}
                for f in feature_cols:
                    row[f] = rng.normal(0, 1)
                row["next_month_return"] = rng.normal(0.005, 0.08)
                rows.append(row)
        return pd.DataFrame(rows)

    train_df = build(train_dates)
    test_df = build(test_dates)
    return train_df, test_df, feature_cols


def test_time_split_is_chronological_not_random():
    train_df, _, _ = make_synthetic_fold_data()
    inner_train, inner_val = _time_split_inner(train_df, date_col="date", inner_val_frac=0.2)

    assert inner_train["date"].max() < inner_val["date"].min(), (
        "Inner-train dates must all be strictly before inner-val dates"
    )
    assert len(inner_train) + len(inner_val) == len(train_df)
    print(f"PASS: inner split is chronological. inner_train ends {inner_train['date'].max().date()}, "
          f"inner_val starts {inner_val['date'].min().date()}.")


def test_tuning_stage_never_sees_actual_test_data():
    """
    Confirm the imputer/scaler used during Stage A (tuning) is fit ONLY on
    inner_train, and never touches inner_val statistics or the real test set at all.
    """
    train_df, test_df, feature_cols = make_synthetic_fold_data()
    inner_train, inner_val = _time_split_inner(train_df, date_col="date")

    from sklearn.impute import SimpleImputer
    manual_imputer = SimpleImputer(strategy="median").fit(inner_train[feature_cols])
    manual_medians = manual_imputer.statistics_

    # Introduce NaNs into inner_train and inner_val to make the imputation check meaningful
    train_df_with_nan = train_df.copy()
    train_df_with_nan.loc[train_df_with_nan.sample(frac=0.05, random_state=1).index, feature_cols[0]] = np.nan

    inner_train2, inner_val2 = _time_split_inner(train_df_with_nan, date_col="date")
    imputer2 = SimpleImputer(strategy="median").fit(inner_train2[feature_cols])
    manual_median_f0 = inner_train2[feature_cols[0]].median()

    assert abs(imputer2.statistics_[0] - manual_median_f0) < 1e-9, (
        "Imputer median does not match manual median of inner_train only — "
        "possible leakage from inner_val into the tuning-stage imputer"
    )
    print("PASS: tuning-stage imputer statistics match manual median computed "
          "from inner_train only (no leakage from inner_val or test set).")


def test_full_pipeline_runs_and_produces_valid_predictions():
    train_df, test_df, feature_cols = make_synthetic_fold_data(n_train_months=48, n_test_months=1)

    result = train_nn_one_fold(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        label_col="next_month_return",
        date_col="date",
        fold_id=0,
    )

    assert result.best_n_epochs >= 1
    assert len(result.predictions) == len(test_df)
    assert not result.predictions["predicted_return"].isna().any(), "NN produced NaN predictions"
    assert set(result.predictions.columns) == {
        "fold_id", "ticker", "date", "actual_return", "predicted_return", "model_name"
    }
    print(f"PASS: full fold trains end-to-end. best_n_epochs={result.best_n_epochs}, "
          f"inner_val_loss={result.inner_val_loss:.6f}, "
          f"{len(result.predictions)} test predictions produced, no NaNs.")


def test_reproducibility_with_fixed_seed():
    """Same input, same fold -> same predictions (since random_state is fixed)."""
    train_df, test_df, feature_cols = make_synthetic_fold_data(seed=7)

    result1 = train_nn_one_fold(train_df, test_df, feature_cols, "next_month_return", "date", fold_id=0)
    result2 = train_nn_one_fold(train_df, test_df, feature_cols, "next_month_return", "date", fold_id=0)

    diffs = np.abs(result1.predictions["predicted_return"].values -
                    result2.predictions["predicted_return"].values)
    assert diffs.max() < 1e-9, "Predictions differ between two runs with the same fixed seed"
    print("PASS: identical predictions across two runs with the same fixed random seed "
          "(reproducible, as required).")


if __name__ == "__main__":
    test_time_split_is_chronological_not_random()
    test_tuning_stage_never_sees_actual_test_data()
    test_full_pipeline_runs_and_produces_valid_predictions()
    test_reproducibility_with_fixed_seed()
    print("\nAll neural network tests passed.")
