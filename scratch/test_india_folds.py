import pandas as pd
from src.validation.walkforward import WalkForwardSplitter

df = pd.read_parquet("data/processed/india_characteristics_panel.parquet")
all_dates = sorted(df["date"].unique())
min_date = all_dates[0]
cutoff_date = all_dates[36]
df_trimmed = df[df["date"] >= cutoff_date].reset_index(drop=True)

print(f"India Raw Panel Dates: {min_date} to {all_dates[-1]} (Total {len(all_dates)} months)")
print(f"Trimmed Cutoff Date (36-month warmup): {cutoff_date}")
print(f"Trimmed Panel Rows: {len(df_trimmed)}, Total Months: {df_trimmed['date'].nunique()}")

splitter = WalkForwardSplitter(date_col="date", min_train_months=48, test_window_months=1, step_months=1)
folds = list(splitter.split(df_trimmed))
print(f"Resulting India Walk-Forward Fold Count: {len(folds)}")
