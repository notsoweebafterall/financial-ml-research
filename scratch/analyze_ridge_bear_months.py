import pandas as pd
import numpy as np

# Load India predictions, backtest results, and benchmark
preds = pd.read_parquet("data/processed/india_oos_predictions.parquet")
bkt = pd.read_csv("results/india_phase5_backtest_results.csv")
bmk = pd.read_parquet("data/raw/india_nifty_benchmark.parquet")

# Prepare benchmark trailing 12m return
bmk_sorted = bmk.sort_values("date").reset_index(drop=True)
bmk_sorted["nifty_trailing_12m"] = bmk_sorted["adj_close"].pct_change(12)
bmk_sorted["nifty_return_1m"] = bmk_sorted["adj_close"].pct_change(1)

# Merge backtest with benchmark to identify bear months
test_dates = set(bkt["date"].unique())
spy_test = bmk_sorted[bmk_sorted["date"].isin(test_dates)].copy()
bear_dates = set(spy_test[spy_test["nifty_trailing_12m"] < 0]["date"])

print(f"Total Test Months: {len(test_dates)}, Bear Months: {len(bear_dates)}, Bull Months: {len(test_dates) - len(bear_dates)}")

ridge_bkt = bkt[(bkt["model_name"] == "Ridge") & (bkt["date"].isin(bear_dates))].sort_values("net_return", ascending=False)

print("\n=== RIDGE PERFORMANCE ACROSS ALL BEAR MONTHS ===")
print(ridge_bkt[["date", "gross_return", "transaction_cost", "net_return"]].to_string(index=False))

def inspect_month(target_date):
    sub = preds[(preds["model_name"] == "Ridge") & (preds["date"] == target_date)].dropna(subset=["actual_return"]).copy()
    sub_sorted = sub.sort_values("predicted_return", ascending=False).reset_index(drop=True)
    
    n_stocks = len(sub_sorted)
    k = int(np.ceil(n_stocks * 0.1)) # Top 6 & Bottom 6
    
    longs = sub_sorted.head(k)
    shorts = sub_sorted.tail(k)
    
    nifty_12m = bmk_sorted[bmk_sorted['date']==target_date]['nifty_trailing_12m'].values[0]
    nifty_1m = bmk_sorted[bmk_sorted['date']==target_date]['nifty_return_1m'].values[0]
    
    long_ret = longs['actual_return'].mean()
    short_ret = shorts['actual_return'].mean()
    spread = long_ret - short_ret
    
    print(f"\n=================== MONTH: {target_date} ===================")
    print(f"Nifty 50 1-Month Return: {nifty_1m*100:+.2f}%, Trailing 12-Month: {nifty_12m*100:+.2f}%")
    print(f"Ridge Long Leg Avg: {long_ret*100:+.2f}%, Short Leg Avg: {short_ret*100:+.2f}%, Gross Spread: {spread*100:+.2f}%")
    
    print("\n  LONG LEG (Top 6 Predicted):")
    for _, row in longs.iterrows():
        print(f"    {row['ticker']:15s} | Pred: {row['predicted_return']:+.4f} | Actual Ret: {row['actual_return']*100:+.2f}%")
        
    print("\n  SHORT LEG (Bottom 6 Predicted):")
    for _, row in shorts.iterrows():
        print(f"    {row['ticker']:15s} | Pred: {row['predicted_return']:+.4f} | Actual Ret: {row['actual_return']*100:+.2f}%")

best_bear_dates = ridge_bkt.head(3)["date"].tolist()
worst_bear_dates = ridge_bkt.tail(3)["date"].tolist()

print("\n\n" + "="*70)
print(">>> TOP 3 BEST BEAR MONTHS FOR RIDGE <<<")
print("="*70)
for d in best_bear_dates:
    inspect_month(d)

print("\n\n" + "="*70)
print(">>> BOTTOM 3 WORST BEAR MONTHS FOR RIDGE <<<")
print("="*70)
for d in worst_bear_dates:
    inspect_month(d)
