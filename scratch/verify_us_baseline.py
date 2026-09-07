import pandas as pd
import hashlib

def file_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

print("=== EXISTING US BASELINE SNAPSHOT ===")

cp = pd.read_parquet("data/processed/characteristics_panel.parquet")
print(f"characteristics_panel.parquet shape: {cp.shape}, hash: {file_hash('data/processed/characteristics_panel.parquet')}")

oos = pd.read_parquet("data/processed/oos_predictions.parquet")
print(f"oos_predictions.parquet shape: {oos.shape}, hash: {file_hash('data/processed/oos_predictions.parquet')}")

bkt = pd.read_csv("results/phase5_backtest_results.csv")
print(f"phase5_backtest_results.csv shape: {bkt.shape}, hash: {file_hash('results/phase5_backtest_results.csv')}")

perf = pd.read_csv("results/phase6_performance_summary.csv")
print(f"phase6_performance_summary.csv shape: {perf.shape}")
print(perf.to_string(index=False))
