import json
import pandas as pd

with open('data/raw/india_source_metadata.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)

print("=== METADATA SUMMARY ===")
print(json.dumps(meta, indent=2))

df_p = pd.read_parquet('data/raw/india_prices.parquet')
print(f"\n=== INDIA PRICES PARQUET (Shape: {df_p.shape}) ===")
print(f"Unique tickers count: {df_p['ticker'].nunique()}")
print(f"Date range: {df_p['date'].min()} to {df_p['date'].max()}")

print("\n=== REPLACEMENT TICKERS CHECK ===")
for tick in ['TVSMOTOR.NS', 'MPHASIS.NS']:
    sub = df_p[df_p['ticker'] == tick]
    min_d = sub['date'].min() if len(sub) > 0 else 'N/A'
    max_d = sub['date'].max() if len(sub) > 0 else 'N/A'
    print(f"Ticker {tick}: {len(sub)} rows, Date Range: {min_d} to {max_d}")

df_snap = pd.read_csv('data/raw/india_market_cap_snapshot.csv')
print(f"\n=== MARKET CAP SNAPSHOT (Shape: {df_snap.shape}) ===")
print(df_snap.head().to_string())
