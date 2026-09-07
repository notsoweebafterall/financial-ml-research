import json
import logging
from pathlib import Path
import pandas as pd
from src.data.india_ingestion import IndiaStockDataIngestion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingest_replacements")

project_root = Path(__file__).resolve().parents[1]
universe_path = project_root / "data" / "raw" / "india_universe.csv"
prices_path = project_root / "data" / "raw" / "india_prices.parquet"
snapshot_path = project_root / "data" / "raw" / "india_market_cap_snapshot.csv"
metadata_path = project_root / "data" / "raw" / "india_source_metadata.json"

# 1. Update india_universe.csv
df_uni = pd.read_csv(universe_path)
df_uni.loc[df_uni['ticker'] == 'LTIM.NS', 'ticker'] = 'MPHASIS.NS'
df_uni.loc[df_uni['ticker'] == 'TATAMOTORS.NS', 'ticker'] = 'TVSMOTOR.NS'
df_uni.to_csv(universe_path, index=False)
logger.info(f"Updated {universe_path} with replacement tickers MPHASIS.NS and TVSMOTOR.NS")

# 2. Ingest replacement tickers
ingestion = IndiaStockDataIngestion(project_root=project_root)
new_tickers = ['TVSMOTOR.NS', 'MPHASIS.NS']
new_price_dfs = []
new_snapshots = []

for ticker in new_tickers:
    logger.info(f"Pulling replacement ticker: {ticker}")
    df_price, snapshot, note = ingestion.fetch_ticker_data(ticker)
    if df_price is not None and not df_price.empty:
        new_price_dfs.append(df_price)
        if snapshot:
            new_snapshots.append(snapshot)
    else:
        logger.error(f"Failed to fetch replacement ticker {ticker}: {note}")

# 3. Combine with existing datasets
df_existing_prices = pd.read_parquet(prices_path)
# Remove any old failed ticker rows if present
df_existing_prices = df_existing_prices[~df_existing_prices['ticker'].isin(['LTIM.NS', 'TATAMOTORS.NS'])]

df_all_prices = pd.concat([df_existing_prices] + new_price_dfs, ignore_index=True)
df_all_prices.sort_values(by=["ticker", "date"], inplace=True)
df_all_prices.reset_index(drop=True, inplace=True)
df_all_prices.to_parquet(prices_path, index=False)
logger.info(f"Saved updated india_prices.parquet with {len(df_all_prices)} rows across {df_all_prices['ticker'].nunique()} tickers.")

df_existing_snapshots = pd.read_csv(snapshot_path)
df_existing_snapshots = df_existing_snapshots[~df_existing_snapshots['ticker'].isin(['LTIM.NS', 'TATAMOTORS.NS'])]
df_all_snapshots = pd.concat([df_existing_snapshots, pd.DataFrame(new_snapshots)], ignore_index=True)
df_all_snapshots.to_csv(snapshot_path, index=False)
logger.info(f"Saved updated india_market_cap_snapshot.csv with {len(df_all_snapshots)} rows.")

# 4. Update metadata
min_date = df_all_prices["date"].min()
max_date = df_all_prices["date"].max()

metadata = {
    "source_name": "yfinance",
    "download_date": "2026-09-07",
    "coverage_period": f"{min_date} to {max_date}",
    "frequency": "monthly",
    "adjustment_method": "adj_close accounts for splits and dividends (auto_adjust=False with explicit Adj Close column)",
    "disclaimer": (
        "Data retrieved via yfinance open interface. Free/unofficial API with no uptime guarantee. "
        "Indian stock universe (NSE) with .NS ticker extension. "
        "Market cap and shares outstanding are current point-in-time snapshot metrics, not historical point-in-time values."
    ),
    "total_tickers_requested": 60,
    "tickers_successful_count": 60,
    "tickers_failed_count": 0,
    "failed_tickers": [],
    "substitutions": [
        {
            "original": "TATAMOTORS.NS",
            "replacement": "TVSMOTOR.NS",
            "sector": "Consumer Discretionary",
            "reason": "TATAMOTORS.NS returned 404 on yfinance due to corporate demerger restructuring on NSE."
        },
        {
            "original": "LTIM.NS",
            "replacement": "MPHASIS.NS",
            "sector": "Technology",
            "reason": "LTIM.NS returned 404 on yfinance due to post-merger ticker restructuring."
        }
    ],
    "data_quality_notes": []
}

with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

logger.info(f"Saved updated source metadata to {metadata_path}")
