"""
Data Ingestion Module for Empirical Asset Pricing Research (Gu, Kelly & Xiu 2020 Replication)

Pulls ~10 years of monthly price/volume data and fundamental snapshot metrics
for a specified stock universe using yfinance.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("src.data.ingestion")


class StockDataIngestion:
    """Handles rate-limited, fault-tolerant ingestion of stock data from yfinance."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        period: str = "10y",
        interval: str = "1mo",
        request_delay: float = 0.3,
    ):
        if project_root is None:
            # Resolves root directory assuming script is in src/data/ingestion.py
            project_root = Path(__file__).resolve().parents[2]
        self.project_root = Path(project_root)
        self.period = period
        self.interval = interval
        self.request_delay = request_delay

        self.universe_path = self.project_root / "data" / "raw" / "universe.csv"
        self.prices_output_path = self.project_root / "data" / "raw" / "prices.parquet"
        self.snapshot_output_path = self.project_root / "data" / "raw" / "market_cap_snapshot.csv"
        self.metadata_output_path = self.project_root / "data" / "raw" / "source_metadata.json"

    def load_universe(self) -> pd.DataFrame:
        """Loads ticker universe from CSV."""
        if not self.universe_path.exists():
            raise FileNotFoundError(f"Universe file not found at {self.universe_path}")
        df_universe = pd.read_csv(self.universe_path)
        if "ticker" not in df_universe.columns:
            raise ValueError("universe.csv must contain a 'ticker' column")
        logger.info(f"Loaded universe with {len(df_universe)} tickers from {self.universe_path}")
        return df_universe

    def fetch_ticker_data(
        self, ticker: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Optional[float]]]]:
        """
        Pulls monthly price history and current fundamental snapshot metrics for a single ticker.
        """
        try:
            t = yf.Ticker(ticker)
            # Fetch price history with auto_adjust=False to get explicit Close and Adj Close
            df_hist = t.history(period=self.period, interval=self.interval, auto_adjust=False)

            if df_hist.empty:
                logger.warning(f"No history returned for ticker '{ticker}'")
                return None, None

            # Handle MultiIndex columns if returned by newer yfinance versions
            if isinstance(df_hist.columns, pd.MultiIndex):
                df_hist.columns = df_hist.columns.get_level_values(0)

            # Reset index to access Date/Datetime
            df_hist = df_hist.reset_index()

            # Find date column name ('Date' or 'Datetime')
            date_col = None
            for col in ["Date", "Datetime", "date", "datetime"]:
                if col in df_hist.columns:
                    date_col = col
                    break
            if date_col is None:
                logger.warning(f"No date column found in history for ticker '{ticker}'")
                return None, None

            # Standardize date format to YYYY-MM-DD
            df_hist["date"] = pd.to_datetime(df_hist[date_col]).dt.strftime("%Y-%m-%d")

            # Map and validate required columns
            col_map = {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
            
            # Filter and rename
            present_cols = [c for c in col_map.keys() if c in df_hist.columns]
            df_clean = df_hist[["date"] + present_cols].rename(columns=col_map).copy()
            
            # If adj_close missing, fallback to close
            if "adj_close" not in df_clean.columns and "close" in df_clean.columns:
                df_clean["adj_close"] = df_clean["close"]

            df_clean["ticker"] = ticker

            # Ensure canonical column order
            required_order = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
            df_clean = df_clean[required_order].copy()

            # Fetch market cap / shares outstanding snapshot
            mcap = None
            shares = None
            try:
                fast_info = getattr(t, "fast_info", None)
                if fast_info:
                    mcap = fast_info.get("market_cap") or fast_info.get("marketCap")
                    shares = fast_info.get("shares") or fast_info.get("sharesOutstanding")
            except Exception as e:
                logger.debug(f"Could not retrieve fast_info for {ticker}: {e}")

            if mcap is None or shares is None:
                try:
                    info = t.info or {}
                    if mcap is None:
                        mcap = info.get("marketCap")
                    if shares is None:
                        shares = info.get("sharesOutstanding")
                except Exception as e:
                    logger.debug(f"Could not retrieve info for {ticker}: {e}")

            snapshot_info = {
                "ticker": ticker,
                "market_cap": float(mcap) if mcap is not None else None,
                "shares_outstanding": float(shares) if shares is not None else None,
                "snapshot_date": datetime.today().strftime("%Y-%m-%d"),
            }

            return df_clean, snapshot_info

        except Exception as exc:
            logger.error(f"Error fetching data for ticker '{ticker}': {exc}")
            return None, None

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Executes ingestion for all tickers in the universe.
        """
        df_universe = self.load_universe()
        tickers = df_universe["ticker"].unique().tolist()

        price_dfs: List[pd.DataFrame] = []
        snapshots: List[Dict] = []
        failed_tickers: List[Dict[str, str]] = []

        logger.info(f"Starting ingestion for {len(tickers)} tickers...")

        for idx, ticker in enumerate(tickers, start=1):
            logger.info(f"[{idx}/{len(tickers)}] Pulling ticker: {ticker}")
            df_price, snapshot = self.fetch_ticker_data(ticker)

            if df_price is not None and not df_price.empty:
                price_dfs.append(df_price)
                if snapshot:
                    snapshots.append(snapshot)
            else:
                failed_tickers.append({"ticker": ticker, "reason": "No data returned or error occurred"})

            time.sleep(self.request_delay)

        if not price_dfs:
            raise RuntimeError("Ingestion failed for all tickers in universe.")

        df_all_prices = pd.concat(price_dfs, ignore_index=True)
        df_all_prices.sort_values(by=["ticker", "date"], inplace=True)
        df_all_prices.reset_index(drop=True, inplace=True)

        df_snapshots = pd.DataFrame(snapshots)

        # Save prices.parquet
        self.prices_output_path.parent.mkdir(parents=True, exist_ok=True)
        df_all_prices.to_parquet(self.prices_output_path, index=False)
        logger.info(f"Saved combined prices dataset ({len(df_all_prices)} rows) to {self.prices_output_path}")

        # Save market_cap_snapshot.csv
        df_snapshots.to_csv(self.snapshot_output_path, index=False)
        logger.info(f"Saved market cap snapshot ({len(df_snapshots)} rows) to {self.snapshot_output_path}")

        # Generate metadata
        min_date = df_all_prices["date"].min()
        max_date = df_all_prices["date"].max()

        metadata = {
            "source_name": "yfinance",
            "download_date": datetime.today().strftime("%Y-%m-%d"),
            "coverage_period": f"{min_date} to {max_date}",
            "frequency": "monthly",
            "adjustment_method": "adj_close accounts for splits and dividends (auto_adjust=False with explicit Adj Close column)",
            "disclaimer": (
                "Data retrieved via yfinance open interface. Free/unofficial API with no uptime guarantee. "
                "Current universe suffers from survivorship bias (only active tickers included). "
                "Market cap and shares outstanding are current point-in-time snapshot metrics, not historical point-in-time values."
            ),
            "total_tickers_requested": len(tickers),
            "tickers_successful_count": len(price_dfs),
            "tickers_failed_count": len(failed_tickers),
            "failed_tickers": failed_tickers,
        }

        with open(self.metadata_output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved source metadata to {self.metadata_output_path}")

        # Fetch SPY benchmark data
        self.fetch_spy_benchmark()

        return df_all_prices, df_snapshots, metadata

    def fetch_spy_benchmark(self) -> pd.DataFrame:
        """Pulls ~10 years of monthly price history for SPY benchmark."""
        spy_output_path = self.project_root / "data" / "raw" / "spy_benchmark.parquet"
        logger.info("Fetching SPY benchmark data...")
        df_spy, _ = self.fetch_ticker_data("SPY")
        if df_spy is not None and not df_spy.empty:
            df_spy.to_parquet(spy_output_path, index=False)
            logger.info(f"Saved SPY benchmark data ({len(df_spy)} rows) to {spy_output_path}")
            return df_spy
        else:
            raise RuntimeError("Failed to fetch SPY benchmark data.")


def run_ingestion() -> None:
    ingestion = StockDataIngestion()
    ingestion.run()


if __name__ == "__main__":
    run_ingestion()
