"""
Characteristic Engineering Module for Gu, Kelly & Xiu (2020) Replication.

Computes 27 point-in-time characteristics per (ticker, month) using backward-looking data up to month t.
Computes target label next_month_return using adj_close at t+1 relative to t.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("src.features.builder")

FEATURE_COLUMNS = [
    "mom_1m",
    "mom_3m",
    "mom_6m",
    "mom_9m",
    "mom_12m",
    "mom_12m_ex_1m",
    "mom_24m",
    "str_reversal",
    "ltr_reversal",
    "vol_3m",
    "vol_6m",
    "vol_12m",
    "downside_vol_12m",
    "max_ret_12m",
    "avg_dollar_volume_3m",
    "avg_dollar_volume_12m",
    "volume_trend",
    "amihud_illiquidity",
    "size_proxy",
    "price_to_52w_high",
    "price_to_52w_low",
    "dist_from_ma_6m",
    "dist_from_ma_12m",
    "skew_12m",
    "kurt_12m",
    "beta_12m",
    "idio_vol_12m",
]


class CharacteristicBuilder:
    """Builds point-in-time compliant characteristics and target labels."""

    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]
        self.project_root = Path(project_root)

        self.prices_path = self.project_root / "data" / "raw" / "prices.parquet"
        self.spy_path = self.project_root / "data" / "raw" / "spy_benchmark.parquet"
        self.output_path = (
            self.project_root / "data" / "processed" / "characteristics_panel.parquet"
        )

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads price dataset and SPY benchmark dataset."""
        if not self.prices_path.exists():
            raise FileNotFoundError(f"Prices parquet not found at {self.prices_path}")
        if not self.spy_path.exists():
            raise FileNotFoundError(f"SPY parquet not found at {self.spy_path}")

        df_prices = pd.read_parquet(self.prices_path)
        df_spy = pd.read_parquet(self.spy_path)

        logger.info(
            f"Loaded prices dataset ({len(df_prices)} rows, {df_prices['ticker'].nunique()} tickers)"
        )
        logger.info(f"Loaded SPY benchmark dataset ({len(df_spy)} rows)")

        return df_prices, df_spy

    @staticmethod
    def _compute_spy_returns(df_spy: pd.DataFrame) -> pd.DataFrame:
        """Computes SPY 1-month returns from adj_close."""
        df_spy_sorted = df_spy.sort_values("date").reset_index(drop=True)
        spy_adj_close = df_spy_sorted["adj_close"].values
        spy_ret = np.full(len(df_spy_sorted), np.nan)
        if len(df_spy_sorted) > 1:
            spy_ret[1:] = (spy_adj_close[1:] / spy_adj_close[:-1]) - 1.0
        return pd.DataFrame({"date": df_spy_sorted["date"], "spy_ret_1m": spy_ret})

    @staticmethod
    def _compute_ticker_characteristics(
        df_ticker: pd.DataFrame, df_spy_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Computes 27 characteristics and next_month_return label for a single ticker.
        """
        df = df_ticker.copy().sort_values("date").reset_index(drop=True)

        adj_close = df["adj_close"].values
        close = df["close"].values
        volume = df["volume"].values
        n = len(df)

        # 1-month return r_t = (adj_close_t / adj_close_{t-1}) - 1
        ret_1m = np.full(n, np.nan)
        if n > 1:
            ret_1m[1:] = (adj_close[1:] / adj_close[:-1]) - 1.0
        df["ret_1m"] = ret_1m

        # Target Label: next_month_return = (adj_close_{t+1} / adj_close_t) - 1
        next_month_return = np.full(n, np.nan)
        if n > 1:
            next_month_return[:-1] = (adj_close[1:] / adj_close[:-1]) - 1.0
        df["next_month_return"] = next_month_return

        # Dollar volume DV_t = close_t * volume_t
        dollar_vol = close * volume
        df["dollar_vol"] = dollar_vol

        # Merge SPY benchmark return
        df = df.merge(df_spy_returns, on="date", how="left")
        spy_ret = df["spy_ret_1m"].values

        # Initialize feature arrays
        mom_1m = np.full(n, np.nan)
        mom_3m = np.full(n, np.nan)
        mom_6m = np.full(n, np.nan)
        mom_9m = np.full(n, np.nan)
        mom_12m = np.full(n, np.nan)
        mom_12m_ex_1m = np.full(n, np.nan)
        mom_24m = np.full(n, np.nan)

        str_reversal = np.full(n, np.nan)
        ltr_reversal = np.full(n, np.nan)

        vol_3m = np.full(n, np.nan)
        vol_6m = np.full(n, np.nan)
        vol_12m = np.full(n, np.nan)
        downside_vol_12m = np.full(n, np.nan)
        max_ret_12m = np.full(n, np.nan)

        avg_dv_3m = np.full(n, np.nan)
        avg_dv_12m = np.full(n, np.nan)
        volume_trend = np.full(n, np.nan)
        amihud_illiquidity = np.full(n, np.nan)
        size_proxy = np.full(n, np.nan)

        p_52w_high = np.full(n, np.nan)
        p_52w_low = np.full(n, np.nan)
        dist_ma_6m = np.full(n, np.nan)
        dist_ma_12m = np.full(n, np.nan)

        skew_12m = np.full(n, np.nan)
        kurt_12m = np.full(n, np.nan)

        beta_12m = np.full(n, np.nan)
        idio_vol_12m = np.full(n, np.nan)

        for i in range(n):
            # Family 1: Momentum & Family 2: Reversal
            if i >= 1:
                mom_1m[i] = (adj_close[i] / adj_close[i - 1]) - 1.0
                str_reversal[i] = mom_1m[i]
            if i >= 3:
                mom_3m[i] = (adj_close[i] / adj_close[i - 3]) - 1.0
            if i >= 6:
                mom_6m[i] = (adj_close[i] / adj_close[i - 6]) - 1.0
            if i >= 9:
                mom_9m[i] = (adj_close[i] / adj_close[i - 9]) - 1.0
            if i >= 12:
                mom_12m[i] = (adj_close[i] / adj_close[i - 12]) - 1.0
                mom_12m_ex_1m[i] = (adj_close[i - 1] / adj_close[i - 12]) - 1.0
            if i >= 24:
                mom_24m[i] = (adj_close[i] / adj_close[i - 24]) - 1.0
            if i >= 36:
                ltr_reversal[i] = (adj_close[i - 12] / adj_close[i - 36]) - 1.0

            # Family 3: Volatility
            if i >= 3:
                window_r3 = ret_1m[i - 2 : i + 1]
                vol_3m[i] = np.std(window_r3, ddof=1)
            if i >= 6:
                window_r6 = ret_1m[i - 5 : i + 1]
                vol_6m[i] = np.std(window_r6, ddof=1)
            if i >= 12:
                window_r12 = ret_1m[i - 11 : i + 1]
                vol_12m[i] = np.std(window_r12, ddof=1)
                max_ret_12m[i] = np.max(window_r12)

                # Downside volatility: std dev of min(r, 0) over 12-month window
                downside_r12 = np.minimum(window_r12, 0.0)
                downside_vol_12m[i] = np.std(downside_r12, ddof=1)

                # Family 6: Distribution shape
                s = pd.Series(window_r12)
                skew_12m[i] = s.skew()
                kurt_12m[i] = s.kurt()

            # Family 4: Volume & Liquidity
            if i >= 2:
                avg_dv_3m[i] = np.mean(dollar_vol[i - 2 : i + 1])
            if i >= 11:
                avg_dv_12m[i] = np.mean(dollar_vol[i - 11 : i + 1])
                size_proxy[i] = (
                    np.log(avg_dv_12m[i])
                    if avg_dv_12m[i] is not None and avg_dv_12m[i] > 0
                    else np.nan
                )
            if not np.isnan(avg_dv_3m[i]) and not np.isnan(avg_dv_12m[i]) and avg_dv_12m[i] > 0:
                volume_trend[i] = avg_dv_3m[i] / avg_dv_12m[i]

            if i >= 12:
                r12 = ret_1m[i - 11 : i + 1]
                dv12 = dollar_vol[i - 11 : i + 1]
                mask = dv12 > 0
                if np.any(mask):
                    amihud_illiquidity[i] = np.mean(np.abs(r12[mask]) / dv12[mask])

            # Family 5: Price-level / technical (using raw close P)
            if i >= 11:
                p_win12 = close[i - 11 : i + 1]
                p_max = np.max(p_win12)
                p_min = np.min(p_win12)
                p_52w_high[i] = close[i] / p_max if p_max > 0 else np.nan
                p_52w_low[i] = close[i] / p_min if p_min > 0 else np.nan

                ma_12m = np.mean(p_win12)
                dist_ma_12m[i] = (close[i] - ma_12m) / ma_12m if ma_12m > 0 else np.nan

            if i >= 5:
                p_win6 = close[i - 5 : i + 1]
                ma_6m = np.mean(p_win6)
                dist_ma_6m[i] = (close[i] - ma_6m) / ma_6m if ma_6m > 0 else np.nan

            # Family 7: Market Sensitivity
            if i >= 12:
                r_stock = ret_1m[i - 11 : i + 1]
                r_mkt = spy_ret[i - 11 : i + 1]

                valid_mask = ~np.isnan(r_stock) & ~np.isnan(r_mkt)
                if np.sum(valid_mask) == 12:
                    y = r_stock[valid_mask]
                    x = r_mkt[valid_mask]

                    var_mkt = np.var(x, ddof=1)
                    if var_mkt > 1e-12:
                        cov_sm = np.cov(y, x, ddof=1)[0, 1]
                        b = cov_sm / var_mkt
                        beta_12m[i] = b

                        a = np.mean(y) - b * np.mean(x)
                        residuals = y - (a + b * x)
                        idio_vol_12m[i] = np.std(residuals, ddof=1)

        # Assign computed features
        df["mom_1m"] = mom_1m
        df["mom_3m"] = mom_3m
        df["mom_6m"] = mom_6m
        df["mom_9m"] = mom_9m
        df["mom_12m"] = mom_12m
        df["mom_12m_ex_1m"] = mom_12m_ex_1m
        df["mom_24m"] = mom_24m

        df["str_reversal"] = str_reversal
        df["ltr_reversal"] = ltr_reversal

        df["vol_3m"] = vol_3m
        df["vol_6m"] = vol_6m
        df["vol_12m"] = vol_12m
        df["downside_vol_12m"] = downside_vol_12m
        df["max_ret_12m"] = max_ret_12m

        df["avg_dollar_volume_3m"] = avg_dv_3m
        df["avg_dollar_volume_12m"] = avg_dv_12m
        df["volume_trend"] = volume_trend
        df["amihud_illiquidity"] = amihud_illiquidity
        df["size_proxy"] = size_proxy

        df["price_to_52w_high"] = p_52w_high
        df["price_to_52w_low"] = p_52w_low
        df["dist_from_ma_6m"] = dist_ma_6m
        df["dist_from_ma_12m"] = dist_ma_12m

        df["skew_12m"] = skew_12m
        df["kurt_12m"] = kurt_12m

        df["beta_12m"] = beta_12m
        df["idio_vol_12m"] = idio_vol_12m

        out_cols = ["ticker", "date"] + FEATURE_COLUMNS + ["next_month_return"]
        return df[out_cols]

    def build_panel(self) -> pd.DataFrame:
        """Processes all tickers and constructs the complete characteristics panel."""
        df_prices, df_spy = self.load_data()
        df_spy_returns = self._compute_spy_returns(df_spy)

        tickers = df_prices["ticker"].unique()
        logger.info(f"Building characteristics panel for {len(tickers)} tickers...")

        ticker_panels: List[pd.DataFrame] = []

        for idx, ticker in enumerate(tickers, start=1):
            df_t = df_prices[df_prices["ticker"] == ticker]
            df_feat = self._compute_ticker_characteristics(df_t, df_spy_returns)
            ticker_panels.append(df_feat)

        df_panel = pd.concat(ticker_panels, ignore_index=True)
        df_panel.sort_values(["ticker", "date"], inplace=True)
        df_panel.reset_index(drop=True, inplace=True)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df_panel.to_parquet(self.output_path, index=False)
        logger.info(
            f"Saved characteristics panel ({len(df_panel)} rows, {df_panel.shape[1]} columns) to {self.output_path}"
        )

        return df_panel


def run_feature_engineering() -> None:
    builder = CharacteristicBuilder()
    builder.build_panel()


if __name__ == "__main__":
    run_feature_engineering()
