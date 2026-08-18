"""
Performance evaluation metrics module for Phase 6.
Calculates performance summary, Information Coefficient (IC/ICIR), Naive & HAC significance,
regime breakdown, and SPY market correlation.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm


class PerformanceEvaluator:
    def __init__(
        self,
        bkt_df: pd.DataFrame,
        oos_df: pd.DataFrame,
        spy_df: pd.DataFrame,
    ):
        self.bkt_df = bkt_df.copy()
        # Clean OOS predictions (exclude unlabelled month 2026-08-01)
        self.oos_df = oos_df[oos_df["date"] != "2026-08-01"].dropna(subset=["actual_return"]).copy()
        
        # Prepare SPY benchmark dataset
        spy = spy_df.sort_values("date").reset_index(drop=True).copy()
        spy["spy_return"] = spy["adj_close"].pct_change()
        spy["spy_trailing_12m"] = spy["adj_close"].pct_change(12)
        spy["spy_trailing_3m_vol"] = spy["spy_return"].rolling(3).std()
        
        test_dates = set(self.bkt_df["date"].unique())
        self.spy_test = spy[spy["date"].isin(test_dates)].copy()
        
        med_vol = self.spy_test["spy_trailing_3m_vol"].median()
        self.spy_test["bull_bear"] = np.where(self.spy_test["spy_trailing_12m"] > 0, "bull", "bear")
        self.spy_test["vol_regime"] = np.where(self.spy_test["spy_trailing_3m_vol"] > med_vol, "high_vol", "low_vol")

    def compute_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        models = sorted(self.bkt_df["model_name"].unique())
        
        perf_rows = []
        regime_rows = []
        market_corr_rows = []

        merged_bkt = pd.merge(
            self.bkt_df,
            self.spy_test[["date", "spy_return", "bull_bear", "vol_regime"]],
            on="date",
        )

        for model in models:
            sub_bkt = merged_bkt[merged_bkt["model_name"] == model].sort_values("date").reset_index(drop=True)
            net_ret = sub_bkt["net_return"].values
            cum_eq = sub_bkt["cumulative_equity"].values
            n_months = len(net_ret)

            # 1. Standard Performance Metrics
            std_ret = np.std(net_ret, ddof=1) if n_months > 1 else 0.0
            sharpe = (np.mean(net_ret) / std_ret * np.sqrt(12)) if std_ret > 0 else 0.0
            
            peaks = np.maximum.accumulate(cum_eq)
            drawdowns = (cum_eq - peaks) / peaks
            max_drawdown = float(np.min(drawdowns))
            
            hit_rate = float(np.mean(net_ret > 0))
            final_eq = float(cum_eq[-1])
            ann_return = (final_eq ** (12.0 / n_months)) - 1.0

            # 2. Information Coefficient (Spearman Rank Correlation - Outlier Robust)
            sub_oos = self.oos_df[self.oos_df["model_name"] == model]
            monthly_ics = []
            for d, group in sub_oos.groupby("date"):
                if len(group) > 1:
                    r, _ = stats.spearmanr(group["predicted_return"], group["actual_return"])
                    if not np.isnan(r):
                        monthly_ics.append(r)
            
            ic = float(np.mean(monthly_ics)) if monthly_ics else 0.0
            ic_std = float(np.std(monthly_ics, ddof=1)) if len(monthly_ics) > 1 else 0.0
            icir = (ic / ic_std) if ic_std > 0 else 0.0

            # 3. Statistical Significance (Naive vs Newey-West HAC)
            t_naive, p_naive = stats.ttest_1samp(net_ret, 0.0)
            
            X = np.ones((n_months, 1))
            ols = sm.OLS(net_ret, X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
            t_hac = float(ols.tvalues[0])
            p_hac = float(ols.pvalues[0])

            perf_rows.append(
                {
                    "model_name": model,
                    "sharpe_ratio": round(sharpe, 4),
                    "max_drawdown": round(max_drawdown, 4),
                    "hit_rate": round(hit_rate, 4),
                    "annualized_return": round(ann_return, 4),
                    "ic": round(ic, 4),
                    "icir": round(icir, 4),
                    "naive_t_stat": round(float(t_naive), 4),
                    "naive_p_value": round(float(p_naive), 4),
                    "hac_t_stat": round(t_hac, 4),
                    "hac_p_value": round(p_hac, 4),
                }
            )

            # 4. Regime Analysis (Bull, Bear, High Vol, Low Vol)
            regime_buckets = {
                "bull": sub_bkt[sub_bkt["bull_bear"] == "bull"],
                "bear": sub_bkt[sub_bkt["bull_bear"] == "bear"],
                "high_vol": sub_bkt[sub_bkt["vol_regime"] == "high_vol"],
                "low_vol": sub_bkt[sub_bkt["vol_regime"] == "low_vol"],
            }

            for b_name, b_df in regime_buckets.items():
                b_count = len(b_df)
                is_low_conf = b_count < 8

                if b_count == 0:
                    r_sharpe = np.nan
                    r_hit = np.nan
                    r_mean = np.nan
                else:
                    b_net = b_df["net_return"].values
                    b_std = np.std(b_net, ddof=1) if b_count > 1 else 0.0
                    r_sharpe = (np.mean(b_net) / b_std * np.sqrt(12)) if b_std > 0 else np.nan
                    r_hit = np.mean(b_net > 0)
                    r_mean = np.mean(b_net)

                regime_rows.append(
                    {
                        "model_name": model,
                        "regime_bucket": b_name,
                        "sharpe_ratio": round(r_sharpe, 4) if not np.isnan(r_sharpe) else np.nan,
                        "hit_rate": round(r_hit, 4) if not np.isnan(r_hit) else np.nan,
                        "mean_net_return": round(r_mean, 4) if not np.isnan(r_mean) else np.nan,
                        "n_months": b_count,
                        "low_confidence_flag": is_low_conf,
                    }
                )

            # 5. SPY Market Correlation Analysis (Pearson Linear Correlation for Market Co-movement)
            spy_ret = sub_bkt["spy_return"].values
            r_spy, p_spy = stats.pearsonr(net_ret, spy_ret)
            market_corr_rows.append(
                {
                    "model_name": model,
                    "spy_correlation": round(float(r_spy), 4),
                    "p_value": round(float(p_spy), 4),
                }
            )

        perf_summary_df = pd.DataFrame(perf_rows)
        regime_breakdown_df = pd.DataFrame(regime_rows)
        market_corr_df = pd.DataFrame(market_corr_rows)

        return perf_summary_df, regime_breakdown_df, market_corr_df
