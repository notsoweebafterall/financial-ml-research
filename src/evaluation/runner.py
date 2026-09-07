"""
Shared runner script for Phase 6: Performance Evaluation, Statistical Significance,
Regime Breakdown, and Market Correlation Analysis.
"""

import os
import pandas as pd
import numpy as np

from src.evaluation.metrics import PerformanceEvaluator


def run_phase6(market: str = "us"):
    market = market.lower()
    print(f"=== STARTING PHASE 6 ({market.upper()}): PERFORMANCE EVALUATION & REGIME ANALYSIS ===", flush=True)

    if market == "us":
        bkt_path = "results/phase5_backtest_results.csv"
        oos_path = "data/processed/oos_predictions.parquet"
        spy_path = "data/raw/spy_benchmark.parquet"
        f1 = "results/phase6_performance_summary.csv"
        f2 = "results/phase6_regime_breakdown.csv"
        f3 = "results/phase6_market_correlation.csv"
    elif market == "india":
        bkt_path = "results/india_phase5_backtest_results.csv"
        oos_path = "data/processed/india_oos_predictions.parquet"
        spy_path = "data/raw/india_nifty_benchmark.parquet"
        f1 = "results/india_phase6_performance_summary.csv"
        f2 = "results/india_phase6_regime_breakdown.csv"
        f3 = "results/india_phase6_market_correlation.csv"
    else:
        raise ValueError(f"Unknown market '{market}'. Expected 'us' or 'india'.")

    for path in [bkt_path, oos_path, spy_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required input file missing: {path}")

    bkt_df = pd.read_csv(bkt_path)
    oos_df = pd.read_parquet(oos_path)
    spy_df = pd.read_parquet(spy_path)

    print(f"Loaded backtest results ({len(bkt_df)} rows), OOS predictions ({len(oos_df)} rows), Benchmark ({len(spy_df)} rows).", flush=True)

    evaluator = PerformanceEvaluator(bkt_df=bkt_df, oos_df=oos_df, spy_df=spy_df)
    perf_summary_df, regime_breakdown_df, market_corr_df = evaluator.compute_all()

    os.makedirs("results", exist_ok=True)

    # Save output CSVs
    perf_summary_df.to_csv(f1, index=False)
    regime_breakdown_df.to_csv(f2, index=False)
    market_corr_df.to_csv(f3, index=False)

    print(f"Saved performance summary to '{f1}'.", flush=True)
    print(f"Saved regime breakdown to '{f2}'.", flush=True)
    print(f"Saved market correlation to '{f3}'.", flush=True)

    # Display Verifications
    print_verifications(perf_summary_df, regime_breakdown_df, market_corr_df)


def print_verifications(
    perf_df: pd.DataFrame, regime_df: pd.DataFrame, corr_df: pd.DataFrame
):
    print("\n" + "=" * 70, flush=True)
    print("PHASE 6 VERIFICATION RESULTS & OUTPUT CSV CONTENTS", flush=True)
    print("=" * 70, flush=True)

    # 1. Performance Summary CSV
    print("\n--- 1. PERFORMANCE SUMMARY ---", flush=True)
    print(perf_df.to_string(index=False), flush=True)

    # 2. Regime Breakdown CSV
    print("\n--- 2. REGIME BREAKDOWN ---", flush=True)
    print(regime_df.to_string(index=False), flush=True)

    # 3. Market Correlation CSV
    print("\n--- 3. MARKET CORRELATION ---", flush=True)
    print(corr_df.to_string(index=False), flush=True)

    # 4. Explicit Callouts
    print("\n" + "=" * 70, flush=True)
    print("EXPLICIT CALLOUTS & DIAGNOSTIC FINDINGS", flush=True)
    print("=" * 70, flush=True)

    # Callout 1: Naive vs HAC Significance Discrepancy
    print("\n--- CALLOUT 1: NAIVE vs. NEWEY-WEST (HAC) SIGNIFICANCE TESTS ---", flush=True)
    for idx, row in perf_df.iterrows():
        model = row["model_name"]
        t_n, p_n = row["naive_t_stat"], row["naive_p_value"]
        t_h, p_h = row["hac_t_stat"], row["hac_p_value"]
        sig_n = p_n < 0.05
        sig_h = p_h < 0.05
        disagree = (sig_n != sig_h)
        status_str = "DISAGREEMENT DETECTED" if disagree else "AGREE (both non-significant at p < 0.05)"
        print(f"Model: {model:12s} | Naive t={t_n:6.4f} (p={p_n:.4f}) | HAC t={t_h:6.4f} (p={p_h:.4f}) -> [{status_str}]", flush=True)

    n_models = len(perf_df)
    print(
        f"\nStatistical Power Note: With N test months, statistical power is inherently limited. "
        f"Model significance evaluated under naive and HAC-adjusted tests.",
        flush=True,
    )

    # Callout 2: Benchmark Correlation & Methodological Note
    print("\n--- CALLOUT 2: MARKET CORRELATION & METHODOLOGICAL DISTINCTION ---", flush=True)
    print(
        "Methodological Note on Correlation Choice:\n"
        "- Spearman Rank Correlation is used for IC because cross-sectional return prediction evaluation requires rank robustness to stock return outliers.\n"
        "- Pearson Linear Correlation is intentionally used for Benchmark Correlation to measure true linear co-movement and market beta exposure.",
        flush=True,
    )
    for idx, row in corr_df.iterrows():
        model = row["model_name"]
        corr = row["spy_correlation"]
        p_val = row["p_value"]
        print(f"Model: {model:12s} | Benchmark Pearson Return Corr: {corr:+.4f} (p = {p_val:.4f})", flush=True)

    # Callout 3: Low-Confidence Regime Flagging (< 8 months) & 0-Month Handling
    print("\n--- CALLOUT 3: LOW-CONFIDENCE REGIME BUCKETS (< 8 MONTHS) ---", flush=True)
    low_conf_buckets = regime_df[regime_df["low_confidence_flag"] == True]
    if len(low_conf_buckets) > 0:
        for idx, row in low_conf_buckets.iterrows():
            m_name = row["model_name"]
            b_name = row["regime_bucket"]
            n_m = row["n_months"]
            if n_m == 0:
                print(f"Model: {m_name:12s} | Regime: {b_name:8s} | Month Count: 0 -> LOW CONFIDENCE / SKIPPED: No bear-market months present in this test period", flush=True)
            else:
                print(f"Model: {m_name:12s} | Regime: {b_name:8s} | Month Count: {n_m} -> LOW CONFIDENCE (< 8 months sample size)", flush=True)
    else:
        print("No low-confidence regimes found.", flush=True)

    print("\n" + "=" * 70, flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Phase 6 evaluation for US or India market.")
    parser.add_argument("--market", type=str, default="us", choices=["us", "india"], help="Target market (default: us)")
    args = parser.parse_args()

    run_phase6(market=args.market)

