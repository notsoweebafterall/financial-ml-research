"""
Runs the Phase 5 backtest for all 3 models (Ridge, RandomForest, XGBoost) using the
real out-of-sample predictions from Phase 4, and saves results.

Usage (from project root, with .venv active):
    python -m src.portfolio.run_backtest
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.portfolio.backtest import run_backtest

PREDICTIONS_PATH = Path("data/processed/oos_predictions.parquet")
RESULTS_DIR = Path("results")
DECILE_FRAC = 0.1
TRANSACTION_COST_BPS = 10.0


def main(market: str = "us"):
    market = market.lower()
    print(f"=== RUNNING PORTFOLIO BACKTEST ({market.upper()}) ===", flush=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    if market == "us":
        predictions_path = Path("data/processed/oos_predictions.parquet")
        results_csv = RESULTS_DIR / "phase5_backtest_results.csv"
        equity_png = RESULTS_DIR / "phase5_equity_curves.png"
    elif market == "india":
        predictions_path = Path("data/processed/india_oos_predictions.parquet")
        results_csv = RESULTS_DIR / "india_phase5_backtest_results.csv"
        equity_png = RESULTS_DIR / "india_phase5_equity_curves.png"
    else:
        raise ValueError(f"Unknown market '{market}'. Expected 'us' or 'india'.")

    predictions = pd.read_parquet(predictions_path)
    model_names = predictions["model_name"].unique()
    print(f"Found models in {predictions_path}: {list(model_names)}")

    all_results = []
    for model_name in model_names:
        print(f"\nRunning backtest for {model_name}...")
        result = run_backtest(
            predictions,
            model_name=model_name,
            decile_frac=DECILE_FRAC,
            transaction_cost_bps=TRANSACTION_COST_BPS,
        )
        all_results.append(result)

        final_equity = result["cumulative_equity"].iloc[-1]
        final_equity_gross = result["cumulative_equity_gross"].iloc[-1]
        avg_monthly_turnover = (result["turnover_long"] + result["turnover_short"]).mean() / 2
        total_cost_drag = (result["cumulative_equity_gross"].iloc[-1]
                            - result["cumulative_equity"].iloc[-1])

        print(f"  Months simulated: {len(result)}")
        print(f"  Final equity (net of costs): {final_equity:.4f} (started at 1.0)")
        print(f"  Final equity (gross, no costs): {final_equity_gross:.4f}")
        print(f"  Average monthly turnover per leg: {avg_monthly_turnover:.2%}")
        print(f"  Cost drag on final equity: {total_cost_drag:.4f}")

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(results_csv, index=False)
    print(f"\nSaved combined results to {results_csv}")

    # Equity curve plot, all models overlaid
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name in model_names:
        model_result = combined[combined["model_name"] == model_name]
        ax.plot(model_result["date"], model_result["cumulative_equity"], label=model_name)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"Long-Short Decile Portfolio ({market.upper()}) — Cumulative Equity (Net of Costs)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (starting at 1.0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(equity_png, dpi=150)
    plt.close(fig)
    print(f"Saved equity curve plot to {equity_png}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run portfolio backtest for US or India market.")
    parser.add_argument("--market", type=str, default="us", choices=["us", "india"], help="Target market (default: us)")
    args = parser.parse_args()

    main(market=args.market)

