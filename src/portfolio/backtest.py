"""
Portfolio construction + backtest logic.

Takes out-of-sample predictions (from Phase 4) and turns them into a long-short
decile portfolio, simulated month by month, with realistic transaction costs
applied based on turnover.

Strategy: each test month, rank stocks by predicted_return. Go long the top decile
(highest predicted returns), short the bottom decile (lowest predicted returns).
Equal-weight within each leg. Dollar-neutral (long leg weights sum to +1, short leg
weights sum to -1, net market exposure = 0).

Costs: a flat transaction cost (in bps) is charged on TURNOVER — i.e. only on the
fraction of each leg's holdings that actually changed from the previous month. If
a stock stays in the long decile two months running, no cost is charged for it that
second month; if it drops out and a new stock enters, that's turnover.
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class MonthlyPortfolioResult:
    date: pd.Timestamp
    long_tickers: frozenset
    short_tickers: frozenset
    gross_return: float          # long leg avg return - short leg avg return, before costs
    turnover_long: float          # fraction of long leg that changed vs previous month
    turnover_short: float
    transaction_cost: float       # total cost deducted this month
    net_return: float             # gross_return - transaction_cost


def select_decile_legs(month_df: pd.DataFrame, decile_frac: float = 0.1) -> tuple[set, set]:
    """
    Given one month's predictions (columns: ticker, predicted_return), return
    (long_tickers, short_tickers) — the top and bottom decile by predicted return.
    """
    n = len(month_df)
    n_leg = max(1, int(round(n * decile_frac)))

    sorted_df = month_df.sort_values("predicted_return", ascending=False)
    long_tickers = set(sorted_df.head(n_leg)["ticker"])
    short_tickers = set(sorted_df.tail(n_leg)["ticker"])

    # Safety: long and short legs must never overlap (would only happen if n_leg
    # is so large relative to n that head/tail overlap — guard against that)
    assert long_tickers.isdisjoint(short_tickers), (
        "Long and short legs overlap — decile_frac is too large for this universe size"
    )
    return long_tickers, short_tickers


def compute_turnover(prev_leg: set, curr_leg: set) -> float:
    """
    Fraction of the CURRENT leg's positions that are new (i.e. were not held last
    month). Range [0, 1]. 0.0 = leg is identical to last month (no trading needed).
    1.0 = leg is entirely different from last month (fully rebuilt).
    First month (prev_leg empty/None) is treated as 100% turnover — you're building
    the position from scratch, which is a real cost.
    """
    if not curr_leg:
        return 0.0
    if not prev_leg:
        return 1.0
    new_positions = curr_leg - prev_leg
    return len(new_positions) / len(curr_leg)


def run_backtest(
    predictions_df: pd.DataFrame,
    model_name: str,
    decile_frac: float = 0.1,
    transaction_cost_bps: float = 10.0,
) -> pd.DataFrame:
    """
    predictions_df: must have columns [ticker, date, actual_return, predicted_return,
    model_name]. This function filters to the given model_name internally.

    Returns a DataFrame, one row per test month, with all MonthlyPortfolioResult
    fields plus a cumulative equity curve column (starting at 1.0).
    """
    df = predictions_df[predictions_df["model_name"] == model_name].copy()
    df["date"] = pd.to_datetime(df["date"])
    months = sorted(df["date"].unique())

    cost_rate = transaction_cost_bps / 10_000.0  # bps -> decimal

    results = []
    prev_long: Optional[set] = None
    prev_short: Optional[set] = None

    for month in months:
        month_df = df[df["date"] == month]
        if month_df.empty:
            continue

        # Guard against the natural data-boundary case: the very last month in the
        # raw price history has no "next month" price, so next_month_return (and
        # therefore actual_return here) is NaN for EVERY ticker that month. If we
        # let this through, .mean() would return NaN for both legs, and NaN
        # silently poisons every subsequent month once fed into .cumprod(). Skip
        # this month entirely rather than recording a fake/undefined return.
        if month_df["actual_return"].isna().all():
            print(f"  Skipping {month.date()}: no valid actual_return for any ticker "
                  f"(likely the final month in the dataset, with no future price to "
                  f"compute a return from)")
            continue

        long_tickers, short_tickers = select_decile_legs(month_df, decile_frac)

        long_actual = month_df[month_df["ticker"].isin(long_tickers)]["actual_return"]
        short_actual = month_df[month_df["ticker"].isin(short_tickers)]["actual_return"]

        # Equal-weight each leg; gross return is long avg minus short avg
        gross_return = long_actual.mean() - short_actual.mean()

        turnover_long = compute_turnover(prev_long, long_tickers)
        turnover_short = compute_turnover(prev_short, short_tickers)

        # Cost is charged on turnover in BOTH legs (both incur trading costs when
        # positions change), scaled by cost_rate
        transaction_cost = (turnover_long + turnover_short) * cost_rate

        net_return = gross_return - transaction_cost

        results.append(MonthlyPortfolioResult(
            date=month,
            long_tickers=frozenset(long_tickers),
            short_tickers=frozenset(short_tickers),
            gross_return=gross_return,
            turnover_long=turnover_long,
            turnover_short=turnover_short,
            transaction_cost=transaction_cost,
            net_return=net_return,
        ))

        prev_long, prev_short = long_tickers, short_tickers

    result_df = pd.DataFrame([r.__dict__ for r in results])
    result_df["model_name"] = model_name
    result_df["cumulative_equity"] = (1 + result_df["net_return"]).cumprod()
    result_df["cumulative_equity_gross"] = (1 + result_df["gross_return"]).cumprod()
    return result_df
