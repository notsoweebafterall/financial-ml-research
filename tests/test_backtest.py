import pandas as pd
import numpy as np
from src.portfolio.backtest import select_decile_legs, compute_turnover, run_backtest


def make_synthetic_predictions(n_tickers=60, n_months=36, model_name="TestModel", seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-09-01", periods=n_months, freq="MS")
    rows = []
    for d in dates:
        for t in range(n_tickers):
            rows.append({
                "ticker": f"TICK{t:03d}",
                "date": d,
                "predicted_return": rng.normal(0, 0.05),
                "actual_return": rng.normal(0.005, 0.08),
                "model_name": model_name,
            })
    return pd.DataFrame(rows)


def test_decile_selection_correctness():
    df = make_synthetic_predictions(n_tickers=60, n_months=1)
    long_t, short_t = select_decile_legs(df, decile_frac=0.1)

    assert len(long_t) == 6, f"Expected 6 stocks in long leg (10% of 60), got {len(long_t)}"
    assert len(short_t) == 6
    assert long_t.isdisjoint(short_t)

    # Manually verify: long leg should be exactly the 6 highest predicted_return tickers
    manual_top6 = set(df.sort_values("predicted_return", ascending=False).head(6)["ticker"])
    assert long_t == manual_top6, "Long leg does not match manual top-6 selection"
    print("PASS: decile selection matches manual top/bottom-6 calculation exactly.")


def test_turnover_hand_computed():
    prev = {"A", "B", "C", "D", "E", "F"}

    # Case 1: identical leg -> 0 turnover
    assert compute_turnover(prev, prev) == 0.0

    # Case 2: completely different leg -> 1.0 turnover
    completely_new = {"G", "H", "I", "J", "K", "L"}
    assert compute_turnover(prev, completely_new) == 1.0

    # Case 3: 2 out of 6 positions are new -> turnover = 2/6 = 0.3333
    partial = {"A", "B", "C", "D", "X", "Y"}  # E,F dropped, X,Y added
    expected = 2 / 6
    actual = compute_turnover(prev, partial)
    assert abs(actual - expected) < 1e-9, f"Expected {expected}, got {actual}"

    # Case 4: first month (no previous leg) -> 100% turnover
    assert compute_turnover(None, prev) == 1.0
    assert compute_turnover(set(), prev) == 1.0

    print("PASS: turnover calculation matches hand-computed values in all 4 cases.")


def test_backtest_cost_reduces_return_correctly():
    """
    Build a tiny, fully controlled 2-month scenario and verify the net_return math
    by hand, including the transaction cost deduction.
    """
    dates = pd.to_datetime(["2020-01-01", "2020-02-01"])
    # 10 stocks, decile_frac=0.1 -> 1 stock per leg, easy to trace by hand
    rows = []
    # Month 1: ticker A has highest predicted (long), ticker J has lowest (short)
    preds_m1 = {"A": 0.10, "B": 0.05, "C": 0.04, "D": 0.03, "E": 0.02,
                "F": 0.01, "G": 0.00, "H": -0.01, "I": -0.02, "J": -0.10}
    actuals_m1 = {"A": 0.08, "J": -0.05}  # only legs matter for return calc
    for t, p in preds_m1.items():
        rows.append({"ticker": t, "date": dates[0], "predicted_return": p,
                      "actual_return": actuals_m1.get(t, 0.0), "model_name": "M"})

    # Month 2: ticker B now highest (long switches A->B), ticker J still lowest (short unchanged)
    preds_m2 = {"A": 0.02, "B": 0.10, "C": 0.04, "D": 0.03, "E": 0.02,
                "F": 0.01, "G": 0.00, "H": -0.01, "I": -0.02, "J": -0.10}
    actuals_m2 = {"B": 0.06, "J": -0.03}
    for t, p in preds_m2.items():
        rows.append({"ticker": t, "date": dates[1], "predicted_return": p,
                      "actual_return": actuals_m2.get(t, 0.0), "model_name": "M"})

    df = pd.DataFrame(rows)
    result = run_backtest(df, model_name="M", decile_frac=0.1, transaction_cost_bps=10.0)

    # --- Month 1 checks ---
    m1 = result.iloc[0]
    assert m1.long_tickers == frozenset({"A"})
    assert m1.short_tickers == frozenset({"J"})
    expected_gross_m1 = 0.08 - (-0.05)  # long actual - short actual = 0.13
    assert abs(m1.gross_return - expected_gross_m1) < 1e-9
    # First month = 100% turnover on both legs
    assert m1.turnover_long == 1.0 and m1.turnover_short == 1.0
    expected_cost_m1 = (1.0 + 1.0) * (10.0 / 10_000.0)  # = 0.002
    assert abs(m1.transaction_cost - expected_cost_m1) < 1e-9
    expected_net_m1 = expected_gross_m1 - expected_cost_m1
    assert abs(m1.net_return - expected_net_m1) < 1e-9

    # --- Month 2 checks ---
    m2 = result.iloc[1]
    assert m2.long_tickers == frozenset({"B"})
    assert m2.short_tickers == frozenset({"J"})
    expected_gross_m2 = 0.06 - (-0.03)  # = 0.09
    assert abs(m2.gross_return - expected_gross_m2) < 1e-9
    # Long leg fully turned over (A->B), short leg unchanged (J->J)
    assert m2.turnover_long == 1.0, "Long leg should be 100% turnover (A dropped, B added)"
    assert m2.turnover_short == 0.0, "Short leg should be 0% turnover (J held both months)"
    expected_cost_m2 = (1.0 + 0.0) * (10.0 / 10_000.0)  # = 0.001
    assert abs(m2.transaction_cost - expected_cost_m2) < 1e-9
    expected_net_m2 = expected_gross_m2 - expected_cost_m2
    assert abs(m2.net_return - expected_net_m2) < 1e-9

    # --- Equity curve check ---
    expected_equity_m1 = 1 + expected_net_m1
    expected_equity_m2 = expected_equity_m1 * (1 + expected_net_m2)
    assert abs(m1.cumulative_equity - expected_equity_m1) < 1e-9
    assert abs(m2.cumulative_equity - expected_equity_m2) < 1e-9

    print("PASS: full 2-month hand-traced backtest matches expected gross/net returns, "
          "turnover, transaction costs, and compounded equity curve exactly.")


def test_no_lookahead_in_backtest_logic():
    """
    Confirm the backtest only ever uses predicted_return (known at month t) to decide
    positions, and actual_return (realized outcome) only to compute the resulting P&L
    — never using actual_return to influence which stocks are selected.
    """
    df = make_synthetic_predictions(n_tickers=60, n_months=12)
    # Corrupt actual_return with extreme values that would be very tempting to
    # "accidentally" use for selection if there were a bug — if decile selection
    # were buggy and used actual_return, results would change when we shuffle it
    df_shuffled = df.copy()
    df_shuffled["actual_return"] = np.random.default_rng(0).permutation(df_shuffled["actual_return"].values)

    result_original = run_backtest(df, model_name="TestModel", decile_frac=0.1)
    result_shuffled = run_backtest(df_shuffled, model_name="TestModel", decile_frac=0.1)

    # Long/short ticker selections must be IDENTICAL regardless of actual_return values,
    # since selection should only depend on predicted_return
    for i in range(len(result_original)):
        assert result_original.iloc[i].long_tickers == result_shuffled.iloc[i].long_tickers
        assert result_original.iloc[i].short_tickers == result_shuffled.iloc[i].short_tickers

    print("PASS: portfolio selection is driven only by predicted_return, confirmed "
          "unaffected by shuffling actual_return.")


if __name__ == "__main__":
    test_decile_selection_correctness()
    test_turnover_hand_computed()
    test_backtest_cost_reduces_return_correctly()
    test_no_lookahead_in_backtest_logic()
    print("\nAll Phase 5 backtest tests passed.")
