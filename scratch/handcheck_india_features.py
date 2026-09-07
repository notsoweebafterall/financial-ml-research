import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/india_characteristics_panel.parquet")
raw = pd.read_parquet("data/raw/india_prices.parquet")

print(f"India panel shape: {df.shape}, tickers: {df['ticker'].nunique()}")

# Pick 2 sample (ticker, date) pairs:
# Pair 1: ('RELIANCE.NS', '2023-01-01')
# Pair 2: ('TCS.NS', '2024-06-01')

for ticker, date in [('RELIANCE.NS', '2023-01-01'), ('TCS.NS', '2024-06-01')]:
    feat_row = df[(df['ticker'] == ticker) & (df['date'] == date)].iloc[0]
    raw_sub = raw[raw['ticker'] == ticker].sort_values('date').reset_index(drop=True)
    idx = raw_sub[raw_sub['date'] == date].index[0]
    
    print(f"\n=================== HAND-CHECK FOR ({ticker}, {date}) ===================")
    
    # 1. mom_1m
    p_t = raw_sub.loc[idx, 'adj_close']
    p_t1 = raw_sub.loc[idx-1, 'adj_close']
    calc_mom_1m = (p_t / p_t1) - 1.0
    panel_mom_1m = feat_row['mom_1m']
    print(f"mom_1m: Calculated={calc_mom_1m:.6f}, Panel={panel_mom_1m:.6f}, Match={np.isclose(calc_mom_1m, panel_mom_1m)}")
    
    # 2. mom_12m
    p_t12 = raw_sub.loc[idx-12, 'adj_close']
    calc_mom_12m = (p_t / p_t12) - 1.0
    panel_mom_12m = feat_row['mom_12m']
    print(f"mom_12m: Calculated={calc_mom_12m:.6f}, Panel={panel_mom_12m:.6f}, Match={np.isclose(calc_mom_12m, panel_mom_12m)}")
    
    # 3. size_proxy (log of 12m avg dollar volume)
    close_12 = raw_sub.loc[idx-11:idx, 'close'].values
    vol_12 = raw_sub.loc[idx-11:idx, 'volume'].values
    dv_12 = close_12 * vol_12
    calc_size_proxy = np.log(np.mean(dv_12))
    panel_size_proxy = feat_row['size_proxy']
    print(f"size_proxy: Calculated={calc_size_proxy:.6f}, Panel={panel_size_proxy:.6f}, Match={np.isclose(calc_size_proxy, panel_size_proxy)}")

    # 4. next_month_return (target)
    if idx + 1 < len(raw_sub):
        p_next = raw_sub.loc[idx+1, 'adj_close']
        calc_target = (p_next / p_t) - 1.0
        panel_target = feat_row['next_month_return']
        print(f"next_month_return: Calculated={calc_target:.6f}, Panel={panel_target:.6f}, Match={np.isclose(calc_target, panel_target)}")
