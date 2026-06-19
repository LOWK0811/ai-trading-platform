# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import yfinance as yf
import pandas as pd
from scipy import stats


# ======================================================================
# SECTION 2: REUSE OUR DATA LOADER FROM MILESTONE 3
# ======================================================================
def get_price_data(ticker, start, end):
    """Loads OHLCV data, using a local cache if one already exists."""
    filename = f"data/{ticker}_{start}_{end}.parquet"

    if os.path.exists(filename):
        data = pd.read_parquet(filename)
    else:
        data = yf.Ticker(ticker).history(start=start, end=end)
        os.makedirs("data", exist_ok=True)
        data.to_parquet(filename)

    return data


# ======================================================================
# SECTION 3: REBUILD THE ANALYSIS FROM MILESTONE 4
# ======================================================================
data = get_price_data("AAPL", "2021-01-01", "2026-06-01")
data["return"] = data["Close"].pct_change()

mean_return = data["return"].mean()
std_return = data["return"].std()
data["z_score"] = (data["return"] - mean_return) / std_return
data["next_day_return"] = data["return"].shift(-1)

big_down_days = data[data["z_score"] < -2]


# ======================================================================
# SECTION 4: SPLIT INTO TWO GROUPS TO COMPARE
# ======================================================================
group_after_drop = data.loc[big_down_days.index, "next_day_return"].dropna()
group_typical = data.loc[~data.index.isin(big_down_days.index), "next_day_return"].dropna()

print(f"Group A (after a big down day): {len(group_after_drop)} days, mean = {group_after_drop.mean():.4%}")
print(f"Group B (all other days): {len(group_typical)} days, mean = {group_typical.mean():.4%}")


# ======================================================================
# SECTION 5: THE STATISTICAL TEST
# ======================================================================
t_stat, p_value = stats.ttest_ind(group_after_drop, group_typical, equal_var=False)

print(f"\nt-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("→ Statistically significant at the 5% level.")
else:
    print("→ Not statistically significant — plausibly just random chance.")