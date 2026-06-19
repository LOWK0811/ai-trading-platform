# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import yfinance as yf
import pandas as pd


# ======================================================================
# SECTION 2: REUSE OUR DATA LOADER
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
# SECTION 3: SPLIT — BEFORE TOUCHING ANYTHING
# ======================================================================
full_data = get_price_data("AAPL", "2021-01-01", "2026-06-01")

split_date = "2024-06-01"
in_sample = full_data[full_data.index < split_date].copy()
out_of_sample = full_data[full_data.index >= split_date].copy()

print(f"In-sample:     {in_sample.index.min().date()} to {in_sample.index.max().date()} ({len(in_sample)} days)")
print(f"Out-of-sample: {out_of_sample.index.min().date()} to {out_of_sample.index.max().date()} ({len(out_of_sample)} days)")


# ======================================================================
# SECTION 4: A REUSABLE BACKTEST FUNCTION (FROM MILESTONE 8'S VECTORIZED VERSION)
# ======================================================================
cost_per_trade = 0.001
starting_cash = 10000

def run_backtest(data, window):
    data = data.copy()  # fresh copy each call, so old runs never leak into new ones
    data["sma"] = data["Close"].rolling(window).mean()
    data["signal"] = (data["Close"] > data["sma"]).shift(1)

    daily_return = data["Close"].pct_change()
    strategy_return = data["signal"] * daily_return

    trade_occurred = data["signal"].diff().fillna(0) != 0
    strategy_return[trade_occurred] -= cost_per_trade

    portfolio_value = starting_cash * (1 + strategy_return.fillna(0)).cumprod()
    return (portfolio_value.iloc[-1] / starting_cash) - 1


# ======================================================================
# SECTION 5: TUNE THE WINDOW — IN-SAMPLE ONLY
# ======================================================================
candidate_windows = [10, 20, 30, 50, 100]

print("\n--- In-sample results (the only data we're allowed to tune on) ---")
in_sample_results = {}
for window in candidate_windows:
    result = run_backtest(in_sample, window)
    in_sample_results[window] = result
    print(f"Window = {window:>3} days  ->  In-sample return: {result:.2%}")

best_window = max(in_sample_results, key=in_sample_results.get)
print(f"\nBest in-sample: {best_window}-day window ({in_sample_results[best_window]:.2%})")


# ======================================================================
# SECTION 6: THE MOMENT OF TRUTH
# ======================================================================
print("\n--- Out-of-sample results (the real test) ---")
for window in candidate_windows:
    result = run_backtest(out_of_sample, window)
    marker = "  <- the one we picked" if window == best_window else ""
    print(f"Window = {window:>3} days  ->  Out-of-sample return: {result:.2%}{marker}")