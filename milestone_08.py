# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import time
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


data = get_price_data("AAPL", "2021-01-01", "2026-06-01")

window = 20
cost_per_trade = 0.001
starting_cash = 10000

data["sma"] = data["Close"].rolling(window).mean()
data["signal"] = (data["Close"] > data["sma"]).shift(1)


# ======================================================================
# SECTION 3: THE OLD WAY — AN EXPLICIT LOOP (FIXED EXECUTION TIMING)
# ======================================================================
def backtest_with_loop(data):
    cash = starting_cash
    shares_held = 0
    portfolio_history = []

    for i in range(len(data)):
        price_today = data["Close"].iloc[i]
        price_yesterday = data["Close"].iloc[i - 1] if i > 0 else price_today
        signal_today = data["signal"].iloc[i]

        if signal_today == True and shares_held == 0:
            shares_held = cash // price_yesterday
            trade_value = shares_held * price_yesterday
            friction = trade_value * cost_per_trade
            cash -= (trade_value + friction)
        elif signal_today == False and shares_held > 0:
            trade_value = shares_held * price_yesterday
            friction = trade_value * cost_per_trade
            cash += (trade_value - friction)
            shares_held = 0

        portfolio_history.append(cash + shares_held * price_today)

    return portfolio_history


# ======================================================================
# SECTION 4: THE NEW WAY — VECTORIZED, NO EXPLICIT LOOP
# ======================================================================
def backtest_vectorized(data):
    daily_return = data["Close"].pct_change()
    strategy_return = data["signal"] * daily_return

    trade_occurred = data["signal"].diff().fillna(0) != 0
    strategy_return[trade_occurred] -= cost_per_trade

    portfolio_value = starting_cash * (1 + strategy_return.fillna(0)).cumprod()
    return portfolio_value


# ======================================================================
# SECTION 5: COMPARE CORRECTNESS AND SPEED ON REAL DATA
# ======================================================================
start_time = time.perf_counter()
loop_result = backtest_with_loop(data)
loop_duration = time.perf_counter() - start_time

start_time = time.perf_counter()
vectorized_result = backtest_vectorized(data)
vectorized_duration = time.perf_counter() - start_time

print(f"Loop-based:   {loop_duration:.4f}s | Final value: ${loop_result[-1]:,.2f}")
print(f"Vectorized:   {vectorized_duration:.4f}s | Final value: ${vectorized_result.iloc[-1]:,.2f}")


# ======================================================================
# SECTION 6: STRESS TEST — WHERE VECTORIZATION ACTUALLY MATTERS
# ======================================================================
big_data = pd.concat([data] * 100, ignore_index=True)
print(f"\nStress test with {len(big_data):,} rows (artificially repeated):")

start_time = time.perf_counter()
backtest_with_loop(big_data)
big_loop_duration = time.perf_counter() - start_time

start_time = time.perf_counter()
backtest_vectorized(big_data)
big_vectorized_duration = time.perf_counter() - start_time

print(f"Loop-based:   {big_loop_duration:.4f}s")
print(f"Vectorized:   {big_vectorized_duration:.4f}s")
print(f"Speedup: {big_loop_duration / big_vectorized_duration:.1f}x faster")
