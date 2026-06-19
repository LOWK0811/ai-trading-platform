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
# SECTION 3: BACKTEST WITH AN OPTIONAL CIRCUIT BREAKER
# ======================================================================
cost_per_trade = 0.001
starting_cash = 10000

def backtest_with_circuit_breaker(data, window, use_circuit_breaker=True, threshold=-0.02):
    data = data.copy()
    data["sma"] = data["Close"].rolling(window).mean()
    data["signal"] = (data["Close"] > data["sma"]).shift(1)

    cash = starting_cash
    shares_held = 0
    portfolio_history = []
    halt_next_day = False
    num_halts = 0
    previous_value = starting_cash

    for i in range(len(data)):
        price_today = data["Close"].iloc[i]
        price_yesterday = data["Close"].iloc[i - 1] if i > 0 else price_today
        signal_today = data["signal"].iloc[i]

        if use_circuit_breaker and halt_next_day:
            num_halts += 1
            if shares_held > 0:
                cash += shares_held * price_today
                shares_held = 0
        else:
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

        current_value = cash + shares_held * price_today
        portfolio_history.append(current_value)

        daily_pnl_pct = (current_value / previous_value) - 1
        halt_next_day = daily_pnl_pct < threshold
        previous_value = current_value

    return portfolio_history, num_halts


# ======================================================================
# SECTION 4: COMPARE WITH AND WITHOUT THE CIRCUIT BREAKER
# ======================================================================
data = get_price_data("AAPL", "2021-01-01", "2026-06-01")
window = 20

with_cb, halts = backtest_with_circuit_breaker(data, window, use_circuit_breaker=True, threshold=-0.02)
without_cb, _ = backtest_with_circuit_breaker(data, window, use_circuit_breaker=False)

print(f"Without circuit breaker: ${without_cb[-1]:,.2f}")
print(f"With circuit breaker:    ${with_cb[-1]:,.2f}")
print(f"Number of days halted:   {halts}")