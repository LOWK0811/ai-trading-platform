# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


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
# SECTION 3: CALCULATE ATR (AVERAGE TRUE RANGE)
# ======================================================================
def calculate_atr(data, period=14):
    """
    ATR measures how much the price typically moves in a single day.
    True Range is the largest of three possible daily moves:
      - High to Low (normal intraday range)
      - Previous Close to High (gap up then volatile)
      - Previous Close to Low (gap down then volatile)
    ATR is just a rolling average of True Range over `period` days.
    """
    high = data["High"]
    low = data["Low"]
    prev_close = data["Close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr


# ======================================================================
# SECTION 4: POSITION SIZING FUNCTION
# ======================================================================
def calculate_shares(cash, price, atr, risk_pct=0.01):
    """
    Returns how many shares to buy so that one ATR move
    never costs more than risk_pct of current cash.
    """
    if pd.isna(atr) or atr <= 0:
        return 0
    max_risk_dollars = cash * risk_pct
    shares = max_risk_dollars / atr
    shares = min(shares, cash // price)  # can't spend more than we have
    return int(shares)


# ======================================================================
# SECTION 5: BACKTEST WITH VOLATILITY-ADJUSTED POSITION SIZING
# ======================================================================
data = get_price_data("AAPL", "2021-01-01", "2026-06-01")
data["sma"] = data["Close"].rolling(20).mean()
data["signal"] = (data["Close"] > data["sma"]).shift(1)
data["atr"] = calculate_atr(data)

cost_per_trade = 0.001
starting_cash = 10000

def backtest_fixed(data):
    """Original approach: buy as many shares as cash allows."""
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
            cash -= trade_value * (1 + cost_per_trade)
        elif signal_today == False and shares_held > 0:
            trade_value = shares_held * price_yesterday
            cash += trade_value * (1 - cost_per_trade)
            shares_held = 0

        portfolio_history.append(cash + shares_held * price_today)
    return portfolio_history


def backtest_atr_sized(data):
    """ATR-based sizing: risk exactly 1% of account per trade."""
    cash = starting_cash
    shares_held = 0
    portfolio_history = []

    for i in range(len(data)):
        price_today = data["Close"].iloc[i]
        price_yesterday = data["Close"].iloc[i - 1] if i > 0 else price_today
        signal_today = data["signal"].iloc[i]
        atr_today = data["atr"].iloc[i]

        if signal_today == True and shares_held == 0:
            shares_held = calculate_shares(cash, price_yesterday, atr_today)
            trade_value = shares_held * price_yesterday
            cash -= trade_value * (1 + cost_per_trade)
        elif signal_today == False and shares_held > 0:
            trade_value = shares_held * price_yesterday
            cash += trade_value * (1 - cost_per_trade)
            shares_held = 0

        portfolio_history.append(cash + shares_held * price_today)
    return portfolio_history


# ======================================================================
# SECTION 6: COMPARE THE TWO APPROACHES
# ======================================================================
fixed_result = backtest_fixed(data)
atr_result = backtest_atr_sized(data)

print(f"Fixed sizing final value:       ${fixed_result[-1]:,.2f}")
print(f"ATR-sized final value:          ${atr_result[-1]:,.2f}")
print(f"Fixed sizing total return:      {(fixed_result[-1]/starting_cash - 1):.2%}")
print(f"ATR-sized total return:         {(atr_result[-1]/starting_cash - 1):.2%}")

data["fixed_portfolio"] = fixed_result
data["atr_portfolio"] = atr_result

plt.plot(data.index, data["fixed_portfolio"], label="Fixed Sizing")
plt.plot(data.index, data["atr_portfolio"], label="ATR Position Sizing")
plt.title("Fixed vs. Volatility-Adjusted Position Sizing")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
plt.legend()
plt.show()