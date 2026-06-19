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
# SECTION 3: BUILD THE SIGNAL (SAME AS MILESTONE 6)
# ======================================================================
data = get_price_data("AAPL", "2021-01-01", "2026-06-01")

window = 20
data["sma"] = data["Close"].rolling(window).mean()
data["signal"] = (data["Close"] > data["sma"]).shift(1)


# ======================================================================
# SECTION 4: WALK THROUGH TIME, NOW WITH TRANSACTION FRICTION
# ======================================================================
starting_cash = 10000
cash = starting_cash
shares_held = 0
portfolio_history = []

cost_per_trade = 0.001  # 0.1% per trade — commission + slippage combined
total_friction_paid = 0.0
num_trades = 0

for i in range(len(data)):
    price_today = data["Close"].iloc[i]
    signal_today = data["signal"].iloc[i]

    if signal_today == True and shares_held == 0:
        shares_held = cash // price_today
        trade_value = shares_held * price_today
        friction = trade_value * cost_per_trade
        cash -= (trade_value + friction)
        total_friction_paid += friction
        num_trades += 1
    elif signal_today == False and shares_held > 0:
        trade_value = shares_held * price_today
        friction = trade_value * cost_per_trade
        cash += (trade_value - friction)
        shares_held = 0
        total_friction_paid += friction
        num_trades += 1

    portfolio_history.append(cash + (shares_held * price_today))

data["portfolio_value"] = portfolio_history


# ======================================================================
# SECTION 5: HOW DID WE DO, FRICTION INCLUDED?
# ======================================================================
final_value = data["portfolio_value"].iloc[-1]

print(f"Number of trades: {num_trades}")
print(f"Total friction paid: ${total_friction_paid:,.2f}")
print(f"Final portfolio value: ${final_value:,.2f}")
print(f"Total return: {(final_value / starting_cash - 1):.2%}")


# ======================================================================
# SECTION 6: COMPARE TO BUY & HOLD
# ======================================================================
buy_and_hold_shares = starting_cash // data["Close"].iloc[0]
data["buy_and_hold_value"] = buy_and_hold_shares * data["Close"]

plt.plot(data.index, data["portfolio_value"], label="SMA Strategy (with friction)")
plt.plot(data.index, data["buy_and_hold_value"], label="Buy & Hold")
plt.title("Strategy (Friction-Adjusted) vs. Buy & Hold")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
plt.legend()
plt.show()