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
# SECTION 3: BUILD THE SIGNAL — CAREFULLY, WITHOUT LOOKING AHEAD
# ======================================================================
data = get_price_data("AAPL", "2021-01-01", "2026-06-01")

window = 20
data["sma"] = data["Close"].rolling(window).mean()
data["signal"] = (data["Close"] > data["sma"]).shift(1)


# ======================================================================
# SECTION 4: WALK THROUGH TIME, DAY BY DAY
# ======================================================================
starting_cash = 10000
cash = starting_cash
shares_held = 0
portfolio_history = []

for i in range(len(data)):
    price_today = data["Close"].iloc[i]
    signal_today = data["signal"].iloc[i]

    if signal_today == True and shares_held == 0:
        shares_held = cash // price_today   # whole shares only
        cash -= shares_held * price_today
    elif signal_today == False and shares_held > 0:
        cash += shares_held * price_today
        shares_held = 0

    portfolio_history.append(cash + (shares_held * price_today))

data["portfolio_value"] = portfolio_history


# ======================================================================
# SECTION 5: HOW DID WE DO?
# ======================================================================
final_value = data["portfolio_value"].iloc[-1]
print(f"Starting cash: ${starting_cash:,.2f}")
print(f"Final portfolio value: ${final_value:,.2f}")
print(f"Total return: {(final_value / starting_cash - 1):.2%}")


# ======================================================================
# SECTION 6: COMPARE TO JUST BUYING AND HOLDING
# ======================================================================
buy_and_hold_shares = starting_cash // data["Close"].iloc[0]
data["buy_and_hold_value"] = buy_and_hold_shares * data["Close"]

plt.plot(data.index, data["portfolio_value"], label="SMA Strategy")
plt.plot(data.index, data["buy_and_hold_value"], label="Buy & Hold")
plt.title("Strategy vs. Buy & Hold")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
plt.legend()
plt.show()