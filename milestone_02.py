# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import yfinance as yf
import matplotlib.pyplot as plt


# ======================================================================
# SECTION 2: DOWNLOAD REAL HISTORICAL DATA
# ======================================================================
ticker = "AAPL"
stock = yf.Ticker(ticker)
data = stock.history(start="2025-12-01", end="2026-06-01")


# ======================================================================
# SECTION 3: TAKE A LOOK AT THE DATA
# ======================================================================
print(data.head())
print(f"\nColumns: {list(data.columns)}")
print(f"Total trading days downloaded: {len(data)}")


# ======================================================================
# SECTION 4: YOUR FIRST PRICE CHART
# ======================================================================
plt.plot(data.index, data["Close"])
plt.title(f"{ticker} Closing Price")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.show()