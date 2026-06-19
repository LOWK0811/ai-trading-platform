# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import yfinance as yf
import pandas as pd


# ======================================================================
# SECTION 2: A REUSABLE, CACHE-AWARE DATA LOADER
# ======================================================================
def get_price_data(ticker, start, end):
    """
    Loads OHLCV data for `ticker` between `start` and `end`.
    Uses a local cached copy if one exists; only contacts
    Yahoo Finance if no local copy is found yet.
    """
    filename = f"data/{ticker}_{start}_{end}.parquet"

    if os.path.exists(filename):
        print(f"Loading {ticker} from local cache...")
        data = pd.read_parquet(filename)
    else:
        print(f"No local copy found — downloading {ticker} from Yahoo Finance...")
        data = yf.Ticker(ticker).history(start=start, end=end)
        os.makedirs("data", exist_ok=True)
        data.to_parquet(filename)

    return data


# ======================================================================
# SECTION 3: USE IT
# ======================================================================
data = get_price_data("AAPL", "2025-12-01", "2026-06-01")
print(data.head())
print(f"\nTotal rows: {len(data)}")