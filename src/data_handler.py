# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 2: PRICE DATA LOADER WITH CACHING
# ======================================================================
def get_price_data(ticker, start, end):
    """
    Loads OHLCV data for `ticker` between `start` and `end`.
    Uses a local Parquet cache if available; downloads otherwise.
    Returns a DataFrame, or None if the download fails.
    """
    filename = f"data/{ticker}_{start}_{end}.parquet"

    if os.path.exists(filename):
        logger.info(f"Cache hit — loading {ticker} from {filename}")
        return pd.read_parquet(filename)

    logger.info(f"No cache found — downloading {ticker} from Yahoo Finance")
    try:
        data = yf.Ticker(ticker).history(start=start, end=end)
        if data.empty:
            logger.error(f"Download returned empty data for {ticker}.")
            return None
        os.makedirs("data", exist_ok=True)
        data.to_parquet(filename)
        logger.info(f"Saved {len(data)} rows to {filename}")
        return data
    except Exception as e:
        logger.error(f"Download failed for {ticker}: {e}")
        return None