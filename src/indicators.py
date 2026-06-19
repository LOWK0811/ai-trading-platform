# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 2: SIMPLE MOVING AVERAGE
# ======================================================================
def add_sma(data, window=20):
    """Adds a simple moving average column to the DataFrame."""
    data = data.copy()
    data["sma"] = data["Close"].rolling(window).mean()
    logger.debug(f"SMA({window}) calculated")
    return data


# ======================================================================
# SECTION 3: AVERAGE TRUE RANGE
# ======================================================================
def add_atr(data, period=14):
    """
    Adds ATR (Average True Range) to the DataFrame.
    ATR measures typical daily price movement including overnight gaps.
    """
    data = data.copy()
    high = data["High"]
    low = data["Low"]
    prev_close = data["Close"].shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    data["atr"] = tr.rolling(period).mean()
    logger.debug(f"ATR({period}) calculated")
    return data


# ======================================================================
# SECTION 4: BUY/SELL SIGNAL
# ======================================================================
def add_signal(data):
    """
    Adds a trading signal: True = price above SMA (buy),
    False = price below SMA (sell/stay out).
    Shifted by 1 day to prevent look-ahead bias.
    Requires add_sma() to have been called first.
    """
    if "sma" not in data.columns:
        logger.error("SMA column missing — run add_sma() before add_signal()")
        return data
    data = data.copy()
    data["signal"] = (data["Close"] > data["sma"]).shift(1)
    return data