# ======================================================================
# VOLATILITY FEATURES
# Volatility measures the magnitude of price uncertainty.
# High volatility = larger position risk; ATR sizing accounts for this.
# All features are normalized to be scale-independent.
# ======================================================================
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

VOLATILITY_FEATURES = [
    "vol_atr_14",
    "vol_atr_pct",
    "vol_realized_10d",
    "vol_realized_20d",
    "vol_rank_52w",
    "vol_ratio",
]


def add_volatility_features(data):
    """
    Adds volatility-based features to the DataFrame.

    Features:
    - ATR(14): Average True Range in dollars
    - ATR%: ATR as percentage of price (normalized)
    - Realized vol: rolling standard deviation of daily returns
    - Vol rank: where current volatility sits in 52-week range
    - Vol ratio: short-term vs long-term volatility comparison
    """
    df = data.copy()

    # ── ATR(14) ──
    high       = df["High"]
    low        = df["Low"]
    prev_close = df["Close"].shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    df["vol_atr_14"] = true_range.rolling(14).mean()

    # ── ATR as percentage of price ──
    df["vol_atr_pct"] = df["vol_atr_14"] / df["Close"]

    # ── Realized volatility (annualized) ──
    daily_return = df["Close"].pct_change()
    df["vol_realized_10d"] = (daily_return.rolling(10).std()
                               * np.sqrt(252))
    df["vol_realized_20d"] = (daily_return.rolling(20).std()
                               * np.sqrt(252))

    # ── Volatility rank: 0 = calmest in past year,
    #                     1 = most volatile in past year ──
    vol_52w_min = df["vol_realized_20d"].rolling(252).min()
    vol_52w_max = df["vol_realized_20d"].rolling(252).max()
    df["vol_rank_52w"] = (
        (df["vol_realized_20d"] - vol_52w_min) /
        (vol_52w_max - vol_52w_min + 1e-10)
    )

    # ── Vol ratio: short vol / long vol ──
    # > 1: volatility expanding (potentially dangerous)
    # < 1: volatility contracting (potentially calm)
    df["vol_ratio"] = (df["vol_realized_10d"] /
                       df["vol_realized_20d"].replace(0, float("inf")))

    logger.debug(f"Volatility features added: {VOLATILITY_FEATURES}")
    return df