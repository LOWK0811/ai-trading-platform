# ======================================================================
# REGIME FEATURES
# Market regime describes the current environment —
# bull/bear/sideways, high/low volatility.
# The same strategy behaves completely differently across regimes.
# This is one of the most underappreciated concepts in retail quant.
# ======================================================================
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

REGIME_FEATURES = [
    "regime_bull_bear",
    "regime_volatility_state",
    "regime_trend_strength",
    "regime_drawdown_pct",
]


def add_regime_features(data):
    """
    Adds market regime classification features to the DataFrame.

    Features:
    - Bull/Bear: +1 if price above SMA(200), -1 if below
    - Volatility state: +1 if high vol (above median), -1 if low
    - Trend strength: ADX proxy using directional movement
    - Drawdown: how far below the 52-week high are we?
    """
    df = data.copy()

    # ── Bull/Bear regime (SMA 200 filter) ──
    sma200 = df["Close"].rolling(200).mean()
    df["regime_bull_bear"] = np.where(
        df["Close"] > sma200, 1, -1
    )

    # ── Volatility state ──
    daily_return = df["Close"].pct_change()
    realized_vol = daily_return.rolling(20).std()
    median_vol   = realized_vol.rolling(252).median()
    df["regime_volatility_state"] = np.where(
        realized_vol > median_vol, 1, -1
    )

    # ── Trend strength proxy (ADX-like) ──
    # Measures consistency of directional moves over 14 days
    up_moves   = (df["High"] - df["High"].shift(1)).clip(lower=0)
    down_moves = (df["Low"].shift(1) - df["Low"]).clip(lower=0)
    directional_diff = (up_moves - down_moves).abs()
    total_movement   = up_moves + down_moves + 1e-10
    df["regime_trend_strength"] = (
        directional_diff / total_movement
    ).rolling(14).mean()

    # ── Current drawdown from 52-week high ──
    high_52w = df["Close"].rolling(252).max()
    df["regime_drawdown_pct"] = (df["Close"] - high_52w) / high_52w

    logger.debug(f"Regime features added: {REGIME_FEATURES}")
    return df