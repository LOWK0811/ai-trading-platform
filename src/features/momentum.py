# ======================================================================
# MOMENTUM FEATURES
# Momentum captures the tendency of assets that have been rising
# to continue rising, and falling to continue falling.
# All features are percentage returns — scale-independent.
# ======================================================================
import pandas as pd
import logging

logger = logging.getLogger(__name__)

MOMENTUM_FEATURES = [
    "mom_return_1d",
    "mom_return_5d",
    "mom_return_10d",
    "mom_return_20d",
    "mom_return_60d",
    "mom_zscore_20d",
    "mom_rsi_14",
]


def add_momentum_features(data):
    """
    Adds momentum-based features to the DataFrame.

    Features:
    - Multi-window returns (1, 5, 10, 20, 60 days)
    - Z-score of 1-day return vs 20-day rolling distribution
    - RSI(14) — Relative Strength Index, normalized 0-100
    """
    df = data.copy()

    # ── Multi-window percentage returns ──
    for n in [1, 5, 10, 20, 60]:
        df[f"mom_return_{n}d"] = df["Close"].pct_change(n)

    # ── Z-score: how unusual is today's move? ──
    rolling_mean = df["mom_return_1d"].rolling(20).mean()
    rolling_std  = df["mom_return_1d"].rolling(20).std()
    df["mom_zscore_20d"] = (
        (df["mom_return_1d"] - rolling_mean) / rolling_std
    )

    # ── RSI(14): measures speed and magnitude of price changes ──
    # RSI > 70: overbought (may reverse down)
    # RSI < 30: oversold (may reverse up)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, float("inf"))
    df["mom_rsi_14"] = 100 - (100 / (1 + rs))

    logger.debug(f"Momentum features added: {MOMENTUM_FEATURES}")
    return df