# ======================================================================
# TREND FEATURES
# Trend features capture whether the asset is in a sustained
# directional move. SMA/EMA distances tell you how far price
# has drifted from its equilibrium — the "drift" in Drift Labs.
# ======================================================================
import pandas as pd
import logging

logger = logging.getLogger(__name__)

TREND_FEATURES = [
    "trend_dist_sma20",
    "trend_dist_sma50",
    "trend_dist_sma200",
    "trend_dist_ema20",
    "trend_sma20_slope",
    "trend_above_sma200",
]


def add_trend_features(data):
    """
    Adds trend-based features to the DataFrame.

    Features:
    - Distance from SMA(20), SMA(50), SMA(200) — normalized by price
    - Distance from EMA(20) — exponentially weighted, more recent-biased
    - SMA(20) slope: is the trend accelerating or decelerating?
    - Binary: is price above SMA(200)? (the classic bull/bear divider)
    """
    df = data.copy()

    # ── SMA distances (normalized) ──
    for window in [20, 50, 200]:
        sma = df["Close"].rolling(window).mean()
        df[f"trend_dist_sma{window}"] = (df["Close"] - sma) / df["Close"]

    # ── EMA(20) distance ──
    # EMA reacts faster than SMA — useful for detecting early reversals
    ema20 = df["Close"].ewm(span=20, adjust=False).mean()
    df["trend_dist_ema20"] = (df["Close"] - ema20) / df["Close"]

    # ── SMA(20) slope: rate of change of the average ──
    # Positive slope = uptrend strengthening
    # Negative slope = downtrend or deceleration
    sma20 = df["Close"].rolling(20).mean()
    df["trend_sma20_slope"] = sma20.pct_change(5)

    # ── Is price above SMA(200)? ──
    # The most widely watched trend filter in institutional finance
    df["trend_above_sma200"] = (
        df["Close"] > df["Close"].rolling(200).mean()
    ).astype(int)

    logger.debug(f"Trend features added: {TREND_FEATURES}")
    return df