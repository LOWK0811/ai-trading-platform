# ======================================================================
# VOLUME FEATURES
# Volume is the market's polygraph — it reveals the conviction
# behind a price move. A price move on high volume is more
# meaningful than the same move on thin volume.
# ======================================================================
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

VOLUME_FEATURES = [
    "vol_relative_volume",
    "vol_volume_spike",
    "vol_price_volume_trend",
    "vol_volume_momentum",
]


def add_volume_features(data):
    """
    Adds volume-based features to the DataFrame.
    Note: uses 'vol_' prefix but these are VOLUME features,
    not volatility — kept consistent with naming convention.

    Features:
    - Relative volume: today vs 20-day average
    - Volume spike: binary flag for unusually high volume (>2x avg)
    - Price-volume trend: cumulative (price_change * volume)
    - Volume momentum: is volume expanding or contracting?
    """
    df = data.copy()

    avg_volume = df["Volume"].rolling(20).mean()

    # ── Relative volume ──
    df["vol_relative_volume"] = df["Volume"] / avg_volume

    # ── Volume spike flag ──
    df["vol_volume_spike"] = (
        df["Volume"] > 2 * avg_volume
    ).astype(int)

    # ── Price-volume trend (normalized) ──
    pct_change = df["Close"].pct_change()
    pvt = (pct_change * df["Volume"]).cumsum()
    df["vol_price_volume_trend"] = pvt / pvt.abs().rolling(20).max()

    # ── Volume momentum: 5-day avg vs 20-day avg ──
    # > 1: volume increasing (rising interest)
    # < 1: volume decreasing (fading interest)
    vol_5d  = df["Volume"].rolling(5).mean()
    df["vol_volume_momentum"] = vol_5d / avg_volume

    logger.debug(f"Volume features added: {VOLUME_FEATURES}")
    return df