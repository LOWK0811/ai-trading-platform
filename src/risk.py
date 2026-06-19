# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 2: POSITION SIZING
# ======================================================================
def calculate_shares(cash, price, atr, risk_pct=0.01):
    """
    Returns how many shares to buy so that one ATR move
    costs at most risk_pct of current cash.
    """
    if pd.isna(atr) or atr <= 0:
        logger.debug(f"ATR invalid ({atr}) — returning 0 shares")
        return 0
    max_risk = cash * risk_pct
    shares = int(min(max_risk / atr, cash // price))
    return shares


# ======================================================================
# SECTION 3: CIRCUIT BREAKER CHECK
# ======================================================================
def check_circuit_breaker(current_value, previous_value, threshold=-0.02):
    """
    Returns True if the portfolio dropped more than `threshold`
    since the previous day — signaling a trading halt tomorrow.
    """
    if previous_value <= 0:
        return False
    daily_pnl = (current_value / previous_value) - 1
    if daily_pnl < threshold:
        logger.warning(f"Circuit breaker triggered: {daily_pnl:.2%} drop (threshold: {threshold:.2%})")
        return True
    return False