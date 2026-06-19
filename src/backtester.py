# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import logging
from src.risk import calculate_shares, check_circuit_breaker

logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 2: THE BACKTEST ENGINE
# ======================================================================
def run_backtest(data, starting_cash=10000, cost_per_trade=0.001,
                 risk_pct=0.01, use_circuit_breaker=True, cb_threshold=-0.02):
    """
    Runs a full backtest on pre-prepared data.
    Expects columns: Close, signal, atr (from indicators.py).
    Returns a list of daily portfolio values.
    """
    cash = starting_cash
    shares_held = 0
    portfolio_history = []
    halt_next_day = False
    previous_value = starting_cash

    num_buys = 0
    num_sells = 0
    num_halts = 0

    for i in range(len(data)):
        date = data.index[i].date()
        price_today = data["Close"].iloc[i]
        price_yesterday = data["Close"].iloc[i - 1] if i > 0 else price_today
        signal_today = data["signal"].iloc[i]
        atr_today = data["atr"].iloc[i]

        if use_circuit_breaker and halt_next_day:
            if shares_held > 0:
                proceeds = shares_held * price_today * (1 - cost_per_trade)
                cash += proceeds
                shares_held = 0
                logger.debug(f"{date} | HALT — forced exit at ${price_today:.2f}")
            num_halts += 1
        else:
            if signal_today == True and shares_held == 0:
                shares = calculate_shares(cash, price_yesterday, atr_today, risk_pct)
                if shares > 0:
                    cost = shares * price_yesterday * (1 + cost_per_trade)
                    cash -= cost
                    shares_held = shares
                    num_buys += 1
                    logger.debug(f"{date} | BUY  {shares} @ ${price_yesterday:.2f} | ATR={atr_today:.2f} | Cash: ${cash:.2f}")
            elif signal_today == False and shares_held > 0:
                proceeds = shares_held * price_yesterday * (1 - cost_per_trade)
                cash += proceeds
                shares_held = 0
                num_sells += 1
                logger.debug(f"{date} | SELL {shares_held} @ ${price_yesterday:.2f} | Cash: ${cash:.2f}")

        current_value = cash + shares_held * price_today
        portfolio_history.append(current_value)

        halt_next_day = check_circuit_breaker(current_value, previous_value, cb_threshold)
        previous_value = current_value

    final_value = portfolio_history[-1]
    total_return = (final_value / starting_cash) - 1

    logger.info(f"Buys: {num_buys} | Sells: {num_sells} | Halts: {num_halts}")
    logger.info(f"Final value: ${final_value:,.2f} | Return: {total_return:.2%}")

    return portfolio_history