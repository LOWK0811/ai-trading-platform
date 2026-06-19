# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
import matplotlib.pyplot as plt
from src.data_handler import get_price_data
from src.indicators import add_sma, add_atr, add_signal
from src.backtester import run_backtest


# ======================================================================
# SECTION 2: LOGGING SETUP
# ======================================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("logs/trading_log.txt", mode="w"),
        logging.StreamHandler()
    ]
)
logging.getLogger("yfinance").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 3: CONFIGURE AND RUN
# ======================================================================
TICKER = "AAPL"
START  = "2021-01-01"
END    = "2026-06-01"

logger.info(f"=== Platform starting: {TICKER} | {START} to {END} ===")

data = get_price_data(TICKER, START, END)

if data is not None:
    data = add_sma(data, window=20)
    data = add_atr(data, period=14)
    data = add_signal(data)

    portfolio = run_backtest(
        data,
        starting_cash=10000,
        cost_per_trade=0.001,
        risk_pct=0.01,
        use_circuit_breaker=True,
        cb_threshold=-0.02
    )

    data["portfolio_value"] = portfolio
    data["buy_and_hold"] = (10000 // data["Close"].iloc[0]) * data["Close"]

    plt.plot(data.index, data["portfolio_value"], label="Strategy")
    plt.plot(data.index, data["buy_and_hold"], label="Buy & Hold")
    plt.title(f"{TICKER} — Full Platform")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (USD)")
    plt.legend()
    plt.show()