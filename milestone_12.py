# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
import yfinance as yf
import pandas as pd


# ======================================================================
# SECTION 2: SET UP THE LOGGER
# ======================================================================
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("logs/trading_log.txt", mode="w"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 3: LOGGED DATA LOADER
# ======================================================================
def get_price_data(ticker, start, end):
    filename = f"data/{ticker}_{start}_{end}.parquet"
    if os.path.exists(filename):
        logger.info(f"Cache hit — loading {ticker} from {filename}")
        data = pd.read_parquet(filename)
    else:
        logger.info(f"No cache found — downloading {ticker} from Yahoo Finance")
        try:
            data = yf.Ticker(ticker).history(start=start, end=end)
            if data.empty:
                logger.error(f"Download returned empty data for {ticker}. Check ticker symbol and date range.")
                return None
            os.makedirs("data", exist_ok=True)
            data.to_parquet(filename)
            logger.info(f"Saved {len(data)} rows to {filename}")
        except Exception as e:
            logger.error(f"Download failed for {ticker}: {e}")
            return None
    return data


# ======================================================================
# SECTION 4: ATR CALCULATOR (FROM MILESTONE 11)
# ======================================================================
def calculate_atr(data, period=14):
    high = data["High"]
    low = data["Low"]
    prev_close = data["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ======================================================================
# SECTION 5: POSITION SIZING (FROM MILESTONE 11)
# ======================================================================
def calculate_shares(cash, price, atr, risk_pct=0.01):
    if pd.isna(atr) or atr <= 0:
        logger.debug(f"ATR invalid ({atr:.4f}) — skipping trade")
        return 0
    max_risk = cash * risk_pct
    shares = int(min(max_risk / atr, cash // price))
    return shares


# ======================================================================
# SECTION 6: BACKTEST WITH FULL LOGGING
# ======================================================================
def run_backtest(ticker, start, end, sma_window=20, risk_pct=0.01):
    logger.info(f"=== Backtest starting: {ticker} | {start} to {end} | SMA={sma_window} | Risk={risk_pct:.0%} ===")

    data = get_price_data(ticker, start, end)
    if data is None:
        logger.error("Aborting backtest — no data available.")
        return

    data["sma"] = data["Close"].rolling(sma_window).mean()
    data["signal"] = (data["Close"] > data["sma"]).shift(1)
    data["atr"] = calculate_atr(data)

    cash = 10000.0
    shares_held = 0
    num_buys = 0
    num_sells = 0
    portfolio_history = []

    for i in range(len(data)):
        date = data.index[i].date()
        price_today = data["Close"].iloc[i]
        price_yesterday = data["Close"].iloc[i - 1] if i > 0 else price_today
        signal_today = data["signal"].iloc[i]
        atr_today = data["atr"].iloc[i]

        if signal_today == True and shares_held == 0:
            shares = calculate_shares(cash, price_yesterday, atr_today, risk_pct)
            if shares > 0:
                cost = shares * price_yesterday * 1.001
                cash -= cost
                shares_held = shares
                num_buys += 1
                logger.debug(f"{date} | BUY  {shares} shares @ ${price_yesterday:.2f} | ATR={atr_today:.2f} | Cash left: ${cash:.2f}")
            else:
                logger.warning(f"{date} | BUY signal fired but 0 shares calculated — skipping")

        elif signal_today == False and shares_held > 0:
            proceeds = shares_held * price_yesterday * 0.999
            logger.debug(f"{date} | SELL {shares_held} shares @ ${price_yesterday:.2f} | Proceeds: ${proceeds:.2f}")
            cash += proceeds
            shares_held = 0
            num_sells += 1

        portfolio_value = cash + shares_held * price_today
        portfolio_history.append(portfolio_value)

    final_value = portfolio_history[-1]
    total_return = (final_value / 10000) - 1

    logger.info(f"Total buys: {num_buys} | Total sells: {num_sells}")
    logger.info(f"Final portfolio value: ${final_value:,.2f}")
    logger.info(f"Total return: {total_return:.2%}")
    logger.info("=== Backtest complete ===")


# ======================================================================
# SECTION 7: RUN IT, INCLUDING A DELIBERATE ERROR TO TEST LOGGING
# ======================================================================
run_backtest("AAPL", "2021-01-01", "2026-06-01")

logger.info("--- Now testing error handling with a bad ticker ---")
run_backtest("INVALIDTICKER999", "2021-01-01", "2026-06-01")