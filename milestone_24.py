# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
from src.experiment_tracker import ExperimentTracker
from src.data_handler import get_price_data
from src.indicators import add_sma, add_atr
from src.risk import calculate_shares
from src.features import build_features, feature_cols
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


# ======================================================================
# SECTION 2: LOGGING SETUP
# ======================================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logging.getLogger("yfinance").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 3: METRIC HELPERS
# ======================================================================
def sharpe_ratio(portfolio, risk_free_rate=0.05):
    values = pd.Series(portfolio)
    dr = values.pct_change().dropna()
    excess = dr - risk_free_rate / 252
    if excess.std() == 0:
        return 0.0
    return round((excess.mean() / excess.std()) * np.sqrt(252), 3)


def max_drawdown(portfolio):
    values = pd.Series(portfolio)
    peak = values.cummax()
    return round(((values - peak) / peak).min() * 100, 2)


def cagr(portfolio, start, end):
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    total = portfolio[-1] / portfolio[0]
    return round((total ** (1 / years) - 1) * 100, 2)


def win_rate(portfolio):
    values = pd.Series(portfolio)
    dr = values.pct_change().dropna()
    return round((dr > 0).mean() * 100, 1)


# ======================================================================
# SECTION 4: RUN A BACKTEST AND LOG IT
# ======================================================================
def run_and_log(tracker, ticker, start, end,
                hypothesis, sma_window=20):
    """Runs the SMA backtest and logs it as an experiment."""
    data = get_price_data(ticker, start, end)
    data = add_sma(data, window=sma_window)
    data = add_atr(data)
    data["signal"] = (data["Close"] > data["sma"]).shift(1)

    cash = 10000
    shares_held = 0
    portfolio = []
    num_trades = 0

    for i in range(len(data)):
        price_today     = data["Close"].iloc[i]
        price_yesterday = data["Close"].iloc[i-1] if i > 0 else price_today
        signal_today    = data["signal"].iloc[i]
        atr_today       = data["atr"].iloc[i]

        if signal_today == True and shares_held == 0:
            shares = calculate_shares(cash, price_yesterday, atr_today)
            if shares > 0:
                cash -= shares * price_yesterday * 1.001
                shares_held = shares
                num_trades += 1
        elif signal_today == False and shares_held > 0:
            cash += shares_held * price_yesterday * 0.999
            shares_held = 0
            num_trades += 1

        portfolio.append(cash + shares_held * price_today)

    metrics = {
        "Total Return": f"{((portfolio[-1]/10000)-1)*100:.2f}%",
        "CAGR":         f"{cagr(portfolio, start, end)}%",
        "Sharpe":       sharpe_ratio(portfolio),
        "Max Drawdown": f"{max_drawdown(portfolio)}%",
        "Win Rate":     f"{win_rate(portfolio)}%",
        "Round Trips":  num_trades // 2
    }

    beat_bh = portfolio[-1] > (10000 // data["Close"].iloc[0]) \
              * data["Close"].iloc[-1]

    conclusion = (
        f"The SMA({sma_window}) strategy on {ticker} produced a "
        f"Sharpe of {metrics['Sharpe']} and CAGR of {metrics['CAGR']} "
        f"over the study period. "
        f"Strategy {'outperformed' if beat_bh else 'underperformed'} "
        f"buy-and-hold. "
        f"ATR-based position sizing kept max drawdown at "
        f"{metrics['Max Drawdown']}."
    )

    exp_id = tracker.log(
        hypothesis  = hypothesis,
        ticker      = ticker,
        start       = start,
        end         = end,
        strategy    = f"SMA_{sma_window}",
        parameters  = {
            "sma_window":      sma_window,
            "cost_per_trade":  "0.1%",
            "position_sizing": "ATR 1%",
            "starting_cash":   "$10,000"
        },
        features    = [],
        metrics     = metrics,
        conclusion  = conclusion,
        tags        = ["sma", "trend-following", "atr-sizing"]
    )

    return exp_id, portfolio[-1]


# ======================================================================
# SECTION 5: LOG MULTIPLE EXPERIMENTS
# ======================================================================
tracker = ExperimentTracker()
START   = "2021-01-01"
END     = "2026-06-01"

experiments = [
    ("AAPL", 20,
     "Does a 20-day SMA crossover with ATR position sizing "
     "generate risk-adjusted returns above the risk-free rate "
     "on AAPL over a 5-year period?"),
    ("MSFT", 20,
     "Does the same SMA(20) system generalize to MSFT, "
     "or is any edge AAPL-specific?"),
    ("NVDA", 20,
     "Does SMA(20) capture the AI-driven momentum in NVDA "
     "despite high volatility causing ATR to reduce position size?"),
    ("AAPL", 10,
     "Does a faster SMA(10) window improve responsiveness "
     "on AAPL at the cost of more whipsaw trades?"),
    ("AAPL", 50,
     "Does a slower SMA(50) window reduce whipsaw and "
     "improve Sharpe on AAPL at the cost of entry lag?"),
]

logger.info("Logging experiments to registry...")

for ticker, window, hypothesis in experiments:
    exp_id, final_val = run_and_log(
        tracker, ticker, START, END, hypothesis, sma_window=window)
    logger.info(f"  {exp_id}: {ticker} SMA({window}) → "
               f"${final_val:,.2f}")


# ======================================================================
# SECTION 6: QUERY THE REGISTRY
# ======================================================================
print("\n")
tracker.summary()

print("\n--- All Experiments ---")
df = tracker.get_all()
print(df.to_string(index=False))

print("\n--- Best by Sharpe Ratio ---")
best = tracker.get_best(metric="Sharpe", n=3)
for exp in best:
    print(f"  {exp['id']} | {exp['ticker']} "
          f"SMA({exp['parameters']['sma_window']}) "
          f"| Sharpe: {exp['metrics']['Sharpe']}")

print("\n--- AAPL Experiments ---")
aapl_exps = tracker.get_by_ticker("AAPL")
for exp in aapl_exps:
    print(f"  {exp['id']} | {exp['strategy']} "
          f"| Return: {exp['metrics']['Total Return']} "
          f"| Sharpe: {exp['metrics']['Sharpe']}")

print(f"\nReports written to: experiments/reports/")
print(f"Open any .md file in VS Code to read the full report.")