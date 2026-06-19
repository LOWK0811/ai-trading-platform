# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from src.data_handler import get_price_data
from src.indicators import add_sma, add_atr
from src.risk import calculate_shares


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
# SECTION 3: METRICS FUNCTIONS
# ======================================================================
def cagr(portfolio_values, start, end):
    """Compound Annual Growth Rate."""
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    if years <= 0:
        return 0.0
    total_return = portfolio_values[-1] / portfolio_values[0]
    return total_return ** (1 / years) - 1


def sharpe_ratio(portfolio_values, risk_free_rate=0.05):
    values = pd.Series(portfolio_values)
    daily_returns = values.pct_change().dropna()
    daily_rf = risk_free_rate / 252
    excess = daily_returns - daily_rf
    if excess.std() == 0:
        return 0.0
    return (excess.mean() / excess.std()) * np.sqrt(252)


def max_drawdown(portfolio_values):
    values = pd.Series(portfolio_values)
    rolling_peak = values.cummax()
    drawdown = (values - rolling_peak) / rolling_peak
    return drawdown.min()


def win_rate(portfolio_values):
    """
    Percentage of days where the strategy made money.
    A 'win' is any day the portfolio value increased.
    """
    values = pd.Series(portfolio_values)
    daily_returns = values.pct_change().dropna()
    wins = (daily_returns > 0).sum()
    return wins / len(daily_returns)


# ======================================================================
# SECTION 4: SINGLE-TICKER BACKTEST FUNCTION
# ======================================================================
def run_backtest(ticker, start, end, sma_window=20,
                 cost_per_trade=0.001, starting_cash=10000):
    """
    Runs the full SMA strategy backtest for one ticker.
    Returns a dict of performance metrics, or None on failure.
    """
    data = get_price_data(ticker, start, end)
    if data is None or len(data) < sma_window + 10:
        logger.warning(f"{ticker}: insufficient data, skipping")
        return None

    data = add_sma(data, window=sma_window)
    data = add_atr(data)
    data["signal"] = (data["Close"] > data["sma"]).shift(1)

    cash = starting_cash
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
                cash -= shares * price_yesterday * (1 + cost_per_trade)
                shares_held = shares
                num_trades += 1
        elif signal_today == False and shares_held > 0:
            cash += shares_held * price_yesterday * (1 - cost_per_trade)
            shares_held = 0
            num_trades += 1

        portfolio.append(cash + shares_held * price_today)

    # Buy and hold over same period
    bh_shares = starting_cash // data["Close"].iloc[0]
    bh_final  = bh_shares * data["Close"].iloc[-1]
    bh_return = (bh_final / starting_cash) - 1

    return {
        "ticker":        ticker,
        "total_return":  (portfolio[-1] / starting_cash) - 1,
        "cagr":          cagr(portfolio, start, end),
        "sharpe":        sharpe_ratio(portfolio),
        "max_drawdown":  max_drawdown(portfolio),
        "win_rate":      win_rate(portfolio),
        "num_trades":    num_trades // 2,       # round trips
        "final_value":   portfolio[-1],
        "bh_return":     bh_return,
        "portfolio":     portfolio,
        "dates":         data.index
    }


# ======================================================================
# SECTION 5: RUN ACROSS MULTIPLE TICKERS
# ======================================================================
START = "2021-01-01"
END   = "2026-06-01"

tickers = [
    "AAPL",   # Apple — large cap tech
    "MSFT",   # Microsoft — large cap tech
    "NVDA",   # Nvidia — semiconductors / AI
    "JPM",    # JPMorgan — financials
    "XOM",    # ExxonMobil — energy
    "JNJ",    # Johnson & Johnson — healthcare
    "TSLA",   # Tesla — high-volatility growth
    "SPY",    # S&P 500 ETF — broad market benchmark
]

results = []
portfolio_curves = {}

for t in tickers:
    logger.info(f"Running backtest: {t}")
    result = run_backtest(t, START, END)
    if result:
        results.append(result)
        portfolio_curves[t] = (result["dates"], result["portfolio"])

logger.info(f"\nCompleted {len(results)}/{len(tickers)} tickers")


# ======================================================================
# SECTION 6: RESULTS TABLE
# ======================================================================
df_results = pd.DataFrame([{
    "Ticker":        r["ticker"],
    "Total Return":  f"{r['total_return']:+.2%}",
    "CAGR":          f"{r['cagr']:+.2%}",
    "Sharpe":        f"{r['sharpe']:.3f}",
    "Max Drawdown":  f"{r['max_drawdown']:.2%}",
    "Win Rate":      f"{r['win_rate']:.1%}",
    "Round Trips":   r["num_trades"],
    "B&H Return":    f"{r['bh_return']:+.2%}",
    "Beat B&H":      "✓" if r["total_return"] > r["bh_return"] else "✗"
} for r in results])

print("\n" + "="*85)
print("  DRIFT LABS — MULTI-TICKER STRATEGY RESULTS")
print("="*85)
print(df_results.to_string(index=False))
print("="*85)

beat_count = sum(1 for r in results if r["total_return"] > r["bh_return"])
print(f"\nStrategy beat Buy & Hold on {beat_count}/{len(results)} tickers")

avg_sharpe = np.mean([r["sharpe"] for r in results])
avg_mdd    = np.mean([r["max_drawdown"] for r in results])
print(f"Average Sharpe ratio: {avg_sharpe:.3f}")
print(f"Average Max Drawdown: {avg_mdd:.2%}")


# ======================================================================
# SECTION 7: COMPARISON CHART
# ======================================================================
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.patch.set_facecolor("#f0f7f4")
axes = axes.flatten()

colors = ["#0d5c3e", "#1a9e6c", "#2d8a5e", "#4aab7e",
          "#6cc49e", "#0a4530", "#3d7a60", "#8ed4b4"]

for idx, result in enumerate(results):
    ax = axes[idx]
    ax.set_facecolor("#f8fbf9")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c5e8d8")
    ax.spines["bottom"].set_color("#c5e8d8")

    portfolio = result["portfolio"]
    normalized = [v / 10000 for v in portfolio]
    ax.plot(normalized, color=colors[idx], linewidth=1.5)
    ax.axhline(y=1.0, color="#c5e8d8", linestyle="--", linewidth=0.8)

    ret_color = "#0d5c3e" if result["total_return"] > 0 else "#e53e3e"
    ax.set_title(result["ticker"],
                 fontsize=11, fontweight="bold", color="#0d5c3e", pad=6)
    ax.text(0.98, 0.05,
            f"{result['total_return']:+.1%}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=9, color=ret_color,
            fontfamily="monospace")
    ax.text(0.98, 0.18,
            f"SR: {result['sharpe']:.2f}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=8, color="#4a7c6a",
            fontfamily="monospace")
    ax.tick_params(colors="#4a7c6a", labelsize=7)
    ax.set_ylabel("Return (×)", fontsize=7, color="#4a7c6a")

fig.suptitle("Drift Labs — SMA Strategy Performance Across Tickers\n"
             "2021–2026 · ATR Position Sizing · 0.1% Friction",
             fontsize=12, fontweight="bold", color="#0d5c3e", y=1.01)
plt.tight_layout()
plt.savefig("data/multi_ticker_results.png",
            dpi=150, bbox_inches="tight",
            facecolor="#f0f7f4")
plt.show()
logger.info("Chart saved to data/multi_ticker_results.png")