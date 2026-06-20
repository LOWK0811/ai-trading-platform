# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from src.data_handler import get_price_data
from src.portfolio import Portfolio


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
# SECTION 3: LOAD PRICE DATA FOR ALL TICKERS
# ======================================================================
tickers = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "TSLA", "SPY"]
START   = "2021-01-01"
END     = "2026-06-01"

prices = {}
for t in tickers:
    data = get_price_data(t, START, END)
    if data is not None:
        prices[t] = data["Close"]
        logger.info(f"Loaded {t}: {len(data)} rows")

portfolio = Portfolio(prices, starting_cash=100000)


# ======================================================================
# SECTION 4: DEFINE ALLOCATION STRATEGIES
# ======================================================================
equal_w    = portfolio.equal_weight()
risk_par_w = portfolio.risk_parity()
momentum_w = portfolio.momentum_weighted(lookback=60)

# Manual concentrated tech portfolio (common mistake)
concentrated = {"AAPL": 0.40, "MSFT": 0.35, "NVDA": 0.25,
                "JPM": 0.0, "XOM": 0.0, "JNJ": 0.0,
                "TSLA": 0.0, "SPY": 0.0}

# Diversified across sectors
diversified  = {"AAPL": 0.15, "MSFT": 0.15, "NVDA": 0.10,
                "JPM": 0.15, "XOM": 0.15, "JNJ": 0.15,
                "TSLA": 0.05, "SPY": 0.10}

weights_dict = {
    "Equal Weight (1/N)":    equal_w,
    "Risk Parity":           risk_par_w,
    "Momentum Weighted":     momentum_w,
    "Concentrated Tech":     concentrated,
    "Diversified":           diversified,
}

portfolio.report(weights_dict)


# ======================================================================
# SECTION 5: VISUALIZE PORTFOLIO CURVES
# ======================================================================
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 8), sharex=False,
    gridspec_kw={"height_ratios": [2, 1]}
)
fig.patch.set_facecolor("#060d1f")

for ax in [ax1, ax2]:
    ax.set_facecolor("#060d1f")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1a3357")
    ax.spines["bottom"].set_color("#1a3357")
    ax.tick_params(colors="#7b9bc0", labelsize=8)
    ax.grid(True, color="#0d1b35", linewidth=0.8, linestyle="--")

colors = ["#00d4aa", "#1a6eff", "#ffd166",
          "#ff4d6a", "#7b61ff"]

for (name, weights), color in zip(weights_dict.items(), colors):
    pv = portfolio.simulate(weights, rebalance_frequency="monthly")
    ax1.plot(pv, color=color, linewidth=1.5, label=name)

ax1.set_ylabel("Portfolio Value ($)", color="#7b9bc0", fontsize=9)
ax1.yaxis.set_major_formatter(
    mticker.StrMethodFormatter("${x:,.0f}"))
ax1.legend(fontsize=8, facecolor="#0f2040",
           edgecolor="#1a3357", labelcolor="#e8f0fe")

# Correlation heatmap
corr = portfolio.correlation_matrix()
im   = ax2.imshow(corr.values, cmap="RdYlGn",
                   vmin=-1, vmax=1, aspect="auto")
ax2.set_xticks(range(len(tickers)))
ax2.set_yticks(range(len(tickers)))
ax2.set_xticklabels(tickers, fontsize=7, color="#7b9bc0")
ax2.set_yticklabels(tickers, fontsize=7, color="#7b9bc0")

for i in range(len(tickers)):
    for j in range(len(tickers)):
        ax2.text(j, i, f"{corr.values[i,j]:.2f}",
                ha="center", va="center",
                fontsize=6, color="#060d1f", fontweight="bold")

ax2.set_title("Correlation Matrix",
              color="#7b9bc0", fontsize=9, pad=8)

fig.suptitle(
    "Aureline Labs — Portfolio Construction & Correlation Analysis\n"
    "8 Assets · 2021–2026 · Monthly Rebalancing",
    color="#e8f0fe", fontsize=11,
    fontfamily="monospace", y=1.01
)
plt.tight_layout()
plt.savefig("data/portfolio_analysis.png",
            dpi=150, bbox_inches="tight",
            facecolor="#060d1f")
plt.show()
logger.info("Chart saved to data/portfolio_analysis.png")