# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.data_handler import get_price_data
from src.indicators import add_sma, add_atr
from src.risk import calculate_shares


# ======================================================================
# SECTION 2: PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="Drift Labs",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ======================================================================
# SECTION 3: CUSTOM CSS — DRIFT LABS VISUAL IDENTITY
# ======================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500&family=Space+Mono:wght@400;700&display=swap');

/* ── Base ── */
.stApp {
    background-color: #f0f7f4;
    font-family: 'Inter', sans-serif;
    color: #2d3748;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #0d5c3e !important;
}
[data-testid="stSidebar"] * {
    color: #e8f5f0 !important;
}
[data-testid="stSidebar"] .stTextInput input {
    background-color: #1a9e6c22 !important;
    border: 1px solid #1a9e6c !important;
    color: #e8f5f0 !important;
    font-family: 'Space Mono', monospace !important;
    border-radius: 6px;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
    margin-top: 4px;
}
[data-testid="stSidebar"] hr {
    border-color: #1a9e6c44 !important;
}

/* ── Main header ── */
h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    color: #0d5c3e !important;
    letter-spacing: -0.5px;
}
h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    color: #0d5c3e !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #c5e8d8;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 1px 4px rgba(26,158,108,0.08);
}
[data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: #4a7c6a !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #0d5c3e !important;
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid #c5e8d8;
    margin: 8px 0 20px 0;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    color: #1a9e6c;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 4px;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #c5e8d8;
    border-radius: 8px;
    overflow: hidden;
}

/* ── Charts background ── */
.stPlotlyChart, .stPyplot {
    background: #ffffff;
    border-radius: 10px;
    border: 1px solid #c5e8d8;
    padding: 8px;
}

/* ── Info/warning boxes ── */
.stAlert {
    border-radius: 8px;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #1a9e6c !important;
}

/* ── Checkbox ── */
[data-testid="stCheckbox"] label {
    color: #2d3748;
    font-family: 'Inter', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# ======================================================================
# SECTION 4: HELPER FUNCTIONS
# ======================================================================
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


def load_paper_account():
    path = "data/paper_account.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"cash": 100000, "positions": {}, "orders": []}


def drift_chart_style(ax, fig):
    """Apply Drift Labs visual style to a matplotlib axes."""
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fbf9")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c5e8d8")
    ax.spines["bottom"].set_color("#c5e8d8")
    ax.tick_params(colors="#4a7c6a", labelsize=9)
    ax.yaxis.label.set_color("#4a7c6a")
    ax.xaxis.label.set_color("#4a7c6a")
    ax.grid(True, color="#e8f5f0", linewidth=0.8, linestyle="--")


# ======================================================================
# SECTION 5: DATA LOADER (CACHED)
# ======================================================================
@st.cache_data
def load_and_run(ticker, start, end, sma_window):
    data = get_price_data(ticker, start, end)
    if data is None or data.empty:
        return None, None, None
    data = add_sma(data, window=sma_window)
    data = add_atr(data)
    data["signal"] = (data["Close"] > data["sma"]).shift(1)

    cost_per_trade = 0.001
    starting_cash  = 10000
    cash = starting_cash
    shares_held = 0
    portfolio = []

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
        elif signal_today == False and shares_held > 0:
            cash += shares_held * price_yesterday * (1 - cost_per_trade)
            shares_held = 0

        portfolio.append(cash + shares_held * price_today)

    bh_shares = starting_cash // data["Close"].iloc[0]
    bh = (bh_shares * data["Close"]).tolist()
    return data, portfolio, bh


# ======================================================================
# SECTION 6: SIDEBAR
# ======================================================================
with st.sidebar:
    st.markdown("""
    <div style='padding: 12px 0 20px 0;'>
        <div style='font-family: Space Grotesk, sans-serif;
                    font-size: 1.5rem; font-weight: 700;
                    color: #e8f5f0; letter-spacing: -0.5px;'>
            📡 Drift Labs
        </div>
        <div style='font-size: 0.72rem; color: #7ecaaa;
                    font-family: Space Mono, monospace;
                    margin-top: 2px; letter-spacing: 0.1em;'>
            QUANT RESEARCH PLATFORM v0.1
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**STRATEGY CONTROLS**")
    ticker     = st.text_input("Ticker Symbol", value="AAPL").upper()
    start_date = st.date_input("Start Date",
                  value=pd.Timestamp("2021-01-01")).strftime("%Y-%m-%d")
    end_date   = st.date_input("End Date",
                  value=pd.Timestamp("2026-06-01")).strftime("%Y-%m-%d")
    sma_window = st.slider("SMA Window (days)", 5, 100, 20)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#7ecaaa; font-family: Space Mono, monospace;
                line-height: 1.8;'>
        Built by Louis Andre<br>
        Ateneo de Manila University<br>
        Applied Mathematics · Math Finance
    </div>
    """, unsafe_allow_html=True)


# ======================================================================
# SECTION 7: HEADER
# ======================================================================
st.markdown("""
<div style='display:flex; align-items:baseline; gap:12px; margin-bottom:4px;'>
    <h1 style='margin:0;'>Drift Labs</h1>
    <span style='font-family: Space Mono, monospace; font-size:0.75rem;
                 color:#1a9e6c; background:#e8f5f0; padding:3px 8px;
                 border-radius:4px; border:1px solid #c5e8d8;'>
        PAPER TRADING
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style='font-family: Space Mono, monospace; font-size:0.8rem;
            color:#4a7c6a; margin-bottom:16px;'>
    {ticker} &nbsp;·&nbsp; {start_date} → {end_date}
    &nbsp;·&nbsp; SMA({sma_window})
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ======================================================================
# SECTION 8: LOAD DATA
# ======================================================================
with st.spinner(f"Loading {ticker}..."):
    data, portfolio, bh = load_and_run(ticker, start_date, end_date, sma_window)

if data is None:
    st.error(f"Could not load data for **{ticker}**. "
             f"Check the ticker symbol and try again.")
    st.stop()


# ======================================================================
# SECTION 9: METRICS ROW
# ======================================================================
latest_price  = data["Close"].iloc[-1]
final_value   = portfolio[-1]
total_return  = (final_value / 10000) - 1
sharpe        = sharpe_ratio(portfolio)
mdd           = max_drawdown(portfolio)
bh_return     = (bh[-1] / 10000) - 1

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Latest Price",   f"${latest_price:.2f}")
col2.metric("Portfolio Value",f"${final_value:,.2f}")
col3.metric("Strategy Return",f"{total_return:+.2%}",
            delta=f"{(total_return - bh_return):+.2%} vs B&H")
col4.metric("Sharpe Ratio",   f"{sharpe:.3f}")
col5.metric("Max Drawdown",   f"{mdd:.2%}")
col6.metric("Buy & Hold Ret.",f"{bh_return:+.2%}")

st.markdown("<br>", unsafe_allow_html=True)


# ======================================================================
# SECTION 10: PRICE + SMA CHART
# ======================================================================
st.markdown('<div class="section-label">Price History & Signal</div>',
            unsafe_allow_html=True)

fig1, ax1 = plt.subplots(figsize=(13, 3.5))
drift_chart_style(ax1, fig1)

ax1.plot(data.index, data["Close"],
         color="#0d5c3e", linewidth=1.2, label="Close Price")
ax1.plot(data.index, data["sma"],
         color="#1a9e6c", linewidth=1, linestyle="--",
         alpha=0.8, label=f"SMA({sma_window})")

# Shade regions where signal is True (in the market)
signal = data["signal"].fillna(False)
in_market = signal.astype(bool)
ax1.fill_between(data.index, data["Close"].min(), data["Close"].max(),
                 where=in_market, alpha=0.06, color="#1a9e6c",
                 label="In Market")

ax1.set_ylabel("Price (USD)", fontsize=9)
ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
legend = ax1.legend(fontsize=8, framealpha=0.9,
                    facecolor="#ffffff", edgecolor="#c5e8d8")
plt.tight_layout()
st.pyplot(fig1)
plt.close()


# ======================================================================
# SECTION 11: PERFORMANCE + DRAWDOWN CHART
# ======================================================================
st.markdown('<div class="section-label">Strategy Performance vs Benchmark</div>',
            unsafe_allow_html=True)

fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(13, 6),
                                  sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})

drift_chart_style(ax2, fig2)
drift_chart_style(ax3, fig2)

ax2.plot(data.index, portfolio,
         color="#0d5c3e", linewidth=1.5, label="SMA Strategy")
ax2.plot(data.index, bh,
         color="#1a9e6c", linewidth=1.2, linestyle="--",
         alpha=0.7, label="Buy & Hold")
ax2.set_ylabel("Portfolio Value (USD)", fontsize=9)
ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
ax2.legend(fontsize=8, facecolor="#ffffff", edgecolor="#c5e8d8")

port_s   = pd.Series(portfolio)
bh_s     = pd.Series(bh)
port_dd  = ((port_s - port_s.cummax()) / port_s.cummax()) * 100
bh_dd    = ((bh_s   - bh_s.cummax())   / bh_s.cummax())   * 100

ax3.fill_between(data.index, port_dd, 0,
                 alpha=0.6, color="#0d5c3e", label="Strategy")
ax3.fill_between(data.index, bh_dd,   0,
                 alpha=0.3, color="#1a9e6c", label="Buy & Hold")
ax3.set_ylabel("Drawdown %", fontsize=9)
ax3.set_xlabel("")
ax3.legend(fontsize=8, facecolor="#ffffff", edgecolor="#c5e8d8")

plt.tight_layout()
st.pyplot(fig2)
plt.close()


# ======================================================================
# SECTION 12: PAPER ACCOUNT + DATA EXPLORER
# ======================================================================
left, right = st.columns([1, 2])

with left:
    st.markdown('<div class="section-label">Paper Account</div>',
                unsafe_allow_html=True)
    account = load_paper_account()

    acol1, acol2 = st.columns(2)
    acol1.metric("Cash",     f"${account['cash']:,.2f}")
    acol2.metric("Positions", len(account["positions"]))
    st.metric("Orders Placed", len(account["orders"]))

    if account["positions"]:
        st.markdown("**Open Positions**")
        pos_df = pd.DataFrame([
            {"Ticker": k,
             "Shares": v["qty"],
             "Avg Entry": f"${v['avg_price']:.2f}"}
            for k, v in account["positions"].items()
        ])
        st.dataframe(pos_df, use_container_width=True, hide_index=True)

with right:
    st.markdown('<div class="section-label">Recent Orders</div>',
                unsafe_allow_html=True)
    if account["orders"]:
        orders_df = pd.DataFrame(account["orders"]).tail(8)
        orders_df["price"] = orders_df["price"].apply(lambda x: f"${x:.2f}")
        st.dataframe(orders_df[["id","timestamp","ticker",
                                 "side","qty","price"]],
                     use_container_width=True, hide_index=True)
    else:
        st.info("No orders placed yet.")


# ======================================================================
# SECTION 13: RAW DATA EXPLORER
# ======================================================================
st.markdown("---")
if st.checkbox("Show raw OHLCV data (last 20 trading days)"):
    st.markdown('<div class="section-label">Raw Data</div>',
                unsafe_allow_html=True)
    st.dataframe(
        data[["Open","High","Low","Close","Volume","sma","atr"]]
        .tail(20).round(2),
        use_container_width=True
    )

st.markdown("""
<div style='text-align:center; font-family: Space Mono, monospace;
            font-size:0.65rem; color:#7ecaaa; margin-top:32px;
            padding-top:16px; border-top:1px solid #c5e8d8;'>
    DRIFT LABS v0.1 &nbsp;·&nbsp; FOR RESEARCH PURPOSES ONLY
    &nbsp;·&nbsp; NOT FINANCIAL ADVICE
</div>
""", unsafe_allow_html=True)