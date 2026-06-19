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
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(__file__))

from src.data_handler import get_price_data
from src.indicators import add_sma, add_atr
from src.risk import calculate_shares
from src.features import build_features, feature_cols


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
# SECTION 3: CUSTOM CSS
# ======================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500&family=Space+Mono:wght@400;700&display=swap');

.stApp {
    background-color: #f0f7f4;
    font-family: 'Inter', sans-serif;
    color: #2d3748;
}
[data-testid="stSidebar"] {
    background-color: #0d5c3e !important;
}
[data-testid="stSidebar"] * { color: #e8f5f0 !important; }
[data-testid="stSidebar"] .stTextInput input {
    background-color: #1a9e6c22 !important;
    border: 1px solid #1a9e6c !important;
    color: #e8f5f0 !important;
    font-family: 'Space Mono', monospace !important;
    border-radius: 6px;
}
[data-testid="stSidebar"] hr { border-color: #1a9e6c44 !important; }
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
hr {
    border: none;
    border-top: 1px solid #c5e8d8;
    margin: 8px 0 20px 0;
}
.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    color: #1a9e6c;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 4px;
}
[data-testid="stDataFrame"] {
    border: 1px solid #c5e8d8;
    border-radius: 8px;
    overflow: hidden;
}
/* Tab styling */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: #e8f5f0;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    color: #4a7c6a;
    border-radius: 6px;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background-color: #0d5c3e !important;
    color: #ffffff !important;
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
    return ((values - rolling_peak) / rolling_peak).min()


def cagr(portfolio_values, start, end):
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    if years <= 0:
        return 0.0
    return (portfolio_values[-1] / portfolio_values[0]) ** (1 / years) - 1


def win_rate(portfolio_values):
    values = pd.Series(portfolio_values)
    daily_returns = values.pct_change().dropna()
    return (daily_returns > 0).mean()


def drift_style(ax, fig):
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


def load_paper_account():
    path = "data/paper_account.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"cash": 100000, "positions": {}, "orders": []}


# ======================================================================
# SECTION 5: BACKTEST FUNCTIONS (CACHED)
# ======================================================================
@st.cache_data
def run_sma_backtest(ticker, start, end, sma_window=20):
    data = get_price_data(ticker, start, end)
    if data is None or data.empty:
        return None, None, None
    data = add_sma(data, window=sma_window)
    data = add_atr(data)
    data["signal"] = (data["Close"] > data["sma"]).shift(1)

    cash = 10000
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
                cash -= shares * price_yesterday * 1.001
                shares_held = shares
        elif signal_today == False and shares_held > 0:
            cash += shares_held * price_yesterday * 0.999
            shares_held = 0

        portfolio.append(cash + shares_held * price_today)

    bh_shares = 10000 // data["Close"].iloc[0]
    bh = (bh_shares * data["Close"]).tolist()
    return data, portfolio, bh


@st.cache_data
def run_ml_backtest(ticker, start, end):
    data = get_price_data(ticker, start, end)
    if data is None or data.empty:
        return None, None
    data = add_atr(data)
    df   = build_features(data)
    df   = df.dropna(subset=feature_cols() + ["label"])

    if len(df) < 600:
        return None, None

    min_train = 500
    signals   = []

    for i in range(min_train, len(df) - 1):
        train = df.iloc[:i]
        model = RandomForestClassifier(
            n_estimators=50, max_depth=4,
            min_samples_leaf=20, random_state=42
        )
        model.fit(train[feature_cols()], train["label"])
        prob = model.predict_proba(df[feature_cols()].iloc[[i]])[0][1]
        signals.append({
            "date":   df.index[i],
            "signal": 1 if prob >= 0.55 else 0
        })

    signals_df = pd.DataFrame(signals).set_index("date")
    bt_data    = df.loc[signals_df.index]

    cash = 10000
    shares_held = 0
    portfolio   = []

    for i in range(len(bt_data)):
        price_today     = bt_data["Close"].iloc[i]
        price_yesterday = bt_data["Close"].iloc[i-1] if i > 0 else price_today
        atr_today       = bt_data["atr"].iloc[i]
        signal_today    = signals_df["signal"].iloc[i]

        if signal_today == 1 and shares_held == 0:
            shares = calculate_shares(cash, price_yesterday, atr_today)
            if shares > 0:
                cash -= shares * price_yesterday * 1.001
                shares_held = shares
        elif signal_today == 0 and shares_held > 0:
            cash += shares_held * price_yesterday * 0.999
            shares_held = 0

        portfolio.append(cash + shares_held * price_today)

    return bt_data.index, portfolio


@st.cache_data
def run_multi_ticker(tickers, start, end, sma_window):
    results = []
    for t in tickers:
        data, portfolio, bh = run_sma_backtest(t, start, end, sma_window)
        if portfolio is None:
            continue
        results.append({
            "Ticker":       t,
            "Return":       (portfolio[-1] / 10000) - 1,
            "CAGR":         cagr(portfolio, start, end),
            "Sharpe":       sharpe_ratio(portfolio),
            "Max DD":       max_drawdown(portfolio),
            "Win Rate":     win_rate(portfolio),
            "B&H Return":   (bh[-1] / 10000) - 1,
            "Beat B&H":     "✓" if portfolio[-1] > bh[-1] else "✗",
            "portfolio":    portfolio,
        })
    return results


# ======================================================================
# SECTION 6: SIDEBAR
# ======================================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:12px 0 20px 0;'>
        <div style='font-family:Space Grotesk,sans-serif; font-size:1.5rem;
                    font-weight:700; color:#e8f5f0; letter-spacing:-0.5px;'>
            📡 Drift Labs
        </div>
        <div style='font-size:0.72rem; color:#7ecaaa;
                    font-family:Space Mono,monospace; margin-top:2px;
                    letter-spacing:0.1em;'>
            QUANT RESEARCH PLATFORM v0.2
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**STRATEGY CONTROLS**")
    ticker     = st.text_input("Ticker Symbol", value="AAPL").upper()
    start_date = st.date_input("Start Date",
                  value=pd.Timestamp("2021-01-01")).strftime("%Y-%m-%d")
    end_date   = st.date_input("End Date",
                  value=pd.Timestamp("2026-06-01")).strftime("%Y-%m-%d")
    sma_window = st.slider("SMA Window", 5, 100, 20)

    st.markdown("---")
    st.markdown("**MULTI-TICKER WATCHLIST**")
    default_tickers = "AAPL,MSFT,NVDA,JPM,XOM,TSLA,SPY"
    ticker_input = st.text_area("Tickers (comma-separated)",
                                 value=default_tickers, height=80)
    watchlist = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#7ecaaa;
                font-family:Space Mono,monospace; line-height:1.8;'>
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
    <span style='font-family:Space Mono,monospace; font-size:0.75rem;
                 color:#1a9e6c; background:#e8f5f0; padding:3px 8px;
                 border-radius:4px; border:1px solid #c5e8d8;'>
        PAPER TRADING
    </span>
    <span style='font-family:Space Mono,monospace; font-size:0.75rem;
                 color:#4a7c6a; background:#e8f5f0; padding:3px 8px;
                 border-radius:4px; border:1px solid #c5e8d8;'>
        v0.2
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style='font-family:Space Mono,monospace; font-size:0.8rem;
            color:#4a7c6a; margin-bottom:16px;'>
    {ticker} &nbsp;·&nbsp; {start_date} → {end_date}
    &nbsp;·&nbsp; SMA({sma_window})
</div>
""", unsafe_allow_html=True)

st.markdown("---")


# ======================================================================
# SECTION 8: TABS
# ======================================================================
tab1, tab2, tab3 = st.tabs([
    "📊  Single Ticker",
    "🌐  Multi-Ticker",
    "🤖  ML vs SMA"
])


# ======================================================================
# TAB 1: SINGLE TICKER DEEP DIVE
# ======================================================================
with tab1:
    with st.spinner(f"Loading {ticker}..."):
        data, portfolio, bh = run_sma_backtest(ticker, start_date,
                                                end_date, sma_window)

    if data is None:
        st.error(f"Could not load data for **{ticker}**.")
    else:
        latest_price  = data["Close"].iloc[-1]
        final_value   = portfolio[-1]
        total_return  = (final_value / 10000) - 1
        sharpe        = sharpe_ratio(portfolio)
        mdd           = max_drawdown(portfolio)
        bh_return     = (bh[-1] / 10000) - 1
        strategy_cagr = cagr(portfolio, start_date, end_date)

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Latest Price",    f"${latest_price:.2f}")
        c2.metric("Final Value",     f"${final_value:,.2f}")
        c3.metric("Total Return",    f"{total_return:+.2%}",
                  delta=f"{(total_return-bh_return):+.2%} vs B&H")
        c4.metric("CAGR",            f"{strategy_cagr:+.2%}")
        c5.metric("Sharpe Ratio",    f"{sharpe:.3f}")
        c6.metric("Max Drawdown",    f"{mdd:.2%}")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Price & SMA Signal</div>',
                    unsafe_allow_html=True)

        fig1, ax1 = plt.subplots(figsize=(13, 3.5))
        drift_style(ax1, fig1)
        ax1.plot(data.index, data["Close"],
                 color="#0d5c3e", linewidth=1.2, label="Close")
        ax1.plot(data.index, data["sma"],
                 color="#1a9e6c", linewidth=1,
                 linestyle="--", label=f"SMA({sma_window})")
        signal = data["signal"].fillna(False).astype(bool)
        ax1.fill_between(data.index,
                         data["Close"].min(), data["Close"].max(),
                         where=signal, alpha=0.06, color="#1a9e6c",
                         label="In Market")
        ax1.yaxis.set_major_formatter(
            mticker.StrMethodFormatter("${x:,.0f}"))
        ax1.legend(fontsize=8, facecolor="#ffffff", edgecolor="#c5e8d8")
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

        st.markdown('<div class="section-label">Performance vs Benchmark</div>',
                    unsafe_allow_html=True)
        fig2, (ax2, ax3) = plt.subplots(
            2, 1, figsize=(13, 6), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]})
        drift_style(ax2, fig2)
        drift_style(ax3, fig2)
        ax2.plot(data.index, portfolio,
                 color="#0d5c3e", linewidth=1.5, label="SMA Strategy")
        ax2.plot(data.index, bh,
                 color="#1a9e6c", linewidth=1.2,
                 linestyle="--", alpha=0.7, label="Buy & Hold")
        ax2.yaxis.set_major_formatter(
            mticker.StrMethodFormatter("${x:,.0f}"))
        ax2.set_ylabel("Portfolio Value (USD)", fontsize=9)
        ax2.legend(fontsize=8, facecolor="#ffffff", edgecolor="#c5e8d8")

        ps = pd.Series(portfolio)
        bs = pd.Series(bh)
        ax3.fill_between(data.index,
                         ((ps-ps.cummax())/ps.cummax())*100, 0,
                         alpha=0.6, color="#0d5c3e", label="Strategy DD")
        ax3.fill_between(data.index,
                         ((bs-bs.cummax())/bs.cummax())*100, 0,
                         alpha=0.3, color="#1a9e6c", label="B&H DD")
        ax3.set_ylabel("Drawdown %", fontsize=9)
        ax3.legend(fontsize=8, facecolor="#ffffff", edgecolor="#c5e8d8")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # Paper account
        st.markdown("---")
        st.markdown('<div class="section-label">Paper Account</div>',
                    unsafe_allow_html=True)
        account = load_paper_account()
        a1, a2, a3 = st.columns(3)
        a1.metric("Cash",          f"${account['cash']:,.2f}")
        a2.metric("Open Positions", len(account["positions"]))
        a3.metric("Total Orders",   len(account["orders"]))

        if account["positions"]:
            pos_df = pd.DataFrame([
                {"Ticker": k, "Shares": v["qty"],
                 "Avg Entry": f"${v['avg_price']:.2f}"}
                for k, v in account["positions"].items()
            ])
            st.dataframe(pos_df, use_container_width=True,
                         hide_index=True)

        if st.checkbox("Show raw OHLCV data"):
            st.dataframe(
                data[["Open","High","Low","Close",
                       "Volume","sma","atr"]].tail(20).round(2),
                use_container_width=True)


# ======================================================================
# TAB 2: MULTI-TICKER COMPARISON
# ======================================================================
with tab2:
    st.markdown('<div class="section-label">Watchlist Backtest Results</div>',
                unsafe_allow_html=True)

    with st.spinner("Running multi-ticker backtest..."):
        multi_results = run_multi_ticker(
            tuple(watchlist), start_date, end_date, sma_window)

    if not multi_results:
        st.error("No results — check your watchlist tickers.")
    else:
        display_df = pd.DataFrame([{
            "Ticker":     r["Ticker"],
            "Return":     f"{r['Return']:+.2%}",
            "CAGR":       f"{r['CAGR']:+.2%}",
            "Sharpe":     f"{r['Sharpe']:.3f}",
            "Max DD":     f"{r['Max DD']:.2%}",
            "Win Rate":   f"{r['Win Rate']:.1%}",
            "B&H Return": f"{r['B&H Return']:+.2%}",
            "Beat B&H":   r["Beat B&H"]
        } for r in multi_results])

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        beat = sum(1 for r in multi_results if r["Return"] > r["B&H Return"])
        avg_sharpe = np.mean([r["Sharpe"] for r in multi_results])

        m1, m2, m3 = st.columns(3)
        m1.metric("Beat Buy & Hold",  f"{beat}/{len(multi_results)}")
        m2.metric("Avg Sharpe",        f"{avg_sharpe:.3f}")
        m3.metric("Tickers Analyzed",  len(multi_results))

        st.markdown('<div class="section-label">Normalized Portfolio Curves</div>',
                    unsafe_allow_html=True)

        greens = ["#0d5c3e","#1a9e6c","#2d8a5e","#4aab7e",
                  "#6cc49e","#0a4530","#3d7a60","#8ed4b4",
                  "#a8dfc8","#c5e8d8"]

        fig3, ax = plt.subplots(figsize=(13, 5))
        drift_style(ax, fig3)

        for idx, r in enumerate(multi_results):
            norm = [v/10000 for v in r["portfolio"]]
            ax.plot(norm, color=greens[idx % len(greens)],
                    linewidth=1.5, label=r["Ticker"], alpha=0.85)

        ax.axhline(y=1.0, color="#c5e8d8",
                   linestyle="--", linewidth=1)
        ax.set_ylabel("Return (×)", fontsize=9)
        ax.set_xlabel("Trading Days", fontsize=9)
        ax.legend(fontsize=8, facecolor="#ffffff",
                  edgecolor="#c5e8d8", ncol=2)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()


# ======================================================================
# TAB 3: ML vs SMA HEAD-TO-HEAD
# ======================================================================
with tab3:
    st.markdown(f"""
    <div style='font-family:Space Mono,monospace; font-size:0.8rem;
                color:#4a7c6a; margin-bottom:12px;'>
        Walk-forward ML model vs SMA baseline · {ticker}
    </div>
    """, unsafe_allow_html=True)

    st.info("Walk-forward training fits ~800 models — takes 1-2 minutes. "
            "Results are cached after the first run.")

    if st.button("▶  Run ML Backtest", type="primary"):
        with st.spinner("Running walk-forward ML training... (~800 iterations)"):
            ml_dates, ml_portfolio = run_ml_backtest(ticker, start_date, end_date)

        if ml_portfolio is None:
            st.error("Not enough data for ML backtest on this ticker/period.")
        else:
            _, sma_portfolio, _ = run_sma_backtest(
                ticker, start_date, end_date, sma_window)

            # Align to ML period
            sma_aligned = sma_portfolio[-len(ml_portfolio):]

            ml_ret   = (ml_portfolio[-1]  / 10000) - 1
            sma_ret  = (sma_aligned[-1]   / 10000) - 1
            ml_sr    = sharpe_ratio(ml_portfolio)
            sma_sr   = sharpe_ratio(sma_aligned)
            ml_mdd   = max_drawdown(ml_portfolio)
            sma_mdd  = max_drawdown(sma_aligned)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🤖 ML Strategy**")
                st.metric("Return",      f"{ml_ret:+.2%}")
                st.metric("Sharpe",      f"{ml_sr:.3f}")
                st.metric("Max Drawdown",f"{ml_mdd:.2%}")
            with c2:
                st.markdown(f"**📈 SMA({sma_window}) Strategy**")
                st.metric("Return",      f"{sma_ret:+.2%}",
                          delta=f"{ml_ret - sma_ret:+.2%} ML edge")
                st.metric("Sharpe",      f"{sma_sr:.3f}",
                          delta=f"{ml_sr - sma_sr:+.3f} ML edge")
                st.metric("Max Drawdown",f"{sma_mdd:.2%}",
                          delta=f"{ml_mdd - sma_mdd:+.2%} ML edge")

            st.markdown('<div class="section-label">Equity Curves</div>',
                        unsafe_allow_html=True)

            fig4, ax4 = plt.subplots(figsize=(13, 4))
            drift_style(ax4, fig4)
            ax4.plot(ml_portfolio,  color="#0d5c3e",
                     linewidth=1.5, label="ML Strategy")
            ax4.plot(sma_aligned,   color="#1a9e6c",
                     linewidth=1.2, linestyle="--",
                     alpha=0.7, label=f"SMA({sma_window})")
            ax4.yaxis.set_major_formatter(
                mticker.StrMethodFormatter("${x:,.0f}"))
            ax4.set_ylabel("Portfolio Value (USD)", fontsize=9)
            ax4.legend(fontsize=8,
                       facecolor="#ffffff", edgecolor="#c5e8d8")
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()
    else:
        st.markdown("""
        <div style='text-align:center; padding:60px;
                    color:#4a7c6a; font-family:Space Mono,monospace;
                    font-size:0.85rem; background:#ffffff;
                    border:1px solid #c5e8d8; border-radius:10px;'>
            Click the button above to run the ML backtest.<br>
            Results are cached after the first run.
        </div>
        """, unsafe_allow_html=True)


# ======================================================================
# SECTION 9: FOOTER
# ======================================================================
st.markdown("""
<div style='text-align:center; font-family:Space Mono,monospace;
            font-size:0.65rem; color:#7ecaaa; margin-top:32px;
            padding-top:16px; border-top:1px solid #c5e8d8;'>
    DRIFT LABS v0.2 &nbsp;·&nbsp; FOR RESEARCH PURPOSES ONLY
    &nbsp;·&nbsp; NOT FINANCIAL ADVICE
</div>
""", unsafe_allow_html=True)