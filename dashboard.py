# ======================================================================
# AURELINE LABS — DASHBOARD v2.0
# Quantitative Research & Intelligence Platform
# Ateneo de Manila University · Applied Mathematics · Math Finance
# ======================================================================

# SECTION 1: IMPORTS
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
import os
import sys
import glob
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from src.data_handler import get_price_data
from src.indicators import add_sma, add_atr
from src.risk import calculate_shares
from src.features import build_all_features as build_features
from src.features import all_feature_cols as feature_cols


# ======================================================================
# SECTION 2: PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="Aureline Labs",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ======================================================================
# SECTION 3: VISUAL IDENTITY — CSS
# ======================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
.stApp { background-color:#060d1f; font-family:'DM Sans',sans-serif; color:#e8f0fe; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background-color:#0d1b35 !important; border-right:1px solid #1a3357; }
[data-testid="stSidebar"] * { color:#e8f0fe !important; }
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stTextArea textarea {
    background-color:#060d1f !important; border:1px solid #1a3357 !important;
    color:#e8f0fe !important; font-family:'JetBrains Mono',monospace !important;
    font-size:0.8rem !important; border-radius:4px;
}
[data-testid="stSidebar"] .stTextInput input:focus { border-color:#00d4aa !important; }
[data-testid="stSidebar"] hr { border-color:#1a3357 !important; }

/* ── Typography ── */
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; font-weight:700 !important; color:#e8f0fe !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background-color:#0f2040; border:1px solid #1a3357;
    border-top:2px solid #00d4aa; border-radius:6px; padding:16px 20px 12px 20px;
}
[data-testid="stMetricLabel"] {
    font-family:'JetBrains Mono',monospace !important; font-size:0.62rem !important;
    color:#7b9bc0 !important; text-transform:uppercase !important; letter-spacing:0.12em !important;
}
[data-testid="stMetricValue"] {
    font-family:'JetBrains Mono',monospace !important; font-size:1.3rem !important;
    font-weight:700 !important; color:#00d4aa !important;
}
[data-testid="stMetricDelta"] { font-family:'JetBrains Mono',monospace !important; font-size:0.72rem !important; }

/* ── Section labels — signature element ── */
.al-section {
    font-family:'JetBrains Mono',monospace; font-size:0.62rem; font-weight:500;
    color:#00d4aa; text-transform:uppercase; letter-spacing:0.2em;
    padding-bottom:6px; border-bottom:1px solid #00d4aa;
    margin-bottom:14px; display:block;
}

/* ── Divider ── */
hr { border:none !important; border-top:1px solid #1a3357 !important; margin:16px 0 !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color:#0d1b35; border:1px solid #1a3357;
    border-radius:6px; padding:4px; gap:2px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family:'JetBrains Mono',monospace; font-size:0.72rem;
    font-weight:500; color:#7b9bc0; border-radius:4px;
}
[data-testid="stTabs"] [aria-selected="true"] { background-color:#00d4aa !important; color:#060d1f !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border:1px solid #1a3357 !important; border-radius:6px; }

/* ── Buttons ── */
.stButton button {
    background-color:#00d4aa !important; color:#060d1f !important; border:none !important;
    font-family:'JetBrains Mono',monospace !important; font-weight:700 !important;
    font-size:0.78rem !important; letter-spacing:0.05em !important;
    border-radius:4px !important; padding:10px 24px !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    background-color:#0f2040 !important; border:1px solid #1a3357 !important;
    border-left:3px solid #00d4aa !important; border-radius:4px !important;
    color:#7b9bc0 !important; font-family:'JetBrains Mono',monospace !important; font-size:0.78rem !important;
}

/* ── Radio (mode toggle) ── */
[data-testid="stRadio"] label p { font-family:'JetBrains Mono',monospace !important; font-size:0.75rem !important; }

/* ── Beginner card ── */
.bcard {
    background:#0f2040; border:1px solid #1a3357; border-radius:8px;
    padding:18px; margin-bottom:10px; font-family:'DM Sans',sans-serif;
}
.bcard-label { font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:#7b9bc0; text-transform:uppercase; letter-spacing:0.1em; }
.bcard-value { font-family:'Space Grotesk',sans-serif; font-size:1.6rem; font-weight:700; color:#00d4aa; margin:4px 0; }
.bcard-desc { font-size:0.8rem; color:#7b9bc0; line-height:1.5; }

/* ── News headline row ── */
.news-row {
    display:flex; align-items:center; gap:10px; padding:8px 12px;
    background:#0f2040; border:1px solid #1a3357; border-radius:4px;
    margin-bottom:4px;
}
</style>
""", unsafe_allow_html=True)


# ======================================================================
# SECTION 4: HELPERS
# ======================================================================
def section(label):
    st.markdown(f'<span class="al-section">{label}</span>', unsafe_allow_html=True)


def aureline_chart_style(ax, fig):
    fig.patch.set_facecolor("#0f2040")
    ax.set_facecolor("#060d1f")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1a3357")
    ax.spines["bottom"].set_color("#1a3357")
    ax.tick_params(colors="#7b9bc0", labelsize=8)
    ax.yaxis.label.set_color("#7b9bc0")
    ax.xaxis.label.set_color("#7b9bc0")
    ax.grid(True, color="#0d1b35", linewidth=0.8, linestyle="--")


# ── Beginner translation layer ──
def plain_signal(signal):
    return {
        "BUY":          "✅ Worth considering",
        "HOLD":         "⏸️ Wait and watch",
        "AVOID":        "⛔ Too risky right now",
        "STRONG BUY":   "🟢 Strong opportunity",
        "STRONG AVOID": "🔴 Avoid — high risk",
        "WATCH ▲":      "👀 Positive news, mixed price signal",
        "WATCH ▼":      "👀 Negative news, watch carefully",
    }.get(signal, signal)


def plain_regime(regime):
    return {
        "BULL_TRENDING":   "📈 Rising steadily",
        "BEAR_TRENDING":   "📉 Falling trend",
        "HIGH_VOLATILITY": "⚠️ Very choppy — unpredictable",
        "SIDEWAYS":        "➡️ No clear direction",
    }.get(regime, regime)


def plain_sharpe(s):
    if s >= 2.0:   return f"Sharpe {s:.2f} — Excellent (very consistent returns)"
    elif s >= 1.0: return f"Sharpe {s:.2f} — Good risk-adjusted performance"
    elif s >= 0.5: return f"Sharpe {s:.2f} — Acceptable, some ups and downs"
    elif s >= 0.0: return f"Sharpe {s:.2f} — Weak"
    else:          return f"Sharpe {s:.2f} — Negative (losing on risk-adjusted basis)"


def plain_drawdown(mdd_pct):
    p = abs(mdd_pct)
    if p < 5:    return f"Worst decline: {mdd_pct:.1f}% — Very low risk"
    elif p < 15: return f"Worst decline: {mdd_pct:.1f}% — Moderate risk"
    elif p < 30: return f"Worst decline: {mdd_pct:.1f}% — High risk"
    else:        return f"Worst decline: {mdd_pct:.1f}% — Very high risk"


def plain_return(ret_pct):
    if ret_pct > 100:  return "More than doubled", "Great long-term growth"
    elif ret_pct > 50: return f"+{ret_pct:.1f}%", "Strong gains"
    elif ret_pct > 20: return f"+{ret_pct:.1f}%", "Good gains"
    elif ret_pct > 0:  return f"+{ret_pct:.1f}%", "Small gain"
    elif ret_pct > -20:return f"{ret_pct:.1f}%", "Small loss"
    else:               return f"{ret_pct:.1f}%", "Significant loss"


# ── Metric helpers ──
def sharpe_ratio(pv, rfr=0.05):
    s  = pd.Series(pv)
    dr = s.pct_change().dropna()
    ex = dr - rfr/252
    return round((ex.mean()/ex.std())*np.sqrt(252), 3) if ex.std() > 0 else 0.0


def max_drawdown(pv):
    s = pd.Series(pv)
    return ((s - s.cummax()) / s.cummax()).min()


def cagr(pv, start, end):
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    return (pv[-1]/pv[0])**(1/years) - 1 if years > 0 else 0.0


def win_rate(pv):
    return (pd.Series(pv).pct_change().dropna() > 0).mean()


def load_paper_account():
    p = "data/paper_account.json"
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"cash": 100000, "positions": {}, "orders": []}


# ======================================================================
# SECTION 5: CACHED DATA + BACKTEST
# ======================================================================
@st.cache_data
def run_sma_backtest(ticker, start, end, sma_window=20):
    data = get_price_data(ticker, start, end)
    if data is None or data.empty:
        return None, None, None
    data = add_sma(data, window=sma_window)
    data = add_atr(data)
    data["signal"] = (data["Close"] > data["sma"]).shift(1)
    cash = 10000; shares = 0; portfolio = []
    for i in range(len(data)):
        pt = data["Close"].iloc[i]
        py = data["Close"].iloc[i-1] if i > 0 else pt
        sig = data["signal"].iloc[i]
        atr = data["atr"].iloc[i]
        if sig and shares == 0:
            sh = calculate_shares(cash, py, atr)
            if sh > 0:
                cash -= sh * py * 1.001; shares = sh
        elif not sig and shares > 0:
            cash += shares * py * 0.999; shares = 0
        portfolio.append(cash + shares * pt)
    bh = (10000 // data["Close"].iloc[0] * data["Close"]).tolist()
    return data, portfolio, bh


@st.cache_data
def run_multi_ticker(tickers, start, end, sma_window):
    results = []
    for t in tickers:
        data, portfolio, bh = run_sma_backtest(t, start, end, sma_window)
        if portfolio is None:
            continue
        results.append({
            "Ticker":     t,
            "Return":     (portfolio[-1]/10000) - 1,
            "CAGR":       cagr(portfolio, start, end),
            "Sharpe":     sharpe_ratio(portfolio),
            "Max DD":     max_drawdown(portfolio),
            "Win Rate":   win_rate(portfolio),
            "B&H Return": (bh[-1]/10000) - 1,
            "Beat B&H":   "✓" if portfolio[-1] > bh[-1] else "✗",
            "portfolio":  portfolio,
        })
    return results


@st.cache_data
def run_ml_backtest(ticker, start, end):
    from sklearn.ensemble import RandomForestClassifier
    data = get_price_data(ticker, start, end)
    if data is None or data.empty:
        return None, None
    data = add_atr(data)
    df   = build_features(data)
    cols = feature_cols()
    df["label"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna(subset=cols + ["label"])
    if len(df) < 600:
        return None, None
    signals = []
    for i in range(500, len(df)-1):
        tr = df.iloc[:i]
        m  = RandomForestClassifier(n_estimators=50, max_depth=4, min_samples_leaf=20, random_state=42)
        m.fit(tr[cols], tr["label"])
        p = m.predict_proba(df[cols].iloc[[i]])[0][1]
        signals.append({"date": df.index[i], "signal": 1 if p >= 0.55 else 0})
    sdf = pd.DataFrame(signals).set_index("date")
    btd = df.loc[sdf.index]; data2 = add_atr(data).loc[sdf.index]
    cash = 10000; sh = 0; pv = []
    for i in range(len(btd)):
        pt = btd["Close"].iloc[i]; py = btd["Close"].iloc[i-1] if i > 0 else pt
        atr = data2["atr"].iloc[i]; sig = sdf["signal"].iloc[i]
        if sig and sh == 0:
            s = calculate_shares(cash, py, atr)
            if s > 0: cash -= s*py*1.001; sh = s
        elif not sig and sh > 0:
            cash += sh*py*0.999; sh = 0
        pv.append(cash + sh*pt)
    return btd.index, pv


# ======================================================================
# SECTION 6: SIDEBAR
# ======================================================================
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 0 24px 0;'>
        <div style='font-family:Space Grotesk,sans-serif; font-size:1.25rem;
                    font-weight:700; color:#e8f0fe; letter-spacing:-0.5px;'>
            ⬡ Aureline Labs
        </div>
        <div style='font-family:JetBrains Mono,monospace; font-size:0.58rem;
                    color:#00d4aa; letter-spacing:0.2em; margin-top:5px;'>
            QUANT RESEARCH PLATFORM v2.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Mode toggle — first class
    st.markdown("""
    <div style='font-family:JetBrains Mono,monospace; font-size:0.6rem;
                color:#7b9bc0; letter-spacing:0.15em; margin-bottom:6px;'>
        INTERFACE MODE
    </div>
    """, unsafe_allow_html=True)
    mode        = st.radio("Mode", ["Professional", "Beginner"],
                           horizontal=True, label_visibility="collapsed")
    is_beginner = (mode == "Beginner")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:JetBrains Mono,monospace; font-size:0.6rem;
                color:#7b9bc0; letter-spacing:0.15em; margin-bottom:8px;'>
        STRATEGY CONTROLS
    </div>""", unsafe_allow_html=True)

    ticker     = st.text_input("Ticker", value="AAPL").upper()
    start_date = st.date_input("Start", value=pd.Timestamp("2021-01-01")).strftime("%Y-%m-%d")
    end_date   = st.date_input("End",   value=pd.Timestamp("2026-06-01")).strftime("%Y-%m-%d")
    sma_window = st.slider("SMA Window", 5, 100, 20)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:JetBrains Mono,monospace; font-size:0.6rem;
                color:#7b9bc0; letter-spacing:0.15em; margin-bottom:8px;'>
        WATCHLIST
    </div>""", unsafe_allow_html=True)
    ticker_input = st.text_area("Tickers (comma-separated)",
                                 value="AAPL,MSFT,NVDA,JPM,XOM,TSLA,SPY",
                                 height=68)
    watchlist = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-family:JetBrains Mono,monospace; font-size:0.62rem;
                color:#1a3357; line-height:2;'>
        <span style='color:#00d4aa;'>RESEARCHER</span><br>
        Louis Andre<br>
        <span style='color:#00d4aa;'>INSTITUTION</span><br>
        Ateneo de Manila University<br>
        <span style='color:#00d4aa;'>MODE</span><br>
        {'Beginner — plain English' if is_beginner else 'Professional — full quant metrics'}
    </div>""", unsafe_allow_html=True)


# ======================================================================
# SECTION 7: HEADER
# ======================================================================
hcol1, hcol2 = st.columns([3, 1])
with hcol1:
    st.markdown(f"""
    <div style='padding-top:8px;'>
        <div style='font-family:Space Grotesk,sans-serif; font-size:2.2rem;
                    font-weight:700; color:#e8f0fe; letter-spacing:-1.5px;
                    line-height:1;'>Aureline Labs</div>
        <div style='font-family:JetBrains Mono,monospace; font-size:0.72rem;
                    color:#7b9bc0; margin-top:5px;'>
            {'Financial Intelligence — Simplified · For the everyday investor' if is_beginner
             else 'Quantitative Research & Intelligence Platform · Applied Mathematics'}
        </div>
    </div>""", unsafe_allow_html=True)
with hcol2:
    st.markdown(f"""
    <div style='display:flex; gap:6px; justify-content:flex-end;
                padding-top:14px; flex-wrap:wrap;'>
        <span style='font-family:JetBrains Mono,monospace; font-size:0.58rem;
                     font-weight:700; color:#060d1f; background:#00d4aa;
                     padding:3px 8px; border-radius:3px; letter-spacing:0.1em;'>
            PAPER TRADING
        </span>
        <span style='font-family:JetBrains Mono,monospace; font-size:0.58rem;
                     color:#7b9bc0; border:1px solid #1a3357;
                     padding:3px 8px; border-radius:3px; letter-spacing:0.1em;'>
            v2.0
        </span>
    </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div style='font-family:JetBrains Mono,monospace; font-size:0.72rem;
            color:#1a3357; margin:8px 0 0 0; padding-bottom:16px;
            border-bottom:1px solid #1a3357;'>
    <span style='color:#7b9bc0;'>{ticker}</span>
    &nbsp;·&nbsp; {start_date} → {end_date}
    &nbsp;·&nbsp; SMA({sma_window})
    &nbsp;·&nbsp; <span style='color:#00d4aa;'>{mode} Mode</span>
</div>""", unsafe_allow_html=True)
st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)


# ======================================================================
# SECTION 8: TABS
# ======================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "OVERVIEW",
    "RESEARCH",
    "ML vs SMA",
    "🧠 INTELLIGENCE"
])


# ======================================================================
# TAB 1: OVERVIEW (replaces "Single Ticker")
# ======================================================================
with tab1:
    with st.spinner(f"Loading {ticker}..."):
        data, portfolio, bh = run_sma_backtest(ticker, start_date, end_date, sma_window)

    if data is None:
        st.error(f"No data found for **{ticker}**. Check the ticker symbol.")
    else:
        latest  = data["Close"].iloc[-1]
        final   = portfolio[-1]
        ret     = (final/10000) - 1
        sh      = sharpe_ratio(portfolio)
        mdd     = max_drawdown(portfolio)
        bh_ret  = (bh[-1]/10000) - 1
        _cagr   = cagr(portfolio, start_date, end_date)
        mdd_pct = mdd * 100

        # ── Metrics row ──
        if is_beginner:
            ret_val, ret_desc = plain_return(ret * 100)
            risk_label = ("Low" if abs(mdd_pct) < 15
                          else "Medium" if abs(mdd_pct) < 30 else "High")
            risk_color = ("#00d4aa" if risk_label == "Low"
                          else "#ffd166" if risk_label == "Medium" else "#ff4d6a")

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"""
            <div class='bcard'>
                <div class='bcard-label'>Current Price</div>
                <div class='bcard-value'>${latest:.2f}</div>
                <div class='bcard-desc'>Latest closing price of {ticker}</div>
            </div>""", unsafe_allow_html=True)
            c2.markdown(f"""
            <div class='bcard'>
                <div class='bcard-label'>Your Strategy Result</div>
                <div class='bcard-value' style='color:{"#00d4aa" if ret > 0 else "#ff4d6a"};'>{ret_val}</div>
                <div class='bcard-desc'>{ret_desc} on ₱10,000 invested → ₱{final:,.0f}</div>
            </div>""", unsafe_allow_html=True)
            c3.markdown(f"""
            <div class='bcard'>
                <div class='bcard-label'>Risk Level</div>
                <div class='bcard-value' style='color:{risk_color};'>{risk_label}</div>
                <div class='bcard-desc'>{plain_drawdown(mdd_pct)}</div>
            </div>""", unsafe_allow_html=True)

            # Plain English summary box
            st.markdown(f"""
            <div style='background:#0f2040; border:1px solid #1a3357;
                        border-left:3px solid #00d4aa; border-radius:6px;
                        padding:16px 20px; margin:12px 0;
                        font-family:DM Sans,sans-serif;'>
                <div style='font-size:0.68rem; color:#00d4aa; font-weight:600;
                            letter-spacing:0.1em; margin-bottom:8px;'>
                    PLAIN ENGLISH SUMMARY
                </div>
                <div style='font-size:0.88rem; color:#e8f0fe; line-height:1.7;'>
                    If you had put <strong>₱10,000</strong> into <strong>{ticker}</strong>
                    at the start of this period using this moving-average strategy,
                    you would now have <strong style='color:#00d4aa;'>₱{final:,.0f}</strong>.
                    Just buying and holding the stock the whole time would have given you
                    <strong>₱{bh[-1]:,.0f}</strong>.
                </div>
                <div style='font-size:0.78rem; color:#7b9bc0; margin-top:8px; line-height:1.6;'>
                    {plain_sharpe(sh)} · {plain_drawdown(mdd_pct)}
                </div>
            </div>""", unsafe_allow_html=True)

        else:
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("Latest Price",  f"${latest:.2f}")
            c2.metric("Final Value",   f"${final:,.2f}")
            c3.metric("Total Return",  f"{ret:+.2%}", delta=f"{(ret-bh_ret):+.2%} vs B&H")
            c4.metric("CAGR",          f"{_cagr:+.2%}")
            c5.metric("Sharpe Ratio",  f"{sh:.3f}")
            c6.metric("Max Drawdown",  f"{mdd:.2%}")

        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

        # ── Price chart ──
        section("PRICE HISTORY & SMA SIGNAL")
        fig1, ax1 = plt.subplots(figsize=(13, 3.5))
        aureline_chart_style(ax1, fig1)
        ax1.plot(data.index, data["Close"], color="#00d4aa", linewidth=1.3,
                 label="Close Price")
        ax1.plot(data.index, data["sma"], color="#1a6eff", linewidth=1,
                 linestyle="--", alpha=0.7, label=f"SMA({sma_window})")
        signal = data["signal"].fillna(False).astype(bool)
        ax1.fill_between(data.index, data["Close"].min(), data["Close"].max(),
                         where=signal, alpha=0.05, color="#00d4aa",
                         label="In Market")
        ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
        ax1.legend(fontsize=8, facecolor="#0f2040", edgecolor="#1a3357",
                   labelcolor="#7b9bc0")
        plt.tight_layout()
        st.pyplot(fig1); plt.close()

        # ── Performance chart ──
        section("STRATEGY vs BENCHMARK")
        fig2, (ax2, ax3) = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1]})
        aureline_chart_style(ax2, fig2); aureline_chart_style(ax3, fig2)
        ax2.plot(data.index, portfolio, color="#00d4aa", linewidth=1.5,
                 label="SMA Strategy")
        ax2.plot(data.index, bh, color="#1a6eff", linewidth=1.2,
                 linestyle="--", alpha=0.6, label="Buy & Hold")
        ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
        ax2.set_ylabel("Portfolio Value", fontsize=8, color="#7b9bc0")
        ax2.legend(fontsize=8, facecolor="#0f2040", edgecolor="#1a3357",
                   labelcolor="#7b9bc0")
        ps = pd.Series(portfolio); bs = pd.Series(bh)
        ax3.fill_between(data.index, ((ps-ps.cummax())/ps.cummax())*100, 0,
                         alpha=0.7, color="#00d4aa", label="Strategy DD")
        ax3.fill_between(data.index, ((bs-bs.cummax())/bs.cummax())*100, 0,
                         alpha=0.3, color="#1a6eff", label="B&H DD")
        ax3.set_ylabel("Drawdown %", fontsize=8, color="#7b9bc0")
        ax3.legend(fontsize=8, facecolor="#0f2040", edgecolor="#1a3357",
                   labelcolor="#7b9bc0")
        plt.tight_layout(); st.pyplot(fig2); plt.close()

        # ── Paper account ──
        st.markdown("<hr>", unsafe_allow_html=True)
        section("PAPER TRADING ACCOUNT")
        acct = load_paper_account()
        a1, a2, a3 = st.columns(3)
        a1.metric("Cash Available",  f"${acct['cash']:,.2f}")
        a2.metric("Open Positions",   len(acct["positions"]))
        a3.metric("Orders Placed",    len(acct["orders"]))
        if acct["positions"]:
            pos_df = pd.DataFrame([
                {"Ticker": k, "Shares": v["qty"],
                 "Avg Entry": f"${v['avg_price']:.2f}"}
                for k, v in acct["positions"].items()
            ])
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
        if st.checkbox("Show raw OHLCV data"):
            st.dataframe(data[["Open","High","Low","Close","Volume","sma","atr"]]
                         .tail(20).round(2), use_container_width=True)


# ======================================================================
# TAB 2: RESEARCH (replaces "Multi-Ticker")
# ======================================================================
with tab2:
    section("WATCHLIST STRATEGY COMPARISON")
    with st.spinner("Running multi-ticker backtest..."):
        multi = run_multi_ticker(tuple(watchlist), start_date, end_date, sma_window)

    if not multi:
        st.error("No results. Check watchlist tickers.")
    else:
        if is_beginner:
            st.markdown("""
            <div style='background:#0f2040; border:1px solid #1a3357;
                        border-left:3px solid #00d4aa; border-radius:6px;
                        padding:14px 18px; margin-bottom:16px;
                        font-family:DM Sans,sans-serif; font-size:0.85rem;
                        color:#7b9bc0; line-height:1.6;'>
                This table shows how a simple moving-average trading strategy
                performed on each stock. <strong style='color:#e8f0fe;'>Return</strong>
                is how much money you'd have made. <strong style='color:#e8f0fe;'>
                Sharpe</strong> tells you how smooth the returns were
                (higher = better). <strong style='color:#e8f0fe;'>Max Drawdown</strong>
                is the biggest loss from peak to trough.
            </div>""", unsafe_allow_html=True)

        disp_df = pd.DataFrame([{
            "Ticker":    r["Ticker"],
            "Return":    f"{r['Return']:+.2%}",
            "CAGR":      f"{r['CAGR']:+.2%}",
            "Sharpe":    f"{r['Sharpe']:.3f}",
            "Max DD":    f"{r['Max DD']:.2%}",
            "Win Rate":  f"{r['Win Rate']:.1%}",
            "B&H Return":f"{r['B&H Return']:+.2%}",
            "Beat B&H":  r["Beat B&H"]
        } for r in multi])
        st.dataframe(disp_df, use_container_width=True, hide_index=True)

        beat       = sum(1 for r in multi if r["Return"] > r["B&H Return"])
        avg_sharpe = np.mean([r["Sharpe"] for r in multi])
        m1, m2, m3 = st.columns(3)
        m1.metric("Beat Buy & Hold",  f"{beat}/{len(multi)}")
        m2.metric("Avg Sharpe Ratio", f"{avg_sharpe:.3f}")
        m3.metric("Tickers Analyzed", len(multi))

        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        section("NORMALIZED PORTFOLIO CURVES")
        colors_palette = ["#00d4aa","#1a6eff","#ffd166","#ff4d6a",
                          "#7b61ff","#06d6a0","#118ab2","#8ed4b4"]
        fig3, ax3 = plt.subplots(figsize=(13, 5))
        aureline_chart_style(ax3, fig3)
        for idx, r in enumerate(multi):
            norm = [v/10000 for v in r["portfolio"]]
            ax3.plot(norm, color=colors_palette[idx % len(colors_palette)],
                     linewidth=1.5, label=r["Ticker"], alpha=0.9)
        ax3.axhline(y=1.0, color="#1a3357", linestyle="--", linewidth=1)
        ax3.set_ylabel("Return Multiple (×)", fontsize=8, color="#7b9bc0")
        ax3.set_xlabel("Trading Days", fontsize=8, color="#7b9bc0")
        ax3.legend(fontsize=8, facecolor="#0f2040", edgecolor="#1a3357",
                   labelcolor="#7b9bc0", ncol=2)
        plt.tight_layout(); st.pyplot(fig3); plt.close()


# ======================================================================
# TAB 3: ML vs SMA
# ======================================================================
with tab3:
    section(f"WALK-FORWARD ML MODEL vs SMA({sma_window}) — {ticker}")

    if is_beginner:
        st.markdown("""
        <div style='background:#0f2040; border:1px solid #1a3357;
                    border-left:2px solid #00d4aa; border-radius:4px;
                    padding:14px 18px; margin-bottom:16px;
                    font-family:DM Sans,sans-serif; font-size:0.85rem;
                    color:#7b9bc0; line-height:1.6;'>
            This test compares a machine learning model against a simple
            moving-average rule. The ML model studies past patterns to
            decide when to buy and sell — it's like having a research
            assistant that reads the history books before making a call.
            <strong style='color:#e8f0fe;'>A higher Sharpe ratio means
            smoother, more consistent returns.</strong>
        </div>""", unsafe_allow_html=True)
    else:
        st.info("Walk-forward training fits a new model at every time step "
                "using only past data — ~800 iterations. "
                "Results are cached after the first run.")

    if st.button("▶  RUN ML BACKTEST"):
        with st.spinner("Running walk-forward ML training..."):
            ml_dates, ml_pv = run_ml_backtest(ticker, start_date, end_date)

        if ml_pv is None:
            st.error("Not enough data for ML backtest. Try a longer date range.")
        else:
            _, sma_pv, _ = run_sma_backtest(ticker, start_date, end_date, sma_window)
            sma_aligned  = sma_pv[-len(ml_pv):]

            ml_ret  = (ml_pv[-1]/10000)  - 1
            sma_ret = (sma_aligned[-1]/10000) - 1
            ml_sh   = sharpe_ratio(ml_pv)
            sma_sh  = sharpe_ratio(sma_aligned)
            ml_mdd  = max_drawdown(ml_pv)
            sma_mdd = max_drawdown(sma_aligned)

            section("HEAD-TO-HEAD METRICS")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""<div style='font-family:JetBrains Mono,monospace;
                    font-size:0.65rem; color:#00d4aa; letter-spacing:0.15em;
                    margin-bottom:8px;'>ML MODEL</div>""",
                    unsafe_allow_html=True)
                st.metric("Return",       f"{ml_ret:+.2%}")
                st.metric("Sharpe Ratio", f"{ml_sh:.3f}")
                st.metric("Max Drawdown", f"{ml_mdd:.2%}")
                if is_beginner:
                    st.markdown(f"""
                    <div class='bcard' style='margin-top:10px;'>
                        <div class='bcard-label'>What this means</div>
                        <div class='bcard-desc'>{plain_sharpe(ml_sh)}</div>
                    </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div style='font-family:JetBrains Mono,monospace;
                    font-size:0.65rem; color:#1a6eff; letter-spacing:0.15em;
                    margin-bottom:8px;'>SMA({sma_window}) BASELINE</div>""",
                    unsafe_allow_html=True)
                st.metric("Return",       f"{sma_ret:+.2%}",
                          delta=f"{ml_ret-sma_ret:+.2%} ML edge")
                st.metric("Sharpe Ratio", f"{sma_sh:.3f}",
                          delta=f"{ml_sh-sma_sh:+.3f} ML edge")
                st.metric("Max Drawdown", f"{sma_mdd:.2%}",
                          delta=f"{ml_mdd-sma_mdd:+.2%} ML edge")

            st.markdown("<div style='margin-top:20px;'></div>",
                        unsafe_allow_html=True)
            section("EQUITY CURVES")
            fig4, ax4 = plt.subplots(figsize=(13, 4))
            aureline_chart_style(ax4, fig4)
            ax4.plot(ml_pv,      color="#00d4aa", linewidth=1.5, label="ML Strategy")
            ax4.plot(sma_aligned, color="#1a6eff", linewidth=1.2,
                     linestyle="--", alpha=0.7, label=f"SMA({sma_window})")
            ax4.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
            ax4.set_ylabel("Portfolio Value", fontsize=8, color="#7b9bc0")
            ax4.legend(fontsize=8, facecolor="#0f2040", edgecolor="#1a3357",
                       labelcolor="#7b9bc0")
            plt.tight_layout(); st.pyplot(fig4); plt.close()
    else:
        st.markdown("""
        <div style='text-align:center; padding:60px 20px;
                    font-family:JetBrains Mono,monospace; font-size:0.78rem;
                    color:#1a3357; background:#0f2040; border:1px solid #1a3357;
                    border-radius:6px;'>
            Press RUN ML BACKTEST to begin walk-forward analysis.
        </div>""", unsafe_allow_html=True)


# ======================================================================
# TAB 4: INTELLIGENCE HUB
# ======================================================================
with tab4:

    # ── Morning Brief ──
    section("MORNING INTELLIGENCE BRIEF")
    brief_files = sorted(glob.glob("experiments/reports/MORNING_BRIEF_*.md"),
                          reverse=True)
    daily_files = sorted(glob.glob("experiments/reports/DAILY_BRIEF_*.md"),
                          reverse=True)
    all_briefs  = brief_files + daily_files

    if all_briefs:
        latest_brief = all_briefs[0]
        brief_date   = Path(latest_brief).stem.replace(
            "MORNING_BRIEF_", "").replace("DAILY_BRIEF_", "")
        st.markdown(f"""
        <div style='font-family:JetBrains Mono,monospace; font-size:0.72rem;
                    color:#7b9bc0; margin-bottom:12px;'>
            Latest: {brief_date} · {len(all_briefs)} briefs in archive
        </div>""", unsafe_allow_html=True)
        with st.expander("📋 View Full Morning Brief", expanded=True):
            with open(latest_brief) as f:
                st.markdown(f.read())
    else:
        st.info("No morning brief found. Run milestone_38.py to generate one.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── News Intelligence ──
    section("NEWS INTELLIGENCE" if not is_beginner
            else "WHAT'S HAPPENING IN THE MARKET?")

    try:
        from src.database import Database as _DB
        _db = _DB()
        recent_news = _db.get_recent_news(limit=30)

        if recent_news:
            scores    = [n.get("impact_score", 0) for n in recent_news]
            avg_score = sum(scores)/len(scores) if scores else 0.0
            pos_count = sum(1 for s in scores if s > 0.2)
            neg_count = sum(1 for s in scores if s < -0.2)
            mood      = ("RISK-ON"  if avg_score > 0.1
                         else "RISK-OFF" if avg_score < -0.1
                         else "NEUTRAL")
            mood_color= ("#00d4aa" if mood == "RISK-ON"
                         else "#ff4d6a" if mood == "RISK-OFF"
                         else "#ffd166")

            st.markdown(f"""
            <div style='background:#0f2040; border:1px solid #1a3357;
                        border-left:3px solid {mood_color}; border-radius:6px;
                        padding:14px 18px; margin-bottom:16px;'>
                <span style='font-family:JetBrains Mono,monospace;
                             font-size:1rem; font-weight:700;
                             color:{mood_color};'>{mood}</span>
                <span style='font-family:JetBrains Mono,monospace;
                             font-size:0.72rem; color:#7b9bc0;
                             margin-left:12px;'>
                    Score: {avg_score:+.3f} ·
                    {pos_count} positive ·
                    {neg_count} negative ·
                    {len(recent_news)} articles
                </span>
                {f"<div style='font-family:DM Sans,sans-serif; font-size:0.83rem; color:#7b9bc0; margin-top:8px; line-height:1.5;'>{'More positive stories than negative — generally a good sign for markets.' if mood == 'RISK-ON' else 'More negative stories than positive — investors may be cautious.' if mood == 'RISK-OFF' else 'About equal positive and negative news — markets are uncertain.'}</div>" if is_beginner else ""}
            </div>""", unsafe_allow_html=True)

            # Ticker sentiment pills
            watch_tickers = ["AAPL","MSFT","NVDA","TSLA","SPY","PSEI.PS"]
            cols = st.columns(len(watch_tickers))
            for col, t in zip(cols, watch_tickers):
                t_news = _db.get_recent_news(ticker=t, limit=30)
                if t_news:
                    t_sc  = [n.get("impact_score",0) for n in t_news]
                    t_avg = sum(t_sc)/len(t_sc)
                    t_col = ("#00d4aa" if t_avg > 0.1
                             else "#ff4d6a" if t_avg < -0.1
                             else "#7b9bc0")
                    col.markdown(f"""
                    <div style='text-align:center; padding:10px 6px;
                                background:#0f2040; border:1px solid #1a3357;
                                border-radius:6px;'>
                        <div style='font-family:JetBrains Mono; font-size:0.6rem;
                                    color:#7b9bc0;'>{t}</div>
                        <div style='font-family:JetBrains Mono; font-size:0.88rem;
                                    font-weight:700; color:{t_col};'>
                            {t_avg:+.2f}</div>
                        <div style='font-family:JetBrains Mono; font-size:0.58rem;
                                    color:#1a3357;'>{len(t_news)} art.</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    col.markdown(f"""
                    <div style='text-align:center; padding:10px 6px;
                                background:#0f2040; border:1px solid #1a3357;
                                border-radius:6px;'>
                        <div style='font-family:JetBrains Mono; font-size:0.6rem;
                                    color:#7b9bc0;'>{t}</div>
                        <div style='font-family:JetBrains Mono; font-size:0.88rem;
                                    color:#1a3357;'>─</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:16px;'></div>",
                        unsafe_allow_html=True)
            section("RECENT HEADLINES")

            for news in recent_news[:15]:
                impact = news.get("market_impact", "neutral")
                score  = news.get("impact_score", 0)
                color  = ("#00d4aa" if impact == "positive"
                          else "#ff4d6a" if impact == "negative"
                          else "#7b9bc0")
                icon   = ("▲" if impact == "positive"
                          else "▼" if impact == "negative" else "─")
                source = news.get("source", "")[:14]
                try:
                    tickers_raw  = news.get("tickers_mentioned", "[]")
                    tickers_list = (json.loads(tickers_raw)
                                    if isinstance(tickers_raw, str)
                                    else tickers_raw)
                    tickers_str  = ", ".join(tickers_list[:2]) if tickers_list else ""
                except Exception:
                    tickers_str = ""
                headline = news.get("headline", "")[:75]

                st.markdown(f"""
                <div style='display:flex; align-items:center; gap:10px;
                            padding:8px 12px; background:#0f2040;
                            border:1px solid #1a3357;
                            border-left:2px solid {color};
                            border-radius:4px; margin-bottom:4px;'>
                    <span style='color:{color}; font-size:0.85rem;
                                 min-width:10px;'>{icon}</span>
                    <span style='font-family:DM Sans,sans-serif;
                                 font-size:0.82rem; color:#e8f0fe;
                                 flex:1;'>{headline}</span>
                    <span style='font-family:JetBrains Mono; font-size:0.6rem;
                                 color:#1a3357; white-space:nowrap;'>
                        {tickers_str}</span>
                    <span style='font-family:JetBrains Mono; font-size:0.6rem;
                                 color:#1a3357; white-space:nowrap;'>
                        {source}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No news data. Run milestone_37.py to populate.")

        _db.close()
    except Exception as e:
        st.error(f"Could not load news: {e}")
        st.info("Run milestone_38.py first.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Philippine Market ──
    section("PHILIPPINE MARKET" if not is_beginner
            else "PHILIPPINE STOCKS & MARKET")

    try:
        from src.philippine_market import (
            load_ph_universe, analyze_ph_ticker, PH_MACRO
        )
        _ph_tickers = ["PSEI.PS","PHI","BDOUY","BPHLY","JBFCF","AYAAF"]
        _today      = datetime.now().strftime("%Y-%m-%d")
        _ph_data    = load_ph_universe(_ph_tickers, "2023-01-01", _today)
        _ph_analyses= [analyze_ph_ticker(t, d)
                       for t, d in _ph_data.items()
                       if analyze_ph_ticker(t, d)]

        if _ph_analyses:
            # Macro row
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("BSP Rate",    f"{PH_MACRO['bsp_rate']}%",
                      delta="Key interest rate")
            m2.metric("Inflation",   f"{PH_MACRO['inflation']}%",
                      delta="Consumer price index")
            m3.metric("GDP Growth",  f"{PH_MACRO['gdp_growth']}%")
            m4.metric("USD/PHP",     f"₱{PH_MACRO['usd_php']:.2f}")

            if is_beginner:
                real_rate = PH_MACRO['bsp_rate'] - PH_MACRO['inflation']
                st.markdown(f"""
                <div style='background:#0f2040; border:1px solid #1a3357;
                            border-left:3px solid #00d4aa; border-radius:6px;
                            padding:14px 18px; margin:12px 0;
                            font-family:DM Sans,sans-serif; font-size:0.85rem;
                            color:#7b9bc0; line-height:1.6;'>
                    <strong style='color:#e8f0fe;'>What this means for you:</strong>
                    The BSP (Bangko Sentral ng Pilipinas) rate is {PH_MACRO['bsp_rate']}%
                    while inflation is {PH_MACRO['inflation']}%, giving a real interest
                    rate of {real_rate:.1f}%. This means your money in savings earns more
                    than inflation loses — which is generally good. GDP growth of
                    {PH_MACRO['gdp_growth']}% means the economy is still expanding.
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:14px;'></div>",
                        unsafe_allow_html=True)

            # Philippine ticker cards
            ph_cols = st.columns(3)
            for idx, a in enumerate(sorted(_ph_analyses,
                                           key=lambda x: x["ticker"])):
                col   = ph_cols[idx % 3]
                sig   = a["signal"]
                clr   = ("#00d4aa" if sig == "BUY"
                         else "#ff4d6a" if sig == "AVOID"
                         else "#7b9bc0")
                sym   = a["symbol"]
                label = plain_signal(sig) if is_beginner else sig
                col.markdown(f"""
                <div style='background:#0f2040; border:1px solid #1a3357;
                            border-top:2px solid {clr}; border-radius:6px;
                            padding:14px; margin-bottom:10px;'>
                    <div style='font-family:JetBrains Mono; font-size:0.62rem;
                                color:#7b9bc0; letter-spacing:0.1em;'>
                        {a['ticker']}</div>
                    <div style='font-family:Space Grotesk; font-size:0.78rem;
                                color:#e8f0fe; font-weight:600; margin:4px 0;'>
                        {a['name'][:24]}</div>
                    <div style='font-family:JetBrains Mono; font-size:1.05rem;
                                font-weight:700; color:{clr};'>
                        {sym}{a['price']:.2f}</div>
                    <div style='font-family:JetBrains Mono; font-size:0.7rem;
                                color:#7b9bc0; margin-top:6px;'>
                        20d: {a['ret_20d']:+.2f}% · RSI {a['rsi']:.1f}</div>
                    <div style='font-family:JetBrains Mono; font-size:0.62rem;
                                font-weight:700; color:{clr}; margin-top:6px;
                                letter-spacing:0.08em;'>
                        {label}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Philippine market data unavailable. "
                    "Run milestone_36.py first.")
    except Exception as e:
        st.error(f"Philippine market error: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Experiment Registry ──
    section("EXPERIMENT REGISTRY")
    try:
        from src.experiment_tracker import ExperimentTracker as _ET
        _tracker = _ET()
        if _tracker.registry:
            _df = _tracker.get_all()
            if not _df.empty:
                cols_to_show = [c for c in ["ID","Date","Ticker","Strategy"]
                                if c in _df.columns]
                st.dataframe(_df[cols_to_show].head(10),
                             use_container_width=True, hide_index=True)
                st.markdown(f"""
                <div style='font-family:JetBrains Mono; font-size:0.65rem;
                            color:#1a3357; margin-top:6px;'>
                    {len(_tracker.registry)} total experiments ·
                    Registry: experiments/registry.json
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No experiments logged yet.")
    except Exception as e:
        st.error(f"Registry error: {e}")


# ======================================================================
# SECTION 9: FOOTER
# ======================================================================
st.markdown(f"""
<div style='text-align:center; font-family:JetBrains Mono,monospace;
            font-size:0.6rem; color:#1a3357; margin-top:40px;
            padding-top:16px; border-top:1px solid #1a3357;'>
    AURELINE LABS v2.0
    &nbsp;·&nbsp; FOR RESEARCH PURPOSES ONLY
    &nbsp;·&nbsp; NOT FINANCIAL ADVICE
    &nbsp;·&nbsp; ATENEO DE MANILA UNIVERSITY
    &nbsp;·&nbsp; {'BEGINNER MODE' if is_beginner else 'PROFESSIONAL MODE'}
</div>""", unsafe_allow_html=True)