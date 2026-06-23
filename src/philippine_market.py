# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from src.data_handler import get_price_data

logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 2: VERIFIED PHILIPPINE MARKET TICKERS
# ======================================================================

# PSE Sector Indices — most reliable, trade in PHP
PSE_INDICES = {
    "PSEI.PS": "PSEi Composite Index",
    "ALL.PS":  "All Shares Index",
    "FIN.PS":  "Financials Index",
    "PRO.PS":  "Property Index",
    "HDG.PS":  "Holding Firms Index",
    "IND.PS":  "Industrial Index",
}

# Philippine companies via US OTC/ADR tickers — trade in USD
PSE_OTC = {
    "PHI":    "PLDT Inc. (NYSE)",
    "BDOUY":  "BDO Unibank (OTC)",
    "BPHLY":  "Bank of Philippine Islands (OTC)",
    "AYAAF":  "Ayala Land (OTC)",
    "JBFCF":  "Jollibee Foods (OTC)",
}

# Combined watchlist — indices first, then OTC stocks
PH_WATCHLIST = list(PSE_INDICES.keys()) + list(PSE_OTC.keys())

# Friendly name lookup across both sources
PH_NAMES = {**PSE_INDICES, **PSE_OTC}

# PSE Sector mapping for indices
PSE_SECTOR_INDICES = {
    "Composite":    ["PSEI.PS", "ALL.PS"],
    "Financials":   ["FIN.PS"],
    "Property":     ["PRO.PS"],
    "Conglomerates":["HDG.PS"],
    "Industrial":   ["IND.PS"],
}

# Philippine macro context (manually updated)
PH_MACRO = {
    "bsp_rate":     6.0,
    "inflation":    3.1,
    "gdp_growth":   5.8,
    "usd_php":      55.8,
    "unemployment": 4.1,
    "psei_pe":      14.2,
    "last_updated": "2026-06-22"
}


# ======================================================================
# SECTION 3: DATA LOADER
# ======================================================================
def get_ph_data(ticker, start, end):
    """
    Downloads Philippine market data.
    Handles both .PS suffix (PHP-denominated) and
    OTC/ADR tickers (USD-denominated).
    """
    data = get_price_data(ticker, start, end)

    if data is None or data.empty:
        logger.warning(f"No data for {ticker}")
        return None

    currency = "PHP" if ticker.endswith(".PS") else "USD"
    name     = PH_NAMES.get(ticker, ticker)
    logger.info(f"Loaded {ticker} ({name}): "
               f"{len(data)} rows | "
               f"{'₱' if currency == 'PHP' else '$'}"
               f"{data['Close'].iloc[-1]:.2f} latest")
    return data


def load_ph_universe(tickers, start, end):
    """
    Loads data for a list of Philippine market tickers.
    Returns dict of {ticker: DataFrame} for successful loads.
    """
    results = {}
    failed  = []

    for ticker in tickers:
        data = get_ph_data(ticker, start, end)
        if data is not None and len(data) > 20:
            results[ticker] = data
        else:
            failed.append(ticker)

    logger.info(f"PH universe: {len(results)} loaded, "
               f"{len(failed)} failed")
    if failed:
        logger.warning(f"Failed: {failed}")

    return results


# ======================================================================
# SECTION 4: SINGLE TICKER ANALYSIS
# ======================================================================
def analyze_ph_ticker(ticker, data):
    """
    Runs full quantitative analysis on a Philippine market ticker.
    Handles both PHP-denominated indices and USD OTC stocks.
    """
    if data is None or len(data) < 20:
        return None

    close        = data["Close"]
    latest_price = float(close.iloc[-1])
    currency     = "PHP" if ticker.endswith(".PS") else "USD"
    symbol       = "₱" if currency == "PHP" else "$"

    # Returns
    ret_1d  = float(close.pct_change(1).iloc[-1]) \
              if len(close) >= 2  else 0.0
    ret_5d  = float(close.pct_change(5).iloc[-1]) \
              if len(close) >= 6  else 0.0
    ret_20d = float(close.pct_change(20).iloc[-1]) \
              if len(close) >= 21 else 0.0
    ret_60d = float(close.pct_change(60).iloc[-1]) \
              if len(close) >= 61 else 0.0

    # 52-week range
    tail    = close.tail(252) if len(close) >= 252 else close
    high_52w= float(tail.max())
    low_52w = float(tail.min())
    pct_from_high = (latest_price - high_52w) / high_52w \
                    if high_52w > 0 else 0.0

    # Volatility
    daily_ret = close.pct_change().dropna()
    vol_20d   = float(daily_ret.tail(20).std() * np.sqrt(252)) \
                if len(daily_ret) >= 20 else 0.0

    # RSI(14)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, float("inf"))
    rsi   = float((100 - (100 / (1 + rs))).iloc[-1]) \
            if len(close) >= 15 else 50.0

    # Moving averages
    sma20 = float(close.rolling(20).mean().iloc[-1]) \
            if len(close) >= 20 else latest_price
    sma50 = float(close.rolling(50).mean().iloc[-1]) \
            if len(close) >= 50 else latest_price

    above_sma20 = latest_price > sma20
    above_sma50 = latest_price > sma50

    # Signal
    if above_sma20 and above_sma50 and rsi < 70:
        signal = "BUY"
    elif not above_sma20 and not above_sma50:
        signal = "AVOID"
    else:
        signal = "HOLD"

    return {
        "ticker":            ticker,
        "name":              PH_NAMES.get(ticker, ticker),
        "price":             round(latest_price, 2),
        "currency":          currency,
        "symbol":            symbol,
        "ret_1d":            round(ret_1d  * 100, 2),
        "ret_5d":            round(ret_5d  * 100, 2),
        "ret_20d":           round(ret_20d * 100, 2),
        "ret_60d":           round(ret_60d * 100, 2),
        "vol_20d":           round(vol_20d * 100, 1),
        "rsi":               round(rsi, 1),
        "above_sma20":       above_sma20,
        "above_sma50":       above_sma50,
        "high_52w":          round(high_52w, 2),
        "low_52w":           round(low_52w, 2),
        "pct_from_52w_high": round(pct_from_high * 100, 1),
        "signal":            signal,
    }


# ======================================================================
# SECTION 5: PSRI INDEX COMPARISON
# ======================================================================
def compare_pse_sectors(index_data):
    """
    Compares PSE sector indices to identify which sectors
    are outperforming or underperforming the composite.
    """
    if "PSEI.PS" not in index_data:
        return {}

    psei_ret = float(
        index_data["PSEI.PS"]["Close"].pct_change(20).iloc[-1]
    ) * 100 if len(index_data["PSEI.PS"]) >= 21 else 0.0

    comparison = {}
    for ticker, data in index_data.items():
        if ticker == "PSEI.PS":
            continue
        if len(data) < 21:
            continue
        ret = float(data["Close"].pct_change(20).iloc[-1]) * 100
        comparison[ticker] = {
            "name":        PSE_INDICES.get(ticker, ticker),
            "ret_20d":     round(ret, 2),
            "vs_psei":     round(ret - psei_ret, 2),
            "outperform":  ret > psei_ret
        }

    return comparison


# ======================================================================
# SECTION 6: PHILIPPINE MARKET INTELLIGENCE BRIEF
# ======================================================================
def generate_ph_brief(analyses, sector_comparison=None,
                       macro=None):
    """
    Generates a complete Philippine market intelligence brief.
    Separates index analysis from individual stock analysis.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    macro    = macro or PH_MACRO

    if not analyses:
        return "No data available for Philippine market brief."

    # Separate indices from stocks
    index_analyses = [a for a in analyses
                      if a["ticker"].endswith(".PS")]
    stock_analyses = [a for a in analyses
                      if not a["ticker"].endswith(".PS")]

    buy_signals   = [a for a in analyses if a["signal"] == "BUY"]
    avoid_signals = [a for a in analyses if a["signal"] == "AVOID"]
    hold_signals  = [a for a in analyses if a["signal"] == "HOLD"]

    sorted_ret = sorted(analyses,
                        key=lambda x: x["ret_20d"],
                        reverse=True)
    best  = sorted_ret[0]  if sorted_ret else None
    worst = sorted_ret[-1] if sorted_ret else None

    # Index table
    index_rows = "\n".join([
        f"| {a['ticker']:<10} | "
        f"{a['name'][:30]:<30} | "
        f"{a['symbol']}{a['price']:>10.2f} | "
        f"{a['ret_1d']:>+6.2f}% | "
        f"{a['ret_20d']:>+7.2f}% | "
        f"{a['rsi']:>5.1f} | "
        f"{a['signal']:<6} |"
        for a in sorted(index_analyses,
                        key=lambda x: x["ticker"])
    ])

    # Stock table
    stock_rows = "\n".join([
        f"| {a['ticker']:<8} | "
        f"{a['name'][:28]:<28} | "
        f"{a['symbol']}{a['price']:>9.2f} | "
        f"{a['ret_1d']:>+6.2f}% | "
        f"{a['ret_20d']:>+7.2f}% | "
        f"{a['rsi']:>5.1f} | "
        f"{a['signal']:<6} |"
        for a in sorted(stock_analyses,
                        key=lambda x: x["ticker"])
    ]) or "_No OTC stock data available_"

    # Sector comparison section
    sector_md = ""
    if sector_comparison:
        sector_md = "\n## Sector Performance vs PSEi\n\n"
        sector_md += "| Sector | 20d Return | vs PSEi | Signal |\n"
        sector_md += "|--------|-----------|---------|--------|\n"
        for ticker, info in sorted(
                sector_comparison.items(),
                key=lambda x: x[1]["ret_20d"], reverse=True):
            icon = "▲" if info["outperform"] else "▼"
            sector_md += (
                f"| {info['name']:<20} | "
                f"{info['ret_20d']:>+8.2f}% | "
                f"{info['vs_psei']:>+7.2f}% | "
                f"{icon} |\n"
            )

    buy_list = "\n".join([
        f"- **{a['name']}** ({a['ticker']}) "
        f"{a['symbol']}{a['price']:.2f} | "
        f"RSI: {a['rsi']:.1f} | "
        f"20d: {a['ret_20d']:+.2f}%"
        for a in buy_signals
    ]) or "_No buy signals today_"

    avoid_list = "\n".join([
        f"- **{a['name']}** ({a['ticker']}) "
        f"{a['symbol']}{a['price']:.2f} | "
        f"RSI: {a['rsi']:.1f} | "
        f"20d: {a['ret_20d']:+.2f}%"
        for a in avoid_signals
    ]) or "_No avoid signals today_"

    best_str  = (f"{best['name']} "
                 f"({best['ret_20d']:+.2f}%)") if best else "N/A"
    worst_str = (f"{worst['name']} "
                 f"({worst['ret_20d']:+.2f}%)") if worst else "N/A"

    return f"""# Aureline Labs — Philippine Market Intelligence Brief
## {date_str}

*Quantitative Research Platform · Ateneo de Manila University*
*For research purposes only. Not financial advice.*

---

## Philippine Macro Dashboard

| Indicator | Value | Context |
|-----------|-------|---------|
| BSP Key Rate | {macro['bsp_rate']}% | {"Restrictive" if macro['bsp_rate'] > 4 else "Accommodative"} |
| CPI Inflation | {macro['inflation']}% | {"Within target" if 2 <= macro['inflation'] <= 4 else "Outside target"} |
| Real Rate | {macro['bsp_rate'] - macro['inflation']:.1f}% | {"Positive — supportive for peso" if macro['bsp_rate'] > macro['inflation'] else "Negative"} |
| GDP Growth | {macro['gdp_growth']}% | {"Expansion" if macro['gdp_growth'] > 0 else "Contraction"} |
| USD/PHP | ₱{macro['usd_php']:.2f} | Exchange rate |
| PSEi Forward P/E | {macro['psei_pe']}x | Valuation multiple |

**Macro Interpretation:** BSP rate of {macro['bsp_rate']}% with
inflation at {macro['inflation']}% gives a real rate of
{macro['bsp_rate'] - macro['inflation']:.1f}%, supportive for
the peso but maintaining upward pressure on cost of capital.
GDP growth of {macro['gdp_growth']}% is consistent with
continued earnings expansion for domestic-focused companies.

---

## PSE Index Performance

| Ticker | Index | Level | 1-Day | 20-Day | RSI | Signal |
|--------|-------|-------|-------|--------|-----|--------|
{index_rows if index_rows else "_No index data loaded_"}

{sector_md}

---

## Philippine Companies (OTC/ADR)

| Ticker | Company | Price | 1-Day | 20-Day | RSI | Signal |
|--------|---------|-------|-------|--------|-----|--------|
{stock_rows}

---

## Signals Summary

**Buy Signals ({len(buy_signals)}):**
{buy_list}

**Avoid Signals ({len(avoid_signals)}):**
{avoid_list}

**Hold:** {len(hold_signals)} tickers

---

## Performance Leaders (20-day)

- **Best:** {best_str}
- **Worst:** {worst_str}

---

## Data Notes

- PSE sector indices (.PS) are denominated in **Philippine Pesos (₱)**
- OTC/ADR stocks are denominated in **US Dollars ($)**
- Data sourced from Yahoo Finance (15-min delayed for PSE)
- Individual PSE stocks (e.g. BDO.PS, ALI.PS) are not reliably
  available via Yahoo Finance — using OTC equivalents instead
- For institutional-grade PSE data, consider PSE DataLink or
  Refinitiv as future data source upgrades

---

*Aureline Labs v1.0 · Philippine Market Intelligence*
*Ateneo de Manila University · BS Applied Mathematics*
*Mathematical Finance · {date_str}*
"""