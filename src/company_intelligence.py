# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import numpy as np
import pandas as pd
import logging
import yfinance as yf
from datetime import datetime
from src.data_handler import get_price_data
from src.regime_detector import RegimeDetector
from src.options import BlackScholes

logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 2: FUNDAMENTAL DATA FETCHER
# ======================================================================
def get_fundamentals(ticker):
    """
    Fetches company fundamental data from Yahoo Finance.
    Returns a cleaned dict of key metrics.
    """
    try:
        info = yf.Ticker(ticker).info
        if not info or "symbol" not in info:
            logger.warning(f"No fundamental data for {ticker}")
            return None

        def safe(key, default="N/A"):
            val = info.get(key, default)
            return val if val is not None else default

        def safe_pct(key):
            val = info.get(key)
            if val is None:
                return "N/A"
            return f"{val*100:.1f}%"

        def safe_billions(key):
            val = info.get(key)
            if val is None:
                return "N/A"
            if val >= 1e12:
                return f"${val/1e12:.2f}T"
            elif val >= 1e9:
                return f"${val/1e9:.2f}B"
            elif val >= 1e6:
                return f"${val/1e6:.2f}M"
            return f"${val:,.0f}"

        result = {
            # Identity
            "ticker":           ticker.upper(),
            "name":             safe("longName", ticker),
            "sector":           safe("sector"),
            "industry":         safe("industry"),
            "country":          safe("country"),
            "website":          safe("website"),
            "description":      safe("longBusinessSummary", ""),
            "employees":        safe("fullTimeEmployees"),

            # Price
            "current_price":    safe("currentPrice", safe("regularMarketPrice")),
            "target_mean":      safe("targetMeanPrice"),
            "target_high":      safe("targetHighPrice"),
            "target_low":       safe("targetLowPrice"),
            "analyst_rating":   safe("recommendationKey", "N/A").upper(),
            "num_analysts":     safe("numberOfAnalystOpinions", 0),

            # Valuation
            "market_cap":       safe_billions("marketCap"),
            "pe_ratio":         safe("trailingPE"),
            "forward_pe":       safe("forwardPE"),
            "pb_ratio":         safe("priceToBook"),
            "ps_ratio":         safe("priceToSalesTrailing12Months"),
            "ev_ebitda":        safe("enterpriseToEbitda"),
            "peg_ratio":        safe("pegRatio"),

            # Financials
            "revenue":          safe_billions("totalRevenue"),
            "revenue_growth":   safe_pct("revenueGrowth"),
            "gross_margin":     safe_pct("grossMargins"),
            "operating_margin": safe_pct("operatingMargins"),
            "profit_margin":    safe_pct("profitMargins"),
            "roe":              safe_pct("returnOnEquity"),
            "roa":              safe_pct("returnOnAssets"),

            # Balance sheet
            "total_cash":       safe_billions("totalCash"),
            "total_debt":       safe_billions("totalDebt"),
            "debt_to_equity":   safe("debtToEquity"),
            "current_ratio":    safe("currentRatio"),
            "free_cashflow":    safe_billions("freeCashflow"),

            # Dividends
            "dividend_yield":   safe_pct("dividendYield"),
            "payout_ratio":     safe_pct("payoutRatio"),

            # 52-week
            "week_52_high":     safe("fiftyTwoWeekHigh"),
            "week_52_low":      safe("fiftyTwoWeekLow"),
            "beta":             safe("beta"),
        }

        # ── Data quality checks for OTC/foreign tickers ──
        div = result.get("dividend_yield", "N/A")
        if div != "N/A":
            try:
                if float(div.replace("%","")) > 30:
                    result["dividend_yield"] = "N/A (data error)"
            except Exception:
                pass

        if result["country"] not in ["United States", "US", None, "N/A"]:
            result["revenue_note"] = "⚠️ Revenue may be in local currency"
        else:
            result["revenue_note"] = ""

        return result
    except Exception as e:
        logger.error(f"Failed to fetch fundamentals for {ticker}: {e}")
        return None

# ======================================================================
# SECTION 3: TECHNICAL ANALYSIS SNAPSHOT
# ======================================================================
def get_technical_snapshot(ticker, start="2021-01-01"):
    """
    Runs our full technical analysis stack on a ticker.
    Returns a compact snapshot dict.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    data  = get_price_data(ticker, start, today)

    if data is None or len(data) < 60:
        return None

    close = data["Close"]
    price = float(close.iloc[-1])

    # Returns
    ret_1d  = float(close.pct_change(1).iloc[-1])  * 100
    ret_5d  = float(close.pct_change(5).iloc[-1])  * 100
    ret_20d = float(close.pct_change(20).iloc[-1]) * 100
    ret_ytd = float(close.pct_change(len(close)-1).iloc[-1]) * 100

    # Volatility
    dr     = close.pct_change().dropna()
    vol_20 = float(dr.tail(20).std() * np.sqrt(252)) * 100
    vol_60 = float(dr.tail(60).std() * np.sqrt(252)) * 100

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, float("inf"))
    rsi   = float((100 - 100/(1+rs)).iloc[-1])

    # Moving averages
    sma20  = float(close.rolling(20).mean().iloc[-1])
    sma50  = float(close.rolling(50).mean().iloc[-1])
    sma200 = float(close.rolling(200).mean().iloc[-1]) \
             if len(close) >= 200 else None

    # 52-week
    high52 = float(close.tail(252).max())
    low52  = float(close.tail(252).min())
    pct_from_high = (price - high52) / high52 * 100

    # Regime
    try:
        detector = RegimeDetector()
        regime_df = detector.detect(data)
        regime    = regime_df["regime"].iloc[-1]
    except Exception:
        regime = "UNKNOWN"

    # Options
    try:
        bs     = BlackScholes(S=price, K=round(price),
                              T=0.25, r=0.05,
                              sigma=vol_20/100)
        call3m = round(bs.call_price(), 2)
        put3m  = round(bs.put_price(),  2)
        iv     = round(vol_20/100, 4)
    except Exception:
        call3m = put3m = iv = "N/A"

    # Signal
    above20  = price > sma20
    above50  = price > sma50
    above200 = (price > sma200) if sma200 else None

    if above20 and above50 and rsi < 70 and regime == "BULL_TRENDING":
        signal = "BUY"
    elif (not above20 and not above50) or regime in ["BEAR_TRENDING","HIGH_VOLATILITY"]:
        signal = "AVOID"
    else:
        signal = "HOLD"

    return {
        "price":         round(price, 2),
        "ret_1d":        round(ret_1d, 2),
        "ret_5d":        round(ret_5d, 2),
        "ret_20d":       round(ret_20d, 2),
        "ret_ytd":       round(ret_ytd, 2),
        "vol_20d":       round(vol_20, 1),
        "vol_60d":       round(vol_60, 1),
        "rsi":           round(rsi, 1),
        "sma20":         round(sma20, 2),
        "sma50":         round(sma50, 2),
        "sma200":        round(sma200, 2) if sma200 else "N/A",
        "above_sma20":   above20,
        "above_sma50":   above50,
        "above_sma200":  above200,
        "high_52w":      round(high52, 2),
        "low_52w":       round(low52,  2),
        "pct_from_high": round(pct_from_high, 1),
        "regime":        regime,
        "signal":        signal,
        "call_3m":       call3m,
        "put_3m":        put3m,
        "iv_20d":        iv,
    }


# ======================================================================
# SECTION 4: BULL / BEAR CASE GENERATOR
# ======================================================================
def generate_bull_bear(fundamentals, technical):
    """
    Generates structured bull and bear cases from
    fundamental and technical data.
    Rule-based — no API needed.
    """
    bull = []
    bear = []

    if fundamentals:
        # Revenue growth
        rg = fundamentals.get("revenue_growth", "N/A")
        if rg != "N/A":
            try:
                rg_val = float(rg.replace("%",""))
                if rg_val > 10:
                    bull.append(f"Revenue growing at {rg} — strong top-line momentum")
                elif rg_val < 0:
                    bear.append(f"Revenue declining ({rg}) — business may be shrinking")
            except Exception:
                pass

        # Profit margin
        pm = fundamentals.get("profit_margin", "N/A")
        if pm != "N/A":
            try:
                pm_val = float(pm.replace("%",""))
                if pm_val > 15:
                    bull.append(f"High profit margin ({pm}) — efficient, scalable business")
                elif pm_val < 0:
                    bear.append(f"Negative profit margin ({pm}) — company is losing money")
            except Exception:
                pass

        # Valuation
        pe = fundamentals.get("pe_ratio", "N/A")
        if pe != "N/A":
            try:
                pe_val = float(pe)
                if pe_val < 15:
                    bull.append(f"Low P/E ratio ({pe_val:.1f}x) — potentially undervalued")
                elif pe_val > 50:
                    bear.append(f"High P/E ratio ({pe_val:.1f}x) — expensive valuation")
            except Exception:
                pass

        # Debt
        dte = fundamentals.get("debt_to_equity", "N/A")
        if dte != "N/A":
            try:
                dte_val = float(dte)
                if dte_val < 50:
                    bull.append(f"Low debt-to-equity ({dte_val:.1f}) — financially conservative")
                elif dte_val > 200:
                    bear.append(f"High debt ({dte_val:.1f} D/E) — heavy financial burden")
            except Exception:
                pass

        # Analyst rating
        rating = fundamentals.get("analyst_rating", "N/A")
        n_analysts = fundamentals.get("num_analysts", 0)
        if rating not in ["N/A", ""]:
            if "buy" in rating.lower() or "outperform" in rating.lower():
                bull.append(f"Analyst consensus: {rating} "
                            f"({n_analysts} analysts)")
            elif "sell" in rating.lower() or "underperform" in rating.lower():
                bear.append(f"Analyst consensus: {rating} "
                            f"({n_analysts} analysts)")

        # Price vs target
        curr  = fundamentals.get("current_price")
        tgt   = fundamentals.get("target_mean")
        if curr not in ["N/A", None] and tgt not in ["N/A", None]:
            try:
                upside = (float(tgt) / float(curr) - 1) * 100
                if upside > 15:
                    bull.append(f"Analyst price target implies "
                                f"{upside:.1f}% upside")
                elif upside < -10:
                    bear.append(f"Analyst price target implies "
                                f"{abs(upside):.1f}% downside")
            except Exception:
                pass

    if technical:
        # Momentum
        ret20 = technical.get("ret_20d", 0)
        if ret20 > 5:
            bull.append(f"Strong 20-day momentum: +{ret20:.1f}%")
        elif ret20 < -10:
            bear.append(f"Weak 20-day momentum: {ret20:.1f}%")

        # RSI
        rsi = technical.get("rsi", 50)
        if rsi < 35:
            bull.append(f"RSI at {rsi:.1f} — oversold, potential bounce")
        elif rsi > 70:
            bear.append(f"RSI at {rsi:.1f} — overbought, pullback risk")

        # Regime
        regime = technical.get("regime", "")
        if regime == "BULL_TRENDING":
            bull.append("Currently in Bull Trending regime — "
                        "favorable technical environment")
        elif regime in ["BEAR_TRENDING", "HIGH_VOLATILITY"]:
            bear.append(f"Currently in {regime} regime — "
                        "unfavorable technical environment")

        # 52-week position
        pct_high = technical.get("pct_from_high", 0)
        if pct_high < -30:
            bull.append(f"Trading {abs(pct_high):.1f}% below "
                        f"52-week high — deep discount")
        elif pct_high > -5:
            bear.append("Near 52-week high — limited near-term upside")

    # Pad with defaults if too few
    if len(bull) < 2:
        bull.append("Long-term fundamental story remains intact")
    if len(bear) < 2:
        bear.append("Execution risk — any guidance miss could pressure stock")

    return bull[:5], bear[:5]


# ======================================================================
# SECTION 5: GENERATE FULL COMPANY REPORT
# ======================================================================
def generate_company_report(ticker, is_beginner=False):
    """
    Generates the complete company intelligence report
    combining fundamentals, technicals, and bull/bear analysis.
    """
    logger.info(f"Generating company report: {ticker}")

    fund  = get_fundamentals(ticker)
    tech  = get_technical_snapshot(ticker)
    bull, bear = generate_bull_bear(fund, tech)

    if fund is None and tech is None:
        return None

    return {
        "ticker":       ticker,
        "fundamentals": fund,
        "technical":    tech,
        "bull_case":    bull,
        "bear_case":    bear,
        "generated_at": datetime.now().isoformat(),
        "is_beginner":  is_beginner,
    }