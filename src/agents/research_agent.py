# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.database import Database
from src.data_handler import get_price_data
from src.indicators import add_sma, add_atr
from src.features import build_all_features, all_feature_cols
from src.regime_detector import RegimeDetector
from src.options import BlackScholes

logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 2: RESEARCH AGENT
# ======================================================================
class ResearchAgent:
    """
    The Aureline Labs Research Agent.

    Runs autonomously on a schedule to:
    1. Update market data for all watched tickers
    2. Detect current market regime for each ticker
    3. Compute key technical signals
    4. Generate a structured daily research brief
    5. Save everything to the database

    This is Level 1 automation from the brainstorm:
    scheduled, autonomous, persistent.
    """

    def __init__(self, db=None, watchlist=None):
        self.db        = db or Database()
        self.watchlist = watchlist or [
            "AAPL", "MSFT", "NVDA", "JPM",
            "XOM", "JNJ", "TSLA", "SPY"
        ]
        self.detector  = RegimeDetector()
        self.agent_name = "research_agent"
        logger.info(f"Research Agent initialized | "
                   f"Watchlist: {self.watchlist}")


    # ======================================================================
    # SECTION 3: UPDATE MARKET DATA
    # ======================================================================
    def update_market_data(self, start="2021-01-01"):
        """Downloads latest data and updates the database."""
        today   = datetime.now().strftime("%Y-%m-%d")
        updated = []

        for ticker in self.watchlist:
            try:
                data = get_price_data(ticker, start, today)
                if data is not None and len(data) > 0:
                    self.db.insert_prices(ticker, data)
                    updated.append(ticker)
                    logger.info(f"Updated {ticker}: "
                               f"{len(data)} rows")
            except Exception as e:
                logger.error(f"Failed to update {ticker}: {e}")

        self.db.set_memory(self.agent_name,
                           "last_data_update", today)
        self.db.set_memory(self.agent_name,
                           "tickers_updated", updated)
        return updated


    # ======================================================================
    # SECTION 4: ANALYZE TICKER
    # ======================================================================
    def analyze_ticker(self, ticker, start="2021-01-01"):
        """
        Runs full analysis on a single ticker:
        - Current price and recent performance
        - Regime detection
        - Technical signals
        - Volatility assessment
        - Options-derived signals
        """
        today = datetime.now().strftime("%Y-%m-%d")
        data  = get_price_data(ticker, start, today)

        if data is None or len(data) < 60:
            logger.warning(f"Insufficient data for {ticker}")
            return None

        # Basic price stats
        close        = data["Close"]
        latest_price = float(close.iloc[-1])
        ret_1d       = float(close.pct_change().iloc[-1])
        ret_5d       = float(close.pct_change(5).iloc[-1])
        ret_20d      = float(close.pct_change(20).iloc[-1])
        ret_60d      = float(close.pct_change(60).iloc[-1])

        # 52-week high and low
        high_52w = float(close.tail(252).max())
        low_52w  = float(close.tail(252).min())
        pct_from_high = (latest_price - high_52w) / high_52w

        # Volatility
        daily_returns = close.pct_change().dropna()
        vol_20d = float(daily_returns.tail(20).std() * np.sqrt(252))
        vol_60d = float(daily_returns.tail(60).std() * np.sqrt(252))

        # Regime detection
        regime_df = self.detector.detect(data)
        current_regime = regime_df["regime"].iloc[-1]

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, float("inf"))
        rsi   = float(100 - (100 / (1 + rs)).iloc[-1])

        # SMA signals
        sma_20  = float(close.rolling(20).mean().iloc[-1])
        sma_50  = float(close.rolling(50).mean().iloc[-1])
        sma_200 = float(close.rolling(200).mean().iloc[-1])
        above_200 = latest_price > sma_200

        # ATR
        high = data["High"]
        low  = data["Low"]
        prev_close = close.shift(1)
        tr  = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])

        # Simple signal
        if (current_regime == "BULL_TRENDING" and
                latest_price > sma_20 and rsi < 70):
            signal = "BUY"
            signal_strength = min(
                (ret_20d * 10 + (1 if above_200 else 0)), 1.0)
        elif (current_regime in ["BEAR_TRENDING","HIGH_VOLATILITY"]
              or latest_price < sma_200):
            signal = "AVOID"
            signal_strength = 0.0
        else:
            signal = "HOLD"
            signal_strength = 0.5

        # Options-based IV estimate
        bs = BlackScholes(S=latest_price, K=round(latest_price),
                          T=0.25, r=0.05, sigma=vol_20d)
        call_price = float(bs.call_price())
        put_price  = float(bs.put_price())

        return {
            "ticker":          ticker,
            "date":            today,
            "price":           round(latest_price, 2),
            "ret_1d":          round(ret_1d * 100, 2),
            "ret_5d":          round(ret_5d * 100, 2),
            "ret_20d":         round(ret_20d * 100, 2),
            "ret_60d":         round(ret_60d * 100, 2),
            "vol_20d":         round(vol_20d * 100, 1),
            "rsi":             round(rsi, 1),
            "regime":          current_regime,
            "above_sma200":    above_200,
            "pct_from_52w_high": round(pct_from_high * 100, 1),
            "atr":             round(atr, 2),
            "signal":          signal,
            "call_price_3m":   round(call_price, 2),
            "put_price_3m":    round(put_price, 2),
        }


    # ======================================================================
    # SECTION 5: GENERATE DAILY BRIEF
    # ======================================================================
    def generate_daily_brief(self, analyses):
        """
        Generates a structured daily research brief
        from ticker analyses.
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        now_str  = datetime.now().strftime("%Y-%m-%d %H:%M")

        buy_signals  = [a for a in analyses
                        if a["signal"] == "BUY"]
        avoid_signals = [a for a in analyses
                         if a["signal"] == "AVOID"]
        hold_signals  = [a for a in analyses
                         if a["signal"] == "HOLD"]

        # Regime summary
        regimes = {}
        for a in analyses:
            r = a["regime"]
            regimes[r] = regimes.get(r, 0) + 1

        # Best and worst performers
        sorted_by_ret = sorted(analyses,
                               key=lambda x: x["ret_20d"],
                               reverse=True)
        best_20d  = sorted_by_ret[0]
        worst_20d = sorted_by_ret[-1]

        # Build report sections
        ticker_rows = "\n".join([
            f"| {a['ticker']:<6} | ${a['price']:>8.2f} | "
            f"{a['ret_1d']:>+6.2f}% | "
            f"{a['ret_20d']:>+7.2f}% | "
            f"{a['rsi']:>5.1f} | "
            f"{a['vol_20d']:>5.1f}% | "
            f"{a['regime']:<20} | "
            f"{a['signal']:<6} |"
            for a in sorted(analyses,
                            key=lambda x: x["ticker"])
        ])

        regime_summary = "\n".join([
            f"- **{regime}**: {count} ticker(s)"
            for regime, count in regimes.items()
        ])

        buy_details = "\n".join([
            f"- **{a['ticker']}** ${a['price']:.2f} | "
            f"RSI: {a['rsi']:.1f} | "
            f"20d Return: {a['ret_20d']:+.2f}% | "
            f"3m ATM Call: ${a['call_price_3m']:.2f}"
            for a in buy_signals
        ]) or "_No buy signals today_"

        avoid_details = "\n".join([
            f"- **{a['ticker']}** ${a['price']:.2f} | "
            f"Regime: {a['regime']} | "
            f"20d Return: {a['ret_20d']:+.2f}%"
            for a in avoid_signals
        ]) or "_No avoid signals today_"

        report = f"""# Aureline Labs — Daily Research Brief
## {date_str}

*Generated automatically by the Research Agent at {now_str}*
*Aureline Labs v1.0 · Ateneo de Manila University*

---

## Market Overview

**Tickers Analyzed:** {len(analyses)}
**Buy Signals:** {len(buy_signals)}
**Hold Signals:** {len(hold_signals)}
**Avoid Signals:** {len(avoid_signals)}

**Regime Distribution:**
{regime_summary}

**Best 20-day Performer:** {best_20d['ticker']} ({best_20d['ret_20d']:+.2f}%)
**Worst 20-day Performer:** {worst_20d['ticker']} ({worst_20d['ret_20d']:+.2f}%)

---

## Watchlist Summary

| Ticker | Price | 1-Day | 20-Day | RSI | Vol | Regime | Signal |
|--------|-------|-------|--------|-----|-----|--------|--------|
{ticker_rows}

---

## Buy Signals

{buy_details}

---

## Avoid Signals

{avoid_details}

---

## Options Intelligence

3-month at-the-money options pricing across watchlist:

| Ticker | Price | Call | Put | Vol (20d) |
|--------|-------|------|-----|-----------|
{chr(10).join([
    f"| {a['ticker']:<6} | "
    f"${a['price']:>8.2f} | "
    f"${a['call_price_3m']:>6.2f} | "
    f"${a['put_price_3m']:>5.2f} | "
    f"{a['vol_20d']:>5.1f}% |"
    for a in sorted(analyses, key=lambda x: x['ticker'])
])}

---

## Research Notes

- All signals are generated algorithmically using the
  Aureline Labs quantitative research framework
- Regime classification uses SMA(200), realized volatility
  vs historical median, and price momentum signals
- Options prices computed via Black-Scholes
  (historical vol as volatility estimate)

---

*This report is for research purposes only.
Not financial advice.*

*Aureline Labs · {date_str}*
"""
        return report


    # ======================================================================
    # SECTION 6: RUN FULL PIPELINE
    # ======================================================================
    def run(self):
        """
        Runs the complete Research Agent pipeline:
        1. Update market data
        2. Analyze all tickers
        3. Generate daily brief
        4. Save to database
        5. Save to file
        """
        start_time = datetime.now()
        logger.info("Research Agent pipeline starting...")

        # Step 1: Update data
        updated = self.update_market_data()
        logger.info(f"Data updated for: {updated}")

        # Step 2: Analyze tickers
        analyses = []
        for ticker in self.watchlist:
            logger.info(f"Analyzing {ticker}...")
            analysis = self.analyze_ticker(ticker)
            if analysis:
                analyses.append(analysis)

        # Step 3: Generate brief
        brief = self.generate_daily_brief(analyses)

        # Step 4: Save to database
        date_str = datetime.now().strftime("%Y-%m-%d")
        report_id = self.db.insert_report(
            title       = f"Daily Research Brief — {date_str}",
            content     = brief,
            report_type = "daily_brief",
            tickers     = self.watchlist
        )

        # Step 5: Save to file
        os.makedirs("experiments/reports", exist_ok=True)
        filename = (f"experiments/reports/"
                   f"DAILY_BRIEF_{date_str}.md")
        with open(filename, "w") as f:
            f.write(brief)

        # Update agent memory
        elapsed = (datetime.now() - start_time).seconds
        self.db.set_memory(self.agent_name,
                           "last_run", date_str)
        self.db.set_memory(self.agent_name,
                           "last_report_id", report_id)
        self.db.set_memory(self.agent_name,
                           "analyses_count", len(analyses))

        logger.info(f"Research Agent complete in {elapsed}s | "
                   f"Report saved: {filename}")
        return brief, analyses