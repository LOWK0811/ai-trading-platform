# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
from datetime import datetime
from src.database import Database
from src.agents.research_agent import ResearchAgent
from src.agents.news_agent import NewsAgent

logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 2: EXECUTIVE AGENT
# ======================================================================
class ExecutiveAgent:
    """
    The Aureline Labs Executive Agent.

    Coordinates the Research Agent and News Agent,
    synthesizes their outputs, and generates a unified
    morning intelligence brief.

    This implements the 'Executive Agent' concept from
    the brainstorm: a coordinator that connects quantitative
    signals with news intelligence to produce actionable
    research output.

    Architecture:
    ExecutiveAgent
        ├── ResearchAgent  (price data, signals, regimes)
        └── NewsAgent      (headlines, sentiment, mentions)
        ↓
    Unified Morning Brief (Markdown + Database)
    """

    def __init__(self, db=None, watchlist=None, include_ph=True):
        self.db         = db or Database()
        self.watchlist  = watchlist or [
            "AAPL", "MSFT", "NVDA", "JPM",
            "XOM", "JNJ", "TSLA", "SPY"
        ]
        self.ph_watchlist = [
            "PSEI.PS", "PHI", "BDOUY", "BPHLY",
            "JBFCF", "AYAAF"
        ]
        self.research_agent = ResearchAgent(
            db=self.db, watchlist=self.watchlist)
        self.news_agent     = NewsAgent(
            db=self.db, include_ph=include_ph)
        self.agent_name     = "executive_agent"
        logger.info("Executive Agent initialized")


    # ======================================================================
    # SECTION 3: SYNTHESIZE SIGNALS + NEWS
    # ======================================================================
    def synthesize(self, analyses, articles):
        """
        Cross-references quantitative signals with news sentiment
        to produce conviction scores and alerts.

        Conviction logic:
        - STRONG BUY:  price signal BUY  + news positive
        - BUY:         price signal BUY  + news neutral
        - WATCH:       price signal HOLD + news positive/negative
        - AVOID:       price signal AVOID + any news
        - STRONG AVOID:price signal AVOID + news negative
        """
        synthesized = []

        for analysis in analyses:
            ticker = analysis["ticker"]

            # Get news sentiment for this ticker
            news_sentiment = self.news_agent.get_ticker_sentiment(
                ticker, days=7)
            news_score  = news_sentiment.get("avg_score", 0.0)
            news_count  = news_sentiment.get("articles", 0)
            news_label  = news_sentiment.get("sentiment", "neutral")

            price_signal = analysis["signal"]
            regime       = analysis["regime"]

            # Synthesize conviction
            if price_signal == "BUY" and news_label == "positive":
                conviction = "STRONG BUY"
                conv_score = 2
            elif price_signal == "BUY":
                conviction = "BUY"
                conv_score = 1
            elif (price_signal == "AVOID"
                  and news_label == "negative"):
                conviction = "STRONG AVOID"
                conv_score = -2
            elif price_signal == "AVOID":
                conviction = "AVOID"
                conv_score = -1
            elif news_label == "positive":
                conviction = "WATCH ▲"
                conv_score = 0.5
            elif news_label == "negative":
                conviction = "WATCH ▼"
                conv_score = -0.5
            else:
                conviction = "HOLD"
                conv_score = 0

            # Generate alert if signal + news align strongly
            alert = None
            if conv_score >= 2:
                alert = (f"BULLISH CONFLUENCE: {ticker} shows "
                        f"price momentum ({price_signal}) "
                        f"with positive news flow "
                        f"({news_count} articles, "
                        f"score {news_score:+.2f})")
            elif conv_score <= -2:
                alert = (f"BEARISH CONFLUENCE: {ticker} shows "
                        f"price weakness ({price_signal}) "
                        f"with negative news flow "
                        f"({news_count} articles, "
                        f"score {news_score:+.2f})")

            synthesized.append({
                **analysis,
                "news_score":   round(news_score, 3),
                "news_count":   news_count,
                "news_label":   news_label,
                "conviction":   conviction,
                "conv_score":   conv_score,
                "alert":        alert,
            })

        # Sort by conviction score descending
        synthesized.sort(key=lambda x: x["conv_score"],
                         reverse=True)
        return synthesized


    # ======================================================================
    # SECTION 4: GENERATE MORNING BRIEF
    # ======================================================================
    def generate_brief(self, synthesized, market_mood,
                        ph_analyses=None):
        """
        Generates the unified morning intelligence brief
        combining price signals, news sentiment, and
        Philippine market overview.
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H:%M PHT")
        now_str  = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Categorize
        strong_buy   = [s for s in synthesized
                        if s["conviction"] == "STRONG BUY"]
        buy          = [s for s in synthesized
                        if s["conviction"] == "BUY"]
        watch_up     = [s for s in synthesized
                        if s["conviction"] == "WATCH ▲"]
        hold         = [s for s in synthesized
                        if s["conviction"] == "HOLD"]
        watch_down   = [s for s in synthesized
                        if s["conviction"] == "WATCH ▼"]
        avoid        = [s for s in synthesized
                        if s["conviction"] == "AVOID"]
        strong_avoid = [s for s in synthesized
                        if s["conviction"] == "STRONG AVOID"]

        # Alerts
        alerts = [s["alert"] for s in synthesized
                  if s["alert"]]

        # Main watchlist table
        rows = "\n".join([
            f"| {s['ticker']:<6} | "
            f"${s['price']:>8.2f} | "
            f"{s['ret_20d']:>+7.2f}% | "
            f"{s['rsi']:>5.1f} | "
            f"{s['regime']:<20} | "
            f"{s['conviction']:<12} | "
            f"{s['news_score']:>+6.3f} |"
            for s in synthesized
        ])

        # Alert section
        alerts_md = "\n".join([
            f"⚡ {alert}" for alert in alerts
        ]) if alerts else "_No high-conviction alerts today_"

        # Strong signals
        strong_buy_md = "\n".join([
            f"- **{s['ticker']}** ${s['price']:.2f} | "
            f"Regime: {s['regime']} | "
            f"RSI: {s['rsi']:.1f} | "
            f"News: {s['news_score']:+.3f} "
            f"({s['news_count']} articles)"
            for s in strong_buy
        ]) or "_None_"

        strong_avoid_md = "\n".join([
            f"- **{s['ticker']}** ${s['price']:.2f} | "
            f"Regime: {s['regime']} | "
            f"RSI: {s['rsi']:.1f} | "
            f"News: {s['news_score']:+.3f} "
            f"({s['news_count']} articles)"
            for s in strong_avoid
        ]) or "_None_"

        # Philippine market section
        ph_section = ""
        if ph_analyses:
            ph_rows = "\n".join([
                f"| {a['ticker']:<10} | "
                f"{a['name'][:25]:<25} | "
                f"{a['symbol']}{a['price']:>9.2f} | "
                f"{a['ret_20d']:>+7.2f}% | "
                f"{a['rsi']:>5.1f} | "
                f"{a['signal']:<6} |"
                for a in ph_analyses
                if a is not None
            ])
            ph_section = f"""
---

## 🇵🇭 Philippine Market Overview

| Ticker | Company | Price | 20-Day | RSI | Signal |
|--------|---------|-------|--------|-----|--------|
{ph_rows}
"""

        # Mood emoji
        mood_emoji = {
            "RISK-ON":  "🟢",
            "RISK-OFF": "🔴",
            "NEUTRAL":  "🟡"
        }.get(market_mood.get("mood", "NEUTRAL"), "🟡")

        return f"""# 📊 Aureline Labs — Morning Intelligence Brief
## {date_str} · {time_str}

*Generated by Executive Agent at {now_str}*
*Aureline Labs v1.0 · Ateneo de Manila University*

---

## Market Mood {mood_emoji}

**Overall Sentiment: {market_mood.get('mood', 'NEUTRAL')}**

| Metric | Value |
|--------|-------|
| News Sentiment Score | {market_mood.get('avg_score', 0):+.3f} |
| Positive News | {market_mood.get('pos_pct', 0):.1f}% |
| Negative News | {market_mood.get('neg_pct', 0):.1f}% |
| Articles Analyzed | {market_mood.get('total_news', 0)} |

---

## ⚡ High-Conviction Alerts

{alerts_md}

---

## 🎯 Strong Buy Signals

{strong_buy_md}

---

## 🚫 Strong Avoid Signals

{strong_avoid_md}

---

## Full Watchlist Intelligence

| Ticker | Price | 20d Ret | RSI | Regime | Conviction | News |
|--------|-------|---------|-----|--------|------------|------|
{rows}

---
{ph_section}

## 📰 Market Context

**Regimes across watchlist:**
{self._regime_summary(synthesized)}

**News flow summary:**
- Tickers with positive news: {sum(1 for s in synthesized if s['news_label'] == 'positive')}
- Tickers with negative news: {sum(1 for s in synthesized if s['news_label'] == 'negative')}
- Tickers with neutral news:  {sum(1 for s in synthesized if s['news_label'] == 'neutral')}

---

## ⚠️ Risk Notes

- All signals are generated algorithmically
- Regime classification uses SMA(200), realized volatility,
  and price momentum
- News sentiment uses keyword-based analysis
- This brief is for **research purposes only**
- **Not financial advice**

---

*Aureline Labs · {date_str}*
*Quantitative Research & Intelligence Platform*
"""

    def _regime_summary(self, synthesized):
        """Builds a compact regime distribution summary."""
        from collections import Counter
        regimes = Counter(s["regime"] for s in synthesized)
        return "\n".join([
            f"- {regime}: {count} ticker(s)"
            for regime, count in regimes.most_common()
        ])


    # ======================================================================
    # SECTION 5: RUN FULL EXECUTIVE PIPELINE
    # ======================================================================
    def run(self):
        """
        Runs the complete Executive Agent pipeline:
        1. Run Research Agent (price data + signals)
        2. Run News Agent (headlines + sentiment)
        3. Run Philippine market analysis
        4. Synthesize signals + news
        5. Generate unified morning brief
        6. Save to database and file
        """
        start_time = datetime.now()
        date_str   = datetime.now().strftime("%Y-%m-%d")

        logger.info("="*50)
        logger.info("EXECUTIVE AGENT — Morning Pipeline Starting")
        logger.info("="*50)

        # Step 1: Research Agent
        logger.info("Step 1/4: Running Research Agent...")
        _, us_analyses = self.research_agent.run()

        # Step 2: News Agent
        logger.info("Step 2/4: Running News Agent...")
        articles    = self.news_agent.run(max_per_feed=8)
        market_mood = self.news_agent.get_market_mood()

        # Step 3: Philippine market
        logger.info("Step 3/4: Analyzing Philippine market...")
        from src.philippine_market import (
            load_ph_universe, analyze_ph_ticker, PH_WATCHLIST
        )
        ph_data     = load_ph_universe(
            self.ph_watchlist, "2023-01-01", date_str)
        ph_analyses = []
        for ticker, data in ph_data.items():
            analysis = analyze_ph_ticker(ticker, data)
            if analysis:
                ph_analyses.append(analysis)

        # Step 4: Synthesize
        logger.info("Step 4/4: Synthesizing signals + news...")
        synthesized = self.synthesize(us_analyses, articles)

        # Step 5: Generate brief
        brief = self.generate_brief(
            synthesized, market_mood, ph_analyses)

        # Step 6: Save
        brief_path = (f"experiments/reports/"
                     f"MORNING_BRIEF_{date_str}.md")
        os.makedirs("experiments/reports", exist_ok=True)
        with open(brief_path, "w") as f:
            f.write(brief)

        report_id = self.db.insert_report(
            title       = f"Morning Intelligence Brief — {date_str}",
            content     = brief,
            report_type = "morning_brief",
            tickers     = self.watchlist + self.ph_watchlist
        )

        # Update memory
        elapsed = (datetime.now() - start_time).seconds
        self.db.set_memory(self.agent_name, "last_run",
                           date_str)
        self.db.set_memory(self.agent_name, "last_runtime_s",
                           elapsed)
        self.db.set_memory(self.agent_name, "report_id",
                           report_id)

        logger.info(f"Executive Agent complete in {elapsed}s")
        logger.info(f"Brief saved: {brief_path}")

        return brief, synthesized, market_mood