# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
# Fix macOS SSL certificate verification
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import logging
import feedparser
import re
from datetime import datetime, timedelta
from src.database import Database

logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 2: NEWS SOURCES — UPDATED URLS
# ======================================================================
NEWS_FEEDS = {
    "WSJ Markets":    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "WSJ Business":   "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",
    "NY Times Biz":   "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "Seeking Alpha":  "https://seekingalpha.com/market_currents.xml",
    "Investopedia":   "https://www.investopedia.com/feedbuilder/feed/getfeed?feedName=rss_headline",
    "CNBC Top News":  "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "CNBC Markets":   "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069",
    "Yahoo Finance Tech": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL,MSFT,NVDA,TSLA&region=US&lang=en-US",
}

PH_NEWS_FEEDS = {
    "BusinessWorld PH":    "https://www.bworldonline.com/category/economy/feed/",
    "Rappler Business":    "https://www.rappler.com/business/feed/",
    "Philippine Star Biz": "https://www.philstar.com/rss/business",
}


# ======================================================================
# SECTION 3: SENTIMENT ANALYZER
# ======================================================================
BULLISH_KEYWORDS = [
    "beat", "beats", "surges", "jumps", "rises", "rally",
    "record", "profit", "growth", "upgrade", "buy",
    "strong", "positive", "gains", "higher", "outperform",
    "exceeds", "boosts", "accelerates", "expands", "wins",
    "breakthrough", "partnership", "dividend", "buyback",
]

BEARISH_KEYWORDS = [
    "miss", "misses", "falls", "drops", "declines", "slumps",
    "loss", "losses", "downgrade", "sell", "weak", "negative",
    "lower", "underperform", "disappoints", "cut", "reduces",
    "lawsuit", "investigation", "recall", "layoff", "bankruptcy",
    "default", "recession", "inflation", "rate hike", "tariff",
]

# Ticker mention patterns
TICKER_PATTERNS = {
    # US stocks
    "AAPL":  ["apple", "aapl", "iphone", "ios", "mac", "tim cook"],
    "MSFT":  ["microsoft", "msft", "azure", "windows", "copilot",
               "satya nadella"],
    "NVDA":  ["nvidia", "nvda", "gpu", "cuda", "jensen huang",
               "geforce", "h100", "blackwell"],
    "JPM":   ["jpmorgan", "jpm", "jamie dimon", "chase bank"],
    "XOM":   ["exxon", "exxonmobil", "xom"],
    "TSLA":  ["tesla", "tsla", "elon musk", "electric vehicle",
               "ev", "autopilot", "cybertruck"],
    "SPY":   ["s&p 500", "s&p500", "spy", "stock market",
               "wall street", "fed", "federal reserve"],
    # Philippine stocks
    "PSEI.PS": ["psei", "philippine stock", "pse", "manila",
                "philippines market", "psei"],
    "PHI":   ["pldt", "smart communications", "philippine long"],
    "BDOUY": ["bdo unibank", "bdo", "banco de oro"],
    "BPHLY": ["bank of the philippine", "bpi", "bpi bank"],
    "JBFCF": ["jollibee", "jfc", "chowking", "mang inasal"],
}


def analyze_sentiment(text):
    """
    Returns sentiment score between -1.0 (very bearish)
    and +1.0 (very bullish) based on keyword analysis.
    """
    text_lower = text.lower()

    bull_count = sum(1 for kw in BULLISH_KEYWORDS
                     if kw in text_lower)
    bear_count = sum(1 for kw in BEARISH_KEYWORDS
                     if kw in text_lower)
    total      = bull_count + bear_count

    if total == 0:
        return 0.0, "neutral"

    score = (bull_count - bear_count) / total

    if score > 0.2:
        sentiment = "positive"
    elif score < -0.2:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return round(score, 3), sentiment


def extract_tickers(text):
    """
    Identifies which tickers from our watchlist are
    mentioned in the news text.
    """
    text_lower = text.lower()
    mentioned  = []

    for ticker, keywords in TICKER_PATTERNS.items():
        if any(kw in text_lower for kw in keywords):
            mentioned.append(ticker)

    return mentioned


# ======================================================================
# SECTION 4: NEWS AGENT
# ======================================================================
class NewsAgent:
    """
    The Aureline Labs News Agent.

    Monitors financial news RSS feeds, extracts relevant
    stories, analyzes sentiment, identifies ticker mentions,
    and stores structured intelligence in the database.

    This is a core component of the multi-agent architecture:
    the News Agent feeds information to the Executive Agent
    which synthesizes it with quantitative signals to produce
    the daily morning brief.
    """

    def __init__(self, db=None, include_ph=True):
        self.db         = db or Database()
        self.include_ph = include_ph
        self.agent_name = "news_agent"
        self.feeds      = dict(NEWS_FEEDS)
        if include_ph:
            self.feeds.update(PH_NEWS_FEEDS)
        logger.info(f"News Agent initialized | "
                   f"{len(self.feeds)} feeds")


    # ======================================================================
    # SECTION 5: FETCH NEWS FROM RSS FEEDS
    # ======================================================================
    def fetch_feed(self, name, url, max_articles=10):
        """
        Fetches articles from a single RSS feed.
        Uses feedparser with a browser User-Agent header
        to avoid being blocked.
        """
        try:
            import urllib.request
            # Many RSS feeds block default feedparser user-agent
            # Sending a browser agent gets past most blocks
            request = urllib.request.Request(
                url,
                headers={"User-Agent":
                         "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) "
                         "Chrome/120.0.0.0 Safari/537.36"}
            )
            response = urllib.request.urlopen(request, timeout=8)
            raw_xml  = response.read()
            feed     = feedparser.parse(raw_xml)

            if not feed.entries:
                logger.debug(f"{name}: feed parsed but 0 entries")
                return []

            articles = []
            for entry in feed.entries[:max_articles]:
                title   = getattr(entry, "title",   "")
                summary = getattr(entry, "summary", "")
                link    = getattr(entry, "link",    "")
                published = getattr(entry, "published", "")

                summary_clean = re.sub(r"<[^>]+>", "", summary)
                full_text     = f"{title} {summary_clean}"

                score, sentiment = analyze_sentiment(full_text)
                tickers          = extract_tickers(full_text)

                articles.append({
                    "headline":    title,
                    "source":      name,
                    "published_at":published[:25] if published else
                                   datetime.now().isoformat(),
                    "url":         link,
                    "summary":     summary_clean[:500],
                    "tickers":     tickers,
                    "sentiment":   sentiment,
                    "score":       score,
                })

            logger.info(f"{name}: {len(articles)} articles fetched")
            return articles

        except Exception as e:
            logger.warning(f"{name}: failed ({type(e).__name__}: {e})")
            return []

    # ======================================================================
    # SECTION 6: RUN FULL NEWS PIPELINE
    # ======================================================================
    def run(self, max_per_feed=8):
        """
        Fetches news from all configured feeds,
        deduplicates, stores to database,
        and returns structured intelligence.
        """
        logger.info("News Agent pipeline starting...")
        all_articles = []
        seen_titles  = set()

        for name, url in self.feeds.items():
            articles = self.fetch_feed(name, url, max_per_feed)
            for article in articles:
                # Deduplicate by headline
                title_key = article["headline"][:50].lower()
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    all_articles.append(article)

        # Store to database
        stored = 0
        for article in all_articles:
            try:
                self.db.insert_news(
                    headline      = article["headline"],
                    source        = article["source"],
                    published_at  = article["published_at"],
                    url           = article["url"],
                    summary       = article["summary"],
                    tickers       = article["tickers"],
                    market_impact = article["sentiment"],
                    impact_score  = article["score"]
                )
                stored += 1
            except Exception as e:
                logger.debug(f"News insert error: {e}")

        # Update agent memory
        self.db.set_memory(self.agent_name, "last_run",
                           datetime.now().isoformat())
        self.db.set_memory(self.agent_name,
                           "articles_stored", stored)
        self.db.set_memory(self.agent_name,
                           "feeds_checked",
                           list(self.feeds.keys()))

        logger.info(f"News Agent complete: "
                   f"{len(all_articles)} articles fetched, "
                   f"{stored} stored")
        return all_articles


    # ======================================================================
    # SECTION 7: QUERY NEWS INTELLIGENCE
    # ======================================================================
    def get_ticker_sentiment(self, ticker, days=7):
        """
        Returns the sentiment summary for a specific ticker
        over the past N days.
        """
        news = self.db.get_recent_news(ticker=ticker,
                                        days=days,
                                        limit=50)
        if not news:
            return {"ticker": ticker, "articles": 0,
                    "avg_score": 0.0, "sentiment": "neutral"}

        scores   = [n.get("impact_score", 0) for n in news]
        avg      = sum(scores) / len(scores) if scores else 0.0
        positive = sum(1 for s in scores if s > 0.2)
        negative = sum(1 for s in scores if s < -0.2)

        return {
            "ticker":    ticker,
            "articles":  len(news),
            "avg_score": round(avg, 3),
            "positive":  positive,
            "negative":  negative,
            "sentiment": "positive" if avg > 0.1
                         else "negative" if avg < -0.1
                         else "neutral"
        }

    def get_market_mood(self):
        """
        Returns the overall market sentiment from recent news.
        """
        all_news = self.db.get_recent_news(limit=100)
        if not all_news:
            return "neutral"

        scores  = [n.get("impact_score", 0) for n in all_news]
        avg     = sum(scores) / len(scores) if scores else 0.0
        pos_pct = sum(1 for s in scores if s > 0.2) / len(scores)
        neg_pct = sum(1 for s in scores if s < -0.2) / len(scores)

        if avg > 0.15 and pos_pct > 0.4:
            mood = "RISK-ON"
        elif avg < -0.15 and neg_pct > 0.4:
            mood = "RISK-OFF"
        else:
            mood = "NEUTRAL"

        return {
            "mood":        mood,
            "avg_score":   round(avg, 3),
            "pos_pct":     round(pos_pct * 100, 1),
            "neg_pct":     round(neg_pct * 100, 1),
            "total_news":  len(all_news)
        }