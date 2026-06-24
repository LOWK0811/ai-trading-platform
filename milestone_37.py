# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
from collections import Counter
from src.database import Database
from src.agents.news_agent import NewsAgent


# ======================================================================
# SECTION 2: LOGGING SETUP
# ======================================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 3: RUN THE NEWS AGENT
# ======================================================================
db    = Database()
agent = NewsAgent(db=db, include_ph=True)

print(f"\n{'='*60}")
print(f"  AURELINE LABS — NEWS AGENT v1.0")
print(f"{'='*60}")
print(f"  Feeds: {len(agent.feeds)}")
for name in agent.feeds:
    print(f"    · {name}")
print(f"\n  Fetching financial news...")

articles = agent.run(max_per_feed=10)


# ======================================================================
# SECTION 4: PRINT NEWS SUMMARY
# ======================================================================
print(f"\n{'='*60}")
print(f"  NEWS INTELLIGENCE SUMMARY")
print(f"{'='*60}")
print(f"  Total articles:  {len(articles)}")

# Sentiment breakdown
sentiments = Counter(a["sentiment"] for a in articles)
print(f"  Positive:        {sentiments.get('positive', 0)}")
print(f"  Neutral:         {sentiments.get('neutral', 0)}")
print(f"  Negative:        {sentiments.get('negative', 0)}")

# Most mentioned tickers
all_tickers = []
for a in articles:
    all_tickers.extend(a["tickers"])
ticker_counts = Counter(all_tickers).most_common(5)
if ticker_counts:
    print(f"\n  Most mentioned tickers:")
    for ticker, count in ticker_counts:
        print(f"    {ticker}: {count} articles")

# Market mood
mood = agent.get_market_mood()
print(f"\n  Market Mood: {mood['mood']} "
      f"(avg score: {mood['avg_score']:+.3f})")
print(f"  Positive news: {mood['pos_pct']:.1f}% | "
      f"Negative news: {mood['neg_pct']:.1f}%")


# ======================================================================
# SECTION 5: SHOW RECENT HEADLINES
# ======================================================================
print(f"\n  RECENT HEADLINES (top 10 by relevance)")
print(f"  {'-'*55}")

# Sort by abs(score) — most impactful first
sorted_articles = sorted(articles,
                          key=lambda x: abs(x["score"]),
                          reverse=True)

for i, a in enumerate(sorted_articles[:10], 1):
    sentiment_icon = {"positive": "▲",
                      "negative": "▼",
                      "neutral":  "─"}.get(
                          a["sentiment"], "?")
    tickers_str = (", ".join(a["tickers"])
                   if a["tickers"] else "general")
    headline    = a["headline"][:55] + "..." \
                  if len(a["headline"]) > 55 \
                  else a["headline"]
    print(f"  {i:>2}. {sentiment_icon} [{tickers_str:<10}] "
          f"{headline}")


# ======================================================================
# SECTION 6: PER-TICKER SENTIMENT
# ======================================================================
watchlist = ["AAPL", "MSFT", "NVDA", "TSLA",
             "SPY", "PSEI.PS", "JBFCF", "PHI"]

print(f"\n  TICKER SENTIMENT ANALYSIS")
print(f"  {'-'*50}")
print(f"  {'Ticker':<12} {'Articles':>8} "
      f"{'Avg Score':>10} {'Sentiment'}")
print(f"  {'-'*50}")

for ticker in watchlist:
    sent = agent.get_ticker_sentiment(ticker)
    icon = {"positive": "▲",
            "negative": "▼",
            "neutral":  "─"}.get(sent["sentiment"], "?")
    print(f"  {icon} {ticker:<10} "
          f"{sent['articles']:>8} "
          f"{sent['avg_score']:>+10.3f} "
          f"{sent['sentiment']}")


# ======================================================================
# SECTION 7: DATABASE STATS
# ======================================================================
stats = db.stats()
print(f"\n{'='*60}")
print(f"  DATABASE — NEWS TABLE")
print(f"{'='*60}")
print(f"  Total news records: {stats['news']:,}")
print(f"  Agent memory:       {stats['agent_memory']} entries")
print(f"{'='*60}")

db.close()