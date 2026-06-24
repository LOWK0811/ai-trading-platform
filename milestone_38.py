# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
from src.database import Database
from src.agents.executive_agent import ExecutiveAgent


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
# SECTION 3: RUN THE EXECUTIVE AGENT
# ======================================================================
print(f"\n{'='*60}")
print(f"  AURELINE LABS — EXECUTIVE AGENT v1.0")
print(f"  Complete Morning Intelligence Pipeline")
print(f"{'='*60}\n")

db    = Database()
agent = ExecutiveAgent(db=db, include_ph=True)
brief, synthesized, mood = agent.run()


# ======================================================================
# SECTION 4: PRINT UNIFIED INTELLIGENCE SUMMARY
# ======================================================================
print(f"\n{'='*65}")
print(f"  UNIFIED MORNING INTELLIGENCE — AURELINE LABS")
print(f"{'='*65}")
print(f"  Market Mood: {mood.get('mood','NEUTRAL')} "
      f"| Score: {mood.get('avg_score',0):+.3f} "
      f"| Articles: {mood.get('total_news',0)}")

# Alerts
alerts = [s["alert"] for s in synthesized if s["alert"]]
if alerts:
    print(f"\n  HIGH-CONVICTION ALERTS:")
    for alert in alerts:
        print(f"  ⚡ {alert}")

# Full synthesis table
print(f"\n  {'Ticker':<6} {'Price':>9} "
      f"{'20d':>7} {'RSI':>5} "
      f"{'Conviction':<14} {'News':>6}")
print(f"  {'-'*60}")

for s in synthesized:
    conv_icon = {
        "STRONG BUY":   "🟢🟢",
        "BUY":          "🟢",
        "WATCH ▲":      "🔵",
        "HOLD":         "⚪",
        "WATCH ▼":      "🟠",
        "AVOID":        "🔴",
        "STRONG AVOID": "🔴🔴",
    }.get(s["conviction"], "⚪")

    print(f"  {s['ticker']:<6} "
          f"${s['price']:>8.2f} "
          f"{s['ret_20d']:>+6.2f}% "
          f"{s['rsi']:>5.1f} "
          f"{conv_icon} {s['conviction']:<12} "
          f"{s['news_score']:>+6.3f}")

# Database stats
stats = db.stats()
print(f"\n  Database: {stats['prices']:,} prices | "
      f"{stats['news']:,} news | "
      f"{stats['reports']} reports")
print(f"\n  Brief saved to experiments/reports/MORNING_BRIEF_*.md")
print(f"{'='*65}")

db.close()