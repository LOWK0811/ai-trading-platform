# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
from src.database import Database
from src.agents.research_agent import ResearchAgent


# ======================================================================
# SECTION 2: LOGGING SETUP
# ======================================================================
os.makedirs("logs", exist_ok=True)
os.makedirs("src/agents", exist_ok=True)

# Create __init__.py for agents package
init_path = "src/agents/__init__.py"
if not os.path.exists(init_path):
    with open(init_path, "w") as f:
        f.write("# Aureline Labs — Agent Package\n")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logging.getLogger("yfinance").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 3: RUN THE RESEARCH AGENT
# ======================================================================
db    = Database()
agent = ResearchAgent(db=db, watchlist=[
    "AAPL", "MSFT", "NVDA", "JPM",
    "XOM", "JNJ", "TSLA", "SPY"
])

print(f"\n{'='*55}")
print(f"  AURELINE LABS — RESEARCH AGENT v1.0")
print(f"{'='*55}")
print(f"  Running automated research pipeline...")
print(f"  Watchlist: {agent.watchlist}\n")

brief, analyses = agent.run()

# Print summary
print(f"\n{'='*55}")
print(f"  RESEARCH AGENT — PIPELINE COMPLETE")
print(f"{'='*55}")

for a in sorted(analyses, key=lambda x: x["ticker"]):
    signal_color = {
        "BUY": "▲", "HOLD": "─", "AVOID": "▼"
    }.get(a["signal"], "?")
    print(f"  {signal_color} {a['ticker']:<6} "
          f"${a['price']:>8.2f} | "
          f"{a['ret_20d']:>+7.2f}% (20d) | "
          f"RSI {a['rsi']:>5.1f} | "
          f"{a['regime']}")

# Check database
stats = db.stats()
print(f"\n  Database reports: {stats['reports']}")
print(f"  Agent memory entries: {stats['agent_memory']}")
print(f"\n  Brief saved to: experiments/reports/")
print(f"  Open DAILY_BRIEF_*.md in VS Code to read the report")
print(f"{'='*55}")

db.close()