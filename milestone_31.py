# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
from pathlib import Path
from src.ai_assistant import AIResearchAssistant
from src.research_journal import ResearchEngine
from src.experiment_tracker import ExperimentTracker


def load_dotenv(dotenv_path=".env"):
    env_path = Path(dotenv_path)
    if not env_path.exists():
        return False

    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    return True

load_dotenv()


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
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 3: INITIALIZE COMPONENTS
# ======================================================================
tracker   = ExperimentTracker()
engine    = ResearchEngine(tracker)
assistant = AIResearchAssistant()


# ======================================================================
# SECTION 4: RUN THREE AI-GENERATED EXPERIMENTS
# ======================================================================

# Research Question 1 — Mean reversion
q1 = assistant.research(
    "Do stocks that have fallen sharply over the past month "
    "tend to bounce back the following days, suggesting a "
    "mean-reversion pattern I could trade?",
    engine
)

# Research Question 2 — Volatility regime
q2 = assistant.research(
    "Does low volatility predict positive returns? "
    "I want to know if calm markets tend to continue "
    "going up on NVDA.",
    engine
)

# Research Question 3 — Trend following
q3 = assistant.research(
    "Is there momentum in AAPL? Do stocks that have been "
    "going up for the past 2-3 months tend to continue "
    "rising in the following weeks?",
    engine
)

print("\nAll experiments complete.")
print(f"Reports saved to experiments/reports/")
print(f"Registry updated with {len(tracker.registry)} total experiments")