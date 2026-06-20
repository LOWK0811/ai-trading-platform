# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
from src.research_journal import ResearchEngine
from src.experiment_tracker import ExperimentTracker

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
# SECTION 3: RUN RESEARCH EXPERIMENTS
# ======================================================================
tracker = ExperimentTracker()
engine  = ResearchEngine(tracker)

# Experiment 1 — Full feature set on AAPL
logger.info("Running Experiment 1: AAPL full feature set")
result1 = engine.run(
    hypothesis = (
        "A Random Forest trained on 27 features spanning momentum, "
        "volatility, trend, volume, and regime signals can predict "
        "next-day direction on AAPL with statistically meaningful "
        "accuracy above 50% on held-out data."
    ),
    ticker      = "AAPL",
    start       = "2021-01-01",
    end         = "2026-06-01",
    split_date  = "2024-01-01",
    n_estimators= 100,
    max_depth   = 4,
    min_samples_leaf = 20,
    prob_threshold   = 0.55,
    tags        = ["ml", "random-forest", "full-features", "aapl"]
)

if result1:
    print(f"\nExperiment 1 complete: {result1['exp_id']}")
    print(f"ROC-AUC:  {result1['metrics']['ROC-AUC']}")
    print(f"Sharpe:   {result1['metrics']['Sharpe']}")
    print(f"Return:   {result1['metrics']['Strategy Return']}")
    print(f"Beat B&H: {result1['metrics']['Beat B&H']}")
    print(f"\nTop 5 features:")
    print(result1["importances"].head(5).to_string())


# Experiment 2 — Momentum features only
logger.info("\nRunning Experiment 2: AAPL momentum features only")

from src.features.momentum import MOMENTUM_FEATURES

result2 = engine.run(
    hypothesis = (
        "Momentum features alone (multi-window returns, z-score, RSI) "
        "are sufficient to generate a predictive edge on AAPL, without "
        "requiring volatility, trend, or regime information."
    ),
    ticker           = "AAPL",
    start            = "2021-01-01",
    end              = "2026-06-01",
    split_date       = "2024-01-01",
    feature_subset   = MOMENTUM_FEATURES,
    n_estimators     = 100,
    max_depth        = 4,
    min_samples_leaf = 20,
    prob_threshold   = 0.55,
    tags        = ["ml", "random-forest", "momentum-only", "aapl"]
)

if result2:
    print(f"\nExperiment 2 complete: {result2['exp_id']}")
    print(f"ROC-AUC:  {result2['metrics']['ROC-AUC']}")
    print(f"Sharpe:   {result2['metrics']['Sharpe']}")
    print(f"Return:   {result2['metrics']['Strategy Return']}")


# ======================================================================
# SECTION 4: COMPARE EXPERIMENTS
# ======================================================================
print(f"\n{'='*60}")
print(f"  AURELINE LABS — RESEARCH JOURNAL SUMMARY")
print(f"{'='*60}")

all_exp = tracker.get_all()
ml_exp  = all_exp[all_exp["Strategy"] == "ML_RandomForest"] \
          if len(all_exp) > 0 else all_exp

if len(ml_exp) > 0:
    print(ml_exp[["ID", "Date", "Ticker",
                  "Strategy", "ROC-AUC",
                  "Sharpe", "Strategy Return",
                  "Beat B&H"]].to_string(index=False))

print(f"\nReports saved to: experiments/reports/")
print(f"Open any .md file in VS Code to read the full report.")
# ======================================================================
# SECTION 5: FOLLOW-UP — TOP 5 FEATURES ONLY
# ======================================================================
logger.info("\nRunning Experiment 3: Top 5 features from full model")

top5_features = [
    "trend_dist_sma50",
    "vol_volume_momentum",
    "mom_return_60d",
    "mom_return_10d",
    "mom_return_20d"
]

result3 = engine.run(
    hypothesis = (
        "The five highest-importance features identified in EXP-6F9682 "
        "(trend_dist_sma50, vol_volume_momentum, mom_return_60d, "
        "mom_return_10d, mom_return_20d) contain the majority of "
        "predictive signal. A model trained on these five features alone "
        "should outperform the full 27-feature model due to reduced noise "
        "and higher training rows per feature."
    ),
    ticker           = "AAPL",
    start            = "2021-01-01",
    end              = "2026-06-01",
    split_date       = "2024-01-01",
    feature_subset   = top5_features,
    n_estimators     = 100,
    max_depth        = 4,
    min_samples_leaf = 20,
    prob_threshold   = 0.55,
    tags        = ["ml", "feature-selection", "top5", "aapl"]
)

if result3:
    print(f"\nExperiment 3 complete: {result3['exp_id']}")
    print(f"ROC-AUC:  {result3['metrics']['ROC-AUC']}")
    print(f"Sharpe:   {result3['metrics']['Sharpe']}")
    print(f"Return:   {result3['metrics']['Strategy Return']}")
    print(f"Beat B&H: {result3['metrics']['Beat B&H']}")

# ======================================================================
# SECTION 6: FINAL COMPARISON TABLE
# ======================================================================
print(f"\n{'='*70}")
print(f"  AURELINE LABS — ML EXPERIMENT COMPARISON")
print(f"{'='*70}")
print(f"{'Experiment':<12} {'Features':<12} {'Train Rows':<12} "
      f"{'ROC-AUC':<10} {'Sharpe':<10} {'Return'}")
print(f"{'-'*70}")

experiments_summary = [
    ("EXP-6F9682", "All 27",    482, 0.4828, -1.577, "-9.98%"),
    ("EXP-5BA75C", "Momentum",  693, 0.5450, -0.569, "+2.72%"),
    ("EXP-???",    "Top 5",     "?",
     result3["metrics"]["ROC-AUC"] if result3 else "N/A",
     result3["metrics"]["Sharpe"]  if result3 else "N/A",
     result3["metrics"]["Strategy Return"] if result3 else "N/A"),
]

for row in experiments_summary:
    print(f"{str(row[0]):<12} {str(row[1]):<12} {str(row[2]):<12} "
          f"{str(row[3]):<10} {str(row[4]):<10} {row[5]}")