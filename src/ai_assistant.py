# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import re
import json
import logging

from src.features.momentum   import MOMENTUM_FEATURES
from src.features.volatility import VOLATILITY_FEATURES
from src.features.trend      import TREND_FEATURES
from src.features.volume     import VOLUME_FEATURES
from src.features.regime     import REGIME_FEATURES

logger = logging.getLogger(__name__)

ALL_FEATURES = (MOMENTUM_FEATURES + VOLATILITY_FEATURES +
                TREND_FEATURES + VOLUME_FEATURES + REGIME_FEATURES)


# ======================================================================
# SECTION 2: CONCEPT KEYWORD MAP
# ======================================================================
# Maps research concepts to the features that measure them.
# This is the core "understanding" layer — each concept has
# a set of keywords that trigger it, and a set of features
# that test it.

CONCEPT_MAP = {
    "mean_reversion": {
        "keywords": [
            "bounce", "revert", "recover", "mean reversion",
            "fallen sharply", "oversold", "snap back",
            "reversal", "overreaction", "too far down"
        ],
        "features": [
            "mom_return_1d", "mom_return_5d", "mom_return_20d",
            "mom_zscore_20d", "mom_rsi_14",
            "regime_drawdown_pct"
        ],
        "hypothesis_template": (
            "Assets that have experienced significant negative "
            "price momentum show statistically meaningful "
            "positive mean-reversion in subsequent periods."
        ),
        "expected": (
            "Mean-reversion effects are notoriously weak in "
            "daily equity data. ROC-AUC likely near 0.50. "
            "RSI and z-score are the most relevant features."
        )
    },

    "momentum": {
        "keywords": [
            "momentum", "trend", "continue", "keep going",
            "upward", "going up", "rising", "bull run",
            "trend following", "continuation", "breakout"
        ],
        "features": [
            "mom_return_5d", "mom_return_20d", "mom_return_60d",
            "trend_dist_sma20", "trend_sma20_slope",
            "trend_above_sma200"
        ],
        "hypothesis_template": (
            "Assets with positive price momentum over multiple "
            "timeframes continue to outperform in subsequent periods, "
            "consistent with the momentum anomaly documented "
            "in academic literature."
        ),
        "expected": (
            "Momentum is one of the most robust factors in "
            "equity markets. Expect ROC-AUC modestly above 0.50, "
            "with longer-window returns (20d, 60d) dominating "
            "feature importance."
        )
    },

    "volatility": {
        "keywords": [
            "volatility", "volatile", "calm", "quiet",
            "vix", "uncertainty", "risk", "stable",
            "low vol", "high vol", "fear"
        ],
        "features": [
            "vol_atr_pct", "vol_realized_10d", "vol_realized_20d",
            "vol_rank_52w", "vol_ratio",
            "regime_volatility_state"
        ],
        "hypothesis_template": (
            "Current volatility regime (high vs low realized "
            "volatility relative to historical norms) has "
            "predictive power for next-day return direction."
        ),
        "expected": (
            "Low volatility environments tend to persist "
            "(volatility clustering). The vol_rank_52w and "
            "vol_ratio features likely show highest importance."
        )
    },

    "volume": {
        "keywords": [
            "volume", "trading activity", "interest",
            "turnover", "participation", "buyers", "sellers",
            "conviction", "accumulation", "distribution"
        ],
        "features": [
            "vol_relative_volume", "vol_volume_spike",
            "vol_volume_momentum", "vol_price_volume_trend",
            "mom_return_1d", "mom_return_5d"
        ],
        "hypothesis_template": (
            "Abnormal trading volume signals informed trading "
            "activity and predicts the direction of subsequent "
            "price moves."
        ),
        "expected": (
            "Volume signals are noisy at the daily level. "
            "Volume spikes on down days may predict continuation "
            "rather than reversal. Expect modest AUC."
        )
    },

    "regime": {
        "keywords": [
            "regime", "bull market", "bear market", "sideways",
            "environment", "market condition", "trend strength",
            "market state", "macro", "drawdown"
        ],
        "features": [
            "regime_bull_bear", "regime_volatility_state",
            "regime_trend_strength", "regime_drawdown_pct",
            "trend_above_sma200", "mom_return_60d"
        ],
        "hypothesis_template": (
            "The current market regime (bull/bear/sideways, "
            "high/low volatility) has independent predictive "
            "power for next-day returns beyond price momentum alone."
        ),
        "expected": (
            "Regime features should add meaningful signal, "
            "especially regime_bull_bear and trend_above_sma200 "
            "which capture the broad market environment."
        )
    },

    "trend": {
        "keywords": [
            "moving average", "sma", "ema", "above average",
            "below average", "distance", "crossover",
            "golden cross", "death cross", "trend strength"
        ],
        "features": [
            "trend_dist_sma20", "trend_dist_sma50",
            "trend_dist_ema20", "trend_sma20_slope",
            "trend_above_sma200", "mom_return_20d"
        ],
        "hypothesis_template": (
            "The distance of price from its moving averages "
            "and the slope of those averages contain predictive "
            "information about next-day return direction."
        ),
        "expected": (
            "Trend features showed moderate importance in prior "
            "experiments. Slope of SMA and distance from SMA50 "
            "likely dominate."
        )
    }
}


# ======================================================================
# SECTION 3: TICKER EXTRACTOR
# ======================================================================
KNOWN_TICKERS = {
    "apple": "AAPL", "aapl": "AAPL",
    "microsoft": "MSFT", "msft": "MSFT",
    "nvidia": "NVDA", "nvda": "NVDA",
    "tesla": "TSLA", "tsla": "TSLA",
    "amazon": "AMZN", "amzn": "AMZN",
    "google": "GOOGL", "alphabet": "GOOGL",
    "meta": "META", "facebook": "META",
    "jpmorgan": "JPM", "jpm": "JPM",
    "exxon": "XOM", "xom": "XOM",
    "spy": "SPY", "s&p": "SPY", "s&p 500": "SPY",
    "jnj": "JNJ", "johnson": "JNJ",
}


def extract_ticker(query):
    """Extracts ticker symbol from natural language query."""
    query_lower = query.lower()
    for name, ticker in KNOWN_TICKERS.items():
        if name in query_lower:
            return ticker
    # Look for uppercase ticker patterns (e.g. "on AAPL")
    matches = re.findall(r'\b([A-Z]{2,5})\b', query)
    if matches:
        return matches[0]
    return "AAPL"


# ======================================================================
# SECTION 4: RULE-BASED AI RESEARCH ASSISTANT
# ======================================================================
class AIResearchAssistant:
    """
    Rule-based hypothesis interpreter for the Aureline Labs
    research pipeline.

    Interprets natural language research questions by matching
    keywords to known research concepts, selecting the appropriate
    feature subsets, and generating structured experiment specs.

    This approach is free, fast, interpretable, and covers the
    core research questions relevant to equity markets.
    """

    def __init__(self):
        logger.info("AI Research Assistant initialized "
                   "(rule-based interpreter)")


    # ======================================================================
    # SECTION 5: INTERPRET HYPOTHESIS
    # ======================================================================
    def interpret_hypothesis(self, user_query):
        """
        Interprets a natural language hypothesis and returns
        a structured experiment specification.
        """
        query_lower = user_query.lower()

        # Score each concept by keyword matches
        scores = {}
        for concept, config in CONCEPT_MAP.items():
            score = sum(
                1 for kw in config["keywords"]
                if kw in query_lower
            )
            scores[concept] = score

        # Pick the top 1-2 matching concepts
        ranked = sorted(scores.items(),
                        key=lambda x: x[1], reverse=True)
        top_concepts = [c for c, s in ranked if s > 0][:2]

        if not top_concepts:
            # Default to momentum if nothing matches
            top_concepts = ["momentum"]
            logger.info("No specific concept matched — "
                       "defaulting to momentum")
        else:
            logger.info(f"Concepts identified: {top_concepts}")

        # Build feature subset from matched concepts
        features = []
        for concept in top_concepts:
            features.extend(CONCEPT_MAP[concept]["features"])
        features = list(dict.fromkeys(features))  # deduplicate, preserve order

        # Extract ticker
        ticker = extract_ticker(user_query)

        # Build hypothesis from templates
        if len(top_concepts) == 1:
            hypothesis = CONCEPT_MAP[top_concepts[0]]["hypothesis_template"]
            expected   = CONCEPT_MAP[top_concepts[0]]["expected"]
        else:
            hypothesis = (
                CONCEPT_MAP[top_concepts[0]]["hypothesis_template"] +
                " This is examined in conjunction with " +
                top_concepts[1] + " signals."
            )
            expected = CONCEPT_MAP[top_concepts[0]]["expected"]

        spec = {
            "interpreted_hypothesis": hypothesis,
            "rationale": (
                f"Query matched concepts: {', '.join(top_concepts)}. "
                f"Selected {len(features)} features from the relevant "
                f"feature modules."
            ),
            "ticker":          ticker,
            "start":           "2021-01-01",
            "end":             "2026-06-01",
            "split_date":      "2024-01-01",
            "feature_subset":  features,
            "n_estimators":    100,
            "max_depth":       4,
            "min_samples_leaf":20,
            "prob_threshold":  0.55,
            "tags":            top_concepts + [ticker.lower()],
            "expected_finding": expected
        }

        return spec


    # ======================================================================
    # SECTION 6: INTERPRET RESULTS
    # ======================================================================
    def interpret_results(self, hypothesis, spec,
                          metrics, importances):
        """
        Generates a rule-based research conclusion from
        experiment results.
        """
        auc      = float(metrics.get("ROC-AUC", 0.5))
        sharpe   = float(str(metrics.get("Sharpe", 0)))
        beat     = metrics.get("Beat B&H", "No") == "Yes"
        top_feat = importances.index[0] if len(importances) > 0 \
                   else "unknown"
        ret      = metrics.get("Strategy Return", "N/A")

        # Interpret AUC
        if auc >= 0.56:
            signal_str = (f"meaningful predictive signal "
                         f"(ROC-AUC: {auc})")
        elif auc >= 0.52:
            signal_str = (f"modest but real predictive signal "
                         f"(ROC-AUC: {auc})")
        elif auc >= 0.50:
            signal_str = (f"weak signal barely above random "
                         f"(ROC-AUC: {auc})")
        else:
            signal_str = (f"no meaningful predictive signal "
                         f"(ROC-AUC: {auc}, below random chance)")

        # Interpret Sharpe
        if sharpe >= 1.0:
            sharpe_str = f"strong risk-adjusted returns (Sharpe: {sharpe})"
        elif sharpe >= 0.5:
            sharpe_str = f"acceptable risk-adjusted returns (Sharpe: {sharpe})"
        elif sharpe >= 0.0:
            sharpe_str = f"weak risk-adjusted returns (Sharpe: {sharpe})"
        else:
            sharpe_str = (f"negative risk-adjusted returns "
                         f"(Sharpe: {sharpe}) — strategy destroyed value")

        # Interpret top feature category
        if top_feat.startswith("mom_"):
            mech = "price momentum"
        elif top_feat.startswith("vol_"):
            mech = "volatility dynamics"
        elif top_feat.startswith("trend_"):
            mech = "trend positioning"
        elif top_feat.startswith("regime_"):
            mech = "market regime"
        else:
            mech = "volume behavior"

        bh_str = ("outperformed" if beat else "underperformed")

        conclusion = (
            f"The experiment found {signal_str}. "
            f"The most important feature was '{top_feat}', suggesting "
            f"the primary mechanism operates through {mech}. "
            f"The strategy produced {sharpe_str} and {bh_str} "
            f"buy-and-hold with a return of {ret}. "
            f"These results {'support' if auc >= 0.52 else 'do not support'} "
            f"the hypothesis that this pattern contains an exploitable edge. "
            f"Recommended next step: validate these features on NVDA and SPY "
            f"to test whether the finding is asset-specific or generalizable."
        )

        return conclusion


    # ======================================================================
    # SECTION 7: FULL RESEARCH PIPELINE
    # ======================================================================
    def research(self, user_query, engine):
        """
        End-to-end research pipeline:
        1. Interpret hypothesis
        2. Run experiment
        3. Interpret results
        4. Return full research output
        """
        print(f"\n{'='*60}")
        print(f"  AURELINE LABS — AI RESEARCH ASSISTANT")
        print(f"{'='*60}")
        print(f"\n  Query: {user_query}")
        print(f"\n  Interpreting hypothesis...")

        spec = self.interpret_hypothesis(user_query)

        print(f"\n  Interpreted as:")
        print(f"  {spec['interpreted_hypothesis']}")
        print(f"\n  Concepts matched: "
              f"{', '.join(spec['tags'])}")
        print(f"\n  Features selected "
              f"({len(spec['feature_subset'])}):")
        for feat in spec["feature_subset"]:
            print(f"    · {feat}")
        print(f"\n  Expected finding:")
        print(f"  {spec['expected_finding']}")
        print(f"\n  Running experiment on {spec['ticker']}...")

        result = engine.run(
            hypothesis       = spec["interpreted_hypothesis"],
            ticker           = spec["ticker"],
            start            = spec["start"],
            end              = spec["end"],
            split_date       = spec["split_date"],
            feature_subset   = spec["feature_subset"],
            n_estimators     = spec["n_estimators"],
            max_depth        = spec["max_depth"],
            min_samples_leaf = spec["min_samples_leaf"],
            prob_threshold   = spec["prob_threshold"],
            tags             = spec["tags"] + ["ai-generated"]
        )

        if not result:
            print("  Experiment failed — insufficient data.")
            return None

        conclusion = self.interpret_results(
            user_query, spec,
            result["metrics"],
            result["importances"]
        )

        print(f"\n{'='*60}")
        print(f"  RESULTS — {result['exp_id']}")
        print(f"{'='*60}")
        print(f"  ROC-AUC:  {result['metrics']['ROC-AUC']}")
        print(f"  Sharpe:   {result['metrics']['Sharpe']}")
        print(f"  Return:   {result['metrics']['Strategy Return']}")
        print(f"  Beat B&H: {result['metrics']['Beat B&H']}")
        print(f"\n  Top 3 features:")
        for feat, score in result["importances"].head(3).items():
            print(f"    · {feat}: {score:.4f}")
        print(f"\n{'='*60}")
        print(f"  AI RESEARCH CONCLUSION")
        print(f"{'='*60}")
        print(f"\n{conclusion}")
        print(f"\n{'='*60}")

        return {
            "spec":       spec,
            "result":     result,
            "conclusion": conclusion
        }