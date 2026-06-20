# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import json
import hashlib
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 2: EXPERIMENT REGISTRY
# ======================================================================
class ExperimentTracker:
    """
    Records every backtest as a structured experiment with a unique ID.
    Stores JSON records for machine querying and Markdown reports
    for human reading. This is Aureline Labs' institutional memory.
    """

    def __init__(self,
                 registry_path="experiments/registry.json",
                 reports_path="experiments/reports"):
        self.registry_path = registry_path
        self.reports_path  = reports_path
        os.makedirs("experiments/reports", exist_ok=True)
        self._load_registry()

    def _load_registry(self):
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r") as f:
                self.registry = json.load(f)
            logger.info(f"Registry loaded: "
                       f"{len(self.registry)} experiments")
        else:
            self.registry = []
            logger.info("New experiment registry created")

    def _save_registry(self):
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)


    # ======================================================================
    # SECTION 3: LOG AN EXPERIMENT
    # ======================================================================
    def log(self,
            hypothesis,
            ticker,
            start,
            end,
            strategy,
            parameters,
            features,
            metrics,
            conclusion,
            tags=None):
        """
        Log a complete experiment to the registry.

        Parameters
        ----------
        hypothesis  : str  — the testable claim being investigated
        ticker      : str  — asset under study
        start/end   : str  — date range
        strategy    : str  — "SMA", "ML_RandomForest", etc.
        parameters  : dict — all configurable settings used
        features    : list — feature names (for ML experiments)
        metrics     : dict — Sharpe, CAGR, Max DD, Win Rate, etc.
        conclusion  : str  — what the result actually means
        tags        : list — optional labels e.g. ["momentum", "regime"]
        """
        timestamp = datetime.now().isoformat()

        # Generate a short unique ID from content + timestamp
        content   = f"{hypothesis}{ticker}{timestamp}"
        exp_id    = "EXP-" + hashlib.md5(
            content.encode()).hexdigest()[:6].upper()

        experiment = {
            "id":          exp_id,
            "timestamp":   timestamp,
            "hypothesis":  hypothesis,
            "ticker":      ticker,
            "start":       start,
            "end":         end,
            "strategy":    strategy,
            "parameters":  parameters,
            "features":    features or [],
            "metrics":     metrics,
            "conclusion":  conclusion,
            "tags":        tags or []
        }

        self.registry.append(experiment)
        self._save_registry()
        self._write_report(experiment)

        logger.info(f"Experiment logged: {exp_id} — {strategy} on {ticker}")
        return exp_id


    # ======================================================================
    # SECTION 4: GENERATE MARKDOWN REPORT
    # ======================================================================
    def _write_report(self, exp):
        """Writes a human-readable Markdown research report."""
        date_str = exp["timestamp"][:10]
        filename = f"{self.reports_path}/{exp['id']}.md"

        metrics_table = "\n".join([
            f"| {k} | {v} |"
            for k, v in exp["metrics"].items()
        ])

        params_table = "\n".join([
            f"| {k} | {v} |"
            for k, v in exp["parameters"].items()
        ])

        features_list = "\n".join([
            f"- `{f}`" for f in exp["features"]
        ]) if exp["features"] else "_No features (rule-based strategy)_"

        tags_str = (", ".join([f"`{t}`" for t in exp["tags"]])
                    if exp["tags"] else "_None_")

        report = f"""# {exp['id']} — Aureline Labs Research Report

**Date:** {date_str}
**Strategy:** {exp['strategy']}
**Asset:** {exp['ticker']}
**Period:** {exp['start']} → {exp['end']}
**Tags:** {tags_str}

---

## Hypothesis

> {exp['hypothesis']}

---

## Parameters

| Parameter | Value |
|-----------|-------|
{params_table}

---

## Features Used

{features_list}

---

## Results

| Metric | Value |
|--------|-------|
{metrics_table}

---

## Conclusion

{exp['conclusion']}

---

_Aureline Labs · Quantitative Research & Intelligence Platform_
_Ateneo de Manila University · Applied Mathematics · Mathematical Finance_
"""

        with open(filename, "w") as f:
            f.write(report)

        logger.info(f"Report written: {filename}")


    # ======================================================================
    # SECTION 5: QUERY AND COMPARE
    # ======================================================================
    def get_all(self):
        """Returns all experiments as a DataFrame."""
        if not self.registry:
            return pd.DataFrame()
        rows = []
        for exp in self.registry:
            row = {
                "ID":         exp["id"],
                "Date":       exp["timestamp"][:10],
                "Ticker":     exp["ticker"],
                "Strategy":   exp["strategy"],
                "Hypothesis": exp["hypothesis"][:60] + "..."
                              if len(exp["hypothesis"]) > 60
                              else exp["hypothesis"],
            }
            row.update(exp["metrics"])
            rows.append(row)
        return pd.DataFrame(rows)

    def get_by_ticker(self, ticker):
        """Returns all experiments for a specific ticker."""
        return [e for e in self.registry
                if e["ticker"] == ticker.upper()]

    def get_by_tag(self, tag):
        """Returns all experiments with a specific tag."""
        return [e for e in self.registry
                if tag in e.get("tags", [])]

    def get_best(self, metric="Sharpe", n=5):
        """Returns the top N experiments by a given metric."""
        scored = [e for e in self.registry
                  if metric in e.get("metrics", {})]
        scored.sort(
            key=lambda e: float(str(e["metrics"][metric])
                                .replace("%","").replace("+","")),
            reverse=True
        )
        return scored[:n]

    def summary(self):
        """Prints a quick summary of the registry."""
        if not self.registry:
            print("No experiments logged yet.")
            return
        df = self.get_all()
        print(f"\n{'='*60}")
        print(f"  AURELINE LABS — EXPERIMENT REGISTRY")
        print(f"{'='*60}")
        print(f"  Total experiments: {len(self.registry)}")
        tickers  = list(set(e["ticker"]
                            for e in self.registry))
        strategies = list(set(e["strategy"]
                              for e in self.registry))
        print(f"  Tickers studied:   {', '.join(tickers)}")
        print(f"  Strategies used:   {', '.join(strategies)}")
        print(f"{'='*60}")