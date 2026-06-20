# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from src.data_handler import get_price_data
from src.features import build_all_features, all_feature_cols
from src.indicators import add_atr
from src.risk import calculate_shares
from src.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 2: METRIC HELPERS
# ======================================================================
def _sharpe(portfolio, rfr=0.05):
    s = pd.Series(portfolio)
    dr = s.pct_change().dropna()
    ex = dr - rfr / 252
    return round((ex.mean() / ex.std()) * np.sqrt(252), 3) \
        if ex.std() > 0 else 0.0


def _mdd(portfolio):
    s = pd.Series(portfolio)
    return round(((s - s.cummax()) / s.cummax()).min() * 100, 2)


def _cagr(portfolio, start, end):
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    return round(((portfolio[-1] / portfolio[0]) ** (1/years) - 1)
                 * 100, 2) if years > 0 else 0.0


def _win_rate(portfolio):
    s = pd.Series(portfolio)
    return round((s.pct_change().dropna() > 0).mean() * 100, 1)


# ======================================================================
# SECTION 3: THE RESEARCH EXPERIMENT ENGINE
# ======================================================================
class ResearchEngine:
    """
    Runs a complete ML research experiment and produces a
    formatted research report. Logs everything to the
    Experiment Tracker registry.

    Workflow:
    1. Load data and build features
    2. Time-based train/test split
    3. Train Random Forest on training set
    4. Evaluate on held-out test set
    5. Run walk-forward backtest on test period
    6. Log to registry and generate report
    """

    def __init__(self, tracker=None):
        self.tracker = tracker or ExperimentTracker()


    # ======================================================================
    # SECTION 4: RUN A COMPLETE EXPERIMENT
    # ======================================================================
    def run(self,
            hypothesis,
            ticker,
            start,
            end,
            split_date,
            feature_subset=None,
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=20,
            prob_threshold=0.55,
            tags=None):
        """
        Runs a complete research experiment.

        Parameters
        ----------
        hypothesis     : str  — the testable claim
        ticker         : str  — asset to study
        start/end      : str  — full date range
        split_date     : str  — train/test boundary
        feature_subset : list — which features to use (None = all 27)
        n_estimators   : int  — RF trees
        max_depth      : int  — max tree depth
        min_samples_leaf: int — min samples per leaf
        prob_threshold : float — signal threshold (0.55 = 55% confident)
        tags           : list — experiment labels
        """
        logger.info(f"Starting experiment: {ticker} | {hypothesis[:50]}...")

        # ── Step 1: Load and build features ──
        raw_data = get_price_data(ticker, start, end)
        if raw_data is None:
            logger.error(f"No data for {ticker}")
            return None

        raw_data = add_atr(raw_data)
        df = build_all_features(raw_data)
        df["label"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

        features = feature_subset or all_feature_cols()
        df = df.dropna(subset=features + ["label"])

        logger.info(f"Features: {len(features)} | "
                   f"Clean rows: {len(df)}")

        # ── Step 2: Train/test split ──
        train = df[df.index < split_date]
        test  = df[df.index >= split_date]

        if len(train) < 200 or len(test) < 50:
            logger.error("Insufficient data for split")
            return None

        X_train = train[features]
        y_train = train["label"]
        X_test  = test[features]
        y_test  = test["label"]

        logger.info(f"Train: {len(train)} rows | "
                   f"Test: {len(test)} rows")

        # ── Step 3: Train model ──
        model = RandomForestClassifier(
            n_estimators    = n_estimators,
            max_depth       = max_depth,
            min_samples_leaf= min_samples_leaf,
            random_state    = 42,
            n_jobs          = -1
        )
        model.fit(X_train, y_train)
        logger.info("Model trained")

        # ── Step 4: Evaluate on test set ──
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc     = round(roc_auc_score(y_test, y_proba), 4)
        accuracy= round((y_pred == y_test).mean() * 100, 1)

        clf_report = classification_report(
            y_test, y_pred,
            target_names=["Down", "Up"],
            output_dict=True
        )

        # ── Step 5: Feature importances ──
        importances = pd.Series(
            model.feature_importances_,
            index=features
        ).sort_values(ascending=False)

        # ── Step 6: Walk-forward backtest on test period ──
        test_data   = raw_data.loc[test.index]
        test_data   = add_atr(test_data)
        signals_df  = pd.DataFrame(
            {"signal": (y_proba >= prob_threshold).astype(int)},
            index=test.index
        )

        cash        = 10000
        shares_held = 0
        portfolio   = []
        num_trades  = 0

        for i in range(len(test_data)):
            price_today     = test_data["Close"].iloc[i]
            price_yesterday = test_data["Close"].iloc[i-1] \
                              if i > 0 else price_today
            atr_today       = test_data["atr"].iloc[i]
            signal_today    = signals_df["signal"].iloc[i]

            if signal_today == 1 and shares_held == 0:
                shares = calculate_shares(
                    cash, price_yesterday, atr_today)
                if shares > 0:
                    cash -= shares * price_yesterday * 1.001
                    shares_held = shares
                    num_trades += 1
            elif signal_today == 0 and shares_held > 0:
                cash += shares_held * price_yesterday * 0.999
                shares_held = 0
                num_trades += 1

            portfolio.append(cash + shares_held * price_today)

        # Buy and hold over test period
        bh_shares = 10000 // test_data["Close"].iloc[0]
        bh_final  = bh_shares * test_data["Close"].iloc[-1]
        bh_return = round((bh_final / 10000 - 1) * 100, 2)

        # ── Step 7: Compile metrics ──
        metrics = {
            "Accuracy":         f"{accuracy}%",
            "ROC-AUC":          auc,
            "Sharpe":           _sharpe(portfolio),
            "CAGR":             f"{_cagr(portfolio, split_date, end)}%",
            "Max Drawdown":     f"{_mdd(portfolio)}%",
            "Win Rate":         f"{_win_rate(portfolio)}%",
            "Round Trips":      num_trades // 2,
            "Strategy Return":  f"{round((portfolio[-1]/10000-1)*100,2)}%",
            "B&H Return":       f"{bh_return}%",
            "Beat B&H":         "Yes" if portfolio[-1] > bh_final else "No",
            "Top Feature":      importances.index[0],
            "Train Rows":       len(train),
            "Test Rows":        len(test),
        }

        # ── Step 8: Generate conclusion ──
        beat    = portfolio[-1] > bh_final
        conclusion = self._generate_conclusion(
            ticker, metrics, importances, beat,
            prob_threshold, features
        )

        # ── Step 9: Log experiment ──
        parameters = {
            "n_estimators":     n_estimators,
            "max_depth":        max_depth,
            "min_samples_leaf": min_samples_leaf,
            "prob_threshold":   prob_threshold,
            "split_date":       split_date,
            "features_used":    len(features),
            "cost_per_trade":   "0.1%",
            "position_sizing":  "ATR 1%",
        }

        exp_id = self.tracker.log(
            hypothesis  = hypothesis,
            ticker      = ticker,
            start       = start,
            end         = end,
            strategy    = "ML_RandomForest",
            parameters  = parameters,
            features    = features,
            metrics     = metrics,
            conclusion  = conclusion,
            tags        = tags or ["ml", "random-forest"]
        )

        # ── Step 10: Write enhanced report ──
        self._write_enhanced_report(
            exp_id, hypothesis, ticker, start, end,
            split_date, parameters, features,
            importances, clf_report, auc, metrics,
            conclusion
        )

        logger.info(f"Experiment complete: {exp_id}")
        return {
            "exp_id":      exp_id,
            "metrics":     metrics,
            "importances": importances,
            "portfolio":   portfolio,
            "model":       model
        }


    # ======================================================================
    # SECTION 5: AUTO-GENERATE CONCLUSION
    # ======================================================================
    def _generate_conclusion(self, ticker, metrics,
                              importances, beat,
                              threshold, features):
        top3 = importances.head(3).index.tolist()
        auc  = float(metrics["ROC-AUC"])

        edge = ("modest but real" if 0.52 <= auc < 0.56
                else "meaningful" if 0.56 <= auc < 0.62
                else "strong" if auc >= 0.62
                else "no discernible")

        return (
            f"The Random Forest model demonstrated {edge} predictive "
            f"edge on {ticker} with a ROC-AUC of {auc} on the held-out "
            f"test set. The three most important features were "
            f"{top3[0]}, {top3[1]}, and {top3[2]}, suggesting the model "
            f"primarily captured "
            f"{'momentum and trend signals' if any('mom' in f or 'trend' in f for f in top3) else 'volatility and regime signals'}. "
            f"The strategy {'outperformed' if beat else 'underperformed'} "
            f"buy-and-hold on the test period with a return of "
            f"{metrics['Strategy Return']} vs {metrics['B&H Return']}. "
            f"Sharpe ratio of {metrics['Sharpe']} suggests "
            f"{'acceptable' if float(str(metrics['Sharpe'])) > 0.5 else 'weak'} "
            f"risk-adjusted performance. "
            f"These results should be validated across additional "
            f"tickers and time periods before drawing broader conclusions."
        )


    # ======================================================================
    # SECTION 6: WRITE ENHANCED MARKDOWN REPORT
    # ======================================================================
    def _write_enhanced_report(self, exp_id, hypothesis,
                                ticker, start, end,
                                split_date, parameters,
                                features, importances,
                                clf_report, auc, metrics,
                                conclusion):
        date_str  = datetime.now().strftime("%Y-%m-%d")
        filename  = f"experiments/reports/{exp_id}.md"
        os.makedirs("experiments/reports", exist_ok=True)

        # Feature importance table (top 10)
        top10 = importances.head(10)
        importance_table = "\n".join([
            f"| {i+1} | `{feat}` | {score:.4f} |"
            for i, (feat, score) in enumerate(top10.items())
        ])

        # Classification metrics
        up   = clf_report.get("Up",   {})
        down = clf_report.get("Down", {})

        report = f"""# {exp_id}
## {ticker} · ML Research Report · {date_str}

---

### Hypothesis

> {hypothesis}

---

### Methodology

| Parameter | Value |
|-----------|-------|
| Asset | {ticker} |
| Study Period | {start} → {end} |
| Training Period | {start} → {split_date} |
| Test Period | {split_date} → {end} |
| Training Rows | {parameters.get('Train Rows', metrics.get('Train Rows', 'N/A'))} |
| Test Rows | {parameters.get('Test Rows', metrics.get('Test Rows', 'N/A'))} |
| Model | Random Forest |
| Trees | {parameters['n_estimators']} |
| Max Depth | {parameters['max_depth']} |
| Min Samples Leaf | {parameters['min_samples_leaf']} |
| Signal Threshold | {parameters['prob_threshold']} |
| Features Used | {len(features)} |
| Position Sizing | ATR 1% |
| Transaction Cost | 0.1% per trade |

---

### Feature Importance (Top 10)

| Rank | Feature | Importance Score |
|------|---------|-----------------|
{importance_table}

---

### Classification Performance (Test Set)

| Metric | Up Days | Down Days |
|--------|---------|-----------|
| Precision | {up.get('precision', 0):.3f} | {down.get('precision', 0):.3f} |
| Recall | {up.get('recall', 0):.3f} | {down.get('recall', 0):.3f} |
| F1-Score | {up.get('f1-score', 0):.3f} | {down.get('f1-score', 0):.3f} |
| ROC-AUC | {auc} | — |

---

### Portfolio Performance (Test Period)

| Metric | Value |
|--------|-------|
| Strategy Return | {metrics['Strategy Return']} |
| Buy & Hold Return | {metrics['B&H Return']} |
| Beat Buy & Hold | {metrics['Beat B&H']} |
| CAGR | {metrics['CAGR']} |
| Sharpe Ratio | {metrics['Sharpe']} |
| Max Drawdown | {metrics['Max Drawdown']} |
| Win Rate | {metrics['Win Rate']} |
| Round Trips | {metrics['Round Trips']} |

---

### Conclusion

{conclusion}

---

### Next Steps

- [ ] Validate on additional tickers (MSFT, NVDA, SPY)
- [ ] Test with different feature subsets
- [ ] Compare against SMA baseline on same test period
- [ ] Run Monte Carlo simulation on strategy returns

---

*Aureline Labs · Quantitative Research & Intelligence Platform*
*Ateneo de Manila University · Applied Mathematics · Mathematical Finance*
"""

        with open(filename, "w") as f:
            f.write(report)

        logger.info(f"Enhanced report written: {filename}")