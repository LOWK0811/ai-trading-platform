# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.data_handler import get_price_data
from src.features import build_all_features, all_feature_cols
from src.features.momentum import MOMENTUM_FEATURES
from src.indicators import add_atr
from src.risk import calculate_shares
from src.experiment_tracker import ExperimentTracker
from src.research_journal import ResearchEngine


# ======================================================================
# SECTION 2: LOGGING
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
# SECTION 3: EXPERIMENT CONFIG
# ======================================================================
TICKERS     = ["AAPL", "NVDA", "SPY", "XOM"]
START       = "2021-01-01"
END         = "2026-06-01"
SPLIT_DATE  = "2024-01-01"

# The three feature sets we're comparing across all tickers
FEATURE_SETS = {
    "momentum_7":   MOMENTUM_FEATURES,      # 7 features, best so far
    "top5_selected":["trend_dist_sma50",     # 5 cross-category features
                     "vol_volume_momentum",
                     "mom_return_60d",
                     "mom_return_10d",
                     "mom_return_20d"],
    "full_27":       all_feature_cols(),     # all 27 features
}


# ======================================================================
# SECTION 4: SINGLE EXPERIMENT RUNNER
# ======================================================================
def run_experiment(ticker, feature_name, features,
                   start, end, split_date):
    """
    Runs a single ML experiment and returns structured results.
    No tracker logging — we collect all results first, then log.
    """
    raw  = get_price_data(ticker, start, end)
    if raw is None:
        return None
    raw  = add_atr(raw)
    df   = build_all_features(raw)
    df["label"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df   = df.dropna(subset=features + ["label"])

    if len(df) < 200:
        logger.warning(f"{ticker}/{feature_name}: "
                      f"insufficient data ({len(df)} rows)")
        return None

    train = df[df.index <  split_date]
    test  = df[df.index >= split_date]

    if len(train) < 100 or len(test) < 50:
        logger.warning(f"{ticker}/{feature_name}: "
                      f"insufficient split data")
        return None

    X_tr = train[features];  y_tr = train["label"]
    X_te = test[features];   y_te = test["label"]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=max(10, len(train)//20),
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_tr, y_tr)

    proba    = model.predict_proba(X_te)[:, 1]
    test_auc = round(roc_auc_score(y_te, proba), 4)

    # Feature importances
    importances = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values(ascending=False)

    # Backtest on test period
    bt_data = add_atr(raw).loc[test.index]
    signals = (proba >= 0.55).astype(int)

    cash = 10000; sh = 0; pv = []
    for i in range(len(bt_data)):
        pt  = bt_data["Close"].iloc[i]
        py  = bt_data["Close"].iloc[i-1] if i > 0 else pt
        atr = bt_data["atr"].iloc[i]
        sig = signals[i]
        if sig and sh == 0:
            s = calculate_shares(cash, py, atr)
            if s > 0: cash -= s*py*1.001; sh = s
        elif not sig and sh > 0:
            cash += sh*py*0.999; sh = 0
        pv.append(cash + sh*pt)

    s  = pd.Series(pv)
    dr = s.pct_change().dropna()
    ex = dr - 0.05/252
    sharpe = round((ex.mean()/ex.std())*np.sqrt(252), 3) \
             if ex.std() > 0 else 0.0

    return {
        "ticker":       ticker,
        "feature_set":  feature_name,
        "n_features":   len(features),
        "train_rows":   len(train),
        "test_rows":    len(test),
        "test_auc":     test_auc,
        "sharpe":       sharpe,
        "top_feature":  importances.index[0],
        "top3_features":importances.index[:3].tolist(),
        "importances":  importances,
        "portfolio":    pv,
    }


# ======================================================================
# SECTION 5: RUN ALL EXPERIMENTS
# ======================================================================
print(f"\n{'='*70}")
print(f"  AURELINE LABS — MULTI-TICKER ML VALIDATION")
print(f"{'='*70}")
print(f"  Tickers:      {TICKERS}")
print(f"  Feature sets: {list(FEATURE_SETS.keys())}")
print(f"  Train period: {START} → {SPLIT_DATE}")
print(f"  Test period:  {SPLIT_DATE} → {END}")
print(f"  Total runs:   {len(TICKERS) * len(FEATURE_SETS)}")
print(f"{'='*70}\n")

all_results = []

for ticker in TICKERS:
    for feat_name, features in FEATURE_SETS.items():
        logger.info(f"Running: {ticker} / {feat_name} "
                   f"({len(features)} features)")
        result = run_experiment(
            ticker, feat_name, features,
            START, END, SPLIT_DATE)
        if result:
            all_results.append(result)
            logger.info(f"  AUC: {result['test_auc']:.4f} | "
                       f"Sharpe: {result['sharpe']:.3f} | "
                       f"Top: {result['top_feature']}")

print(f"\nCompleted {len(all_results)} experiments")


# ======================================================================
# SECTION 6: RESULTS ANALYSIS
# ======================================================================
df_results = pd.DataFrame([{
    "Ticker":      r["ticker"],
    "Feature Set": r["feature_set"],
    "N Features":  r["n_features"],
    "Train Rows":  r["train_rows"],
    "ROC-AUC":     r["test_auc"],
    "Sharpe":      r["sharpe"],
    "Top Feature": r["top_feature"],
} for r in all_results])

print(f"\n{'='*80}")
print(f"  FULL RESULTS TABLE")
print(f"{'='*80}")
print(df_results.to_string(index=False))
print(f"{'='*80}")


# ======================================================================
# SECTION 7: KEY FINDINGS
# ======================================================================
print(f"\n{'='*80}")
print(f"  KEY FINDINGS")
print(f"{'='*80}")

# Finding 1: AUC by feature set across tickers
print(f"\n  1. AVERAGE ROC-AUC BY FEATURE SET:")
auc_by_set = df_results.groupby("Feature Set")["ROC-AUC"].agg(
    ["mean", "min", "max", "std"]).round(4)
print(auc_by_set.to_string())

# Finding 2: AUC by ticker
print(f"\n  2. AVERAGE ROC-AUC BY TICKER:")
auc_by_ticker = df_results.groupby("Ticker")["ROC-AUC"].agg(
    ["mean", "min", "max"]).round(4)
print(auc_by_ticker.to_string())

# Finding 3: Feature dominance
print(f"\n  3. TOP FEATURE FREQUENCY:")
top_feat_counts = df_results["Top Feature"].value_counts()
for feat, count in top_feat_counts.items():
    pct = count / len(df_results) * 100
    print(f"  {feat:<35} {count:>3} / {len(df_results)} "
          f"({pct:.0f}%)")

# Finding 4: Dimensionality curse check
small = df_results[df_results["N Features"] <= 7]["ROC-AUC"].mean()
large = df_results[df_results["N Features"] > 7]["ROC-AUC"].mean()
print(f"\n  4. CURSE OF DIMENSIONALITY (CROSS-TICKER):")
print(f"  ≤7 features:  mean AUC = {small:.4f}")
print(f"  >7 features:  mean AUC = {large:.4f}")
print(f"  Gap: {small - large:+.4f} "
      f"({'Confirmed' if small > large else 'NOT confirmed'})")

# Finding 5: Best single experiment
best = df_results.loc[df_results["ROC-AUC"].idxmax()]
print(f"\n  5. BEST SINGLE EXPERIMENT:")
print(f"  {best['Ticker']} / {best['Feature Set']} → "
      f"AUC {best['ROC-AUC']:.4f} | Sharpe {best['Sharpe']:.3f}")

# Finding 6: mom_return_20d dominance
mom20_dominant = (df_results["Top Feature"] == "mom_return_20d").sum()
print(f"\n  6. mom_return_20d as top feature: "
      f"{mom20_dominant}/{len(df_results)} experiments "
      f"({mom20_dominant/len(df_results):.0%})")

print(f"{'='*80}")


# ======================================================================
# SECTION 8: VISUALIZATION
# ======================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor("#060d1f")

def style(ax):
    ax.set_facecolor("#060d1f")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1a3357")
    ax.spines["bottom"].set_color("#1a3357")
    ax.tick_params(colors="#7b9bc0", labelsize=8)
    ax.yaxis.label.set_color("#7b9bc0")
    ax.xaxis.label.set_color("#7b9bc0")
    ax.grid(True, color="#0d1b35", linewidth=0.7, linestyle="--")

# ── Panel 1: AUC by ticker and feature set ──
ax1 = axes[0][0]; style(ax1)
feat_sets   = list(FEATURE_SETS.keys())
x           = np.arange(len(TICKERS))
width       = 0.25
colors_bar  = ["#00d4aa", "#1a6eff", "#ffd166"]

for fi, (fs, col) in enumerate(zip(feat_sets, colors_bar)):
    vals = [df_results[
                (df_results["Ticker"] == t) &
                (df_results["Feature Set"] == fs)
            ]["ROC-AUC"].values[0]
            if len(df_results[
                (df_results["Ticker"] == t) &
                (df_results["Feature Set"] == fs)
            ]) > 0 else 0
            for t in TICKERS]
    ax1.bar(x + fi*width, vals,
            width=width, color=col, alpha=0.8,
            label=fs)

ax1.axhline(y=0.5, color="#ff4d6a", linewidth=1,
            linestyle="--", alpha=0.7, label="Random (0.5)")
ax1.set_xticks(x + width)
ax1.set_xticklabels(TICKERS, color="#7b9bc0", fontsize=8)
ax1.set_ylabel("ROC-AUC", fontsize=8)
ax1.set_title("ROC-AUC by Ticker & Feature Set",
              color="#e8f0fe", fontsize=9,
              fontfamily="monospace")
ax1.legend(fontsize=7, facecolor="#0f2040",
           edgecolor="#1a3357", labelcolor="#7b9bc0")
ax1.set_ylim(0.44, 0.60)

# ── Panel 2: Feature importance heatmap ──
ax2 = axes[0][1]; ax2.set_facecolor("#060d1f")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_color("#1a3357")
ax2.spines["bottom"].set_color("#1a3357")
ax2.tick_params(colors="#7b9bc0", labelsize=7)

# Build importance matrix for momentum_7 across tickers
mom_results = [r for r in all_results
               if r["feature_set"] == "momentum_7"]
if mom_results:
    imp_data = pd.DataFrame({
        r["ticker"]: r["importances"]
        for r in mom_results
    }).fillna(0)

    im2 = ax2.imshow(imp_data.values,
                     cmap="YlOrRd", aspect="auto",
                     vmin=0, vmax=0.25)
    ax2.set_xticks(range(len(imp_data.columns)))
    ax2.set_yticks(range(len(imp_data.index)))
    ax2.set_xticklabels(imp_data.columns, fontsize=8)
    ax2.set_yticklabels(imp_data.index,   fontsize=7)
    for i in range(len(imp_data.index)):
        for j in range(len(imp_data.columns)):
            ax2.text(j, i,
                     f"{imp_data.values[i,j]:.3f}",
                     ha="center", va="center",
                     fontsize=6, color="#060d1f",
                     fontweight="bold")
    ax2.set_title("Feature Importances (Momentum-7 set)",
                  color="#e8f0fe", fontsize=9,
                  fontfamily="monospace")

# ── Panel 3: Portfolio curves (momentum_7) ──
ax3 = axes[1][0]; style(ax3)
ticker_colors = {"AAPL":"#00d4aa", "NVDA":"#1a6eff",
                 "SPY":"#ffd166",  "XOM":"#ff4d6a"}

for r in all_results:
    if r["feature_set"] == "momentum_7":
        norm = [v/10000 for v in r["portfolio"]]
        ax3.plot(norm,
                 color=ticker_colors.get(r["ticker"],"#7b9bc0"),
                 linewidth=1.5,
                 label=f"{r['ticker']} "
                       f"(AUC={r['test_auc']:.3f})")

ax3.axhline(y=1.0, color="#1a3357",
            linestyle="--", linewidth=0.8)
ax3.set_ylabel("Return Multiple (×)", fontsize=8)
ax3.set_xlabel("Test Period Days", fontsize=8)
ax3.set_title("Momentum-7 Strategy (Test Period Only)",
              color="#e8f0fe", fontsize=9,
              fontfamily="monospace")
ax3.legend(fontsize=7, facecolor="#0f2040",
           edgecolor="#1a3357", labelcolor="#7b9bc0")

# ── Panel 4: AUC vs n_features scatter ──
ax4 = axes[1][1]; style(ax4)
for r in all_results:
    col = ticker_colors.get(r["ticker"], "#7b9bc0")
    ax4.scatter(r["n_features"], r["test_auc"],
                color=col, s=80, alpha=0.8,
                zorder=5)
    ax4.annotate(
        f"{r['ticker'][:4]}\n{r['feature_set'][:5]}",
        (r["n_features"], r["test_auc"]),
        fontsize=5.5, color="#7b9bc0",
        textcoords="offset points", xytext=(4, 2)
    )

ax4.axhline(y=0.5, color="#ff4d6a", linewidth=1,
            linestyle="--", alpha=0.7)
ax4.set_xlabel("Number of Features", fontsize=8)
ax4.set_ylabel("ROC-AUC", fontsize=8)
ax4.set_title("Curse of Dimensionality — Cross-Ticker",
              color="#e8f0fe", fontsize=9,
              fontfamily="monospace")

# Trend line
x_pts = np.array([r["n_features"] for r in all_results])
y_pts = np.array([r["test_auc"]   for r in all_results])
z     = np.polyfit(x_pts, y_pts, 1)
p     = np.poly1d(z)
x_line= np.linspace(x_pts.min(), x_pts.max(), 100)
ax4.plot(x_line, p(x_line),
         color="#7b9bc0", linewidth=1,
         linestyle="--", alpha=0.5,
         label=f"Trend (slope={z[0]:.5f})")
ax4.legend(fontsize=7, facecolor="#0f2040",
           edgecolor="#1a3357", labelcolor="#7b9bc0")

fig.suptitle(
    "Aureline Labs — Multi-Ticker ML Validation\n"
    "4 Tickers × 3 Feature Sets × Test Period 2024–2026",
    color="#e8f0fe", fontsize=11,
    fontfamily="monospace", y=1.01
)
plt.tight_layout()
os.makedirs("data", exist_ok=True)
plt.savefig("data/multi_ticker_ml_validation.png",
            dpi=150, bbox_inches="tight",
            facecolor="#060d1f")
plt.show()
logger.info("Chart saved: data/multi_ticker_ml_validation.png")


# ======================================================================
# SECTION 9: LOG TO EXPERIMENT TRACKER
# ======================================================================
print(f"\n  Logging to experiment registry...")
tracker = ExperimentTracker()

for r in all_results:
    metrics = {
        "ROC-AUC":       r["test_auc"],
        "Sharpe":        r["sharpe"],
        "Top Feature":   r["top_feature"],
        "Train Rows":    r["train_rows"],
        "Test Rows":     r["test_rows"],
    }
    tracker.log(
        hypothesis  = (
            f"Does the {r['feature_set']} feature set "
            f"show predictive edge on {r['ticker']} "
            f"over the test period 2024-2026?"
        ),
        ticker      = r["ticker"],
        start       = START,
        end         = END,
        strategy    = "ML_RandomForest",
        parameters  = {
            "feature_set":    r["feature_set"],
            "n_features":     r["n_features"],
            "split_date":     SPLIT_DATE,
            "n_estimators":   100,
            "prob_threshold": 0.55,
        },
        features    = list(FEATURE_SETS[r["feature_set"]]),
        metrics     = metrics,
        conclusion  = (
            f"ROC-AUC {r['test_auc']:.4f} on {r['ticker']} "
            f"using {r['n_features']} features. "
            f"Top feature: {r['top_feature']}."
        ),
        tags = ["ml", "cross-ticker-validation",
                r["feature_set"], r["ticker"].lower()]
    )

print(f"  Logged {len(all_results)} experiments")
print(f"  Registry total: {len(tracker.registry)}")