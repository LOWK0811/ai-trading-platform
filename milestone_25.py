# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
import pandas as pd
from src.data_handler import get_price_data
from src.features import build_all_features, all_feature_cols

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
# SECTION 3: RUN THE FACTORY
# ======================================================================
data = get_price_data("AAPL", "2021-01-01", "2026-06-01")
df   = build_all_features(data)
df   = df.dropna(subset=all_feature_cols())

feature_list = all_feature_cols()

print(f"\n{'='*60}")
print(f"  AURELINE LABS — FEATURE FACTORY")
print(f"{'='*60}")
print(f"  Total features built: {len(feature_list)}")
print(f"  Clean rows:           {len(df)}")
print(f"  Date range:           {df.index.min().date()} "
      f"→ {df.index.max().date()}")
print(f"{'='*60}")

# ── Print by module ──
modules = {
    "MOMENTUM":   [f for f in feature_list if f.startswith("mom_")],
    "VOLATILITY": [f for f in feature_list if f.startswith("vol_")],
    "TREND":      [f for f in feature_list if f.startswith("trend_")],
    "VOLUME":     [f for f in feature_list if f.startswith("vol_rel")
                   or f.startswith("vol_volume")
                   or f.startswith("vol_price")],
    "REGIME":     [f for f in feature_list if f.startswith("regime_")],
}

for module, features in modules.items():
    print(f"\n  {module} ({len(features)} features)")
    for f in features:
        latest = df[f].iloc[-1]
        print(f"    {f:<35} {latest:+.4f}")

print(f"\n{'='*60}")
print(f"  Sample — last 3 rows of feature matrix:")
print(f"{'='*60}")
print(df[feature_list].tail(3).T.to_string())