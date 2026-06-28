# EXP-834AC8 — Aureline Labs Research Report

**Date:** 2026-06-28
**Strategy:** ML_RandomForest
**Asset:** SPY
**Period:** 2021-01-01 → 2026-06-01
**Tags:** `ml`, `cross-ticker-validation`, `full_27`, `spy`

---

## Hypothesis

> Does the full_27 feature set show predictive edge on SPY over the test period 2024-2026?

---

## Parameters

| Parameter | Value |
|-----------|-------|
| feature_set | full_27 |
| n_features | 27 |
| split_date | 2024-01-01 |
| n_estimators | 100 |
| prob_threshold | 0.55 |

---

## Features Used

- `mom_return_1d`
- `mom_return_5d`
- `mom_return_10d`
- `mom_return_20d`
- `mom_return_60d`
- `mom_zscore_20d`
- `mom_rsi_14`
- `vol_atr_14`
- `vol_atr_pct`
- `vol_realized_10d`
- `vol_realized_20d`
- `vol_rank_52w`
- `vol_ratio`
- `trend_dist_sma20`
- `trend_dist_sma50`
- `trend_dist_sma200`
- `trend_dist_ema20`
- `trend_sma20_slope`
- `trend_above_sma200`
- `vol_relative_volume`
- `vol_volume_spike`
- `vol_price_volume_trend`
- `vol_volume_momentum`
- `regime_bull_bear`
- `regime_volatility_state`
- `regime_trend_strength`
- `regime_drawdown_pct`

---

## Results

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.4765 |
| Sharpe | 2.997 |
| Top Feature | mom_return_1d |
| Train Rows | 482 |
| Test Rows | 604 |

---

## Conclusion

ROC-AUC 0.4765 on SPY using 27 features. Top feature: mom_return_1d.

---

_Aureline Labs · Quantitative Research & Intelligence Platform_
_Ateneo de Manila University · Applied Mathematics · Mathematical Finance_
