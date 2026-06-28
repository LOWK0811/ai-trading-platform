# EXP-4A1B39 — Aureline Labs Research Report

**Date:** 2026-06-28
**Strategy:** ML_RandomForest
**Asset:** XOM
**Period:** 2021-01-01 → 2026-06-01
**Tags:** `ml`, `cross-ticker-validation`, `top5_selected`, `xom`

---

## Hypothesis

> Does the top5_selected feature set show predictive edge on XOM over the test period 2024-2026?

---

## Parameters

| Parameter | Value |
|-----------|-------|
| feature_set | top5_selected |
| n_features | 5 |
| split_date | 2024-01-01 |
| n_estimators | 100 |
| prob_threshold | 0.55 |

---

## Features Used

- `trend_dist_sma50`
- `vol_volume_momentum`
- `mom_return_60d`
- `mom_return_10d`
- `mom_return_20d`

---

## Results

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.5107 |
| Sharpe | -1.524 |
| Top Feature | trend_dist_sma50 |
| Train Rows | 693 |
| Test Rows | 604 |

---

## Conclusion

ROC-AUC 0.5107 on XOM using 5 features. Top feature: trend_dist_sma50.

---

_Aureline Labs · Quantitative Research & Intelligence Platform_
_Ateneo de Manila University · Applied Mathematics · Mathematical Finance_
