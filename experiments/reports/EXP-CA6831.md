# EXP-CA6831 — Aureline Labs Research Report

**Date:** 2026-06-28
**Strategy:** ML_RandomForest
**Asset:** SPY
**Period:** 2021-01-01 → 2026-06-01
**Tags:** `ml`, `cross-ticker-validation`, `momentum_7`, `spy`

---

## Hypothesis

> Does the momentum_7 feature set show predictive edge on SPY over the test period 2024-2026?

---

## Parameters

| Parameter | Value |
|-----------|-------|
| feature_set | momentum_7 |
| n_features | 7 |
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

---

## Results

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.4831 |
| Sharpe | 1.581 |
| Top Feature | mom_return_1d |
| Train Rows | 693 |
| Test Rows | 604 |

---

## Conclusion

ROC-AUC 0.4831 on SPY using 7 features. Top feature: mom_return_1d.

---

_Aureline Labs · Quantitative Research & Intelligence Platform_
_Ateneo de Manila University · Applied Mathematics · Mathematical Finance_
