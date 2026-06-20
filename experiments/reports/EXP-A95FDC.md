# EXP-A95FDC — Aureline Labs Research Report

**Date:** 2026-06-20
**Strategy:** SMA_20
**Asset:** MSFT
**Period:** 2021-01-01 → 2026-06-01
**Tags:** `sma`, `trend-following`, `atr-sizing`

---

## Hypothesis

> Does the same SMA(20) system generalize to MSFT, or is any edge AAPL-specific?

---

## Parameters

| Parameter | Value |
|-----------|-------|
| sma_window | 20 |
| cost_per_trade | 0.1% |
| position_sizing | ATR 1% |
| starting_cash | $10,000 |

---

## Features Used

_No features (rule-based strategy)_

---

## Results

| Metric | Value |
|--------|-------|
| Total Return | 7.73% |
| CAGR | 1.39% |
| Sharpe | -0.415 |
| Max Drawdown | -18.08% |
| Win Rate | 29.1% |
| Round Trips | 75 |

---

## Conclusion

The SMA(20) strategy on MSFT produced a Sharpe of -0.415 and CAGR of 1.39% over the study period. Strategy underperformed buy-and-hold. ATR-based position sizing kept max drawdown at -18.08%.

---

_Aureline Labs · Quantitative Research & Intelligence Platform_
_Ateneo de Manila University · Applied Mathematics · Mathematical Finance_
