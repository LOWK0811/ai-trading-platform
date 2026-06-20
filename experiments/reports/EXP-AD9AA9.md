# EXP-AD9AA9 — Aureline Labs Research Report

**Date:** 2026-06-20
**Strategy:** SMA_20
**Asset:** NVDA
**Period:** 2021-01-01 → 2026-06-01
**Tags:** `sma`, `trend-following`, `atr-sizing`

---

## Hypothesis

> Does SMA(20) capture the AI-driven momentum in NVDA despite high volatility causing ATR to reduce position size?

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
| Total Return | 91.58% |
| CAGR | 12.76% |
| Sharpe | 0.739 |
| Max Drawdown | -14.24% |
| Win Rate | 32.4% |
| Round Trips | 69 |

---

## Conclusion

The SMA(20) strategy on NVDA produced a Sharpe of 0.739 and CAGR of 12.76% over the study period. Strategy underperformed buy-and-hold. ATR-based position sizing kept max drawdown at -14.24%.

---

_Aureline Labs · Quantitative Research & Intelligence Platform_
_Ateneo de Manila University · Applied Mathematics · Mathematical Finance_
