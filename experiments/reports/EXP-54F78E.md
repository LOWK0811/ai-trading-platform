# EXP-54F78E
## AAPL · ML Research Report · 2026-06-21

---

### Hypothesis

> Assets that have experienced significant negative price momentum show statistically meaningful positive mean-reversion in subsequent periods.

---

### Methodology

| Parameter | Value |
|-----------|-------|
| Asset | AAPL |
| Study Period | 2021-01-01 → 2026-06-01 |
| Training Period | 2021-01-01 → 2024-01-01 |
| Test Period | 2024-01-01 → 2026-06-01 |
| Training Rows | 502 |
| Test Rows | 604 |
| Model | Random Forest |
| Trees | 100 |
| Max Depth | 4 |
| Min Samples Leaf | 20 |
| Signal Threshold | 0.55 |
| Features Used | 6 |
| Position Sizing | ATR 1% |
| Transaction Cost | 0.1% per trade |

---

### Feature Importance (Top 10)

| Rank | Feature | Importance Score |
|------|---------|-----------------|
| 1 | `mom_return_20d` | 0.1822 |
| 2 | `mom_rsi_14` | 0.1684 |
| 3 | `mom_return_1d` | 0.1677 |
| 4 | `mom_zscore_20d` | 0.1664 |
| 5 | `regime_drawdown_pct` | 0.1638 |
| 6 | `mom_return_5d` | 0.1515 |

---

### Classification Performance (Test Set)

| Metric | Up Days | Down Days |
|--------|---------|-----------|
| Precision | 0.575 | 0.495 |
| Recall | 0.550 | 0.520 |
| F1-Score | 0.562 | 0.507 |
| ROC-AUC | 0.5674 | — |

---

### Portfolio Performance (Test Period)

| Metric | Value |
|--------|-------|
| Strategy Return | 29.54% |
| Buy & Hold Return | 68.51% |
| Beat Buy & Hold | No |
| CAGR | 11.32% |
| Sharpe Ratio | 1.22 |
| Max Drawdown | -3.45% |
| Win Rate | 14.4% |
| Round Trips | 71 |

---

### Conclusion

The Random Forest model demonstrated meaningful predictive edge on AAPL with a ROC-AUC of 0.5674 on the held-out test set. The three most important features were mom_return_20d, mom_rsi_14, and mom_return_1d, suggesting the model primarily captured momentum and trend signals. The strategy underperformed buy-and-hold on the test period with a return of 29.54% vs 68.51%. Sharpe ratio of 1.22 suggests acceptable risk-adjusted performance. These results should be validated across additional tickers and time periods before drawing broader conclusions.

---

### Next Steps

- [ ] Validate on additional tickers (MSFT, NVDA, SPY)
- [ ] Test with different feature subsets
- [ ] Compare against SMA baseline on same test period
- [ ] Run Monte Carlo simulation on strategy returns

---

*Aureline Labs · Quantitative Research & Intelligence Platform*
*Ateneo de Manila University · Applied Mathematics · Mathematical Finance*
