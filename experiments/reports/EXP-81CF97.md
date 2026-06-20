# EXP-81CF97
## AAPL · ML Research Report · 2026-06-20

---

### Hypothesis

> Momentum features alone (multi-window returns, z-score, RSI) are sufficient to generate a predictive edge on AAPL, without requiring volatility, trend, or regime information.

---

### Methodology

| Parameter | Value |
|-----------|-------|
| Asset | AAPL |
| Study Period | 2021-01-01 → 2026-06-01 |
| Training Period | 2021-01-01 → 2024-01-01 |
| Test Period | 2024-01-01 → 2026-06-01 |
| Training Rows | 693 |
| Test Rows | 604 |
| Model | Random Forest |
| Trees | 100 |
| Max Depth | 4 |
| Min Samples Leaf | 20 |
| Signal Threshold | 0.55 |
| Features Used | 7 |
| Position Sizing | ATR 1% |
| Transaction Cost | 0.1% per trade |

---

### Feature Importance (Top 10)

| Rank | Feature | Importance Score |
|------|---------|-----------------|
| 1 | `mom_return_20d` | 0.1608 |
| 2 | `mom_return_60d` | 0.1597 |
| 3 | `mom_return_10d` | 0.1549 |
| 4 | `mom_return_1d` | 0.1525 |
| 5 | `mom_rsi_14` | 0.1523 |
| 6 | `mom_return_5d` | 0.1160 |
| 7 | `mom_zscore_20d` | 0.1038 |

---

### Classification Performance (Test Set)

| Metric | Up Days | Down Days |
|--------|---------|-----------|
| Precision | 0.558 | 0.488 |
| Recall | 0.661 | 0.383 |
| F1-Score | 0.605 | 0.429 |
| ROC-AUC | 0.545 | — |

---

### Portfolio Performance (Test Period)

| Metric | Value |
|--------|-------|
| Strategy Return | 2.72% |
| Buy & Hold Return | 68.51% |
| Beat Buy & Hold | No |
| CAGR | 1.12% |
| Sharpe Ratio | -0.569 |
| Max Drawdown | -11.07% |
| Win Rate | 17.2% |
| Round Trips | 73 |

---

### Conclusion

The Random Forest model demonstrated modest but real predictive edge on AAPL with a ROC-AUC of 0.545 on the held-out test set. The three most important features were mom_return_20d, mom_return_60d, and mom_return_10d, suggesting the model primarily captured momentum and trend signals. The strategy underperformed buy-and-hold on the test period with a return of 2.72% vs 68.51%. Sharpe ratio of -0.569 suggests weak risk-adjusted performance. These results should be validated across additional tickers and time periods before drawing broader conclusions.

---

### Next Steps

- [ ] Validate on additional tickers (MSFT, NVDA, SPY)
- [ ] Test with different feature subsets
- [ ] Compare against SMA baseline on same test period
- [ ] Run Monte Carlo simulation on strategy returns

---

*Aureline Labs · Quantitative Research & Intelligence Platform*
*Ateneo de Manila University · Applied Mathematics · Mathematical Finance*
