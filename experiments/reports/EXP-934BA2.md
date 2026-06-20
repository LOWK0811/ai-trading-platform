# EXP-934BA2
## AAPL · ML Research Report · 2026-06-20

---

### Hypothesis

> The five highest-importance features identified in EXP-6F9682 (trend_dist_sma50, vol_volume_momentum, mom_return_60d, mom_return_10d, mom_return_20d) contain the majority of predictive signal. A model trained on these five features alone should outperform the full 27-feature model due to reduced noise and higher training rows per feature.

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
| Features Used | 5 |
| Position Sizing | ATR 1% |
| Transaction Cost | 0.1% per trade |

---

### Feature Importance (Top 10)

| Rank | Feature | Importance Score |
|------|---------|-----------------|
| 1 | `mom_return_20d` | 0.2504 |
| 2 | `vol_volume_momentum` | 0.2134 |
| 3 | `trend_dist_sma50` | 0.1980 |
| 4 | `mom_return_10d` | 0.1888 |
| 5 | `mom_return_60d` | 0.1493 |

---

### Classification Performance (Test Set)

| Metric | Up Days | Down Days |
|--------|---------|-----------|
| Precision | 0.558 | 0.484 |
| Recall | 0.615 | 0.426 |
| F1-Score | 0.585 | 0.453 |
| ROC-AUC | 0.5143 | — |

---

### Portfolio Performance (Test Period)

| Metric | Value |
|--------|-------|
| Strategy Return | -9.56% |
| Buy & Hold Return | 68.51% |
| Beat Buy & Hold | No |
| CAGR | -4.08% |
| Sharpe Ratio | -1.308 |
| Max Drawdown | -15.31% |
| Win Rate | 12.8% |
| Round Trips | 58 |

---

### Conclusion

The Random Forest model demonstrated no discernible predictive edge on AAPL with a ROC-AUC of 0.5143 on the held-out test set. The three most important features were mom_return_20d, vol_volume_momentum, and trend_dist_sma50, suggesting the model primarily captured momentum and trend signals. The strategy underperformed buy-and-hold on the test period with a return of -9.56% vs 68.51%. Sharpe ratio of -1.308 suggests weak risk-adjusted performance. These results should be validated across additional tickers and time periods before drawing broader conclusions.

---

### Next Steps

- [ ] Validate on additional tickers (MSFT, NVDA, SPY)
- [ ] Test with different feature subsets
- [ ] Compare against SMA baseline on same test period
- [ ] Run Monte Carlo simulation on strategy returns

---

*Aureline Labs · Quantitative Research & Intelligence Platform*
*Ateneo de Manila University · Applied Mathematics · Mathematical Finance*
