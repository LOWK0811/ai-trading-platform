# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from src.data_handler import get_price_data
from src.features import build_features, feature_cols
from src.indicators import add_atr
from src.risk import calculate_shares


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
# SECTION 3: METRICS FUNCTIONS
# ======================================================================
def sharpe_ratio(portfolio_values, risk_free_rate=0.05):
    """
    Annualized Sharpe ratio.
    risk_free_rate: annual rate (e.g. 0.05 = 5% Treasury yield).
    We convert it to a daily rate before subtracting.
    """
    values = pd.Series(portfolio_values)
    daily_returns = values.pct_change().dropna()
    daily_rf = risk_free_rate / 252     # 252 trading days in a year
    excess_returns = daily_returns - daily_rf
    if excess_returns.std() == 0:
        return 0.0
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)


def max_drawdown(portfolio_values):
    """
    Maximum peak-to-trough decline as a percentage.
    """
    values = pd.Series(portfolio_values)
    rolling_peak = values.cummax()      # highest value seen so far at each point
    drawdown = (values - rolling_peak) / rolling_peak
    return drawdown.min()               # most negative value = worst drawdown


def print_metrics(label, portfolio_values, starting_cash=10000):
    final    = portfolio_values[-1]
    ret      = (final / starting_cash) - 1
    sharpe   = sharpe_ratio(portfolio_values)
    mdd      = max_drawdown(portfolio_values)
    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"{'='*40}")
    print(f"  Total return:    {ret:.2%}")
    print(f"  Final value:     ${final:,.2f}")
    print(f"  Sharpe ratio:    {sharpe:.3f}")
    print(f"  Max drawdown:    {mdd:.2%}")


# ======================================================================
# SECTION 4: REBUILD WALK-FORWARD SIGNALS (FROM MILESTONE 17)
# ======================================================================
data = get_price_data("AAPL", "2021-01-01", "2026-06-01")
data = add_atr(data)
df   = build_features(data)
df   = df.dropna(subset=feature_cols() + ["label"])

min_train_size = 500
signals = []
logger.info("Running walk-forward training...")

for i in range(min_train_size, len(df) - 1):
    train_slice = df.iloc[:i]
    model = RandomForestClassifier(
        n_estimators=50, max_depth=4,
        min_samples_leaf=20, random_state=42
    )
    model.fit(train_slice[feature_cols()], train_slice["label"])
    prob_up = model.predict_proba(df[feature_cols()].iloc[[i]])[0][1]
    signals.append({
        "date":   df.index[i],
        "signal": 1 if prob_up >= 0.55 else 0
    })

    if i % 200 == 0:
        logger.info(f"Progress: {i}/{len(df)}")

signals_df = pd.DataFrame(signals).set_index("date")
bt_data    = df.loc[signals_df.index]


# ======================================================================
# SECTION 5: BACKTEST — ML STRATEGY
# ======================================================================
cost_per_trade = 0.001
starting_cash  = 10000

def run_simple_backtest(bt_data, signals_df):
    cash = starting_cash
    shares_held = 0
    portfolio = []

    for i in range(len(bt_data)):
        price_today     = bt_data["Close"].iloc[i]
        price_yesterday = bt_data["Close"].iloc[i-1] if i > 0 else price_today
        atr_today       = bt_data["atr"].iloc[i]
        signal_today    = signals_df["signal"].iloc[i]

        if signal_today == 1 and shares_held == 0:
            shares = calculate_shares(cash, price_yesterday, atr_today, risk_pct=0.01)
            if shares > 0:
                cash -= shares * price_yesterday * (1 + cost_per_trade)
                shares_held = shares
        elif signal_today == 0 and shares_held > 0:
            cash += shares_held * price_yesterday * (1 - cost_per_trade)
            shares_held = 0

        portfolio.append(cash + shares_held * price_today)
    return portfolio


ml_portfolio = run_simple_backtest(bt_data, signals_df)


# ======================================================================
# SECTION 6: BUY AND HOLD OVER THE SAME PERIOD
# ======================================================================
bh_shares    = starting_cash // bt_data["Close"].iloc[0]
bh_portfolio = (bh_shares * bt_data["Close"]).tolist()


# ======================================================================
# SECTION 7: PRINT ALL METRICS
# ======================================================================
print_metrics("ML Strategy",   ml_portfolio)
print_metrics("Buy & Hold",    bh_portfolio)


# ======================================================================
# SECTION 8: DRAWDOWN CHART
# ======================================================================
def drawdown_series(portfolio_values):
    values = pd.Series(portfolio_values)
    rolling_peak = values.cummax()
    return ((values - rolling_peak) / rolling_peak) * 100

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(bt_data.index, ml_portfolio, label="ML Strategy")
ax1.plot(bt_data.index, bh_portfolio, label="Buy & Hold")
ax1.set_ylabel("Portfolio Value (USD)")
ax1.set_title("ML Strategy vs Buy & Hold")
ax1.legend()

ax2.fill_between(bt_data.index,
                 drawdown_series(ml_portfolio),
                 0, alpha=0.4, color="red", label="ML Drawdown")
ax2.fill_between(bt_data.index,
                 drawdown_series(bh_portfolio),
                 0, alpha=0.4, color="orange", label="B&H Drawdown")
ax2.set_ylabel("Drawdown (%)")
ax2.set_xlabel("Date")
ax2.legend()

plt.tight_layout()
plt.show()