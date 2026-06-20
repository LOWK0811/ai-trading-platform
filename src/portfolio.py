# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 2: PORTFOLIO BUILDER
# ======================================================================
class Portfolio:
    """
    Builds and analyzes a multi-asset portfolio.
    Handles correlation analysis, risk budgeting,
    and position sizing across multiple simultaneous holdings.
    """

    def __init__(self, prices_dict, starting_cash=100000):
        """
        Parameters
        ----------
        prices_dict : dict — {ticker: pd.Series of Close prices}
        starting_cash : float — total capital to allocate
        """
        self.prices        = pd.DataFrame(prices_dict).dropna()
        self.returns       = self.prices.pct_change().dropna()
        self.tickers       = list(prices_dict.keys())
        self.starting_cash = starting_cash
        logger.info(f"Portfolio initialized: {self.tickers}")
        logger.info(f"Date range: {self.prices.index[0].date()} "
                   f"→ {self.prices.index[-1].date()}")
        logger.info(f"Trading days: {len(self.prices)}")


    # ======================================================================
    # SECTION 3: CORRELATION ANALYSIS
    # ======================================================================
    def correlation_matrix(self):
        """Returns the pairwise correlation matrix of daily returns."""
        return self.returns.corr()

    def find_low_correlation_pairs(self, threshold=0.3):
        """
        Identifies pairs of assets with correlation below threshold.
        Low correlation = genuine diversification benefit.
        """
        corr = self.correlation_matrix()
        pairs = []

        for i in range(len(self.tickers)):
            for j in range(i+1, len(self.tickers)):
                t1 = self.tickers[i]
                t2 = self.tickers[j]
                c  = corr.loc[t1, t2]
                if abs(c) < threshold:
                    pairs.append((t1, t2, round(c, 3)))

        pairs.sort(key=lambda x: abs(x[2]))
        return pairs

    def find_high_correlation_pairs(self, threshold=0.7):
        """
        Identifies pairs with correlation above threshold.
        High correlation = redundant risk — owning both
        doesn't add meaningful diversification.
        """
        corr = self.correlation_matrix()
        pairs = []

        for i in range(len(self.tickers)):
            for j in range(i+1, len(self.tickers)):
                t1 = self.tickers[i]
                t2 = self.tickers[j]
                c  = corr.loc[t1, t2]
                if abs(c) >= threshold:
                    pairs.append((t1, t2, round(c, 3)))

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs


    # ======================================================================
    # SECTION 4: INDIVIDUAL ASSET STATISTICS
    # ======================================================================
    def asset_statistics(self):
        """
        Computes annualized return, volatility, and Sharpe
        for each asset individually.
        """
        stats = {}
        for ticker in self.tickers:
            ret = self.returns[ticker]
            ann_return = ret.mean() * 252
            ann_vol    = ret.std() * np.sqrt(252)
            sharpe     = (ann_return - 0.05) / ann_vol \
                         if ann_vol > 0 else 0.0
            stats[ticker] = {
                "Ann. Return": f"{ann_return:.2%}",
                "Ann. Volatility": f"{ann_vol:.2%}",
                "Sharpe Ratio": round(sharpe, 3),
                "Max Drawdown": self._max_dd(ticker)
            }
        return pd.DataFrame(stats).T


    def _max_dd(self, ticker):
        prices = self.prices[ticker]
        peak   = prices.cummax()
        dd     = (prices - peak) / peak
        return f"{dd.min():.2%}"


    # ======================================================================
    # SECTION 5: PORTFOLIO VOLATILITY
    # ======================================================================
    def portfolio_volatility(self, weights):
        """
        Computes annualized portfolio volatility given weights.
        This is where correlation actually matters:
        portfolio vol < weighted sum of individual vols
        when assets are less than perfectly correlated.

        Parameters
        ----------
        weights : dict — {ticker: weight} summing to 1.0
        """
        w      = np.array([weights[t] for t in self.tickers])
        cov    = self.returns.cov() * 252
        port_var = w.T @ cov.values @ w
        return np.sqrt(port_var)

    def diversification_ratio(self, weights):
        """
        The ratio of weighted average individual volatility
        to portfolio volatility. A ratio > 1 means you're
        getting genuine diversification benefit.
        DR = 1.0 means no diversification (all perfectly correlated).
        DR = 2.0 means portfolio vol is half of individual vol average.
        """
        w        = np.array([weights[t] for t in self.tickers])
        ind_vols = self.returns.std() * np.sqrt(252)
        weighted_avg_vol = (w * ind_vols.values).sum()
        port_vol = self.portfolio_volatility(weights)
        return weighted_avg_vol / port_vol if port_vol > 0 else 1.0


    # ======================================================================
    # SECTION 6: ALLOCATION STRATEGIES
    # ======================================================================
    def equal_weight(self):
        """1/N allocation — simplest possible diversification."""
        n = len(self.tickers)
        return {t: 1/n for t in self.tickers}

    def risk_parity(self):
        """
        Risk parity: allocate inversely proportional to volatility.
        Higher volatility assets get smaller positions so each
        asset contributes equally to total portfolio risk.
        This is the core insight behind Bridgewater's All Weather fund.
        """
        vols    = self.returns.std() * np.sqrt(252)
        inv_vol = 1 / vols
        total   = inv_vol.sum()
        return {t: inv_vol[t] / total for t in self.tickers}

    def momentum_weighted(self, lookback=60):
        """
        Allocate more to recent winners, less to recent losers.
        Only allocates to assets with positive momentum.
        Classic momentum portfolio construction.
        """
        momentum = self.returns.rolling(lookback).mean().iloc[-1]
        positive = momentum[momentum > 0]
        if len(positive) == 0:
            return self.equal_weight()
        total = positive.sum()
        weights = {t: positive[t]/total if t in positive.index else 0.0
                   for t in self.tickers}
        return weights


    # ======================================================================
    # SECTION 7: PORTFOLIO SIMULATION
    # ======================================================================
    def simulate(self, weights, rebalance_frequency="monthly"):
        """
        Simulates a buy-and-hold portfolio with periodic rebalancing.

        Parameters
        ----------
        weights : dict — target allocation per ticker
        rebalance_frequency : "monthly", "quarterly", or "never"
        """
        cash    = self.starting_cash
        holdings= {t: 0 for t in self.tickers}
        portfolio_values = []

        # Initial allocation
        for t in self.tickers:
            alloc = cash * weights.get(t, 0)
            price = self.prices[t].iloc[0]
            if price > 0 and alloc > 0:
                shares = alloc // price
                holdings[t] = shares
                cash -= shares * price

        # Determine rebalance dates
        if rebalance_frequency == "monthly":
            rebalance_dates = set(
                self.prices.resample("ME").last().index
            )
        elif rebalance_frequency == "quarterly":
            rebalance_dates = set(
                self.prices.resample("QE").last().index
            )
        else:
            rebalance_dates = set()

        for date, row in self.prices.iterrows():
            # Portfolio value today
            port_val = cash + sum(
                holdings[t] * row[t] for t in self.tickers
            )
            portfolio_values.append(port_val)

            # Rebalance if it's time
            if date in rebalance_dates and port_val > 0:
                # Sell everything back to cash
                for t in self.tickers:
                    cash += holdings[t] * row[t]
                    holdings[t] = 0
                # Reallocate according to target weights
                for t in self.tickers:
                    alloc = cash * weights.get(t, 0)
                    price = row[t]
                    if price > 0 and alloc > 0:
                        shares = alloc // price
                        holdings[t] = shares
                        cash -= shares * price

        return portfolio_values


    # ======================================================================
    # SECTION 8: FULL PORTFOLIO REPORT
    # ======================================================================
    def report(self, weights_dict):
        """
        Prints a comprehensive portfolio analysis report.

        Parameters
        ----------
        weights_dict : dict of dicts — {name: {ticker: weight}}
        """
        print(f"\n{'='*65}")
        print(f"  AURELINE LABS — PORTFOLIO ANALYSIS REPORT")
        print(f"{'='*65}")
        print(f"  Assets: {', '.join(self.tickers)}")
        print(f"  Period: {self.prices.index[0].date()} → "
              f"{self.prices.index[-1].date()}")

        print(f"\n  INDIVIDUAL ASSET STATISTICS")
        print(f"  {'-'*60}")
        stats = self.asset_statistics()
        print(stats.to_string())

        print(f"\n  CORRELATION MATRIX")
        print(f"  {'-'*60}")
        corr = self.correlation_matrix().round(3)
        print(corr.to_string())

        high_pairs = self.find_high_correlation_pairs(0.7)
        low_pairs  = self.find_low_correlation_pairs(0.3)

        if high_pairs:
            print(f"\n  HIGH CORRELATION PAIRS (>0.70) — "
                  f"Redundant risk:")
            for t1, t2, c in high_pairs:
                print(f"    {t1} ↔ {t2}: {c:+.3f}")

        if low_pairs:
            print(f"\n  LOW CORRELATION PAIRS (<0.30) — "
                  f"Genuine diversification:")
            for t1, t2, c in low_pairs:
                print(f"    {t1} ↔ {t2}: {c:+.3f}")

        print(f"\n  ALLOCATION STRATEGIES")
        print(f"  {'-'*60}")
        print(f"  {'Strategy':<25} {'Port. Vol':>10} "
              f"{'Div. Ratio':>12} {'Weights'}")
        print(f"  {'-'*60}")

        for name, weights in weights_dict.items():
            port_vol = self.portfolio_volatility(weights)
            div_rat  = self.diversification_ratio(weights)
            w_str    = "  ".join(
                [f"{t}:{v:.0%}" for t, v in weights.items()
                 if v > 0.01]
            )
            print(f"  {name:<25} {port_vol:>9.2%} "
                  f"{div_rat:>12.3f}  {w_str}")

        print(f"\n  PORTFOLIO SIMULATION (Monthly Rebalancing)")
        print(f"  {'-'*60}")
        print(f"  {'Strategy':<25} {'Return':>9} "
              f"{'Sharpe':>9} {'Max DD':>10}")
        print(f"  {'-'*60}")

        for name, weights in weights_dict.items():
            pv   = self.simulate(weights)
            ret  = (pv[-1] / self.starting_cash - 1) * 100
            s    = pd.Series(pv)
            dr   = s.pct_change().dropna()
            ex   = dr - 0.05/252
            sh   = round((ex.mean()/ex.std())*np.sqrt(252), 3) \
                   if ex.std() > 0 else 0.0
            peak = s.cummax()
            mdd  = round(((s-peak)/peak).min()*100, 2)
            print(f"  {name:<25} {ret:>+8.2f}%  "
                  f"{sh:>8.3f}  {mdd:>9.2f}%")

        print(f"{'='*65}")