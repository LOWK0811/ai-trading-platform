# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
from src.company_intelligence import (
    get_fundamentals, get_technical_snapshot,
    generate_bull_bear, generate_company_report
)

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logging.getLogger("yfinance").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ======================================================================
# SECTION 2: TEST THREE COMPANIES
# ======================================================================
for ticker in ["AAPL", "NVDA", "PHI"]:
    print(f"\n{'='*65}")
    print(f"  AURELINE LABS — COMPANY INTELLIGENCE: {ticker}")
    print(f"{'='*65}")

    report = generate_company_report(ticker)

    if not report:
        print(f"  No data available for {ticker}")
        continue

    f = report["fundamentals"]
    t = report["technical"]

    if f:
        print(f"\n  COMPANY PROFILE")
        print(f"  Name:     {f['name']}")
        print(f"  Sector:   {f['sector']} · {f['industry']}")
        print(f"  Market Cap: {f['market_cap']}")
        desc = f['description']
        if desc and len(desc) > 0:
            print(f"\n  DESCRIPTION (first 200 chars)")
            print(f"  {desc[:200]}...")

        print(f"\n  KEY METRICS")
        print(f"  {'P/E Ratio:':<25} {f['pe_ratio']}")
        print(f"  {'Forward P/E:':<25} {f['forward_pe']}")
        print(f"  {'Revenue:':<25} {f['revenue']}")
        print(f"  {'Revenue Growth:':<25} {f['revenue_growth']}")
        print(f"  {'Profit Margin:':<25} {f['profit_margin']}")
        print(f"  {'Gross Margin:':<25} {f['gross_margin']}")
        print(f"  {'Debt/Equity:':<25} {f['debt_to_equity']}")
        print(f"  {'ROE:':<25} {f['roe']}")
        print(f"  {'Dividend Yield:':<25} {f['dividend_yield']}")
        print(f"  {'Analyst Rating:':<25} {f['analyst_rating']} "
              f"({f['num_analysts']} analysts)")
        print(f"  {'Price Target:':<25} ${f['target_mean']} "
              f"(range ${f['target_low']}–${f['target_high']})")

    if t:
        print(f"\n  TECHNICAL SNAPSHOT")
        print(f"  Price:     ${t['price']:.2f}")
        print(f"  20d Ret:   {t['ret_20d']:+.2f}%")
        print(f"  RSI:       {t['rsi']:.1f}")
        print(f"  Regime:    {t['regime']}")
        print(f"  Signal:    {t['signal']}")
        print(f"  ATM Call (3m): ${t['call_3m']}")
        print(f"  ATM Put  (3m): ${t['put_3m']}")

    bull, bear = report["bull_case"], report["bear_case"]
    print(f"\n  BULL CASE")
    for b in bull:
        print(f"  ▲ {b}")
    print(f"\n  BEAR CASE")
    for b in bear:
        print(f"  ▼ {b}")