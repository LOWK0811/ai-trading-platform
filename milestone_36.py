# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import logging
from datetime import datetime
from src.database import Database
from src.philippine_market import (
    PSE_INDICES, PSE_OTC, PH_NAMES, PH_MACRO,
    PH_WATCHLIST, load_ph_universe,
    analyze_ph_ticker, compare_pse_sectors,
    generate_ph_brief
)


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
# SECTION 3: CONFIGURATION
# ======================================================================
START = "2023-01-01"
END   = "2026-06-22"

# Try indices first (most reliable), then OTC stocks
CORE_WATCHLIST = [
    # PSE sector indices — denominated in PHP
    "PSEI.PS",   # PSEi composite
    "FIN.PS",    # Financials sector
    "PRO.PS",    # Property sector
    "HDG.PS",    # Holding firms sector
    "IND.PS",    # Industrial sector
    # Philippine OTC/ADR stocks — denominated in USD
    "PHI",       # PLDT (NYSE-listed)
    "BDOUY",     # BDO Unibank
    "BPHLY",     # Bank of the Philippine Islands
    "AYAAF",     # Ayala Land
    "JBFCF",     # Jollibee Foods
]


# ======================================================================
# SECTION 4: LOAD PHILIPPINE MARKET DATA
# ======================================================================
print(f"\n{'='*65}")
print(f"  AURELINE LABS — PHILIPPINE MARKET INTEGRATION")
print(f"{'='*65}")
print(f"  Tickers:  {len(CORE_WATCHLIST)} "
      f"(indices + OTC/ADR stocks)")
print(f"  Period:   {START} → {END}")
print(f"  Sources:  Yahoo Finance (.PS indices + OTC tickers)\n")

ph_data = load_ph_universe(CORE_WATCHLIST, START, END)

successful = list(ph_data.keys())
failed     = [t for t in CORE_WATCHLIST if t not in ph_data]

print(f"\n  Loaded:  {len(successful)} tickers")
if successful:
    print(f"           {successful}")
if failed:
    print(f"  Failed:  {failed}")


# ======================================================================
# SECTION 5: ANALYZE ALL LOADED TICKERS
# ======================================================================
print(f"\n  Running quantitative analysis...")

analyses = []
for ticker, data in ph_data.items():
    analysis = analyze_ph_ticker(ticker, data)
    if analysis:
        analyses.append(analysis)

# Sector comparison (indices only)
index_data = {t: ph_data[t] for t in ph_data
              if t.endswith(".PS")}
sector_comparison = compare_pse_sectors(index_data)


# ======================================================================
# SECTION 6: PRINT RESULTS TABLE
# ======================================================================
if analyses:
    print(f"\n{'='*75}")
    print(f"  PHILIPPINE MARKET ANALYSIS — {END}")
    print(f"{'='*75}")
    print(f"  {'Sig'} {'Ticker':<10} "
          f"{'Name':<30} "
          f"{'Price':>12} "
          f"{'20d Ret':>8} "
          f"{'RSI':>6} "
          f"{'Signal'}")
    print(f"  {'-'*70}")

    # Indices first
    print(f"  --- PSE SECTOR INDICES (₱ PHP) ---")
    for a in sorted([a for a in analyses
                     if a["ticker"].endswith(".PS")],
                    key=lambda x: x["ticker"]):
        icon = {"BUY":"▲","HOLD":"─","AVOID":"▼"}.get(
            a["signal"], "?")
        print(f"  {icon} {a['ticker']:<10} "
              f"{a['name'][:28]:<28} "
              f"₱{a['price']:>10.2f} "
              f"{a['ret_20d']:>+7.2f}% "
              f"{a['rsi']:>6.1f} "
              f"{a['signal']}")

    # OTC stocks
    otc = [a for a in analyses if not a["ticker"].endswith(".PS")]
    if otc:
        print(f"\n  --- PHILIPPINE OTC/ADR STOCKS ($ USD) ---")
        for a in sorted(otc, key=lambda x: x["ticker"]):
            icon = {"BUY":"▲","HOLD":"─","AVOID":"▼"}.get(
                a["signal"], "?")
            print(f"  {icon} {a['ticker']:<10} "
                  f"{a['name'][:28]:<28} "
                  f"${a['price']:>10.2f} "
                  f"{a['ret_20d']:>+7.2f}% "
                  f"{a['rsi']:>6.1f} "
                  f"{a['signal']}")

    # Sector comparison
    if sector_comparison:
        print(f"\n  --- SECTOR vs PSEi (20d) ---")
        for ticker, info in sorted(
                sector_comparison.items(),
                key=lambda x: x[1]["ret_20d"],
                reverse=True):
            icon = "▲" if info["outperform"] else "▼"
            print(f"  {icon} {info['name']:<30} "
                  f"{info['ret_20d']:>+8.2f}% "
                  f"({info['vs_psei']:>+6.2f}% vs PSEi)")

    # Summary stats
    print(f"\n  SUMMARY")
    print(f"  {'-'*40}")
    avg_ret = sum(a["ret_20d"] for a in analyses) / len(analyses)
    buy_ct  = sum(1 for a in analyses if a["signal"] == "BUY")
    avoid_ct= sum(1 for a in analyses if a["signal"] == "AVOID")
    hold_ct = sum(1 for a in analyses if a["signal"] == "HOLD")
    print(f"  Avg 20d Return:  {avg_ret:>+.2f}%")
    print(f"  Buy / Hold / Avoid: "
          f"{buy_ct} / {hold_ct} / {avoid_ct}")
    print(f"{'='*75}")
else:
    print(f"\n  No analysis data available.")
    print(f"  All tickers failed to load from Yahoo Finance.")
    print(f"  This is a data source limitation, not a code bug.")


# ======================================================================
# SECTION 7: GENERATE PHILIPPINE MARKET BRIEF
# ======================================================================
print(f"\n  Generating Philippine Market Intelligence Brief...")

brief     = generate_ph_brief(analyses, sector_comparison, PH_MACRO)
date_str  = datetime.now().strftime("%Y-%m-%d")
brief_path = (f"experiments/reports/"
              f"PH_MARKET_BRIEF_{date_str}.md")

os.makedirs("experiments/reports", exist_ok=True)
with open(brief_path, "w") as f:
    f.write(brief)

print(f"  Saved: {brief_path}")


# ======================================================================
# SECTION 8: SAVE TO DATABASE
# ======================================================================
db = Database()

# Save price data
for ticker, data in ph_data.items():
    db.insert_prices(ticker, data)

# Save company/index profiles
for ticker in ph_data:
    name     = PH_NAMES.get(ticker, ticker)
    sector   = "Index" if ticker.endswith(".PS") else "Equity"
    industry = PSE_INDICES.get(ticker,
               PSE_OTC.get(ticker, "Philippine Market"))
    country  = "PH"
    db.insert_company(ticker, name, sector, industry, country)

# Save report
report_id = db.insert_report(
    title       = f"Philippine Market Brief — {date_str}",
    content     = brief,
    report_type = "ph_market_brief",
    tickers     = successful
)

# Update agent memory
db.set_memory("research_agent",
              "ph_last_run", date_str)
db.set_memory("research_agent",
              "ph_tickers_loaded", successful)
db.set_memory("research_agent",
              "ph_tickers_failed", failed)

stats = db.stats()
print(f"\n{'='*65}")
print(f"  DATABASE UPDATED")
print(f"{'='*65}")
print(f"  Total prices:    {stats['prices']:>8,} rows")
print(f"  Total companies: {stats['companies']:>8}")
print(f"  Total reports:   {stats['reports']:>8}")
print(f"  Agent memory:    {stats['agent_memory']:>8} entries")
print(f"{'='*65}")

db.close()