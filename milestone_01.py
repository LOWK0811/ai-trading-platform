# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import pandas as pd


# ======================================================================
# SECTION 2: PYTHON BASICS WARM-UP
# ======================================================================
ticker = "AAPL"       # a string (text) — always wrapped in quotes
price = 184.32        # a float — a number with decimals
shares = 10           # an int — a whole number

print(f"You own {shares} shares of {ticker}, currently worth ${price} each.")


# ======================================================================
# SECTION 3: BUILDING YOUR FIRST DATAFRAME
# ======================================================================
data = {
    "date": ["2026-06-10", "2026-06-11", "2026-06-12", "2026-06-13", "2026-06-16"],
    "close_price": [184.32, 186.10, 183.95, 188.40, 190.05],
}

df = pd.DataFrame(data)


# ======================================================================
# SECTION 4: TAKING A LOOK AT THE DATA
# ======================================================================
print(df)
print(f"\nAverage closing price this period: {df['close_price'].mean():.2f}")

# ======================================================================
# SECTION 5: LOOPS & CONDITIONALS — UP DAYS VS DOWN DAYS
# ======================================================================
prices = df["close_price"].tolist()  # pull the column out as a plain Python list

for i in range(1, len(prices)):
    change = prices[i] - prices[i - 1]
    if change > 0:
        print(f"Day {i}: UP   ({change:+.2f})")
    else:
        print(f"Day {i}: DOWN ({change:+.2f})")