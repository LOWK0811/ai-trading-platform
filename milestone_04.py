# ======================================================================
# SECTION 1: IMPORTS
# ======================================================================
import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================================
# SECTION 2: REUSE OUR DATA LOADER FROM MILESTONE 3
# ======================================================================
def get_price_data(ticker, start, end):
    """Loads OHLCV data, using a local cache if one already exists."""
    filename = f"data/{ticker}_{start}_{end}.parquet"

    if os.path.exists(filename):
        data = pd.read_parquet(filename)
    else:
        data = yf.Ticker(ticker).history(start=start, end=end)
        os.makedirs("data", exist_ok=True)
        data.to_parquet(filename)

    return data


# ======================================================================
# SECTION 3: CALCULATE DAILY RETURNS
# ======================================================================
data = get_price_data("AAPL", "2021-01-01", "2026-06-01")
data["return"] = data["Close"].pct_change()


# ======================================================================
# SECTION 4: MEAN, STANDARD DEVIATION, AND Z-SCORES
# ======================================================================
mean_return = data["return"].mean()
std_return = data["return"].std()
data["z_score"] = (data["return"] - mean_return) / std_return

print(f"Average daily return: {mean_return:.4%}")
print(f"Standard deviation of daily returns: {std_return:.4%}")


# ======================================================================
# SECTION 5: TESTING THE HYPOTHESIS
# ======================================================================
big_down_days = data[data["z_score"] < -2]
data["next_day_return"] = data["return"].shift(-1)

next_day_after_drop = data.loc[big_down_days.index, "next_day_return"].mean()
typical_next_day = data["next_day_return"].mean()

print(f"\nNumber of 'big down days' (z-score < -2): {len(big_down_days)}")
print(f"Average next-day return after a big down day: {next_day_after_drop:.4%}")
print(f"Average next-day return on a typical day: {typical_next_day:.4%}")


# ======================================================================
# SECTION 6: VISUALIZING THE DISTRIBUTION
# ======================================================================
plt.hist(data["return"].dropna(), bins=50)
plt.axvline(mean_return, color="black", linestyle="--", label="Mean")
plt.axvline(mean_return - 2 * std_return, color="red", linestyle="--", label="±2 std")
plt.axvline(mean_return + 2 * std_return, color="red", linestyle="--")
plt.title("Distribution of AAPL Daily Returns")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.legend()
plt.show()