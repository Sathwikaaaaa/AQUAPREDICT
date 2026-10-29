import wf4bwdf
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_DMA = "DMA 3"

# Require at least 22 valid hourly measurements
# before accepting a day's demand value.
MIN_VALID_HOURS = 22

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 65)
print("AQUAPREDICT - DAILY DATA PREPARATION")
print("=" * 65)

dataset = wf4bwdf.load_complete_dataset()

inflows = dataset["dma-inflows"].copy()
weather = dataset["weather"].copy()
calendar = dataset["calendar"].copy()

print("\nLoaded dataset successfully.")
print("Hourly rows:", len(inflows))
print("Target:", TARGET_DMA)


# ============================================================
# 2. SELECT TARGET DMA
# ============================================================

flow = inflows[TARGET_DMA]

print("\nOriginal missing hourly values:")
print(flow.isna().sum())


# ============================================================
# 3. DAILY FLOW STATISTICS
# ============================================================

# Count how many valid hourly observations exist each day.
daily_valid_hours = flow.resample("D").count()

# Mean hourly flow for each day.
#
# IMPORTANT:
# The source variable is an hourly flow-rate measurement.
# For our first experiment we keep DAILY MEAN FLOW as the
# forecasting target instead of pretending a simple sum is
# necessarily daily volume.
daily_flow = flow.resample("D").mean()

# Days with insufficient observations are invalid.
daily_flow[daily_valid_hours < MIN_VALID_HOURS] = np.nan


# ============================================================
# 4. DAILY WEATHER FEATURES
# ============================================================

daily_weather = pd.DataFrame(index=daily_flow.index)

# Rain -> total rainfall during day
daily_weather["Rain"] = weather["Rain"].resample("D").sum()

# Temperature -> daily mean
daily_weather["Temperature"] = (
    weather["Temperature"]
    .resample("D")
    .mean()
)

# Humidity -> daily mean
daily_weather["Humidity"] = (
    weather["Humidity"]
    .resample("D")
    .mean()
)

# Windspeed -> daily mean
daily_weather["Windspeed"] = (
    weather["Windspeed"]
    .resample("D")
    .mean()
)


# ============================================================
# 5. HANDLE WEATHER MISSING VALUES
# ============================================================

# Do NOT blindly interpolate the target water-demand series.
#
# For weather only, perform time interpolation at daily level.
daily_weather = daily_weather.interpolate(
    method="time",
    limit_direction="both"
)


# ============================================================
# 6. CALENDAR FEATURES
# ============================================================

daily_calendar = pd.DataFrame(index=daily_flow.index)

# If any hour in a day is marked as a holiday,
# consider that date a holiday.
daily_calendar["Holiday"] = (
    calendar["Holiday"]
    .resample("D")
    .max()
    .astype(int)
)

# Date-derived features.
daily_calendar["day_of_week"] = daily_calendar.index.dayofweek

daily_calendar["month"] = daily_calendar.index.month

daily_calendar["day_of_year"] = daily_calendar.index.dayofyear

daily_calendar["is_weekend"] = (
    daily_calendar["day_of_week"] >= 5
).astype(int)


# ============================================================
# 7. COMBINE EVERYTHING
# ============================================================

daily = pd.concat(
    [
        daily_flow.rename("Demand"),
        daily_valid_hours.rename("ValidHours"),
        daily_weather,
        daily_calendar
    ],
    axis=1
)


# ============================================================
# 8. QUALITY REPORT BEFORE DROPPING INVALID TARGET DAYS
# ============================================================

print("\n" + "=" * 65)
print("DAILY DATA QUALITY")
print("=" * 65)

print("\nTotal calendar days:", len(daily))

print("\nValid hourly observations per day:")
print(daily["ValidHours"].describe())

print("\nMissing values BEFORE target filtering:")
print(daily.isna().sum())

invalid_days = daily["Demand"].isna().sum()

print("\nDays rejected because target had insufficient data:",
      invalid_days)


# ============================================================
# 9. REMOVE DAYS WITHOUT RELIABLE TARGET
# ============================================================

daily_clean = daily.dropna(subset=["Demand"]).copy()


# ============================================================
# 10. FINAL CHECKS
# ============================================================

print("\n" + "=" * 65)
print("FINAL DAILY DATASET")
print("=" * 65)

print("Rows:", len(daily_clean))

print(
    "Date range:",
    daily_clean.index.min(),
    "to",
    daily_clean.index.max()
)

print("\nMissing values:")
print(daily_clean.isna().sum())

print("\nFirst 10 rows:")
print(daily_clean.head(10))

print("\nDemand statistics:")
print(daily_clean["Demand"].describe())


# ============================================================
# 11. SAVE
# ============================================================

output_file = OUTPUT_DIR / "dma3_daily.csv"

daily_clean.to_csv(
    output_file,
    index=True
)

print("\nSaved:")
print(output_file)

print("\nDaily preprocessing completed successfully!")