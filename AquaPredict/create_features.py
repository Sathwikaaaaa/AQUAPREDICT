
import pandas as pd

print("Loading prepared daily datasets...")

demand = pd.read_csv("daily_demand.csv", index_col=0)
weather = pd.read_csv("daily_weather.csv", index_col=0)

# Convert CSV indexes into proper datetime indexes
demand.index = pd.to_datetime(demand.index, utc=True)
weather.index = pd.to_datetime(weather.index, utc=True)

# Use timezone-naive daily dates consistently
demand.index = demand.index.tz_convert(None).normalize()
weather.index = weather.index.tz_convert(None).normalize()

print("Demand index type:", type(demand.index).__name__)

# Use DMA 1 for the first experiment
dma = "DMA 1"

data = pd.DataFrame(index=demand.index)

# Target: demand on the day being predicted
data["target_demand"] = demand[dma]

# Previous 7 days of demand
for lag in range(1, 8):
    data[f"demand_lag_{lag}"] = demand[dma].shift(lag)

# Calendar features
data["month"] = data.index.month
data["day_of_week"] = data.index.dayofweek

# Weather features
data["daily_rainfall"] = weather["Rain"]
data["daily_temperature"] = weather["Temperature"]

# Remove rows with unavailable lag values
data = data.dropna()

print("\n========== FEATURE DATASET ==========")
print("DMA:", dma)
print("Shape:", data.shape)
print("\nColumns:")
print(data.columns.tolist())
print("\nFirst 5 rows:")
print(data.head())
print("\nMissing values:", data.isna().sum().sum())

data.to_csv("dma1_features.csv")

print("\nSaved dma1_features.csv")
print("Feature preparation complete!")
