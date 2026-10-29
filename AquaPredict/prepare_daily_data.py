
import wf4bwdf as bwdf
import pandas as pd
from sklearn.impute import KNNImputer

print("Loading BWDF dataset...")
dataset = bwdf.load_complete_dataset()

inflows = dataset["dma-inflows"].copy()
weather = dataset["weather"].copy()

# 1. Impute missing hourly water-demand values
print("\nImputing missing water-demand values...")
demand_imputer = KNNImputer(n_neighbors=5)
inflows_imputed = pd.DataFrame(
    demand_imputer.fit_transform(inflows),
    index=inflows.index,
    columns=inflows.columns
)

# 2. Impute missing hourly weather values
print("Imputing missing weather values...")
weather_imputer = KNNImputer(n_neighbors=5)
weather_imputed = pd.DataFrame(
    weather_imputer.fit_transform(weather),
    index=weather.index,
    columns=weather.columns
)

# 3. Aggregate hourly demand into daily demand
daily_demand = inflows_imputed.resample("D").mean()

# 4. Aggregate hourly weather into daily features
daily_weather = weather_imputed.resample("D").agg({
    "Rain": "sum",
    "Temperature": "mean",
    "Humidity": "mean",
    "Windspeed": "mean"
})

# 5. Check the results
print("\n========== PREPROCESSING RESULTS ==========")
print("Daily demand shape:", daily_demand.shape)
print("Daily weather shape:", daily_weather.shape)

print("\nRemaining missing demand values:")
print(daily_demand.isna().sum().sum())

print("\nRemaining missing weather values:")
print(daily_weather.isna().sum().sum())

print("\nDaily demand preview:")
print(daily_demand.head())

print("\nDaily weather preview:")
print(daily_weather.head())

# 6. Save intermediate datasets
daily_demand.to_csv("daily_demand.csv")
daily_weather.to_csv("daily_weather.csv")

print("\nSaved daily_demand.csv and daily_weather.csv")
print("Preprocessing complete!")
