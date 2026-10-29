
# ============================================================
# AquaPredict: 7-Day Water Demand Forecasting using LSTM
# Predicts the next 7 days for all 10 DMAs
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import wf4bwdf as bwdf

from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

MODEL_DIR = "lstm_models"
OUTPUT_FILE = os.path.join(
    MODEL_DIR,
    "next_7_days_forecast.csv"
)

NUM_DMAS = 10
TRAIN_RATIO = 0.70

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 65)
print("AquaPredict - LSTM Water Demand Forecasting")
print("=" * 65)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

print("\n[1/5] Loading BWDF dataset...")

raw = bwdf.load_complete_dataset()

inflows = raw["dma-inflows"].copy()
weather = raw["weather"].copy()
calendar = raw["calendar"].copy()

# Ensure datetime indexes and chronological ordering
for dataframe in (inflows, weather, calendar):
    dataframe.index = pd.to_datetime(dataframe.index)
    dataframe.sort_index(inplace=True)

# Check required columns
required_dmas = [f"DMA {i}" for i in range(1, NUM_DMAS + 1)]

missing_dmas = [
    dma for dma in required_dmas
    if dma not in inflows.columns
]

if missing_dmas:
    raise ValueError(
        f"Missing DMA columns in dataset: {missing_dmas}"
    )

required_weather = ["Rain", "Temperature"]

missing_weather = [
    col for col in required_weather
    if col not in weather.columns
]

if missing_weather:
    raise ValueError(
        f"Missing weather columns: {missing_weather}"
    )

if "Holiday" not in calendar.columns:
    raise ValueError(
        "The calendar section does not contain a 'Holiday' column."
    )

print("Dataset loaded successfully.")
print("Hourly demand shape:", inflows.shape)
print("Weather shape:", weather.shape)
print("Calendar shape:", calendar.shape)


# ------------------------------------------------------------
# 3. HANDLE MISSING VALUES
#    Match train_lstm_all_dmas.py
# ------------------------------------------------------------

print("\n[2/5] Handling missing values...")

# Calculate fallback medians using the earliest 70%
# of the hourly demand timestamps.
hourly_cut = int(len(inflows) * TRAIN_RATIO)

demand_fallback = inflows.iloc[:hourly_cut].median()
weather_fallback = weather.iloc[:hourly_cut].median()

# Forward-fill using previous observations only.
# Leading missing values use training-period medians.
inflows = inflows.ffill().fillna(demand_fallback)
weather = weather.ffill().fillna(weather_fallback)

if inflows.isna().any().any():
    raise ValueError(
        "Missing values remain in demand data after imputation."
    )

if weather.isna().any().any():
    raise ValueError(
        "Missing values remain in weather data after imputation."
    )

print("Missing-value handling completed.")


# ------------------------------------------------------------
# 4. CREATE DAILY FEATURES
#    Match train_lstm_all_dmas.py
# ------------------------------------------------------------

print("\n[3/5] Creating daily features...")

# Daily average water demand for each DMA
daily_demand = inflows.resample("D").mean()

# Daily rainfall total and average temperature
daily_weather = weather.resample("D").agg({
    "Rain": "sum",
    "Temperature": "mean",
})

# Daily holiday indicator
daily_holiday = (
    calendar["Holiday"]
    .resample("D")
    .max()
    .reindex(daily_demand.index)
    .fillna(0)
)

# Normalize holiday values to numeric 0/1
holiday_numeric = pd.to_numeric(
    daily_holiday.astype(str).replace({
        "True": "1",
        "False": "0",
    }),
    errors="coerce",
).fillna(0)

daily_holiday = (holiday_numeric > 0).astype(int)

dates = daily_demand.index

# Create shared calendar and weather features
common_features = pd.DataFrame(index=dates)

common_features["rain"] = daily_weather["Rain"]
common_features["temperature"] = daily_weather["Temperature"]
common_features["month"] = dates.month
common_features["day_of_week"] = dates.dayofweek
common_features["holiday"] = daily_holiday

# Treat public holidays as Sundays
common_features.loc[
    common_features["holiday"] == 1,
    "day_of_week"
] = 6

# Prepare one dataframe per DMA.
# Keep all six expected features available; the saved training
# configuration will determine their exact order.
daily_data = {}

for dma_name in required_dmas:
    frame = pd.DataFrame(index=dates)

    frame["demand"] = daily_demand[dma_name]
    frame["rain"] = common_features["rain"]
    frame["temperature"] = common_features["temperature"]
    frame["month"] = common_features["month"]
    frame["day_of_week"] = common_features["day_of_week"]
    frame["holiday"] = common_features["holiday"]

    daily_data[dma_name] = frame

print("Daily features created.")
print("Daily date range:", dates.min().date(), "to", dates.max().date())
print("Number of daily records:", len(dates))


# ------------------------------------------------------------
# 5. LOAD MODELS AND GENERATE FORECASTS
# ------------------------------------------------------------

print("\n[4/5] Loading LSTM models and forecasting...")

all_predictions = []

# Forecast dates begin one day after the dataset ends
last_observed_date = dates[-1]

forecast_dates = pd.date_range(
    start=last_observed_date + pd.Timedelta(days=1),
    periods=7,
    freq="D",
)

for dma_number in range(1, NUM_DMAS + 1):

    dma_name = f"DMA {dma_number}"

    # Convert "DMA 1" into "DMA_1" to match saved filenames
    dma_file_name = dma_name.replace(" ", "_")

    model_path = os.path.join(
        MODEL_DIR,
        f"{dma_file_name}_lstm_7day.keras"
    )

    config_path = os.path.join(
        MODEL_DIR,
        f"{dma_file_name}_preprocessing.joblib"
    )

    print(f"\nProcessing {dma_name}...")

    # Check that model and preprocessing files exist
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Preprocessing configuration not found: {config_path}"
        )

    # Load saved model and preprocessing configuration
    model = load_model(model_path, compile=False)
    config = joblib.load(config_path)

    required_config_keys = [
        "x_scaler",
        "y_scaler",
        "feature_columns",
        "lookback",
        "horizon",
    ]

    missing_keys = [
        key for key in required_config_keys
        if key not in config
    ]

    if missing_keys:
        raise KeyError(
            f"{dma_name} configuration is missing keys: {missing_keys}"
        )

    x_scaler = config["x_scaler"]
    y_scaler = config["y_scaler"]
    feature_columns = list(config["feature_columns"])
    lookback = int(config["lookback"])
    horizon = int(config["horizon"])

    if horizon != 7:
        raise ValueError(
            f"{dma_name} model has horizon={horizon}; expected 7."
        )

    # Verify that every training feature is available
    frame = daily_data[dma_name]

    missing_features = [
        col for col in feature_columns
        if col not in frame.columns
    ]

    if missing_features:
        raise ValueError(
            f"{dma_name} requires unavailable features: {missing_features}"
        )

    # Match the exact feature order used during training
    features = frame.loc[:, feature_columns].to_numpy(
        dtype=np.float64
    )

    # Take the most recent lookback days
    if len(features) < lookback:
        raise ValueError(
            f"Not enough daily history for {dma_name}. "
            f"Need {lookback}, found {len(features)}."
        )

    last_window = features[-lookback:]

    # Scale using the saved training scaler
    scaled_window = x_scaler.transform(last_window)

    # LSTM input shape: (batch, time steps, features)
    X_input = scaled_window.reshape(
        1,
        lookback,
        len(feature_columns),
    ).astype(np.float32)

    # Predict all seven days at once
    scaled_predictions = model.predict(
        X_input,
        verbose=0,
    )

    scaled_predictions = np.asarray(scaled_predictions)

    if scaled_predictions.size != horizon:
        raise ValueError(
            f"{dma_name} returned {scaled_predictions.size} predictions; "
            f"expected {horizon}."
        )

    # Convert predictions back to original demand units
    predictions = y_scaler.inverse_transform(
        scaled_predictions.reshape(-1, 1)
    ).ravel()

    # Demand cannot be negative
    predictions = np.maximum(predictions, 0)

    # Store the seven daily forecasts
    for forecast_date, predicted_demand in zip(
        forecast_dates,
        predictions,
    ):
        all_predictions.append({
            "date": forecast_date.strftime("%Y-%m-%d"),
            "dma": dma_name,
            "predicted_demand": round(float(predicted_demand), 4),
        })

    print(f"  Model loaded: {os.path.basename(model_path)}")
    print(f"  Lookback: {lookback} days")
    print(f"  Features: {feature_columns}")
    print(f"  Forecast generated: {horizon} days")


# ------------------------------------------------------------
# 6. SAVE AND DISPLAY FORECASTS
# ------------------------------------------------------------

print("\n[5/5] Saving forecast results...")

forecast_df = pd.DataFrame(all_predictions)

forecast_df["date"] = pd.to_datetime(forecast_df["date"])

forecast_df.sort_values(
    ["date", "dma"],
    inplace=True,
)

forecast_df["date"] = forecast_df["date"].dt.strftime("%Y-%m-%d")

forecast_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\n" + "=" * 65)
print("7-DAY WATER DEMAND FORECAST")
print("=" * 65)

print(
    forecast_df.pivot(
        index="date",
        columns="dma",
        values="predicted_demand",
    ).to_string()
)

print("\n" + "-" * 65)
print("FORECAST SUMMARY")
print("-" * 65)

print("Number of DMAs:", forecast_df["dma"].nunique())
print("Forecast days:", forecast_df["date"].nunique())
print("Total predictions:", len(forecast_df))
print("Forecast starts:", forecast_df["date"].min())
print("Forecast ends:", forecast_df["date"].max())
print("CSV saved to:", os.path.abspath(OUTPUT_FILE))

if len(forecast_df) != NUM_DMAS * 7:
    raise ValueError(
        f"Expected {NUM_DMAS * 7} forecast rows, "
        f"but generated {len(forecast_df)}."
    )

print("\nSUCCESS: All 10 DMAs have 7-day forecasts!")
