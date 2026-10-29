
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import wf4bwdf as bwdf

# from sklearn.impute import KNNImputer
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (
#     mean_absolute_error,
#     mean_squared_error,
#     mean_absolute_percentage_error,
# )

# SEED = 42
# LOOKBACK = 14
# HORIZON = 7
# DMA = "DMA 1"

# np.random.seed(SEED)
# tf.random.set_seed(SEED)

# print("Loading original BWDF dataset...")
# raw = bwdf.load_complete_dataset()

# inflows = raw["dma-inflows"].copy()
# weather = raw["weather"].copy()
# calendar = raw["calendar"].copy()

# # Fit imputers using only the earliest 70% of hourly records.
# cut = int(len(inflows) * 0.70)

# demand_imputer = KNNImputer(n_neighbors=5)
# weather_imputer = KNNImputer(n_neighbors=5)

# demand_imputer.fit(inflows.iloc[:cut])
# weather_imputer.fit(weather.iloc[:cut])

# inflows = pd.DataFrame(
#     demand_imputer.transform(inflows),
#     index=inflows.index,
#     columns=inflows.columns,
# )

# weather = pd.DataFrame(
#     weather_imputer.transform(weather),
#     index=weather.index,
#     columns=weather.columns,
# )

# # Aggregate hourly data to daily data.
# daily_demand = inflows.resample("D").mean()
# daily_weather = weather.resample("D").agg({
#     "Rain": "sum",
#     "Temperature": "mean",
# })

# # Calendar features.
# daily_holiday = (
#     calendar["Holiday"]
#     .resample("D")
#     .max()
#     .fillna(False)
#     .astype(bool)
# )

# dates = daily_demand.index
# day_of_week = pd.Series(dates.dayofweek, index=dates)

# # Treat public holidays as Sundays.
# day_of_week.loc[daily_holiday] = 6

# # Build features for the pilot DMA.
# frame = pd.DataFrame(index=dates)

# frame["demand"] = daily_demand[DMA]
# frame["rain"] = daily_weather["Rain"]
# frame["temperature"] = daily_weather["Temperature"]
# frame["month"] = dates.month
# frame["day_of_week"] = day_of_week
# frame["holiday"] = daily_holiday.astype(int)

# frame = frame.replace([np.inf, -np.inf], np.nan).dropna()

# feature_columns = list(frame.columns)
# values = frame.to_numpy(dtype=np.float32)
# demand_values = frame["demand"].to_numpy(dtype=np.float32)

# # Each sample uses 14 historical days to predict the next 7 days.
# X, y, target_starts = [], [], []

# for start in range(len(frame) - LOOKBACK - HORIZON + 1):
#     target_start = start + LOOKBACK
#     target_end = target_start + HORIZON

#     X.append(values[start:target_start])
#     y.append(demand_values[target_start:target_end])
#     target_starts.append(target_start)

# X = np.asarray(X, dtype=np.float32)
# y = np.asarray(y, dtype=np.float32)
# target_starts = np.asarray(target_starts)

# n_days = len(frame)
# train_boundary = int(n_days * 0.70)
# val_boundary = int(n_days * 0.85)

# # Keep each complete 7-day target block inside one partition.
# train_mask = target_starts + HORIZON <= train_boundary

# val_mask = (
#     (target_starts >= train_boundary)
#     & (target_starts + HORIZON <= val_boundary)
# )

# test_mask = target_starts >= val_boundary

# X_train, y_train = X[train_mask], y[train_mask]
# X_val, y_val = X[val_mask], y[val_mask]
# X_test, y_test = X[test_mask], y[test_mask]

# if min(len(X_train), len(X_val), len(X_test)) == 0:
#     raise ValueError("A data split is empty. Check dataset coverage.")

# print("\nDataset shapes:")
# print("Train:", X_train.shape, y_train.shape)
# print("Validation:", X_val.shape, y_val.shape)
# print("Test:", X_test.shape, y_test.shape)

# # Fit feature scaling on training sequences only.
# x_scaler = StandardScaler()
# y_scaler = StandardScaler()

# n_features = X_train.shape[-1]

# x_scaler.fit(X_train.reshape(-1, n_features))

# def scale_inputs(array):
#     shape = array.shape
#     scaled = x_scaler.transform(array.reshape(-1, n_features))
#     return scaled.reshape(shape).astype(np.float32)

# X_train_s = scale_inputs(X_train)
# X_val_s = scale_inputs(X_val)
# X_test_s = scale_inputs(X_test)

# # Fit the target scaler only on training-period targets.
# y_scaler.fit(y_train.reshape(-1, 1))

# y_train_s = y_scaler.transform(
#     y_train.reshape(-1, 1)
# ).reshape(y_train.shape)

# y_val_s = y_scaler.transform(
#     y_val.reshape(-1, 1)
# ).reshape(y_val.shape)

# # Single-layer, unidirectional LSTM with 7 output values.
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(LOOKBACK, n_features)),
#     tf.keras.layers.LSTM(16),
#     tf.keras.layers.Dense(HORIZON),
# ])

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#     loss="mse",
# )

# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor="val_loss",
#     patience=10,
#     restore_best_weights=True,
# )

# print("\nTraining LSTM pilot...")

# model.fit(
#     X_train_s,
#     y_train_s,
#     validation_data=(X_val_s, y_val_s),
#     epochs=100,
#     batch_size=16,
#     shuffle=False,
#     callbacks=[early_stopping],
#     verbose=1,
# )

# # Evaluate on future dates that were not used for training or tuning.
# pred_scaled = model.predict(X_test_s, verbose=0)

# predictions = y_scaler.inverse_transform(
#     pred_scaled.reshape(-1, 1)
# ).reshape(-1, HORIZON)

# actual = y_test

# mae = mean_absolute_error(actual.ravel(), predictions.ravel())
# mse = mean_squared_error(actual.ravel(), predictions.ravel())
# rmse = np.sqrt(mse)

# # MAPE can be unstable if actual demand is zero or near zero.
# mape = mean_absolute_percentage_error(
#     actual.ravel(), predictions.ravel()
# ) * 100

# print("\n========== DMA 1: 7-DAY LSTM RESULTS ==========")
# print(f"MAE:  {mae:.4f}")
# print(f"MSE:  {mse:.4f}")
# print(f"RMSE: {rmse:.4f}")
# print(f"MAPE: {mape:.2f}%")
# print(f"Test forecast windows: {len(X_test)}")

# # Save each forecast origin and its seven daily predictions.
# output = []

# for i in range(len(X_test)):
#     origin = target_starts[test_mask][i]

#     for horizon_day in range(HORIZON):
#         target_position = origin + horizon_day
#         output.append({
#             "forecast_origin": frame.index[origin - 1],
#             "date": frame.index[target_position],
#             "horizon_day": horizon_day + 1,
#             "actual_demand": actual[i, horizon_day],
#             "predicted_demand": predictions[i, horizon_day],
#         })

# pd.DataFrame(output).to_csv(
#     "dma1_lstm_7day_test_predictions.csv",
#     index=False,
# )

# model.save("dma1_lstm_7day.keras")

# print("\nSaved model: dma1_lstm_7day.keras")
# print("Saved test predictions: dma1_lstm_7day_test_predictions.csv")
# print("Pilot training complete.")




import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import numpy as np
import pandas as pd
import tensorflow as tf
import wf4bwdf as bwdf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
DMA = "DMA 1"

LOOKBACK = 14       # Use the previous 14 days
HORIZON = 7         # Predict the next 7 days

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 60)
print("AquaPredict - LSTM Water Demand Forecasting")
print("=" * 60)


# ============================================================
# 2. LOAD THE ORIGINAL DATASET
# ============================================================

print("\n[1/8] Loading BWDF dataset...")

raw = bwdf.load_complete_dataset()

inflows = raw["dma-inflows"].copy()
weather = raw["weather"].copy()
calendar = raw["calendar"].copy()

# Use consistent timestamps.
# The dataset has timezone-aware timestamps; convert to local
# wall-clock timestamps and remove timezone information.
def make_naive_index(df):
    idx = pd.DatetimeIndex(df.index)

    if idx.tz is not None:
        idx = idx.tz_localize(None)

    result = df.copy()
    result.index = idx
    return result.sort_index()

inflows = make_naive_index(inflows)
weather = make_naive_index(weather)
calendar = make_naive_index(calendar)

if DMA not in inflows.columns:
    raise ValueError(f"{DMA} is not available in the dataset.")

print("Hourly demand shape:", inflows.shape)
print("Hourly weather shape:", weather.shape)
print("Hourly calendar shape:", calendar.shape)


# ============================================================
# 3. CAUSAL MISSING-VALUE HANDLING
# ============================================================

print("\n[2/8] Handling missing values without backward filling...")

# Fit fallback medians on the earliest 70% of hourly records.
# Do not calculate fallback values from the full dataset.
hourly_cut = int(len(inflows) * TRAIN_RATIO)

demand_fallback = inflows.iloc[:hourly_cut].median()
weather_fallback = weather.iloc[:hourly_cut].median()

# Forward fill uses only the most recent available observation.
# Remaining leading gaps are filled with training-period medians.
inflows = inflows.ffill().fillna(demand_fallback)
weather = weather.ffill().fillna(weather_fallback)

if inflows.isna().any().any():
    raise ValueError("Demand data still contains missing values.")

if weather.isna().any().any():
    raise ValueError("Weather data still contains missing values.")

print("Missing values handled.")
print("No backward fill or full-series KNN imputation is used.")


# ============================================================
# 4. CONVERT HOURLY DATA TO DAILY DATA
# ============================================================

print("\n[3/8] Aggregating hourly observations to daily data...")

daily_demand = inflows.resample("D").mean()

daily_weather = weather.resample("D").agg({
    "Rain": "sum",
    "Temperature": "mean",
})

daily_calendar = calendar.resample("D").agg({
    "Holiday": "max",
})

# Build one daily feature table for the selected DMA.
dates = daily_demand.index

frame = pd.DataFrame(index=dates)

frame["demand"] = daily_demand[DMA]
frame["rain"] = daily_weather["Rain"]
frame["temperature"] = daily_weather["Temperature"]

frame["month"] = dates.month
frame["day_of_week"] = dates.dayofweek

# Holiday values can be booleans or numeric indicators.
holiday = (
    daily_calendar["Holiday"]
    .reindex(dates)
    .fillna(0)
)

holiday_numeric = pd.to_numeric(
    holiday.astype(str).replace({
        "True": "1",
        "False": "0",
    }),
    errors="coerce",
).fillna(0)

frame["holiday"] = (holiday_numeric > 0).astype(int)

# Treat public holidays as Sundays.
frame.loc[frame["holiday"] == 1, "day_of_week"] = 6

frame = frame.replace([np.inf, -np.inf], np.nan).dropna()

if len(frame) < LOOKBACK + HORIZON + 30:
    raise ValueError("Not enough daily data to train and evaluate the model.")

print("Daily dataset shape:", frame.shape)
print("Date range:", frame.index.min().date(), "to", frame.index.max().date())


# ============================================================
# 5. CREATE SEQUENTIAL INPUTS AND 7-DAY TARGETS
# ============================================================

print("\n[4/8] Creating LSTM sequences...")

feature_columns = [
    "demand",
    "rain",
    "temperature",
    "month",
    "day_of_week",
    "holiday",
]

values = frame[feature_columns].to_numpy(dtype=np.float32)
demand_values = frame["demand"].to_numpy(dtype=np.float32)

X_list = []
y_list = []
target_starts = []

for start in range(len(frame) - LOOKBACK - HORIZON + 1):

    target_start = start + LOOKBACK
    target_end = target_start + HORIZON

    # Shape per sample: (14 days, 6 features)
    X_list.append(values[start:target_start])

    # Seven consecutive future daily demand values.
    y_list.append(demand_values[target_start:target_end])

    target_starts.append(target_start)

X = np.asarray(X_list, dtype=np.float32)
y = np.asarray(y_list, dtype=np.float32)
target_starts = np.asarray(target_starts, dtype=int)

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 6. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n[5/8] Splitting data chronologically...")

n_days = len(frame)

train_boundary = int(n_days * TRAIN_RATIO)
val_boundary = int(n_days * (TRAIN_RATIO + VAL_RATIO))

# Keep each full target horizon within one partition.
train_mask = (target_starts + HORIZON) <= train_boundary

val_mask = (
    (target_starts >= train_boundary)
    & ((target_starts + HORIZON) <= val_boundary)
)

test_mask = target_starts >= val_boundary

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

test_starts = target_starts[test_mask]

if min(len(X_train), len(X_val), len(X_test)) == 0:
    raise ValueError("An empty split was created. Check dataset size.")

print("Training windows:", len(X_train))
print("Validation windows:", len(X_val))
print("Testing windows:", len(X_test))

print("Training period ends:", frame.index[train_boundary - 1].date())
print("Validation period ends:", frame.index[val_boundary - 1].date())
print("Test period starts:", frame.index[val_boundary].date())


# ============================================================
# 7. SCALE INPUTS AND TARGETS WITHOUT TEST DATA
# ============================================================

print("\n[6/8] Scaling the data...")

n_features = X_train.shape[-1]

x_scaler = StandardScaler()
y_scaler = StandardScaler()

# Fit feature scaler on training windows only.
x_scaler.fit(X_train.reshape(-1, n_features))

def scale_X(data):
    original_shape = data.shape

    transformed = x_scaler.transform(
        data.reshape(-1, n_features)
    )

    return transformed.reshape(original_shape).astype(np.float32)

X_train_scaled = scale_X(X_train)
X_val_scaled = scale_X(X_val)
X_test_scaled = scale_X(X_test)

# Fit target scaler using training-period targets only.
y_scaler.fit(y_train.reshape(-1, 1))

def scale_y(data):
    return y_scaler.transform(
        data.reshape(-1, 1)
    ).reshape(data.shape).astype(np.float32)

y_train_scaled = scale_y(y_train)
y_val_scaled = scale_y(y_val)


# ============================================================
# 8. BUILD, TRAIN AND EVALUATE THE LSTM
# ============================================================

print("\n[7/8] Training the LSTM...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(LOOKBACK, n_features)
    ),
    tf.keras.layers.LSTM(
        16,
        activation="tanh",
        recurrent_activation="sigmoid",
    ),
    tf.keras.layers.Dense(HORIZON),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="mse",
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)

history = model.fit(
    X_train_scaled,
    y_train_scaled,
    validation_data=(X_val_scaled, y_val_scaled),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=False,
    callbacks=[early_stopping],
    verbose=1,
)

# Predict the untouched test windows.
pred_scaled = model.predict(X_test_scaled, verbose=0)

predictions = y_scaler.inverse_transform(
    pred_scaled.reshape(-1, 1)
).reshape(-1, HORIZON)

actual = y_test

# Overall test metrics across all seven horizons.
mae = mean_absolute_error(
    actual.ravel(),
    predictions.ravel(),
)

mse = mean_squared_error(
    actual.ravel(),
    predictions.ravel(),
)

rmse = np.sqrt(mse)

# Exclude zero or near-zero actual values for MAPE.
nonzero = np.abs(actual.ravel()) > 1e-8

mape = (
    np.mean(
        np.abs(
            (
                actual.ravel()[nonzero]
                - predictions.ravel()[nonzero]
            )
            / actual.ravel()[nonzero]
        )
    ) * 100
    if nonzero.any()
    else np.nan
)

print("\n" + "=" * 60)
print("DMA 1 - SEVEN-DAY TEST RESULTS")
print("=" * 60)

print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")
print("Test windows:", len(X_test))


# ============================================================
# 9. SAVE PREDICTIONS AND HORIZON METRICS
# ============================================================

print("\n[8/8] Saving model, predictions and metrics...")

prediction_rows = []

for i, target_start in enumerate(test_starts):

    forecast_origin = frame.index[target_start - 1]

    for h in range(HORIZON):
        target_position = target_start + h

        prediction_rows.append({
            "dma": DMA,
            "forecast_origin": forecast_origin,
            "date": frame.index[target_position],
            "horizon_day": h + 1,
            "actual_demand": float(actual[i, h]),
            "predicted_demand": float(predictions[i, h]),
            "error": float(predictions[i, h] - actual[i, h]),
            "absolute_error": float(
                abs(predictions[i, h] - actual[i, h])
            ),
        })

prediction_df = pd.DataFrame(prediction_rows)

prediction_df.to_csv(
    "dma1_lstm_7day_test_predictions.csv",
    index=False,
)

# Per-horizon evaluation.
metric_rows = []

for h in range(HORIZON):
    y_true = actual[:, h]
    y_pred = predictions[:, h]

    h_mae = mean_absolute_error(y_true, y_pred)
    h_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    valid = np.abs(y_true) > 1e-8

    h_mape = (
        np.mean(
            np.abs(
                (y_true[valid] - y_pred[valid]) / y_true[valid]
            )
        ) * 100
        if valid.any()
        else np.nan
    )

    metric_rows.append({
        "forecast_day": h + 1,
        "MAE": h_mae,
        "RMSE": h_rmse,
        "MAPE_percent": h_mape,
        "samples": len(y_true),
    })

metrics_df = pd.DataFrame(metric_rows)

metrics_df.to_csv(
    "dma1_lstm_horizon_metrics.csv",
    index=False,
)

# Save model and preprocessing parameters.
model.save("dma1_lstm_7day.keras")

import joblib

joblib.dump(
    {
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "feature_columns": feature_columns,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "dma": DMA,
    },
    "dma1_lstm_preprocessing.joblib",
)

# Save training history for later inspection.
pd.DataFrame(history.history).to_csv(
    "dma1_lstm_training_history.csv",
    index_label="epoch",
)

print("\nPERFORMANCE BY FORECAST DAY")
print(metrics_df.round(4).to_string(index=False))

print("\nFiles saved:")
print("1. dma1_lstm_7day.keras")
print("2. dma1_lstm_preprocessing.joblib")
print("3. dma1_lstm_7day_test_predictions.csv")
print("4. dma1_lstm_horizon_metrics.csv")
print("5. dma1_lstm_training_history.csv")

print("\nPilot complete!")
