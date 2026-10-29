
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import re
import joblib
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
LOOKBACK = 14
HORIZON = 7

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001
PATIENCE = 10

OUTPUT_DIR = "lstm_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name)


def make_naive_index(df):
    """Keep local wall-clock timestamps, without timezone info."""
    result = df.copy()
    idx = pd.DatetimeIndex(result.index)

    if idx.tz is not None:
        idx = idx.tz_localize(None)

    result.index = idx
    return result.sort_index()


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual).ravel()
    predicted = np.asarray(predicted).ravel()

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # MAPE is calculated only where actual demand is not near zero.
    valid = np.abs(actual) > 1e-8

    mape = (
        np.mean(
            np.abs(
                (actual[valid] - predicted[valid]) / actual[valid]
            )
        ) * 100
        if valid.any()
        else np.nan
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE_percent": float(mape),
    }


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 65)
print("AquaPredict - LSTM Forecasting for All 10 DMAs")
print("=" * 65)

print("\n[1/6] Loading BWDF dataset...")

raw = bwdf.load_complete_dataset()

inflows = make_naive_index(raw["dma-inflows"])
weather = make_naive_index(raw["weather"])
calendar = make_naive_index(raw["calendar"])

dma_names = list(inflows.columns)

print("DMAs found:", dma_names)
print("Hourly demand shape:", inflows.shape)
print("Hourly weather shape:", weather.shape)


# ============================================================
# 4. CAUSAL IMPUTATION
# ============================================================

print("\n[2/6] Handling missing values...")

# Use the earliest 70% of timestamps to calculate fallback medians.
hourly_cut = int(len(inflows) * TRAIN_RATIO)

demand_fallback = inflows.iloc[:hourly_cut].median()
weather_fallback = weather.iloc[:hourly_cut].median()

# Forward filling uses past observations only.
# Leading missing values use training-period medians.
inflows = inflows.ffill().fillna(demand_fallback)
weather = weather.ffill().fillna(weather_fallback)

if inflows.isna().any().any():
    raise ValueError("Missing values remain in demand data.")

if weather.isna().any().any():
    raise ValueError("Missing values remain in weather data.")

print("Missing values handled without backward filling.")


# ============================================================
# 5. DAILY FEATURES
# ============================================================

print("\n[3/6] Creating daily demand, weather and calendar features...")

daily_demand = inflows.resample("D").mean()

daily_weather = weather.resample("D").agg({
    "Rain": "sum",
    "Temperature": "mean",
})

daily_holiday = (
    calendar["Holiday"]
    .resample("D")
    .max()
    .reindex(daily_demand.index)
    .fillna(0)
)

# Normalize holiday values to numeric 0/1.
holiday_numeric = pd.to_numeric(
    daily_holiday.astype(str).replace({
        "True": "1",
        "False": "0",
    }),
    errors="coerce",
).fillna(0)

daily_holiday = (holiday_numeric > 0).astype(int)

dates = daily_demand.index

common_features = pd.DataFrame(index=dates)
common_features["rain"] = daily_weather["Rain"]
common_features["temperature"] = daily_weather["Temperature"]
common_features["month"] = dates.month
common_features["day_of_week"] = dates.dayofweek
common_features["holiday"] = daily_holiday

# Treat public holidays as Sundays.
common_features.loc[
    common_features["holiday"] == 1, "day_of_week"
] = 6

print("Daily date range:", dates.min().date(), "to", dates.max().date())
print("Daily rows:", len(dates))


# ============================================================
# 6. CHRONOLOGICAL SPLITS
# ============================================================

n_days = len(dates)

train_boundary = int(n_days * TRAIN_RATIO)
val_boundary = int(n_days * (TRAIN_RATIO + VAL_RATIO))

print("\nTrain ends:", dates[train_boundary - 1].date())
print("Validation ends:", dates[val_boundary - 1].date())
print("Test starts:", dates[val_boundary].date())


# ============================================================
# 7. TRAIN AND EVALUATE ONE LSTM PER DMA
# ============================================================

all_overall_metrics = []
all_horizon_metrics = []
all_predictions = []

for dma_number, dma in enumerate(dma_names, start=1):

    print("\n" + "=" * 65)
    print(f"DMA {dma_number}/{len(dma_names)}: {dma}")
    print("=" * 65)

    tf.keras.backend.clear_session()

    # Reset seeds so runs are reproducible.
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Demand history for this DMA plus common daily features.
    frame = common_features.copy()
    frame.insert(0, "demand", daily_demand[dma])

    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()

    feature_columns = list(frame.columns)
    values = frame.to_numpy(dtype=np.float32)
    demand_values = frame["demand"].to_numpy(dtype=np.float32)

    # Construct sequences.
    X_list, y_list, target_starts = [], [], []

    for start in range(len(frame) - LOOKBACK - HORIZON + 1):

        target_start = start + LOOKBACK
        target_end = target_start + HORIZON

        X_list.append(values[start:target_start])
        y_list.append(demand_values[target_start:target_end])
        target_starts.append(target_start)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    target_starts = np.asarray(target_starts, dtype=int)

    # Each complete target horizon stays in its own partition.
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
        print(f"Skipping {dma}: an empty split was created.")
        continue

    print("Train windows:", len(X_train))
    print("Validation windows:", len(X_val))
    print("Test windows:", len(X_test))

    # Fit input scaler on training windows only.
    n_features = X_train.shape[-1]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(X_train.reshape(-1, n_features))

    def scale_X(data):
        original_shape = data.shape
        transformed = x_scaler.transform(
            data.reshape(-1, n_features)
        )
        return transformed.reshape(original_shape).astype(np.float32)

    X_train_s = scale_X(X_train)
    X_val_s = scale_X(X_val)
    X_test_s = scale_X(X_test)

    # Fit target scaler on training targets only.
    y_scaler.fit(y_train.reshape(-1, 1))

    def scale_y(data):
        return y_scaler.transform(
            data.reshape(-1, 1)
        ).reshape(data.shape).astype(np.float32)

    y_train_s = scale_y(y_train)
    y_val_s = scale_y(y_val)

    # LSTM: 14 days of input -> 7 future daily demand values.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(LOOKBACK, n_features)
        ),
        tf.keras.layers.LSTM(16),
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
        patience=PATIENCE,
        restore_best_weights=True,
    )

    print("Training LSTM...")

    history = model.fit(
        X_train_s,
        y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[early_stopping],
        verbose=0,
    )

    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    best_val_loss = float(min(history.history["val_loss"]))

    print("Best epoch:", best_epoch)
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Evaluate on the held-out test period.
    pred_scaled = model.predict(X_test_s, verbose=0)

    predictions = y_scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).reshape(-1, HORIZON)

    actual = y_test

    overall = calculate_metrics(actual, predictions)

    all_overall_metrics.append({
        "dma": dma,
        "train_windows": len(X_train),
        "validation_windows": len(X_val),
        "test_windows": len(X_test),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        **overall,
    })

    print(
        f"Test MAE={overall['MAE']:.4f}, "
        f"RMSE={overall['RMSE']:.4f}, "
        f"MAPE={overall['MAPE_percent']:.2f}%"
    )

    # Save horizon-specific metrics and predictions.
    prediction_rows = []

    for h in range(HORIZON):

        horizon_metrics = calculate_metrics(
            actual[:, h],
            predictions[:, h],
        )

        all_horizon_metrics.append({
            "dma": dma,
            "forecast_day": h + 1,
            "samples": len(actual),
            **horizon_metrics,
        })

    for i, target_start in enumerate(test_starts):

        forecast_origin = frame.index[target_start - 1]

        for h in range(HORIZON):
            target_position = target_start + h

            prediction_rows.append({
                "dma": dma,
                "forecast_origin": forecast_origin,
                "date": frame.index[target_position],
                "horizon_day": h + 1,
                "actual_demand": float(actual[i, h]),
                "predicted_demand": float(predictions[i, h]),
                "error": float(
                    predictions[i, h] - actual[i, h]
                ),
                "absolute_error": float(
                    abs(predictions[i, h] - actual[i, h])
                ),
            })

    all_predictions.extend(prediction_rows)

    # Save this DMA's model and preprocessing configuration.
    prefix = safe_filename(dma)

    model_path = os.path.join(
        OUTPUT_DIR, f"{prefix}_lstm_7day.keras"
    )
    preprocessing_path = os.path.join(
        OUTPUT_DIR, f"{prefix}_preprocessing.joblib"
    )
    history_path = os.path.join(
        OUTPUT_DIR, f"{prefix}_training_history.csv"
    )

    model.save(model_path)

    joblib.dump({
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "feature_columns": feature_columns,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "dma": dma,
    }, preprocessing_path)

    pd.DataFrame(history.history).to_csv(
        history_path, index_label="epoch"
    )

    print("Saved model:", model_path)


# ============================================================
# 8. SAVE GLOBAL RESULTS
# ============================================================

print("\n" + "=" * 65)
print("SAVING ALL-DMA RESULTS")
print("=" * 65)

overall_df = pd.DataFrame(all_overall_metrics)
horizon_df = pd.DataFrame(all_horizon_metrics)
predictions_df = pd.DataFrame(all_predictions)

overall_path = os.path.join(
    OUTPUT_DIR, "all_dmas_overall_metrics.csv"
)
horizon_path = os.path.join(
    OUTPUT_DIR, "all_dmas_horizon_metrics.csv"
)
predictions_path = os.path.join(
    OUTPUT_DIR, "all_dmas_test_predictions.csv"
)

overall_df.to_csv(overall_path, index=False)
horizon_df.to_csv(horizon_path, index=False)
predictions_df.to_csv(predictions_path, index=False)

if not overall_df.empty:
    print("\nOVERALL TEST METRICS BY DMA")
    print(overall_df.round(4).to_string(index=False))

if not horizon_df.empty:
    print("\nHORIZON METRICS")
    print(horizon_df.round(4).to_string(index=False))

print("\nFiles saved:")
print(overall_path)
print(horizon_path)
print(predictions_path)
print("\nAll-DMA LSTM training finished.")
