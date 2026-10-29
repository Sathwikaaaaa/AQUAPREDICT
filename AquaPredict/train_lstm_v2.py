
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
import wf4bwdf as bwdf

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

np.random.seed(42)
tf.random.set_seed(42)

print("Loading original BWDF dataset...")
raw = bwdf.load_complete_dataset()

inflows = raw["dma-inflows"].copy()
weather = raw["weather"].copy()
calendar = raw["calendar"].copy()

# Split the HOURLY data chronologically before imputation.
n_hourly = len(inflows)
hourly_cut = int(n_hourly * 0.70)

demand_imputer = KNNImputer(n_neighbors=5)
weather_imputer = KNNImputer(n_neighbors=5)

demand_imputer.fit(inflows.iloc[:hourly_cut])
weather_imputer.fit(weather.iloc[:hourly_cut])

inflows = pd.DataFrame(
    demand_imputer.transform(inflows),
    index=inflows.index,
    columns=inflows.columns
)

weather = pd.DataFrame(
    weather_imputer.transform(weather),
    index=weather.index,
    columns=weather.columns
)

# Aggregate hourly measurements into daily data.
daily_demand = inflows.resample("D").mean()
daily_weather = weather.resample("D").agg({
    "Rain": "sum",
    "Temperature": "mean"
})

# Treat a day containing a public holiday as Sunday.
holiday = calendar["Holiday"].resample("D").max().fillna(False)
day_of_week = pd.Series(
    daily_demand.index.dayofweek,
    index=daily_demand.index
)
day_of_week.loc[holiday.reindex(day_of_week.index).fillna(False).astype(bool)] = 6

# Build features for DMA 1.
dma = "DMA 1"
data = pd.DataFrame(index=daily_demand.index)
data["target"] = daily_demand[dma]

for lag in range(1, 8):
    data[f"lag_{lag}"] = daily_demand[dma].shift(lag)

data["month"] = data.index.month
data["day_of_week"] = day_of_week
data["rain"] = daily_weather["Rain"]
data["temperature"] = daily_weather["Temperature"]
data = data.dropna()

feature_cols = [c for c in data.columns if c != "target"]
X = data[feature_cols].to_numpy(dtype=np.float32)
y = data["target"].to_numpy(dtype=np.float32)
dates = data.index

# Chronological train / validation / test split.
n = len(data)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
test_dates = dates[val_end:]

# Fit normalization on training samples only.
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train)
X_val = x_scaler.transform(X_val)
X_test = x_scaler.transform(X_test)

y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_val_s = y_scaler.transform(y_val.reshape(-1, 1))

# Paper-inspired single-layer, unidirectional LSTM.
# Hyperparameters are tested using validation loss.
best_model = None
best_val_loss = float("inf")
best_config = None

for units in [8, 16, 32]:
    for batch_size in [1, 2, 4]:
        for learning_rate in [0.0001, 0.001, 0.005]:
            tf.keras.backend.clear_session()
            tf.random.set_seed(42)

            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(1, len(feature_cols))),
                tf.keras.layers.LSTM(units),
                tf.keras.layers.Dense(1)
            ])

            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=learning_rate
                ),
                loss="mse"
            )

            stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=6,
                restore_best_weights=True
            )

            model.fit(
                X_train.reshape(-1, 1, len(feature_cols)),
                y_train_s,
                validation_data=(
                    X_val.reshape(-1, 1, len(feature_cols)),
                    y_val_s
                ),
                epochs=50,
                batch_size=batch_size,
                shuffle=False,
                callbacks=[stop],
                verbose=0
            )

            val_loss = model.evaluate(
                X_val.reshape(-1, 1, len(feature_cols)),
                y_val_s,
                verbose=0
            )

            print(
                f"units={units}, batch={batch_size}, "
                f"lr={learning_rate}, val_loss={val_loss:.5f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = tf.keras.models.clone_model(model)
                best_model.set_weights(model.get_weights())
                best_config = (units, batch_size, learning_rate)

print("\nBest configuration:", best_config)
print("Best validation loss:", best_val_loss)

# Test once using the selected model.
X_test_3d = X_test.reshape(-1, 1, len(feature_cols))
pred_s = best_model.predict(X_test_3d, verbose=0)
pred = y_scaler.inverse_transform(pred_s).ravel()

mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, pred) * 100

print("\n========== IMPROVED LSTM RESULTS ==========")
print(f"Test samples: {len(y_test)}")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")

pd.DataFrame({
    "date": test_dates,
    "actual_demand": y_test,
    "predicted_demand": pred
}).to_csv("dma1_lstm_v2_predictions.csv", index=False)

best_model.save("dma1_lstm_v2.keras")
print("\nSaved dma1_lstm_v2.keras and dma1_lstm_v2_predictions.csv")
