
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow:", tf.__version__)
print("Loading feature dataset...")

data = pd.read_csv(
    "dma1_features.csv",
    index_col=0,
    parse_dates=True
).sort_index()

# Separate features and target
feature_columns = [
    f"demand_lag_{i}" for i in range(1, 8)
] + [
    "month",
    "day_of_week",
    "daily_rainfall",
    "daily_temperature"
]

X = data[feature_columns].values.astype(np.float32)
y = data["target_demand"].values.astype(np.float32)
dates = data.index

# Chronological split: first 70% train, next 15% validation,
# final 15% test. This is a baseline split, not the paper's
# exact temporal evaluation protocol.
n = len(data)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, X_val, X_test = (
    X[:train_end],
    X[train_end:val_end],
    X[val_end:]
)

y_train, y_val, y_test = (
    y[:train_end],
    y[train_end:val_end],
    y[val_end:]
)

test_dates = dates[val_end:]

# Fit scalers on training data only
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train = x_scaler.fit_transform(X_train)
X_val = x_scaler.transform(X_val)
X_test = x_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(
    y_train.reshape(-1, 1)
)
y_val_scaled = y_scaler.transform(
    y_val.reshape(-1, 1)
)

# LSTM input shape: samples, time steps, features
# Each row contains engineered features for one forecast day.
X_train = X_train.reshape(len(X_train), 1, len(feature_columns))
X_val = X_val.reshape(len(X_val), 1, len(feature_columns))
X_test = X_test.reshape(len(X_test), 1, len(feature_columns))

print("\nTrain samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))
print("Input shape:", X_train.shape)

# Single-layer, unidirectional LSTM baseline
model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(1, len(feature_columns))
    ),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

print("\nTraining LSTM on CPU...")

model.fit(
    X_train,
    y_train_scaled,
    validation_data=(X_val, y_val_scaled),
    epochs=100,
    batch_size=2,
    callbacks=[early_stopping],
    shuffle=False,
    verbose=1
)

# Predict and return values to original demand units
pred_scaled = model.predict(X_test, verbose=0)
predictions = y_scaler.inverse_transform(pred_scaled).ravel()

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

# MAPE is unstable when actual values are zero or near zero
mape = mean_absolute_percentage_error(y_test, predictions) * 100

print("\n========== TEST RESULTS: DMA 1 ==========")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")

results = pd.DataFrame({
    "date": test_dates,
    "actual_demand": y_test,
    "predicted_demand": predictions
})

results.to_csv("dma1_lstm_predictions.csv", index=False)
model.save("dma1_lstm_baseline.keras")

print("\nSaved predictions to dma1_lstm_predictions.csv")
print("Saved model to dma1_lstm_baseline.keras")
print("Baseline training complete!")
