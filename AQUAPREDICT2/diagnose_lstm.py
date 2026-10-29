
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

path = Path("outputs/lstm/lstm_test_predictions.csv")
df = pd.read_csv(path, parse_dates=["Date"])

actual = df["Actual"].to_numpy()
pred = df["LSTM_Predicted"].to_numpy()
errors = pred - actual

print("=" * 55)
print("AQUAPREDICT - LSTM PREDICTION DIAGNOSTICS")
print("=" * 55)

print("\nTest dates:", df["Date"].min(), "to", df["Date"].max())
print("Test samples:", len(df))

print("\nActual demand:")
print(df["Actual"].describe().round(4))

print("\nLSTM predictions:")
print(df["LSTM_Predicted"].describe().round(4))

print("\nPrediction bias (prediction - actual):", round(errors.mean(), 6))
print("Prediction standard deviation:", round(pred.std(), 6))
print("Actual standard deviation:", round(actual.std(), 6))

print("\nMAE:", round(mean_absolute_error(actual, pred), 6))
print("RMSE:", round(np.sqrt(mean_squared_error(actual, pred)), 6))
print("R2:", round(r2_score(actual, pred), 6))

print("\nFirst 15 predictions:")
print(df.head(15).round(4).to_string(index=False))

print("\nLast 15 predictions:")
print(df.tail(15).round(4).to_string(index=False))
