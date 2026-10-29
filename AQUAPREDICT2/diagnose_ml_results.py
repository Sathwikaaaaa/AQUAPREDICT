
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUTPUT = Path("outputs")
ML_DIR = OUTPUT / "ml_models"
BASE_DIR = OUTPUT / "baselines"

# --------------------------------------------------
# 1. Inspect saved ML predictions
# --------------------------------------------------
ml = pd.read_csv(
    ML_DIR / "test_predictions.csv",
    parse_dates=["Date"]
).set_index("Date")

print("\n========== ML TEST DATA ==========")
print("Number of rows:", len(ml))
print("First test date:", ml.index.min())
print("Last test date:", ml.index.max())
print("\nMissing values:")
print(ml.isna().sum())

actual = ml["Actual"].to_numpy()

print("\nActual demand statistics:")
print(pd.Series(actual).describe())

print("\n========== MODEL DIAGNOSTICS ==========")

for model_name in ["Linear Regression", "Random Forest"]:
    pred = ml[model_name].to_numpy()
    errors = pred - actual

    print(f"\n{model_name}")
    print("Prediction mean:", round(pred.mean(), 6))
    print("Actual mean:", round(actual.mean(), 6))
    print("Mean error (prediction - actual):", round(errors.mean(), 6))
    print("Prediction standard deviation:", round(pred.std(), 6))
    print("Actual standard deviation:", round(actual.std(), 6))
    print("MAE:", round(mean_absolute_error(actual, pred), 6))
    print("RMSE:", round(np.sqrt(mean_squared_error(actual, pred)), 6))
    print("R2:", round(r2_score(actual, pred), 6))

# --------------------------------------------------
# 2. Compare exact dates against baseline predictions
# --------------------------------------------------
baseline_files = {
    "Persistence": BASE_DIR / "persistence_predictions.csv",
    "Weekly seasonal": BASE_DIR / "weekly_predictions.csv"
}

print("\n========== COMMON-DATE COMPARISON ==========")

common = ml[["Actual", "Linear Regression", "Random Forest"]].copy()

for name, path in baseline_files.items():
    baseline = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")

    print(f"\n{name} columns:", baseline.columns.tolist())
    print("Baseline rows:", len(baseline))
    print("Overlapping dates:", len(ml.index.intersection(baseline.index)))

    # Inspect column names before selecting the prediction column.
    print(baseline.head(2))

print("\nDiagnostic completed.")
