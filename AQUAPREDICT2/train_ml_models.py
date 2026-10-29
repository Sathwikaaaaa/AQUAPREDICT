
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# 1. LOAD DAILY DATA
# ============================================================

DATA_FILE = Path("data/processed/dma3_daily.csv")
OUTPUT_DIR = Path("outputs/ml_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

date_col = df.columns[0]
df["Date"] = pd.to_datetime(
    df[date_col].astype(str).str[:10],
    errors="raise"
)

df = df.drop(columns=[date_col])
df = df.set_index("Date").sort_index()

# Restore missing dates without filling target values.
df = df.reindex(
    pd.date_range(
        df.index.min(),
        df.index.max(),
        freq="D"
    )
)
df.index.name = "Date"

# ============================================================
# 2. CREATE PREDICTIVE FEATURES
# ============================================================

data = df.copy()

# Previous observed demand.
data["demand_lag_1"] = data["Demand"].shift(1)
data["demand_lag_2"] = data["Demand"].shift(2)
data["demand_lag_7"] = data["Demand"].shift(7)
data["demand_lag_14"] = data["Demand"].shift(14)

# Rolling statistics use only previous observations.
data["mean_7_days"] = (
    data["Demand"].shift(1).rolling(7).mean()
)
data["std_7_days"] = (
    data["Demand"].shift(1).rolling(7).std()
)
data["mean_14_days"] = (
    data["Demand"].shift(1).rolling(14).mean()
)

# Calendar information is known in advance.
data["day_of_week"] = data.index.dayofweek
data["month"] = data.index.month
data["day_of_year"] = data.index.dayofyear
data["is_weekend"] = (
    data.index.dayofweek >= 5
).astype(int)

# Use past weather measurements, not target-day observations.
for column in ["Rain", "Temperature", "Humidity", "Windspeed"]:
    data[f"{column}_lag_1"] = df[column].shift(1)

FEATURES = [
    "demand_lag_1",
    "demand_lag_2",
    "demand_lag_7",
    "demand_lag_14",
    "mean_7_days",
    "std_7_days",
    "mean_14_days",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "Holiday",
    "Rain_lag_1",
    "Temperature_lag_1",
    "Humidity_lag_1",
    "Windspeed_lag_1"
]

TARGET = "Demand"

# ============================================================
# 3. CHRONOLOGICAL SPLIT
# ============================================================

n = len(data)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_dates = data.index[:train_end]
val_dates = data.index[train_end:val_end]
test_dates = data.index[val_end:]

X = data[FEATURES]
y = data[TARGET]

# The imputer and scaler will be fitted using training data only.
# No future target values are used to construct the features.

train_mask = (
    data.index.isin(train_dates)
    & y.notna()
    & X.notna().all(axis=1)
)

val_mask = (
    data.index.isin(val_dates)
    & y.notna()
    & X.notna().all(axis=1)
)

test_mask = (
    data.index.isin(test_dates)
    & y.notna()
    & X.notna().all(axis=1)
)

X_train, y_train = X.loc[train_mask], y.loc[train_mask]
X_val, y_val = X.loc[val_mask], y.loc[val_mask]
X_test, y_test = X.loc[test_mask], y.loc[test_mask]

print("=" * 60)
print("AQUAPREDICT - ML MODEL TRAINING")
print("=" * 60)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))

if min(len(X_train), len(X_val), len(X_test)) == 0:
    raise ValueError("One of the data splits has no valid samples.")

# ============================================================
# 4. DEFINE MODELS
# ============================================================

models = {
    "Linear Regression": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LinearRegression()
    ),

    "Random Forest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
    )
}

# ============================================================
# 5. TRAIN AND VALIDATE
# ============================================================

trained_models = {}
validation_results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    trained_models[name] = model

    val_pred = model.predict(X_val)

    val_mae = mean_absolute_error(y_val, val_pred)
    val_rmse = np.sqrt(
        mean_squared_error(y_val, val_pred)
    )

    validation_results.append({
        "Model": name,
        "Validation_MAE": val_mae,
        "Validation_RMSE": val_rmse
    })

    print("Validation MAE:", round(val_mae, 6))
    print("Validation RMSE:", round(val_rmse, 6))

# ============================================================
# 6. EVALUATE ON THE TEST PERIOD
# ============================================================

test_results = []
prediction_frame = pd.DataFrame(index=X_test.index)
prediction_frame["Actual"] = y_test

for name, model in trained_models.items():
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    nonzero = y_test.to_numpy() != 0
    mape = (
        np.mean(
            np.abs(
                (y_test.to_numpy()[nonzero]
                 - predictions[nonzero])
                / y_test.to_numpy()[nonzero]
            )
        ) * 100
        if nonzero.any()
        else np.nan
    )

    r2 = r2_score(y_test, predictions)

    test_results.append({
        "Model": name,
        "Test_Samples": len(y_test),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape,
        "R2": r2
    })

    prediction_frame[name] = predictions

    print(f"\n{name} - TEST RESULTS")
    print("MAE:", round(mae, 6))
    print("RMSE:", round(rmse, 6))
    print("MAPE (%):", round(mape, 6))
    print("R2:", round(r2, 6))

# ============================================================
# 7. SAVE REPORTS
# ============================================================

pd.DataFrame(validation_results).to_csv(
    OUTPUT_DIR / "validation_metrics.csv",
    index=False
)

pd.DataFrame(test_results).to_csv(
    OUTPUT_DIR / "test_metrics.csv",
    index=False
)

prediction_frame.to_csv(
    OUTPUT_DIR / "test_predictions.csv"
)

# Plot actual vs predictions.
fig, ax = plt.subplots(figsize=(13, 5))

ax.plot(
    prediction_frame.index,
    prediction_frame["Actual"],
    label="Actual",
    linewidth=2
)

for name in trained_models:
    ax.plot(
        prediction_frame.index,
        prediction_frame[name],
        label=name,
        alpha=0.8
    )

ax.set_title("DMA 3 - ML Model Test Predictions")
ax.set_xlabel("Date")
ax.set_ylabel("Daily mean flow (source units)")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "ml_model_comparison.png",
    dpi=150
)
plt.close(fig)

print("\nVALIDATION COMPARISON")
print(pd.DataFrame(validation_results).to_string(index=False))

print("\nTEST COMPARISON")
print(pd.DataFrame(test_results).to_string(index=False))

print("\nSaved results in:", OUTPUT_DIR)
print("ML model training and evaluation completed.")
