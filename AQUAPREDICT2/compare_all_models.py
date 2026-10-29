
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = Path("outputs")
ML_DIR = BASE / "ml_models"
BASE_DIR = BASE / "baselines"

# Read the existing baseline predictions.
persistence = pd.read_csv(
    BASE_DIR / "persistence_predictions.csv",
    parse_dates=["Date"]
).set_index("Date")

weekly = pd.read_csv(
    BASE_DIR / "weekly_predictions.csv",
    parse_dates=["Date"]
).set_index("Date")

# Read predictions from the original ML script.
original_ml = pd.read_csv(
    ML_DIR / "test_predictions.csv",
    parse_dates=["Date"]
).set_index("Date")

# Load the daily target and recreate the simplified model.
df = pd.read_csv("data/processed/dma3_daily.csv")
date_col = df.columns[0]
df["Date"] = pd.to_datetime(df[date_col].astype(str).str[:10])
df = df.drop(columns=[date_col]).set_index("Date").sort_index()
df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))

data = df.copy()
data["lag1"] = data["Demand"].shift(1)
data["lag7"] = data["Demand"].shift(7)
data["mean7"] = data["Demand"].shift(1).rolling(7).mean()
data["day_of_week"] = data.index.dayofweek
data["month"] = data.index.month
data["is_weekend"] = (data.index.dayofweek >= 5).astype(int)

features = ["lag1", "lag7", "mean7", "day_of_week", "month", "is_weekend"]
valid = data[features].notna().all(axis=1) & data["Demand"].notna()

n = len(data)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_mask = valid.copy()
train_mask.iloc[train_end:] = False

test_mask = valid.copy()
test_mask.iloc[:val_end] = False

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(data.loc[train_mask, features], data.loc[train_mask, "Demand"])

simplified_pred = pd.Series(
    model.predict(data.loc[test_mask, features]),
    index=data.index[test_mask],
    name="Simplified Linear Regression"
)

# Align every model on the exact same dates and actual target values.
comparison = pd.DataFrame({
    "Actual": persistence["Actual"],
    "Persistence": persistence["Predicted"],
    "Weekly Seasonal": weekly["Predicted"],
    "Original Linear Regression": original_ml["Linear Regression"],
    "Simplified Linear Regression": simplified_pred
}).dropna()

print("Common test samples:", len(comparison))
print("First test date:", comparison.index.min())
print("Last test date:", comparison.index.max())

results = []

for name in comparison.columns:
    if name == "Actual":
        continue

    actual = comparison["Actual"].to_numpy()
    pred = comparison[name].to_numpy()

    results.append({
        "Model": name,
        "Samples": len(comparison),
        "MAE": mean_absolute_error(actual, pred),
        "RMSE": np.sqrt(mean_squared_error(actual, pred)),
        "R2": r2_score(actual, pred)
    })

results = pd.DataFrame(results).sort_values("MAE")
print("\nFINAL COMMON-DATE COMPARISON")
print(results.to_string(index=False))

comparison.to_csv(BASE / "common_date_predictions.csv")
results.to_csv(BASE / "common_date_metrics.csv", index=False)

print("\nSaved common-date predictions and metrics.")
