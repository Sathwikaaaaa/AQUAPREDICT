
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# ==================================================
# 1. LOAD DATA
# ==================================================

DATA_FILE = Path("data/processed/dma3_daily.csv")
OUTPUT_DIR = Path("data/lstm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

# Convert timezone-aware date strings to date-only values.
df["Datetime"] = pd.to_datetime(
    df["Datetime"].astype(str).str[:10],
    errors="raise"
)

df = df.sort_values("Datetime").reset_index(drop=True)

FEATURES = [
    "Demand",
    "Rain",
    "Temperature",
    "Humidity",
    "Windspeed",
    "Holiday",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend"
]

TARGET = "Demand"
LOOKBACK = 14

if df[FEATURES + [TARGET]].isna().any().any():
    raise ValueError("Missing values found. Resolve them before sequence creation.")

# ==================================================
# 2. CHRONOLOGICAL SPLIT
# ==================================================

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

print("=" * 60)
print("AQUAPREDICT - LSTM DATA PREPARATION")
print("=" * 60)

print("Total daily rows:", n)
print("Training period:", df.loc[0, "Datetime"], "to", df.loc[train_end - 1, "Datetime"])
print("Validation period:", df.loc[train_end, "Datetime"], "to", df.loc[val_end - 1, "Datetime"])
print("Test period:", df.loc[val_end, "Datetime"], "to", df.loc[n - 1, "Datetime"])

# Fit scaling parameters only on the training period.
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

feature_scaler.fit(df.loc[:train_end - 1, FEATURES])
target_scaler.fit(df.loc[:train_end - 1, [TARGET]])

scaled_features = feature_scaler.transform(df[FEATURES]).astype(np.float32)
scaled_target = target_scaler.transform(df[[TARGET]]).astype(np.float32).ravel()

# ==================================================
# 3. CREATE SEQUENCES
# ==================================================

X, y, target_indices = [], [], []

for target_idx in range(LOOKBACK, n):
    X.append(scaled_features[target_idx - LOOKBACK:target_idx])
    y.append(scaled_target[target_idx])
    target_indices.append(target_idx)

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.float32)
target_indices = np.asarray(target_indices)

# Assign each sequence according to the date being predicted.
train_mask = target_indices < train_end
val_mask = (target_indices >= train_end) & (target_indices < val_end)
test_mask = target_indices >= val_end

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

print("\nSequence shapes:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:  ", X_val.shape, "y_val:  ", y_val.shape)
print("X_test: ", X_test.shape, "y_test: ", y_test.shape)

print("\nTarget dates:")
print("First train target:", df.loc[target_indices[train_mask][0], "Datetime"])
print("First validation target:", df.loc[target_indices[val_mask][0], "Datetime"])
print("First test target:", df.loc[target_indices[test_mask][0], "Datetime"])

# ==================================================
# 4. SAVE ARRAYS AND SCALERS
# ==================================================


np.savez_compressed(
    OUTPUT_DIR / "lstm_sequences.npz",
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
    test_dates=np.asarray(
        df.loc[target_indices[test_mask], "Datetime"].dt.strftime("%Y-%m-%d"),
        dtype="U10"
    )
)


joblib.dump(feature_scaler, OUTPUT_DIR / "feature_scaler.pkl")
joblib.dump(target_scaler, OUTPUT_DIR / "target_scaler.pkl")

print("\nSaved prepared data to:", OUTPUT_DIR)
print("Sequence preparation completed successfully.")
