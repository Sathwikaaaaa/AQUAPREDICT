
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

OUT = Path("data/lstm_demand_only")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/processed/dma3_daily.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"].astype(str).str[:10])
df = df.sort_values("Datetime").reset_index(drop=True)

values = df[["Demand"]].to_numpy(dtype=np.float32)

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)
lookback = 14

# Fit the scaler on training targets only.
scaler = StandardScaler()
scaler.fit(values[:train_end])
scaled = scaler.transform(values).astype(np.float32)

X, y, indices = [], [], []

for i in range(lookback, n):
    X.append(scaled[i-lookback:i])
    y.append(scaled[i, 0])
    indices.append(i)

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.float32)
indices = np.asarray(indices)

train = indices < train_end
val = (indices >= train_end) & (indices < val_end)
test = indices >= val_end

np.savez_compressed(
    OUT / "sequences.npz",
    X_train=X[train], y_train=y[train],
    X_val=X[val], y_val=y[val],
    X_test=X[test], y_test=y[test],
    test_dates=df.loc[indices[test], "Datetime"].dt.strftime("%Y-%m-%d").to_numpy(dtype="U10")
)

joblib.dump(scaler, OUT / "target_scaler.pkl")

print("Demand-only LSTM data prepared.")
print("Train:", X[train].shape, y[train].shape)
print("Validation:", X[val].shape, y[val].shape)
print("Test:", X[test].shape, y[test].shape)
print("Test period:", df.loc[indices[test], "Datetime"].min(),
      "to", df.loc[indices[test], "Datetime"].max())
