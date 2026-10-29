
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from wf4bwdf import load_complete_dataset


# ============================================================
# 1. CONFIGURATION
# ============================================================
SEED = 42
LOOKBACK = 28
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001
MAX_EPOCHS = 150
PATIENCE = 15
MIN_VALID_HOURS = 18

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path("outputs/multidma_lstm")
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("AQUAPREDICT - MULTI-DMA LSTM")
print("=" * 65)
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. LOAD ALL 10 DMAs AND SOURCE FEATURES
# ============================================================
sections = load_complete_dataset()

inflows = sections["dma-inflows"].copy()
weather = sections["weather"].copy()
calendar = sections["calendar"].copy()

# Ensure a common, sorted timestamp index.
for frame in (inflows, weather, calendar):
    frame.index = pd.to_datetime(frame.index)
    frame.sort_index(inplace=True)

# Normalize column names to strings.
inflows.columns = inflows.columns.astype(str)
weather.columns = weather.columns.astype(str)
calendar.columns = calendar.columns.astype(str)

dma_cols = [f"DMA {i}" for i in range(1, 11)]
missing_dmas = [c for c in dma_cols if c not in inflows.columns]
if missing_dmas:
    raise ValueError(
        f"Expected all 10 DMA columns; missing: {missing_dmas}. "
        f"Found: {inflows.columns.tolist()}"
    )

inflows = inflows[dma_cols].apply(pd.to_numeric, errors="coerce")

weather_cols = ["Rain", "Temperature", "Humidity", "Windspeed"]
missing_weather = [c for c in weather_cols if c not in weather.columns]
if missing_weather:
    raise ValueError(
        f"Missing weather columns: {missing_weather}"
    )

weather = weather[weather_cols].apply(
    pd.to_numeric, errors="coerce"
)

# ============================================================
# 3. AGGREGATE HOURLY DATA TO DAILY DATA
# ============================================================
# Demand target: daily mean of observed hourly flow.
daily_demand = inflows.resample("D").mean()
valid_hours = inflows.resample("D").count()

# Mark low-coverage days as missing targets for that DMA.
daily_demand = daily_demand.where(
    valid_hours >= MIN_VALID_HOURS
)

daily_weather = pd.DataFrame(index=weather.resample("D").mean().index)
daily_weather["Rain"] = weather["Rain"].resample("D").sum(min_count=1)
for col in ["Temperature", "Humidity", "Windspeed"]:
    daily_weather[col] = weather[col].resample("D").mean()

if "Holiday" in calendar.columns:
    daily_holiday = calendar["Holiday"].resample("D").max()
else:
    daily_holiday = pd.Series(
        0.0, index=daily_demand.index, name="Holiday"
    )

daily = daily_demand.join(daily_weather, how="outer")
daily["Holiday"] = daily_holiday
daily = daily.sort_index()

# Add deterministic calendar features.
dow = daily.index.dayofweek.to_numpy()
doy = daily.index.dayofyear.to_numpy()

daily["dow_sin"] = np.sin(2 * np.pi * dow / 7)
daily["dow_cos"] = np.cos(2 * np.pi * dow / 7)
daily["year_sin"] = np.sin(2 * np.pi * doy / 365.25)
daily["year_cos"] = np.cos(2 * np.pi * doy / 365.25)

daily = daily.replace([np.inf, -np.inf], np.nan)

print("\nDaily dataset:", daily.shape)
print("Date range:", daily.index.min(), "to", daily.index.max())
print("Observed target values by DMA:")
print(daily[dma_cols].notna().sum())


# ============================================================
# 4. CHRONOLOGICAL SPLIT
# ============================================================
n = len(daily)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

if n <= LOOKBACK + 10:
    raise ValueError("Not enough daily data for this lookback window.")

train_dates = daily.index[:train_end]
val_dates = daily.index[train_end:val_end]
test_dates = daily.index[val_end:]

print("\nChronological periods")
print("Train:", train_dates.min().date(), "to", train_dates.max().date())
print("Validation:", val_dates.min().date(), "to", val_dates.max().date())
print("Test:", test_dates.min().date(), "to", test_dates.max().date())


# ============================================================
# 5. PREPROCESS WITHOUT TEST-DATA LEAKAGE
# ============================================================
feature_cols = dma_cols + weather_cols + [
    "Holiday", "dow_sin", "dow_cos", "year_sin", "year_cos"
]

features = daily[feature_cols].copy()

# Learn input fill values from training period only.
train_features = features.iloc[:train_end]
fill_values = train_features.median().fillna(0.0)

# Forward-fill is causal: it uses past observations only.
features = features.ffill()
features = features.fillna(fill_values)
features = features.fillna(0.0)

# Fit feature scaling on training rows only.
x_mean = features.iloc[:train_end].mean()
x_std = features.iloc[:train_end].std().replace(0, 1).fillna(1)
X = ((features - x_mean) / x_std).to_numpy(dtype=np.float32)

# Target scaling is calculated separately for each DMA,
# using observed target values from the training period only.
raw_y = daily[dma_cols].to_numpy(dtype=np.float64)
train_y = raw_y[:train_end]

y_mean = np.nanmean(train_y, axis=0)
y_std = np.nanstd(train_y, axis=0)
y_std = np.where(np.isfinite(y_std) & (y_std > 1e-8), y_std, 1.0)
y_mean = np.where(np.isfinite(y_mean), y_mean, 0.0)

Y = ((raw_y - y_mean) / y_std).astype(np.float32)
Y_mask = np.isfinite(raw_y)

# Missing targets must not enter the loss.
Y = np.where(Y_mask, Y, 0.0).astype(np.float32)


# ============================================================
# 6. BUILD SEQUENCES BY TARGET DATE
# ============================================================
def make_sequences(target_indices):
    xs, ys, masks, indices = [], [], [], []

    for target_i in target_indices:
        start = target_i - LOOKBACK
        if start < 0:
            continue

        # Every input is strictly before its target date.
        xs.append(X[start:target_i])
        ys.append(Y[target_i])
        masks.append(Y_mask[target_i])
        indices.append(target_i)

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(masks, dtype=bool),
        np.asarray(indices, dtype=int),
    )


train_idx = np.arange(LOOKBACK, train_end)
val_idx = np.arange(train_end, val_end)
test_idx = np.arange(val_end, n)

X_train, y_train, m_train, i_train = make_sequences(train_idx)
X_val, y_val, m_val, i_val = make_sequences(val_idx)
X_test, y_test, m_test, i_test = make_sequences(test_idx)

print("\nSequence shapes")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

if not m_train.any() or not m_val.any() or not m_test.any():
    raise ValueError(
        "One split has no observed targets. Inspect data coverage and split."
    )

def make_loader(x, y, mask, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(mask.astype(np.float32)),
    )
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=shuffle
    )

train_loader = make_loader(X_train, y_train, m_train, True)
val_loader = make_loader(X_val, y_val, m_val, False)


# ============================================================
# 7. MULTI-OUTPUT LSTM
# ============================================================
class MultiDMALSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        sequence_output, _ = self.lstm(x)
        return self.head(sequence_output[:, -1, :])


model = MultiDMALSTM(
    input_size=X_train.shape[2],
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    output_size=len(dma_cols),
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)


def masked_mse(pred, target, mask):
    squared_error = (pred - target) ** 2
    return (squared_error * mask).sum() / mask.sum().clamp(min=1)


def evaluate_loss(loader):
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            pred = model(xb)
            loss = masked_mse(pred, yb, mb)
            count = mb.sum().item()
            total_loss += loss.item() * count
            total_count += count

    return total_loss / max(total_count, 1)


# ============================================================
# 8. TRAIN WITH EARLY STOPPING
# ============================================================
best_val = float("inf")
best_state = None
best_epoch = 0
wait = 0
history = []

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    train_loss_sum = 0.0
    train_count = 0

    for xb, yb, mb in train_loader:
        xb, yb, mb = xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = masked_mse(pred, yb, mb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        count = mb.sum().item()
        train_loss_sum += loss.item() * count
        train_count += count

    train_loss = train_loss_sum / max(train_count, 1)
    val_loss = evaluate_loss(val_loader)
    scheduler.step(val_loss)

    history.append({
        "epoch": epoch,
        "train_masked_mse": train_loss,
        "validation_masked_mse": val_loss,
        "learning_rate": optimizer.param_groups[0]["lr"],
    })

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Train: {train_loss:.5f} | "
            f"Validation: {val_loss:.5f}"
        )

    if val_loss < best_val:
        best_val = val_loss
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch
        wait = 0
    else:
        wait += 1

    if wait >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        break

if best_state is None:
    raise RuntimeError("Training failed to produce a valid checkpoint.")

model.load_state_dict(best_state)
torch.save(model.state_dict(), OUT / "best_multidma_lstm.pt")
pd.DataFrame(history).to_csv(OUT / "training_history.csv", index=False)

print("\nBest epoch:", best_epoch)
print("Best validation masked MSE:", round(best_val, 6))


# ============================================================
# 9. FINAL TEST EVALUATION
# ============================================================
model.eval()
with torch.no_grad():
    pred_scaled = model(
        torch.from_numpy(X_test).to(DEVICE)
    ).cpu().numpy()

pred_raw = pred_scaled * y_std + y_mean
actual_raw = raw_y[i_test]

prediction_frame = pd.DataFrame({"Date": daily.index[i_test]})
for j, dma in enumerate(dma_cols):
    prediction_frame[f"{dma}_Actual"] = actual_raw[:, j]
    prediction_frame[f"{dma}_Predicted"] = pred_raw[:, j]

prediction_frame.to_csv(OUT / "test_predictions.csv", index=False)

metric_rows = []
all_actual = []
all_predicted = []

for j, dma in enumerate(dma_cols):
    observed = np.isfinite(actual_raw[:, j])
    actual = actual_raw[observed, j]
    predicted = pred_raw[observed, j]

    if len(actual) == 0:
        continue

    metric_rows.append({
        "DMA": dma,
        "Test_Samples": len(actual),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "MAPE_percent": (
            np.mean(np.abs((actual - predicted) / actual)) * 100
            if np.all(actual != 0) else np.nan
        ),
        "R2": r2_score(actual, predicted) if len(actual) > 1 else np.nan,
        "Mean_Error_PredMinusActual": np.mean(predicted - actual),
    })

    all_actual.extend(actual.tolist())
    all_predicted.extend(predicted.tolist())

metrics = pd.DataFrame(metric_rows)
metrics.to_csv(OUT / "per_dma_metrics.csv", index=False)

overall = pd.DataFrame([{
    "Model": "Multi-DMA LSTM",
    "Test_Target_Values": len(all_actual),
    "MAE": mean_absolute_error(all_actual, all_predicted),
    "RMSE": np.sqrt(mean_squared_error(all_actual, all_predicted)),
    "MAPE_percent": (
        np.mean(np.abs(
            (np.asarray(all_actual) - np.asarray(all_predicted))
            / np.asarray(all_actual)
        )) * 100
        if np.all(np.asarray(all_actual) != 0) else np.nan
    ),
    "R2": r2_score(all_actual, all_predicted),
    "Best_Epoch": best_epoch,
    "Best_Validation_Masked_MSE": best_val,
}])
overall.to_csv(OUT / "overall_metrics.csv", index=False)

print("\nPER-DMA TEST RESULTS")
print(metrics.round(4).to_string(index=False))
print("\nOVERALL TEST RESULTS")
print(overall.round(4).to_string(index=False))

print("\nSaved outputs in:", OUT)
print(" - best_multidma_lstm.pt")
print(" - training_history.csv")
print(" - test_predictions.csv")
print(" - per_dma_metrics.csv")
print(" - overall_metrics.csv")
