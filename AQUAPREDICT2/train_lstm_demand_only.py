
import copy
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. REPRODUCIBILITY AND DEVICE
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

DATA_DIR = Path("data/lstm_demand_only")
OUTPUT_DIR = Path("outputs/lstm_demand_only")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("AQUAPREDICT - DEMAND-ONLY LSTM")
print("=" * 60)
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# 2. LOAD SEQUENCES
# ============================================================

data = np.load(DATA_DIR / "sequences.npz")

X_train = data["X_train"]
y_train = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]
X_test = data["X_test"]
y_test = data["y_test"]

test_dates = pd.to_datetime(data["test_dates"])
scaler = joblib.load(DATA_DIR / "target_scaler.pkl")

print("\nTrain:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# ============================================================
# 3. PYTORCH DATA
# ============================================================

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

# ============================================================
# 4. LSTM MODEL
# ============================================================

class DemandOnlyLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        sequence_output, _ = self.lstm(x)
        return self.output_layer(sequence_output[:, -1, :])


model = DemandOnlyLSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ============================================================
# 5. TRAIN WITH EARLY STOPPING
# ============================================================

MAX_EPOCHS = 150
PATIENCE = 15
best_val_loss = float("inf")
best_state = None
wait = 0
history = []

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    train_losses = []

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    model.eval()
    val_losses = []

    with torch.no_grad():
        for start in range(0, len(X_val_t), 32):
            bx = X_val_t[start:start + 32].to(device)
            by = y_val_t[start:start + 32].to(device)
            val_losses.append(criterion(model(bx), by).item())

    train_loss = float(np.mean(train_losses))
    val_loss = float(np.mean(val_losses))

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Train loss: {train_loss:.5f} | "
            f"Validation loss: {val_loss:.5f}"
        )

    if wait >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        break

if best_state is None:
    raise RuntimeError("No valid model checkpoint was created.")

model.load_state_dict(best_state)
torch.save(model.state_dict(), OUTPUT_DIR / "best_lstm.pt")

pd.DataFrame(history).to_csv(
    OUTPUT_DIR / "training_history.csv",
    index=False
)

print("\nBest validation MSE (scaled target):", round(best_val_loss, 6))

# ============================================================
# 6. TEST EVALUATION
# ============================================================

model.eval()

with torch.no_grad():
    pred_scaled = model(X_test_t.to(device)).cpu().numpy().reshape(-1)

# Convert scaled values back to the original demand units.
pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
actual = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

mae = mean_absolute_error(actual, pred)
rmse = np.sqrt(mean_squared_error(actual, pred))
r2 = r2_score(actual, pred)

nonzero = actual != 0
mape = (
    np.mean(np.abs((actual[nonzero] - pred[nonzero]) / actual[nonzero])) * 100
    if nonzero.any()
    else np.nan
)

print("\nDEMAND-ONLY LSTM TEST RESULTS")
print("Test samples:", len(actual))
print("MAE:", round(mae, 6))
print("RMSE:", round(rmse, 6))
print("MAPE (%):", round(mape, 6))
print("R2:", round(r2, 6))

pd.DataFrame({
    "Date": test_dates,
    "Actual": actual,
    "LSTM_DemandOnly": pred
}).to_csv(
    OUTPUT_DIR / "lstm_demand_only_predictions.csv",
    index=False
)

pd.DataFrame([{
    "Model": "Demand-only LSTM",
    "Test_Samples": len(actual),
    "MAE": mae,
    "RMSE": rmse,
    "MAPE_percent": mape,
    "R2": r2,
    "Best_Validation_MSE_Scaled": best_val_loss
}]).to_csv(
    OUTPUT_DIR / "lstm_demand_only_metrics.csv",
    index=False
)

print("\nSaved results to:", OUTPUT_DIR)
print("Demand-only LSTM training completed.")
