
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

print("=" * 60)
print("AQUAPREDICT - LSTM TRAINING")
print("=" * 60)
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# 2. LOAD PREPARED SEQUENCES
# ============================================================

DATA_DIR = Path("data/lstm")
OUTPUT_DIR = Path("outputs/lstm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data = np.load(DATA_DIR / "lstm_sequences.npz")

X_train = data["X_train"]
y_train = data["y_train"]
X_val = data["X_val"]
y_val = data["y_val"]
X_test = data["X_test"]
y_test_scaled = data["y_test"]
test_dates = pd.to_datetime(data["test_dates"])

target_scaler = joblib.load(DATA_DIR / "target_scaler.pkl")

print("\nSequence shapes:")
print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test_scaled.shape)

# ============================================================
# 3. CONVERT TO PYTORCH TENSORS
# ============================================================

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

X_test_t = torch.tensor(X_test, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=32,
    shuffle=True
)

# ============================================================
# 4. DEFINE THE LSTM
# ============================================================

class DemandLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x):
        sequence_output, _ = self.lstm(x)

        # Use the output corresponding to the final input day.
        last_output = sequence_output[:, -1, :]
        return self.regressor(last_output)


model = DemandLSTM(
    input_size=X_train.shape[2],
    hidden_size=32
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nModel architecture:")
print(model)

# ============================================================
# 5. TRAIN WITH VALIDATION-BASED EARLY STOPPING
# ============================================================

EPOCHS = 150
PATIENCE = 15

best_val_loss = float("inf")
best_state = None
epochs_without_improvement = 0
history = []

for epoch in range(1, EPOCHS + 1):
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
            batch_X = X_val_t[start:start + 32].to(device)
            batch_y = y_val_t[start:start + 32].to(device)

            predictions = model(batch_X)
            val_losses.append(criterion(predictions, batch_y).item())

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
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Train loss: {train_loss:.5f} | "
            f"Validation loss: {val_loss:.5f}"
        )

    if epochs_without_improvement >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}.")
        break

if best_state is None:
    raise RuntimeError("Training did not produce a valid checkpoint.")

model.load_state_dict(best_state)
torch.save(model.state_dict(), OUTPUT_DIR / "best_lstm.pt")

pd.DataFrame(history).to_csv(
    OUTPUT_DIR / "training_history.csv",
    index=False
)

print("\nBest validation MSE (scaled target):", round(best_val_loss, 6))

# ============================================================
# 6. EVALUATE ON THE HELD-OUT TEST SET
# ============================================================

model.eval()

with torch.no_grad():
    test_predictions_scaled = model(
        X_test_t.to(device)
    ).cpu().numpy().reshape(-1)

# Convert predictions and targets back to source units.
test_predictions = target_scaler.inverse_transform(
    test_predictions_scaled.reshape(-1, 1)
).reshape(-1)

test_actual = target_scaler.inverse_transform(
    y_test_scaled.reshape(-1, 1)
).reshape(-1)

mae = mean_absolute_error(test_actual, test_predictions)
rmse = np.sqrt(mean_squared_error(test_actual, test_predictions))
r2 = r2_score(test_actual, test_predictions)

nonzero = test_actual != 0
mape = (
    np.mean(
        np.abs(
            (test_actual[nonzero] - test_predictions[nonzero])
            / test_actual[nonzero]
        )
    ) * 100
    if nonzero.any()
    else np.nan
)

print("\nLSTM TEST RESULTS")
print("Test samples:", len(test_actual))
print("MAE:", round(mae, 6))
print("RMSE:", round(rmse, 6))
print("MAPE (%):", round(mape, 6))
print("R2:", round(r2, 6))

predictions_df = pd.DataFrame({
    "Date": test_dates,
    "Actual": test_actual,
    "LSTM_Predicted": test_predictions
})

predictions_df.to_csv(
    OUTPUT_DIR / "lstm_test_predictions.csv",
    index=False
)

pd.DataFrame([{
    "Model": "LSTM",
    "Test_Samples": len(test_actual),
    "MAE": mae,
    "RMSE": rmse,
    "MAPE_percent": mape,
    "R2": r2,
    "Best_Validation_MSE_Scaled": best_val_loss
}]).to_csv(
    OUTPUT_DIR / "lstm_test_metrics.csv",
    index=False
)

print("\nSaved model and results to:", OUTPUT_DIR)
print("LSTM training and evaluation completed.")
