
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_FILE = Path("data/processed/dma3_daily.csv")
OUTPUT_DIR = Path("outputs/baselines")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

# Preserve local calendar dates and avoid mixed UTC offsets.
date_column = df.columns[0]
df["Date"] = pd.to_datetime(
    df[date_column].astype(str).str[:10],
    errors="raise"
)

df = df.drop(columns=[date_column])
df = df.set_index("Date").sort_index()

if not df.index.is_unique:
    raise ValueError("Duplicate daily dates found.")

print("=" * 60)
print("AQUAPREDICT - BASELINE FORECASTING")
print("=" * 60)

print("Loaded records:", len(df))
print("Start date:", df.index.min())
print("End date:", df.index.max())


# ============================================================
# 2. RESTORE MISSING CALENDAR DAYS
# ============================================================

# Missing target dates must remain missing.
# Do not drop the dates and accidentally make nonconsecutive
# observations look like consecutive days.

full_dates = pd.date_range(
    start=df.index.min(),
    end=df.index.max(),
    freq="D"
)

df = df.reindex(full_dates)
df.index.name = "Date"

print("\nFull calendar days:", len(df))
print("Days with missing Demand:", df["Demand"].isna().sum())


# ============================================================
# 3. CHRONOLOGICAL SPLIT
# ============================================================

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_dates = df.index[:train_end]
val_dates = df.index[train_end:val_end]
test_dates = df.index[val_end:]

print("\nChronological split:")
print(
    "Train:",
    train_dates.min().date(),
    "to",
    train_dates.max().date()
)
print(
    "Validation:",
    val_dates.min().date(),
    "to",
    val_dates.max().date()
)
print(
    "Test:",
    test_dates.min().date(),
    "to",
    test_dates.max().date()
)


# ============================================================
# 4. CREATE BASELINE PREDICTIONS
# ============================================================

actual = df["Demand"]

# Yesterday's observed value predicts today's value.
pred_persistence = actual.shift(1)

# The observation seven calendar days ago predicts today.
pred_weekly = actual.shift(7)

results = []

def evaluate_baseline(name, predictions, dates):
    actual_values = actual.loc[dates]
    predicted_values = predictions.loc[dates]

    # Evaluate only dates with a valid target and the required
    # historical observation.
    valid = (
        actual_values.notna()
        & predicted_values.notna()
    )

    y_true = actual_values[valid].to_numpy()
    y_pred = predicted_values[valid].to_numpy()

    if len(y_true) == 0:
        print(f"\n{name}: no valid test predictions.")
        return None

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE is only valid for nonzero actual targets.
    nonzero = y_true != 0
    mape = (
        np.mean(
            np.abs(
                (y_true[nonzero] - y_pred[nonzero])
                / y_true[nonzero]
            )
        ) * 100
        if nonzero.any()
        else np.nan
    )

    r2 = (
        r2_score(y_true, y_pred)
        if len(y_true) >= 2
        else np.nan
    )

    row = {
        "Model": name,
        "Evaluated_Days": len(y_true),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape,
        "R2": r2
    }

    results.append(row)

    print(f"\n{name}")
    for key, value in row.items():
        print(f"{key}: {value}")

    return pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_pred
    }, index=actual_values[valid].index)


# ============================================================
# 5. EVALUATE ON THE UNTOUCHED TEST PERIOD
# ============================================================

persistence_test = evaluate_baseline(
    "Persistence",
    pred_persistence,
    test_dates
)

weekly_test = evaluate_baseline(
    "Weekly seasonal",
    pred_weekly,
    test_dates
)


# ============================================================
# 6. SAVE RESULTS
# ============================================================

metrics = pd.DataFrame(results)

metrics.to_csv(
    OUTPUT_DIR / "baseline_metrics.csv",
    index=False
)

if persistence_test is not None:
    persistence_test.to_csv(
        OUTPUT_DIR / "persistence_predictions.csv"
    )

if weekly_test is not None:
    weekly_test.to_csv(
        OUTPUT_DIR / "weekly_predictions.csv"
    )

print("\n" + "=" * 60)
print("BASELINE COMPARISON")
print("=" * 60)
print(metrics.to_string(index=False))

# Plot the actual and predicted values for the test period.
fig, ax = plt.subplots(figsize=(13, 5))

test_actual = actual.loc[test_dates].dropna()

ax.plot(
    test_actual.index,
    test_actual.values,
    label="Actual demand",
    linewidth=2
)

if persistence_test is not None:
    ax.plot(
        persistence_test.index,
        persistence_test["Predicted"],
        label="Persistence prediction",
        alpha=0.8
    )

if weekly_test is not None:
    ax.plot(
        weekly_test.index,
        weekly_test["Predicted"],
        label="Weekly seasonal prediction",
        alpha=0.8
    )

ax.set_title("DMA 3: Baseline Forecast Evaluation")
ax.set_xlabel("Date")
ax.set_ylabel("Daily mean flow (source units)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "baseline_comparison.png",
    dpi=150
)

plt.close(fig)

print("\nSaved metrics and prediction files to:", OUTPUT_DIR)
print("Saved plot: baseline_comparison.png")
print("\nBaseline evaluation completed.")
