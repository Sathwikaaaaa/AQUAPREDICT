
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

FILE = "dma1_lstm_7day_test_predictions.csv"

df = pd.read_csv(FILE, parse_dates=["forecast_origin", "date"])

# Evaluate each forecast horizon separately.
results = []

for day, group in df.groupby("horizon_day"):
    actual = group["actual_demand"].to_numpy()
    predicted = group["predicted_demand"].to_numpy()

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # Avoid misleading percentage errors when actual demand is near zero.
    nonzero = np.abs(actual) > 1e-8
    mape = (
        np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero])
                / actual[nonzero]
            )
        ) * 100
        if nonzero.any()
        else np.nan
    )

    results.append({
        "forecast_day": day,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape,
        "samples": len(group),
    })

metrics = pd.DataFrame(results)

print("\n===== PERFORMANCE BY FORECAST DAY =====")
print(metrics.round(4).to_string(index=False))

metrics.to_csv("dma1_lstm_horizon_metrics.csv", index=False)

# Plot MAE by forecast horizon.
plt.figure(figsize=(9, 5))
plt.plot(
    metrics["forecast_day"],
    metrics["MAE"],
    marker="o",
)
plt.xticks(range(1, 8))
plt.xlabel("Forecast horizon (days ahead)")
plt.ylabel("MAE")
plt.title("DMA 1: LSTM Error by Forecast Day")
plt.grid(True)
plt.tight_layout()
plt.savefig("dma1_lstm_horizon_mae.png", dpi=150)
plt.show()

# Plot one complete seven-day forecast window.
# Choose the latest available test window.
latest_origin = df["forecast_origin"].max()
window = df[df["forecast_origin"] == latest_origin].sort_values("horizon_day")

plt.figure(figsize=(9, 5))
plt.plot(
    window["horizon_day"],
    window["actual_demand"],
    marker="o",
    label="Actual demand",
)
plt.plot(
    window["horizon_day"],
    window["predicted_demand"],
    marker="o",
    label="LSTM prediction",
)
plt.xticks(range(1, 8))
plt.xlabel("Forecast day")
plt.ylabel("Daily demand")
plt.title(f"DMA 1: Seven-Day Forecast from {latest_origin.date()}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("dma1_lstm_example_forecast.png", dpi=150)
plt.show()

print("\nSaved:")
print("- dma1_lstm_horizon_metrics.csv")
print("- dma1_lstm_horizon_mae.png")
print("- dma1_lstm_example_forecast.png")
