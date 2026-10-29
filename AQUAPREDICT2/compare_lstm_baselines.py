
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# File paths
baseline_path = Path("outputs/common_date_predictions.csv")
lstm_path = Path("outputs/lstm_demand_only/lstm_demand_only_predictions.csv")

# Load predictions
baseline = pd.read_csv(baseline_path, index_col=0)
lstm = pd.read_csv(lstm_path)

# Convert dates
baseline.index = pd.to_datetime(baseline.index)
baseline.index.name = "Date"
lstm["Date"] = pd.to_datetime(lstm["Date"])
lstm = lstm.set_index("Date")

# Merge on dates shared by both files
merged = baseline.join(
    lstm[["Actual", "LSTM_DemandOnly"]],
    how="inner",
    rsuffix="_lstm"
)

# Check that actual values agree
if not np.allclose(
    merged["Actual"].to_numpy(),
    merged["Actual_lstm"].to_numpy(),
    atol=1e-6
):
    raise ValueError("Actual demand values do not match between files.")

actual = merged["Actual"].to_numpy()

models = {
    "Persistence": "Persistence",
    "Weekly Seasonal": "Weekly Seasonal",
    "Original Linear Regression": "Original Linear Regression",
    "Simplified Linear Regression": "Simplified Linear Regression",
    "Demand-Only LSTM": "LSTM_DemandOnly"
}

results = []

for name, column in models.items():
    predicted = merged[column].to_numpy()

    results.append({
        "Model": name,
        "Test_Samples": len(merged),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "MAPE_percent": np.mean(
            np.abs((actual - predicted) / actual)
        ) * 100,
        "R2": r2_score(actual, predicted),
        "Mean_Error": np.mean(predicted - actual)
    })

results_df = pd.DataFrame(results).sort_values("MAE")

# Save results
Path("outputs").mkdir(exist_ok=True)
results_df.to_csv("outputs/fair_model_comparison.csv", index=False)
merged.to_csv("outputs/fair_model_predictions.csv")

print("=" * 65)
print("AQUAPREDICT - FAIR MODEL COMPARISON")
print("=" * 65)
print("Common test dates:", len(merged))
print("Test period:", merged.index.min().date(), "to", merged.index.max().date())
print("\nResults (sorted by MAE; lower is better):")
print(results_df.to_string(
    index=False,
    formatters={
        "MAE": "{:.5f}".format,
        "RMSE": "{:.5f}".format,
        "MAPE_percent": "{:.3f}".format,
        "R2": "{:.4f}".format,
        "Mean_Error": "{:+.5f}".format
    }
))

print("\nSaved:")
print("outputs/fair_model_comparison.csv")
print("outputs/fair_model_predictions.csv")
