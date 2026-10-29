
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT = "lstm_models/all_dmas_test_predictions.csv"
OUTPUT = "lstm_models/all_dmas_nonoverlapping_metrics.csv"

df = pd.read_csv(
    INPUT,
    parse_dates=["forecast_origin", "date"],
)

results = []

for dma, group in df.groupby("dma"):
    origins = sorted(group["forecast_origin"].unique())

    # Select one forecast origin every 7 days.
    selected_origins = set(origins[::7])
    selected = group[
        group["forecast_origin"].isin(selected_origins)
    ].copy()

    actual = selected["actual_demand"].to_numpy()
    predicted = selected["predicted_demand"].to_numpy()

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    valid = np.abs(actual) > 1e-8
    mape = (
        np.mean(
            np.abs(
                (actual[valid] - predicted[valid]) / actual[valid]
            )
        ) * 100
        if valid.any()
        else np.nan
    )

    results.append({
        "dma": dma,
        "forecast_windows": len(selected_origins),
        "daily_predictions": len(selected),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_percent": mape,
    })

results_df = pd.DataFrame(results).sort_values("dma")
results_df.to_csv(OUTPUT, index=False)

print("\n===== NON-OVERLAPPING EVALUATION: ALL DMAs =====")
print(results_df.round(4).to_string(index=False))
print(f"\nSaved: {OUTPUT}")
