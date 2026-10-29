
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv(
    "dma1_lstm_7day_test_predictions.csv",
    parse_dates=["forecast_origin", "date"],
)

# Keep one forecast origin every 7 days.
origins = sorted(df["forecast_origin"].unique())
selected_origins = set(origins[::7])

weekly = df[df["forecast_origin"].isin(selected_origins)].copy()

# Overall metrics on the selected forecast windows.
actual = weekly["actual_demand"].to_numpy()
predicted = weekly["predicted_demand"].to_numpy()

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))

valid = np.abs(actual) > 1e-8
mape = (
    np.mean(np.abs((actual[valid] - predicted[valid]) / actual[valid]))
    * 100
    if valid.any()
    else np.nan
)

print("DMA 1 - Non-overlapping weekly forecast evaluation")
print("Forecast windows:", len(selected_origins))
print("Forecasted daily values:", len(weekly))
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")

weekly.to_csv("dma1_lstm_nonoverlapping_test.csv", index=False)
print("Saved: dma1_lstm_nonoverlapping_test.csv")
