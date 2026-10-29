
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_FILE = Path("data/processed/dma3_daily.csv")

df = pd.read_csv(DATA_FILE)
date_col = df.columns[0]
df["Date"] = pd.to_datetime(df[date_col].astype(str).str[:10])
df = df.drop(columns=[date_col]).set_index("Date").sort_index()

df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))
df.index.name = "Date"

# Use the exact same test period as the previous models.
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

data = df.copy()
data["lag1"] = data["Demand"].shift(1)
data["lag7"] = data["Demand"].shift(7)
data["mean7"] = data["Demand"].shift(1).rolling(7).mean()

data["day_of_week"] = data.index.dayofweek
data["month"] = data.index.month
data["is_weekend"] = (data.index.dayofweek >= 5).astype(int)

features = ["lag1", "lag7", "mean7", "day_of_week", "month", "is_weekend"]

X = data[features]
y = data["Demand"]

valid_features = X.notna().all(axis=1) & y.notna()

train_mask = valid_features.copy()
train_mask.iloc[train_end:] = False

val_mask = valid_features.copy()
val_mask.iloc[:train_end] = False
val_mask.iloc[val_end:] = False

test_mask = valid_features.copy()
test_mask.iloc[:val_end] = False

X_train, y_train = X.loc[train_mask], y.loc[train_mask]
X_val, y_val = X.loc[val_mask], y.loc[val_mask]
X_test, y_test = X.loc[test_mask], y.loc[test_mask]

models = {
    "Training Mean": DummyRegressor(strategy="mean"),
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
}

print("Train:", len(X_train), "Validation:", len(X_val), "Test:", len(X_test))
print("\nVALIDATION RESULTS")

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    print(
        name,
        "MAE =", round(mean_absolute_error(y_val, pred), 4),
        "RMSE =", round(np.sqrt(mean_squared_error(y_val, pred)), 4)
    )

print("\nTEST RESULTS")

for name, model in models.items():
    pred = model.predict(X_test)
    print(
        name,
        "MAE =", round(mean_absolute_error(y_test, pred), 4),
        "RMSE =", round(np.sqrt(mean_squared_error(y_test, pred)), 4),
        "R2 =", round(r2_score(y_test, pred), 4)
    )
