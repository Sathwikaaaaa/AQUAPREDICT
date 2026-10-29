
import pandas as pd
from pathlib import Path

path = Path("data/processed/dma3_daily.csv")
df = pd.read_csv(path)

print("=" * 55)
print("AQUAPREDICT - LSTM DATA INSPECTION")
print("=" * 55)

print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head().to_string())

print("\nLast 5 rows:")
print(df.tail().to_string())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nDemand statistics:")
print(df["Demand"].describe())

print("\nDate column:", df.columns[0])
print("First date:", df.iloc[0, 0])
print("Last date:", df.iloc[-1, 0])

print("\nGPU check:")
import torch
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")
