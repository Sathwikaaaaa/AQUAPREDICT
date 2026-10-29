
import wf4bwdf
import pandas as pd

dataset = wf4bwdf.load_complete_dataset()

inflows = dataset["dma-inflows"].copy()
weather = dataset["weather"].copy()
calendar = dataset["calendar"].copy()

print("=" * 60)
print("AQUAPREDICT - DATA QUALITY REPORT")
print("=" * 60)

print("\n1. TIMESTAMP INFORMATION")
print("First timestamp:", inflows.index.min())
print("Last timestamp :", inflows.index.max())
print("Total rows     :", len(inflows))
print("Index unique   :", inflows.index.is_unique)
print("Index sorted   :", inflows.index.is_monotonic_increasing)

print("\n2. MISSING DATA PERCENTAGES")
print("\nDMA inflows:")
print((inflows.isna().mean() * 100).round(2))

print("\nWeather:")
print((weather.isna().mean() * 100).round(2))

print("\n3. CONTINUOUS MISSING GAPS")
for column in inflows.columns:
    missing = inflows[column].isna()

    # Identify consecutive missing-value groups.
    groups = missing.ne(missing.shift()).cumsum()
    gap_lengths = missing.groupby(groups).sum()
    gap_lengths = gap_lengths[gap_lengths > 0]

    print(f"\n{column}")
    print("Number of missing groups:", len(gap_lengths))
    print("Longest missing gap (hours):",
          int(gap_lengths.max()) if len(gap_lengths) else 0)

print("\n4. INFLOW SUMMARY")
print(inflows.describe().round(3))

print("\n5. WEATHER SUMMARY")
print(weather.describe().round(3))

print("\n6. DUPLICATE TIMESTAMPS")
print("Duplicate inflow timestamps:",
      inflows.index.duplicated().sum())

print("\n7. CALENDAR COLUMNS")
print(calendar.dtypes)
print("\n", calendar.head())

print("\nInspection complete.")
