
from pathlib import Path
import wf4bwdf as bwdf

# Create folders
output_dir = Path("data") / "raw"
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading BWDF dataset...")
dataset = bwdf.load_complete_dataset()

# Export each DataFrame to a separate CSV
for name, df in dataset.items():
    if hasattr(df, "to_csv"):
        file_path = output_dir / f"{name}.csv"
        df.to_csv(file_path, index=True)
        print(f"Saved: {file_path} | Shape: {df.shape}")

print("\nDataset export completed!")
