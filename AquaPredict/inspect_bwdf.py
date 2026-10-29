
import wf4bwdf as bwdf

print("Loading BWDF dataset...")

dataset = bwdf.load_complete_dataset()

print("\nDataset loaded successfully!")
print("Available sections:", list(dataset.keys()))

for name, data in dataset.items():
    print(f"\n--- {name} ---")
    print("Type:", type(data).__name__)

    if hasattr(data, "shape"):
        print("Shape:", data.shape)
        print("Columns:", list(data.columns))
        print(data.head())
    elif isinstance(data, dict):
        print("Subsections:", list(data.keys())[:20])
    else:
        print(data)
