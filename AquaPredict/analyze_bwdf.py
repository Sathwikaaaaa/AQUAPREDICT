import wf4bwdf

dataset = wf4bwdf.load_complete_dataset()

print("Available sections:", dataset.keys())

for name, df in dataset.items():
    print("\nSection:", name)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head())
    print("Missing values:\n", df.isnull().sum())