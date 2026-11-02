
import numpy as np
from pathlib import Path
from src.data.load_dataset import load_wafer_dataset
from src.preprocessing.wafer_map import create_valid_wafer_mask
from src.preprocessing.profiles import (
    extract_radial_profile,
    extract_angular_profile,
)

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "LSWMD.pkl"

df = load_wafer_dataset(file_path=DATASET_PATH)

print("Dataset shape:", df.shape)

# Inspect a sample of actual wafer maps.
sample_indices = [0, 100, 1000, 10000, 50000]

for idx in sample_indices:
    wafer = np.asarray(df.iloc[idx]["waferMap"])

    print(f"\nOriginal row index: {idx}")
    print("Map shape:", wafer.shape)
    print("Unique values:", np.unique(wafer))
    print("Value counts:", {
        int(v): int(c)
        for v, c in zip(*np.unique(wafer, return_counts=True))
    })

    mask = create_valid_wafer_mask(wafer)
    print("Valid wafer positions:", int(mask.sum()))

    radial = extract_radial_profile(wafer)
    angular = extract_angular_profile(wafer)

    print("Radial profile shape:", radial.shape)
    print("Angular profile shape:", angular.shape)
    print("Radial profile:", radial)
    print("Angular profile:", angular)

print("\nEncoding inspection completed.")
