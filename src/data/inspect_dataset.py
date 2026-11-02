
"""
WM-811K Dataset Inspection and Inventory
Team 18 | Track E | Card E3 | RevD2

Inspects the original dataset and generates a wafer-level inventory.
The original pickle is never modified.
"""

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from src.data.load_dataset import load_wafer_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "LSWMD.pkl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "wafer_inventory.csv"


def normalize_single_string(value):
    """
    RevD2 normalization rule:
    - Plain string: retain it.
    - Array containing exactly one string: extract that string.
    - Anything else: treat as unlabelled.
    """
    if isinstance(value, str):
        return value.strip() or None

    if isinstance(value, np.ndarray) and value.size == 1:
        item = value.reshape(-1)[0]

        if isinstance(item, str):
            return item.strip() or None

    return None


def get_map_shape(wafer_map):
    """Return the original wafer-map dimensions."""
    if wafer_map is None:
        return None

    try:
        return tuple(np.asarray(wafer_map).shape)
    except (TypeError, ValueError):
        return None


def count_die(wafer_map):
    """
    Count valid wafer-map cells.

    WM-811K encoding:
    0 = outside wafer
    1 = good die
    2 = defective die

    Therefore, every non-zero cell is counted as a die.
    """
    if wafer_map is None:
        return 0

    try:
        wafer_array = np.asarray(wafer_map)
        return int(np.count_nonzero(wafer_array != 0))
    except (TypeError, ValueError):
        return 0


def inspect_dataset(df):
    print("\n" + "=" * 65)
    print("WM-811K DATASET INSPECTION")
    print("=" * 65)

    print("\n1. DATASET OVERVIEW")
    print("-" * 40)
    print(f"Total records       : {len(df):,}")
    print(f"Total columns       : {len(df.columns)}")
    print(f"Columns             : {list(df.columns)}")

    print("\n2. MISSING VALUES")
    print("-" * 40)
    print(df.isnull().sum())

    print("\n3. NORMALIZED FAILURE LABEL COUNTS")
    print("-" * 40)

    normalized_labels = df["failureType"].map(
        normalize_single_string
    )

    label_counts = normalized_labels.value_counts(dropna=False)

    for label, count in label_counts.items():
        display_label = label if label is not None else "Unlabelled"
        print(f"{str(display_label):20s}: {count:,}")

    print("\n4. BUILT-IN TRAIN/TEST MEMBERSHIP")
    print("-" * 40)

    memberships = df["trianTestLabel"].map(
        normalize_single_string
    )

    membership_counts = memberships.value_counts(dropna=False)

    for membership, count in membership_counts.items():
        display_value = (
            membership if membership is not None else "Unassigned"
        )
        print(f"{str(display_value):20s}: {count:,}")

    print("\n5. ORIGINAL WAFER-MAP SHAPES")
    print("-" * 40)

    shape_counts = Counter(
        get_map_shape(value) for value in df["waferMap"]
    )

    for shape, count in shape_counts.most_common(15):
        print(f"{str(shape):20s}: {count:,}")

    print("\n6. LOT INFORMATION")
    print("-" * 40)

    print(f"Unique non-null lots : {df['lotName'].nunique():,}")
    print(f"Missing lot names    : {df['lotName'].isnull().sum():,}")

    print("\n7. DIE COUNT SUMMARY")
    print("-" * 40)

    die_counts = [
        count_die(value) for value in df["waferMap"]
    ]

    die_series = pd.Series(die_counts)

    print(die_series.describe())


def generate_inventory(df):
    print("\n" + "=" * 65)
    print("GENERATING WAFER INVENTORY")
    print("=" * 65)

    inventory = pd.DataFrame({
        "original_row_index": np.arange(len(df)),
        "lotName": df["lotName"].values,
        "normalized_label": df["failureType"].map(
            normalize_single_string
        ).values,
        "wafer_map_shape": [
            str(get_map_shape(value))
            for value in df["waferMap"]
        ],
        "die_count": [
            count_die(value)
            for value in df["waferMap"]
        ],
        "built_in_membership": df["trianTestLabel"].map(
            normalize_single_string
        ).values,
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    inventory.to_csv(OUTPUT_PATH, index=False)

    print(f"\nInventory saved to:\n{OUTPUT_PATH}")
    print(f"Inventory records: {len(inventory):,}")
    print(f"Inventory columns: {list(inventory.columns)}")

    return inventory


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    print("Loading WM-811K dataset...")

    df = load_wafer_dataset(DATASET_PATH)

    inspect_dataset(df)

    inventory = generate_inventory(df)

    print("\n" + "=" * 65)
    print("INSPECTION COMPLETED")
    print("=" * 65)

    print(f"Original dataset records : {len(df):,}")
    print(f"Inventory records        : {len(inventory):,}")
    print("Original pickle modified : No")


if __name__ == "__main__":
    main()
