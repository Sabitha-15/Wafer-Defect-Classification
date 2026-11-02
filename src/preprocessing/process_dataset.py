
import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.data.load_dataset import load_wafer_dataset
from src.preprocessing.wafer_map import resize_wafer_map
from src.preprocessing.profiles import extract_spatial_profiles


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "LSWMD.pkl"

BATCH_SIZE = 128

# Fixed nine-class order for the entire project.
CLASS_NAMES = [
    "none",
    "Edge-Ring",
    "Edge-Loc",
    "Center",
    "Loc",
    "Scratch",
    "Random",
    "Donut",
    "Near-full",
]

LABEL_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# Membership codes:
# -1 = unlabelled, 0 = Training, 1 = Test
MEMBERSHIP_TO_ID = {
    "Training": 0,
    "Test": 1,
}


def normalize_single_string(value):
    """Apply the RevD2 single-string normalization rule."""
    if isinstance(value, str):
        return value

    if isinstance(value, np.ndarray) and value.size == 1:
        item = value.reshape(-1)[0]
        if isinstance(item, str):
            return item

    return None


def process_dataset(output_dir, limit=None, batch_size=BATCH_SIZE):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    print("Loading original WM-811K dataset...")
    df = load_wafer_dataset(file_path=DATASET_PATH)

    total_rows = len(df)
    rows_to_process = min(total_rows, limit) if limit else total_rows

    print(f"Original dataset rows: {total_rows:,}")
    print(f"Rows to process: {rows_to_process:,}")
    print(f"Batch size: {batch_size}")

    # Disk-backed output arrays.
    maps = np.lib.format.open_memmap(
        output_dir / "wafer_maps_64x64.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(rows_to_process, 64, 64),
    )

    radial = np.lib.format.open_memmap(
        output_dir / "radial_profiles.npy",
        mode="w+",
        dtype=np.float32,
        shape=(rows_to_process, 64),
    )

    angular = np.lib.format.open_memmap(
        output_dir / "angular_profiles.npy",
        mode="w+",
        dtype=np.float32,
        shape=(rows_to_process, 64),
    )

    row_indices = np.lib.format.open_memmap(
        output_dir / "original_row_indices.npy",
        mode="w+",
        dtype=np.int64,
        shape=(rows_to_process,),
    )

    labels = np.lib.format.open_memmap(
        output_dir / "label_ids.npy",
        mode="w+",
        dtype=np.int8,
        shape=(rows_to_process,),
    )

    membership = np.lib.format.open_memmap(
        output_dir / "membership_ids.npy",
        mode="w+",
        dtype=np.int8,
        shape=(rows_to_process,),
    )

    start_time = time.perf_counter()

    for batch_start in range(0, rows_to_process, batch_size):
        batch_end = min(batch_start + batch_size, rows_to_process)

        for row_index in range(batch_start, batch_end):
            original_map = np.asarray(
                df["waferMap"].iloc[row_index]
            )

            if original_map.ndim != 2:
                raise ValueError(
                    f"Invalid wafer map at row {row_index}"
                )

            if not np.isin(original_map, [0, 1, 2]).all():
                raise ValueError(
                    f"Unexpected wafer-map values at row {row_index}"
                )

            # Profiles are computed from the ORIGINAL map.
            profiles = extract_spatial_profiles(original_map)

            # Resize only for the image-model representation.
            resized_map = resize_wafer_map(original_map)

            if not np.isin(resized_map, [0, 1, 2]).all():
                raise ValueError(
                    f"Unexpected resized values at row {row_index}"
                )

            maps[row_index] = resized_map.astype(np.uint8)
            radial[row_index] = profiles["radial_profile"]
            angular[row_index] = profiles["angular_profile"]

            row_indices[row_index] = row_index

            label = normalize_single_string(
                df["failureType"].iloc[row_index]
            )

            labels[row_index] = (
                LABEL_TO_ID[label] if label in LABEL_TO_ID else -1
            )

            built_in = normalize_single_string(
                df["trianTestLabel"].iloc[row_index]
            )

            membership[row_index] = MEMBERSHIP_TO_ID.get(
                built_in, -1
            )

        # Flush each batch to disk.
        maps.flush()
        radial.flush()
        angular.flush()
        row_indices.flush()
        labels.flush()
        membership.flush()

        elapsed = time.perf_counter() - start_time
        processed = batch_end
        rate = processed / elapsed if elapsed else 0

        print(
            f"Processed {processed:,}/{rows_to_process:,} "
            f"({processed / rows_to_process * 100:.2f}%) "
            f"| {rate:.1f} wafers/sec"
        )

    elapsed = time.perf_counter() - start_time

    # Final flush.
    maps.flush()
    radial.flush()
    angular.flush()
    row_indices.flush()
    labels.flush()
    membership.flush()

    metadata = {
        "source_dataset": "WM-811K",
        "source_rows": total_rows,
        "processed_rows": rows_to_process,
        "batch_size": batch_size,
        "map_shape": [64, 64],
        "radial_bins": 64,
        "angular_sectors": 64,
        "class_names": CLASS_NAMES,
        "label_encoding": {
            name: index for index, name in enumerate(CLASS_NAMES)
        },
        "unlabelled_label_id": -1,
        "membership_encoding": {
            "Training": 0,
            "Test": 1,
            "unknown_or_unlabelled": -1,
        },
        "profile_definition": (
            "Per-bin defective-die proportion among valid wafer positions; "
            "computed from original maps."
        ),
        "elapsed_seconds": round(elapsed, 2),
    }

    with open(output_dir / "preprocessing_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nPREPROCESSING COMPLETED")
    print(f"Rows processed: {rows_to_process:,}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows for a trial run.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            PROJECT_ROOT / "data" / "processed" / "preprocessing_trial"
        ),
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")

    process_dataset(
        output_dir=args.output_dir,
        limit=args.limit,
        batch_size=args.batch_size,
    )
