
import json
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/processed/full_dataset")

EXPECTED_ROWS = 811457

EXPECTED_LABEL_COUNTS = {
    -1: 638507,
    0: 147431,
    1: 9680,
    2: 5189,
    3: 4294,
    4: 3593,
    5: 1193,
    6: 866,
    7: 555,
    8: 149,
}

EXPECTED_MEMBERSHIP_COUNTS = {
    -1: 638507,
    0: 54355,
    1: 118595,
}

maps = np.load(DATA_DIR / "wafer_maps_64x64.npy", mmap_mode="r")
radial = np.load(DATA_DIR / "radial_profiles.npy", mmap_mode="r")
angular = np.load(DATA_DIR / "angular_profiles.npy", mmap_mode="r")
indices = np.load(DATA_DIR / "original_row_indices.npy", mmap_mode="r")
labels = np.load(DATA_DIR / "label_ids.npy", mmap_mode="r")
membership = np.load(DATA_DIR / "membership_ids.npy", mmap_mode="r")

assert maps.shape == (EXPECTED_ROWS, 64, 64)
assert radial.shape == (EXPECTED_ROWS, 64)
assert angular.shape == (EXPECTED_ROWS, 64)
assert len(indices) == EXPECTED_ROWS

print("[PASS] All output shapes are correct")

label_counts = dict(zip(*np.unique(labels, return_counts=True)))
membership_counts = dict(zip(*np.unique(membership, return_counts=True)))

print("\nLabel counts:")
print(label_counts)

print("\nMembership counts:")
print(membership_counts)

assert label_counts == EXPECTED_LABEL_COUNTS
assert membership_counts == EXPECTED_MEMBERSHIP_COUNTS

print("\n[PASS] Label counts match RevD2")
print("[PASS] Membership counts match RevD2")

assert np.array_equal(indices, np.arange(EXPECTED_ROWS))
print("[PASS] Original row indices preserved")

# Validate large arrays in batches to limit RAM usage.
for start in range(0, EXPECTED_ROWS, 4096):
    end = min(start + 4096, EXPECTED_ROWS)

    map_batch = maps[start:end]
    radial_batch = radial[start:end]
    angular_batch = angular[start:end]

    assert np.isin(map_batch, [0, 1, 2]).all()
    assert np.isfinite(radial_batch).all()
    assert np.isfinite(angular_batch).all()
    assert ((radial_batch >= 0) & (radial_batch <= 1)).all()
    assert ((angular_batch >= 0) & (angular_batch <= 1)).all()

print("[PASS] All wafer maps contain valid values")
print("[PASS] All profiles are finite and within [0, 1]")

metadata_path = DATA_DIR / "preprocessing_metadata.json"
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

assert metadata["processed_rows"] == EXPECTED_ROWS
print("[PASS] Metadata confirms all rows processed")

print("\nALL FULL-DATASET VALIDATIONS PASSED")
