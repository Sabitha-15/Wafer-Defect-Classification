
from pathlib import Path
import json
import numpy as np

OUTPUT = Path("data/processed/preprocessing_trial")

maps = np.load(OUTPUT / "wafer_maps_64x64.npy", mmap_mode="r")
radial = np.load(OUTPUT / "radial_profiles.npy", mmap_mode="r")
angular = np.load(OUTPUT / "angular_profiles.npy", mmap_mode="r")
indices = np.load(OUTPUT / "original_row_indices.npy", mmap_mode="r")
labels = np.load(OUTPUT / "label_ids.npy", mmap_mode="r")
membership = np.load(OUTPUT / "membership_ids.npy", mmap_mode="r")

checks = {
    "Map values are only 0, 1, 2":
        np.isin(maps, [0, 1, 2]).all(),

    "Radial profiles are finite":
        np.isfinite(radial).all(),

    "Angular profiles are finite":
        np.isfinite(angular).all(),

    "Radial profiles are within [0, 1]":
        ((radial >= 0) & (radial <= 1)).all(),

    "Angular profiles are within [0, 1]":
        ((angular >= 0) & (angular <= 1)).all(),

    "Original row indices preserved":
        np.array_equal(indices, np.arange(256)),

    "Label IDs are valid":
        np.isin(labels, [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]).all(),

    "Membership IDs are valid":
        np.isin(membership, [-1, 0, 1]).all(),
}

for name, passed in checks.items():
    print(f"[{'PASS' if passed else 'FAIL'}] {name}")

metadata = json.loads(
    (OUTPUT / "preprocessing_metadata.json").read_text()
)

print("\nMetadata:")
print(json.dumps(metadata, indent=2))

if not all(checks.values()):
    raise SystemExit("TRIAL VALIDATION FAILED")

print("\nALL TRIAL VALIDATIONS PASSED")
