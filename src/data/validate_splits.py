
"""
RevD2 E3 split validation.
Team 18 | Track E | Card E3

Validates the instructor-issued validation lots and
pseudo-label row indices against the original WM-811K pickle.
Does not train models or inspect test performance.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_dataset import load_wafer_dataset


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "LSWMD.pkl"
SPLIT_PATH = ROOT / "data" / "splits" / "TrackE_split_lot_lists.json"

EXPECTED = {
    "total_rows": 811457,
    "training_wafers": 54355,
    "test_wafers": 118595,
    "training_lots": 5809,
    "test_lots": 4953,
    "validation_lots": 1161,
    "eligible_unlabelled": 620038,
    "eligible_unlabelled_lots": 39496,
}


def normalize(value):
    """Apply the exact RevD2 single-string normalization rule."""
    if isinstance(value, str):
        return value.strip() or None

    if isinstance(value, np.ndarray) and value.size == 1:
        item = value.reshape(-1)[0]
        if isinstance(item, str):
            return item.strip() or None

    return None


def sha256_row_index(index):
    """RevD2 hash ordering for eligible unlabelled wafer IDs."""
    return hashlib.sha256(f"22|{index}".encode("utf-8")).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)
    print(f"[PASS] {message}")


def main():
    print("=" * 65)
    print("REV D2 - E3 SPLIT VALIDATION")
    print("=" * 65)

    require(DATA_PATH.exists(), "Original dataset file exists")
    require(SPLIT_PATH.exists(), "Issued split JSON exists")

    print("\nLoading dataset...")
    df = load_wafer_dataset(DATA_PATH)

    print("Loading issued split lists...")
    with open(SPLIT_PATH, "r", encoding="utf-8") as file:
        split = json.load(file)

    required_keys = {
        "E1E3_val_lots",
        "E2_test_lots",
        "E2_val_lots",
        "E2_train_lots",
        "E3_cap2000_idx",
        "E3_cap10000_idx",
        "meta",
    }

    require(set(split.keys()) == required_keys,
            "JSON contains exactly the seven expected top-level keys")

    require(len(df) == EXPECTED["total_rows"],
            "Dataset row count matches RevD2")

    # Normalize labels and built-in membership.
    labels = df["failureType"].map(normalize)
    membership = df["trianTestLabel"].map(normalize)
    lots = df["lotName"]

    training_mask = membership.eq("Training")
    test_mask = membership.eq("Test")
    unlabelled_mask = labels.isna()

    training_lots = set(lots[training_mask].unique())
    test_lots = set(lots[test_mask].unique())

    require(int(training_mask.sum()) == EXPECTED["training_wafers"],
            "Built-in training wafer count matches RevD2")

    require(int(test_mask.sum()) == EXPECTED["test_wafers"],
            "Built-in test wafer count matches RevD2")

    require(len(training_lots) == EXPECTED["training_lots"],
            "Built-in training lot count matches RevD2")

    require(len(test_lots) == EXPECTED["test_lots"],
            "Built-in test lot count matches RevD2")

    require(training_lots.isdisjoint(test_lots),
            "Built-in training and test lots are disjoint")

    # Validate the issued E1/E3 validation lots.
    val_lots_list = split["E1E3_val_lots"]
    val_lots = set(val_lots_list)

    require(len(val_lots_list) == EXPECTED["validation_lots"],
            "Issued E1/E3 validation list has 1,161 entries")

    require(len(val_lots) == len(val_lots_list),
            "Validation lot list has no duplicates")

    require(val_lots.issubset(training_lots),
            "Every validation lot belongs to built-in training")

    require(val_lots.isdisjoint(test_lots),
            "Validation lots do not overlap built-in test lots")

    validation_mask = training_mask & lots.isin(val_lots)
    fit_mask = training_mask & ~lots.isin(val_lots)

    validation_labels = labels[validation_mask]
    fit_labels = labels[fit_mask]

    print("\nE1/E3 built-in-protocol split:")
    print("Training lots for fitting:", len(training_lots - val_lots))
    print("Validation lots:", len(val_lots))
    print("Fitting wafers:", int(fit_mask.sum()))
    print("Validation wafers:", int(validation_mask.sum()))
    print("Built-in test wafers:", int(test_mask.sum()))

    print("\nValidation class support:")
    print(validation_labels.value_counts().sort_index())

    require(validation_labels.notna().all(),
            "All validation wafers have normalized labels")

    require(validation_labels.nunique() == 9,
            "All nine classes have validation support")

    require(fit_labels.notna().all(),
            "All fitting wafers have normalized labels")

    # Eligible unlabelled pool:
    # Exclude unlabelled wafers from validation/test lots.
    excluded_lots = val_lots | test_lots

    eligible_mask = unlabelled_mask & ~lots.isin(excluded_lots)
    eligible_indices = np.flatnonzero(eligible_mask.to_numpy())

    eligible_lots = set(lots.iloc[eligible_indices].unique())

    print("\nEligible unlabelled pool:")
    print("Wafers:", len(eligible_indices))
    print("Lots:", len(eligible_lots))

    require(len(eligible_indices) == EXPECTED["eligible_unlabelled"],
            "Eligible unlabelled wafer count matches RevD2")

    require(len(eligible_lots) == EXPECTED["eligible_unlabelled_lots"],
            "Eligible unlabelled lot count matches RevD2")

    # Validate the two issued nested caps.
    cap2k = split["E3_cap2000_idx"]
    cap10k = split["E3_cap10000_idx"]

    require(len(cap2k) == 2000, "Issued 2,000-wafer cap has 2,000 IDs")
    require(len(cap10k) == 10000, "Issued 10,000-wafer cap has 10,000 IDs")

    require(len(set(cap2k)) == 2000, "2,000-wafer cap has unique IDs")
    require(len(set(cap10k)) == 10000, "10,000-wafer cap has unique IDs")

    require(set(cap2k).issubset(set(cap10k)),
            "2,000-wafer cap is nested inside 10,000-wafer cap")

    eligible_set = set(map(int, eligible_indices))

    require(set(cap10k).issubset(eligible_set),
            "All 10,000 issued IDs belong to the eligible unlabelled pool")

    # Independently reproduce the required hash ordering.
    print("\nReproducing eligible-pool hash ordering...")
    expected_order = sorted(
        map(int, eligible_indices),
        key=sha256_row_index
    )

    require(cap2k == expected_order[:2000],
            "Issued 2,000 IDs match the first 2,000 hash-ordered IDs")

    require(cap10k == expected_order[:10000],
            "Issued 10,000 IDs match the first 10,000 hash-ordered IDs")

    print("\nJSON metadata:")
    print(json.dumps(split["meta"], indent=2))

    print("\n" + "=" * 65)
    print("ALL E3 SPLIT VALIDATIONS PASSED")
    print("=" * 65)


if __name__ == "__main__":
    main()
