
import numpy as np

from src.preprocessing.profiles import (
    create_valid_wafer_mask,
    extract_radial_profile,
    extract_angular_profile,
    extract_spatial_profiles,
)


def test_valid_mask():
    wafer = np.array([
        [0, 0, 0],
        [0, 1, 2],
        [0, 1, 0]
    ])

    mask = create_valid_wafer_mask(wafer)

    assert mask.shape == wafer.shape
    assert mask.sum() == 3


def test_radial_profile_shape():
    wafer = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, 1, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0]
    ])

    profile = extract_radial_profile(wafer)

    assert profile.shape == (64,)
    assert np.all(profile >= 0)
    assert np.all(profile <= 1)


def test_angular_profile_shape():
    wafer = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, 1, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0]
    ])

    profile = extract_angular_profile(wafer)

    assert profile.shape == (64,)
    assert np.all(profile >= 0)
    assert np.all(profile <= 1)


def test_empty_wafer():
    wafer = np.zeros((5, 5), dtype=np.uint8)

    radial = extract_radial_profile(wafer)
    angular = extract_angular_profile(wafer)

    assert np.all(radial == 0)
    assert np.all(angular == 0)


def test_combined_profiles():
    wafer = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 2, 1, 0],
        [1, 2, 1, 2, 1],
        [0, 1, 2, 1, 0],
        [0, 0, 1, 0, 0]
    ])

    profiles = extract_spatial_profiles(wafer)

    assert profiles["radial_profile"].shape == (64,)
    assert profiles["angular_profile"].shape == (64,)
