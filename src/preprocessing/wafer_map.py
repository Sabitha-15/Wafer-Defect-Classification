import numpy as np
from scipy.ndimage import zoom

TARGET_SIZE = (64, 64)


def resize_wafer_map(wafer_map, target_size=TARGET_SIZE):
    """Resize a categorical wafer map using nearest-neighbour interpolation."""

    wafer_map = np.asarray(wafer_map)

    if wafer_map.ndim != 2:
        raise ValueError("Wafer map must be a 2D array.")

    if wafer_map.size == 0:
        raise ValueError("Wafer map cannot be empty.")

    resized = zoom(
        wafer_map,
        (
            target_size[0] / wafer_map.shape[0],
            target_size[1] / wafer_map.shape[1],
        ),
        order=0,
        mode="nearest",
        prefilter=False,
    )

    if resized.shape != target_size:
        raise RuntimeError(f"Unexpected resized shape: {resized.shape}")

    if not set(np.unique(resized)).issubset(set(np.unique(wafer_map))):
        raise RuntimeError("Resizing introduced new categorical values.")

    return resized.astype(wafer_map.dtype, copy=False)


def create_valid_wafer_mask(wafer_map):
    """Identify valid wafer positions; zero means outside the wafer."""

    wafer_map = np.asarray(wafer_map)

    if wafer_map.ndim != 2:
        raise ValueError("Wafer map must be a 2D array.")

    return wafer_map != 0