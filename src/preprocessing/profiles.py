
import numpy as np


def create_valid_wafer_mask(wafer_map):
    """Identify valid wafer positions; zero means outside the wafer."""
    wafer_map = np.asarray(wafer_map)

    if wafer_map.ndim != 2:
        raise ValueError("Wafer map must be a 2D array.")

    return wafer_map != 0


def _calculate_profile(wafer_map, bin_indices, num_bins):
    valid_mask = create_valid_wafer_mask(wafer_map)
    defective_mask = (wafer_map == 2) & valid_mask

    valid_bins = bin_indices[valid_mask]
    defect_bins = bin_indices[defective_mask]

    total_counts = np.bincount(valid_bins, minlength=num_bins)
    defect_counts = np.bincount(defect_bins, minlength=num_bins)

    profile = np.zeros(num_bins, dtype=np.float32)
    occupied = total_counts > 0
    profile[occupied] = (
        defect_counts[occupied] / total_counts[occupied]
    )

    return profile


def extract_radial_profile(wafer_map, num_bins=64):
    """Calculate defective-die proportions in concentric radial bins."""
    wafer_map = np.asarray(wafer_map)

    if wafer_map.ndim != 2:
        raise ValueError("Wafer map must be a 2D array.")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")

    valid_mask = create_valid_wafer_mask(wafer_map)
    if not np.any(valid_mask):
        return np.zeros(num_bins, dtype=np.float32)

    rows, cols = np.indices(wafer_map.shape)
    center_row = (wafer_map.shape[0] - 1) / 2
    center_col = (wafer_map.shape[1] - 1) / 2

    distances = np.sqrt(
        (rows - center_row) ** 2 + (cols - center_col) ** 2
    )

    max_distance = distances[valid_mask].max()
    if max_distance == 0:
        return np.zeros(num_bins, dtype=np.float32)

    bin_indices = np.minimum(
        (distances / max_distance * num_bins).astype(int),
        num_bins - 1
    )

    return _calculate_profile(wafer_map, bin_indices, num_bins)


def extract_angular_profile(wafer_map, num_sectors=64):
    """Calculate defective-die proportions in angular sectors."""
    wafer_map = np.asarray(wafer_map)

    if wafer_map.ndim != 2:
        raise ValueError("Wafer map must be a 2D array.")
    if num_sectors <= 0:
        raise ValueError("num_sectors must be positive.")

    valid_mask = create_valid_wafer_mask(wafer_map)
    if not np.any(valid_mask):
        return np.zeros(num_sectors, dtype=np.float32)

    rows, cols = np.indices(wafer_map.shape)
    center_row = (wafer_map.shape[0] - 1) / 2
    center_col = (wafer_map.shape[1] - 1) / 2

    angles = np.arctan2(
        rows - center_row,
        cols - center_col
    )

    normalized_angles = (angles + np.pi) / (2 * np.pi)

    sector_indices = np.minimum(
        (normalized_angles * num_sectors).astype(int),
        num_sectors - 1
    )

    return _calculate_profile(wafer_map, sector_indices, num_sectors)


def extract_spatial_profiles(wafer_map):
    """Return 64-bin radial and 64-sector angular profiles."""
    return {
        "radial_profile": extract_radial_profile(wafer_map),
        "angular_profile": extract_angular_profile(wafer_map)
    }
