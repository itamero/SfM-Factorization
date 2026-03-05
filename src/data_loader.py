import numpy as np
from scipy.spatial.transform import Rotation


def generate_3d_points(num_points, shape="full_3d", rng=None):
    """Generate synthetic 3D points.

    Parameters
    ----------
    num_points : int
    shape : str
        One of "full_3d", "near_planar", "sphere".
    rng : np.random.Generator or None

    Returns shape (3, P).
    """
    if rng is None:
        rng = np.random.default_rng()
    if shape == "full_3d":
        points = rng.uniform(-1, 1, size=(3, num_points))
    elif shape == "near_planar":
        points = np.empty((3, num_points))
        points[0] = rng.uniform(-1, 1, size=num_points)
        points[1] = rng.uniform(-1, 1, size=num_points)
        points[2] = rng.normal(0, 0.001, size=num_points)
    elif shape == "sphere":
        # Points uniformly sampled on the unit sphere surface
        points = rng.normal(size=(3, num_points))
        points /= np.linalg.norm(points, axis=0, keepdims=True)
    else:
        raise ValueError(f"Unknown shape: {shape}")
    return points

def generate_rotations(num_frames, rng=None):
    """Generate rotation matrices for each frame.

    Returns shape (F, 3, 3).
    """
    if rng is None:
        rng = np.random.default_rng()

    rotations = np.empty((num_frames, 3, 3))

    for f in range(num_frames):
        # Random axis (unit vector)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)

        angle = rng.uniform(np.radians(10), np.radians(60))

        rotvec = axis * angle
        rotations[f] = Rotation.from_rotvec(rotvec).as_matrix()

    return rotations


def project_orthographic(rotations, points_3d):
    """Orthographic projection: measurement matrix of shape (2F, P).

    Rows are ordered [u1, v1, u2, v2, ..., uF, vF].
    """
    num_frames = rotations.shape[0]
    num_points = points_3d.shape[1]
    measurement = np.empty((2 * num_frames, num_points))

    for f in range(num_frames):
        projected = rotations[f] @ points_3d  # (3, P)
        measurement[2 * f] = projected[0]      # u_f
        measurement[2 * f + 1] = projected[1]  # v_f

    return measurement


def add_noise(measurement_matrix, sigma, rng=None):
    """Add Gaussian noise N(0, sigma) to all entries. Returns a copy."""
    if rng is None:
        rng = np.random.default_rng()

    return measurement_matrix + rng.normal(0, sigma, size=measurement_matrix.shape)


def apply_missing_mask(measurement_matrix, missing_rate, rng=None):
    """Randomly mask out point-frame pairs.

    Returns (corrupted_matrix, mask) where mask is boolean (True = observed).
    Missing entries are set to NaN. Masking is per point-frame pair: both
    x and y rows for a given (frame, point) go missing together.
    """
    if rng is None:
        rng = np.random.default_rng()

    two_f, num_points = measurement_matrix.shape
    num_frames = two_f // 2

    # Decide visibility per (frame, point) pair
    visible = rng.random(size=(num_frames, num_points)) >= missing_rate

    # Expand to (2F, P): each frame-point pair controls two rows
    mask = np.repeat(visible, 2, axis=0)  # (2F, P)

    corrupted = measurement_matrix.copy()
    corrupted[~mask] = np.nan

    return corrupted, mask


def add_outliers(measurement_matrix, outlier_rate, outlier_magnitude=5.0, rng=None):
    """Replace a fraction of point-frame entries with large outliers.

    Returns (corrupted_matrix, outlier_mask) where outlier_mask is boolean
    (True = outlier). Outliers are applied per point-frame pair.
    """
    if rng is None:
        rng = np.random.default_rng()

    two_f, num_points = measurement_matrix.shape
    num_frames = two_f // 2

    # Decide which (frame, point) pairs are outliers
    is_outlier = rng.random(size=(num_frames, num_points)) < outlier_rate

    # Expand to (2F, P)
    outlier_mask = np.repeat(is_outlier, 2, axis=0)

    corrupted = measurement_matrix.copy()
    # Add large random offsets to outlier entries
    num_outlier_entries = outlier_mask.sum()
    corrupted[outlier_mask] += rng.uniform(
        -outlier_magnitude, outlier_magnitude, size=num_outlier_entries
    )

    return corrupted, outlier_mask


def generate_sfm_data(
    num_points=50,
    num_frames=10,
    sigma=0.0,
    missing_rate=0.0,
    outlier_rate=0.0,
    outlier_magnitude=5.0,
    point_shape="full_3d",
    degeneracy_motion="diverse",
    seed=None,
):
    """Generate a complete synthetic SfM dataset.

    Returns a dict with keys:
        points_3d            — (3, P)
        rotations            — (F, 3, 3)
        measurement_matrix_clean — (2F, P)
        measurement_matrix   — (2F, P), with noise/missing/outliers applied
        visibility_mask      — (2F, P) boolean, True = observed
        outlier_mask         — (2F, P) boolean, True = outlier
    """
    rng = np.random.default_rng(seed)

    points_3d = generate_3d_points(num_points, shape=point_shape, rng=rng)
    rotations = generate_rotations(num_frames, rng=rng)
    clean = project_orthographic(rotations, points_3d)

    # Apply corruptions in order: noise → outliers → missing
    corrupted = clean.copy()

    if sigma > 0:
        corrupted = add_noise(corrupted, sigma, rng=rng)

    outlier_mask = np.zeros_like(corrupted, dtype=bool)
    if outlier_rate > 0:
        corrupted, outlier_mask = add_outliers(
            corrupted, outlier_rate, outlier_magnitude, rng=rng
        )

    visibility_mask = np.ones_like(corrupted, dtype=bool)
    if missing_rate > 0:
        corrupted, visibility_mask = apply_missing_mask(
            corrupted, missing_rate, rng=rng
        )

    return {
        "points_3d": points_3d,
        "rotations": rotations,
        "measurement_matrix_clean": clean,
        "measurement_matrix": corrupted,
        "visibility_mask": visibility_mask,
        "outlier_mask": outlier_mask,
    }
