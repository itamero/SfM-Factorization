import numpy as np
import matplotlib.pyplot as plt


def procrustes_align(S_estimated, S_ground_truth):
    """Align estimated structure to ground truth via Procrustes (rotation + scale).

    Both inputs are (3, P). Returns the aligned estimate and the RMSE.
    Solves: min ||s * R @ S_est - S_gt||_F over rotation R and scale s.
    """
    # Center both
    S_est = S_estimated - S_estimated.mean(axis=1, keepdims=True)
    S_gt = S_ground_truth - S_ground_truth.mean(axis=1, keepdims=True)

    # Optimal rotation via SVD of cross-covariance
    H = S_est @ S_gt.T  # (3, 3)
    U, _, Vt = np.linalg.svd(H)
    # Ensure proper rotation (det = +1)
    D = np.eye(3)
    D[2, 2] = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ D @ U.T

    S_est_rot = R @ S_est

    # Optimal scale
    scale = np.trace(S_gt @ S_est_rot.T) / np.trace(S_est_rot @ S_est_rot.T)
    S_aligned = scale * S_est_rot

    rmse = np.sqrt(np.mean((S_aligned - S_gt) ** 2))
    return S_aligned, rmse


def reprojection_rmse(measurements, motion, structure, translations):
    """Compute RMSE between original measurements and M @ S + t (reprojected).

    Parameters
    ----------
    measurements : ndarray, shape (2F, P)
    motion : ndarray, shape (2F, 3)
    structure : ndarray, shape (3, P)
    translations : ndarray, shape (F, 2)

    Returns
    -------
    rmse : float
    """
    two_f, num_points = measurements.shape
    num_frames = two_f // 2

    reprojected = motion @ structure
    # Add back translations
    for f in range(num_frames):
        reprojected[2 * f] += translations[f, 0]
        reprojected[2 * f + 1] += translations[f, 1]

    mask = ~np.isnan(measurements)
    diff = (reprojected - measurements)[mask]
    return np.sqrt(np.mean(diff ** 2))


def plot_structure_comparison(S_gt, S_est, title_gt="Ground Truth", title_est="Reconstructed"):
    """Side-by-side 3D scatter plots comparing ground truth and estimated structure."""
    fig = plt.figure(figsize=(12, 5))

    num_points = S_gt.shape[1]
    colors = plt.cm.hsv(np.linspace(0, 1, num_points, endpoint=False))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(S_gt[0], S_gt[1], S_gt[2], c=colors, s=20)
    ax1.set_title(title_gt)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(S_est[0], S_est[1], S_est[2], c=colors, s=20)
    ax2.set_title(title_est)
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")

    fig.tight_layout()
    plt.show()


def plot_sfm_data(data, frames=None, title=None):
    """Plot 3D points and their 2D projections with consistent per-point colors.

    Parameters
    ----------
    data : dict
        Output of generate_sfm_data(). Expected keys: points_3d, rotations,
        and measurement_matrix or measurement_matrix_clean.
    frames : list[int] or None
        Frame indices to display. Default: up to 4 evenly spaced frames.
    title : str or None
        Figure suptitle.
    """
    points_3d = data["points_3d"]  # (3, P)
    num_points = points_3d.shape[1]

    # Choose measurement matrix: prefer corrupted, fall back to clean
    mm = data.get("measurement_matrix")
    if mm is None or np.all(np.isnan(mm)):
        mm = data["measurement_matrix_clean"]

    num_frames = mm.shape[0] // 2

    # Default frames: up to 4 evenly spaced
    if frames is None:
        if num_frames <= 4:
            frames = list(range(num_frames))
        else:
            frames = np.linspace(0, num_frames - 1, 4, dtype=int).tolist()

    # Consistent colors per point
    colors = plt.cm.hsv(np.linspace(0, 1, num_points, endpoint=False))

    num_subplots = 1 + len(frames)
    fig = plt.figure(figsize=(5 * num_subplots, 5))

    # 3D scatter
    ax3d = fig.add_subplot(1, num_subplots, 1, projection="3d")
    ax3d.scatter(points_3d[0], points_3d[1], points_3d[2], c=colors, s=20)
    ax3d.set_title("3D Points")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    # 2D projections
    for i, f in enumerate(frames):
        ax = fig.add_subplot(1, num_subplots, 2 + i)
        u = mm[2 * f]
        v = mm[2 * f + 1]

        # Handle NaN (missing data): only plot observed points
        observed = ~(np.isnan(u) | np.isnan(v))
        ax.scatter(u[observed], v[observed], c=colors[observed], s=20)
        ax.set_title(f"Frame {f}")
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_aspect("equal", adjustable="datalim")

    if title:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()
    plt.show()
