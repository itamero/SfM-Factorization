import numpy as np


def center_data(measurements):
    """Center the measurement matrix by removing per-frame translation.

    For each frame f, subtracts the centroid of that frame's 2D points
    from both the u and v rows.

    Parameters
    ----------
    measurements : ndarray, shape (2F, P)
        Raw measurement matrix with rows [u1, v1, u2, v2, ..., uF, vF].

    Returns
    -------
    centered : ndarray, shape (2F, P)
        Centered measurement matrix.
    translations : ndarray, shape (F, 2)
        Per-frame translation vectors (centroid u, centroid v) that were subtracted.
    """
    two_f, num_points = measurements.shape
    num_frames = two_f // 2

    centered = measurements.copy()
    translations = np.empty((num_frames, 2))

    for f in range(num_frames):
        u_row = centered[2 * f]
        v_row = centered[2 * f + 1]

        mean_u = np.nanmean(u_row)
        mean_v = np.nanmean(v_row)

        translations[f] = [mean_u, mean_v]

        centered[2 * f] -= mean_u
        centered[2 * f + 1] -= mean_v

    return centered, translations


def factorize_tomasi_kanade(measurements, missing_strat="zero-fill",
                            max_iter=100, tol=1e-6, trim_fraction=0.0,
                            outlier_mask=None):
    """Tomasi-Kanade factorization for rigid SfM under orthographic projection.

    Centers the measurement matrix, computes the rank-3 SVD approximation,
    and recovers the motion and structure matrices.

    Parameters
    ----------
    measurements : ndarray, shape (2F, P)
        Raw measurement matrix. May contain NaN for missing observations.
    missing_strat : str, default "zero-fill"
        Strategy for handling missing (NaN) entries after centering:
        - "zero-fill": replace NaNs with 0.0 before SVD.
        - "iterative-svd": iteratively compute rank-3 SVD, updating only
          missing entries until convergence.
    max_iter : int, default 100
        Maximum iterations (only used with "iterative-svd").
    tol : float, default 1e-6
        Convergence threshold on relative change in filled values
        (only used with "iterative-svd").
    trim_fraction : float, default 0.0
        Fraction of observed entries (by largest residual) to treat as
        suspected outliers each iteration. Only used with "iterative-svd".
        Set to e.g. 0.05 to trim the top 5% residuals.
    outlier_mask : ndarray of bool, shape (2F, P), optional
        Ground-truth outlier mask (True = outlier). When provided, the
        returned trim_stats dict reports how many trimmed entries were
        true outliers vs. inliers.

    Returns
    -------
    motion : ndarray, shape (2F, 3)
        Motion matrix. Each pair of rows [2f, 2f+1] gives the first two rows
        of frame f's rotation matrix.
    structure : ndarray, shape (3, P)
        3D point coordinates.
    translations : ndarray, shape (F, 2)
        Per-frame translations that were removed during centering.
    history : list of float
        Reprojection RMSE (observed entries) at each iteration.
        Only returned when missing_strat="iterative-svd".
    trim_stats : dict
        Only returned when missing_strat="iterative-svd" and outlier_mask
        is provided. Keys: "outliers_ignored" (true outliers correctly
        trimmed) and "inliers_ignored" (inliers incorrectly trimmed).
    """
    centered, translations = center_data(measurements)
    observed = ~np.isnan(centered)
    np.nan_to_num(centered, copy=False, nan=0.0)
    # Force iterative loop if trimming is requested
    is_iterative = (missing_strat == "iterative-svd") or (trim_fraction > 0.0)

    if is_iterative:
        working = centered.copy()
        history = []
        inliers = observed.copy()

        for iteration in range(max_iter):
            U, sigma, Vt = np.linalg.svd(working, full_matrices=False)
            recon = U[:, :3] @ np.diag(sigma[:3]) @ Vt[:3, :]

            # Dynamic trimming based on current residuals against pristine data
            if trim_fraction > 0.0:
                residuals = np.abs(centered - recon)
                obs_residuals = residuals[observed]
                threshold = np.percentile(obs_residuals, 100.0 * (1.0 - trim_fraction))
                inliers = observed & (residuals <= threshold)
            else:
                inliers = observed

            # Build new working matrix: use original data for inliers, recon for the rest
            new_working = recon.copy()
            new_working[inliers] = centered[inliers]

            mask_changed = ~inliers
            if mask_changed.any():
                change = np.linalg.norm(new_working[mask_changed] - working[mask_changed]) / \
                         (np.linalg.norm(new_working[mask_changed]) + 1e-16)
            else:
                change = 0.0

            diff = (recon - centered)[inliers]
            rmse = np.sqrt(np.mean(diff ** 2))
            history.append(rmse)

            working = new_working

            if change < tol:
                break

        centered = working
    elif missing_strat != "zero-fill":
        raise ValueError(f"Unknown missing_strat: {missing_strat!r}")

    # SVD of the centered measurement matrix
    U, sigma, Vt = np.linalg.svd(centered, full_matrices=False)

    # Rank-3 truncation
    U3 = U[:, :3]
    sigma3 = np.diag(np.sqrt(sigma[:3]))
    V3t = Vt[:3, :]

    motion = U3 @ sigma3        # (2F, 3)
    structure = sigma3 @ V3t    # (3, P)

    if is_iterative:
        if outlier_mask is not None:
            trimmed = observed & ~inliers
            trim_stats = {
                "outliers_ignored": int((trimmed & outlier_mask).sum()),
                "inliers_ignored": int((trimmed & ~outlier_mask).sum()),
            }
            return motion, structure, translations, history, trim_stats
        return motion, structure, translations, history
    return motion, structure, translations
