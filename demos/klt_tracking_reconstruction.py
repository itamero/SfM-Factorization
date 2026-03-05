"""
Utility functions for KLT point tracking and image/video loading.

Provides:
- track_points_klt: Track 2D points across frames with Lucas-Kanade optical flow
- initialize_feature_points: Detect good-to-track points in a ROI
- select_roi_and_points: Interactive ROI selection + feature detection
- load_video_or_images: Load frames from a video file or image directory
- load_images: Load all images from a directory
"""

import os
import numpy as np
import cv2


def track_points_klt(video_frames: np.ndarray, query_points: np.ndarray):
    """
    Track points using KLT (Lucas-Kanade) optical flow.

    Parameters
    ----------
    video_frames : np.ndarray, shape (T, H, W, 3), uint8 RGB
    query_points : np.ndarray, shape (N, 2) — (x, y) on frame 0

    Returns
    -------
    tracks : np.ndarray, shape (T, N, 2)
    visibility : np.ndarray, shape (T, N)
    """
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    T, H, W, _ = video_frames.shape
    N = query_points.shape[0]

    tracks = np.zeros((T, N, 2), dtype=np.float64)
    visibility = np.ones((T, N), dtype=bool)

    pts = query_points.astype(np.float32).reshape(-1, 1, 2)
    tracks[0] = query_points

    prev_gray = cv2.cvtColor(video_frames[0], cv2.COLOR_RGB2GRAY)

    for t in range(1, T):
        curr_gray = cv2.cvtColor(video_frames[t], cv2.COLOR_RGB2GRAY)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts, None, **lk_params
        )
        status = status.flatten().astype(bool)

        # Back-track for validation
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, next_pts, None, **lk_params
        )
        back_status = back_status.flatten().astype(bool)
        fb_error = np.linalg.norm(
            pts.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1
        )
        good = status & back_status & (fb_error < 2.0)

        tracks[t] = next_pts.reshape(-1, 2)
        visibility[t] = good

        # Update: keep tracking from latest position (even if lost, for continuity)
        pts = next_pts.copy()
        prev_gray = curr_gray

    return tracks, visibility


def initialize_feature_points(
    frame: np.ndarray,
    roi: tuple,
    max_corners: int = 500,
    quality_level: float = 0.01,
    min_distance: float = 10,
) -> np.ndarray:
    """
    Detect good-to-track feature points inside a region of interest.

    Parameters
    ----------
    frame : np.ndarray (H, W, 3)
    roi : (x, y, w, h) bounding box of the object
    max_corners : maximum number of corners to return
    quality_level : minimum accepted quality of corners
    min_distance : minimum Euclidean distance between returned corners

    Returns
    -------
    points : np.ndarray, shape (N, 2) — (x, y)
    """
    x0, y0, w, h = roi
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Create a mask so detection is restricted to the ROI
    mask = np.zeros(gray.shape, dtype=np.uint8)
    mask[y0 : y0 + h, x0 : x0 + w] = 255

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        mask=mask,
    )
    if corners is None or len(corners) == 0:
        raise RuntimeError("No feature points detected in the ROI.")

    points = corners.reshape(-1, 2).astype(np.float64)
    return points


def select_roi_and_points(
    frame: np.ndarray,
    max_corners: int = 500,
    quality_level: float = 0.01,
    min_distance: float = 10,
):
    """Let the user draw a bounding box, then detect features inside it."""
    print("Draw a bounding box around the object, then press ENTER or SPACE.")
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    roi = cv2.selectROI("Select Object", frame_bgr, fromCenter=False)
    cv2.destroyWindow("Select Object")
    if roi[2] == 0 or roi[3] == 0:
        raise RuntimeError("No ROI selected.")
    points = initialize_feature_points(
        frame, roi,
        max_corners=max_corners,
        quality_level=quality_level,
        min_distance=min_distance,
    )
    print(f"Detected {len(points)} feature points.")
    return points


def load_video_or_images(path: str) -> np.ndarray:
    """Load frames from either a directory of images or a single video file."""
    if os.path.isfile(path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if len(frames) == 0:
            raise RuntimeError(f"Could not read any frames from video: {path}")
        return np.array(frames)

    elif os.path.isdir(path):
        return load_images(path)

    else:
        raise RuntimeError(f"Path {path} is neither a file nor a directory.")


def load_images(directory: str) -> np.ndarray:
    """Load all images from a directory into an (T, H, W, 3) uint8 RGB array."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    fnames = sorted(
        f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in exts
    )
    if len(fnames) == 0:
        raise RuntimeError(f"No images found in {directory}")
    frames = []
    for fname in fnames:
        img = cv2.imread(os.path.join(directory, fname))
        if img is None:
            print(f"  Warning: could not read {fname}, skipping.")
            continue
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if len(frames) == 0:
        raise RuntimeError(f"Could not read any images from {directory}")
    # Verify all frames have the same shape
    h0, w0 = frames[0].shape[:2]
    for i, f in enumerate(frames):
        if f.shape[:2] != (h0, w0):
            raise RuntimeError(
                f"Image {fnames[i]} has shape {f.shape[:2]} but expected {(h0, w0)}. "
                "All images must have the same resolution."
            )
    return np.array(frames)


def run_klt_pipeline(path, max_corners=1000, quality_level=0.01,
                     min_distance=10, min_visible=0.7):
    """Run the full KLT tracking + Tomasi-Kanade reconstruction pipeline."""
    import sys
    import matplotlib.pyplot as plt

    script_dir = os.path.dirname(__file__)
    sys.path.insert(0, os.path.join(script_dir, "..", "src"))
    from factorization import factorize_tomasi_kanade

    print(f"Loading frames from: {path}")
    video = load_video_or_images(path)
    T, H, W, _ = video.shape
    print(f"  {T} frames, {W}x{H}")

    query_points = select_roi_and_points(
        video[0],
        max_corners=max_corners,
        quality_level=quality_level,
        min_distance=min_distance,
    )

    print(f"Tracking {len(query_points)} points with KLT...")
    tracks, visibility = track_points_klt(video, query_points)

    vis_frac = visibility.mean(axis=0)
    keep = vis_frac >= min_visible
    print(
        f"  Keeping {keep.sum()} / {len(keep)} points "
        f"(visible >= {min_visible * 100:.0f}% of frames)"
    )
    if keep.sum() < 4:
        print("ERROR: Too few surviving points. Try a larger ROI or lower min_visible.")
        sys.exit(1)

    tracks = tracks[:, keep, :]
    visibility = visibility[:, keep]
    n_points = tracks.shape[1]

    # Build measurement matrix
    measurements = np.zeros((2 * T, n_points))
    for f in range(T):
        measurements[2 * f] = tracks[f, :, 0]
        measurements[2 * f + 1] = tracks[f, :, 1]

    print("Running Tomasi-Kanade factorization...")
    motion, S, translations = factorize_tomasi_kanade(measurements)

    # Show tracking overlay on 4 evenly spaced frames
    frame_indices = np.linspace(0, T - 1, 4, dtype=int)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True, sharey=True)
    for ax, t in zip(axes, frame_indices):
        ax.imshow(video[t])
        ax.scatter(tracks[t, :, 0], tracks[t, :, 1], s=8, c="lime", edgecolors="k", linewidths=0.3)
        ax.set_title(f"Frame {t}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # 3D reconstruction plot
    X, Y, Z = S[0], S[1], S[2]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X, Y, Z, s=10, c=Z, cmap="viridis")
    ax.set_axis_off()
    ax.set_title(f"KLT Reconstruction ({n_points} pts, {T} frames)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "hotel")
    run_klt_pipeline(data_path)
