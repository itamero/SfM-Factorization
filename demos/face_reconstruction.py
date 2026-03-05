"""
Utility function for detecting face landmarks across a directory of images.

Provides:
- detect_face_landmarks: Detect 68-point face landmarks in each image
"""

import os
import cv2
import numpy as np
import dlib
from imutils import face_utils


def detect_face_landmarks(image_dir, predictor_path):
    """Detect 68-point face landmarks in every image in a directory.

    Parameters
    ----------
    image_dir : str
        Path to a directory of face images.
    predictor_path : str
        Path to dlib's shape_predictor_68_face_landmarks.dat file.

    Returns
    -------
    landmarks : np.ndarray, shape (F, 68, 2)
        Detected landmark coordinates for each frame.
    first_annotated_image : np.ndarray or None
        The first image with landmarks drawn (RGB uint8), or None if no
        faces were detected in the first frame.
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    fnames = sorted(
        f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in exts
    )
    if not fnames:
        raise RuntimeError(f"No images found in {image_dir}")

    landmarks = np.zeros((len(fnames), 68, 2))
    first_annotated_image = None

    for idx, fname in enumerate(fnames):
        image = cv2.imread(os.path.join(image_dir, fname))
        gray = clahe.apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        for rect in detector(gray, 0):
            shape = face_utils.shape_to_np(predictor(gray, rect))
            landmarks[idx] = shape
            if idx == 0:
                for (x, y) in shape:
                    cv2.circle(image, (x, y), 4, (0, 255, 0), -1)
                first_annotated_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return landmarks, first_annotated_image


def run_face_pipeline(path, predictor_path, point_size=18):
    """Run the full face landmark detection + Tomasi-Kanade reconstruction pipeline."""
    import sys
    import matplotlib.pyplot as plt

    script_dir = os.path.dirname(__file__)
    sys.path.insert(0, os.path.join(script_dir, "..", "src"))
    from factorization import factorize_tomasi_kanade

    if not os.path.isfile(predictor_path):
        print(f"ERROR: shape predictor not found at {predictor_path}")
        print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        sys.exit(1)

    print(f"Detecting face landmarks in: {path}")
    landmarks, first_image = detect_face_landmarks(path, predictor_path)
    n_frames = landmarks.shape[0]

    # Show first frame with landmarks
    if first_image is not None:
        plt.figure(figsize=(6, 6))
        plt.imshow(first_image)
        plt.title("Frame 1 — Detected Landmarks")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # Build measurement matrix
    measurements = np.zeros((2 * n_frames, 68))
    for f in range(n_frames):
        measurements[2 * f] = landmarks[f, :, 0]
        measurements[2 * f + 1] = landmarks[f, :, 1]

    print("Running Tomasi-Kanade factorization...")
    motion, S, translations = factorize_tomasi_kanade(measurements)

    # 3D reconstruction plot
    X, Y, Z = S[0], S[1], S[2]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X, Y, Z, s=point_size, c=Z, cmap="viridis")
    ax.set_axis_off()
    ax.set_title(f"Face Reconstruction (68 landmarks, {n_frames} frames)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _script_dir = os.path.dirname(__file__)
    run_face_pipeline(
        os.path.join(_script_dir, "..", "data", "face_images"),
        os.path.join(_script_dir, "..", "data", "shape_predictor_68_face_landmarks.dat"),
    )

