"""
CLI for 3D reconstruction via Tomasi-Kanade factorization.

Usage:
    python run.py path/to/images_or_video           # KLT tracking (default)
    python run.py --face path/to/face_images        # Face landmark tracking
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="3D reconstruction via Tomasi-Kanade factorization"
    )
    parser.add_argument(
        "path",
        help="Path to a video file or directory of images",
    )
    parser.add_argument(
        "--face", action="store_true",
        help="Use dlib face landmark tracking instead of KLT",
    )
    parser.add_argument(
        "--predictor", default="data/shape_predictor_68_face_landmarks.dat",
        help="Path to dlib shape predictor (default: data/shape_predictor_68_face_landmarks.dat)",
    )
    parser.add_argument(
        "--max-corners", type=int, default=1000,
        help="KLT: max feature points to detect (default: 1000)",
    )
    parser.add_argument(
        "--quality-level", type=float, default=0.01,
        help="KLT: minimum corner quality (default: 0.01)",
    )
    parser.add_argument(
        "--min-distance", type=float, default=10,
        help="KLT: minimum distance between corners (default: 10)",
    )
    parser.add_argument(
        "--min-visible", type=float, default=0.7,
        help="KLT: keep points visible in >= this fraction of frames (default: 0.7)",
    )

    args = parser.parse_args()

    if args.face:
        from demos.face_reconstruction import run_face_pipeline
        run_face_pipeline(args.path, args.predictor)
    else:
        from demos.klt_tracking_reconstruction import run_klt_pipeline
        run_klt_pipeline(
            args.path,
            max_corners=args.max_corners,
            quality_level=args.quality_level,
            min_distance=args.min_distance,
            min_visible=args.min_visible,
        )


if __name__ == "__main__":
    main()
