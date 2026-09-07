from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2

from .vision import FaceDetector, normalize_face


PRIVATE_DATA_DIR = Path("data/private")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture local face examples for Glasses Guard")
    parser.add_argument("--label", required=True, choices=("glasses", "no_glasses"))
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dir = PRIVATE_DATA_DIR / args.label
    target_dir.mkdir(parents=True, exist_ok=True)

    detector = FaceDetector()
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError("Unable to open webcam")

    print("SPACE: save detected face | q: quit")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                continue

            detection = detector.detect_largest(frame)
            preview = frame.copy()
            if detection is not None:
                x, y, w, h = detection.box
                cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 255, 255), 2)

            cv2.imshow("Glasses Guard capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32 and detection is not None:
                stamp = time.time_ns()
                output = target_dir / f"sample-{stamp}.png"
                cv2.imwrite(str(output), normalize_face(detection.crop))
                print(f"saved {args.label} sample")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
