from __future__ import annotations

import argparse
from pathlib import Path
import platform
import subprocess
import time

import cv2
import joblib

from .vision import FaceDetector, extract_features


MODEL_PATH = Path("models/private/glasses_classifier.joblib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor whether glasses appear to be worn")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--sample-interval", type=float, default=5.0)
    parser.add_argument("--alert-after", type=float, default=30.0)
    parser.add_argument("--cooldown", type=float, default=900.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def notify(message: str) -> None:
    if platform.system() == "Darwin":
        safe_message = message.replace('"', '\\"')
        subprocess.run(
            ["osascript", "-e", f'display notification "{safe_message}" with title "Glasses Guard"'],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    print(f"[Glasses Guard] {message}")


def main() -> None:
    args = parse_args()
    if not MODEL_PATH.exists():
        raise RuntimeError("No trained model found. Run glasses-guard-train first.")

    model = joblib.load(MODEL_PATH)
    detector = FaceDetector()
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError("Unable to open webcam")

    no_glasses_since: float | None = None
    last_alert = float("-inf")

    print("Glasses Guard is running locally. Press Ctrl+C to stop.")
    try:
        while True:
            started = time.monotonic()
            ok, frame = camera.read()
            if ok:
                detection = detector.detect_largest(frame)
                if detection is None:
                    no_glasses_since = None
                else:
                    features = extract_features(detection.crop).reshape(1, -1)
                    glasses_probability = float(model.predict_proba(features)[0][1])
                    if glasses_probability < args.threshold:
                        if no_glasses_since is None:
                            no_glasses_since = started
                        persistent_for = started - no_glasses_since
                        cooldown_elapsed = started - last_alert >= args.cooldown
                        if persistent_for >= args.alert_after and cooldown_elapsed:
                            notify("Your glasses do not appear to be on.")
                            last_alert = started
                    else:
                        no_glasses_since = None

            elapsed = time.monotonic() - started
            time.sleep(max(0.0, args.sample_interval - elapsed))
    except KeyboardInterrupt:
        print("Glasses Guard stopped.")
    finally:
        camera.release()


if __name__ == "__main__":
    main()
