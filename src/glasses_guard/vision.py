from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.feature import hog


@dataclass(frozen=True)
class FaceDetection:
    crop: np.ndarray
    box: tuple[int, int, int, int]


class FaceDetector:
    """Small OpenCV-based frontal face detector for the V0 prototype."""

    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError("Unable to load OpenCV face cascade")

    def detect_largest(self, frame: np.ndarray) -> FaceDetection | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda item: int(item[2]) * int(item[3]))
        crop = gray[y : y + h, x : x + w]
        return FaceDetection(crop=crop, box=(int(x), int(y), int(w), int(h)))


def normalize_face(face: np.ndarray, size: int = 128) -> np.ndarray:
    resized = cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(resized)


def extract_features(face: np.ndarray) -> np.ndarray:
    normalized = normalize_face(face)
    features = hog(
        normalized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return np.asarray(features, dtype=np.float32)
