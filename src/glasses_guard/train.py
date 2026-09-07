from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .vision import extract_features


DATA_DIR = Path("data/private")
MODEL_PATH = Path("models/private/glasses_classifier.joblib")
LABELS = {"no_glasses": 0, "glasses": 1}


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []

    for label_name, label_value in LABELS.items():
        directory = DATA_DIR / label_name
        for path in sorted(directory.glob("*.png")):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            features.append(extract_features(image))
            labels.append(label_value)

    if not features:
        raise RuntimeError("No training samples found under data/private")
    if len(set(labels)) < 2:
        raise RuntimeError("Training requires both glasses and no_glasses samples")

    return np.vstack(features), np.asarray(labels, dtype=np.int64)


def main() -> None:
    x, y = load_dataset()
    if len(y) < 20:
        raise RuntimeError("Collect at least 20 total samples before training")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    print(classification_report(y_test, predictions, target_names=["no_glasses", "glasses"]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
