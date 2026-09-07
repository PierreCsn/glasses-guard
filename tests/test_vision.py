import numpy as np

from glasses_guard.vision import extract_features, normalize_face


def test_normalize_face_shape() -> None:
    image = np.zeros((200, 160), dtype=np.uint8)
    normalized = normalize_face(image)
    assert normalized.shape == (128, 128)


def test_extract_features_is_stable_size() -> None:
    image = np.zeros((128, 128), dtype=np.uint8)
    first = extract_features(image)
    second = extract_features(image)
    assert first.ndim == 1
    assert first.shape == second.shape
    assert np.array_equal(first, second)
