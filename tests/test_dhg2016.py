import numpy as np
from src.dhg2016 import DHG14_CLASS_NAMES, DHG_TO_MEDIAPIPE21, dhg22_to_features, resample_sequence


def test_dhg_mapping_and_feature_shape():
    seq = np.zeros((24, 22, 3), dtype=np.float32)
    for t in range(24):
        for j in range(22):
            seq[t, j] = [j * 0.01 + t * 0.001, j * 0.02, j * 0.005]
    x = dhg22_to_features(seq, 30)
    assert x.shape == (30, 66)
    assert np.isfinite(x).all()


def test_dhg_constants():
    assert DHG14_CLASS_NAMES == [
        "grab", "expand", "pinch", "rotation_cw", "rotation_ccw", "tap",
        "swipe_right", "swipe_left", "swipe_up", "swipe_down",
        "swipe_x", "swipe_v", "swipe_plus", "shake",
    ]
    assert DHG_TO_MEDIAPIPE21.tolist() == [0] + list(range(2, 22))


def test_resample_length():
    seq = np.zeros((45, 21, 3), dtype=np.float32)
    out = resample_sequence(seq, 30)
    assert out.shape == (30, 21, 3)
