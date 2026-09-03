from __future__ import annotations
from pathlib import Path
import re
import numpy as np

# Official DHG-14 gesture ids:
# 1 Grab, 2 Expand, 3 Pinch, 4 Rotation CW, 5 Rotation CCW, 6 Tap,
# 7 Swipe Right, 8 Swipe Left, 9 Swipe Up, 10 Swipe Down,
# 11 Swipe X, 12 Swipe V, 13 Swipe +, 14 Shake.
DHG14_CLASS_NAMES = [
    "grab", "expand", "pinch", "rotation_cw", "rotation_ccw", "tap",
    "swipe_right", "swipe_left", "swipe_up", "swipe_down",
    "swipe_x", "swipe_v", "swipe_plus", "shake",
]

# DHG joint layout: 0 wrist, 1 palm, 2..21 = 4 joints for each finger.
# MediaPipe has 21 landmarks and no explicit palm joint, so dropping DHG joint 1
# gives a topology-compatible 21-point hand skeleton for this baseline.
DHG_TO_MEDIAPIPE21 = np.array([0] + list(range(2, 22)), dtype=np.int64)


class DHGCompatibleFeatureExtractor:
    """21-point local pose + scale-normalized wrist velocity for DHG/MediaPipe transfer."""
    def __init__(self) -> None:
        self.previous_wrist: np.ndarray | None = None

    def reset(self) -> None:
        self.previous_wrist = None

    def extract(self, landmarks: np.ndarray) -> np.ndarray:
        landmarks = np.asarray(landmarks, dtype=np.float32)
        if landmarks.shape != (21, 3):
            raise ValueError(f"Expected landmarks shape (21,3), got {landmarks.shape}")
        wrist = landmarks[0].copy()
        scale = max(float(np.linalg.norm(landmarks[9] - wrist)), 1e-6)
        local = (landmarks - wrist) / scale
        if self.previous_wrist is None:
            velocity = np.zeros(3, dtype=np.float32)
        else:
            velocity = (wrist - self.previous_wrist) / scale
        self.previous_wrist = wrist
        return np.concatenate([local.reshape(-1), velocity]).astype(np.float32)


def resample_sequence(sequence: np.ndarray, target_length: int = 30) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 3 or sequence.shape[1:] != (21, 3):
        raise ValueError(f"Expected [T,21,3], got {sequence.shape}")
    if len(sequence) < 2:
        return np.repeat(sequence[:1], target_length, axis=0)
    old_t = np.linspace(0.0, 1.0, len(sequence), dtype=np.float32)
    new_t = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    out = np.empty((target_length, 21, 3), dtype=np.float32)
    for joint in range(21):
        for axis in range(3):
            out[:, joint, axis] = np.interp(new_t, old_t, sequence[:, joint, axis])
    return out


def dhg22_to_features(skeleton22: np.ndarray, target_length: int = 30) -> np.ndarray:
    skeleton22 = np.asarray(skeleton22, dtype=np.float32)
    if skeleton22.ndim != 3 or skeleton22.shape[1:] != (22, 3):
        raise ValueError(f"Expected [T,22,3], got {skeleton22.shape}")
    points21 = resample_sequence(skeleton22[:, DHG_TO_MEDIAPIPE21], target_length)
    extractor = DHGCompatibleFeatureExtractor()
    return np.stack([extractor.extract(frame) for frame in points21]).astype(np.float32)


def subject_id_from_name(path: Path) -> int:
    m = re.search(r"_s(\d+)_", path.stem)
    if not m:
        raise ValueError(f"Cannot parse subject id from {path.name}")
    return int(m.group(1))
