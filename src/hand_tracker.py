from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import platform
import cv2
import numpy as np

HAND_CONNECTIONS = (
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
)

@dataclass
class HandResult:
    detected: bool
    landmarks: Optional[np.ndarray] = None
    handedness: Optional[str] = None

class HandTracker:
    """MediaPipe Tasks HandLandmarker wrapper for VIDEO or IMAGE mode."""
    def __init__(self, model_path: Path, max_num_hands: int = 1, running_mode: str = "video") -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing MediaPipe model: {model_path}. Run: python tools/download_models.py"
            )
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("mediapipe is not installed. Run pip install -r requirements.txt") from exc

        # mediapipe 1.0.1 currently aborts inside DrishtiMetalHelper while opening
        # HandLandmarker graphs on Apple Silicon macOS. Native SIGABRT cannot be
        # caught by Python, so fail before graph construction with an actionable
        # environment fix instead of crashing the whole preprocessing job.
        mp_version = getattr(mp, "__version__", "unknown")
        if platform.system() == "Darwin" and platform.machine() == "arm64" and mp_version == "1.0.1":
            raise RuntimeError(
                "mediapipe 1.0.1 is not supported for HandLandmarker on Apple Silicon in this project "
                "because it can abort inside DrishtiMetalHelper. Reinstall project dependencies with:\n"
                "  python -m pip uninstall -y mediapipe\n"
                "  python -m pip install -r requirements.txt\n"
                "This installs mediapipe==0.10.30 on arm64 macOS."
            )

        mode = running_mode.strip().lower()
        if mode not in {"video", "image"}:
            raise ValueError("running_mode must be 'video' or 'image'")

        self.mp = mp
        self.running_mode = mode
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode
        mp_running_mode = RunningMode.VIDEO if mode == "video" else RunningMode.IMAGE

        base_options = BaseOptions(
            model_asset_path=str(model_path),
            delegate=BaseOptions.Delegate.CPU,
        )
        options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_running_mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self.timestamp_ms = 0

    def process(self, frame_bgr: np.ndarray) -> HandResult:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        if self.running_mode == "video":
            self.timestamp_ms += 1
            result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
        else:
            result = self.landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return HandResult(False)
        pts = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
        handedness = None
        if result.handedness and result.handedness[0]:
            handedness = result.handedness[0][0].category_name
        return HandResult(True, pts, handedness)

    def close(self) -> None:
        self.landmarker.close()

    @staticmethod
    def draw(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        pts = [(int(x*w), int(y*h)) for x, y, _ in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 4, (0, 0, 255), -1)
        return frame
