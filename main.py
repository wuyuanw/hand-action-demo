from __future__ import annotations
import argparse
from pathlib import Path
import cv2
from config import *
from src.camera import Camera
from src.feature_extractor import StatefulFeatureExtractor
from src.dhg2016 import DHGCompatibleFeatureExtractor
from src.hand_tracker import HandTracker
from src.heuristic_features import compute_heuristics
from src.predictor import Predictor
from src.sequence_buffer import SequenceBuffer
from src.smoothing import PredictionSmoother
from src.utils import FPSMeter
from src.visualization import draw_status

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=CAMERA_INDEX)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--checkpoint", type=Path, default=MODEL_PATH,
                    help="Checkpoint to use. For DHG-2016: models/hand_action_dhg14_lstm.pth")
    ap.add_argument("--feature-mode", choices=["legacy", "dhg"], default="legacy",
                    help="Use 'dhg' with a checkpoint trained by tools/train_dhg2016.py")
    args = ap.parse_args()

    camera = Camera(args.camera, CAMERA_WIDTH, CAMERA_HEIGHT)
    tracker = HandTracker(HAND_LANDMARKER_MODEL_PATH, MAX_NUM_HANDS)
    extractor = DHGCompatibleFeatureExtractor() if args.feature_mode == "dhg" else StatefulFeatureExtractor()
    buffer = SequenceBuffer(SEQUENCE_LENGTH, INPUT_SIZE)
    predictor = Predictor(args.checkpoint)
    smoother = PredictionSmoother(SMOOTHING_WINDOW, CONFIDENCE_THRESHOLD)
    fps_meter = FPSMeter()
    no_hand = 0
    frame_id = 0
    action, raw_action, confidence = "collecting sequence", "", 0.0
    previous_debug_wrist = None

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            result = tracker.process(frame)
            frame_id += 1
            if result.detected and result.landmarks is not None:
                no_hand = 0
                tracker.draw(frame, result.landmarks)
                if args.debug:
                    h = compute_heuristics(result.landmarks, previous_debug_wrist)
                    previous_debug_wrist = result.landmarks[0].copy()
                    print(" ".join(f"{k}={v:.4f}" for k, v in h.items()))
                feature = extractor.extract(result.landmarks)
                buffer.append(feature)
                if buffer.is_ready() and frame_id % INFERENCE_STRIDE == 0:
                    pred = predictor.predict(buffer.get_sequence())
                    raw_action, confidence = pred.class_name, pred.confidence
                    action = smoother.update(raw_action, confidence)
                elif not buffer.is_ready():
                    action, confidence = "collecting sequence", 0.0
            else:
                no_hand += 1
                previous_debug_wrist = None
                if no_hand >= NO_HAND_RESET_FRAMES:
                    buffer.clear(); extractor.reset(); smoother.clear()
                    action, raw_action, confidence = "no hand", "", 0.0

            fps = fps_meter.tick()
            draw_status(frame, action, confidence, fps, len(buffer), SEQUENCE_LENGTH, raw_action if args.debug else None)
            cv2.imshow("Hand Action Demo", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                break
    finally:
        tracker.close(); camera.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
