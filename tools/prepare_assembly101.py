from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import HAND_LANDMARKER_MODEL_PATH, SEQUENCE_LENGTH
from src.feature_extractor import StatefulFeatureExtractor
from src.hand_tracker import HandTracker

ANNOTATION_FPS = 30.0
SPLIT_FILES = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}
SPLIT_SEED_OFFSET = {"train": 0, "validation": 1, "test": 2}


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def parse_filter(value: str | None) -> set[str] | None:
    if not value:
        return None
    p = Path(value)
    if p.exists():
        items = [line.strip() for line in p.read_text(encoding="utf-8").splitlines()]
    else:
        items = [x.strip() for x in value.split(",")]
    items = [x.lower() for x in items if x]
    return set(items) if items else None


def parse_recordings(value: str | None) -> set[str] | None:
    if not value:
        return None
    p = Path(value)
    if p.exists():
        items = [line.strip() for line in p.read_text(encoding="utf-8").splitlines()]
    else:
        items = [x.strip() for x in value.split(",")]
    items = [x for x in items if x]
    return set(items) if items else None


def load_rows(annotation_dir: Path, split: str) -> list[dict[str, str]]:
    path = annotation_dir / SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"Missing Assembly101 annotation file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def local_video_index(video_root: Path) -> set[str]:
    if not video_root.exists():
        return set()
    return {p.relative_to(video_root).as_posix() for p in video_root.rglob("*.mp4")}


def interpolate_missing(points: list[np.ndarray | None]) -> np.ndarray | None:
    valid = [i for i, p in enumerate(points) if p is not None]
    if not valid:
        return None
    arr = np.full((len(points), 21, 3), np.nan, dtype=np.float32)
    for i in valid:
        arr[i] = points[i]
    t = np.arange(len(points), dtype=np.float32)
    good_t = np.asarray(valid, dtype=np.float32)
    for joint in range(21):
        for axis in range(3):
            arr[:, joint, axis] = np.interp(t, good_t, arr[valid, joint, axis])
    return arr


def extract_segment(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    tracker: HandTracker,
    sequence_length: int,
    min_hand_ratio: float,
) -> tuple[np.ndarray | None, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0.0
    start_sec = max(0.0, float(start_frame) / ANNOTATION_FPS)
    end_sec = max(start_sec, float(end_frame) / ANNOTATION_FPS)
    if end_sec <= start_sec:
        cap.release()
        return None, 0.0
    times = np.linspace(start_sec, end_sec, sequence_length, endpoint=False, dtype=np.float64)
    landmarks: list[np.ndarray | None] = []
    try:
        for sec in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(sec * 1000.0))
            ok, frame = cap.read()
            if not ok:
                landmarks.append(None)
                continue
            result = tracker.process(frame)
            landmarks.append(result.landmarks.copy() if result.detected and result.landmarks is not None else None)
    finally:
        cap.release()
    valid_ratio = sum(p is not None for p in landmarks) / max(1, len(landmarks))
    if valid_ratio < min_hand_ratio:
        return None, valid_ratio
    filled = interpolate_missing(landmarks)
    if filled is None:
        return None, valid_ratio
    extractor = StatefulFeatureExtractor()
    features = np.stack([extractor.extract(frame) for frame in filled]).astype(np.float32)
    return features, valid_ratio


def split_limit(args: argparse.Namespace, split: str) -> int:
    specific = {
        "train": args.train_limit,
        "validation": args.validation_limit,
        "test": args.test_limit,
    }[split]
    return specific if specific > 0 else args.limit


def main() -> None:
    ap = argparse.ArgumentParser(description="Assembly101 -> MediaPipe -> fixed 30x66 training samples.")
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--annotation-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=ROOT / "dataset_assembly101")
    ap.add_argument("--label-level", choices=["verb", "action"], default="verb")
    ap.add_argument("--labels", type=str, default=None)
    ap.add_argument(
        "--recordings",
        type=str,
        default=None,
        help="Optional comma-separated recording whitelist or text file with one recording per line.",
    )
    ap.add_argument("--splits", nargs="+", choices=list(SPLIT_FILES), default=list(SPLIT_FILES))
    ap.add_argument("--min-hand-ratio", type=float, default=0.60)
    ap.add_argument("--limit", type=int, default=0,
                    help="Fallback maximum accepted samples PER CLASS in each split; 0 means unlimited.")
    ap.add_argument("--train-limit", type=int, default=0,
                    help="Per-class train limit; overrides --limit when > 0.")
    ap.add_argument("--validation-limit", type=int, default=0,
                    help="Per-class validation limit; overrides --limit when > 0.")
    ap.add_argument("--test-limit", type=int, default=0,
                    help="Per-class test limit; overrides --limit when > 0.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Deterministic sampling seed used to mix segments from multiple recordings.")
    args = ap.parse_args()

    if not HAND_LANDMARKER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing MediaPipe hand model: {HAND_LANDMARKER_MODEL_PATH}. "
            "Run python tools/download_models.py first."
        )

    label_filter = parse_filter(args.labels)
    recording_filter = parse_recordings(args.recordings)
    video_index = local_video_index(args.video_root)
    if not video_index:
        raise RuntimeError(f"No .mp4 files found under {args.video_root}")
    print(f"Indexed {len(video_index)} local Assembly101 video(s).")
    if recording_filter is not None:
        print(f"Restricting preprocessing to {len(recording_filter)} selected recording(s).")

    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "metadata.csv"
    metadata_exists = metadata_path.exists()
    tracker = HandTracker(HAND_LANDMARKER_MODEL_PATH, max_num_hands=1, running_mode="image")
    class_names: set[str] = set()
    stats: dict[str, dict[str, object]] = {}

    try:
        with metadata_path.open("a", encoding="utf-8", newline="") as meta_f:
            writer = csv.DictWriter(
                meta_f,
                fieldnames=[
                    "split", "sample", "label", "label_text", "video", "start_frame", "end_frame",
                    "action_id", "verb_id", "noun_id", "action_cls", "verb_cls", "noun_cls", "hand_ratio",
                ],
            )
            if not metadata_exists:
                writer.writeheader()

            for split in args.splits:
                rows = load_rows(args.annotation_dir, split)
                filtered_rows: list[dict[str, str]] = []
                skipped_filter = skipped_missing_video = skipped_recording_filter = 0
                available_by_class: Counter[str] = Counter()
                for row in rows:
                    label_text = row["verb_cls"] if args.label_level == "verb" else row["action_cls"]
                    if label_filter is not None and label_text.strip().lower() not in label_filter:
                        skipped_filter += 1
                        continue
                    video_rel = row["video"].replace("\\", "/")
                    recording = video_rel.split("/", 1)[0]
                    if recording_filter is not None and recording not in recording_filter:
                        skipped_recording_filter += 1
                        continue
                    if video_rel not in video_index:
                        skipped_missing_video += 1
                        continue
                    filtered_rows.append(row)
                    available_by_class[slugify(label_text)] += 1

                rng = random.Random(args.seed + SPLIT_SEED_OFFSET[split])
                rng.shuffle(filtered_rows)
                current_limit = split_limit(args, split)

                print(
                    f"{split}: local target segments={len(filtered_rows)} "
                    f"(filtered={skipped_filter}, recording_filter={skipped_recording_filter}, "
                    f"missing_video={skipped_missing_video})"
                )
                print(f"{split}: available by class={dict(sorted(available_by_class.items()))}")
                print(f"{split}: per-class target={current_limit if current_limit > 0 else 'unlimited'} seed={args.seed}")

                accepted_by_class: Counter[str] = Counter()
                accepted = skipped_hand = skipped_limit = 0
                target_classes = set(available_by_class)
                accepted_recordings: set[str] = set()

                for row in tqdm(filtered_rows, desc=f"Assembly101 {split}"):
                    label_text = row["verb_cls"] if args.label_level == "verb" else row["action_cls"]
                    label = slugify(label_text)
                    if current_limit > 0 and accepted_by_class[label] >= current_limit:
                        skipped_limit += 1
                        if target_classes and all(accepted_by_class[c] >= current_limit for c in target_classes):
                            break
                        continue

                    video_rel = row["video"].replace("\\", "/")
                    features, hand_ratio = extract_segment(
                        args.video_root / video_rel,
                        int(row["start_frame"]),
                        int(row["end_frame"]),
                        tracker,
                        SEQUENCE_LENGTH,
                        args.min_hand_ratio,
                    )
                    if features is None:
                        skipped_hand += 1
                        continue

                    out_dir = args.output / split / label
                    out_dir.mkdir(parents=True, exist_ok=True)
                    sample_id = f"{split}_{int(row['id']):08d}"
                    out_path = out_dir / f"{sample_id}.npy"
                    np.save(out_path, features)
                    class_names.add(label)
                    accepted_recordings.add(video_rel.split("/", 1)[0])
                    writer.writerow({
                        "split": split,
                        "sample": str(out_path.relative_to(args.output)),
                        "label": label,
                        "label_text": label_text,
                        "video": row["video"],
                        "start_frame": row["start_frame"],
                        "end_frame": row["end_frame"],
                        "action_id": row.get("action_id", ""),
                        "verb_id": row.get("verb_id", ""),
                        "noun_id": row.get("noun_id", ""),
                        "action_cls": row.get("action_cls", ""),
                        "verb_cls": row.get("verb_cls", ""),
                        "noun_cls": row.get("noun_cls", ""),
                        "hand_ratio": f"{hand_ratio:.3f}",
                    })
                    meta_f.flush()
                    accepted += 1
                    accepted_by_class[label] += 1

                    if current_limit > 0 and target_classes and all(
                        accepted_by_class[c] >= current_limit for c in target_classes
                    ):
                        break

                stats[split] = {
                    "available_target_segments": len(filtered_rows),
                    "available_by_class": dict(sorted(available_by_class.items())),
                    "accepted": accepted,
                    "accepted_by_class": dict(sorted(accepted_by_class.items())),
                    "accepted_recordings": len(accepted_recordings),
                    "accepted_recording_names": sorted(accepted_recordings),
                    "skipped_filter": skipped_filter,
                    "skipped_recording_filter": skipped_recording_filter,
                    "skipped_missing_video": skipped_missing_video,
                    "skipped_hand": skipped_hand,
                    "skipped_limit": skipped_limit,
                }
    finally:
        tracker.close()

    manifest = {
        "dataset": "Assembly101",
        "annotation_fps": ANNOTATION_FPS,
        "sequence_length": SEQUENCE_LENGTH,
        "input_size": 66,
        "label_level": args.label_level,
        "class_names": sorted(class_names),
        "sampling_seed": args.seed,
        "selected_recordings": sorted(recording_filter) if recording_filter is not None else None,
        "stats": stats,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
