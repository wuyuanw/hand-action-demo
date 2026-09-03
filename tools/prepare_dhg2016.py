from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import SEQUENCE_LENGTH
from src.dhg2016 import DHG14_CLASS_NAMES, dhg22_to_features


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert DHG-14/28 skeleton_world.txt files to this demo's 30x66 feature format.")
    ap.add_argument("--dhg-root", type=Path, required=True, help="Root containing informations_troncage_sequences.txt and gesture_* folders")
    ap.add_argument("--output", type=Path, default=ROOT / "dataset_dhg2016")
    args = ap.parse_args()

    info_path = args.dhg_root / "informations_troncage_sequences.txt"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing {info_path}")

    rows = np.loadtxt(info_path, dtype=int)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in DHG14_CLASS_NAMES:
        (args.output / name).mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    for row in rows:
        gesture, finger, subject, trial, begin_frame, end_frame = map(int, row[:6])
        if not 1 <= gesture <= 14:
            skipped += 1
            continue
        src = (args.dhg_root / f"gesture_{gesture}" / f"finger_{finger}" /
               f"subject_{subject}" / f"essai_{trial}" / "skeleton_world.txt")
        if not src.exists():
            print(f"skip missing: {src}")
            skipped += 1
            continue
        raw = np.loadtxt(src, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 66:
            print(f"skip bad shape {raw.shape}: {src}")
            skipped += 1
            continue
        # DHG annotations are used by common reference implementations as Python
        # slice indices [begin_frame:end_frame + 1].
        raw = raw[begin_frame:end_frame + 1]
        skeleton = raw.reshape(-1, 22, 3)
        feat = dhg22_to_features(skeleton, SEQUENCE_LENGTH)
        out_name = f"g{gesture:02d}_f{finger}_s{subject:02d}_e{trial}.npy"
        np.save(args.output / DHG14_CLASS_NAMES[gesture - 1] / out_name, feat)
        converted += 1

    print(f"converted={converted} skipped={skipped} output={args.output}")


if __name__ == "__main__":
    main()
