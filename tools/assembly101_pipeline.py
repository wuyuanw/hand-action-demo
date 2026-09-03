from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n+", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="End-to-end Assembly101: download -> MediaPipe feature extraction -> optional LSTM training."
    )
    ap.add_argument("--root", type=Path, default=ROOT / "data" / "assembly101",
                    help="Raw Assembly101 root (recordings + official annotations).")
    ap.add_argument("--output", type=Path, default=ROOT / "dataset_assembly101",
                    help="Generated 30x66 MediaPipe feature dataset.")
    ap.add_argument("--source", choices=["hf720", "hf", "gdrive"], default="hf720",
                    help="Default hf720 uses the public pablovela5620/assembly101-720p video variant.")
    ap.add_argument("--mirror", choices=["official", "cn"], default="official",
                    help="Hugging Face endpoint. 'cn' uses https://hf-mirror.com.")
    ap.add_argument("--views", default="v8",
                    help="v1..v8, e1..e4, fixed, egocentric or all. Default: v8 fixed RGB.")
    ap.add_argument("--videos", default="all", help="all or comma-separated recording names.")
    ap.add_argument("--labels", default="pick up,put down,position,screw,unscrew",
                    help="Comma-separated verb whitelist. Empty string means all verbs.")
    ap.add_argument("--max-recordings", type=int, default=0,
                    help="Limit matched recordings for a quick end-to-end test; 0 means no limit.")
    ap.add_argument("--hf-token", default=None,
                    help="Only required when official gated annotations/videos must be downloaded.")
    ap.add_argument("--min-hand-ratio", type=float, default=0.60)
    ap.add_argument("--limit", type=int, default=0,
                    help="Maximum accepted feature samples per class in each split; 0 means unlimited.")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--client-secrets", type=Path, default=None)
    ap.add_argument("--credentials", type=Path, default=None)
    ap.add_argument("--authenticate", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    output = args.output.resolve()
    annotations = root / "annotations" / "fine-grained-annotations"
    recordings = root / "recordings"

    if not args.skip_download:
        cmd = [
            sys.executable, "tools/download_assembly101.py",
            "--root", str(root),
            "--source", args.source,
            "--mirror", args.mirror,
            "--views", args.views,
            "--videos", args.videos,
        ]
        if args.labels:
            cmd.extend(["--labels", args.labels])
        if args.max_recordings > 0:
            cmd.extend(["--max-recordings", str(args.max_recordings)])
        if args.hf_token:
            cmd.extend(["--hf-token", args.hf_token])
        if args.resume:
            cmd.append("--resume")
        if args.client_secrets:
            cmd.extend(["--client-secrets", str(args.client_secrets)])
        if args.credentials:
            cmd.extend(["--credentials", str(args.credentials)])
        if args.authenticate:
            cmd.append("--authenticate")
        run(cmd)

    if not args.skip_prepare:
        cmd = [
            sys.executable, "tools/prepare_assembly101.py",
            "--video-root", str(recordings),
            "--annotation-dir", str(annotations),
            "--output", str(output),
            "--label-level", "verb",
            "--min-hand-ratio", str(args.min_hand_ratio),
        ]
        if args.labels:
            cmd.extend(["--labels", args.labels])
        if args.limit > 0:
            cmd.extend(["--limit", str(args.limit)])
        run(cmd)

    if args.train:
        run([
            sys.executable, "tools/train_assembly101.py",
            "--dataset", str(output),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
        ])

    print("\nAssembly101 pipeline complete.")
    print("raw root:", root)
    print("feature dataset:", output)
    if not args.train:
        print("train next with:")
        print(f"  {sys.executable} tools/train_assembly101.py --dataset \"{output}\"")


if __name__ == "__main__":
    main()
