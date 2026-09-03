from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_LABELS = "pick up,put down,position,screw,unscrew"
DEFAULT_DATA_ROOT = ROOT / "data" / "assembly101"
DEFAULT_DATASET = ROOT / "dataset_assembly101"
DEFAULT_MODEL = ROOT / "models" / "hand_action_assembly101_lstm.pth"
DEFAULT_METRICS = ROOT / "outputs" / "assembly101_baseline_metrics.json"


def run(cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print("+", " ".join(str(x) for x in cmd))
    print("=" * 80)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def target_classes(labels: str) -> list[str]:
    return sorted({slugify(x) for x in labels.split(",") if x.strip()})


def load_manifest(dataset: Path) -> dict | None:
    path = dataset / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def dataset_meets_targets(
    dataset: Path,
    classes: list[str],
    train_limit: int,
    validation_limit: int,
    test_limit: int,
    min_train_recordings: int,
    min_eval_recordings: int,
) -> tuple[bool, str]:
    manifest = load_manifest(dataset)
    if manifest is None:
        return False, "manifest.json 不存在或不可读"

    found_classes = sorted(manifest.get("class_names", []))
    if found_classes != classes:
        return False, f"类别不匹配: 当前={found_classes}, 目标={classes}"

    stats = manifest.get("stats", {})
    requirements = {
        "train": (train_limit, min_train_recordings),
        "validation": (validation_limit, min_eval_recordings),
        "test": (test_limit, min_eval_recordings),
    }
    for split, (limit, min_recordings) in requirements.items():
        split_stats = stats.get(split, {})
        accepted = split_stats.get("accepted_by_class", {})
        for cls in classes:
            if int(accepted.get(cls, 0)) < limit:
                return False, f"{split}/{cls} 只有 {accepted.get(cls, 0)} 个，目标 {limit} 个"
        recording_count = int(split_stats.get("accepted_recordings", 0))
        if recording_count < min_recordings:
            return False, f"{split} 只有 {recording_count} 个有效 recording，目标至少 {min_recordings} 个"

    return True, "现有训练集满足要求"


def print_dataset_summary(dataset: Path, classes: list[str]) -> None:
    manifest = load_manifest(dataset)
    if manifest is None:
        return
    print("\n数据集摘要:")
    stats = manifest.get("stats", {})
    for split in ("train", "validation", "test"):
        s = stats.get(split, {})
        accepted = s.get("accepted_by_class", {})
        recording_count = s.get("accepted_recordings", "?")
        counts = ", ".join(f"{cls}={accepted.get(cls, 0)}" for cls in classes)
        print(f"  {split:10s} recordings={recording_count} | {counts}")


def choose_baseline_recordings(
    annotation_dir: Path,
    labels: str,
    recordings_per_split: int,
    views: str,
    mirror: str,
) -> tuple[dict[str, list[str]], list[str]]:
    from tools.download_assembly101 import (
        available_720p_recordings,
        configure_hf_endpoint,
        recordings_per_split as select_per_split,
    )

    configure_hf_endpoint(mirror)
    available = available_720p_recordings(views)
    label_filter = {x.strip().lower() for x in labels.split(",") if x.strip()}
    by_split, coverage = select_per_split(
        annotation_dir,
        label_filter,
        recordings_per_split,
        available,
        fill_to_limit=True,
    )

    print("\nBaseline recording selection:")
    for split in ("train", "validation", "test"):
        names = by_split.get(split, [])
        covered = ", ".join(sorted(coverage.get(split, set()))) or "none"
        print(f"  {split}: {len(names)} recording(s), covered verbs: {covered}")
        if len(names) < recordings_per_split:
            raise RuntimeError(
                f"{split} only has {len(names)} selectable 720p recordings; requested {recordings_per_split}."
            )
        missing = label_filter - coverage.get(split, set())
        if missing:
            raise RuntimeError(f"{split} cannot cover target verbs: {', '.join(sorted(missing))}")

    selected = sorted({name for names in by_split.values() for name in names})
    print(f"  total unique recordings: {len(selected)}")
    return by_split, selected


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Assembly101 基础动作 baseline：自动准备数据 -> MediaPipe -> LSTM -> 验证/测试 -> 保存最佳模型。"
    )
    ap.add_argument("--quick", action="store_true", help="快速冒烟测试：1 recording/split、5 samples/class、5 epochs。")
    ap.add_argument("--force-rebuild", action="store_true", help="强制重新生成骨架训练集；原始 720p 视频仍会复用。")
    ap.add_argument("--reuse-dataset", action="store_true", help="只训练现有 dataset_assembly101；仍会检查类别、样本量和 recording 数。")
    ap.add_argument("--labels", default=DEFAULT_LABELS, help="逗号分隔的 Assembly101 verb 类别。")
    ap.add_argument("--recordings", type=int, default=None, help="每个官方 split 固定选择多少个 recording。")
    ap.add_argument("--train-per-class", type=int, default=None)
    ap.add_argument("--validation-per-class", type=int, default=None)
    ap.add_argument("--test-per-class", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.02)
    ap.add_argument("--early-stop-patience", type=int, default=12)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--views", default="v8")
    ap.add_argument("--mirror", choices=["official", "cn"], default="official")
    ap.add_argument("--min-hand-ratio", type=float, default=0.60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS)
    args = ap.parse_args()

    if args.force_rebuild and args.reuse_dataset:
        ap.error("--force-rebuild 和 --reuse-dataset 不能同时使用。")

    if args.quick:
        recordings = args.recordings or 1
        train_limit = args.train_per_class or 5
        validation_limit = args.validation_per_class or 5
        test_limit = args.test_per_class or 5
        epochs = args.epochs or 5
        min_train_recordings = min_eval_recordings = 1
    else:
        recordings = args.recordings or 6
        train_limit = args.train_per_class or 150
        validation_limit = args.validation_per_class or 50
        test_limit = args.test_per_class or 50
        epochs = args.epochs or 80
        min_train_recordings = min(4, recordings)
        min_eval_recordings = min(3, recordings)

    for name, value in {
        "--recordings": recordings,
        "--train-per-class": train_limit,
        "--validation-per-class": validation_limit,
        "--test-per-class": test_limit,
        "--epochs": epochs,
        "--batch-size": args.batch_size,
        "--early-stop-patience": args.early_stop_patience,
    }.items():
        if value < 1:
            ap.error(f"{name} 必须 >= 1")
    if args.lr <= 0 or args.weight_decay < 0 or args.grad_clip < 0:
        ap.error("lr 必须 > 0，weight-decay/grad-clip 必须 >= 0")
    if not 0.0 <= args.label_smoothing < 1.0:
        ap.error("--label-smoothing 必须在 [0, 1) 范围内")

    data_root = args.data_root.resolve()
    dataset = args.dataset.resolve()
    model_out = args.model_out.resolve()
    metrics_out = args.metrics_out.resolve()
    classes = target_classes(args.labels)
    if not classes:
        ap.error("--labels 不能为空")

    print("\nAssembly101 基础动作模型 baseline v1")
    print(f"  模式: {'QUICK' if args.quick else 'BASELINE'}")
    print(f"  类别: {', '.join(classes)}")
    print(f"  视频: hf720 / {args.views} / 每 split 固定 {recordings} 个 recording")
    print(f"  最低多样性: train>={min_train_recordings} recordings, val/test>={min_eval_recordings}")
    print(f"  样本: train={train_limit}/class, validation={validation_limit}/class, test={test_limit}/class")
    print(f"  训练: epochs<={epochs}, batch={args.batch_size}, lr={args.lr}, early_stop={args.early_stop_patience}")
    print("  选模指标: validation Macro-F1（同分时看 Accuracy）")
    print(f"  模型: {model_out}")
    print(f"  指标: {metrics_out}")

    ready, reason = dataset_meets_targets(
        dataset,
        classes,
        train_limit,
        validation_limit,
        test_limit,
        min_train_recordings,
        min_eval_recordings,
    )

    if args.reuse_dataset:
        if not ready:
            raise RuntimeError(f"--reuse-dataset 指定了复用，但现有数据不满足 baseline 要求：{reason}")
        print("\n复用现有骨架数据集（跳过下载和 MediaPipe 抽取）。")
        print_dataset_summary(dataset, classes)
    elif args.force_rebuild or not ready:
        print(f"\n需要重新构建骨架训练集: {reason}")
        if dataset.exists():
            print(f"清理旧的生成数据: {dataset}")
            shutil.rmtree(dataset)
        print("原始视频 data/assembly101/recordings 不会删除，已下载文件会自动复用。")

        from config import HAND_LANDMARKER_MODEL_PATH
        if not HAND_LANDMARKER_MODEL_PATH.exists():
            print("\nMediaPipe hand landmarker 模型不存在，自动下载...")
            run([sys.executable, "tools/download_models.py"])

        annotations = data_root / "annotations" / "fine-grained-annotations"
        if not all((annotations / name).exists() for name in ("train.csv", "validation.csv", "test.csv")):
            raise FileNotFoundError(
                f"Missing official Assembly101 fine-grained annotations under {annotations}."
            )

        _, selected_recordings = choose_baseline_recordings(
            annotations,
            args.labels,
            recordings,
            args.views,
            args.mirror,
        )
        selected_csv = ",".join(selected_recordings)

        # Download exactly the deterministic recording set chosen above. Existing files are reused.
        run([
            sys.executable,
            "tools/download_assembly101.py",
            "--root", str(data_root),
            "--source", "hf720",
            "--mirror", args.mirror,
            "--views", args.views,
            "--videos", selected_csv,
            "--skip-annotations",
        ])

        # Preprocess only this fixed set. Extra recordings already present in the local cache
        # must not silently change a repeat of the same baseline experiment.
        run([
            sys.executable,
            "tools/prepare_assembly101.py",
            "--video-root", str(data_root / "recordings"),
            "--annotation-dir", str(annotations),
            "--output", str(dataset),
            "--label-level", "verb",
            "--labels", args.labels,
            "--recordings", selected_csv,
            "--min-hand-ratio", str(args.min_hand_ratio),
            "--train-limit", str(train_limit),
            "--validation-limit", str(validation_limit),
            "--test-limit", str(test_limit),
            "--seed", str(args.seed),
        ])

        ready, reason = dataset_meets_targets(
            dataset,
            classes,
            train_limit,
            validation_limit,
            test_limit,
            min_train_recordings,
            min_eval_recordings,
        )
        print_dataset_summary(dataset, classes)
        if not ready:
            raise RuntimeError(
                "数据准备完成，但没有达到 baseline 目标规模："
                f"{reason}。可以增加 --recordings，或降低每类样本数。"
            )
    else:
        print(f"\n{reason}，直接复用，不重新跑 MediaPipe。")
        print_dataset_summary(dataset, classes)

    run([
        sys.executable,
        "tools/train_assembly101.py",
        "--dataset", str(dataset),
        "--epochs", str(epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--label-smoothing", str(args.label_smoothing),
        "--early-stop-patience", str(args.early_stop_patience),
        "--grad-clip", str(args.grad_clip),
        "--seed", str(args.seed),
        "--model-out", str(model_out),
        "--metrics-out", str(metrics_out),
    ])

    print("\n" + "=" * 80)
    print("Baseline v1 完成")
    print(f"最佳模型: {model_out}")
    print(f"训练/测试指标: {metrics_out}")
    print("摄像头测试:")
    print(f'  {sys.executable} main.py --camera 0 --checkpoint "{model_out}"')
    print("=" * 80)


if __name__ == "__main__":
    main()
