from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from config import HIDDEN_SIZE, INPUT_SIZE, NUM_LAYERS, SEQUENCE_LENGTH
from src.model import HandActionLSTM


class Assembly101Dataset(Dataset):
    def __init__(self, root: Path, split: str, class_names: list[str], augment: bool = False) -> None:
        self.root = root
        self.augment = augment
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.samples: list[tuple[Path, int]] = []
        split_root = root / split
        for name in class_names:
            for p in sorted((split_root / name).glob("*.npy")):
                self.samples.append((p, self.class_to_idx[name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        x = np.load(path).astype(np.float32)
        if x.shape != (SEQUENCE_LENGTH, INPUT_SIZE):
            raise ValueError(f"Invalid sequence shape in {path}: {x.shape}")
        if self.augment:
            x = x.copy()
            # Pose scale jitter + small coordinate/velocity noise. Keep this mild so the
            # first baseline measures the representation/model rather than aggressive augmentation.
            x[:, :63] *= np.random.uniform(0.95, 1.05)
            x += np.random.normal(0.0, 0.003, size=x.shape).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)


def evaluate(model, loader, device, loss_fn) -> dict[str, object]:
    model.eval()
    total_loss = total = 0
    ys: list[int] = []
    ps: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * len(y)
            total += len(y)
            ps.extend(logits.argmax(1).cpu().numpy().tolist())
            ys.extend(y.cpu().numpy().tolist())
    return {
        "loss": total_loss / max(1, total),
        "accuracy": accuracy_score(ys, ps) if ys else 0.0,
        "macro_f1": f1_score(ys, ps, average="macro", zero_division=0) if ys else 0.0,
        "y_true": ys,
        "y_pred": ps,
    }


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train HandActionLSTM baseline on Assembly101 MediaPipe features.")
    ap.add_argument("--dataset", type=Path, default=ROOT / "dataset_assembly101")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.02)
    ap.add_argument("--early-stop-patience", type=int, default=12)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-out", type=Path, default=ROOT / "models" / "hand_action_assembly101_lstm.pth")
    ap.add_argument("--metrics-out", type=Path, default=ROOT / "outputs" / "assembly101_baseline_metrics.json")
    args = ap.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.early_stop_patience < 1:
        ap.error("epochs, batch-size and early-stop-patience must be >= 1")
    if args.lr <= 0 or args.weight_decay < 0 or args.grad_clip < 0:
        ap.error("lr must be > 0 and weight-decay/grad-clip must be >= 0")
    if not 0.0 <= args.label_smoothing < 1.0:
        ap.error("label-smoothing must be in [0, 1)")

    manifest_path = args.dataset / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}. Run tools/prepare_assembly101.py first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    class_names = manifest.get("class_names", [])
    if not class_names:
        raise RuntimeError("Assembly101 manifest contains no classes.")

    set_seed(args.seed)
    train_ds = Assembly101Dataset(args.dataset, "train", class_names, augment=True)
    val_ds = Assembly101Dataset(args.dataset, "validation", class_names, augment=False)
    test_ds = Assembly101Dataset(args.dataset, "test", class_names, augment=False)
    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError("Train/validation/test dataset must all be non-empty for the formal baseline.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = choose_device()
    model = HandActionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, len(class_names)).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=4, factor=0.5)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"device={device} params={param_count:,}")
    print(f"samples train={len(train_ds)} validation={len(val_ds)} test={len(test_ds)}")
    print(f"classes={class_names}")

    best_macro_f1 = -1.0
    best_acc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_total = 0
        train_true: list[int] = []
        train_pred: list[int] = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            train_loss_sum += loss.item() * len(y)
            train_total += len(y)
            train_true.extend(y.detach().cpu().numpy().tolist())
            train_pred.extend(logits.argmax(1).detach().cpu().numpy().tolist())

        train_loss = train_loss_sum / max(1, train_total)
        train_acc = accuracy_score(train_true, train_pred)
        train_macro_f1 = f1_score(train_true, train_pred, average="macro", zero_division=0)
        val = evaluate(model, val_loader, device, loss_fn)
        scheduler.step(float(val["loss"]))
        lr_now = float(opt.param_groups[0]["lr"])

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "train_macro_f1": float(train_macro_f1),
            "val_loss": float(val["loss"]),
            "val_accuracy": float(val["accuracy"]),
            "val_macro_f1": float(val["macro_f1"]),
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} lr={lr_now:.2e} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_macro_f1:.4f} "
            f"val_loss={float(val['loss']):.4f} val_acc={float(val['accuracy']):.4f} val_f1={float(val['macro_f1']):.4f}"
        )

        val_f1 = float(val["macro_f1"])
        val_acc = float(val["accuracy"])
        improved = val_f1 > best_macro_f1 + 1e-12 or (
            abs(val_f1 - best_macro_f1) <= 1e-12 and val_acc > best_acc
        )
        if improved:
            best_macro_f1 = val_f1
            best_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "input_size": INPUT_SIZE,
                "sequence_length": SEQUENCE_LENGTH,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dataset": "Assembly101-MediaPipe",
                "label_level": manifest.get("label_level", "verb"),
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_macro_f1,
                "best_val_accuracy": best_acc,
                "seed": args.seed,
                "training_config": {
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "label_smoothing": args.label_smoothing,
                    "early_stop_patience": args.early_stop_patience,
                    "grad_clip": args.grad_clip,
                },
            }, args.model_out)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch}: no validation Macro-F1 improvement for "
                    f"{args.early_stop_patience} epochs."
                )
                break

    ckpt = torch.load(args.model_out, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test = evaluate(model, test_loader, device, loss_fn)
    labels = list(range(len(class_names)))
    report = classification_report(
        test["y_true"],
        test["y_pred"],
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    matrix = confusion_matrix(test["y_true"], test["y_pred"], labels=labels).tolist()

    result = {
        "baseline": "assembly101_mediapipe_lstm_v1",
        "device": str(device),
        "params": param_count,
        "class_names": class_names,
        "dataset_manifest": manifest,
        "training": {
            "seed": args.seed,
            "requested_epochs": args.epochs,
            "completed_epochs": len(history),
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_macro_f1,
            "best_val_accuracy": best_acc,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "early_stop_patience": args.early_stop_patience,
            "grad_clip": args.grad_clip,
        },
        "test": {
            "loss": float(test["loss"]),
            "accuracy": float(test["accuracy"]),
            "macro_f1": float(test["macro_f1"]),
            "classification_report": report,
            "confusion_matrix": matrix,
        },
        "history": history,
        "model_path": str(args.model_out.resolve()),
    }
    args.metrics_out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(
        f"best_epoch={best_epoch} best_val_f1={best_macro_f1:.4f} best_val_acc={best_acc:.4f} "
        f"test_loss={float(test['loss']):.4f} test_acc={float(test['accuracy']):.4f} "
        f"test_f1={float(test['macro_f1']):.4f}"
    )
    print(classification_report(
        test["y_true"],
        test["y_pred"],
        labels=labels,
        target_names=class_names,
        zero_division=0,
    ))
    print(np.asarray(matrix, dtype=np.int64))
    print(f"model={args.model_out}")
    print(f"metrics={args.metrics_out}")


if __name__ == "__main__":
    main()
