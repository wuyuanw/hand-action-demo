from __future__ import annotations
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

from config import HIDDEN_SIZE, INPUT_SIZE, NUM_LAYERS, OUTPUT_DIR, SEQUENCE_LENGTH
from src.dhg2016 import DHG14_CLASS_NAMES, subject_id_from_name
from src.model import HandActionLSTM


class DHGDataset(Dataset):
    def __init__(self, root: Path, subjects: set[int], augment: bool = False) -> None:
        self.samples: list[tuple[Path, int]] = []
        self.augment = augment
        for label, name in enumerate(DHG14_CLASS_NAMES):
            for p in sorted((root / name).glob("*.npy")):
                if subject_id_from_name(p) in subjects:
                    self.samples.append((p, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        x = np.load(path).astype(np.float32)
        if x.shape != (SEQUENCE_LENGTH, INPUT_SIZE):
            raise ValueError(f"Invalid sequence shape in {path}: {x.shape}")
        if self.augment:
            x = x.copy()
            x[:, :63] *= np.random.uniform(0.95, 1.05)
            x += np.random.normal(0.0, 0.003, size=x.shape).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the LSTM baseline on DHG-2016/DHG-14.")
    ap.add_argument("--dataset", type=Path, default=ROOT / "dataset_dhg2016")
    ap.add_argument("--test-subject", type=int, default=20, choices=range(1, 21))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model-out", type=Path, default=ROOT / "models" / "hand_action_dhg14_lstm.pth")
    args = ap.parse_args()

    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    train_subjects = set(range(1, 21)) - {args.test_subject}
    test_subjects = {args.test_subject}
    train_ds = DHGDataset(args.dataset, train_subjects, augment=True)
    test_ds = DHGDataset(args.dataset, test_subjects, augment=False)
    if not train_ds or not test_ds:
        raise RuntimeError("DHG dataset is empty. Run tools/prepare_dhg2016.py first.")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    model = HandActionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, len(DHG14_CLASS_NAMES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    best = -1.0
    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train(); tl = tc = tn = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); logits = model(x); loss = loss_fn(logits, y); loss.backward(); opt.step()
            tl += loss.item() * len(y); tc += (logits.argmax(1) == y).sum().item(); tn += len(y)

        model.eval(); vl = vc = vn = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x); loss = loss_fn(logits, y)
                vl += loss.item() * len(y); vc += (logits.argmax(1) == y).sum().item(); vn += len(y)

        tr_loss, tr_acc = tl / tn, tc / tn
        te_loss, te_acc = vl / vn, vc / vn
        scheduler.step(te_loss)
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss); history["test_acc"].append(te_acc)
        print(f"Epoch {epoch:02d} train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} test_loss={te_loss:.4f} test_acc={te_acc:.3f}")

        if te_acc > best:
            best = te_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": DHG14_CLASS_NAMES,
                "input_size": INPUT_SIZE,
                "sequence_length": SEQUENCE_LENGTH,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dataset": "DHG-2016/DHG-14",
                "test_subject": args.test_subject,
            }, args.model_out)

    plt.figure(); plt.plot(history["train_loss"], label="train loss"); plt.plot(history["test_loss"], label="test loss")
    plt.legend(); plt.tight_layout(); plt.savefig(OUTPUT_DIR / "dhg2016_training_curve.png"); plt.close()

    ckpt = torch.load(args.model_out, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in test_loader:
            p = model(x.to(device)).argmax(1).cpu().numpy(); ps.extend(p); ys.extend(y.numpy())
    print(classification_report(ys, ps, labels=list(range(len(DHG14_CLASS_NAMES))), target_names=DHG14_CLASS_NAMES, zero_division=0))
    print(confusion_matrix(ys, ps, labels=list(range(len(DHG14_CLASS_NAMES)))))
    print(f"best_test_acc={best:.4f} test_subject={args.test_subject} model={args.model_out}")


if __name__ == "__main__":
    main()
