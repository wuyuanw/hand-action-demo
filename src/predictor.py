from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from .model import HandActionLSTM

@dataclass
class PredictionResult:
    class_id: int
    class_name: str
    confidence: float
    probabilities: np.ndarray

class Predictor:
    def __init__(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}. Run python train.py")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.class_names = ckpt["class_names"]
        self.model = HandActionLSTM(
            input_size=ckpt["input_size"],
            hidden_size=ckpt["hidden_size"],
            num_layers=ckpt.get("num_layers", 2),
            num_classes=len(self.class_names),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def predict(self, sequence: np.ndarray) -> PredictionResult:
        x = torch.from_numpy(sequence.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        return PredictionResult(idx, self.class_names[idx], float(probs[idx]), probs)
