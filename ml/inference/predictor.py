from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class Prediction:
    label: int
    confidence: float
    probs: np.ndarray  # shape: (10,)


class DigitPredictor:
    """
    Loads a trained PyTorch MLP digit model (0-9) and performs inference on a single feature vector.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_classes: int = 10,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.num_classes = num_classes

    @staticmethod
    def load_from_checkpoint(
        model_ctor,
        checkpoint_path: str | Path,
        device: Optional[str] = None,
        num_classes: int = 10,
    ) -> "DigitPredictor":
        """
        model_ctor: callable that returns the model instance, e.g. lambda: MLPClassifier(...)
        checkpoint_path: path to .pt / .pth
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_ctor()

        state = torch.load(str(ckpt_path), map_location=dev)
        # supports either full dict {'model_state_dict': ...} or raw state_dict
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)

        return DigitPredictor(model=model, device=dev, num_classes=num_classes)

    @torch.inference_mode()
    def predict(self, features: np.ndarray) -> Prediction:
        """
        features: shape (F,) float32
        returns: label + confidence + full probs
        """
        if features.ndim != 1:
            raise ValueError(f"Expected 1D feature vector, got shape {features.shape}")

        x = torch.from_numpy(features.astype(np.float32)).unsqueeze(0).to(self.device)  # (1, F)
        logits = self.model(x)  # (1, 10)
        probs_t = torch.softmax(logits, dim=-1).squeeze(0)  # (10,)
        probs = probs_t.detach().cpu().numpy()

        label = int(np.argmax(probs))
        confidence = float(probs[label])
        return Prediction(label=label, confidence=confidence, probs=probs)