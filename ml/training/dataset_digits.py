from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DigitsDatasetConfig:
    csv_path: str
    label_column: str = "label"


class DigitsCSVDataset(Dataset):
    """
    Dataset for 0–9 digit landmark features.

    Expected CSV format:
        label, f0, f1, ..., f62

    - label: int (0–9)
    - 63 wrist-relative normalized features
    """

    def __init__(self, config: DigitsDatasetConfig):
        self.config = config

        csv_path = Path(config.csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        self.df = pd.read_csv(csv_path, header=None)

        # Assign proper column names
        num_columns = self.df.shape[1]

        if num_columns != 64:
            raise ValueError(f"Expected 64 columns (1 label + 63 features), got {num_columns}")

        self.df.columns = ["label"] + [f"f{i}" for i in range(63)]

        if config.label_column not in self.df.columns:
            raise ValueError(f"Missing label column: {config.label_column}")

        # Separate features and labels
        self.labels = self.df[config.label_column].values

        if not set(self.labels).issubset(set(range(10))):
            raise ValueError("Labels must be integers 0–9")
        
        self.features = self.df.drop(columns=[config.label_column]).values

        if self.features.shape[1] != 63:
            raise ValueError(
                f"Expected 63 features, got {self.features.shape[1]}"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y