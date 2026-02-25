from typing import Tuple

import torch
from torch.utils.data import random_split


def train_test_split_dataset(
    dataset,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple:
    """
    Deterministic train/test split.

    Args:
        dataset: PyTorch Dataset
        test_ratio: proportion of dataset for testing
        seed: random seed for reproducibility
    """

    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    total_size = len(dataset)
    test_size = int(total_size * test_ratio)
    train_size = total_size - test_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=generator
    )

    return train_dataset, test_dataset