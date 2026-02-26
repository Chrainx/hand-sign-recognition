from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Subset


def stratified_split(
    dataset,
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[Subset, Subset]:
    """
    Stratified train/test split for classification datasets.

    Ensures each class maintains the same proportion in train and test sets.

    Args:
        dataset: Dataset with `labels` attribute
        test_ratio: proportion of dataset for testing
        seed: random seed for reproducibility
    """

    if not hasattr(dataset, "labels"):
        raise AttributeError("Dataset must have `labels` attribute for stratified split")

    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")

    labels = np.array(dataset.labels)
    unique_classes = np.unique(labels)

    train_indices = []
    test_indices = []

    rng = np.random.default_rng(seed)

    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)

        test_size = int(len(cls_indices) * test_ratio)
        test_cls_indices = cls_indices[:test_size]
        train_cls_indices = cls_indices[test_size:]

        train_indices.extend(train_cls_indices)
        test_indices.extend(test_cls_indices)

    # Shuffle final indices to avoid ordered classes
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset