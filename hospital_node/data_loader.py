"""
data_loader.py -- Data utilities for hospital nodes
===============================================
Default path uses local PneumoniaMNIST NPZ (real data). Synthetic generation
is retained only as optional fallback for quick debugging.
"""

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


def _default_npz_path() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "simulation", "data", "pneumoniamnist.npz")
    )


def load_real_data(npz_path: str | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load PneumoniaMNIST training split from local NPZ into tensors."""
    path = npz_path or _default_npz_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = np.load(path)
    img_key = "train_images" if "train_images" in data.files else "train_imgs"
    lbl_key = "train_labels" if "train_labels" in data.files else "train_lbls"

    x = torch.from_numpy(data[img_key]).float()
    if x.ndim == 3:
        x = x.unsqueeze(1)
    x = x / 255.0
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    x = (x - 0.5) / 0.5

    y = torch.from_numpy(data[lbl_key]).long().view(-1)
    return x, y


def generate_synthetic_data(
    num_samples: int = 1000,
    image_size: int = 28,
    num_classes: int = 2,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optional synthetic data generator for debugging."""
    rng = np.random.RandomState(seed)
    images = rng.randn(num_samples, 1, image_size, image_size).astype(np.float32)
    labels = rng.randint(0, num_classes, size=num_samples).astype(np.int64)

    x = torch.from_numpy(images)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    y = torch.from_numpy(labels)
    return x, y


def get_data_loaders(
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
    use_synthetic: bool = False,
    num_samples: int = 1000,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/validation loaders. Uses real data by default."""
    if use_synthetic:
        images, labels = generate_synthetic_data(num_samples=num_samples, seed=seed)
    else:
        images, labels = load_real_data()

    dataset = TensorDataset(images, labels)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
