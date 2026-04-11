"""
capacity_manager.py -- Proof of Capacity and Training Manager
=============================================================
Runs a Proof of Capacity (PoC), assigns a capacity class, trains a model
on a private local shard from PneumoniaMNIST, computes reliability metrics,
and returns values ready for FLCoordinator submission.
"""

import os
import time
import hashlib
import struct
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset, Subset

from eth_account import Account
from eth_account.messages import encode_defunct

SCALE = 10_000
NUM_CLASSES = 2


class DummyNet(nn.Module):
    """Small fixed model used only for PoC benchmarking."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(16 * 112 * 112, 10)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def compute_benchmark_hash(
    throughput: float,
    k_steps: int,
    batch_size: int,
    capacity_class: str,
) -> bytes:
    """Return SHA-256(throughput, k_steps, batch_size, capacity_class)."""
    h = hashlib.sha256()
    h.update(struct.pack("!d", throughput))
    h.update(struct.pack("!I", k_steps))
    h.update(struct.pack("!I", batch_size))
    h.update(capacity_class.encode("utf-8"))
    return h.digest()


def sign_benchmark_hash(benchmark_hash: bytes, private_key: str) -> bytes:
    """Sign benchmark hash with node private key (EIP-191)."""
    msg = encode_defunct(hexstr="0x" + benchmark_hash.hex())
    signed = Account.sign_message(msg, private_key=private_key)
    return bytes(signed.signature)


def proof_of_capacity(
    k_steps: int = 50,
    batch_size: int = 16,
    device: str = "cpu",
) -> Tuple[str, float, bytes]:
    """
    Run fixed K-step SGD benchmark and classify capacity.
    Weak: <100, Medium: <300, Strong: >=300 samples/sec.
    """
    model = DummyNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    inputs = torch.randn(batch_size, 3, 224, 224).to(device)
    targets = torch.randint(0, 10, (batch_size,)).to(device)

    model.train()

    for _ in range(5):
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()

    start = time.perf_counter()
    for _ in range(k_steps):
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - start

    throughput = (k_steps * batch_size) / elapsed

    if throughput < 100:
        capacity_class = "Weak"
    elif throughput < 300:
        capacity_class = "Medium"
    else:
        capacity_class = "Strong"

    benchmark_hash = compute_benchmark_hash(throughput, k_steps, batch_size, capacity_class)
    return capacity_class, throughput, benchmark_hash


def compute_ece(
    model: nn.Module,
    data_x: torch.Tensor,
    data_y: torch.Tensor,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error with equal-width binning."""
    model.eval()
    with torch.no_grad():
        logits = model(data_x)
        probs = F.softmax(logits, dim=1)

    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(data_y).float()

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=confidences.device)
    ece = 0.0
    n_total = float(data_x.size(0))

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)

        n_bin = in_bin.sum().item()
        if n_bin == 0:
            continue

        avg_conf = confidences[in_bin].mean().item()
        avg_acc = accuracies[in_bin].mean().item()
        ece += (n_bin / n_total) * abs(avg_conf - avg_acc)

    return ece


def assign_model(capacity_class: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Assign architecture by capacity class (ImageNet-pretrained) with task-specific output head."""
    if capacity_class == "Weak":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if capacity_class == "Medium":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if capacity_class == "Strong":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unknown capacity class: {capacity_class}")


def _default_npz_path() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "simulation", "data", "pneumoniamnist.npz")
    )


def load_pneumonia_npz(npz_path: str | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load local PneumoniaMNIST NPZ and return normalized training tensors."""
    path = npz_path or _default_npz_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pneumonia dataset not found: {path}")

    data = np.load(path)

    image_key_candidates = ["train_images", "train_imgs"]
    label_key_candidates = ["train_labels", "train_lbls", "train_targets"]

    image_key = next((k for k in image_key_candidates if k in data.files), None)
    label_key = next((k for k in label_key_candidates if k in data.files), None)

    if image_key is None or label_key is None:
        raise KeyError(f"Unexpected NPZ keys: {data.files}")

    images = data[image_key]
    labels = data[label_key]

    x = torch.from_numpy(images).float()
    if x.ndim == 3:
        x = x.unsqueeze(1)
    x = x / 255.0
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    # ImageNet normalization (matches experiment.py pretrained backbone preprocessing)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std

    y = torch.from_numpy(labels).long().view(-1)
    return x, y


def create_non_iid_splits(
    labels: torch.Tensor,
    num_hospitals: int = 3,
    ratios: List[Dict[int, float]] | None = None,
    seed: int = 42,
) -> List[List[int]]:
    """Create hospital index shards with label skew."""
    if ratios is None:
        if num_hospitals == 3:
            ratios = [
                {0: 0.2, 1: 0.8},
                {0: 0.5, 1: 0.5},
                {0: 0.8, 1: 0.2},
            ]
        else:
            ratios = [{0: 1.0, 1: 1.0} for _ in range(num_hospitals)]
    else:
        num_hospitals = len(ratios)
    rng = np.random.default_rng(seed)
    y = labels.cpu().numpy()

    cls_indices = {}
    for cls in range(NUM_CLASSES):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        cls_indices[cls] = idx.tolist()

    splits = [[] for _ in range(num_hospitals)]
    for cls in range(NUM_CLASSES):
        idx = cls_indices[cls]
        n = len(idx)
        total_ratio = sum(r.get(cls, 0.0) for r in ratios)
        if total_ratio == 0:
            continue
        start = 0
        for h in range(num_hospitals):
            frac = ratios[h].get(cls, 0.0) / total_ratio
            count = int(n * frac) if h < num_hospitals - 1 else n - start
            splits[h].extend(idx[start:start + count])
            start += count

    for h in range(num_hospitals):
        rng.shuffle(splits[h])

    return splits


def _train_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
) -> None:
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()


def mock_training_manager(
    device: str = "cpu",
    hospital_id: int = 0,
    num_hospitals: int = 3,
    epochs: int = 1,
    batch_size: int = 32,
    max_samples: int = 512,
    private_key: str | None = None,
    run_poc: bool = True,
    capacity_override: str | None = None,
) -> Dict:
    """
    Run PoC (optional), train on hospital-local real data shard, and return
    benchmark/model hashes + reliability metrics for smart contract submission.
    """
    device_t = torch.device(device)

    if run_poc:
        capacity_class, throughput, benchmark_hash = proof_of_capacity(
            k_steps=50,
            batch_size=16,
            device=device,
        )
    else:
        if not capacity_override:
            raise ValueError("capacity_override is required when run_poc=False")
        capacity_class = capacity_override
        throughput = 0.0
        benchmark_hash = b"\x00" * 32

    benchmark_signature = None
    if private_key is not None and run_poc:
        benchmark_signature = sign_benchmark_hash(benchmark_hash, private_key)

    model = assign_model(capacity_class).to(device_t)

    x_all, y_all = load_pneumonia_npz()
    splits = create_non_iid_splits(y_all, num_hospitals=num_hospitals)
    shard = splits[hospital_id % len(splits)]
    if max_samples > 0:
        shard = shard[:max_samples]

    subset = TensorDataset(x_all[shard], y_all[shard])
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    _train_on_loader(model, loader, device_t, epochs=epochs)

    eval_x = x_all[shard][: min(256, len(shard))].to(device_t)
    eval_y = y_all[shard][: min(256, len(shard))].to(device_t)

    ece_raw = compute_ece(model, eval_x, eval_y, n_bins=10)
    with torch.no_grad():
        probs = F.softmax(model(eval_x), dim=1)
        confidence_raw = probs.max(dim=1).values.mean().item()

    confidence = min(int(confidence_raw * SCALE), SCALE)
    ece = min(int(ece_raw * SCALE), SCALE)

    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    model_hash = "0x" + hashlib.sha256(buf.getvalue()).hexdigest()

    return {
        "capacity_class": capacity_class,
        "throughput": throughput,
        "benchmark_hash": benchmark_hash,
        "benchmark_signature": benchmark_signature,
        "model_hash": model_hash,
        "confidence": confidence,
        "ece": ece,
        "model": model,
    }


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = mock_training_manager(device=dev, hospital_id=0, run_poc=True)
    print(
        {
            "capacity_class": res["capacity_class"],
            "throughput": round(res["throughput"], 2),
            "model_hash": res["model_hash"][:18] + "...",
            "confidence": res["confidence"],
            "ece": res["ece"],
        }
    )
