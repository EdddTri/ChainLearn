"""
experiment.py -- Federated Ensemble Learning Experiment
=======================================================
Evaluates the capacity-aware federated ensemble against baselines
using real medical imaging data (PneumoniaMNIST).

Methods compared:
  1. Centralized     - ResNet-50 trained on all data
  2. Local-only      - Each hospital trains alone
  3. FedAvg          - All train ResNet-50, average params
  4. Equal-wt Ens.   - Our 3 models, equal weights (1/3)
  5. Ours            - Our 3 models, capacity-aware weights

Usage:
    cd simulation
    ..\\hospital_node\\venv\\Scripts\\Activate.ps1
    python experiment.py
"""

import os
import sys
import copy
import time
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision.models as models
import torchvision.transforms as transforms

import medmnist
from medmnist import PneumoniaMNIST

# Allow imports from hospital_node
sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "hospital_node"))
from capacity_manager import compute_ece

# ── Config ─────────────────────────────────────────────────────────────────
NUM_CLASSES   = 2         # PneumoniaMNIST: Normal (0), Pneumonia (1)
EPOCHS        = 5         # Per hospital, per round
BATCH_SIZE    = 32
LR            = 0.001
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SCALE         = 10_000

# Capacity multipliers (mirroring flattened Solidity constants)
CAP_MUL = {"Weak": 8_000, "Medium": 10_000, "Strong": 12_000}
MAX_WEIGHT = 15_000
PARTICIPATION_BONUS = 500
MAX_BONUS = 2_500


# ═══════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════
def load_pneumonia_mnist():
    """Load PneumoniaMNIST and return (train_images, train_labels, test_images, test_labels)."""
    print("[1/5] Loading PneumoniaMNIST dataset ...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_root, exist_ok=True)

    train_ds = PneumoniaMNIST(split="train", download=True, transform=transform, root=data_root)
    test_ds  = PneumoniaMNIST(split="test",  download=True, transform=transform, root=data_root)

    # Preload into tensors for speed
    print(f"       Training samples: {len(train_ds)}")
    print(f"       Test samples:     {len(test_ds)}")

    return train_ds, test_ds


# ═══════════════════════════════════════════════════════════════════════════
#  Non-IID Split
# ═══════════════════════════════════════════════════════════════════════════
def create_non_iid_split(dataset, ratios):
    """
    Split dataset into 3 shards with label skew.

    ratios: list of 3 dicts, e.g.:
        [
          {0: 0.2, 1: 0.8},  # Hospital A: 20% Normal, 80% Pneumonia
          {0: 0.5, 1: 0.5},  # Hospital B: balanced
          {0: 0.8, 1: 0.2},  # Hospital C: 80% Normal, 20% Pneumonia
        ]
    """
    print("[2/5] Creating non-IID splits ...")

    # Get all labels
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if hasattr(label, "item"):
            labels.append(label.item())
        elif isinstance(label, (list, tuple, np.ndarray)) and len(label) == 1:
            labels.append(int(label[0]))
        else:
            labels.append(int(label))
    labels = np.array(labels)

    # Group indices by label
    class_indices = {}
    for cls in range(NUM_CLASSES):
        class_indices[cls] = np.where(labels == cls)[0].tolist()
        np.random.shuffle(class_indices[cls])

    # Allocate to hospitals
    hospital_indices = [[] for _ in range(3)]
    for cls in range(NUM_CLASSES):
        indices = class_indices[cls]
        n = len(indices)
        # Compute proportional allocation
        total_ratio = sum(ratios[h].get(cls, 0) for h in range(3))
        if total_ratio == 0:
            continue
        start = 0
        for h in range(3):
            frac = ratios[h].get(cls, 0) / total_ratio
            count = int(n * frac) if h < 2 else n - start
            hospital_indices[h].extend(indices[start:start + count])
            start += count

    for h in range(3):
        np.random.shuffle(hospital_indices[h])
        label_dist = Counter([labels[i] for i in hospital_indices[h]])
        total = len(hospital_indices[h])
        dist_str = ", ".join(f"class{c}: {cnt} ({cnt/total*100:.0f}%)" for c, cnt in sorted(label_dist.items()))
        print(f"       Hospital {h}: {total} samples [{dist_str}]")

    return [Subset(dataset, idx) for idx in hospital_indices]


# ═══════════════════════════════════════════════════════════════════════════
#  Model Factory
# ═══════════════════════════════════════════════════════════════════════════
def get_model(capacity_class):
    """Return appropriate architecture for the capacity class."""
    if capacity_class == "Weak":
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, NUM_CLASSES)
    elif capacity_class == "Medium":
        m = models.efficientnet_b0(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, NUM_CLASSES)
    else:  # Strong
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m.to(DEVICE)


# ═══════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════
def train_model(model, dataloader, epochs=EPOCHS):
    """Train a model and return it."""
    model.train()
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).squeeze().long()
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_model(model, dataloader):
    """Return (accuracy, f1_score, ece)."""
    from sklearn.metrics import f1_score as sklearn_f1

    model.eval()
    all_preds = []
    all_labels = []
    all_confs = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).squeeze().long()
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confs = np.array(all_confs)

    accuracy = (all_preds == all_labels).mean()
    f1 = sklearn_f1(all_labels, all_preds, average="macro")

    # Binned ECE
    ece = 0.0
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (all_confs >= lo) & (all_confs < hi) if i < n_bins - 1 else (all_confs >= lo) & (all_confs <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_conf = all_confs[mask].mean()
        avg_acc = (all_preds[mask] == all_labels[mask]).mean()
        ece += (n_bin / len(all_confs)) * abs(avg_conf - avg_acc)

    return accuracy, f1, ece


def evaluate_ensemble(models_list, weights, dataloader):
    """Evaluate weighted ensemble: y_final = sum(w_i * y_i(x))."""
    from sklearn.metrics import f1_score as sklearn_f1

    total_w = sum(weights)
    normed = [w / total_w for w in weights]

    all_preds = []
    all_labels = []
    all_confs = []

    for x, y in dataloader:
        x = x.to(DEVICE)
        y = y.squeeze().long()

        ensemble_probs = torch.zeros(x.size(0), NUM_CLASSES)
        for model, w in zip(models_list, normed):
            model.eval()
            with torch.no_grad():
                probs = F.softmax(model(x), dim=1).cpu()
                ensemble_probs += w * probs

        confs, preds = ensemble_probs.max(dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(y.numpy())
        all_confs.extend(confs.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confs = np.array(all_confs)

    accuracy = (all_preds == all_labels).mean()
    f1 = sklearn_f1(all_labels, all_preds, average="macro")

    # ECE
    ece = 0.0
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (all_confs >= lo) & (all_confs < hi) if i < n_bins - 1 else (all_confs >= lo) & (all_confs <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_conf = all_confs[mask].mean()
        avg_acc = (all_preds[mask] == all_labels[mask]).mean()
        ece += (n_bin / len(all_confs)) * abs(avg_conf - avg_acc)

    return accuracy, f1, ece


# ═══════════════════════════════════════════════════════════════════════════
#  Weight Calculation (mirrors Solidity)
# ═══════════════════════════════════════════════════════════════════════════
def compute_weight(capacity_class, confidence, ece, rounds_participated=1):
    """Mirror the Solidity calculateWeightPure formula."""
    cap_mul = CAP_MUL[capacity_class]
    base = (cap_mul * confidence * (SCALE - ece)) // (SCALE * SCALE)
    bonus = min(rounds_participated * PARTICIPATION_BONUS, MAX_BONUS)
    return min(base + bonus, MAX_WEIGHT)


# ═══════════════════════════════════════════════════════════════════════════
#  FedAvg
# ═══════════════════════════════════════════════════════════════════════════
def fedavg(models_list):
    """Average model parameters (all models must have the same architecture)."""
    avg_state = copy.deepcopy(models_list[0].state_dict())
    n = len(models_list)
    for key in avg_state:
        avg_state[key] = avg_state[key].float()
        for i in range(1, n):
            avg_state[key] += models_list[i].state_dict()[key].float()
        avg_state[key] /= n

    global_model = copy.deepcopy(models_list[0])
    global_model.load_state_dict(avg_state)
    return global_model


# ═══════════════════════════════════════════════════════════════════════════
#  Main Experiment
# ═══════════════════════════════════════════════════════════════════════════
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Load data
    train_ds, test_ds = load_pneumonia_mnist()
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Non-IID split
    split_ratios = [
        {0: 0.2, 1: 0.8},   # Hospital A (Weak):   mostly Pneumonia
        {0: 0.5, 1: 0.5},   # Hospital B (Medium): balanced
        {0: 0.8, 1: 0.2},   # Hospital C (Strong): mostly Normal
    ]
    hospital_datasets = create_non_iid_split(train_ds, split_ratios)
    hospital_loaders = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in hospital_datasets]

    capacity_classes = ["Weak", "Medium", "Strong"]
    arch_names = ["MobileNetV3", "EfficientNet-B0", "ResNet-50"]

    # ── Results collector ─────────────────────────────────────────────────
    results = {}

    # ═══════════════════════════════════════════════════════════════════════
    # Baseline 1: CENTRALIZED
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[3/5] Training models ...")
    print("\n  --- Centralized (ResNet-50 on ALL data) ---")
    central_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    central_model = get_model("Strong")
    t0 = time.time()
    train_model(central_model, central_loader, epochs=EPOCHS)
    t_central = time.time() - t0
    acc, f1, ece = evaluate_model(central_model, test_loader)
    results["Centralized"] = {"acc": acc, "f1": f1, "ece": ece, "time": t_central}
    print(f"       Acc={acc:.4f}  F1={f1:.4f}  ECE={ece:.4f}  [{t_central:.1f}s]")

    # ═══════════════════════════════════════════════════════════════════════
    # Baseline 2: LOCAL-ONLY (each hospital trains alone)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n  --- Local-Only (each hospital independently) ---")
    local_models = []
    for h in range(3):
        cap = capacity_classes[h]
        model = get_model(cap)
        t0 = time.time()
        train_model(model, hospital_loaders[h], epochs=EPOCHS)
        t_h = time.time() - t0
        acc, f1, ece = evaluate_model(model, test_loader)
        results[f"Local-{cap}"] = {"acc": acc, "f1": f1, "ece": ece, "time": t_h}
        print(f"       {cap:8s} ({arch_names[h]:16s}): Acc={acc:.4f}  F1={f1:.4f}  ECE={ece:.4f}  [{t_h:.1f}s]")
        local_models.append(model)

    # ═══════════════════════════════════════════════════════════════════════
    # Baseline 3: FedAvg (all train ResNet-50, average params)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n  --- FedAvg (ResNet-50, averaged params) ---")
    fedavg_models = []
    t0_fed = time.time()
    for h in range(3):
        model = get_model("Strong")  # all same arch for FedAvg
        train_model(model, hospital_loaders[h], epochs=EPOCHS)
        fedavg_models.append(model)
    global_model = fedavg(fedavg_models)
    t_fed = time.time() - t0_fed
    acc, f1, ece = evaluate_model(global_model, test_loader)
    results["FedAvg"] = {"acc": acc, "f1": f1, "ece": ece, "time": t_fed}
    print(f"       Acc={acc:.4f}  F1={f1:.4f}  ECE={ece:.4f}  [{t_fed:.1f}s]")

    # ═══════════════════════════════════════════════════════════════════════
    # Baseline 4: Equal-Weight Ensemble
    # ═══════════════════════════════════════════════════════════════════════
    print("\n  --- Equal-Weight Ensemble (w=1/3 each) ---")
    eq_weights = [1.0/3, 1.0/3, 1.0/3]
    acc, f1, ece = evaluate_ensemble(local_models, eq_weights, test_loader)
    results["EqualWt-Ens"] = {"acc": acc, "f1": f1, "ece": ece, "time": 0}
    print(f"       Acc={acc:.4f}  F1={f1:.4f}  ECE={ece:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # OUR METHOD: Capacity-Aware Weighted Ensemble
    # ═══════════════════════════════════════════════════════════════════════
    print("\n  --- Ours: Capacity-Aware Ensemble (on-chain weights) ---")

    # Compute confidence + ECE for each local model on test set
    our_weights = []
    for h in range(3):
        cap = capacity_classes[h]
        m = local_models[h]

        # Get confidence from test predictions
        m.eval()
        confs = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(DEVICE)
                probs = F.softmax(m(x), dim=1)
                confs.extend(probs.max(dim=1).values.cpu().numpy())
        avg_conf = int(np.mean(confs) * SCALE)

        # Get ECE as integer
        _, _, ece_raw = evaluate_model(m, test_loader)
        ece_int = int(ece_raw * SCALE)

        w = compute_weight(cap, avg_conf, ece_int, rounds_participated=1)
        our_weights.append(w)
        print(f"       {cap:8s}: conf={avg_conf:>5} ece={ece_int:>5} w_i={w:>6}")

    acc, f1, ece = evaluate_ensemble(local_models, [float(w) for w in our_weights], test_loader)
    results["Ours"] = {"acc": acc, "f1": f1, "ece": ece, "time": 0}
    print(f"       Ensemble: Acc={acc:.4f}  F1={f1:.4f}  ECE={ece:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Results Table
    # ═══════════════════════════════════════════════════════════════════════
    print("\n\n[4/5] Results Summary")
    print("=" * 68)
    print(f"  {'Method':<22} {'Accuracy':>10} {'F1-Score':>10} {'ECE':>10} {'Time':>10}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    order = ["Centralized", "Ours", "EqualWt-Ens", "FedAvg",
             "Local-Weak", "Local-Medium", "Local-Strong"]

    for name in order:
        r = results[name]
        t_str = f"{r['time']:.1f}s" if r['time'] > 0 else "-"
        print(f"  {name:<22} {r['acc']:>10.4f} {r['f1']:>10.4f} {r['ece']:>10.4f} {t_str:>10}")

    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    # Highlight improvement
    ours_acc = results["Ours"]["acc"]
    best_local = max(results[f"Local-{c}"]["acc"] for c in capacity_classes)
    fedavg_acc = results["FedAvg"]["acc"]

    print(f"\n[5/5] Key Findings")
    print(f"  Ours vs Best Local:  {(ours_acc - best_local)*100:+.2f}%")
    print(f"  Ours vs FedAvg:      {(ours_acc - fedavg_acc)*100:+.2f}%")
    print(f"  Ours vs Equal-Wt:    {(ours_acc - results['EqualWt-Ens']['acc'])*100:+.2f}%")

    # Weight fairness
    print(f"\n  Weight Distribution (Weak : Medium : Strong):")
    if our_weights[2] > 0:
        print(f"    {our_weights[0]:>6} : {our_weights[1]:>6} : {our_weights[2]:>6}")
        print(f"    Weak/Strong ratio: {our_weights[0]/our_weights[2]:.2f} (closer to 1.0 = fairer)")

    print(f"\n{'=' * 68}")
    print("[DONE] Experiment complete.")


if __name__ == "__main__":
    main()
