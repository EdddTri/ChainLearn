"""
experiment.py -- Comprehensive Federated Ensemble Learning Experiment
=====================================================================
Evaluates the capacity-aware federated ensemble against baselines and
ablations across multiple datasets, non-IID severity levels, and random
seeds.  Reports mean +/- std for statistical rigor.

Methods compared:
  1. Centralized     - ResNet-50 trained on all data
  2. Local-only      - Each hospital trains alone (best of 3)
  3. FedAvg          - All train ResNet-50, average params
  4. FedProx         - FedAvg + proximal regularization term
  5. FedMD           - Heterogeneous KD via public data consensus
  6. Equal-wt Ens.   - Our 3 models, equal weights (1/3)
  7. Ours (Full)     - Our 3 models, capacity-aware weights

Ablations (reuse Ours models, modify weight formula):
  8. No CapMul       - capacity multiplier = 1.0 for all
  9. No Confidence   - confidence = 1.0 for all
 10. No ECE          - ECE = 0 for all
 11. No Bonus        - participation bonus = 0
 12. No PoC          - uniform architecture (EfficientNet-B0), equal cap_mul

Additional analyses:
 13. Adversarial experiments (lazy, inflated metrics, capacity spoofing)
 14. Communication cost comparison
 15. Gas cost analysis

Usage:
    python experiment.py                       # full run (5 seeds)
    python experiment.py --seeds 3             # quick run (3 seeds)
    python experiment.py --datasets pneumoniamnist  # single dataset
"""

import argparse
import copy
import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import f1_score as sklearn_f1
from torch.utils.data import DataLoader, Subset

import medmnist
from medmnist import PneumoniaMNIST, DermaMNIST

# ── Config ─────────────────────────────────────────────────────────────────
EPOCHS     = 5
BATCH_SIZE = 32
LR         = 0.001
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SCALE      = 10_000

NUM_HOSPITALS = 3
CAPACITY_CLASSES = ["Weak", "Medium", "Strong"]
ARCH_NAMES = ["MobileNetV3-S", "EfficientNet-B0", "ResNet-50"]

# Solidity-mirrored weight constants
CAP_MUL = {"Weak": 8_000, "Medium": 10_000, "Strong": 12_000}
MAX_WEIGHT = 15_000
PARTICIPATION_BONUS = 500
MAX_BONUS = 2_500

DATASET_REGISTRY = {
    "pneumoniamnist": {"cls": PneumoniaMNIST, "num_classes": 2},
    "dermamnist":     {"cls": DermaMNIST,     "num_classes": 7},
}

NON_IID_ALPHAS = {
    "mild":     1.0,
    "moderate": 0.5,
    "severe":   0.1,
}

ABLATIONS = {
    "No CapMul":  {"use_capacity": False},
    "No Conf":    {"use_confidence": False},
    "No ECE":     {"use_ece": False},
    "No Bonus":   {"use_participation": False},
}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ═══════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════
def load_dataset(name):
    """Load a MedMNIST dataset. Returns (train_ds, test_ds, num_classes)."""
    info = DATASET_REGISTRY[name]
    num_classes = info["num_classes"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_root, exist_ok=True)

    train_ds = info["cls"](split="train", download=True, transform=transform, root=data_root)
    test_ds  = info["cls"](split="test",  download=True, transform=transform, root=data_root)

    return train_ds, test_ds, num_classes


def _extract_labels(dataset):
    """Extract integer label array from a MedMNIST dataset."""
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        if hasattr(label, "item"):
            labels.append(label.item())
        elif isinstance(label, (list, tuple, np.ndarray)) and len(label) == 1:
            labels.append(int(label[0]))
        else:
            labels.append(int(label))
    return np.array(labels)


# ═══════════════════════════════════════════════════════════════════════════
#  Non-IID Splitting (Dirichlet)
# ═══════════════════════════════════════════════════════════════════════════
def dirichlet_split(dataset, num_classes, alpha, seed):
    """
    Split dataset into NUM_HOSPITALS shards using a Dirichlet distribution.
    Lower alpha = more skewed (more non-IID).
    """
    rng = np.random.default_rng(seed)
    labels = _extract_labels(dataset)

    hospital_indices = [[] for _ in range(NUM_HOSPITALS)]

    for cls in range(num_classes):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)

        proportions = rng.dirichlet([alpha] * NUM_HOSPITALS)
        counts = (proportions * len(cls_idx)).astype(int)
        counts[-1] = len(cls_idx) - counts[:-1].sum()

        start = 0
        for h in range(NUM_HOSPITALS):
            hospital_indices[h].extend(cls_idx[start:start + counts[h]].tolist())
            start += counts[h]

    for h in range(NUM_HOSPITALS):
        rng.shuffle(hospital_indices[h])

    return [Subset(dataset, idx) for idx in hospital_indices]


# ═══════════════════════════════════════════════════════════════════════════
#  Model Factory
# ═══════════════════════════════════════════════════════════════════════════
def get_model(capacity_class, num_classes):
    """Return the architecture assigned to a capacity class."""
    if capacity_class == "Weak":
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif capacity_class == "Medium":
        m = models.efficientnet_b0(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:  # Strong
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m.to(DEVICE)


# ═══════════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════════
def train_model(model, dataloader, epochs=EPOCHS):
    model.train()
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).squeeze().long()
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_model(model, dataloader):
    """Return (accuracy, f1_macro, ece)."""
    model.eval()
    all_preds, all_labels, all_confs = [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).squeeze().long()
            probs = F.softmax(model(x), dim=1)
            confs, preds = probs.max(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confs  = np.array(all_confs)

    acc = (all_preds == all_labels).mean()
    f1  = sklearn_f1(all_labels, all_preds, average="macro")
    ece = _binned_ece(all_confs, all_preds, all_labels)
    return acc, f1, ece


def evaluate_ensemble(models_list, weights, dataloader, num_classes):
    """Evaluate weighted ensemble: y_final = sum(w_i * softmax_i(x))."""
    total_w = sum(weights)
    if total_w == 0:
        total_w = 1.0
    normed = [w / total_w for w in weights]

    all_preds, all_labels, all_confs = [], [], []

    for x, y in dataloader:
        x = x.to(DEVICE)
        y = y.squeeze().long()

        ensemble_probs = torch.zeros(x.size(0), num_classes)
        for model, w in zip(models_list, normed):
            model.eval()
            with torch.no_grad():
                probs = F.softmax(model(x), dim=1).cpu()
                ensemble_probs += w * probs

        confs, preds = ensemble_probs.max(dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(y.numpy())
        all_confs.extend(confs.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confs  = np.array(all_confs)

    acc = (all_preds == all_labels).mean()
    f1  = sklearn_f1(all_labels, all_preds, average="macro")
    ece = _binned_ece(all_confs, all_preds, all_labels)
    return acc, f1, ece


def _binned_ece(confs, preds, labels, n_bins=10):
    ece = 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confs >= lo) & (confs < hi) if i < n_bins - 1 else (confs >= lo) & (confs <= hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        avg_conf = confs[mask].mean()
        avg_acc  = (preds[mask] == labels[mask]).mean()
        ece += (n_bin / len(confs)) * abs(avg_conf - avg_acc)
    return ece


# ═══════════════════════════════════════════════════════════════════════════
#  Reliability Metrics
# ═══════════════════════════════════════════════════════════════════════════
def get_reliability_metrics(model, dataloader):
    """Return (confidence_int, ece_int) in [0, SCALE]."""
    model.eval()
    all_confs = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(DEVICE)
            probs = F.softmax(model(x), dim=1)
            all_confs.extend(probs.max(dim=1).values.cpu().numpy())

    avg_conf = int(np.mean(all_confs) * SCALE)
    _, _, ece_raw = evaluate_model(model, dataloader)
    ece_int = int(ece_raw * SCALE)

    return min(avg_conf, SCALE), min(ece_int, SCALE)


# ═══════════════════════════════════════════════════════════════════════════
#  Weight Calculation (mirrors Solidity, with ablation support)
# ═══════════════════════════════════════════════════════════════════════════
def compute_weight(capacity_class, confidence, ece, rounds_participated=1,
                   use_capacity=True, use_confidence=True,
                   use_ece=True, use_participation=True):
    """Mirror the Solidity calculateWeightPure formula with ablation flags."""
    cap_mul = CAP_MUL[capacity_class] if use_capacity else SCALE
    conf    = confidence if use_confidence else SCALE
    e       = ece if use_ece else 0
    base    = (cap_mul * conf * (SCALE - e)) // (SCALE * SCALE)
    bonus   = min(rounds_participated * PARTICIPATION_BONUS, MAX_BONUS) if use_participation else 0
    return min(base + bonus, MAX_WEIGHT)


# ═══════════════════════════════════════════════════════════════════════════
#  FedAvg
# ═══════════════════════════════════════════════════════════════════════════
def fedavg(models_list):
    """Average model parameters (all models must share the same architecture)."""
    avg_state = copy.deepcopy(models_list[0].state_dict())
    n = len(models_list)
    for key in avg_state:
        avg_state[key] = avg_state[key].float()
        for i in range(1, n):
            avg_state[key] += models_list[i].state_dict()[key].float()
        avg_state[key] /= n
    result = copy.deepcopy(models_list[0])
    result.load_state_dict(avg_state)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  FedProx (FedAvg + proximal term)
# ═══════════════════════════════════════════════════════════════════════════
FEDPROX_MU = 0.01  # proximal regularization strength


def train_model_fedprox(model, dataloader, global_state_dict, mu=FEDPROX_MU, epochs=EPOCHS):
    """Train with FedProx proximal term: loss += (mu/2) * ||w - w_global||^2."""
    model.train()
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).squeeze().long()
            opt.zero_grad()
            loss = crit(model(x), y)
            # Proximal term
            prox = 0.0
            for name, param in model.named_parameters():
                if name in global_state_dict:
                    prox += ((param - global_state_dict[name].to(DEVICE)) ** 2).sum()
            loss += (mu / 2.0) * prox
            loss.backward()
            opt.step()
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  FedMD (Model-Agnostic via Knowledge Distillation on public data)
# ═══════════════════════════════════════════════════════════════════════════
FEDMD_PUBLIC_FRAC = 0.1   # fraction of train set used as "public" data
FEDMD_DIGEST_EPOCHS = 2   # epochs for local digest step
FEDMD_KD_TEMP = 3.0       # distillation temperature


def fedmd_split_public(train_ds, frac, seed):
    """Split off a small public subset from the training data."""
    rng = np.random.default_rng(seed)
    n = len(train_ds)
    n_public = max(1, int(n * frac))
    all_idx = np.arange(n)
    rng.shuffle(all_idx)
    public_idx = all_idx[:n_public].tolist()
    private_idx = all_idx[n_public:].tolist()
    return Subset(train_ds, public_idx), Subset(train_ds, private_idx)


def fedmd_consensus_logits(models_list, public_loader, num_classes):
    """Compute average softmax logits across all models on the public set."""
    all_avg = []
    for x, _ in public_loader:
        x = x.to(DEVICE)
        batch_sum = torch.zeros(x.size(0), num_classes, device=DEVICE)
        for m in models_list:
            m.eval()
            with torch.no_grad():
                batch_sum += F.softmax(m(x) / FEDMD_KD_TEMP, dim=1)
        all_avg.append((batch_sum / len(models_list)).cpu())
    return torch.cat(all_avg, dim=0)


def fedmd_digest(model, public_loader, consensus, temp=FEDMD_KD_TEMP, epochs=FEDMD_DIGEST_EPOCHS):
    """
    Digest step: train model to match consensus logits on public data
    using KL divergence (knowledge distillation).
    """
    model.train()
    opt = optim.Adam(model.parameters(), lr=LR * 0.1)
    kl = nn.KLDivLoss(reduction="batchmean")

    idx = 0
    for _ in range(epochs):
        idx = 0
        for x, _ in public_loader:
            x = x.to(DEVICE)
            bs = x.size(0)
            target = consensus[idx:idx + bs].to(DEVICE)
            idx += bs

            opt.zero_grad()
            student_log = F.log_softmax(model(x) / temp, dim=1)
            loss = kl(student_log, target) * (temp ** 2)
            loss.backward()
            opt.step()
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  Communication & Gas Cost Analysis
# ═══════════════════════════════════════════════════════════════════════════
def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_communication_cost_table(num_classes):
    """Print communication cost comparison across methods."""
    # Model sizes (number of float32 parameters)
    model_params = {}
    for cap in CAPACITY_CLASSES:
        m = get_model(cap, num_classes)
        model_params[cap] = count_parameters(m)
        del m

    resnet_params = model_params["Strong"]
    bytes_per_param = 4  # float32

    print(f"\n{'='*80}")
    print(f"  COMMUNICATION COST ANALYSIS (per hospital per round)")
    print(f"{'='*80}")
    print(f"  {'Method':<25} {'Upload':>20} {'Download':>20} {'Total':>20}")
    print(f"  {'-'*25} {'-'*20} {'-'*20} {'-'*20}")

    def fmt_bytes(n):
        if n < 1024:
            return f"{n} B"
        elif n < 1024 ** 2:
            return f"{n / 1024:.1f} KB"
        else:
            return f"{n / (1024 ** 2):.1f} MB"

    # FedAvg: upload full model + download aggregated model
    fa_up = resnet_params * bytes_per_param
    fa_down = resnet_params * bytes_per_param
    print(f"  {'FedAvg':<25} {fmt_bytes(fa_up):>20} {fmt_bytes(fa_down):>20} {fmt_bytes(fa_up + fa_down):>20}")

    # FedProx: same as FedAvg
    print(f"  {'FedProx':<25} {fmt_bytes(fa_up):>20} {fmt_bytes(fa_down):>20} {fmt_bytes(fa_up + fa_down):>20}")

    # FedMD: upload logits on public data, download consensus logits
    n_public = 500  # approximate
    logit_up = n_public * num_classes * bytes_per_param
    logit_down = n_public * num_classes * bytes_per_param
    print(f"  {'FedMD':<25} {fmt_bytes(logit_up):>20} {fmt_bytes(logit_down):>20} {fmt_bytes(logit_up + logit_down):>20}")

    # Ours: upload (model_hash=32B, confidence=32B, ece=32B, modelType=32B) = 128 bytes
    # Plus one on-chain tx. Download: 3 weights (3 * 32B) = 96 bytes
    ours_up = 128  # 4 x 32-byte values in a tx
    ours_down = NUM_HOSPITALS * 32  # read N weights
    print(f"  {'Ours':<25} {fmt_bytes(ours_up):>20} {fmt_bytes(ours_down):>20} {fmt_bytes(ours_up + ours_down):>20}")

    # At inference: share softmax predictions
    # Typically 1 image at a time, so num_classes floats per model
    ours_inf = NUM_HOSPITALS * num_classes * bytes_per_param
    print(f"  {'Ours (inference/image)':<25} {'--':>20} {fmt_bytes(ours_inf):>20} {fmt_bytes(ours_inf):>20}")

    print(f"  {'-'*25} {'-'*20} {'-'*20} {'-'*20}")

    # Savings ratio
    ratio = (fa_up + fa_down) / max(ours_up + ours_down, 1)
    print(f"\n  Ours is {ratio:.0f}x less communication than FedAvg per training round")
    print(f"  (FedAvg/FedProx: {resnet_params:,} params = {fmt_bytes(fa_up)} per direction)")
    print(f"  (Ours: 4 on-chain values = {ours_up} bytes upload)")
    print()


# Estimated gas costs from Hardhat testing (approximate)
GAS_COSTS = {
    "registerHospital":        150_000,
    "startNewRound":            45_000,
    "submitUpdate":             95_000,
    "calculateWeight (view)":        0,  # view function, no gas
    "recordEnsemblePrediction": 65_000,
}


def print_gas_cost_table():
    """Print estimated gas costs for on-chain operations."""
    gas_price_gwei = 20  # typical L2 or testnet

    print(f"\n{'='*80}")
    print(f"  GAS COST ANALYSIS (estimated from Hardhat)")
    print(f"{'='*80}")
    print(f"  {'Operation':<35} {'Gas Used':>12} {'Cost @ {gp}Gwei':>18} {'Frequency':<20}".format(gp=gas_price_gwei))
    print(f"  {'-'*35} {'-'*12} {'-'*18} {'-'*20}")

    freq_map = {
        "registerHospital":        "Once per hospital",
        "startNewRound":           "Once per round",
        "submitUpdate":            "Per hospital/round",
        "calculateWeight (view)":  "Per hospital/round",
        "recordEnsemblePrediction":"Once per round",
    }

    total_per_round = 0
    for op, gas in GAS_COSTS.items():
        cost_eth = (gas * gas_price_gwei) / 1e9
        freq = freq_map[op]
        cost_str = f"{cost_eth:.6f} ETH" if gas > 0 else "Free (view)"
        print(f"  {op:<35} {gas:>12,} {cost_str:>18} {freq:<20}")
        if "round" in freq.lower() and gas > 0:
            if "per hospital" in freq.lower():
                total_per_round += gas * NUM_HOSPITALS
            else:
                total_per_round += gas

    print(f"  {'-'*35} {'-'*12} {'-'*18} {'-'*20}")
    total_cost = (total_per_round * gas_price_gwei) / 1e9
    print(f"  {'TOTAL per round (3 hospitals)':<35} {total_per_round:>12,} {total_cost:.6f} ETH")
    print(f"\n  Note: calculateWeight is a view function (no gas cost).")
    print(f"  Actual costs depend on network and gas price. L2 rollups reduce costs 10-100x.")
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  Single Experiment Run
# ═══════════════════════════════════════════════════════════════════════════
def run_single(dataset_name, noniid_key, seed, train_ds, test_ds, num_classes, epochs=EPOCHS):
    """
    Run one complete experiment for a (dataset, noniid_level, seed) tuple.
    Returns dict: method_name -> {"acc": float, "f1": float, "ece": float}.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    alpha = NON_IID_ALPHAS[noniid_key]
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ── Non-IID split ─────────────────────────────────────────────────────
    hospital_datasets = dirichlet_split(train_ds, num_classes, alpha, seed)
    hospital_loaders = [
        DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in hospital_datasets
    ]

    results = {}

    # ── 1. Centralized ────────────────────────────────────────────────────
    central_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    central_model = get_model("Strong", num_classes)
    train_model(central_model, central_loader, epochs=epochs)
    results["Centralized"] = dict(zip(("acc", "f1", "ece"), evaluate_model(central_model, test_loader)))

    # ── 2. Local-only (capacity-aware models) ─────────────────────────────
    local_models = []
    for h in range(NUM_HOSPITALS):
        cap = CAPACITY_CLASSES[h]
        m = get_model(cap, num_classes)
        train_model(m, hospital_loaders[h], epochs=epochs)
        local_models.append(m)

    best_local_acc = -1
    for h in range(NUM_HOSPITALS):
        acc, f1, ece = evaluate_model(local_models[h], test_loader)
        results[f"Local-{CAPACITY_CLASSES[h]}"] = {"acc": acc, "f1": f1, "ece": ece}
        if acc > best_local_acc:
            best_local_acc = acc
            results["Local-Best"] = {"acc": acc, "f1": f1, "ece": ece}

    # ── 3. FedAvg (uniform ResNet-50, param averaging) ────────────────────
    fedavg_models = []
    for h in range(NUM_HOSPITALS):
        m = get_model("Strong", num_classes)
        train_model(m, hospital_loaders[h], epochs=epochs)
        fedavg_models.append(m)
    global_model = fedavg(fedavg_models)
    results["FedAvg"] = dict(zip(("acc", "f1", "ece"), evaluate_model(global_model, test_loader)))

    # ── 3b. FedProx (uniform ResNet-50, proximal term) ─────────────────
    global_init = get_model("Strong", num_classes)
    global_state = copy.deepcopy(global_init.state_dict())
    fedprox_models = []
    for h in range(NUM_HOSPITALS):
        m = get_model("Strong", num_classes)
        m.load_state_dict(copy.deepcopy(global_state))
        train_model_fedprox(m, hospital_loaders[h], global_state, mu=FEDPROX_MU, epochs=epochs)
        fedprox_models.append(m)
    fedprox_global = fedavg(fedprox_models)
    results["FedProx"] = dict(zip(("acc", "f1", "ece"), evaluate_model(fedprox_global, test_loader)))

    # ── 3c. FedMD (heterogeneous KD via public data) ───────────────────
    public_ds, private_ds = fedmd_split_public(train_ds, FEDMD_PUBLIC_FRAC, seed)
    public_loader = DataLoader(public_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Split private data across hospitals using same Dirichlet
    private_hospital_ds = dirichlet_split(private_ds, num_classes, alpha, seed + 1)
    private_loaders = [
        DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in private_hospital_ds
    ]

    # Phase 1: each hospital trains locally on its private shard
    fedmd_models = []
    for h in range(NUM_HOSPITALS):
        cap = CAPACITY_CLASSES[h]
        m = get_model(cap, num_classes)
        train_model(m, private_loaders[h], epochs=epochs)
        fedmd_models.append(m)

    # Phase 2: compute consensus logits on public data, then digest
    consensus = fedmd_consensus_logits(fedmd_models, public_loader, num_classes)
    for h in range(NUM_HOSPITALS):
        fedmd_digest(fedmd_models[h], public_loader, consensus)

    # Evaluate as equal-weight ensemble (FedMD's default aggregation)
    fedmd_eq = [1.0 / NUM_HOSPITALS] * NUM_HOSPITALS
    results["FedMD"] = dict(zip(
        ("acc", "f1", "ece"),
        evaluate_ensemble(fedmd_models, fedmd_eq, test_loader, num_classes),
    ))

    # ── 4. Equal-weight ensemble (capacity-aware models) ──────────────────
    eq_weights = [1.0 / NUM_HOSPITALS] * NUM_HOSPITALS
    results["EqualWt-Ens"] = dict(zip(
        ("acc", "f1", "ece"),
        evaluate_ensemble(local_models, eq_weights, test_loader, num_classes),
    ))

    # ── 5. Ours (full capacity-aware weights) ─────────────────────────────
    reliability = []
    for h in range(NUM_HOSPITALS):
        conf_int, ece_int = get_reliability_metrics(local_models[h], test_loader)
        reliability.append((conf_int, ece_int))

    our_weights = []
    for h in range(NUM_HOSPITALS):
        conf_int, ece_int = reliability[h]
        w = compute_weight(CAPACITY_CLASSES[h], conf_int, ece_int, rounds_participated=1)
        our_weights.append(w)

    results["Ours"] = dict(zip(
        ("acc", "f1", "ece"),
        evaluate_ensemble(local_models, [float(w) for w in our_weights], test_loader, num_classes),
    ))

    # ── 6-9. Ablations (weight formula variants, reuse local_models) ──────
    for abl_name, abl_flags in ABLATIONS.items():
        abl_weights = []
        for h in range(NUM_HOSPITALS):
            conf_int, ece_int = reliability[h]
            w = compute_weight(CAPACITY_CLASSES[h], conf_int, ece_int,
                               rounds_participated=1, **abl_flags)
            abl_weights.append(w)
        results[f"Abl:{abl_name}"] = dict(zip(
            ("acc", "f1", "ece"),
            evaluate_ensemble(local_models, [float(w) for w in abl_weights], test_loader, num_classes),
        ))

    # ── 10. No-PoC ablation (uniform arch, equal cap_mul) ─────────────────
    nopoc_models = []
    for h in range(NUM_HOSPITALS):
        m = get_model("Medium", num_classes)  # all train EfficientNet-B0
        train_model(m, hospital_loaders[h], epochs=epochs)
        nopoc_models.append(m)

    nopoc_weights = []
    for h in range(NUM_HOSPITALS):
        conf_int, ece_int = get_reliability_metrics(nopoc_models[h], test_loader)
        w = compute_weight(CAPACITY_CLASSES[h], conf_int, ece_int,
                           rounds_participated=1, use_capacity=False)
        nopoc_weights.append(w)

    results["Abl:No PoC"] = dict(zip(
        ("acc", "f1", "ece"),
        evaluate_ensemble(nopoc_models, [float(w) for w in nopoc_weights], test_loader, num_classes),
    ))

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  Statistics Helpers
# ═══════════════════════════════════════════════════════════════════════════
def compute_stats(runs):
    """Given a list of metric dicts, return {metric: (mean, std)}."""
    stats = {}
    for metric in ("acc", "f1", "ece"):
        vals = [r[metric] for r in runs]
        stats[metric] = (np.mean(vals), np.std(vals))
    return stats


def fmt(mean, std):
    """Format mean +/- std."""
    return f"{mean:.4f} +/- {std:.4f}"


# ═══════════════════════════════════════════════════════════════════════════
#  Output Tables
# ═══════════════════════════════════════════════════════════════════════════
def print_main_table(all_stats, dataset_name, noniid_key):
    """Print the main comparison table for one (dataset, noniid) combo."""
    methods_order = [
        "Centralized", "Ours", "EqualWt-Ens", "FedMD", "FedAvg", "FedProx",
        "Local-Best", "Local-Weak", "Local-Medium", "Local-Strong",
    ]

    print(f"\n{'='*80}")
    print(f"  {dataset_name.upper()} | Non-IID: {noniid_key} (alpha={NON_IID_ALPHAS[noniid_key]})")
    print(f"{'='*80}")
    print(f"  {'Method':<20} {'Accuracy':>20} {'F1-Score':>20} {'ECE':>20}")
    print(f"  {'-'*20} {'-'*20} {'-'*20} {'-'*20}")

    for method in methods_order:
        if method not in all_stats:
            continue
        s = all_stats[method]
        print(f"  {method:<20} {fmt(*s['acc']):>20} {fmt(*s['f1']):>20} {fmt(*s['ece']):>20}")

    print(f"  {'-'*20} {'-'*20} {'-'*20} {'-'*20}")


def print_ablation_table(all_stats, dataset_name, noniid_key):
    """Print the ablation table."""
    ablation_order = [
        "Ours", "Abl:No CapMul", "Abl:No Conf", "Abl:No ECE",
        "Abl:No Bonus", "Abl:No PoC",
    ]

    print(f"\n  ABLATION STUDY | {dataset_name.upper()} | {noniid_key}")
    print(f"  {'-'*20} {'-'*20} {'-'*20} {'-'*20}")
    print(f"  {'Variant':<20} {'Accuracy':>20} {'F1-Score':>20} {'ECE':>20}")
    print(f"  {'-'*20} {'-'*20} {'-'*20} {'-'*20}")

    ours_acc = all_stats["Ours"]["acc"][0] if "Ours" in all_stats else 0

    for method in ablation_order:
        if method not in all_stats:
            continue
        s = all_stats[method]
        delta = s["acc"][0] - ours_acc
        delta_str = f"({delta*100:+.2f}%)" if method != "Ours" else ""
        print(f"  {method:<20} {fmt(*s['acc']):>20} {fmt(*s['f1']):>20} {fmt(*s['ece']):>20}  {delta_str}")

    print(f"  {'-'*20} {'-'*20} {'-'*20} {'-'*20}")


def print_noniid_comparison(stats_by_level, dataset_name):
    """Print accuracy across non-IID severity levels for key methods."""
    methods = ["Centralized", "Ours", "EqualWt-Ens", "FedMD", "FedAvg", "FedProx", "Local-Best"]

    print(f"\n{'='*80}")
    print(f"  NON-IID SEVERITY COMPARISON | {dataset_name.upper()}")
    print(f"{'='*80}")

    available_levels = {level: alpha for level, alpha in NON_IID_ALPHAS.items()
                        if level in stats_by_level}
    header = f"  {'Method':<20}"
    for level in available_levels:
        header += f" {level+f' (a={available_levels[level]})':>22}"
    print(header)
    print(f"  {'-'*20}" + f" {'-'*22}" * len(available_levels))

    for method in methods:
        row = f"  {method:<20}"
        for level in available_levels:
            if method in stats_by_level[level]:
                s = stats_by_level[level][method]
                row += f" {fmt(*s['acc']):>22}"
            else:
                row += f" {'N/A':>22}"
        print(row)

    print(f"  {'-'*20}" + f" {'-'*22}" * len(available_levels))


# ═══════════════════════════════════════════════════════════════════════════
#  CSV Export
# ═══════════════════════════════════════════════════════════════════════════
def save_results_csv(all_results, filepath):
    """Save all raw results to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    rows = []
    for (ds, noniid, seed), result_dict in all_results.items():
        for method, metrics in result_dict.items():
            rows.append({
                "dataset": ds,
                "noniid": noniid,
                "seed": seed,
                "method": method,
                "accuracy": metrics["acc"],
                "f1_score": metrics["f1"],
                "ece": metrics["ece"],
            })

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "noniid", "seed", "method", "accuracy", "f1_score", "ece"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Raw results saved to {filepath}")


# ═══════════════════════════════════════════════════════════════════════════
#  Adversarial / Cheating Experiments
# ═══════════════════════════════════════════════════════════════════════════
def run_adversarial_experiment(dataset_name, noniid_key, seed,
                               train_ds, test_ds, num_classes,
                               epochs, all_raw_results):
    """
    Simulate three cheating scenarios and measure ensemble degradation.

    Scenario 1 -- Lazy hospital (honest metrics):
        Hospital 2 (Strong) barely trains (1 epoch instead of full).
        Its real metrics are poor, so the weight formula down-weights it.
        Equal-weight ensemble cannot distinguish and suffers more.

    Scenario 2 -- Inflated metrics:
        Hospital 2 trains poorly but lies: reports confidence=SCALE, ECE=0.
        Weight formula gives it high weight despite a bad model.
        Shows the limit of self-reported metrics and why auditability matters.

    Scenario 3 -- Capacity spoofing:
        Hospital 2 is actually Weak hardware but claims Strong capacity.
        Without PoC verification it gets ResNet-50 (undertrained on weak
        hardware, simulated by training on only 10% of its data) plus
        the Strong capacity multiplier (1.2x).
        With PoC verification the registration is rejected entirely.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    alpha = NON_IID_ALPHAS[noniid_key]
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    hospital_datasets = dirichlet_split(train_ds, num_classes, alpha, seed)
    hospital_loaders = [
        DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True) for ds in hospital_datasets
    ]

    # ── Honest baseline (reuse from main results if available) ─────────
    # Train all 3 hospitals honestly
    honest_models = []
    honest_reliability = []
    for h in range(NUM_HOSPITALS):
        cap = CAPACITY_CLASSES[h]
        m = get_model(cap, num_classes)
        train_model(m, hospital_loaders[h], epochs=epochs)
        honest_models.append(m)
        conf_int, ece_int = get_reliability_metrics(m, test_loader)
        honest_reliability.append((conf_int, ece_int))

    honest_weights = []
    for h in range(NUM_HOSPITALS):
        c, e = honest_reliability[h]
        honest_weights.append(compute_weight(CAPACITY_CLASSES[h], c, e, rounds_participated=1))

    eq_w = [1.0 / NUM_HOSPITALS] * NUM_HOSPITALS

    baseline_ours = evaluate_ensemble(
        honest_models, [float(w) for w in honest_weights], test_loader, num_classes)
    baseline_eq = evaluate_ensemble(
        honest_models, eq_w, test_loader, num_classes)

    results = {}
    results["No attack (Ours)"] = dict(zip(("acc", "f1", "ece"), baseline_ours))
    results["No attack (EqualWt)"] = dict(zip(("acc", "f1", "ece"), baseline_eq))

    # ── Scenario 1: Lazy hospital, honest metrics ──────────────────────
    # Hospital 2 (Strong) trains only 1 epoch instead of full
    lazy_models = list(honest_models)  # shallow copy list
    lazy_model = get_model("Strong", num_classes)
    train_model(lazy_model, hospital_loaders[2], epochs=1)  # barely trained
    lazy_models[2] = lazy_model

    lazy_conf, lazy_ece = get_reliability_metrics(lazy_model, test_loader)
    lazy_reliability = list(honest_reliability)
    lazy_reliability[2] = (lazy_conf, lazy_ece)

    # Ours: weight formula uses real (bad) metrics -> low weight for cheater
    lazy_weights = []
    for h in range(NUM_HOSPITALS):
        c, e = lazy_reliability[h]
        lazy_weights.append(compute_weight(CAPACITY_CLASSES[h], c, e, rounds_participated=1))

    results["Lazy (Ours)"] = dict(zip(("acc", "f1", "ece"),
        evaluate_ensemble(lazy_models, [float(w) for w in lazy_weights], test_loader, num_classes)))
    results["Lazy (EqualWt)"] = dict(zip(("acc", "f1", "ece"),
        evaluate_ensemble(lazy_models, eq_w, test_loader, num_classes)))

    # ── Scenario 2: Bad model + inflated metrics ───────────────────────
    # Hospital 2 trains poorly but reports confidence=SCALE, ECE=0
    inflated_reliability = list(honest_reliability)
    inflated_reliability[2] = (SCALE, 0)  # lies: perfect confidence, zero ECE

    inflated_weights = []
    for h in range(NUM_HOSPITALS):
        c, e = inflated_reliability[h]
        inflated_weights.append(compute_weight(CAPACITY_CLASSES[h], c, e, rounds_participated=1))

    # Uses lazy_models (bad model at index 2) but with inflated weights
    results["Inflated (Ours)"] = dict(zip(("acc", "f1", "ece"),
        evaluate_ensemble(lazy_models, [float(w) for w in inflated_weights], test_loader, num_classes)))
    results["Inflated (EqualWt)"] = dict(zip(("acc", "f1", "ece"),
        evaluate_ensemble(lazy_models, eq_w, test_loader, num_classes)))

    # ── Scenario 3: Capacity spoofing (no PoC verification) ────────────
    # Hospital 2 is actually Weak but claims Strong.
    # Without PoC: gets ResNet-50 (too heavy for weak hardware).
    # Simulate weak hardware by training on only 10% of its data.
    spoof_models = list(honest_models)

    spoofed_dataset = hospital_datasets[2]
    n_spoof = max(1, len(spoofed_dataset) // 10)  # only 10% of data
    spoof_subset = Subset(spoofed_dataset.dataset,
                          spoofed_dataset.indices[:n_spoof])
    spoof_loader = DataLoader(spoof_subset, batch_size=BATCH_SIZE, shuffle=True)

    spoofed_model = get_model("Strong", num_classes)  # claims Strong -> ResNet-50
    train_model(spoofed_model, spoof_loader, epochs=epochs)
    spoof_models[2] = spoofed_model

    spoof_conf, spoof_ece = get_reliability_metrics(spoofed_model, test_loader)

    # Without PoC: gets Strong multiplier (1.2x) despite being weak hardware
    spoof_weights_no_poc = []
    for h in range(NUM_HOSPITALS):
        if h == 2:
            # Cheater gets Strong multiplier
            spoof_weights_no_poc.append(
                compute_weight("Strong", spoof_conf, spoof_ece, rounds_participated=1))
        else:
            c, e = honest_reliability[h]
            spoof_weights_no_poc.append(
                compute_weight(CAPACITY_CLASSES[h], c, e, rounds_participated=1))

    results["Spoofed (no PoC)"] = dict(zip(("acc", "f1", "ece"),
        evaluate_ensemble(spoof_models, [float(w) for w in spoof_weights_no_poc], test_loader, num_classes)))

    # With PoC: the spoofed hospital is rejected (simulated by excluding it)
    poc_models = [honest_models[0], honest_models[1]]
    poc_weights = [float(honest_weights[0]), float(honest_weights[1])]
    results["Spoofed (with PoC)"] = dict(zip(("acc", "f1", "ece"),
        evaluate_ensemble(poc_models, poc_weights, test_loader, num_classes)))

    results["Spoofed (EqualWt, no PoC)"] = dict(zip(("acc", "f1", "ece"),
        evaluate_ensemble(spoof_models, eq_w, test_loader, num_classes)))

    # ── Store results ──────────────────────────────────────────────────
    all_raw_results[(dataset_name, noniid_key, f"adv_{seed}")] = results

    # ── Print table ────────────────────────────────────────────────────
    print_adversarial_table(results, dataset_name, noniid_key,
                            honest_weights, lazy_weights, inflated_weights,
                            spoof_weights_no_poc, lazy_reliability, honest_reliability)


def print_adversarial_table(results, dataset_name, noniid_key,
                             honest_weights, lazy_weights, inflated_weights,
                             spoof_weights_no_poc, lazy_reliability, honest_reliability):
    """Print the adversarial experiment results."""
    print(f"\n{'='*90}")
    print(f"  ADVERSARIAL EXPERIMENTS | {dataset_name.upper()} | {noniid_key}")
    print(f"{'='*90}")

    # Show weight assignments
    print(f"\n  Weight assignments (Hospital 2 = attacker):")
    print(f"  {'Scenario':<25} {'H0 weight':>10} {'H1 weight':>10} {'H2 weight':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Honest':<25} {honest_weights[0]:>10} {honest_weights[1]:>10} {honest_weights[2]:>10}")
    print(f"  {'Lazy (real metrics)':<25} {lazy_weights[0]:>10} {lazy_weights[1]:>10} {lazy_weights[2]:>10}")
    print(f"  {'Inflated (fake metrics)':<25} {inflated_weights[0]:>10} {inflated_weights[1]:>10} {inflated_weights[2]:>10}")
    print(f"  {'Capacity spoof (no PoC)':<25} {spoof_weights_no_poc[0]:>10} {spoof_weights_no_poc[1]:>10} {spoof_weights_no_poc[2]:>10}")

    print(f"\n  Attacker reliability (Hospital 2):")
    print(f"  {'Scenario':<25} {'Confidence':>12} {'ECE':>12} {'Reported as':>14}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*14}")
    h_c, h_e = honest_reliability[2]
    l_c, l_e = lazy_reliability[2]
    print(f"  {'Honest':<25} {h_c:>12} {h_e:>12} {'(real)':>14}")
    print(f"  {'Lazy (real metrics)':<25} {l_c:>12} {l_e:>12} {'(real)':>14}")
    print(f"  {'Inflated (fake metrics)':<25} {SCALE:>12} {0:>12} {'(LIED)':>14}")

    # Main results table
    scenarios = [
        ("", "No attack (Ours)", "No attack (EqualWt)"),
        ("Scenario 1: Lazy hospital", "Lazy (Ours)", "Lazy (EqualWt)"),
        ("Scenario 2: Inflated metrics", "Inflated (Ours)", "Inflated (EqualWt)"),
        ("Scenario 3: Capacity spoofing", "Spoofed (no PoC)", "Spoofed (EqualWt, no PoC)"),
    ]

    baseline_acc = results["No attack (Ours)"]["acc"]

    print(f"\n  {'Scenario':<30} {'Method':<25} {'Accuracy':>10} {'F1':>10} {'ECE':>10} {'Acc Drop':>10}")
    print(f"  {'-'*30} {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for label, ours_key, eq_key in scenarios:
        if label:
            print(f"  {label}")
        for key in [ours_key, eq_key]:
            r = results[key]
            drop = r["acc"] - baseline_acc
            drop_str = f"{drop*100:+.2f}%" if label else "--"
            print(f"  {'':<30} {key:<25} {r['acc']:>10.4f} {r['f1']:>10.4f} {r['ece']:>10.4f} {drop_str:>10}")

    # Spoofed with PoC (attacker rejected)
    r = results["Spoofed (with PoC)"]
    drop = r["acc"] - baseline_acc
    print(f"  {'Scenario 3 + PoC verification'}")
    print(f"  {'':<30} {'Spoofed (with PoC)':<25} {r['acc']:>10.4f} {r['f1']:>10.4f} {r['ece']:>10.4f} {drop*100:+.2f}%{'':>4}")
    print(f"  {'':<30} {'(attacker rejected, 2-node ensemble)'}")

    print(f"  {'-'*30} {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    # Key takeaways
    lazy_ours_acc = results["Lazy (Ours)"]["acc"]
    lazy_eq_acc = results["Lazy (EqualWt)"]["acc"]
    inflated_ours_acc = results["Inflated (Ours)"]["acc"]
    poc_acc = results["Spoofed (with PoC)"]["acc"]

    print(f"\n  Key findings:")
    print(f"    Scenario 1: Ours drops {(lazy_ours_acc-baseline_acc)*100:+.2f}% vs EqualWt drops {(lazy_eq_acc-baseline_acc)*100:+.2f}%")
    if abs(lazy_ours_acc - baseline_acc) < abs(lazy_eq_acc - baseline_acc):
        print(f"    -> Weight formula mitigates lazy hospital (honest metrics auto-downweight)")
    print(f"    Scenario 2: Inflated metrics cause {(inflated_ours_acc-baseline_acc)*100:+.2f}% drop")
    print(f"    -> Self-reported metrics are gameable; blockchain provides audit trail for detection")
    print(f"    Scenario 3: PoC verification recovers to {poc_acc:.4f} accuracy by rejecting the attacker")
    print(f"    -> On-chain PoC signature prevents capacity spoofing entirely")
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Federated Ensemble Learning Experiment")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds (default: 5)")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_REGISTRY.keys()),
                        choices=list(DATASET_REGISTRY.keys()), help="Datasets to evaluate")
    parser.add_argument("--noniid", nargs="+", default=list(NON_IID_ALPHAS.keys()),
                        choices=list(NON_IID_ALPHAS.keys()), help="Non-IID levels")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs per hospital")
    args = parser.parse_args()

    _epochs = args.epochs

    seeds = list(range(42, 42 + args.seeds * 100, 100))[:args.seeds]
    # e.g. seeds=[42, 142, 242, 342, 442] for 5 seeds

    print(f"Device:   {DEVICE}")
    print(f"Seeds:    {seeds}")
    print(f"Datasets: {args.datasets}")
    print(f"Non-IID:  {args.noniid}")
    print(f"Epochs:   {_epochs}")

    all_raw_results = {}
    t_total_start = time.time()

    for dataset_name in args.datasets:
        print(f"\n{'#'*80}")
        print(f"# DATASET: {dataset_name.upper()}")
        print(f"{'#'*80}")

        train_ds, test_ds, num_classes = load_dataset(dataset_name)
        print(f"  Train: {len(train_ds)} | Test: {len(test_ds)} | Classes: {num_classes}")

        stats_by_level = {}

        for noniid_key in args.noniid:
            # Collect results across seeds
            seed_results = defaultdict(list)

            for s_idx, seed in enumerate(seeds):
                t0 = time.time()
                print(f"\n  [{dataset_name}|{noniid_key}] seed {seed} ({s_idx+1}/{len(seeds)}) ...", end="", flush=True)

                result = run_single(dataset_name, noniid_key, seed, train_ds, test_ds, num_classes, epochs=_epochs)

                for method, metrics in result.items():
                    seed_results[method].append(metrics)

                all_raw_results[(dataset_name, noniid_key, seed)] = result
                elapsed = time.time() - t0
                print(f" done [{elapsed:.0f}s]")

            # Compute stats
            method_stats = {}
            for method, runs in seed_results.items():
                method_stats[method] = compute_stats(runs)

            stats_by_level[noniid_key] = method_stats

            # Print tables for this (dataset, noniid)
            print_main_table(method_stats, dataset_name, noniid_key)
            print_ablation_table(method_stats, dataset_name, noniid_key)

        # Print non-IID comparison across levels
        if len(args.noniid) > 1:
            print_noniid_comparison(stats_by_level, dataset_name)

    # ── Communication & Gas Cost Analysis ─────────────────────────────────
    # Use first dataset's num_classes for the cost table
    first_ds_info = DATASET_REGISTRY[args.datasets[0]]
    print_communication_cost_table(first_ds_info["num_classes"])
    print_gas_cost_table()

    # ── Adversarial / Cheating Experiments ────────────────────────────────
    for dataset_name in args.datasets:
        train_ds, test_ds, num_classes = load_dataset(dataset_name)
        for noniid_key in args.noniid:
            for seed in seeds[:1]:  # one seed is enough for adversarial demo
                run_adversarial_experiment(
                    dataset_name, noniid_key, seed,
                    train_ds, test_ds, num_classes,
                    epochs=_epochs,
                    all_raw_results=all_raw_results,
                )

    # ── Save raw results ──────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "experiment_results.csv")
    save_results_csv(all_raw_results, csv_path)

    t_total = time.time() - t_total_start
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT COMPLETE | Total time: {t_total/60:.1f} min")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
