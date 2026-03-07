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
  4. Equal-wt Ens.   - Our 3 models, equal weights (1/3)
  5. Ours (Full)     - Our 3 models, capacity-aware weights

Ablations (reuse Ours models, modify weight formula):
  6. No CapMul       - capacity multiplier = 1.0 for all
  7. No Confidence   - confidence = 1.0 for all
  8. No ECE          - ECE = 0 for all
  9. No Bonus        - participation bonus = 0
 10. No PoC          - uniform architecture (EfficientNet-B0), equal cap_mul

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
        "Centralized", "Ours", "EqualWt-Ens", "FedAvg", "Local-Best",
        "Local-Weak", "Local-Medium", "Local-Strong",
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
    methods = ["Centralized", "Ours", "EqualWt-Ens", "FedAvg", "Local-Best"]

    print(f"\n{'='*80}")
    print(f"  NON-IID SEVERITY COMPARISON | {dataset_name.upper()}")
    print(f"{'='*80}")

    header = f"  {'Method':<20}"
    for level in NON_IID_ALPHAS:
        header += f" {level+f' (a={NON_IID_ALPHAS[level]})':>22}"
    print(header)
    print(f"  {'-'*20}" + f" {'-'*22}" * len(NON_IID_ALPHAS))

    for method in methods:
        row = f"  {method:<20}"
        for level in NON_IID_ALPHAS:
            if method in stats_by_level[level]:
                s = stats_by_level[level][method]
                row += f" {fmt(*s['acc']):>22}"
            else:
                row += f" {'N/A':>22}"
        print(row)

    print(f"  {'-'*20}" + f" {'-'*22}" * len(NON_IID_ALPHAS))


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

    # ── Save raw results ──────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "experiment_results.csv")
    save_results_csv(all_raw_results, csv_path)

    t_total = time.time() - t_total_start
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT COMPLETE | Total time: {t_total/60:.1f} min")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
