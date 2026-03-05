"""
simulate_federation.py — End-to-end local simulation
======================================================
Simulates a full federated learning workflow locally:

1. Spins up N virtual hospital nodes.
2. Each node trains a local model on a synthetic data partition.
3. Model weights are aggregated via FedAvg.
4. Repeats for a configurable number of rounds.

Usage:
    python simulate_federation.py --nodes 3 --rounds 5 --epochs 2
"""

import argparse
import copy
import sys
import os

# Allow importing from sibling packages
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from hospital_node.model import FederatedCNN, get_model_hash
from hospital_node.trainer import run_training
from hospital_node.data_loader import get_data_loaders
from hospital_node.aggregator import federated_average


def simulate(num_nodes: int, num_rounds: int, epochs_per_round: int):
    """Run a local federated learning simulation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Nodes : {num_nodes}")
    print(f"Rounds: {num_rounds}")
    print(f"Epochs per round: {epochs_per_round}")
    print("=" * 60)

    # Initialize global model
    global_model = FederatedCNN(num_classes=10)

    # Create data partitions — each node gets a different random seed
    node_loaders = []
    for i in range(num_nodes):
        train_loader, val_loader = get_data_loaders(
            use_synthetic=True,
            num_samples=500,
            batch_size=32,
            seed=42 + i,
        )
        node_loaders.append((train_loader, val_loader))

    # ── Federated rounds ────────────────────────────────────────────────
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'─' * 60}")
        print(f"  ROUND {round_num}/{num_rounds}")
        print(f"{'─' * 60}")

        local_state_dicts = []
        data_sizes = []

        for node_id in range(num_nodes):
            print(f"\n  🏥 Hospital Node {node_id + 1}")

            # Each node starts from the current global model
            local_model = copy.deepcopy(global_model)
            train_loader, val_loader = node_loaders[node_id]

            # Local training
            local_model = run_training(
                model=local_model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs_per_round,
                lr=0.001,
                device=device,
            )

            # Collect state dict and data size
            local_state_dicts.append(copy.deepcopy(local_model.state_dict()))
            data_sizes.append(len(train_loader.dataset))

            model_hash = get_model_hash(local_model)
            print(f"    Model hash: {model_hash[:24]}…")

        # ── Aggregation ────────────────────────────────────────────────
        print(f"\n  ⚙ Aggregating {len(local_state_dicts)} model updates (FedAvg)...")
        global_model = federated_average(
            global_model=global_model,
            local_state_dicts=local_state_dicts,
            weights=[float(s) for s in data_sizes],
        )
        global_hash = get_model_hash(global_model)
        print(f"  ✓ New global model hash: {global_hash[:24]}…")

    print(f"\n{'=' * 60}")
    print("Simulation complete!")
    print(f"Final global model hash: {get_model_hash(global_model)[:32]}…")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Simulation")
    parser.add_argument("--nodes", type=int, default=3, help="Number of hospital nodes")
    parser.add_argument("--rounds", type=int, default=3, help="Number of FL rounds")
    parser.add_argument("--epochs", type=int, default=2, help="Local epochs per round")
    args = parser.parse_args()

    simulate(
        num_nodes=args.nodes,
        num_rounds=args.rounds,
        epochs_per_round=args.epochs,
    )



