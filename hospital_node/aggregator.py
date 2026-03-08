"""
aggregator.py — Federated Averaging (FedAvg) implementation
=============================================================
Implements the classic FedAvg algorithm for aggregating model
weights received from multiple hospital nodes. This runs on
the coordinating server side (or can be used in simulation).
"""

import copy
import torch
import torch.nn as nn
from typing import List, Dict


def federated_average(
    global_model: nn.Module,
    local_state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[float] | None = None,
) -> nn.Module:
    """
    Perform weighted Federated Averaging across local model updates.

    Args:
        global_model:      The global model to update.
        local_state_dicts: List of state dicts from participating nodes.
        weights:           Optional per-node weights (e.g. proportional to
                           dataset size). If None, uniform weighting is used.

    Returns:
        Updated global model with averaged parameters.
    """
    if not local_state_dicts:
        raise ValueError("No local state dicts provided for aggregation")

    n = len(local_state_dicts)

    if weights is None:
        weights = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise ValueError(
                f"weights length ({len(weights)}) != state_dicts length ({n})"
            )
        total = sum(weights)
        if total == 0:
            raise ValueError("Sum of weights is zero, cannot aggregate")
        weights = [w / total for w in weights]

    # Start with zeros
    avg_state = {}
    for key in local_state_dicts[0]:
        avg_state[key] = torch.zeros_like(local_state_dicts[0][key], dtype=torch.float32)

    # Weighted sum
    for state_dict, w in zip(local_state_dicts, weights):
        for key in avg_state:
            avg_state[key] += state_dict[key].float() * w

    # Load into a copy of the global model
    updated_model = copy.deepcopy(global_model)
    updated_model.load_state_dict(avg_state)
    return updated_model
