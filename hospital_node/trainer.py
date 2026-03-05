"""
trainer.py — Local model training loop
=======================================
Handles the local training process at each hospital node.  After training,
the updated model weights are serialized and their hash is computed for
submission to the federated learning smart contract.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Returns:
        (avg_loss, accuracy) for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on a validation / test set.

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int = 5,
    lr: float = 0.001,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Full local training routine.

    Args:
        model:        The neural network to train.
        train_loader: DataLoader for training data.
        val_loader:   Optional DataLoader for validation data.
        epochs:       Number of local epochs.
        lr:           Learning rate.
        device:       Device (cpu / cuda).

    Returns:
        The trained model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        print(
            f"  Epoch {epoch}/{epochs} — "
            f"loss: {train_loss:.4f}, acc: {train_acc:.4f}",
            end="",
        )

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(f" | val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
        else:
            print()

    return model
