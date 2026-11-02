"""Checkpointing and history persistence utilities for VAE models."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    val_loss: float,
    val_recon_loss: float,
    val_kl_loss: float,
    model_type: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Saves a model checkpoint with metadata.
    
    Args:
        checkpoint_path: Destination path for checkpoint file (.pt).
        model: VAE model instance.
        optimizer: Optimizer instance.
        epoch: Epoch index (1-indexed).
        val_loss: Validation total loss at this epoch.
        val_recon_loss: Validation reconstruction NLL.
        val_kl_loss: Validation KL loss.
        model_type: One of 'A', 'B', 'C', 'D'.
        extra_metadata: Optional additional metadata dict.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_type": model_type,
        "val_loss": float(val_loss),
        "val_recon_loss": float(val_recon_loss),
        "val_kl_loss": float(val_kl_loss),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metadata": extra_metadata or {},
    }

    torch.save(payload, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Loads a model checkpoint into an existing model and optional optimizer.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model instance into which weights are loaded.
        optimizer: Optional optimizer into which state is loaded.
        device: Target device for loading tensors.
        
    Returns:
        Checkpoint dictionary containing metadata.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def save_history(
    history: Dict[str, List[float]],
    json_path: Path,
    csv_path: Optional[Path] = None,
    extra_summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Saves training history to JSON and optional CSV.
    
    Args:
        history: Dict mapping metric names to lists of per-epoch values.
        json_path: Path to JSON history output.
        csv_path: Optional path to CSV history output.
        extra_summary: Optional metadata (e.g. best epoch, early stopping info).
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "history": history,
        "summary": extra_summary or {},
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        keys = list(history.keys())
        num_epochs = len(history[keys[0]]) if keys else 0

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch"] + keys)
            for ep in range(num_epochs):
                row = [ep + 1] + [history[k][ep] for k in keys]
                writer.writerow(row)
