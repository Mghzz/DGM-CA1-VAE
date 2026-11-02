"""Training engine for VAE models (A, B, C, D).

Implements:
1. Unified training loop supporting all four priors.
2. Separate per-batch and per-epoch tracking for reconstruction NLL, KL, and total loss.
3. Isolated, reproducible validation RNG that does not advance the training RNG stream.
4. Exact early-stopping monitoring validation total loss (patience = 15).
5. Automatic restoration of the best-epoch model weights upon completion.
6. History logging (JSON and CSV) and checkpoint persistence.
7. Safe handling of incomplete batches and NaN/Inf detection.
"""

from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.config import ProjectConfig
from src.models.vae import VAE
from src.training.checkpointing import save_checkpoint, save_history


@dataclass
class TrainingResult:
    """Container for training execution output."""
    model: VAE
    model_type: str
    history: Dict[str, List[float]]
    best_epoch: int
    best_val_loss: float
    best_val_recon: float
    best_val_kl: float
    stopped_early: bool
    total_epochs: int
    checkpoint_path: Optional[Path] = None
    history_path: Optional[Path] = None


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for reproducibility.
    
    Args:
        seed: Random seed integer.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """Reusable training manager for VAE models."""

    def __init__(
        self,
        model: VAE,
        model_type: str,
        config: Optional[ProjectConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.model = model
        self.model_type = model_type.upper()
        self.config = config or ProjectConfig()

        # 1. Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # 2. Optimizer: includes all trainable parameters (encoder, decoder, and learnable prior parameters)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.lr,
        )

        # 3. Dedicated validation generator for isolated, reproducible validation sampling
        self.val_generator = torch.Generator(device=self.device.type)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Runs one training epoch over the training DataLoader.
        
        Args:
            train_loader: DataLoader yielding training batches.
            
        Returns:
            Dict containing mean train total_loss, recon_loss, and kl_loss.
        """
        if len(train_loader) == 0:
            raise ValueError("train_loader must not be empty.")

        self.model.train()

        total_recon_sum = 0.0
        total_kl_sum = 0.0
        total_loss_sum = 0.0
        total_samples = 0

        for batch in train_loader:
            x = batch["x"].to(self.device)
            batch_size = x.shape[0]

            self.optimizer.zero_grad()
            out = self.model(x)  # Reparameterization uses global training RNG
            loss = self.model.compute_loss(x, out)

            # Check for non-finite losses
            if not torch.isfinite(loss.total_loss):
                raise FloatingPointError(
                    f"Non-finite total loss encountered in Model {self.model_type}: "
                    f"total={loss.total_loss.item()}, recon={loss.recon_loss.item()}, kl={loss.kl_loss.item()}"
                )

            loss.total_loss.backward()
            self.optimizer.step()

            total_recon_sum += loss.recon_loss.item() * batch_size
            total_kl_sum += loss.kl_loss.item() * batch_size
            total_loss_sum += loss.total_loss.item() * batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("No samples were processed in train_epoch; total_samples is 0.")

        return {
            "total_loss": total_loss_sum / total_samples,
            "recon_loss": total_recon_sum / total_samples,
            "kl_loss": total_kl_sum / total_samples,
        }

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Runs one validation pass over the validation DataLoader.
        
        Resets the dedicated validation generator to a fixed seed so that validation
        sampling is strictly deterministic across epochs without consuming the training RNG stream.
        Guarantees that global training RNG states (CPU and CUDA) are completely preserved.
        
        Args:
            val_loader: DataLoader yielding validation batches.
            
        Returns:
            Dict containing mean validation total_loss, recon_loss, and kl_loss.
        """
        if len(val_loader) == 0:
            raise ValueError("val_loader must not be empty.")

        self.model.eval()

        # Capture global RNG state before validation to guarantee training RNG stream is undisturbed
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        try:
            # Reset validation generator to guarantee identical latent sampling sequences per epoch
            self.val_generator.manual_seed(self.config.training.val_eval_seed)

            val_recon_sum = 0.0
            val_kl_sum = 0.0
            val_loss_sum = 0.0
            val_samples = 0

            for batch in val_loader:
                x = batch["x"].to(self.device)
                batch_size = x.shape[0]

                # Forward pass using dedicated validation generator
                out = self.model(x, generator=self.val_generator)
                loss = self.model.compute_loss(x, out)

                if not torch.isfinite(loss.total_loss):
                    raise FloatingPointError(
                        f"Non-finite validation loss encountered in Model {self.model_type}: "
                        f"total={loss.total_loss.item()}, recon={loss.recon_loss.item()}, kl={loss.kl_loss.item()}"
                    )

                val_recon_sum += loss.recon_loss.item() * batch_size
                val_kl_sum += loss.kl_loss.item() * batch_size
                val_loss_sum += loss.total_loss.item() * batch_size
                val_samples += batch_size

            if val_samples == 0:
                raise ValueError("No samples were processed in validate; val_samples is 0.")

            return {
                "total_loss": val_loss_sum / val_samples,
                "recon_loss": val_recon_sum / val_samples,
                "kl_loss": val_kl_sum / val_samples,
            }
        finally:
            # Restore global RNG state exactly as it was before validation
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: Optional[int] = None,
        patience: Optional[int] = None,
        verbose: bool = True,
        save_checkpoints: bool = True,
        checkpoint_dir: Optional[Path] = None,
        history_dir: Optional[Path] = None,
    ) -> TrainingResult:
        """Executes full training with early stopping and best-model restoration.
        
        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            max_epochs: Max epochs to train (defaults to config: 100).
            patience: Early stopping patience epochs (defaults to config: 15).
            verbose: Whether to print per-epoch progress.
            save_checkpoints: Whether to persist best weights to disk.
            checkpoint_dir: Output directory for checkpoints.
            history_dir: Output directory for history JSON/CSV.
            
        Returns:
            TrainingResult containing the restored best model and metrics history.
        """
        if max_epochs is None:
            max_epochs = self.config.training.max_epochs
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {max_epochs}")

        if patience is None:
            patience = self.config.training.early_stopping_patience
        if patience <= 0:
            raise ValueError(f"patience must be positive, got {patience}")

        chk_dir = Path(checkpoint_dir or self.config.paths.checkpoints_dir)
        hist_dir = Path(history_dir or self.config.paths.histories_dir)
        chk_dir.mkdir(parents=True, exist_ok=True)
        hist_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = chk_dir / f"model_{self.model_type}_best.pt"
        history_json_path = hist_dir / f"model_{self.model_type}_history.json"
        history_csv_path = hist_dir / f"model_{self.model_type}_history.csv"

        if verbose:
            device_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU"
            print(f"=== Starting Training: Model {self.model_type} ===")
            print(f"Device: {self.device} ({device_name}) | Max Epochs: {max_epochs} | Early Stopping Patience: {patience}")
            print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

        history = {
            "train_total_loss": [],
            "train_recon_loss": [],
            "train_kl_loss": [],
            "val_total_loss": [],
            "val_recon_loss": [],
            "val_kl_loss": [],
        }

        best_val_loss = float("inf")
        best_val_recon = float("inf")
        best_val_kl = float("inf")
        best_epoch = 0
        patience_counter = 0
        best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        stopped_early = False

        for epoch in range(1, max_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            # Record metrics
            history["train_total_loss"].append(train_metrics["total_loss"])
            history["train_recon_loss"].append(train_metrics["recon_loss"])
            history["train_kl_loss"].append(train_metrics["kl_loss"])
            history["val_total_loss"].append(val_metrics["total_loss"])
            history["val_recon_loss"].append(val_metrics["recon_loss"])
            history["val_kl_loss"].append(val_metrics["kl_loss"])

            # Check for improvement in validation total loss
            is_best = val_metrics["total_loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["total_loss"]
                best_val_recon = val_metrics["recon_loss"]
                best_val_kl = val_metrics["kl_loss"]
                best_epoch = epoch
                patience_counter = 0

                # Cache best weights in CPU memory
                best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

                if save_checkpoints:
                    metadata = {
                        "lr": float(self.config.training.lr),
                        "batch_size": int(self.config.training.batch_size),
                        "max_epochs": int(max_epochs),
                        "patience": int(patience),
                        "model_seed": int(self.config.training.model_seed),
                        "val_eval_seed": int(self.config.training.val_eval_seed),
                        "likelihood_sigma": float(self.config.model.likelihood_sigma),
                        "latent_dim": int(self.config.model.latent_dim),
                    }
                    save_checkpoint(
                        checkpoint_path=checkpoint_path,
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        val_loss=best_val_loss,
                        val_recon_loss=best_val_recon,
                        val_kl_loss=best_val_kl,
                        model_type=self.model_type,
                        extra_metadata=metadata,
                    )
            else:
                patience_counter += 1

            if verbose:
                flag = " * (Best)" if is_best else ""
                print(
                    f"Epoch [{epoch:3d}/{max_epochs:3d}] | "
                    f"Train Loss: {train_metrics['total_loss']:8.4f} (Recon: {train_metrics['recon_loss']:8.4f}, KL: {train_metrics['kl_loss']:7.4f}) | "
                    f"Val Loss: {val_metrics['total_loss']:8.4f} (Recon: {val_metrics['recon_loss']:8.4f}, KL: {val_metrics['kl_loss']:7.4f}) | "
                    f"Best Val: {best_val_loss:8.4f}{flag}"
                )

            # Check early stopping trigger
            if patience_counter >= patience:
                stopped_early = True
                if verbose:
                    print(f"\n[Early Stopping Triggered] No validation improvement for {patience} consecutive epochs.")
                break

        # Restore the best model weights
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            self.model.to(self.device)
            if verbose:
                print(f"[Restoration] Loaded weights from best epoch {best_epoch} with val_loss: {best_val_loss:.4f}")

        # Persist histories
        if save_checkpoints:
            summary = {
                "model_type": self.model_type,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_recon": best_val_recon,
                "best_val_kl": best_val_kl,
                "stopped_early": stopped_early,
                "total_epochs": epoch,
            }
            save_history(
                history=history,
                json_path=history_json_path,
                csv_path=history_csv_path,
                extra_summary=summary,
            )

        if verbose:
            print(f"=== Completed Training: Model {self.model_type} (Best Epoch: {best_epoch}) ===\n")

        return TrainingResult(
            model=self.model,
            model_type=self.model_type,
            history=history,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            best_val_recon=best_val_recon,
            best_val_kl=best_val_kl,
            stopped_early=stopped_early,
            total_epochs=epoch,
            checkpoint_path=checkpoint_path if save_checkpoints else None,
            history_path=history_json_path if save_checkpoints else None,
        )


def train_model(
    model: VAE,
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[ProjectConfig] = None,
    device: Optional[Union[str, torch.device]] = None,
    max_epochs: Optional[int] = None,
    patience: Optional[int] = None,
    verbose: bool = True,
    save_checkpoints: bool = True,
) -> TrainingResult:
    """Convenience functional wrapper for Trainer.fit.
    
    Args:
        model: VAE model instance.
        model_type: Model identifier ('A', 'B', 'C', 'D').
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Optional project configuration.
        device: Optional target device.
        max_epochs: Max epochs to train.
        patience: Early stopping patience.
        verbose: Verbose printing flag.
        save_checkpoints: Whether to save checkpoints and histories.
        
    Returns:
        TrainingResult instance.
    """
    trainer = Trainer(
        model=model,
        model_type=model_type,
        config=config,
        device=device,
    )
    return trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=max_epochs,
        patience=patience,
        verbose=verbose,
        save_checkpoints=save_checkpoints,
    )
