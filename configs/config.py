"""Global project configuration for VAE HW1 assignment."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple
import torch


@dataclass
class DatasetConfig:
    data_seed: int = 2026  # Mandated by dataset.py
    n_clusters: int = 8
    n_per_cluster: int = 1200
    total_samples: int = 9600
    input_dim: int = 20
    latent_dim: int = 2
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    cluster_radius: float = 4.0
    cluster_std: float = 0.35
    noise_std: float = 0.07


@dataclass
class ModelConfig:
    input_dim: int = 20
    latent_dim: int = 2
    hidden_dims: Tuple[int, int] = (128, 128)
    likelihood_sigma: float = 0.30  # Strictly fixed in all models
    gmm_covariance_type: str = "diagonal"  # Explicit decision: "diagonal" for Models C & D
    init_std_gmm: float = 0.40  # Initial std for Models C & D
    b_radius: float = 2.0  # Circle radius for Model B fixed centers
    b_cov: float = 0.16  # Fixed variance for Model B (std = 0.4)


@dataclass
class TrainingConfig:
    lr: float = 0.001
    batch_size: int = 256
    max_epochs: int = 100
    early_stopping_patience: int = 15
    model_seed: int = 42  # Experimental reproducibility choice
    val_eval_seed: int = 999  # Dedicated validation seed for isolated deterministic evaluation


@dataclass
class PathConfig:
    project_root: Path = Path(__file__).resolve().parent.parent
    checkpoints_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "checkpoints")
    results_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "results")
    figures_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "results" / "figures")
    metrics_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "results" / "metrics")
    histories_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "results" / "histories")
    experiments_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "results" / "experiments")


@dataclass
class ProjectConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
