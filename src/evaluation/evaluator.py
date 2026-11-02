"""Evaluator engine for Phase 5: Test Reconstruction, Sampling, MMD², and Cluster Coverage.

Strictly preserves existing checkpoints, histories, dataset, and Phase 1 PCA object.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import torch
import torch.nn as nn

from configs.config import ProjectConfig
from src.data.dataset_loader import get_dataloaders
from src.models.vae import VAE, build_vae
from src.training.checkpointing import load_checkpoint
from src.evaluation.metrics import (
    compute_reconstruction_mse,
    compute_unbiased_rbf_mmd2,
    compute_bandwidth_median_heuristic,
    compute_cluster_reference_regions,
    evaluate_cluster_coverage,
    compute_mixture_component_assignments,
)


class Evaluator:
    """Evaluation coordinator for the four trained VAE models."""

    def __init__(
        self,
        config: Optional[ProjectConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
        generation_seed: int = 42,
        bandwidth_seed: int = 42,
        eval_seed: Optional[int] = None,
    ) -> None:
        """Initializes the evaluator.
        
        Args:
            config: Project configuration.
            device: Evaluation device. Defaults to config.device.
            generation_seed: Seed for reproducible prior evaluation generation.
            bandwidth_seed: Fixed seed for bandwidth selection calculation (separated from generation).
            eval_seed: Optional legacy alias for generation_seed.
        """
        self.config = config or ProjectConfig()
        self.device = torch.device(device or self.config.device)
        self.generation_seed = eval_seed if eval_seed is not None else generation_seed
        self.bandwidth_seed = bandwidth_seed

        # Device-matched random generator for reproducible sampling
        # Note on PyTorch reproducibility:
        # torch.Generator stream algorithms on CPU and CUDA produce different sequences for the
        # same seed. Within the same device type (e.g. cuda:0), execution is strictly deterministic.
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(self.generation_seed)

        # Load canonical Phase 1 PCA object
        pca_path = self.config.paths.metrics_dir / "fitted_pca.joblib"
        if not pca_path.exists():
            pca_path = self.config.paths.checkpoints_dir / "fitted_pca.joblib"
        if not pca_path.exists():
            raise FileNotFoundError(f"Canonical fitted PCA object not found at {pca_path}")
        self.pca = joblib.load(pca_path)

        # Load dataset splits (reproducibility guaranteed)
        _, self.val_loader, self.test_loader, self.raw_data = get_dataloaders(
            batch_size=self.config.training.batch_size,
            seed=self.config.training.model_seed,
        )

        # Extract full test, validation, and training numpy arrays
        if hasattr(self.raw_data, "x_test"):
            self.x_test = self.raw_data.x_test  # (1440, 20)
            self.y_test = self.raw_data.y_test  # (1440,)
            self.x_val = self.raw_data.x_val    # (1440, 20)
            self.y_val = self.raw_data.y_val    # (1440,)
            self.x_train = self.raw_data.x_train  # (6720, 20)
        else:
            self.x_test = self.raw_data["x_test"]
            self.y_test = self.raw_data["y_test"]
            self.x_val = self.raw_data["x_val"]
            self.y_val = self.raw_data["y_val"]
            self.x_train = self.raw_data["x_train"]

        # Precompute reference cluster regions on validation set (20D space)
        self.reference_regions = compute_cluster_reference_regions(
            val_x=self.x_val,
            val_y=self.y_val,
            n_clusters=self.config.dataset.n_clusters,
            percentile=95.0,
        )

        # Precompute fixed bandwidths once for all models using full precision
        self.bandwidths = self._compute_fixed_bandwidths()

        # Mandatory: Load recorded best epochs from Phase 4 experiment summary
        self.best_epochs_record = self._load_recorded_best_epochs()

    def _compute_fixed_bandwidths(self) -> Dict[str, Any]:
        """Computes data and latent space bandwidths once using median heuristic in full precision."""
        # Data space: deterministic slice of first 600 training samples
        data_bw = compute_bandwidth_median_heuristic(self.x_train, subsample_size=600)

        # Latent space: 600 samples from 1000 standard normal draws with dedicated bandwidth_seed
        gen = torch.Generator(device="cpu").manual_seed(self.bandwidth_seed)
        z_norm_1000 = torch.randn(1000, 2, generator=gen).numpy()
        latent_bw = compute_bandwidth_median_heuristic(z_norm_1000, subsample_size=600)

        # Preserve full precision floats without rounding truncation
        bandwidth_info = {
            "data_space_bandwidth": float(data_bw),
            "latent_space_bandwidth": float(latent_bw),
            "data_subsample_size": 600,
            "latent_subsample_size": 600,
            "bandwidth_seed": self.bandwidth_seed,
            "subsample_rule": "Deterministic slice [0:600]",
        }

        # Save to disk
        bw_path = self.config.paths.metrics_dir / "evaluation_bandwidths.json"
        bw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(bw_path, "w", encoding="utf-8") as f:
            json.dump(bandwidth_info, f, indent=2)

        return bandwidth_info

    def _load_recorded_best_epochs(self) -> Dict[str, Dict[str, Any]]:
        """Loads best epoch metadata from experiment_summary.json. Mandatory requirement for evaluation."""
        summary_path = self.config.paths.experiments_dir / "experiment_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Mandatory experiment_summary.json not found at {summary_path}. "
                "Phase 4 training experiments must be completed before running evaluation."
            )

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        records = {}
        for m in summary.get("models", []):
            if "model_type" in m and "best_epoch" in m and "best_val_loss" in m:
                records[m["model_type"]] = {
                    "best_epoch": int(m["best_epoch"]),
                    "best_val_loss": float(m["best_val_loss"]),
                }

        for req_model in ("A", "B", "C", "D"):
            if req_model not in records:
                raise ValueError(
                    f"Mandatory experiment_summary.json is missing record for Model {req_model}! "
                    f"Found models: {list(records.keys())}"
                )

        return records

    def load_model(self, model_type: str) -> Tuple[VAE, Dict[str, Any]]:
        """Loads a model from its best checkpoint and verifies it matches experiment_summary.json.
        
        Args:
            model_type: One of 'A', 'B', 'C', 'D'.
            
        Returns:
            Tuple of (restored VAE instance in eval mode, checkpoint metadata dict).
        """
        model_type = model_type.upper()
        ckpt_path = self.config.paths.checkpoints_dir / f"model_{model_type}_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint for Model {model_type} does not exist at {ckpt_path}")

        model = build_vae(model_type, self.config).to(self.device)
        checkpoint = load_checkpoint(ckpt_path, model=model, device=self.device)

        # Mandatory verification against Phase 4 experiment summary record
        expected_ep = self.best_epochs_record[model_type]["best_epoch"]
        expected_loss = self.best_epochs_record[model_type]["best_val_loss"]

        if checkpoint["epoch"] != expected_ep:
            raise ValueError(
                f"Model {model_type} checkpoint epoch {checkpoint['epoch']} does not match "
                f"recorded best epoch {expected_ep} in experiment_summary.json!"
            )
        if not np.isclose(checkpoint["val_loss"], expected_loss, atol=1e-5):
            raise ValueError(
                f"Model {model_type} checkpoint val_loss {checkpoint['val_loss']} does not match "
                f"recorded best val_loss {expected_loss} in experiment_summary.json!"
            )

        model.eval()
        checkpoint_meta = {
            "checkpoint_path": str(ckpt_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_val_loss": float(checkpoint["val_loss"]),
        }
        return model, checkpoint_meta

    def evaluate_model(
        self,
        model_type: str,
        n_gen_samples: int = 2000,
        n_mmd_samples: int = 1440,
    ) -> Dict[str, Any]:
        """Runs the complete Phase 5 evaluation pipeline for a single model.
        
        Args:
            model_type: One of 'A', 'B', 'C', 'D'.
            n_gen_samples: Number of samples to generate from prior (default 2,000).
            n_mmd_samples: Number of samples to use for MMD comparisons (default 1,440).
            
        Returns:
            Dictionary containing all computed metrics, sample arrays, and metadata.
        """
        model, checkpoint_meta = self.load_model(model_type)

        # 1. Test Reconstruction MSE (Unsupervised: x -> mu_z -> mu_x)
        x_test_t = torch.from_numpy(self.x_test).float().to(self.device)
        with torch.no_grad():
            mu_z, _ = model.encoder(x_test_t)
            mu_x = model.decoder(mu_z)

        mse_metrics = compute_reconstruction_mse(x_test_t, mu_x)
        z_test_np = mu_z.cpu().numpy()
        x_recon_np = mu_x.cpu().numpy()

        # 2. Prior Sampling (2,000 samples, no observation noise)
        self.generator.manual_seed(self.generation_seed)
        with torch.no_grad():
            z_prior = model.prior.sample(n_gen_samples, generator=self.generator)
            x_gen = model.decoder(z_prior)

        z_prior_np = z_prior.cpu().numpy()
        x_gen_np = x_gen.cpu().numpy()

        # Save generated samples array
        gen_path = self.config.paths.metrics_dir / f"generated_samples_model_{model_type}.npz"
        np.savez_compressed(
            gen_path,
            z_prior=z_prior_np,
            x_gen=x_gen_np,
            generation_seed=self.generation_seed,
            model_type=model_type,
        )

        # 3. 20D Cluster Coverage and Validity (on 2,000 generated samples)
        coverage_metrics = evaluate_cluster_coverage(
            gen_x=x_gen_np,
            centroids=self.reference_regions["centroids"],
            radii=self.reference_regions["radii"],
            min_samples_covered=20,
        )

        # 4. Unbiased RBF MMD² (Equal sample sizes: 1,440 each) using full-precision bandwidths
        data_bw = float(self.bandwidths["data_space_bandwidth"])
        latent_bw = float(self.bandwidths["latent_space_bandwidth"])

        data_mmd2 = compute_unbiased_rbf_mmd2(
            x=self.x_test[0:n_mmd_samples],
            y=x_gen_np[0:n_mmd_samples],
            bandwidth=data_bw,
        )

        latent_mmd2 = compute_unbiased_rbf_mmd2(
            x=z_test_np[0:n_mmd_samples],
            y=z_prior_np[0:n_mmd_samples],
            bandwidth=latent_bw,
        )

        # 5. Mixture Component Assignments (Models B, C, D)
        component_metrics = None
        if model_type in ["B", "C", "D"]:
            component_metrics = compute_mixture_component_assignments(
                z=z_test_np,
                centers=model.prior.centers.detach().cpu().numpy(),
                log_stds=model.prior.log_stds.detach().cpu().numpy(),
                weights=model.prior.weights.detach().cpu().numpy(),
            )

        # Document exact deterministic sample-selection rules and counts
        sample_selection_rules = {
            "bandwidth_data_rule": "Deterministic slice x_train[0:600] (N=600 of 6720)",
            "bandwidth_latent_rule": f"Deterministic slice z[0:600] of 1000 draws from N(0, I_2) with CPU seed={self.bandwidth_seed}",
            "prior_sampling_rule": f"Direct decoder output on 2000 prior samples with device-matched generator seed={self.generation_seed}",
            "mmd_data_comparison_rule": f"Real test set x_test[0:{n_mmd_samples}] vs generated subset x_gen[0:{n_mmd_samples}] (equal sizes m=n={n_mmd_samples})",
            "mmd_latent_comparison_rule": f"Test encodings mu_z(x_test)[0:{n_mmd_samples}] vs prior sample subset z_prior[0:{n_mmd_samples}] (equal sizes m=n={n_mmd_samples})",
            "cluster_coverage_rule": "All 2000 generated samples evaluated against validation centroids with 95th-percentile radius; covered if count >= 20",
            "n_train_total": len(self.x_train),
            "n_val_total": len(self.x_val),
            "n_test_total": len(self.x_test),
            "n_gen_samples": n_gen_samples,
            "n_mmd_samples": n_mmd_samples,
            "bandwidth_subsample_size": 600,
        }

        model_results = {
            "model_type": model_type,
            "checkpoint_path": checkpoint_meta["checkpoint_path"],
            "checkpoint_epoch": checkpoint_meta["checkpoint_epoch"],
            "checkpoint_val_loss": checkpoint_meta["checkpoint_val_loss"],
            "mse_per_element": mse_metrics["mse_per_element"],
            "mean_squared_l2_per_sample": mse_metrics["mean_squared_l2_per_sample"],
            "data_mmd2": data_mmd2,
            "latent_mmd2": latent_mmd2,
            "data_space_bandwidth": data_bw,
            "latent_space_bandwidth": latent_bw,
            "cluster_validity_rate_pct": coverage_metrics["validity_rate_pct"],
            "covered_clusters_count": coverage_metrics["covered_clusters_count"],
            "total_clusters": coverage_metrics["total_clusters"],
            "covered_clusters": coverage_metrics["covered_clusters"],
            "valid_samples_per_cluster": coverage_metrics["valid_counts_per_cluster"],
            "n_gen_samples": n_gen_samples,
            "n_mmd_samples": n_mmd_samples,
            "generation_seed": self.generation_seed,
            "bandwidth_seed": self.bandwidth_seed,
            "device": str(self.device),
            "generated_samples_path": str(gen_path),
            "sample_selection_rules": sample_selection_rules,
        }

        # Pack internal objects for visualization
        internal_payload = {
            "model": model,
            "checkpoint_meta": checkpoint_meta,
            "z_test": z_test_np,
            "z_prior": z_prior_np,
            "x_test": self.x_test,
            "x_gen": x_gen_np,
            "x_recon": x_recon_np,
            "coverage_metrics": coverage_metrics,
            "component_metrics": component_metrics,
            "summary": model_results,
        }

        return internal_payload
