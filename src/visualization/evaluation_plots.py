"""Plotting and visualization engine for Phase 5: Learning Curves, PCA, Latent Spaces, and Cluster Coverage.

All functions are strictly reproducible, self-contained, and independently testable.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch

from src.models.vae import VAE


def plot_learning_curves(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: Optional[Path] = None,
    required_models: Optional[Tuple[str, ...]] = ("A", "B", "C", "D"),
) -> plt.Figure:
    """Plots training and validation learning curves (Total, Recon, KL) across all four models.
    
    Validates that all required models exist in histories and contain expected non-empty curves.
    
    Args:
        histories: Dict mapping model_type ('A', 'B', 'C', 'D') to its history dictionary
                   containing 'train_total_loss', 'val_total_loss', etc.
        save_path: Optional destination Path for saving the figure.
        required_models: Tuple of model types required to be present (default ('A', 'B', 'C', 'D')).
                         Pass None or empty tuple to skip strict model presence check.
                         
    Returns:
        Matplotlib Figure object.
    """
    if required_models:
        for m in required_models:
            if m not in histories:
                raise ValueError(
                    f"plot_learning_curves validation error: Missing history for Model {m}! "
                    f"Present models: {list(histories.keys())}"
                )

    expected_curve_keys = [
        "train_total_loss",
        "val_total_loss",
        "train_recon_loss",
        "val_recon_loss",
        "train_kl_loss",
        "val_kl_loss",
    ]

    for m_type, hist in histories.items():
        for k in expected_curve_keys:
            if k not in hist:
                raise ValueError(
                    f"Model {m_type} history is missing required curve '{k}'!"
                )
            if not isinstance(hist[k], (list, np.ndarray)) or len(hist[k]) == 0:
                raise ValueError(
                    f"Model {m_type} history curve '{k}' must be a non-empty sequence!"
                )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728"}

    metrics = [
        ("total_loss", "Total Loss (ELBO NLL)"),
        ("recon_loss", "Reconstruction NLL (sigma_x=0.30)"),
        ("kl_loss", "KL Divergence (Analytical / MC)"),
    ]

    for ax, (metric_key, title) in zip(axes, metrics):
        for m_type, hist in histories.items():
            epochs = hist.get("epoch", list(range(1, len(hist[f"train_{metric_key}"]) + 1)))
            train_vals = hist[f"train_{metric_key}"]
            val_vals = hist[f"val_{metric_key}"]
            c = colors.get(m_type, "black")

            ax.plot(epochs, train_vals, label=f"Model {m_type} (Train)", color=c, linestyle="-", alpha=0.8, linewidth=1.5)
            ax.plot(epochs, val_vals, label=f"Model {m_type} (Val)", color=c, linestyle="--", alpha=0.9, linewidth=1.5)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss per sample", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)

    # Consolidate legend on middle plot
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_pca_comparison(
    pca: Any,
    x_test: Union[torch.Tensor, np.ndarray],
    x_gen: Union[torch.Tensor, np.ndarray],
    model_name: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Projects real test samples and generated samples onto Phase 1 canonical PCA coordinates.
    
    IMPORTANT: PCA object is NEVER refitted. pca.transform is applied identically to both sets.
    
    Args:
        pca: Pre-fitted sklearn PCA object from Phase 1.
        x_test: Real test data array of shape (N_test, 20).
        x_gen: Generated data array of shape (N_gen, 20).
        model_name: Model identifier ('A', 'B', 'C', 'D').
        save_path: Destination path for saving PNG.
        
    Returns:
        Matplotlib Figure object.
    """
    if isinstance(x_test, torch.Tensor):
        x_test = x_test.detach().cpu().numpy()
    if isinstance(x_gen, torch.Tensor):
        x_gen = x_gen.detach().cpu().numpy()

    # Transform without refitting
    test_pca = pca.transform(x_test)
    gen_pca = pca.transform(x_gen)

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150, sharex=True, sharey=True)

    # Subplot 1: Real Test Data
    axes[0].scatter(test_pca[:, 0], test_pca[:, 1], c="#1f77b4", alpha=0.5, s=15, edgecolors="none")
    axes[0].set_title(f"Real Test Data (N={len(test_pca)})", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=11)
    axes[0].set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.5)

    # Subplot 2: Generated Data
    axes[1].scatter(gen_pca[:, 0], gen_pca[:, 1], c="#ff7f0e", alpha=0.5, s=15, edgecolors="none")
    axes[1].set_title(f"Model {model_name} Generated Samples (N={len(gen_pca)})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=11)
    axes[1].set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.5)

    fig.suptitle(f"Model {model_name} — Canonical 2D PCA Projection (Phase 1 Coordinates)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_latent_space(
    z_test: Union[torch.Tensor, np.ndarray],
    z_prior: Union[torch.Tensor, np.ndarray],
    model: VAE,
    model_name: str,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plots 2D latent representations: encoded test posterior means vs prior samples.
    
    For mixture priors (B, C, D), also renders:
    - Component centers mu_k with numeric labels.
    - 1-sigma solid covariance contour (39.3% mass).
    - 95% confidence dashed covariance contour (chi^2_2(0.95) = 5.991, s = 2.4477).
    
    Args:
        z_test: Encoded test means mu_z(x) of shape (N_test, 2).
        z_prior: Prior samples z ~ p(z) of shape (N_prior, 2).
        model: VAE instance (used to access prior centers and log_stds).
        model_name: Model identifier ('A', 'B', 'C', 'D').
        save_path: Destination path for saving PNG.
        
    Returns:
        Matplotlib Figure object.
    """
    if isinstance(z_test, torch.Tensor):
        z_test = z_test.detach().cpu().numpy()
    if isinstance(z_prior, torch.Tensor):
        z_prior = z_prior.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    # 1. Scatter of prior samples and test encodings
    ax.scatter(z_prior[:, 0], z_prior[:, 1], c="#ff7f0e", alpha=0.35, s=12, label="Prior Samples $z \\sim p(z)$", edgecolors="none")
    ax.scatter(z_test[:, 0], z_test[:, 1], c="#1f77b4", alpha=0.45, s=14, label="Test Latent Encodings $\\mu_z(x)$", edgecolors="none")

    # 2. Render Component Structure for Mixture Models
    chi2_95_scale = np.sqrt(5.99146)  # ~ 2.4477 for 2D Gaussian 95% confidence

    if model_name in ["B", "C", "D"]:
        centers = model.prior.centers.detach().cpu().numpy()  # (K, 2)
        stds = np.exp(model.prior.log_stds.detach().cpu().numpy())  # (K, 2)

        for k in range(len(centers)):
            c_x, c_y = centers[k, 0], centers[k, 1]
            s_x, s_y = stds[k, 0], stds[k, 1]

            # 1-sigma contour (semi-axes: s_x, s_y -> width=2*s_x, height=2*s_y)
            ell_1sigma = Ellipse(
                xy=(c_x, c_y),
                width=2.0 * s_x,
                height=2.0 * s_y,
                angle=0.0,
                edgecolor="#d62728",
                linestyle="-",
                linewidth=1.5,
                fill=False,
                label="$1\\sigma$ Prior Contour" if k == 0 else "",
            )
            ax.add_patch(ell_1sigma)

            # 95% confidence contour (semi-axes: chi * s_x, chi * s_y)
            ell_95 = Ellipse(
                xy=(c_x, c_y),
                width=2.0 * chi2_95_scale * s_x,
                height=2.0 * chi2_95_scale * s_y,
                edgecolor="#d62728",
                linestyle="--",
                linewidth=1.2,
                fill=False,
                alpha=0.8,
                label="95% Conf Contour ($\\chi^2_2$)" if k == 0 else "",
            )
            ax.add_patch(ell_95)

            # Center marker and label
            ax.plot(c_x, c_y, marker="P", markersize=8, color="#9467bd", markeredgecolor="black")
            ax.text(c_x + 0.08, c_y + 0.08, f"$k_{k+1}$", fontsize=9, fontweight="bold", color="#333333")

    elif model_name == "A":
        # Standard normal prior: center at (0, 0), std = 1.0
        ell_1sigma = Ellipse(
            xy=(0, 0),
            width=2.0,
            height=2.0,
            angle=0.0,
            edgecolor="#d62728",
            linestyle="-",
            linewidth=1.5,
            fill=False,
            label="$1\\sigma$ Prior Contour",
        )
        ell_95 = Ellipse(
            xy=(0, 0),
            width=2.0 * chi2_95_scale,
            height=2.0 * chi2_95_scale,
            angle=0.0,
            edgecolor="#d62728",
            linestyle="--",
            linewidth=1.2,
            fill=False,
            alpha=0.8,
            label="95% Conf Contour ($\\chi^2_2$)",
        )
        ax.add_patch(ell_1sigma)
        ax.add_patch(ell_95)
        ax.plot(0, 0, marker="P", markersize=8, color="#9467bd", markeredgecolor="black")

    ax.set_title(f"Model {model_name} — 2D Latent Space & Prior Geometry", fontsize=13, fontweight="bold")
    ax.set_xlabel("$z_1$", fontsize=12)
    ax.set_ylabel("$z_2$", fontsize=12)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_cluster_coverage_bar(
    coverage_details: Dict[str, Dict[str, Any]],
    min_samples_threshold: int = 20,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Plots a grouped bar chart of valid generated sample counts per cluster across models.
    
    Includes horizontal threshold line at min_samples_threshold (20 samples).
    
    Args:
        coverage_details: Dict mapping model_type to its evaluate_cluster_coverage output dict.
        min_samples_threshold: Coverage threshold line value (default 20).
        save_path: Destination path for saving PNG.
        
    Returns:
        Matplotlib Figure object.
    """
    models = [m for m in ["A", "B", "C", "D"] if m in coverage_details]
    n_models = len(models)
    n_clusters = 8
    clusters = np.arange(n_clusters)

    bar_width = 0.8 / n_models
    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728"}

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    for idx, m_type in enumerate(models):
        res = coverage_details[m_type]
        valid_counts = [res["valid_counts_per_cluster"].get(k, res["valid_counts_per_cluster"].get(str(k), 0)) for k in range(n_clusters)]
        cov_count = res["covered_clusters_count"]
        val_pct = res["validity_rate_pct"]
        offset = (idx - (n_models - 1) / 2.0) * bar_width

        ax.bar(
            clusters + offset,
            valid_counts,
            width=bar_width,
            label=f"Model {m_type} ({cov_count}/8 covered, {val_pct}% valid)",
            color=colors.get(m_type, "gray"),
            alpha=0.85,
        )

    # Draw horizontal threshold line
    ax.axhline(
        y=min_samples_threshold,
        color="crimson",
        linestyle="--",
        linewidth=1.8,
        label=f"Coverage Threshold ({min_samples_threshold} valid samples)",
    )

    ax.set_title("20D Cluster Coverage: Valid Generated Sample Counts per Cluster (out of 2,000)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Cluster Index ($k \\in \\{0, \\dots, 7\\}$)", fontsize=11)
    ax.set_ylabel("Valid Generated Samples in Cluster Region", fontsize=11)
    ax.set_xticks(clusters)
    ax.set_xticklabels([f"Cluster {k}" for k in range(n_clusters)], fontsize=10)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    return fig
