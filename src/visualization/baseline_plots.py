"""Baseline visualization module for Stage 1.

Generates:
1. Ground-truth 2D source coordinates s colored by cluster labels.
2. 2D PCA projection of the 20D training observations x_train colored by y_train.
Saves the fitted PCA object for all future evaluations (validation, test, generated).
"""

import json
from pathlib import Path
from typing import Dict, Any
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.data.dataset_loader import load_dataset


def generate_baseline_visualizations(
    figures_dir: Path = None,
    metrics_dir: Path = None,
    checkpoints_dir: Path = None,
) -> Dict[str, Any]:
    """Generates and saves baseline visualizations and fits the canonical PCA model."""
    project_root = Path(__file__).resolve().parent.parent.parent

    if figures_dir is None:
        figures_dir = project_root / "results" / "figures"
    if metrics_dir is None:
        metrics_dir = project_root / "results" / "metrics"
    if checkpoints_dir is None:
        checkpoints_dir = project_root / "checkpoints"

    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    splits = load_dataset()

    # Set clean plotting style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "figure.titlesize": 14,
        "figure.autolayout": True,
    })

    # -------------------------------------------------------------
    # 1. Ground-truth 2D Source Coordinates Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    # Use discrete colormap for 8 clusters
    cmap = plt.get_cmap("tab10", 8)
    scatter = ax.scatter(
        splits.s_train[:, 0],
        splits.s_train[:, 1],
        c=splits.y_train,
        cmap=cmap,
        alpha=0.6,
        s=15,
        edgecolors="none",
    )

    # Mark the true geometric centers
    ax.scatter(
        splits.centers[:, 0],
        splits.centers[:, 1],
        c="black",
        marker="X",
        s=90,
        linewidths=1.5,
        label="True Cluster Centers ($R=4$)",
        zorder=5,
    )

    # Draw reference circle of radius 4
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(
        4 * np.cos(theta),
        4 * np.sin(theta),
        "k--",
        alpha=0.3,
        linewidth=1.2,
        label="Radius 4 Circle",
    )

    cbar = plt.colorbar(scatter, ax=ax, ticks=range(8))
    cbar.set_label("Cluster Label $y$", rotation=270, labelpad=15)

    ax.set_title("Ground-Truth 2D Source Coordinates ($s_{train}$)")
    ax.set_xlabel("$s_1$ (Source Coordinate 1)")
    ax.set_ylabel("$s_2$ (Source Coordinate 2)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    fig1_path = figures_dir / "source_2d_scatter.png"
    fig.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------
    # 2. PCA Projection of 20D Observations on Training Set
    # -------------------------------------------------------------
    pca = PCA(n_components=2, random_state=2026)
    x_train_pca = pca.fit_transform(splits.x_train)

    ev_ratio = pca.explained_variance_ratio_
    ev_total = float(ev_ratio.sum() * 100.0)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    scatter_pca = ax.scatter(
        x_train_pca[:, 0],
        x_train_pca[:, 1],
        c=splits.y_train,
        cmap=cmap,
        alpha=0.6,
        s=15,
        edgecolors="none",
    )

    cbar_pca = plt.colorbar(scatter_pca, ax=ax, ticks=range(8))
    cbar_pca.set_label("Cluster Label $y_{train}$", rotation=270, labelpad=15)

    ax.set_title("2D PCA Projection of 20D Observations ($x_{train}$)")
    ax.set_xlabel(f"PC 1 ({ev_ratio[0]*100:.2f}% Variance)")
    ax.set_ylabel(f"PC 2 ({ev_ratio[1]*100:.2f}% Variance)")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig2_path = figures_dir / "train_pca_2d_scatter.png"
    fig.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------
    # 3. Save Fitted PCA Model and Metrics
    # -------------------------------------------------------------
    pca_metrics_path = metrics_dir / "fitted_pca.joblib"
    pca_checkpoint_path = checkpoints_dir / "fitted_pca.joblib"
    joblib.dump(pca, pca_metrics_path)
    joblib.dump(pca, pca_checkpoint_path)

    summary = {
        "pca_components": 2,
        "explained_variance_ratio": [float(r) for r in ev_ratio],
        "total_explained_variance_pct": round(ev_total, 2),
        "source_figure_path": str(fig1_path),
        "pca_figure_path": str(fig2_path),
        "saved_pca_paths": [str(pca_metrics_path), str(pca_checkpoint_path)],
    }

    with open(metrics_dir / "pca_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    res = generate_baseline_visualizations()
    print("Baseline visualizations generated successfully:")
    print(json.dumps(res, indent=2))
