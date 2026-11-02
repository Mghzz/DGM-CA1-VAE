"""CLI runner for Phase 5: Evaluation and Analysis across Models A, B, C, and D.

Executes test evaluation, metric aggregation, persistence, and figure generation.
"""

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np

from configs.config import ProjectConfig
from src.evaluation.evaluator import Evaluator
from src.visualization.evaluation_plots import (
    plot_learning_curves,
    plot_pca_comparison,
    plot_latent_space,
    plot_cluster_coverage_bar,
)


def run_full_evaluation(
    config: Optional[ProjectConfig] = None,
    device: Optional[str] = None,
    models: List[str] = ("A", "B", "C", "D"),
    eval_seed: int = 42,
) -> Dict[str, Any]:
    """Executes full evaluation pipeline across all specified models.
    
    Args:
        config: Project configuration.
        device: Device to use for evaluation.
        models: List of model identifiers.
        eval_seed: Seed for reproducible generation.
        
    Returns:
        Summary dictionary containing metrics across all evaluated models.
    """
    config = config or ProjectConfig()
    evaluator = Evaluator(
        config=config,
        device=device,
        generation_seed=eval_seed,
        bandwidth_seed=42,
    )

    print("\n=======================================================")
    print(f"Launching Phase 5 Evaluation: Models {list(models)}")
    print(f"Device: {evaluator.device} | Evaluation Seed: {eval_seed}")
    print(f"Data Bandwidth: {evaluator.bandwidths['data_space_bandwidth']:.4f} | Latent Bandwidth: {evaluator.bandwidths['latent_space_bandwidth']:.4f}")
    print("=======================================================\n")

    summary_records = []
    coverage_details = {}
    component_details = {}
    internal_results = {}

    t_start = time.perf_counter()

    for m_type in models:
        print(f"Evaluating Model {m_type}...")
        res = evaluator.evaluate_model(m_type)
        summary_records.append(res["summary"])
        coverage_details[m_type] = res["coverage_metrics"]
        if res["component_metrics"] is not None:
            # Convert ndarrays to serializable structures
            comp = res["component_metrics"]
            component_details[m_type] = {
                "hard_proportions": comp["hard_proportions"],
                "mean_responsibilities": comp["mean_responsibilities"],
            }
        internal_results[m_type] = res

    elapsed_time = time.perf_counter() - t_start

    # 1. Save Evaluation Summary JSON & CSV
    metrics_dir = config.paths.metrics_dir
    figures_dir = config.paths.figures_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = metrics_dir / "evaluation_summary.json"
    summary_csv_path = metrics_dir / "evaluation_summary.csv"
    coverage_json_path = metrics_dir / "cluster_coverage_details.json"
    component_json_path = metrics_dir / "component_assignments.json"

    eval_summary = {
        "evaluation_seed": eval_seed,
        "device": str(evaluator.device),
        "total_evaluation_time_sec": round(elapsed_time, 2),
        "bandwidths": evaluator.bandwidths,
        "models": summary_records,
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    # Convert coverage details for JSON serialization including full reference regions metadata
    cov_serializable = {
        "reference_regions": {
            "percentile": float(evaluator.reference_regions["percentile"]),
            "percentile_method": "linear",
            "cluster_counts": {str(k): int(v) for k, v in evaluator.reference_regions["cluster_counts"].items()},
            "radii": {str(k): float(v) for k, v in enumerate(evaluator.reference_regions["radii"])},
            "centroids": {str(k): [float(x) for x in c] for k, c in enumerate(evaluator.reference_regions["centroids"])},
        },
        "model_coverage": {},
    }
    for m, c in coverage_details.items():
        cov_serializable["model_coverage"][m] = {
            "validity_rate_pct": c["validity_rate_pct"],
            "total_valid_samples": c["total_valid_samples"],
            "total_generated_samples": c["total_generated_samples"],
            "covered_clusters_count": c["covered_clusters_count"],
            "total_clusters": c["total_clusters"],
            "covered_clusters": c["covered_clusters"],
            "valid_counts_per_cluster": {str(k): v for k, v in c["valid_counts_per_cluster"].items()},
            "total_assigned_per_cluster": {str(k): v for k, v in c["total_assigned_per_cluster"].items()},
            "min_samples_covered_threshold": c["min_samples_covered_threshold"],
        }

    with open(coverage_json_path, "w", encoding="utf-8") as f:
        json.dump(cov_serializable, f, indent=2)

    with open(component_json_path, "w", encoding="utf-8") as f:
        json.dump(component_details, f, indent=2)

    # Save summary CSV
    csv_fields = [
        "model_type",
        "checkpoint_path",
        "checkpoint_epoch",
        "checkpoint_val_loss",
        "mse_per_element",
        "mean_squared_l2_per_sample",
        "data_mmd2",
        "latent_mmd2",
        "cluster_validity_rate_pct",
        "covered_clusters_count",
        "total_clusters",
    ]
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in summary_records:
            writer.writerow({k: r[k] for k in csv_fields})

    # 2. Generate and Save Visualizations
    print("\nGenerating figures...")

    # A. Learning curves (loaded from history files - strictly validated)
    histories = {}
    for m_type in models:
        hist_file = config.paths.histories_dir / f"model_{m_type}_history.json"
        if not hist_file.exists():
            raise FileNotFoundError(
                f"Required learning history file not found for Model {m_type} at {hist_file}!"
            )
        with open(hist_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            histories[m_type] = data.get("history", data)

    plot_learning_curves(
        histories=histories,
        save_path=figures_dir / "learning_curves_all_models.png",
        required_models=tuple(models),
    )

    # B. PCA test vs generated plots
    for m_type in models:
        res = internal_results[m_type]
        plot_pca_comparison(
            pca=evaluator.pca,
            x_test=res["x_test"],
            x_gen=res["x_gen"],
            model_name=m_type,
            save_path=figures_dir / f"pca_test_vs_generated_model_{m_type}.png",
        )

    # C. Latent space plots
    for m_type in models:
        res = internal_results[m_type]
        plot_latent_space(
            z_test=res["z_test"],
            z_prior=res["z_prior"],
            model=res["model"],
            model_name=m_type,
            save_path=figures_dir / f"latent_space_model_{m_type}.png",
        )

    # D. Cluster coverage comparison bar chart
    plot_cluster_coverage_bar(
        coverage_details=coverage_details,
        min_samples_threshold=20,
        save_path=figures_dir / "cluster_coverage_comparison.png",
    )

    print(f"\n=======================================================")
    print(f"Phase 5 Evaluation Completed in {elapsed_time:.2f}s")
    print(f"Summary saved to: {summary_json_path}")
    print(f"Figures saved to: {figures_dir}")
    print("=======================================================\n")

    return eval_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 5 Model Evaluation.")
    parser.add_argument("--device", type=str, default=None, help="Target device (e.g. cuda:0 or cpu).")
    parser.add_argument("--seed", type=int, default=42, help="Evaluation generation seed.")
    args = parser.parse_args()

    run_full_evaluation(device=args.device, eval_seed=args.seed)
