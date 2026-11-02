"""Dataset verification module.

Performs exhaustive verification of dataset dimensions, split sizes,
source coordinate alignment, cluster distributions, and numerical properties.
"""

import json
from pathlib import Path
from typing import Dict, Any
import numpy as np

from src.data.dataset_loader import load_dataset


def verify_dataset(save_metrics_path: Path = None) -> Dict[str, Any]:
    """Runs all dataset verification checks and returns the diagnostic report."""
    splits = load_dataset()

    # 1. Dimensions and split sizes
    n_train, dim_x = splits.x_train.shape
    n_val, _ = splits.x_val.shape
    n_test, _ = splits.x_test.shape
    n_total = n_train + n_val + n_test

    assert n_train == 6720, f"Expected 6720 train samples, got {n_train}"
    assert n_val == 1440, f"Expected 1440 val samples, got {n_val}"
    assert n_test == 1440, f"Expected 1440 test samples, got {n_test}"
    assert dim_x == 20, f"Expected 20 observation features, got {dim_x}"
    assert splits.s_train.shape[1] == 2, f"Expected 2D source coordinates, got {splits.s_train.shape[1]}"

    # 2. NaN and Inf checks
    for name, arr in [
        ("x_train", splits.x_train),
        ("x_val", splits.x_val),
        ("x_test", splits.x_test),
        ("s_train", splits.s_train),
        ("s_val", splits.s_val),
        ("s_test", splits.s_test),
    ]:
        assert not np.isnan(arr).any(), f"NaN detected in {name}"
        assert not np.isinf(arr).any(), f"Inf detected in {name}"

    # 3. Label distributions
    train_counts = {int(k): int(v) for k, v in zip(*np.unique(splits.y_train, return_counts=True))}
    val_counts = {int(k): int(v) for k, v in zip(*np.unique(splits.y_val, return_counts=True))}
    test_counts = {int(k): int(v) for k, v in zip(*np.unique(splits.y_test, return_counts=True))}

    n_clusters = len(train_counts)
    assert n_clusters == 8, f"Expected 8 clusters, got {n_clusters}"

    # 4. Alignment verification between s and labels
    # Each sample in cluster k should be within ~3-4 stds of centers[k]
    # cluster radius is 4, within-cluster std is 0.35
    s_perm = np.vstack([splits.s_train, splits.s_val, splits.s_test])
    y_perm = np.concatenate([splits.y_train, splits.y_val, splits.y_test])
    centers = splits.centers

    assigned_centers = centers[y_perm]
    distances_to_centers = np.linalg.norm(s_perm - assigned_centers, axis=1)

    max_dist = float(distances_to_centers.max())
    mean_dist = float(distances_to_centers.mean())
    # 99.9% of samples from 2D Gaussian with sigma=0.35 fall within r < 0.35 * sqrt(2 * ln(1000)) ~ 1.3
    # Max distance must be safely below 2.0 (cluster separation is 4 * sin(pi/8) ~ 3.06)
    assert max_dist < 2.0, f"Alignment check failed! Max distance to assigned center is {max_dist:.4f}"

    # 5. Normalization checks on x_train
    train_mean = splits.x_train.mean(axis=0)
    train_std = splits.x_train.std(axis=0)
    np.testing.assert_allclose(train_mean, 0.0, atol=1e-5, err_msg="x_train mean is not 0")
    np.testing.assert_allclose(train_std, 1.0, atol=1e-5, err_msg="x_train std is not 1")

    report = {
        "status": "PASSED",
        "split_sizes": {
            "train": n_train,
            "validation": n_val,
            "test": n_test,
            "total": n_total,
        },
        "dimensions": {
            "observation_dim": dim_x,
            "latent_source_dim": 2,
            "n_clusters": n_clusters,
        },
        "cluster_counts": {
            "train": train_counts,
            "validation": val_counts,
            "test": test_counts,
        },
        "source_alignment": {
            "mean_distance_to_center": round(mean_dist, 4),
            "max_distance_to_center": round(max_dist, 4),
            "theoretical_expected_distance": round(float(0.35 * np.sqrt(np.pi / 2)), 4),
            "alignment_valid": True,
        },
        "normalization": {
            "train_mean_abs_max": float(np.abs(train_mean).max()),
            "train_std_min": float(train_std.min()),
            "train_std_max": float(train_std.max()),
        },
    }

    if save_metrics_path is not None:
        save_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_metrics_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    metrics_path = Path(__file__).resolve().parent.parent.parent / "results" / "metrics" / "dataset_verification.json"
    rep = verify_dataset(metrics_path)
    print("Dataset verification passed successfully:")
    print(json.dumps(rep, indent=2))
