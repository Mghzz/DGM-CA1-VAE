"""Evaluation metrics for Phase 5: Reconstruction MSE, Unbiased RBF MMD², Cluster Coverage, and Mixture Assignments.

All functions are purely numerical, self-contained, and independently testable.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial.distance import cdist, pdist
import torch


def compute_reconstruction_mse(
    x: Union[torch.Tensor, np.ndarray],
    x_recon: Union[torch.Tensor, np.ndarray],
) -> Dict[str, float]:
    """Computes test reconstruction Mean Squared Error under both standard reporting conventions.
    
    Args:
        x: Ground-truth test samples of shape (N, D).
        x_recon: Reconstructed samples (decoder output on encoder posterior mean) of shape (N, D).
        
    Returns:
        Dict with keys:
            - 'mse_per_element': Mean squared error per dimension (sum_diff_sq / (N * D)).
            - 'mean_squared_l2_per_sample': Mean squared Euclidean error per sample (sum_diff_sq / N).
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x_recon, torch.Tensor):
        x_recon = x_recon.detach().cpu().numpy()

    if x.shape != x_recon.shape:
        raise ValueError(f"Shape mismatch: x has shape {x.shape} but x_recon has shape {x_recon.shape}")

    if not np.isfinite(x).all() or not np.isfinite(x_recon).all():
        raise ValueError("Input tensors contain non-finite (NaN or Inf) values.")

    diff_sq = (x - x_recon) ** 2
    mse_per_element = float(np.mean(diff_sq))
    mean_squared_l2_per_sample = float(np.mean(np.sum(diff_sq, axis=-1)))

    return {
        "mse_per_element": mse_per_element,
        "mean_squared_l2_per_sample": mean_squared_l2_per_sample,
    }


def compute_unbiased_rbf_mmd2(
    x: Union[torch.Tensor, np.ndarray],
    y: Union[torch.Tensor, np.ndarray],
    bandwidth: float,
) -> float:
    """Computes the exact one-sample U-statistic unbiased RBF MMD² estimator.
    
    Formula (Gretton et al., 2012):
        MMD_u^2(X, Y) = 1/(m(m-1)) * sum_{i!=j} k(x_i, x_j)
                      + 1/(n(n-1)) * sum_{i!=j} k(y_i, y_j)
                      - 2/(mn) * sum_{i,j} k(x_i, y_j)
                      
    where k(u, v) = exp(-||u - v||^2 / (2 * bandwidth^2)).
    
    IMPORTANT: Finite-sample estimates can be negative, including when X and Y are
    drawn from the same distribution or when X == Y. Negative values are strictly preserved
    and never clipped to zero.
    
    Args:
        x: Sample array of shape (m, d).
        y: Sample array of shape (n, d).
        bandwidth: Kernel bandwidth sigma > 0.
        
    Returns:
        Unbiased MMD² value (float).
    """
    if bandwidth <= 0:
        raise ValueError(f"Bandwidth must be strictly positive, got {bandwidth}")

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Inputs must be 2D arrays, got x shape {x.shape} and y shape {y.shape}")

    if x.shape[1] != y.shape[1]:
        raise ValueError(f"Feature dimension mismatch: x has {x.shape[1]} but y has {y.shape[1]}")

    m = x.shape[0]
    n = y.shape[0]

    if m < 2 or n < 2:
        raise ValueError(f"At least 2 samples per set required for unbiased MMD, got m={m}, n={n}")

    gamma = 1.0 / (2.0 * bandwidth ** 2)

    # Compute pairwise squared Euclidean distances
    d_xx_sq = cdist(x, x, metric="sqeuclidean")
    d_yy_sq = cdist(y, y, metric="sqeuclidean")
    d_xy_sq = cdist(x, y, metric="sqeuclidean")

    k_xx = np.exp(-gamma * d_xx_sq)
    k_yy = np.exp(-gamma * d_yy_sq)
    k_xy = np.exp(-gamma * d_xy_sq)

    # Sum of off-diagonal elements (diag is 1.0 for RBF)
    sum_k_xx_offdiag = float(np.sum(k_xx) - np.trace(k_xx))
    sum_k_yy_offdiag = float(np.sum(k_yy) - np.trace(k_yy))
    sum_k_xy = float(np.sum(k_xy))

    term_xx = sum_k_xx_offdiag / (m * (m - 1))
    term_yy = sum_k_yy_offdiag / (n * (n - 1))
    term_xy = 2.0 * sum_k_xy / (m * n)

    mmd2 = term_xx + term_yy - term_xy
    return float(mmd2)


def compute_bandwidth_median_heuristic(
    samples: Union[torch.Tensor, np.ndarray],
    subsample_size: Optional[int] = 600,
) -> float:
    """Computes RBF kernel bandwidth using the median pairwise Euclidean distance heuristic.
    
    To guarantee strict reproducibility across systems, if len(samples) > subsample_size,
    the deterministic slice samples[0:subsample_size] is used.
    
    Args:
        samples: Array of shape (N, D).
        subsample_size: Number of samples to use for pairwise distance calculation (default 600).
        
    Returns:
        Median pairwise distance (float).
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()

    if samples.ndim != 2:
        raise ValueError(f"Samples must be 2D array, got shape {samples.shape}")

    if subsample_size is not None and len(samples) > subsample_size:
        sub_samples = samples[0:subsample_size]
    else:
        sub_samples = samples

    if len(sub_samples) < 2:
        raise ValueError(f"At least 2 samples required to compute pairwise distances, got {len(sub_samples)}")

    distances = pdist(sub_samples, metric="euclidean")
    median_dist = float(np.median(distances))

    if median_dist <= 0:
        raise ValueError(f"Computed median distance is non-positive ({median_dist})!")

    return median_dist


def compute_cluster_reference_regions(
    val_x: Union[torch.Tensor, np.ndarray],
    val_y: Union[torch.Tensor, np.ndarray],
    n_clusters: int = 8,
    percentile: float = 95.0,
) -> Dict[str, Any]:
    """Calculates cluster centroids and 95th-percentile acceptance radii from validation data in 20D space.
    
    Args:
        val_x: Validation feature vectors of shape (N_val, 20).
        val_y: Validation cluster labels of shape (N_val,).
        n_clusters: Expected number of clusters (default 8).
        percentile: Acceptance radius percentile (default 95.0).
        
    Returns:
        Dict containing:
            - 'centroids': np.ndarray of shape (n_clusters, D).
            - 'radii': np.ndarray of shape (n_clusters,).
            - 'cluster_counts': Dict[int, int] sample count per cluster.
    """
    if isinstance(val_x, torch.Tensor):
        val_x = val_x.detach().cpu().numpy()
    if isinstance(val_y, torch.Tensor):
        val_y = val_y.detach().cpu().numpy()

    d = val_x.shape[1]
    centroids = np.zeros((n_clusters, d), dtype=np.float32)
    radii = np.zeros(n_clusters, dtype=np.float32)
    cluster_counts = {}

    for k in range(n_clusters):
        mask = (val_y == k)
        cluster_points = val_x[mask]
        n_k = len(cluster_points)
        if n_k == 0:
            raise ValueError(f"Validation cluster {k} has 0 samples!")

        cluster_counts[k] = n_k
        c_k = np.mean(cluster_points, axis=0)
        centroids[k] = c_k

        # Euclidean distances to centroid
        dists = np.linalg.norm(cluster_points - c_k, axis=1)
        # Direct empirical percentile calculation
        radii[k] = float(np.percentile(dists, percentile, method="linear"))

    return {
        "centroids": centroids,
        "radii": radii,
        "cluster_counts": cluster_counts,
        "percentile": percentile,
    }


def evaluate_cluster_coverage(
    gen_x: Union[torch.Tensor, np.ndarray],
    centroids: Union[torch.Tensor, np.ndarray],
    radii: Union[torch.Tensor, np.ndarray],
    min_samples_covered: int = 20,
) -> Dict[str, Any]:
    """Evaluates generated samples against reference cluster regions in 20D space.
    
    Validity rule:
        Each generated sample is assigned to its nearest centroid k*.
        The sample is valid if and only if ||x_gen - c_{k*}|| <= R_{k*}.
        
    Coverage rule:
        A cluster is considered covered if at least min_samples_covered (default 20)
        valid generated samples are assigned to it.
        
    Args:
        gen_x: Generated sample array of shape (N_gen, 20).
        centroids: Cluster centroids of shape (n_clusters, 20).
        radii: Cluster acceptance radii of shape (n_clusters,).
        min_samples_covered: Threshold for cluster coverage (default 20).
        
    Returns:
        Dict containing validity and coverage statistics.
    """
    if isinstance(gen_x, torch.Tensor):
        gen_x = gen_x.detach().cpu().numpy()
    if isinstance(centroids, torch.Tensor):
        centroids = centroids.detach().cpu().numpy()
    if isinstance(radii, torch.Tensor):
        radii = radii.detach().cpu().numpy()

    n_gen = len(gen_x)
    n_clusters = len(centroids)

    # Compute distances from all generated samples to all centroids: shape (N_gen, n_clusters)
    dist_matrix = cdist(gen_x, centroids, metric="euclidean")

    nearest_clusters = np.argmin(dist_matrix, axis=1)  # shape (N_gen,)
    min_distances = np.min(dist_matrix, axis=1)        # shape (N_gen,)

    cluster_radii_matched = radii[nearest_clusters]
    is_valid = (min_distances <= cluster_radii_matched)

    valid_counts_per_cluster = {}
    total_assigned_per_cluster = {}
    covered_clusters = []

    for k in range(n_clusters):
        assigned_to_k = (nearest_clusters == k)
        total_assigned_per_cluster[k] = int(np.sum(assigned_to_k))
        valid_in_k = int(np.sum(assigned_to_k & is_valid))
        valid_counts_per_cluster[k] = valid_in_k

        if valid_in_k >= min_samples_covered:
            covered_clusters.append(k)

    total_valid = int(np.sum(is_valid))
    validity_rate_pct = float(100.0 * total_valid / n_gen) if n_gen > 0 else 0.0

    return {
        "validity_rate_pct": round(validity_rate_pct, 2),
        "total_valid_samples": total_valid,
        "total_generated_samples": n_gen,
        "covered_clusters_count": len(covered_clusters),
        "total_clusters": n_clusters,
        "covered_clusters": sorted(covered_clusters),
        "valid_counts_per_cluster": valid_counts_per_cluster,
        "total_assigned_per_cluster": total_assigned_per_cluster,
        "min_samples_covered_threshold": min_samples_covered,
    }


def compute_mixture_component_assignments(
    z: Union[torch.Tensor, np.ndarray],
    centers: Union[torch.Tensor, np.ndarray],
    log_stds: Union[torch.Tensor, np.ndarray],
    weights: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Computes posterior responsibilities and hard component assignments for mixture priors.
    
    For latent samples z_i in R^2 and K Gaussian components with centers mu_k and diagonal stds sigma_k:
        log p(z_i | k) = -log(2*pi) - sum_d log sigma_{kd} - 0.5 * sum_d ((z_{id} - mu_{kd}) / sigma_{kd})^2
        gamma_{ik} = softmax_k(log pi_k + log p(z_i | k))
        
    Under equal weights (pi_k = 1/K), argmax_k gamma_{ik} == argmax_k log p(z_i | k).
    
    Args:
        z: Latent samples of shape (N, 2).
        centers: Component centers of shape (K, 2).
        log_stds: Component diagonal log standard deviations of shape (K, 2).
        weights: Component weights of shape (K,). Defaults to uniform (1/K).
        
    Returns:
        Dict containing hard assignments, hard proportions, and mean responsibilities.
    """
    if isinstance(z, np.ndarray):
        z = torch.from_numpy(z).float()
    if isinstance(centers, np.ndarray):
        centers = torch.from_numpy(centers).float()
    if isinstance(log_stds, np.ndarray):
        log_stds = torch.from_numpy(log_stds).float()

    n_samples, latent_dim = z.shape
    n_components, _ = centers.shape

    if weights is None:
        weights = torch.full((n_components,), 1.0 / n_components, dtype=torch.float32)
    elif isinstance(weights, np.ndarray):
        weights = torch.from_numpy(weights).float()

    log_weights = torch.log(weights).unsqueeze(0)  # (1, K)

    # z: (N, 1, 2), centers: (1, K, 2), log_stds: (1, K, 2)
    z_exp = z.unsqueeze(1)
    c_exp = centers.unsqueeze(0)
    ls_exp = log_stds.unsqueeze(0)

    # Quadratic term: sum_d ((z - mu) / sigma)^2
    stds_exp = torch.exp(ls_exp)
    diff_norm_sq = torch.sum(((z_exp - c_exp) / stds_exp) ** 2, dim=-1)  # (N, K)

    log_det = torch.sum(ls_exp, dim=-1)  # (1, K)
    log_2pi = float(np.log(2.0 * np.pi))

    # Log-likelihood per component: (N, K)
    log_p_z_k = -0.5 * (latent_dim * log_2pi + 2.0 * log_det + diff_norm_sq)

    # Joint log prob: log(pi_k) + log p(z | k)
    log_joint = log_weights + log_p_z_k

    # Posterior responsibilities gamma_ik: (N, K)
    responsibilities = torch.softmax(log_joint, dim=-1)

    # Hard assignment: argmax component
    hard_assignments = torch.argmax(responsibilities, dim=-1).cpu().numpy()
    resp_np = responsibilities.cpu().numpy()

    hard_proportions = {}
    mean_responsibilities = {}

    for k in range(n_components):
        hard_proportions[k] = float(np.mean(hard_assignments == k))
        mean_responsibilities[k] = float(np.mean(resp_np[:, k]))

    return {
        "hard_assignments": hard_assignments,
        "hard_proportions": hard_proportions,
        "mean_responsibilities": mean_responsibilities,
        "responsibilities": resp_np,
    }
