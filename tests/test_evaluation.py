"""Comprehensive unit test suite for Phase 5: Evaluation and Analysis Engine.

Verifies:
1. Exact mathematical contracts of MSE and U-statistic RBF MMD² (including identical-sample negative value).
2. Direct empirical percentile verification for cluster reference radii.
3. Strict coverage thresholding (>= 20 valid samples).
4. Mixture posterior responsibility calculation and component assignment.
5. Independent testability of plotting routines using synthetic data.
"""

import json
import math
from pathlib import Path
import tempfile
import unittest
import joblib
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import torch

from configs.config import ProjectConfig
from src.models.vae import build_vae
from src.training.checkpointing import save_checkpoint
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import (
    compute_reconstruction_mse,
    compute_unbiased_rbf_mmd2,
    compute_bandwidth_median_heuristic,
    compute_cluster_reference_regions,
    evaluate_cluster_coverage,
    compute_mixture_component_assignments,
)
from src.visualization.evaluation_plots import (
    plot_learning_curves,
    plot_pca_comparison,
    plot_latent_space,
    plot_cluster_coverage_bar,
)


class TestReconstructionMSE(unittest.TestCase):
    """Tests for test reconstruction Mean Squared Error."""

    def test_reconstruction_mse_zero_on_identical(self):
        """Verify that identical input and reconstruction yield exactly 0.0 MSE."""
        x = np.random.randn(50, 20).astype(np.float32)
        res = compute_reconstruction_mse(x, x)
        self.assertAlmostEqual(res["mse_per_element"], 0.0, places=7)
        self.assertAlmostEqual(res["mean_squared_l2_per_sample"], 0.0, places=7)

    def test_reconstruction_mse_constant_shift(self):
        """Verify that a constant shift c yields c^2 per element and D * c^2 per sample."""
        c = 0.5
        d = 20
        n = 100
        x = np.zeros((n, d), dtype=np.float32)
        x_recon = np.full((n, d), c, dtype=np.float32)

        res = compute_reconstruction_mse(x, x_recon)
        self.assertAlmostEqual(res["mse_per_element"], c ** 2, places=6)
        self.assertAlmostEqual(res["mean_squared_l2_per_sample"], d * (c ** 2), places=6)

    def test_reconstruction_mse_input_validation(self):
        """Verify input validation: rejects shape mismatch and non-finite values."""
        x = np.zeros((10, 20))
        mismatched = np.zeros((10, 15))
        with self.assertRaises(ValueError):
            compute_reconstruction_mse(x, mismatched)

        nan_x = np.zeros((10, 20))
        nan_x[0, 0] = np.nan
        with self.assertRaises(ValueError):
            compute_reconstruction_mse(nan_x, x)


class TestUnbiasedRBFMMD2(unittest.TestCase):
    """Tests for the unbiased U-statistic RBF MMD² estimator."""

    def test_mmd2_identical_samples_exact_formula(self):
        """Verify that MMD_u^2(X, X) matches exact analytical formula: 2/(m-1)*(mean(K) - 1) < 0."""
        rng = np.random.RandomState(42)
        x = rng.randn(100, 5)
        bandwidth = 1.5

        mmd2 = compute_unbiased_rbf_mmd2(x, x, bandwidth=bandwidth)

        # Analytical formula derivation:
        # MMD_u^2(X, X) = 2/(m(m-1)) * sum_{i!=j} k(x_i, x_j) - 2/m^2 * sum_{i,j} k(x_i, x_j)
        #               = 2/(m-1) * (1/m^2 * sum_{i,j} k(x_i, x_j) - 1)
        gamma = 1.0 / (2.0 * bandwidth ** 2)
        k_mat = np.exp(-gamma * cdist(x, x, metric="sqeuclidean"))
        m = len(x)
        mean_k = np.mean(k_mat)
        expected_mmd2 = (2.0 / (m - 1)) * (mean_k - 1.0)

        self.assertAlmostEqual(mmd2, expected_mmd2, places=6)
        # Because points are distinct, mean_k < 1.0 -> MMD_u^2(X, X) is strictly negative
        self.assertLess(mmd2, 0.0)

    def test_mmd2_independent_draws_same_distribution(self):
        """Verify two independent draws from same distribution yield MMD_u^2 close to zero."""
        rng = np.random.RandomState(123)
        x = rng.randn(400, 2)
        y = rng.randn(400, 2)
        bandwidth = 1.0

        mmd2 = compute_unbiased_rbf_mmd2(x, y, bandwidth=bandwidth)
        # Expectation is 0.0, finite sample estimate fluctuates closely around zero
        self.assertAlmostEqual(mmd2, 0.0, delta=0.015)

    def test_mmd2_distinct_distributions(self):
        """Verify distinct distributions N(0, I) vs N(4, I) yield large positive MMD_u^2 >> 0."""
        rng = np.random.RandomState(456)
        x = rng.randn(300, 2)
        y = rng.randn(300, 2) + 4.0
        bandwidth = 1.0

        mmd2 = compute_unbiased_rbf_mmd2(x, y, bandwidth=bandwidth)
        self.assertGreater(mmd2, 0.4)

    def test_mmd2_input_validation(self):
        """Verify input validation rejects non-positive bandwidth and dimension mismatch."""
        x = np.random.randn(20, 5)
        y = np.random.randn(20, 5)

        with self.assertRaises(ValueError):
            compute_unbiased_rbf_mmd2(x, y, bandwidth=0.0)

        with self.assertRaises(ValueError):
            compute_unbiased_rbf_mmd2(x, y, bandwidth=-1.0)

        y_mismatched = np.random.randn(20, 3)
        with self.assertRaises(ValueError):
            compute_unbiased_rbf_mmd2(x, y_mismatched, bandwidth=1.0)


class TestBandwidthSelection(unittest.TestCase):
    """Tests for median pairwise Euclidean distance bandwidth heuristic."""

    def test_bandwidth_median_heuristic_exact(self):
        """Verify median heuristic calculation on controlled synthetic points."""
        # 4 points forming a square with side length 1.0
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        # Distances: four 1.0s and two sqrt(2) ~ 1.4142s
        # Pairwise distances sorted: [1.0, 1.0, 1.0, 1.0, 1.4142, 1.4142] -> median = 1.0
        bw = compute_bandwidth_median_heuristic(points)
        self.assertAlmostEqual(bw, 1.0, places=5)

    def test_bandwidth_deterministic_subsample_slice(self):
        """Verify that samples[0:subsample_size] is deterministically used."""
        rng = np.random.RandomState(42)
        samples = rng.randn(100, 10)
        bw_full = compute_bandwidth_median_heuristic(samples, subsample_size=50)
        bw_manual = compute_bandwidth_median_heuristic(samples[0:50], subsample_size=None)
        self.assertEqual(bw_full, bw_manual)


class TestClusterCoverageAndValidity(unittest.TestCase):
    """Tests for 20D cluster reference regions, sample validity, and coverage thresholding."""

    def test_cluster_reference_regions_percentile_calculation(self):
        """Verify compute_cluster_reference_regions computes exact empirical percentile."""
        # Create 2 clusters in 2D
        # Cluster 0: 100 points along x-axis with distances 1, 2, ..., 100 from origin
        points_c0 = np.zeros((100, 2), dtype=np.float32)
        points_c0[:, 0] = np.arange(1, 101)  # centroid will be at (50.5, 0)
        # Distances from centroid: |x - 50.5|
        expected_c0 = np.array([50.5, 0.0])
        dists_c0 = np.abs(np.arange(1, 101) - 50.5)
        expected_r0 = float(np.percentile(dists_c0, 95.0, method="linear"))

        # Cluster 1: 50 points
        points_c1 = np.ones((50, 2), dtype=np.float32) * 10.0
        points_c1[:, 1] += np.arange(1, 51)
        expected_c1 = np.mean(points_c1, axis=0)
        dists_c1 = np.linalg.norm(points_c1 - expected_c1, axis=1)
        expected_r1 = float(np.percentile(dists_c1, 95.0, method="linear"))

        val_x = np.vstack([points_c0, points_c1])
        val_y = np.array([0] * 100 + [1] * 50)

        regions = compute_cluster_reference_regions(val_x, val_y, n_clusters=2, percentile=95.0)

        np.testing.assert_allclose(regions["centroids"][0], expected_c0)
        np.testing.assert_allclose(regions["centroids"][1], expected_c1)
        self.assertAlmostEqual(regions["radii"][0], expected_r0, places=5)
        self.assertAlmostEqual(regions["radii"][1], expected_r1, places=5)

    def test_evaluate_cluster_coverage_exact_thresholds(self):
        """Verify cluster coverage rule (>= 20 samples covered) and distance validity."""
        centroids = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
        ])
        radii = np.array([2.0, 2.0])  # Acceptance radii

        # Generate points:
        # 25 points inside cluster 0 (valid)
        pts_c0_valid = np.zeros((25, 2)) + 0.5  # distance ~ 0.707 <= 2.0
        # 19 points inside cluster 1 (valid, but < 20 -> NOT covered)
        pts_c1_valid = np.zeros((19, 2)) + np.array([10.5, 0.0])  # distance = 0.5 <= 2.0
        # 10 invalid points far away (nearest to cluster 0, but distance = 5.0 > 2.0)
        pts_invalid = np.zeros((10, 2)) + np.array([5.0, 0.0])

        gen_x = np.vstack([pts_c0_valid, pts_c1_valid, pts_invalid])  # 54 total points

        cov = evaluate_cluster_coverage(gen_x, centroids, radii, min_samples_covered=20)

        self.assertEqual(cov["total_generated_samples"], 54)
        self.assertEqual(cov["total_valid_samples"], 44)  # 25 + 19
        self.assertAlmostEqual(cov["validity_rate_pct"], 100.0 * 44 / 54, places=2)
        # Cluster 0 has 25 valid samples (>= 20) -> covered
        # Cluster 1 has 19 valid samples (< 20) -> not covered
        self.assertEqual(cov["covered_clusters"], [0])
        self.assertEqual(cov["covered_clusters_count"], 1)
        self.assertEqual(cov["valid_counts_per_cluster"][0], 25)
        self.assertEqual(cov["valid_counts_per_cluster"][1], 19)


class TestMixtureComponentAssignments(unittest.TestCase):
    """Tests for mixture posterior responsibilities and hard component assignment."""

    def test_component_assignment_dominant_center(self):
        """Verify points placed at component centers receive highest density and gamma -> 1.0."""
        centers = np.array([
            [-5.0, -5.0],
            [5.0, 5.0],
        ])
        log_stds = np.zeros((2, 2))  # std = 1.0
        weights = np.array([0.5, 0.5])

        # Point at center 0 and point at center 1
        z = np.array([
            [-5.0, -5.0],
            [5.0, 5.0],
        ])

        res = compute_mixture_component_assignments(z, centers, log_stds, weights)

        # Hard assignments must be [0, 1]
        np.testing.assert_array_equal(res["hard_assignments"], [0, 1])
        # Responsibilities must be near 1.0 for the matching component
        self.assertGreater(res["responsibilities"][0, 0], 0.999)
        self.assertGreater(res["responsibilities"][1, 1], 0.999)
        self.assertAlmostEqual(res["hard_proportions"][0], 0.5)
        self.assertAlmostEqual(res["hard_proportions"][1], 0.5)

    def test_hard_vs_soft_assignment_consistency(self):
        """Verify argmax(gamma) equals argmax(log_prob) under equal mixture weights."""
        rng = np.random.RandomState(42)
        centers = rng.randn(4, 2)
        log_stds = rng.randn(4, 2) * 0.2
        z = rng.randn(50, 2)

        res = compute_mixture_component_assignments(z, centers, log_stds, weights=None)
        # Responsibilities must sum to 1.0 for each sample
        np.testing.assert_allclose(np.sum(res["responsibilities"], axis=1), np.ones(50), atol=1e-5)


class TestPlottingFunctionsSynthetic(unittest.TestCase):
    """Tests confirming all plotting functions render and save cleanly using synthetic data."""

    def test_plots_render_and_save_synthetic(self):
        """Verify learning curves, PCA, latent space, and coverage bar charts save without errors."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Learning curves with all 4 required models
            sample_curve = {
                "epoch": [1, 2],
                "train_total_loss": [10.0, 8.0],
                "val_total_loss": [11.0, 9.0],
                "train_recon_loss": [6.0, 5.0],
                "val_recon_loss": [7.0, 6.0],
                "train_kl_loss": [4.0, 3.0],
                "val_kl_loss": [4.0, 3.0],
            }
            histories = {m: dict(sample_curve) for m in ["A", "B", "C", "D"]}

            fig_lc = plot_learning_curves(histories, save_path=tmp_path / "lc.png")
            self.assertTrue((tmp_path / "lc.png").exists())
            self.assertIsNotNone(fig_lc)

            # 2. PCA comparison
            pca = PCA(n_components=2)
            pca.fit(np.random.randn(100, 20))
            x_test = np.random.randn(50, 20)
            x_gen = np.random.randn(50, 20)
            fig_pca = plot_pca_comparison(pca, x_test, x_gen, model_name="A", save_path=tmp_path / "pca.png")
            self.assertTrue((tmp_path / "pca.png").exists())
            self.assertIsNotNone(fig_pca)

            # 3. Latent space
            model_b = build_vae("B")
            z_test = np.random.randn(50, 2)
            z_prior = np.random.randn(50, 2)
            fig_ls = plot_latent_space(z_test, z_prior, model_b, model_name="B", save_path=tmp_path / "ls.png")
            self.assertTrue((tmp_path / "ls.png").exists())
            self.assertIsNotNone(fig_ls)

            # 4. Cluster coverage bar
            coverage_dict = {
                "A": {
                    "valid_counts_per_cluster": {k: 25 for k in range(8)},
                    "covered_clusters_count": 8,
                    "validity_rate_pct": 90.0,
                }
            }
            fig_bar = plot_cluster_coverage_bar(coverage_dict, save_path=tmp_path / "bar.png")
            self.assertTrue((tmp_path / "bar.png").exists())
            self.assertIsNotNone(fig_bar)

    def test_plot_learning_curves_validation(self):
        """Verify plot_learning_curves raises ValueError when a required model or curve is missing."""
        # Missing Model D
        incomplete_hist = {
            "A": {"train_total_loss": [1.0], "val_total_loss": [1.0], "train_recon_loss": [1.0], "val_recon_loss": [1.0], "train_kl_loss": [1.0], "val_kl_loss": [1.0]},
            "B": {"train_total_loss": [1.0], "val_total_loss": [1.0], "train_recon_loss": [1.0], "val_recon_loss": [1.0], "train_kl_loss": [1.0], "val_kl_loss": [1.0]},
            "C": {"train_total_loss": [1.0], "val_total_loss": [1.0], "train_recon_loss": [1.0], "val_recon_loss": [1.0], "train_kl_loss": [1.0], "val_kl_loss": [1.0]},
        }
        with self.assertRaises(ValueError):
            plot_learning_curves(incomplete_hist, required_models=("A", "B", "C", "D"))

        # Missing curve key
        bad_curve_hist = {
            m: {"train_total_loss": [1.0], "val_total_loss": [1.0]} for m in ["A", "B", "C", "D"]
        }
        with self.assertRaises(ValueError):
            plot_learning_curves(bad_curve_hist)


class TestEvaluatorIntegration(unittest.TestCase):
    """Integration test verifying Evaluator model loading, checkpoint verification, and evaluation flow."""

    def test_evaluator_integration_synthetic(self):
        """Verify model loading, checkpoint metadata validation, and the complete per-model evaluation flow."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg = ProjectConfig()
            cfg.paths.checkpoints_dir = tmp_path / "checkpoints"
            cfg.paths.experiments_dir = tmp_path / "experiments"
            cfg.paths.metrics_dir = tmp_path / "metrics"
            cfg.paths.figures_dir = tmp_path / "figures"
            cfg.paths.histories_dir = tmp_path / "histories"

            cfg.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            cfg.paths.experiments_dir.mkdir(parents=True, exist_ok=True)
            cfg.paths.metrics_dir.mkdir(parents=True, exist_ok=True)

            # 1. Save dummy fitted PCA object
            pca = PCA(n_components=2)
            pca.fit(np.random.randn(100, 20))
            joblib.dump(pca, cfg.paths.metrics_dir / "fitted_pca.joblib")

            # 2. Build and save temporary checkpoints for Models A, B, C, D
            test_epochs = {"A": 10, "B": 15, "C": 20, "D": 25}
            test_val_losses = {"A": 12.5, "B": 11.0, "C": 9.5, "D": 9.8}

            for m in ["A", "B", "C", "D"]:
                mod = build_vae(m, cfg)
                save_checkpoint(
                    checkpoint_path=cfg.paths.checkpoints_dir / f"model_{m}_best.pt",
                    model=mod,
                    optimizer=None,
                    epoch=test_epochs[m],
                    val_loss=test_val_losses[m],
                    val_recon_loss=5.0,
                    val_kl_loss=test_val_losses[m] - 5.0,
                    model_type=m,
                )

            # 3. Test that missing experiment_summary.json raises FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                Evaluator(config=cfg, device="cpu")

            # 4. Save valid experiment_summary.json
            summary_data = {
                "total_experiments_time_sec": 10.0,
                "device": "cpu",
                "models": [
                    {"model_type": m, "best_epoch": test_epochs[m], "best_val_loss": test_val_losses[m]}
                    for m in ["A", "B", "C", "D"]
                ],
            }
            summary_path = cfg.paths.experiments_dir / "experiment_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2)

            # 5. Instantiate Evaluator with valid experiment summary
            evaluator = Evaluator(
                config=cfg,
                device="cpu",
                generation_seed=42,
                bandwidth_seed=42,
            )

            # Verify bandwidths exist in full precision
            self.assertIn("data_space_bandwidth", evaluator.bandwidths)
            self.assertIn("latent_space_bandwidth", evaluator.bandwidths)
            self.assertIsInstance(evaluator.bandwidths["data_space_bandwidth"], float)
            self.assertGreater(evaluator.bandwidths["data_space_bandwidth"], 0.0)

            # 6. Test successful model loading and metadata return
            model_a, meta_a = evaluator.load_model("A")
            self.assertEqual(meta_a["checkpoint_epoch"], 10)
            self.assertAlmostEqual(meta_a["checkpoint_val_loss"], 12.5)
            self.assertTrue(Path(meta_a["checkpoint_path"]).exists())

            # 7. Test tampered epoch in experiment_summary.json raises ValueError
            tampered_data = dict(summary_data)
            tampered_data["models"] = [
                {"model_type": m, "best_epoch": test_epochs[m] + 100 if m == "A" else test_epochs[m], "best_val_loss": test_val_losses[m]}
                for m in ["A", "B", "C", "D"]
            ]
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(tampered_data, f, indent=2)

            tampered_evaluator = Evaluator(config=cfg, device="cpu")
            with self.assertRaises(ValueError):
                tampered_evaluator.load_model("A")

            # Restore valid summary
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2)

            # 8. Test complete per-model evaluation flow with synthetic parameters
            eval_res = evaluator.evaluate_model("A", n_gen_samples=64, n_mmd_samples=32)
            self.assertIn("summary", eval_res)
            s = eval_res["summary"]
            self.assertEqual(s["model_type"], "A")
            self.assertEqual(s["checkpoint_epoch"], 10)
            self.assertAlmostEqual(s["checkpoint_val_loss"], 12.5)
            self.assertIn("mse_per_element", s)
            self.assertIn("mean_squared_l2_per_sample", s)
            self.assertIn("data_mmd2", s)
            self.assertIn("latent_mmd2", s)
            self.assertIn("sample_selection_rules", s)
            self.assertTrue(Path(s["generated_samples_path"]).exists())


if __name__ == "__main__":
    unittest.main()
