"""Unit tests for Phase 1: Dataset loading, alignment, and verification."""

import unittest
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA

from src.data.dataset_loader import load_dataset, get_dataloaders
from src.data.verify_dataset import verify_dataset
from src.visualization.baseline_plots import generate_baseline_visualizations


class TestDataset(unittest.TestCase):
    """Test suite for dataset loader and data integrity."""

    @classmethod
    def setUpClass(cls):
        cls.splits = load_dataset()

    def test_split_dimensions(self):
        """Verify train, validation, and test split shapes."""
        self.assertEqual(self.splits.x_train.shape, (6720, 20))
        self.assertEqual(self.splits.y_train.shape, (6720,))
        self.assertEqual(self.splits.s_train.shape, (6720, 2))

        self.assertEqual(self.splits.x_val.shape, (1440, 20))
        self.assertEqual(self.splits.y_val.shape, (1440,))
        self.assertEqual(self.splits.s_val.shape, (1440, 2))

        self.assertEqual(self.splits.x_test.shape, (1440, 20))
        self.assertEqual(self.splits.y_test.shape, (1440,))
        self.assertEqual(self.splits.s_test.shape, (1440, 2))

        total_samples = len(self.splits.x_train) + len(self.splits.x_val) + len(self.splits.x_test)
        self.assertEqual(total_samples, 9600)

    def test_finite_values(self):
        """Ensure no NaNs or Infs are present in any data split."""
        for name, arr in [
            ("x_train", self.splits.x_train),
            ("x_val", self.splits.x_val),
            ("x_test", self.splits.x_test),
            ("s_train", self.splits.s_train),
            ("s_val", self.splits.s_val),
            ("s_test", self.splits.s_test),
        ]:
            self.assertFalse(np.isnan(arr).any(), f"NaN found in {name}")
            self.assertFalse(np.isinf(arr).any(), f"Inf found in {name}")

    def test_cluster_labels(self):
        """Verify 8 clusters exist and are distributed across all splits."""
        for split_name, y in [
            ("train", self.splits.y_train),
            ("val", self.splits.y_val),
            ("test", self.splits.y_test),
        ]:
            unique_labels = np.unique(y)
            self.assertEqual(len(unique_labels), 8, f"Expected 8 clusters in {split_name}, got {len(unique_labels)}")
            self.assertTrue(set(unique_labels) == set(range(8)))

    def test_source_coordinate_alignment(self):
        """Verify that s[order] restores 1-to-1 correspondence with labels and centers."""
        s_all = np.vstack([self.splits.s_train, self.splits.s_val, self.splits.s_test])
        y_all = np.concatenate([self.splits.y_train, self.splits.y_val, self.splits.y_test])
        centers = self.splits.centers

        # Distance of each s to its assigned center
        assigned_centers = centers[y_all]
        distances = np.linalg.norm(s_all - assigned_centers, axis=1)

        mean_dist = float(distances.mean())
        max_dist = float(distances.max())

        # For 2D Gaussian with std=0.35, expected distance is sigma * sqrt(pi/2) ~ 0.438
        self.assertAlmostEqual(mean_dist, 0.35 * np.sqrt(np.pi / 2), delta=0.05)
        # Max distance should be strictly within cluster radius (< 2.0; cluster separation is ~3.06)
        self.assertLess(max_dist, 2.0)

    def test_normalization_properties(self):
        """Verify that normalization was strictly computed on x_train."""
        mean_x = self.splits.x_train.mean(axis=0)
        std_x = self.splits.x_train.std(axis=0)

        np.testing.assert_allclose(mean_x, 0.0, atol=1e-5, err_msg="Train mean not 0")
        np.testing.assert_allclose(std_x, 1.0, atol=1e-5, err_msg="Train std not 1")

    def test_dataloader_batch_shapes(self):
        """Verify DataLoader batch generation, types, and shapes."""
        train_loader, val_loader, test_loader, _ = get_dataloaders(batch_size=256, shuffle_train=False)

        batch = next(iter(train_loader))
        self.assertIn("x", batch)
        self.assertEqual(batch["x"].shape, (256, 20))
        self.assertEqual(batch["x"].dtype, torch.float32)

        val_batch = next(iter(val_loader))
        self.assertEqual(val_batch["x"].shape, (256, 20))

        # Check total number of batches
        # 6720 / 256 = 26 full batches + 1 batch of 64
        self.assertEqual(len(train_loader), int(np.ceil(6720 / 256)))
        # 1440 / 256 = 5 full batches + 1 batch of 160
        self.assertEqual(len(val_loader), int(np.ceil(1440 / 256)))
        self.assertEqual(len(test_loader), int(np.ceil(1440 / 256)))


class TestVisualizationsAndPCA(unittest.TestCase):
    """Test suite for PCA fitting and figure generation."""

    def test_generate_and_save_visualizations(self):
        """Run baseline visualizations and verify output artifacts."""
        summary = generate_baseline_visualizations()

        fig1_path = Path(summary["source_figure_path"])
        fig2_path = Path(summary["pca_figure_path"])

        self.assertTrue(fig1_path.exists(), f"Source figure missing at {fig1_path}")
        self.assertTrue(fig2_path.exists(), f"PCA figure missing at {fig2_path}")
        self.assertGreater(fig1_path.stat().st_size, 1000, "Source figure is empty")
        self.assertGreater(fig2_path.stat().st_size, 1000, "PCA figure is empty")

        # Verify saved PCA object
        for path_str in summary["saved_pca_paths"]:
            pca_path = Path(path_str)
            self.assertTrue(pca_path.exists(), f"Saved PCA missing at {pca_path}")
            pca = joblib.load(pca_path)
            self.assertIsInstance(pca, PCA)
            self.assertEqual(pca.n_components, 2)


if __name__ == "__main__":
    unittest.main()
