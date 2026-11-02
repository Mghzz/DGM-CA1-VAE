"""Unit tests for Phase 4: Experiment Runner and Verification Suite."""

from pathlib import Path
import tempfile
import unittest
import torch
import torch.nn as nn

from configs.config import ProjectConfig
from src.models.vae import build_vae
from src.training.run_experiments import (
    verify_saved_checkpoint,
    run_single_experiment,
    benchmark_training_runtime,
)
from src.training.checkpointing import load_checkpoint, save_checkpoint
from tests.test_training import create_synthetic_loader


class TestExperimentRunner(unittest.TestCase):
    """Test suite for experiment runner and checkpoint verification."""

    def setUp(self):
        self.config = ProjectConfig()
        self.train_loader = create_synthetic_loader(n_samples=64, dim=20, batch_size=32)
        self.val_loader = create_synthetic_loader(n_samples=32, dim=20, batch_size=32)

    def test_verify_saved_checkpoint_success_and_tamper_detection(self):
        """Verify that verify_saved_checkpoint passes on genuine weights and fails on tampered ones."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ckpt_path = tmp_path / "model_C_best.pt"

            model = build_vae("C", self.config)
            save_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                optimizer=None,
                epoch=5,
                val_loss=25.5,
                val_recon_loss=20.0,
                val_kl_loss=5.5,
                model_type="C",
            )

            # 1. Genuine verification must pass
            res = verify_saved_checkpoint(
                checkpoint_path=ckpt_path,
                model=model,
                model_type="C",
                expected_epoch=5,
                expected_val_loss=25.5,
                device=torch.device("cpu"),
            )
            self.assertTrue(res["verified"])
            self.assertTrue(res["learnable_prior_verified"])

            # 2. Tampered expected epoch must fail
            with self.assertRaises(ValueError):
                verify_saved_checkpoint(
                    checkpoint_path=ckpt_path,
                    model=model,
                    model_type="C",
                    expected_epoch=4,
                    expected_val_loss=25.5,
                    device=torch.device("cpu"),
                )

            # 3. Tampered model weights must fail
            with torch.no_grad():
                model.encoder.trunk[0].weight.add_(1.0)
            with self.assertRaises(AssertionError):
                verify_saved_checkpoint(
                    checkpoint_path=ckpt_path,
                    model=model,
                    model_type="C",
                    expected_epoch=5,
                    expected_val_loss=25.5,
                    device=torch.device("cpu"),
                )

    def test_run_single_experiment_fast(self):
        """Verify end-to-end single experiment execution and artifact generation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg = ProjectConfig()
            cfg.paths.checkpoints_dir = tmp_path / "checkpoints"
            cfg.paths.experiments_dir = tmp_path / "experiments"
            cfg.paths.histories_dir = tmp_path / "histories"

            meta = run_single_experiment(
                model_type="B",
                config=cfg,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                device="cpu",
                max_epochs=2,
                patience=5,
                verbose=False,
            )

            self.assertEqual(meta["model_type"], "B")
            self.assertEqual(meta["total_epochs"], 2)
            self.assertTrue(Path(meta["checkpoint_path"]).exists())
            self.assertTrue((cfg.paths.experiments_dir / "model_B" / "history.json").exists())
            self.assertTrue((cfg.paths.experiments_dir / "model_B" / "history.csv").exists())
            self.assertTrue((cfg.paths.experiments_dir / "model_B" / "metadata.json").exists())
            self.assertTrue(meta["verification"]["verified"])

    def test_benchmark_training_runtime_execution(self):
        """Verify benchmark runs cleanly and produces all expected timing and VRAM metrics."""
        rep = benchmark_training_runtime(
            config=self.config,
            device="cpu",
            warmup_steps=1,
            measure_steps=2,
        )
        self.assertIn("measured_metrics", rep)
        self.assertIn("projections", rep)
        self.assertIn("projection_assumptions", rep)

        metrics = rep["measured_metrics"]
        self.assertIn("training_batch_breakdown_ms", metrics)
        self.assertIn("validation_batch_breakdown_ms", metrics)
        self.assertIn("checkpoint_save_ms", metrics)
        self.assertIn("complete_epoch_time_sec", metrics)

        t_breakdown = metrics["training_batch_breakdown_ms"]
        self.assertIn("dataloader_iteration_ms", t_breakdown)
        self.assertIn("host_to_device_transfer_ms", t_breakdown)
        self.assertIn("model_computation_ms", t_breakdown)
        self.assertIn("complete_training_batch_ms", t_breakdown)

        v_breakdown = metrics["validation_batch_breakdown_ms"]
        self.assertIn("dataloader_iteration_ms", v_breakdown)
        self.assertIn("host_to_device_transfer_ms", v_breakdown)
        self.assertIn("model_computation_ms", v_breakdown)
        self.assertIn("complete_validation_batch_ms", v_breakdown)

        self.assertGreater(metrics["complete_epoch_time_sec"], 0.0)

    def test_model_b_contract_buffers_optimizer_checkpoint_reload(self):
        """Verify the full 4-point contract for Model B fixed prior.
        
        1. Registered as non-trainable buffers, not learnable parameters.
        2. Optimizer state contains no moments or LR entries for fixed prior components.
        3. Saved checkpoint contains exact analytical values of centers and covariance.
        4. Reloaded checkpoint preserves these fixed values without modification.
        """
        import math

        # 1. Registered as non-trainable buffers, not learnable parameters
        model_b = build_vae("B", self.config)
        prior_named_params = list(model_b.prior.named_parameters())
        self.assertEqual(
            len(prior_named_params),
            0,
            f"Model B prior must have 0 parameters, but has: {prior_named_params}",
        )

        buffer_dict = dict(model_b.prior.named_buffers())
        self.assertIn("centers", buffer_dict)
        self.assertIn("log_stds", buffer_dict)
        self.assertIn("weights", buffer_dict)

        self.assertFalse(model_b.prior.centers.requires_grad)
        self.assertFalse(model_b.prior.log_stds.requires_grad)
        self.assertFalse(model_b.prior.weights.requires_grad)

        # 2. Optimizer state contains no moments or learning-rate entries for prior
        optimizer = torch.optim.Adam(model_b.parameters(), lr=0.001)
        optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}

        self.assertNotIn(id(model_b.prior.centers), optimizer_param_ids)
        self.assertNotIn(id(model_b.prior.log_stds), optimizer_param_ids)
        self.assertNotIn(id(model_b.prior.weights), optimizer_param_ids)

        # Execute training step
        x = torch.randn(32, 20)
        out = model_b(x)
        loss = model_b.compute_loss(x, out)
        optimizer.zero_grad()
        loss.total_loss.backward()
        optimizer.step()

        # Prior buffers must not have gradients
        self.assertIsNone(model_b.prior.centers.grad)
        self.assertIsNone(model_b.prior.log_stds.grad)
        self.assertIsNone(model_b.prior.weights.grad)

        # Optimizer state table must not track prior buffers
        for p in optimizer.state:
            self.assertNotEqual(id(p), id(model_b.prior.centers))
            self.assertNotEqual(id(p), id(model_b.prior.log_stds))
            self.assertNotEqual(id(p), id(model_b.prior.weights))

        # 3. Checkpoint contains exact analytical values of centers and covariance
        angles = 2.0 * math.pi * torch.arange(8, dtype=torch.float32) / 8.0
        analytical_centers = 2.0 * torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        # Covariance 0.16 * I_2 -> std = 0.4 -> log_std = log(0.4)
        analytical_log_stds = torch.full((8, 2), math.log(0.4), dtype=torch.float32)
        analytical_weights = torch.full((8,), 1.0 / 8.0, dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "model_B_test.pt"
            save_checkpoint(
                checkpoint_path=ckpt_path,
                model=model_b,
                optimizer=optimizer,
                epoch=3,
                val_loss=15.0,
                val_recon_loss=10.0,
                val_kl_loss=5.0,
                model_type="B",
            )

            # Direct state_dict inspection from raw disk checkpoint
            raw_ckpt = torch.load(ckpt_path, weights_only=False)
            sd = raw_ckpt["model_state_dict"]

            torch.testing.assert_close(sd["prior.centers"], analytical_centers, rtol=0.0, atol=0.0)
            torch.testing.assert_close(sd["prior.log_stds"], analytical_log_stds, rtol=0.0, atol=0.0)
            torch.testing.assert_close(sd["prior.weights"], analytical_weights, rtol=0.0, atol=0.0)

            # 4. Reloaded checkpoint preserves these fixed values without modification
            reloaded_model = build_vae("B", self.config)
            # Perturb before reload to verify it genuinely overwrites with checkpoint values
            with torch.no_grad():
                reloaded_model.prior.centers.add_(999.0)
                reloaded_model.prior.log_stds.add_(999.0)
                reloaded_model.prior.weights.add_(999.0)

            load_checkpoint(ckpt_path, model=reloaded_model)

            torch.testing.assert_close(reloaded_model.prior.centers, analytical_centers, rtol=0.0, atol=0.0)
            torch.testing.assert_close(reloaded_model.prior.log_stds, analytical_log_stds, rtol=0.0, atol=0.0)
            torch.testing.assert_close(reloaded_model.prior.weights, analytical_weights, rtol=0.0, atol=0.0)

            # Run verify_saved_checkpoint
            v_res = verify_saved_checkpoint(
                checkpoint_path=ckpt_path,
                model=model_b,
                model_type="B",
                expected_epoch=3,
                expected_val_loss=15.0,
                device=torch.device("cpu"),
            )
            self.assertTrue(v_res["verified"])
            self.assertTrue(v_res["fixed_prior_verified"])


if __name__ == "__main__":
    unittest.main()
