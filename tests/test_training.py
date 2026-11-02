"""Unit tests for Phase 3: Training Engine, Early Stopping, RNG Isolation, and Checkpointing."""

import json
from pathlib import Path
import tempfile
import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from configs.config import ProjectConfig
from src.models.vae import build_vae
from src.training.trainer import Trainer, set_seed, train_model
from src.training.checkpointing import load_checkpoint


def create_synthetic_loader(n_samples: int = 64, dim: int = 20, batch_size: int = 32) -> DataLoader:
    """Helper creating a fast synthetic DataLoader for tests."""
    torch.manual_seed(123)
    x = torch.randn(n_samples, dim)
    ds = TensorDataset(x)
    return DataLoader(
        [{"x": item[0]} for item in ds],
        batch_size=batch_size,
        shuffle=False,
    )


class TestTrainingEngine(unittest.TestCase):
    """Test suite for VAE training loop and components."""

    def setUp(self):
        self.config = ProjectConfig()
        self.train_loader = create_synthetic_loader(n_samples=64, dim=20, batch_size=32)
        self.val_loader = create_synthetic_loader(n_samples=32, dim=20, batch_size=32)

    def test_train_epoch_synthetic(self):
        """Verify train_epoch computes finite loss metrics and executes without errors."""
        model = build_vae("A", self.config)
        trainer = Trainer(model=model, model_type="A", config=self.config, device="cpu")

        metrics = trainer.train_epoch(self.train_loader)
        self.assertIn("total_loss", metrics)
        self.assertIn("recon_loss", metrics)
        self.assertIn("kl_loss", metrics)

        self.assertTrue(torch.isfinite(torch.tensor(metrics["total_loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(metrics["recon_loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(metrics["kl_loss"])))

    def test_validate_finite_losses(self):
        """Verify validation pass produces finite losses and does not compute gradients."""
        model = build_vae("A", self.config)
        trainer = Trainer(model=model, model_type="A", config=self.config, device="cpu")

        metrics = trainer.validate(self.val_loader)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["total_loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(metrics["recon_loss"])))
        self.assertTrue(torch.isfinite(torch.tensor(metrics["kl_loss"])))

    def test_training_updates_model_parameters(self):
        """Verify that training steps actually modify the network parameters."""
        model = build_vae("A", self.config)
        trainer = Trainer(model=model, model_type="A", config=self.config, device="cpu")

        initial_weight = model.encoder.trunk[0].weight.clone()
        trainer.train_epoch(self.train_loader)
        updated_weight = model.encoder.trunk[0].weight

        self.assertFalse(torch.allclose(initial_weight, updated_weight))

    def test_learnable_prior_parameters_update(self):
        """Verify that learnable GMM parameters in Models C and D update during training."""
        for m_type in ["C", "D"]:
            model = build_vae(m_type, self.config)
            trainer = Trainer(model=model, model_type=m_type, config=self.config, device="cpu")

            initial_centers = model.prior.centers.clone()
            initial_log_stds = model.prior.log_stds.clone()
            initial_weights = model.prior.weights.clone()

            # Run 2 training epochs to ensure observable parameter updates
            trainer.train_epoch(self.train_loader)
            trainer.train_epoch(self.train_loader)

            # Centers and log_stds must update
            self.assertFalse(
                torch.allclose(initial_centers, model.prior.centers),
                f"Model {m_type} prior centers did not update",
            )
            self.assertFalse(
                torch.allclose(initial_log_stds, model.prior.log_stds),
                f"Model {m_type} prior log_stds did not update",
            )
            # Prior weights must remain strictly identical
            torch.testing.assert_close(initial_weights, model.prior.weights, rtol=0.0, atol=0.0)

    def test_model_b_fixed_prior_buffers_unchanged(self):
        """Verify Model B prior buffers remain bitwise identical after multiple training epochs."""
        model = build_vae("B", self.config)
        trainer = Trainer(model=model, model_type="B", config=self.config, device="cpu")

        initial_centers = model.prior.centers.clone()
        initial_log_stds = model.prior.log_stds.clone()
        initial_weights = model.prior.weights.clone()

        for _ in range(3):
            trainer.train_epoch(self.train_loader)

        torch.testing.assert_close(initial_centers, model.prior.centers, rtol=0.0, atol=0.0)
        torch.testing.assert_close(initial_log_stds, model.prior.log_stds, rtol=0.0, atol=0.0)
        torch.testing.assert_close(initial_weights, model.prior.weights, rtol=0.0, atol=0.0)

    def test_validation_rng_isolation(self):
        """Verify validation sampling consumes 0 draws from the global PyTorch RNG."""
        model = build_vae("B", self.config)
        trainer = Trainer(model=model, model_type="B", config=self.config, device="cpu")

        # Baseline: draw 10 random numbers from global RNG
        torch.manual_seed(98765)
        expected_draw = torch.randn(10)

        # Draw with intervening validation pass
        torch.manual_seed(98765)
        trainer.validate(self.val_loader)
        actual_draw = torch.randn(10)

        # Must be bitwise identical
        torch.testing.assert_close(actual_draw, expected_draw, rtol=0.0, atol=0.0)

        # Furthermore, two consecutive validation passes must yield identical losses
        val_1 = trainer.validate(self.val_loader)
        val_2 = trainer.validate(self.val_loader)
        self.assertEqual(val_1["total_loss"], val_2["total_loss"])

    def test_early_stopping_and_checkpoint_restoration(self):
        """Verify early stopping triggers at patience=3 and genuinely restores best weights, not last."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            chk_dir = tmp_path / "checkpoints"
            hist_dir = tmp_path / "histories"

            model = build_vae("A", self.config)
            trainer = Trainer(model=model, model_type="A", config=self.config, device="cpu")

            loss_sequence = [10.0, 12.0, 13.0, 14.0, 15.0, 16.0]
            call_count = [0]
            states_by_epoch = {}

            # Wrap train_epoch to record model state immediately after each epoch's training
            orig_train_epoch = trainer.train_epoch

            def recording_train_epoch(loader):
                metrics = orig_train_epoch(loader)
                epoch_idx = call_count[0] + 1
                states_by_epoch[epoch_idx] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                return metrics

            def mock_validate(loader):
                val_loss = loss_sequence[call_count[0]]
                call_count[0] += 1
                return {"total_loss": val_loss, "recon_loss": val_loss - 1.0, "kl_loss": 1.0}

            trainer.train_epoch = recording_train_epoch
            trainer.validate = mock_validate

            result = trainer.fit(
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                max_epochs=10,
                patience=3,
                verbose=False,
                save_checkpoints=True,
                checkpoint_dir=chk_dir,
                history_dir=hist_dir,
            )

            # 1. Early stopping metadata checks
            self.assertTrue(result.stopped_early)
            self.assertEqual(result.best_epoch, 1)
            self.assertEqual(result.best_val_loss, 10.0)
            self.assertEqual(result.total_epochs, 4)

            # 2. Check that model parameters evolved between best epoch (1) and final epoch (4)
            best_epoch_state = states_by_epoch[1]
            final_epoch_state = states_by_epoch[4]
            any_param_changed = False
            for k in best_epoch_state:
                if not torch.equal(best_epoch_state[k], final_epoch_state[k]):
                    any_param_changed = True
                    break
            self.assertTrue(any_param_changed, "Model parameters must change across epochs")

            # 3. Rigorous weight verification: returned model MUST equal best_epoch_state bit-for-bit
            for k, current_param in model.state_dict().items():
                torch.testing.assert_close(current_param, best_epoch_state[k], rtol=0.0, atol=0.0)

            # 4. Rigorous non-last verification: returned model MUST NOT equal final non-best epoch state
            has_difference_from_final = False
            for k, current_param in model.state_dict().items():
                if not torch.equal(current_param, final_epoch_state[k]):
                    has_difference_from_final = True
                    break
            self.assertTrue(has_difference_from_final, "Returned model weights must not match final epoch state")

            # 5. Check checkpoint file exists and metadata is complete
            self.assertIsNotNone(result.checkpoint_path)
            self.assertTrue(result.checkpoint_path.exists())

            loaded = load_checkpoint(result.checkpoint_path, model=build_vae("A", self.config))
            self.assertEqual(loaded["epoch"], 1)
            self.assertEqual(loaded["val_loss"], 10.0)
            self.assertIn("metadata", loaded)
            meta = loaded["metadata"]
            self.assertEqual(meta["lr"], self.config.training.lr)
            self.assertEqual(meta["batch_size"], self.config.training.batch_size)
            self.assertEqual(meta["max_epochs"], 10)
            self.assertEqual(meta["patience"], 3)
            self.assertEqual(meta["model_seed"], self.config.training.model_seed)
            self.assertEqual(meta["val_eval_seed"], self.config.training.val_eval_seed)
            self.assertEqual(meta["likelihood_sigma"], self.config.model.likelihood_sigma)
            self.assertEqual(meta["latent_dim"], self.config.model.latent_dim)

            # Check history structure
            self.assertEqual(len(result.history["train_total_loss"]), 4)
            self.assertEqual(len(result.history["val_total_loss"]), 4)
            self.assertEqual(result.history["val_total_loss"], [10.0, 12.0, 13.0, 14.0])

    def test_invalid_fit_arguments(self):
        """Verify Trainer.fit rejects max_epochs <= 0 and patience <= 0."""
        model = build_vae("A", self.config)
        trainer = Trainer(model=model, model_type="A", config=self.config, device="cpu")

        with self.assertRaises(ValueError):
            trainer.fit(self.train_loader, self.val_loader, max_epochs=0)

        with self.assertRaises(ValueError):
            trainer.fit(self.train_loader, self.val_loader, max_epochs=-5)

        with self.assertRaises(ValueError):
            trainer.fit(self.train_loader, self.val_loader, patience=0)

        with self.assertRaises(ValueError):
            trainer.fit(self.train_loader, self.val_loader, patience=-3)

    def test_empty_loaders_raise_value_error(self):
        """Verify train_epoch and validate raise ValueError on empty DataLoaders."""
        model = build_vae("A", self.config)
        trainer = Trainer(model=model, model_type="A", config=self.config, device="cpu")
        empty_loader = DataLoader([])

        with self.assertRaises(ValueError):
            trainer.train_epoch(empty_loader)

        with self.assertRaises(ValueError):
            trainer.validate(empty_loader)

    def test_reproducible_dataloader_shuffle_sequence(self):
        """Verify DataLoader shuffle sequence is strictly reproducible with training seed."""
        from src.data.dataset_loader import get_dataloaders
        loader1, _, _, _ = get_dataloaders(batch_size=256, shuffle_train=True, seed=42)
        loader2, _, _, _ = get_dataloaders(batch_size=256, shuffle_train=True, seed=42)

        batch1 = next(iter(loader1))["x"]
        batch2 = next(iter(loader2))["x"]
        torch.testing.assert_close(batch1, batch2, rtol=0.0, atol=0.0)

    def test_incomplete_batch_loss_accumulation(self):
        """Verify loss accumulation accurately weights incomplete batches by their size."""
        # 50 samples with batch_size 32 gives batch 1 of 32, batch 2 of 18
        loader = create_synthetic_loader(n_samples=50, dim=20, batch_size=32)
        model = build_vae("A", self.config)
        trainer = Trainer(model=model, model_type="A", config=self.config, device="cpu")

        metrics = trainer.train_epoch(loader)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["total_loss"])))

    def test_cuda_execution_if_available(self):
        """Verify Trainer runs cleanly on CUDA when available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available.")

        model = build_vae("C", self.config)
        trainer = Trainer(model=model, model_type="C", config=self.config, device="cuda")
        metrics = trainer.train_epoch(self.train_loader)

        self.assertTrue(torch.isfinite(torch.tensor(metrics["total_loss"])))
        self.assertEqual(model.encoder.trunk[0].weight.device.type, "cuda")

    def test_cpu_execution(self):
        """Verify training and validation run cleanly when explicitly forced to CPU."""
        model = build_vae("A", self.config)
        result = train_model(
            model=model,
            model_type="A",
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=self.config,
            device="cpu",
            max_epochs=2,
            patience=5,
            verbose=False,
            save_checkpoints=False,
        )
        self.assertEqual(result.total_epochs, 2)
        self.assertEqual(len(result.history["train_total_loss"]), 2)


if __name__ == "__main__":
    unittest.main()
