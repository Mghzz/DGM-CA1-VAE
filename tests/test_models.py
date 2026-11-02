"""Comprehensive unit tests for Phase 2: VAE Architectures, Priors, and Loss Engine."""

import math
import unittest
import numpy as np
import torch
import torch.nn as nn

from configs.config import ProjectConfig
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.priors import (
    StandardNormalPrior,
    FixedGMM8Prior,
    LearnableGMMPrior,
)
from src.models.losses import (
    gaussian_reconstruction_nll,
    compute_vae_loss,
)
from src.models.vae import VAE, build_vae


class TestVAEArchitecture(unittest.TestCase):
    """Tests for Encoder, Decoder, and Reparameterization."""

    def setUp(self):
        self.config = ProjectConfig()
        self.batch_size = 256
        self.input_dim = 20
        self.latent_dim = 2

    def test_encoder_output_shapes(self):
        """Verify encoder outputs two distinct heads of shape (batch, 2)."""
        encoder = Encoder(
            input_dim=self.input_dim,
            hidden_dims=(128, 128),
            latent_dim=self.latent_dim,
        )
        x = torch.randn(self.batch_size, self.input_dim)
        mu_z, logvar_z = encoder(x)

        self.assertEqual(mu_z.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar_z.shape, (self.batch_size, self.latent_dim))

    def test_decoder_output_shapes(self):
        """Verify decoder output shape (batch, 20) with no activation."""
        decoder = Decoder(
            latent_dim=self.latent_dim,
            hidden_dims=(128, 128),
            output_dim=self.input_dim,
        )
        z = torch.randn(self.batch_size, self.latent_dim)
        mu_x = decoder(z)

        self.assertEqual(mu_x.shape, (self.batch_size, self.input_dim))

    def test_reparameterization_shape_and_gradient_flow(self):
        """Verify reparameterization trick preserves shapes and propagates gradients."""
        model = build_vae("A", self.config)
        mu = torch.randn(self.batch_size, self.latent_dim, requires_grad=True)
        logvar = torch.randn(self.batch_size, self.latent_dim, requires_grad=True)

        z = model.reparameterize(mu, logvar)
        self.assertEqual(z.shape, (self.batch_size, self.latent_dim))

        # Backward test
        loss = z.sum()
        loss.backward()

        self.assertIsNotNone(mu.grad)
        self.assertIsNotNone(logvar.grad)
        # Gradient w.r.t mu must be 1.0 everywhere
        torch.testing.assert_close(mu.grad, torch.ones_like(mu))
        # Gradient w.r.t logvar must be non-zero
        self.assertTrue((logvar.grad != 0).all())


class TestLossEngine(unittest.TestCase):
    """Tests for exact Gaussian NLL, Analytical KL, and MC KL."""

    def test_gaussian_reconstruction_nll_exact_constant(self):
        """Verify Gaussian NLL constant: when x == mu_x, NLL == 10 * log(0.18*pi) ~ -5.70059."""
        sigma_x = 0.30
        var_x = sigma_x ** 2
        d = 20
        expected_constant = 0.5 * d * math.log(2.0 * math.pi * var_x)  # ~ -5.7005914

        x = torch.zeros(16, d)
        mu_x = torch.zeros(16, d)  # Zero reconstruction error

        nll = gaussian_reconstruction_nll(x, mu_x, sigma_x=sigma_x)
        self.assertAlmostEqual(nll.item(), expected_constant, places=5)

    def test_gaussian_reconstruction_nll_input_validation(self):
        """Verify input validation: rejects non-positive sigma_x and mismatched shapes."""
        x = torch.zeros(16, 20)
        mu_x = torch.zeros(16, 20)

        with self.assertRaises(ValueError):
            gaussian_reconstruction_nll(x, mu_x, sigma_x=0.0)

        with self.assertRaises(ValueError):
            gaussian_reconstruction_nll(x, mu_x, sigma_x=-0.30)

        mismatched_mu = torch.zeros(16, 10)
        with self.assertRaises(ValueError):
            gaussian_reconstruction_nll(x, mismatched_mu, sigma_x=0.30)

    def test_gaussian_reconstruction_nll_scaling(self):
        """Verify that MSE error is scaled by 0.5 / sigma_x^2 = 0.5 / 0.09 = 1 / 0.18."""
        sigma_x = 0.30
        d = 20
        x = torch.ones(1, d)
        mu_x = torch.zeros(1, d)  # Squared error per dimension = 1.0 -> sum = 20.0

        expected_nll = 0.5 * (20.0 / 0.09 + 20.0 * math.log(2.0 * math.pi * 0.09))
        nll = gaussian_reconstruction_nll(x, mu_x, sigma_x=sigma_x)
        self.assertAlmostEqual(nll.item(), expected_nll, places=5)

    def test_analytical_kl_standard_normal_manual_example(self):
        """Verify analytical KL against manually derived numerical example.
        
        Let mu = [1.0, 2.0], logvar = [0.0, log(4.0)].
        j=1: 1 + 0 - 1^2 - 1 = -1 -> -0.5 * (-1) = 0.5
        j=2: 1 + log(4) - 4 - 4 = log(4) - 7 -> -0.5 * (log(4) - 7) = 3.5 - 0.5*log(4)
        Total KL = 0.5 + 3.5 - 0.5 * log(4) = 4.0 - log(2) ~ 3.3068528
        """
        prior = StandardNormalPrior(latent_dim=2)
        mu = torch.tensor([[1.0, 2.0]])
        logvar = torch.tensor([[0.0, math.log(4.0)]])
        z = torch.zeros_like(mu)

        kl = prior.kl_divergence(z, mu, logvar)
        expected_kl = 4.0 - math.log(2.0)
        self.assertAlmostEqual(kl.item(), expected_kl, places=5)

    def test_gmm_log_prob_manual_example(self):
        """Verify GMM log_prob against hand-calculated 2-component mixture.
        
        K=2, weights=[0.5, 0.5].
        mu_1 = [0, 0], sigma_1 = [1, 1] -> log N(0; 0, I) = -log(2*pi)
        mu_2 = [2, 0], sigma_2 = [1, 1] -> log N(0; [2, 0], I) = -log(2*pi) - 2.0
        p(0) = 0.5 * (2*pi)^(-1) * (1 + exp(-2))
        log p(0) = -log(2) - log(2*pi) + log(1 + exp(-2)) ~ -2.4040963
        """
        prior = LearnableGMMPrior(n_components=2, init_std=1.0, latent_dim=2)
        # Override parameters to match manual test
        with torch.no_grad():
            prior.centers.copy_(torch.tensor([[0.0, 0.0], [2.0, 0.0]]))
            prior.log_stds.copy_(torch.zeros(2, 2))  # std = 1.0

        z = torch.zeros(1, 2)
        log_prob = prior.log_prob(z)

        expected = -math.log(2.0) - math.log(2.0 * math.pi) + math.log(1.0 + math.exp(-2.0))
        self.assertAlmostEqual(log_prob.item(), expected, places=5)

    def test_mixture_log_prob_numerical_stability(self):
        """Verify logsumexp maintains finite outputs without NaNs on extreme values."""
        prior = FixedGMM8Prior(radius=2.0, covariance=0.16)
        # Moderate and large latent points
        z_extreme = torch.tensor([
            [10.0, 10.0],
            [-15.0, 20.0],
            [35.0, -35.0],
        ])
        log_probs = prior.log_prob(z_extreme)
        self.assertEqual(log_probs.shape, (3,))
        self.assertFalse(torch.isnan(log_probs).any())
        self.assertFalse((log_probs > 0).any(), "Probabilities cannot exceed 1.0")


class TestPriorsAndModels(unittest.TestCase):
    """Tests for Priors A, B, C, D and VAE factory."""

    def setUp(self):
        self.config = ProjectConfig()

    def test_prior_sampling_shapes(self):
        """Verify sampling shapes for all 4 priors."""
        priors = [
            StandardNormalPrior(latent_dim=2),
            FixedGMM8Prior(radius=2.0, covariance=0.16, n_components=8),
            LearnableGMMPrior(n_components=8, init_std=0.4),
            LearnableGMMPrior(n_components=4, init_std=0.4),
        ]
        for prior in priors:
            samples = prior.sample(2000)
            self.assertEqual(samples.shape, (2000, 2))
            self.assertFalse(torch.isnan(samples).any())

    def test_learnable_gmm_prior_input_validation(self):
        """Verify LearnableGMMPrior input validation: rejects n_components < 1 and init_std <= 0."""
        with self.assertRaises(ValueError):
            LearnableGMMPrior(n_components=0, init_std=0.4)

        with self.assertRaises(ValueError):
            LearnableGMMPrior(n_components=-5, init_std=0.4)

        with self.assertRaises(ValueError):
            LearnableGMMPrior(n_components=8, init_std=0.0)

        with self.assertRaises(ValueError):
            LearnableGMMPrior(n_components=8, init_std=-0.4)

    def test_model_b_parameters_remain_fixed(self):
        """Verify Model B prior has 0 trainable parameters and values remain constant."""
        model_b = build_vae("B", self.config)

        # Count trainable parameters in prior
        prior_params = list(model_b.prior.parameters())
        self.assertEqual(len(prior_params), 0, "Model B prior must have 0 trainable parameters")

        # Verify persistent buffers exist with requires_grad=False
        self.assertFalse(model_b.prior.centers.requires_grad)
        self.assertFalse(model_b.prior.log_stds.requires_grad)
        self.assertFalse(model_b.prior.weights.requires_grad)

        # Clone original centers, log_stds, and weights before optimizer step
        orig_centers = model_b.prior.centers.clone()
        orig_log_stds = model_b.prior.log_stds.clone()
        orig_weights = model_b.prior.weights.clone()

        # Simulate training step to confirm centers and buffers do not drift
        optimizer = torch.optim.Adam(model_b.parameters(), lr=0.001)
        x = torch.randn(32, 20)
        out = model_b(x)
        loss = model_b.compute_loss(x, out)
        optimizer.zero_grad()
        loss.total_loss.backward()
        optimizer.step()

        # Compare actual buffers afterward with clones using rtol=0, atol=0
        torch.testing.assert_close(model_b.prior.centers, orig_centers, rtol=0.0, atol=0.0)
        torch.testing.assert_close(model_b.prior.log_stds, orig_log_stds, rtol=0.0, atol=0.0)
        torch.testing.assert_close(model_b.prior.weights, orig_weights, rtol=0.0, atol=0.0)

    def test_models_c_and_d_parameters_receive_gradients(self):
        """Verify Models C and D prior centers and log_stds are learnable and receive gradients."""
        for m_type, expected_k in [("C", 8), ("D", 4)]:
            model = build_vae(m_type, self.config)

            self.assertEqual(model.prior.centers.shape, (expected_k, 2))
            self.assertEqual(model.prior.log_stds.shape, (expected_k, 2))

            self.assertTrue(model.prior.centers.requires_grad)
            self.assertTrue(model.prior.log_stds.requires_grad)
            self.assertFalse(model.prior.weights.requires_grad)

            # Initial std check
            expected_log_std = math.log(0.4)
            torch.testing.assert_close(
                model.prior.log_stds,
                torch.full((expected_k, 2), expected_log_std),
            )

            # Forward and backward check
            x = torch.randn(64, 20)
            out = model(x)
            loss = model.compute_loss(x, out)
            loss.total_loss.backward()

            self.assertIsNotNone(model.prior.centers.grad)
            self.assertIsNotNone(model.prior.log_stds.grad)
            self.assertTrue((model.prior.centers.grad != 0).any())
            self.assertTrue((model.prior.log_stds.grad != 0).any())

    def test_equal_mixture_weights(self):
        """Verify all mixture priors have strictly equal fixed weights."""
        model_b = build_vae("B", self.config)
        model_c = build_vae("C", self.config)
        model_d = build_vae("D", self.config)

        torch.testing.assert_close(model_b.prior.weights, torch.full((8,), 1.0 / 8.0))
        torch.testing.assert_close(model_c.prior.weights, torch.full((8,), 1.0 / 8.0))
        torch.testing.assert_close(model_d.prior.weights, torch.full((4,), 1.0 / 4.0))

    def test_finite_loss_and_gradients_normal_inputs(self):
        """Verify all 4 models produce finite loss and finite gradients on standard inputs."""
        for m_type in ["A", "B", "C", "D"]:
            model = build_vae(m_type, self.config)
            x = torch.randn(128, 20)
            out = model(x)
            loss = model.compute_loss(x, out)

            self.assertTrue(torch.isfinite(loss.total_loss))
            self.assertTrue(torch.isfinite(loss.recon_loss))
            self.assertTrue(torch.isfinite(loss.kl_loss))

            loss.total_loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.assertTrue(
                        torch.isfinite(param.grad).all(),
                        f"Non-finite gradient in {m_type} parameter {name}",
                    )

    def test_architecture_identity_across_models(self):
        """Verify all 4 models share the exact same encoder and decoder architecture."""
        models = [build_vae(m, self.config) for m in ["A", "B", "C", "D"]]

        for i in range(1, 4):
            # Check encoder parameter shapes
            enc_0_shapes = [p.shape for p in models[0].encoder.parameters()]
            enc_i_shapes = [p.shape for p in models[i].encoder.parameters()]
            self.assertEqual(enc_0_shapes, enc_i_shapes, f"Encoder architecture mismatch for Model {i}")

            # Check decoder parameter shapes
            dec_0_shapes = [p.shape for p in models[0].decoder.parameters()]
            dec_i_shapes = [p.shape for p in models[i].decoder.parameters()]
            self.assertEqual(dec_0_shapes, dec_i_shapes, f"Decoder architecture mismatch for Model {i}")

    def test_cuda_forward_backward_operation(self):
        """Verify forward and backward passes execute cleanly on CUDA if GPU is available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available on this machine.")

        device = torch.device("cuda")
        for m_type in ["A", "B", "C", "D"]:
            model = build_vae(m_type, self.config).to(device)
            x = torch.randn(64, 20, device=device)

            out = model(x)
            loss = model.compute_loss(x, out)

            self.assertEqual(out["mu_x"].device.type, "cuda")
            self.assertEqual(loss.total_loss.device.type, "cuda")

            loss.total_loss.backward()
            self.assertTrue(torch.isfinite(loss.total_loss))


if __name__ == "__main__":
    unittest.main()
