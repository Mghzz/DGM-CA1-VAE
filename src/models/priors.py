"""Prior distribution implementations for VAE models (A, B, C, D).

Priors supported:
- Model A: Standard normal N(0, I_2) with analytical KL.
- Model B: Fixed 8-component GMM on circle of radius 2 with covariance 0.16 * I_2. Non-trainable.
- Model C: Learnable 8-component GMM with centers initialized ~ N(0, I_2) and diagonal stds initialized to 0.4.
- Model D: Same as C, but with 4 components.

Mixture log-probabilities use logsumexp for strict numerical stability.
"""

from abc import ABC, abstractmethod
import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn


class BasePrior(nn.Module, ABC):
    """Abstract base class for VAE latent priors."""

    def __init__(self, latent_dim: int = 2) -> None:
        super().__init__()
        self.latent_dim = latent_dim

    @abstractmethod
    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Draws n random samples from the prior distribution.
        
        Args:
            n: Number of samples to draw.
            generator: Optional dedicated pseudo-random generator.
            
        Returns:
            Tensor of shape (n, latent_dim).
        """
        pass

    @abstractmethod
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluates log p(z) for a batch of latent vectors.
        
        Args:
            z: Latent vectors of shape (batch_size, latent_dim).
            
        Returns:
            Tensor of shape (batch_size,) with log-probability densities.
        """
        pass

    @abstractmethod
    def kl_divergence(
        self,
        z: torch.Tensor,
        mu_q: torch.Tensor,
        logvar_q: torch.Tensor,
    ) -> torch.Tensor:
        """Computes D_KL( q_phi(z|x) || p(z) ) per sample.
        
        Args:
            z: Sampled latent vectors z ~ q_phi(z|x) of shape (batch_size, latent_dim).
            mu_q: Posterior mean of shape (batch_size, latent_dim).
            logvar_q: Posterior log-variance of shape (batch_size, latent_dim).
            
        Returns:
            Tensor of shape (batch_size,) with KL divergence per sample.
        """
        pass

    @property
    @abstractmethod
    def has_analytical_kl(self) -> bool:
        """Whether this prior supports closed-form analytical KL divergence."""
        pass


class StandardNormalPrior(BasePrior):
    """Model A: Standard isotropic Gaussian prior p(z) = N(0, I_2).
    
    Supports exact closed-form analytical KL divergence with diagonal Gaussian posterior.
    """

    def __init__(self, latent_dim: int = 2) -> None:
        super().__init__(latent_dim=latent_dim)
        # Register standard normal constant as buffer
        self.register_buffer("log_2pi", torch.tensor(math.log(2.0 * math.pi)))

    @property
    def has_analytical_kl(self) -> bool:
        return True

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        device = self.log_2pi.device
        return torch.randn(n, self.latent_dim, device=device, generator=generator)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        # log N(z; 0, I) = -0.5 * (d * log(2*pi) + sum(z^2))
        d = self.latent_dim
        sq_norm = torch.sum(z ** 2, dim=-1)
        return -0.5 * (d * self.log_2pi + sq_norm)

    def kl_divergence(
        self,
        z: torch.Tensor,
        mu_q: torch.Tensor,
        logvar_q: torch.Tensor,
    ) -> torch.Tensor:
        # Analytical KL for diagonal Gaussians:
        # D_KL = -0.5 * sum_j [ 1 + logvar_j - mu_j^2 - exp(logvar_j) ]
        return -0.5 * torch.sum(1.0 + logvar_q - mu_q.pow(2) - logvar_q.exp(), dim=-1)


class FixedGMM8Prior(BasePrior):
    """Model B: Fixed 8-component Gaussian mixture prior.
    
    - Centers: 8 points equally spaced on circle of radius 2.
    - Covariance: 0.16 * I_2 (standard deviation = 0.4).
    - Weights: Fixed and equal (1/8).
    - Non-trainable: All parameters registered as persistent buffers.
    - KL: Monte Carlo approximation using logsumexp.
    """

    def __init__(
        self,
        radius: float = 2.0,
        covariance: float = 0.16,
        n_components: int = 8,
        latent_dim: int = 2,
    ) -> None:
        super().__init__(latent_dim=latent_dim)
        self.n_components = n_components

        # Centers on circle of radius 2
        angles = 2.0 * math.pi * torch.arange(n_components, dtype=torch.float32) / n_components
        centers = radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # (8, 2)

        # Covariance = 0.16 * I_2 -> std = 0.4 -> log_std = log(0.4)
        std_val = math.sqrt(covariance)
        log_stds = torch.full((n_components, latent_dim), math.log(std_val), dtype=torch.float32)
        weights = torch.full((n_components,), 1.0 / n_components, dtype=torch.float32)

        # Registered as persistent buffers with requires_grad=False
        self.register_buffer("centers", centers)
        self.register_buffer("log_stds", log_stds)
        self.register_buffer("weights", weights)
        self.register_buffer("log_2pi", torch.tensor(math.log(2.0 * math.pi)))

    @property
    def has_analytical_kl(self) -> bool:
        return False

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        device = self.centers.device
        # Equal weights: choose component index uniformly
        comp_indices = torch.randint(0, self.n_components, (n,), device=device, generator=generator)
        selected_centers = self.centers[comp_indices]  # (n, 2)
        selected_stds = torch.exp(self.log_stds[comp_indices])  # (n, 2)

        eps = torch.randn(n, self.latent_dim, device=device, generator=generator)
        return selected_centers + selected_stds * eps

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, 2) -> unsqueeze to (B, 1, 2)
        # centers: (1, K, 2), log_stds: (1, K, 2)
        z_expanded = z.unsqueeze(1)
        centers_expanded = self.centers.unsqueeze(0)
        log_stds_expanded = self.log_stds.unsqueeze(0)

        # Vectorized Gaussian log-density per component:
        # log N(z; mu_k, diag(sigma_k^2)) = -0.5 * [ d*log(2*pi) + 2*sum(log_std) + sum( (z - mu)^2 * exp(-2*log_std) ) ]
        diff = z_expanded - centers_expanded  # (B, K, 2)
        inv_vars = torch.exp(-2.0 * log_stds_expanded)  # (1, K, 2)
        quad = torch.sum(diff.pow(2) * inv_vars, dim=-1)  # (B, K)
        log_det = 2.0 * torch.sum(log_stds_expanded, dim=-1)  # (1, K)

        component_log_probs = -0.5 * (self.latent_dim * self.log_2pi + log_det + quad)  # (B, K)

        # log p(z) = logsumexp( component_log_probs + log(weights) )
        log_weights = torch.log(self.weights).unsqueeze(0)  # (1, K)
        return torch.logsumexp(component_log_probs + log_weights, dim=-1)  # (B,)

    def kl_divergence(
        self,
        z: torch.Tensor,
        mu_q: torch.Tensor,
        logvar_q: torch.Tensor,
    ) -> torch.Tensor:
        # Monte Carlo KL: log q(z|x) - log p(z)
        # log q(z|x) = -0.5 * [ d*log(2*pi) + sum(logvar_q) + sum( (z - mu_q)^2 * exp(-logvar_q) ) ]
        inv_var_q = torch.exp(-logvar_q)
        quad_q = torch.sum((z - mu_q).pow(2) * inv_var_q, dim=-1)
        log_det_q = torch.sum(logvar_q, dim=-1)
        log_q = -0.5 * (self.latent_dim * self.log_2pi + log_det_q + quad_q)

        log_p = self.log_prob(z)
        return log_q - log_p


class LearnableGMMPrior(BasePrior):
    """Models C & D: Learnable Gaussian mixture prior.
    
    - Centers: Initialized ~ N(0, I_2). Trainable parameters.
    - Covariance: Diagonal. Standard deviations initialized to 0.4.
                  Parameterized via unconstrained log_stds = log(0.4). Trainable parameters.
    - Weights: Fixed and equal (1 / n_components). Non-trainable buffer.
    - KL: Monte Carlo approximation using logsumexp.
    
    Used for:
    - Model C: n_components = 8
    - Model D: n_components = 4
    """

    def __init__(
        self,
        n_components: int,
        init_std: float = 0.4,
        latent_dim: int = 2,
    ) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be at least 1, got {n_components}")
        if init_std <= 0:
            raise ValueError(f"init_std must be positive, got {init_std}")

        super().__init__(latent_dim=latent_dim)
        self.n_components = n_components

        # Centers initialized from standard normal N(0, I_2)
        self.centers = nn.Parameter(torch.randn(n_components, latent_dim))

        # Learnable diagonal standard deviations initialized to init_std (0.4)
        # Parameterized as unconstrained log-standard-deviation
        self.log_stds = nn.Parameter(
            torch.full((n_components, latent_dim), math.log(init_std), dtype=torch.float32)
        )

        # Equal mixing weights strictly fixed as non-trainable buffer
        weights = torch.full((n_components,), 1.0 / n_components, dtype=torch.float32)
        self.register_buffer("weights", weights)
        self.register_buffer("log_2pi", torch.tensor(math.log(2.0 * math.pi)))

    @property
    def has_analytical_kl(self) -> bool:
        return False

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        device = self.centers.device
        comp_indices = torch.randint(0, self.n_components, (n,), device=device, generator=generator)
        selected_centers = self.centers[comp_indices]  # (n, 2)
        selected_stds = torch.exp(self.log_stds[comp_indices])  # (n, 2)

        eps = torch.randn(n, self.latent_dim, device=device, generator=generator)
        return selected_centers + selected_stds * eps

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        z_expanded = z.unsqueeze(1)  # (B, 1, 2)
        centers_expanded = self.centers.unsqueeze(0)  # (1, K, 2)
        log_stds_expanded = self.log_stds.unsqueeze(0)  # (1, K, 2)

        diff = z_expanded - centers_expanded  # (B, K, 2)
        # exp(-2 * log_std) computes 1 / sigma^2 stably without explicit division
        inv_vars = torch.exp(-2.0 * log_stds_expanded)  # (1, K, 2)
        quad = torch.sum(diff.pow(2) * inv_vars, dim=-1)  # (B, K)
        log_det = 2.0 * torch.sum(log_stds_expanded, dim=-1)  # (1, K)

        component_log_probs = -0.5 * (self.latent_dim * self.log_2pi + log_det + quad)  # (B, K)

        log_weights = torch.log(self.weights).unsqueeze(0)  # (1, K)
        return torch.logsumexp(component_log_probs + log_weights, dim=-1)  # (B,)

    def kl_divergence(
        self,
        z: torch.Tensor,
        mu_q: torch.Tensor,
        logvar_q: torch.Tensor,
    ) -> torch.Tensor:
        inv_var_q = torch.exp(-logvar_q)
        quad_q = torch.sum((z - mu_q).pow(2) * inv_var_q, dim=-1)
        log_det_q = torch.sum(logvar_q, dim=-1)
        log_q = -0.5 * (self.latent_dim * self.log_2pi + log_det_q + quad_q)

        log_p = self.log_prob(z)
        return log_q - log_p
