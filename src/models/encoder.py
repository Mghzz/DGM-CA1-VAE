"""Encoder network for VAE.

Architecture:
    Linear(20, 128) -> ReLU
    Linear(128, 128) -> ReLU
    Separate linear heads for latent mean (mu_z) and log-variance (logvar_z).
"""

from typing import Tuple
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Gaussian diagonal approximate posterior encoder q_phi(z|x).
    
    Produces mean mu_phi(x) and log-variance logvar_phi(x) for latent dimension d=2.
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dims: Tuple[int, int] = (128, 128),
        latent_dim: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Shared feature trunk
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )

        # Separate output heads for mean and log-variance
        self.mu_head = nn.Linear(hidden_dims[1], latent_dim)
        self.logvar_head = nn.Linear(hidden_dims[1], latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through encoder.
        
        Args:
            x: Input observations of shape (batch_size, 20).
            
        Returns:
            Tuple of:
                mu_z: Posterior mean of shape (batch_size, 2).
                logvar_z: Posterior log-variance of shape (batch_size, 2).
        """
        features = self.trunk(x)
        mu_z = self.mu_head(features)
        logvar_z = self.logvar_head(features)
        return mu_z, logvar_z
