"""Decoder network for VAE.

Architecture:
    Linear(2, 128) -> ReLU
    Linear(128, 128) -> ReLU
    Linear(128, 20) (Linear output, no activation)

The output represents the mean mu_theta(z) of the Gaussian observation likelihood.
"""

from typing import Tuple
import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Gaussian observation likelihood decoder p_theta(x|z).
    
    Produces observation mean mu_theta(z) of shape (batch_size, 20) from latent vector z (d=2).
    """

    def __init__(
        self,
        latent_dim: int = 2,
        hidden_dims: Tuple[int, int] = (128, 128),
        output_dim: int = 20,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder.
        
        Args:
            z: Latent vectors of shape (batch_size, 2).
            
        Returns:
            mu_x: Reconstructed observation mean of shape (batch_size, 20).
        """
        return self.net(z)
