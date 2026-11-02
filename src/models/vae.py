"""Unified Variational Autoencoder (VAE) architecture and model factory.

Integrates:
- Encoder: q_phi(z|x)
- Reparameterization trick
- Decoder: p_theta(x|z)
- Latent Priors: Models A, B, C, D
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from configs.config import ProjectConfig
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.priors import (
    BasePrior,
    StandardNormalPrior,
    FixedGMM8Prior,
    LearnableGMMPrior,
)
from src.models.losses import LossOutput, compute_vae_loss


class VAE(nn.Module):
    """Unified VAE model for 20D observation and 2D latent space."""

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        prior: BasePrior,
        likelihood_sigma: float = 0.30,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.likelihood_sigma = likelihood_sigma

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Applies reparameterization trick: z = mu + exp(0.5 * logvar) * eps.
        
        Args:
            mu: Posterior mean of shape (batch_size, 2).
            logvar: Posterior log-variance of shape (batch_size, 2).
            generator: Optional dedicated generator for deterministic validation.
            
        Returns:
            Sampled latent vector z of shape (batch_size, 2).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.shape, device=mu.device, dtype=mu.dtype, generator=generator)
        return mu + std * eps

    def forward(
        self,
        x: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass through VAE.
        
        Args:
            x: Input observations of shape (batch_size, 20).
            generator: Optional dedicated generator for validation sampling.
            
        Returns:
            Dictionary containing:
                mu_x: Reconstructed observation mean (batch_size, 20).
                z: Sampled latent vector (batch_size, 2).
                mu_z: Latent posterior mean (batch_size, 2).
                logvar_z: Latent posterior log-variance (batch_size, 2).
        """
        mu_z, logvar_z = self.encoder(x)
        z = self.reparameterize(mu_z, logvar_z, generator=generator)
        mu_x = self.decoder(z)

        return {
            "mu_x": mu_x,
            "z": z,
            "mu_z": mu_z,
            "logvar_z": logvar_z,
        }

    def compute_loss(
        self,
        x: torch.Tensor,
        forward_output: Dict[str, torch.Tensor],
    ) -> LossOutput:
        """Computes total ELBO loss and its components from forward outputs."""
        return compute_vae_loss(
            x=x,
            mu_x=forward_output["mu_x"],
            z=forward_output["z"],
            mu_q=forward_output["mu_z"],
            logvar_q=forward_output["logvar_z"],
            prior=self.prior,
            sigma_x=self.likelihood_sigma,
        )

    @torch.no_grad()
    def sample(
        self,
        n: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Generates new samples from the model prior.
        
        PDF Specification:
        Sample z ~ p(z), decode directly using decoder mean without adding observation noise.
        
        Args:
            n: Number of samples to generate.
            generator: Optional dedicated random generator.
            
        Returns:
            Generated observation vectors of shape (n, 20).
        """
        z = self.prior.sample(n, generator=generator)
        mu_x = self.decoder(z)
        return mu_x

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic reconstruction using posterior mean mu_phi(x).
        
        PDF Specification (Stage 4a):
        Reconstruct test data using the encoder posterior mean without sampling noise.
        
        Args:
            x: Observations of shape (batch_size, 20).
            
        Returns:
            Reconstructed observations of shape (batch_size, 20).
        """
        mu_z, _ = self.encoder(x)
        mu_x = self.decoder(mu_z)
        return mu_x


def build_vae(
    model_type: str,
    config: Optional[ProjectConfig] = None,
) -> VAE:
    """Factory function creating VAE instances for Models A, B, C, or D.
    
    Guarantees that all four models share identical encoder and decoder architectures,
    identical latent dimension (d=2), and identical observation likelihood (sigma=0.30).
    
    Args:
        model_type: One of 'A', 'B', 'C', 'D'.
        config: Optional project configuration instance.
        
    Returns:
        Configured VAE instance.
    """
    if config is None:
        config = ProjectConfig()

    model_type = model_type.upper()
    if model_type not in ["A", "B", "C", "D"]:
        raise ValueError(f"Unknown model_type '{model_type}'. Expected one of ['A', 'B', 'C', 'D'].")

    # Identical Encoder and Decoder across all models
    encoder = Encoder(
        input_dim=config.model.input_dim,
        hidden_dims=config.model.hidden_dims,
        latent_dim=config.model.latent_dim,
    )
    decoder = Decoder(
        latent_dim=config.model.latent_dim,
        hidden_dims=config.model.hidden_dims,
        output_dim=config.model.input_dim,
    )

    # Model-specific prior
    if model_type == "A":
        prior = StandardNormalPrior(latent_dim=config.model.latent_dim)
    elif model_type == "B":
        prior = FixedGMM8Prior(
            radius=config.model.b_radius,
            covariance=config.model.b_cov,
            n_components=8,
            latent_dim=config.model.latent_dim,
        )
    elif model_type == "C":
        prior = LearnableGMMPrior(
            n_components=8,
            init_std=config.model.init_std_gmm,
            latent_dim=config.model.latent_dim,
        )
    elif model_type == "D":
        prior = LearnableGMMPrior(
            n_components=4,
            init_std=config.model.init_std_gmm,
            latent_dim=config.model.latent_dim,
        )

    vae = VAE(
        encoder=encoder,
        decoder=decoder,
        prior=prior,
        likelihood_sigma=config.model.likelihood_sigma,
    )

    return vae
