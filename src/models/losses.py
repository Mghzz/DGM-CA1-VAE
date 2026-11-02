"""Loss functions and ELBO engine for VAE models.

Implements:
1. Exact Gaussian observation negative log-likelihood (sigma_x = 0.30):
   NLL = 0.5 * sum_d [ (x_d - mu_x_d)^2 / sigma_x^2 + log(2*pi*sigma_x^2) ]
   Summed over features, averaged over batch. Includes exact constant ~ -5.70059.
2. Analytical KL for Standard Normal Prior (Model A).
3. Monte Carlo KL for GMM Priors (Models B, C, D) using logsumexp.
"""

from dataclasses import dataclass
import math
import torch

from src.models.priors import BasePrior


@dataclass
class LossOutput:
    """Container for individual and total loss components."""
    total_loss: torch.Tensor
    recon_loss: torch.Tensor
    kl_loss: torch.Tensor


def gaussian_reconstruction_nll(
    x: torch.Tensor,
    mu_x: torch.Tensor,
    sigma_x: float = 0.30,
) -> torch.Tensor:
    """Computes exact Gaussian negative log-likelihood -log p_theta(x|z).
    
    Formula:
        NLL(x, mu_x) = 0.5 * sum_{d=1}^D [ (x_d - mu_{x,d})^2 / sigma_x^2 + log(2*pi*sigma_x^2) ]
    
    Args:
        x: Ground-truth target observations of shape (batch_size, D).
        mu_x: Reconstructed observation mean of shape (batch_size, D).
        sigma_x: Fixed likelihood standard deviation (PDF specifies 0.30).
        
    Returns:
        Scalar tensor containing the batch-averaged negative log-likelihood.
    """
    if sigma_x <= 0:
        raise ValueError(f"sigma_x must be positive, got {sigma_x}")
    if x.shape != mu_x.shape:
        raise ValueError(f"Shape mismatch: x has shape {x.shape} but mu_x has shape {mu_x.shape}")

    d = x.shape[-1]
    var_x = sigma_x ** 2
    log_2pi_var = math.log(2.0 * math.pi * var_x)

    sq_error = torch.sum((x - mu_x) ** 2, dim=-1)  # (batch_size,)
    nll_per_sample = 0.5 * (sq_error / var_x + d * log_2pi_var)  # (batch_size,)

    return nll_per_sample.mean()


def compute_vae_loss(
    x: torch.Tensor,
    mu_x: torch.Tensor,
    z: torch.Tensor,
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    prior: BasePrior,
    sigma_x: float = 0.30,
) -> LossOutput:
    """Computes total VAE negative ELBO loss and its separate components.
    
    total_loss = reconstruction_nll + KL_divergence
    
    Args:
        x: Observations of shape (batch_size, 20).
        mu_x: Reconstructed means of shape (batch_size, 20).
        z: Sampled latents of shape (batch_size, 2).
        mu_q: Posterior mean of shape (batch_size, 2).
        logvar_q: Posterior log-variance of shape (batch_size, 2).
        prior: Latent prior instance (BasePrior).
        sigma_x: Fixed likelihood standard deviation (0.30).
        
    Returns:
        LossOutput containing total_loss, recon_loss, and kl_loss.
    """
    recon_loss = gaussian_reconstruction_nll(x, mu_x, sigma_x=sigma_x)
    kl_per_sample = prior.kl_divergence(z, mu_q, logvar_q)
    kl_loss = kl_per_sample.mean()

    total_loss = recon_loss + kl_loss

    return LossOutput(
        total_loss=total_loss,
        recon_loss=recon_loss,
        kl_loss=kl_loss,
    )
