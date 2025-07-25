# src/core/learning/training/losses.py

import torch
import torch.nn.functional as F
from typing import Dict, Callable


def vae_loss_bce(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
        **kwargs
) -> torch.Tensor:
    """
    VAE loss using Binary Cross-Entropy (BCE) for reconstruction.

    This loss is suitable for images with pixel values normalized to [0, 1],
    treating each pixel as a Bernoulli distribution.

    Args:
        recon_x: The reconstructed image tensor from the decoder.
        x: The original input image tensor.
        mu: The latent mean from the encoder.
        logvar: The latent log-variance from the encoder.
        beta: The weight for the KL divergence term (for β-VAE).

    Returns:
        The total VAE loss.
    """
    # Reconstruction loss
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence regularization
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return bce + beta * kld


def vae_loss_mse(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
        **kwargs
) -> torch.Tensor:
    """
    VAE loss using Mean Squared Error (MSE) for reconstruction.

    This loss is suitable for real-valued images and assumes a Gaussian
    likelihood for the decoder's output.

    Args:
        recon_x, x, mu, logvar: See vae_loss_bce.
        beta: The weight for the KL divergence term.

    Returns:
        The total VAE loss.
    """
    # Reconstruction loss
    mse = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence regularization
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return mse + beta * kld


def vae_loss_l1(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
        **kwargs
) -> torch.Tensor:
    """
    VAE loss using L1 (Mean Absolute Error) for reconstruction.

    L1 loss can be more robust to outliers than MSE and sometimes produces
    less blurry reconstructions.

    Args:
        recon_x, x, mu, logvar: See vae_loss_bce.
        beta: The weight for the KL divergence term.

    Returns:
        The total VAE loss.
    """
    # Reconstruction loss
    l1 = torch.abs(recon_x - x).sum()

    # KL divergence regularization
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return l1 + beta * kld


def vae_loss_bce_l1(
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0,
        l1_weight: float = 0.1,
        **kwargs
) -> torch.Tensor:
    """
    VAE loss combining BCE and L1 for reconstruction.

    This hybrid loss can leverage the probabilistic nature of BCE while using
    L1 to encourage sharper features.

    Args:
        recon_x, x, mu, logvar: See vae_loss_bce.
        beta: The weight for the KL divergence term.
        l1_weight: The weight for the L1 component of the reconstruction loss.

    Returns:
        The total VAE loss.
    """
    # Reconstruction loss
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    l1 = torch.abs(recon_x - x).sum()

    # KL divergence regularization
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return bce + l1_weight * l1 + beta * kld


# Dictionary to easily access loss functions by name
losses_dict: Dict[str, Callable] = {
    'BCE': vae_loss_bce,
    'MSE': vae_loss_mse,
    'L1': vae_loss_l1,
    'BCE+L1': vae_loss_bce_l1
}



