# src/core/learning/solvers/nn_solvers.py

import torch
import numpy as np
import streamlit as st
from typing import Dict, Any, Tuple

from ..models.vae import VAE
from src.core.reconstruction.solvers.oamp import oamp_solver


# #######################################
# Main Dispatcher Function
# #######################################

def solve_with_nn_prior(
        measurement_data: Dict[str, Any],
        recon_config: Dict[str, Any],
        device: torch.device,
) -> np.ndarray:
    """
    Entry point for all VAE-based learned-prior reconstructions.

    Handles both real and complex-valued images by processing real and imaginary parts separately.
    If input is complex, both parts are reconstructed and recombined.
    Adds robust error handling for model shape/latent dim mismatches.
    """
    y = measurement_data['y']
    A = measurement_data['A']
    grid_shape = measurement_data['grid_shape']

    # 1. Load the pre-trained VAE model
    try:
        model = VAE(
            img_size=grid_shape[0],
            latent_dim=recon_config.get('latent_dim', 32)
        ).to(device)
        model.load_state_dict(torch.load(recon_config['nn_model_path'], map_location=device))
        model.eval()
        st.info(f"Successfully loaded VAE model from {recon_config['nn_model_path']}")
    except Exception as e:
        st.error(f"Failed to load or initialize VAE model: {e}")
        return np.zeros(grid_shape)

    # 2. Dispatch to the correct solver based on UI config
    formulation = recon_config.get('formulation')
    solver = recon_config.get('solver')

    def run_oamp_with_vae(y_input):
        def vae_denoiser(u_noisy: np.ndarray, sigma_n: float) -> Tuple[np.ndarray, float]:
            mu = recon_config.get('mu_penalty', 0.5)
            with torch.no_grad():
                u_torch = torch.from_numpy(u_noisy).float().to(device).unsqueeze(0).unsqueeze(0)
                u_denoised_torch, _, _ = model(u_torch)
                u_pnp = (1 - mu) * u_torch + mu * u_denoised_torch
                epsilon = 1e-4
                v = torch.randn_like(u_torch)
                v = v / torch.linalg.norm(v)
                u_perturbed_denoised, _, _ = model(u_torch + epsilon * v)
                u_perturbed_pnp = (1 - mu) * (u_torch + epsilon * v) + mu * u_perturbed_denoised
                divergence_val = torch.real(torch.sum(v * (u_perturbed_pnp - u_pnp))) / epsilon
                divergence_val /= u_torch.numel()
                return u_pnp.cpu().numpy(), divergence_val.item()
        x_recon = oamp_solver(
            y_input, A, A.conj().T,
            vae_denoiser,
            grid_shape,
            recon_config
        )
        return x_recon

    # --- Complex-valued handling ---
    if np.iscomplexobj(y):
        st.warning("Input is complex-valued. Real and imaginary parts will be processed separately and recombined. If you want only the real part, use a real-valued measurement.")
        try:
            x_recon_real = run_oamp_with_vae(y.real)
            x_recon_imag = run_oamp_with_vae(y.imag)
            x_recon_complex = x_recon_real + 1j * x_recon_imag
            return np.abs(x_recon_complex)
        except Exception as e:
            st.error(f"NN-Prior (OAMP) complex-valued reconstruction failed: {e}")
            return np.zeros(grid_shape)
    else:
        try:
            return run_oamp_with_vae(y)
        except Exception as e:
            st.error(f"NN-Prior (OAMP) reconstruction failed: {e}")
            return np.zeros(grid_shape)


# #######################################
# Plug-and-Play OAMP Solver
# #######################################

def _solve_pnp_oamp(
        model: VAE, A: np.ndarray, y: np.ndarray, grid_shape: tuple,
        config: Dict, device: torch.device
) -> np.ndarray:
    """
    Solves the inverse problem using OAMP with a VAE denoiser.
    This corresponds to the "Image Space Penalty" formulation solved via PnP-OAMP.
    """
    st.write("Running PnP-OAMP with VAE Denoiser...")

    # --- Define the VAE-based Denoiser for PnP-OAMP ---
    def vae_denoiser(u_noisy: np.ndarray, sigma_n: float) -> Tuple[np.ndarray, float]:
        """
        The VAE acts as the denoiser η(u). The penalty weight mu is used as a
        mixing parameter to control the strength of the projection onto the learned manifold.
        """
        # The 'mu_penalty' from the UI controls the mixing strength.
        mu = config.get('mu_penalty', 0.5)

        with torch.no_grad():
            # Convert numpy input to a torch tensor suitable for the model
            u_torch = torch.from_numpy(u_noisy.real).float().to(device).unsqueeze(0).unsqueeze(0)

            # Get the denoised version by passing through the VAE
            u_denoised_torch, _, _ = model(u_torch)

            # PnP update combines the noisy signal with the denoised signal
            u_pnp = (1 - mu) * u_torch + mu * u_denoised_torch

            # Estimate the divergence of the denoiser using a Monte Carlo perturbation
            # This is crucial for the Onsager correction term in OAMP.
            epsilon = 1e-4
            # Create a random perturbation vector
            v = torch.randn_like(u_torch)
            v = v / torch.linalg.norm(v) # Normalize the perturbation

            # Denoise the perturbed input
            u_perturbed_denoised, _, _ = model(u_torch + epsilon * v)
            u_perturbed_pnp = (1 - mu) * (u_torch + epsilon * v) + mu * u_perturbed_denoised

            # Calculate divergence: <v, (f(u+ev)-f(u))/e>
            divergence_val = torch.real(torch.sum(v * (u_perturbed_pnp - u_pnp))) / epsilon
            divergence_val /= u_torch.numel()  # Return the average divergence

            # Return the denoised image (as a numpy array) and the calculated divergence
            return u_pnp.cpu().numpy(), divergence_val.item()

    # --- Run the main OAMP solver ---
    # The oamp_solver function contains the iterative logic.
    x_recon = oamp_solver(
        y, A, A.conj().T,
        vae_denoiser,
        grid_shape,
        config  # Pass the full config dict which includes nn_iter, etc.
    )

    return x_recon
