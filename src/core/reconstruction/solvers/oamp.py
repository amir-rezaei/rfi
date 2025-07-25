# src/core/reconstruction/solvers/oamp.py

import numpy as np
import streamlit as st
from typing import Callable, Dict, Any, Tuple


def oamp_solver(
        y: np.ndarray,
        A: np.ndarray,
        A_H: np.ndarray,
        denoiser: Callable[[np.ndarray, float], Tuple[np.ndarray, float]],
        grid_shape: Tuple[int, int],
        params: Dict[str, Any]
) -> np.ndarray:
    """
    A Plug-and-Play Orthogonal Approximate Message Passing (PnP-OAMP) solver.

    This algorithm solves inverse problems by alternating between a linear
    estimation step based on the measurement matrix A and a denoising step
    performed by a powerful, pre-trained denoiser (like a VAE).

    It solves problems of the form:
        min_x ||Ax - y||₂² + R(x)
    where R(x) is an implicit regularizer defined by the denoiser.

    Args:
        y: The complex-valued measurement vector.
        A: The measurement matrix (forward operator).
        A_H: The Hermitian transpose of A (adjoint operator).
        denoiser: A function that takes a noisy image `r` and a noise level `sigma`
                  and returns a tuple of (denoised_image, divergence).
        grid_shape: The (rows, cols) shape of the image x.
        params: A dictionary of parameters containing:
            - 'nn_iter' (int): The number of iterations.
            - 'step_size' (float): Damping factor for state updates.

    Returns:
        The reconstructed image as a 2D numpy array.
    """
    max_iter = params.get('nn_iter', 50)
    damping = params.get('step_size', 0.7)  # Damping factor for stability

    m, n = A.shape

    # --- Initialization ---
    x_k = np.zeros(n, dtype=y.dtype)  # Image estimate
    w_k = np.zeros(n, dtype=y.dtype)  # Onsager term correction

    # UI placeholders for progress reporting
    status_text = st.empty()
    progress_bar = st.progress(0)

    for i in range(max_iter):
        # 1. Linear MMSE Estimation Step
        # Pre-denoising residual
        z_k = A @ x_k

        # This is a key step in OAMP. A simple gradient step with Onsager correction
        # is a common and effective approximation in PnP settings.
        r_k = x_k + damping * (A_H @ (y - z_k)) - damping * w_k

        # Estimate the effective noise level for the denoiser input.
        # This is typically derived from state evolution, but can be approximated.
        v_k = np.mean(np.abs(y - z_k) ** 2) - m / n * np.mean(np.abs(x_k) ** 2)
        tau_k = np.sqrt(np.maximum(v_k, 1e-10))

        # 2. Non-linear Denoising Step
        # The denoiser takes the pseudo-data r_k and its estimated noise level tau_k.
        # It returns the denoised image and the divergence (average derivative).
        x_k_plus_1, divergence = denoiser(r_k.reshape(grid_shape), tau_k)
        x_k_plus_1 = x_k_plus_1.flatten()

        # Ensure divergence is a scalar average.
        if hasattr(divergence, "__len__"):
            divergence = np.mean(divergence)

        # 3. Onsager Correction Term Update
        # This term ensures asymptotic orthogonality of the error vector, which is key to OAMP's performance.
        w_k_plus_1 = (1.0 / damping) * (x_k_plus_1 - r_k) * divergence

        # Update state for the next iteration
        x_k = x_k_plus_1
        w_k = w_k_plus_1

        # Update UI
        progress_bar.progress((i + 1) / max_iter)
        status_text.text(f"OAMP Iteration: {i + 1}/{max_iter} | Est. Noise Level: {tau_k:.2e}")

    status_text.text(f"OAMP reconstruction complete after {max_iter} iterations.")

    # Return the magnitude of the final complex-valued reconstruction
    return np.abs(x_k).reshape(grid_shape)
