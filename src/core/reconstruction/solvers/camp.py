# src/core/reconstruction/solvers/camp.py

import numpy as np
import streamlit as st
from typing import Callable, Dict, Any, Tuple


def camp_solver(
        y: np.ndarray,
        s_len: int,
        forward_op: Callable[[np.ndarray], np.ndarray],
        adjoint_op: Callable[[np.ndarray], np.ndarray],
        prox_op: Callable[[np.ndarray, float], np.ndarray],
        divergence_op: Callable[[np.ndarray, float], float],
        params: Dict[str, Any]
) -> np.ndarray:
    """
    A generic Complex Approximate Message Passing (CAMP) solver.

    This algorithm is generalized to work with operator handles for problems
    with fast transforms (e.g., Wavelet). It solves problems of the form:

        min_s  || forward_op(s) - y ||₂² + τ * g(s)

    where `prox_op` is the proximal operator for g(s) and `divergence_op`
    is its average derivative.

    Args:
        y: The complex-valued measurement vector of shape (m,).
        s_len: The length of the complex sparse coefficient vector `s` to be solved for.
        forward_op: The forward operator, e.g., lambda s: A @ D @ s.
        adjoint_op: The adjoint of the forward operator, e.g., lambda r: D.H @ A.H @ r.
        prox_op: The complex proximal operator (denoiser η).
        divergence_op: A function to compute the average divergence <η'> of the prox_op.
        params: A dictionary of parameters containing:
            - 'max_iter' (int): The number of iterations.
            - 'alpha' (float): The regularization strength, used here as the threshold tau.

    Returns:
        The reconstructed complex-valued sparse coefficient vector s.
    """
    max_iter = params.get('max_iter', 50)
    # In c-LASSO, the regularization strength alpha directly serves as the threshold tau
    tau = params.get('alpha', 0.1)

    m = y.shape[0]
    n = s_len
    if n == 0:
        st.warning("Coefficient vector length is zero. Cannot run CAMP.")
        return np.zeros(0, dtype=np.complex128)

    delta = m / n  # Undersampling ratio

    # Initialization
    s_t = np.zeros(n, dtype=np.complex)
    z_t = y.copy()  # The residual state is initialized with the measurements

    # UI placeholders for progress reporting
    status_text = st.empty()
    progress_bar = st.progress(0)

    for i in range(max_iter):
        # 0) residual variance and adaptive threshold ------------------------
        sigma2 = np.mean(np.abs(z_t) ** 2)  # σ_t^2
        tau_t = params.get('alpha', 0.1) * np.sqrt(sigma2)

        # 1. Form the "pseudo-data" by applying the adjoint operator to the current residual state
        pseudo_data = s_t + adjoint_op(z_t)

        # 2. Calculate the Onsager correction term *before* updating s_t
        divergence = divergence_op(pseudo_data, tau_t)
        onsager_term = (1 / delta) * z_t * divergence

        # 3. Denoising Step: Apply the proximal operator to get the new estimate for s
        s_t_plus_1 = prox_op(pseudo_data, tau_t)

        # 4. Residual Update: Update z_t using the new s_t and the Onsager term
        z_t_plus_1 = y - forward_op(s_t_plus_1) + onsager_term

        # Update state for the next iteration
        s_t, z_t = s_t_plus_1, z_t_plus_1

        # Update UI periodically
        if (i + 1) % 5 == 0 or i == max_iter - 1:
            progress_bar.progress((i + 1) / max_iter)
            status_text.text(f"CAMP Iteration: {i + 1}/{max_iter}")

    status_text.text(f"CAMP reconstruction complete after {max_iter} iterations.")

    return s_t




