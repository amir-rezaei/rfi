# src/core/reconstruction/solvers/fista.py

import numpy as np
import streamlit as st
from typing import Callable, Dict, Any


def _estimate_lipschitz(
        forward_op: Callable,
        adjoint_op: Callable,
        s_len: int,
        dtype: np.dtype,
        max_power_iter: int = 10
) -> float:
    """Estimates the Lipschitz constant L using power iteration."""
    if np.issubdtype(dtype, np.complexfloating):
        s = np.random.randn(s_len) + 1j * np.random.randn(s_len)
    else:
        s = np.random.randn(s_len)

    s = s.astype(dtype)
    if np.linalg.norm(s) > 0:
        s = s / np.linalg.norm(s)

    for _ in range(max_power_iter):
        # Power iteration step to estimate L
        s = adjoint_op(forward_op(s))
        if np.linalg.norm(s) > 0:
            s = s / np.linalg.norm(s)

    L = np.linalg.norm(forward_op(s)) ** 2
    return L if L > 1e-6 else 1.0


def fista_solver(
        y: np.ndarray,
        s_len: int,
        forward_op: Callable[[np.ndarray], np.ndarray],
        adjoint_op: Callable[[np.ndarray], np.ndarray],
        prox_op: Callable[[np.ndarray, float], np.ndarray],
        params: Dict[str, Any]
) -> np.ndarray:
    """
    A generic Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) solver.

    FISTA is an accelerated version of ISTA that uses a momentum term to achieve
    a faster convergence rate for L1-regularized least squares problems.
    It solves problems of the form: min_s ||forward_op(s) - y||₂² + α * ||s||₁

    Args:
        y: The complex-valued measurement vector.
        s_len: The length of the sparse coefficient vector 's' to be solved for.
        forward_op: The forward operator, e.g., lambda s: A @ D(s).
        adjoint_op: The adjoint of the forward operator, e.g., lambda r: D.T(A.H @ r).
        prox_op: The complex proximal operator (e.g., complex_soft_threshold).
        params: A dictionary of parameters containing:
            - 'max_iter' (int): The number of iterations.
            - 'alpha' (float): The regularization strength (α).

    Returns:
        The reconstructed complex-valued sparse coefficient vector 's'.
    """
    max_iter = params.get('max_iter', 100)
    alpha = params.get('alpha', 0.1)
    dtype = y.dtype if np.iscomplexobj(y) else float

    # Estimate the Lipschitz constant L for step size calculation
    try:
        L = _estimate_lipschitz(forward_op, adjoint_op, s_len, dtype)
    except Exception as e:
        st.warning(f"Power iteration for Lipschitz constant failed: {e}. Using L=1.0.")
        L = 1.0

    step_size = 1.0 / L
    threshold = alpha * step_size

    # UI placeholders
    status_text = st.empty()
    progress_bar = st.progress(0)

    # --- FISTA Initialization ---
    s_k_minus_1 = np.zeros(s_len, dtype=dtype)  # s_{k-1}
    z_k = np.zeros(s_len, dtype=dtype)  # Extrapolation point
    t_k = 1.0  # Momentum term

    for i in range(max_iter):
        # --- Main FISTA Loop ---

        # 1. Gradient descent step from the extrapolated point z_k
        y_hat = forward_op(z_k)
        grad = adjoint_op(y_hat - y)
        s_k = z_k - step_size * grad

        # 2. Proximal step (shrinkage/thresholding)
        s_k = prox_op(s_k, threshold)

        # 3. Update momentum terms
        t_k_plus_1 = (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0

        # 4. Update extrapolation point for the next iteration
        z_k_plus_1 = s_k + ((t_k - 1) / t_k_plus_1) * (s_k - s_k_minus_1)

        # Update state for next iteration
        s_k_minus_1 = s_k
        t_k = t_k_plus_1
        z_k = z_k_plus_1

        # Update UI periodically
        if (i + 1) % 10 == 0 or i == max_iter - 1:
            progress_bar.progress((i + 1) / max_iter)
            status_text.text(f"FISTA Iteration: {i + 1}/{max_iter}")

    status_text.text(f"FISTA reconstruction complete after {max_iter} iterations.")

    return s_k
