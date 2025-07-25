# src/core/reconstruction/solvers/ista.py
# src/core/reconstruction/solvers/ista.py
# This file contains the core, corrected implementation of the ISTA solver.

import numpy as np
import streamlit as st
from typing import Callable, Dict, Any


def _estimate_lipschitz(
        forward_op: Callable,
        adjoint_op: Callable,
        s_len: int,
        dtype: np.dtype,
        max_power_iter: int = 20
) -> float:
    """
    Estimates the Lipschitz constant L of (adjoint_op @ forward_op) using
    the power iteration method. This is the largest eigenvalue of the operator,
    which is crucial for setting a stable step size in gradient-based methods.
    """
    # Start with a random vector of the correct data type
    if np.issubdtype(dtype, np.complexfloating):
        s = np.random.randn(s_len) + 1j * np.random.randn(s_len)
    else:
        s = np.random.randn(s_len)

    s = s.astype(dtype)
    if np.linalg.norm(s) == 0:  # Avoid division by zero for an all-zero vector
        return 1.0
    s = s / np.linalg.norm(s)

    for _ in range(max_power_iter):
        # Apply the operator A_eff.H * A_eff (where A_eff = A @ D)
        s_new = adjoint_op(forward_op(s))
        norm_s_new = np.linalg.norm(s_new)
        if norm_s_new == 0:
            return 1.0
        s = s_new / norm_s_new

    # The Lipschitz constant L is the Rayleigh quotient s.H * (A_eff.H * A_eff * s) / s.H * s
    # Since s is normalized, L is the norm of the resulting vector.
    # A more stable way is to calculate ||A_eff * s||^2
    L = np.linalg.norm(forward_op(s)) ** 2

    # Return a safe, non-zero value
    return L if L > 1e-6 else 1.0


def ista_solver(
        y: np.ndarray,
        s_len: int,
        forward_op: Callable[[np.ndarray], np.ndarray],
        adjoint_op: Callable[[np.ndarray], np.ndarray],
        prox_op: Callable[[np.ndarray, float], np.ndarray],
        params: Dict[str, Any]
) -> np.ndarray:
    """
    A generic Iterative Shrinkage-Thresholding Algorithm (ISTA) solver.

    This algorithm is a form of proximal gradient descent used to solve
    L1-regularized least squares problems (LASSO). For this project, it solves
    the **synthesis formulation** common in compressed sensing:

    .. math::
        \hat{\mathbf{s}} = \\arg\min_{\mathbf{s}} \\frac{1}{2} \| \mathbf{A D s} - \mathbf{y} \|_2^2 + \\alpha \|\mathbf{s}\|_1

    where:
    - **s** is the sparse coefficient vector we want to find.
    - **D** is a synthesis dictionary (e.g., an inverse wavelet transform).
    - **A** is the measurement matrix.
    - **y** is the measurement vector.
    - **α** is the regularization strength.

    Args:
        y: The measurement vector (can be complex).
        s_len: The length of the sparse coefficient vector 's' to be solved for.
        forward_op: The effective forward operator, `A_eff = A @ D`.
        adjoint_op: The adjoint of the effective operator, `A_eff.H = D.H @ A.H`.
        prox_op: The proximal operator for the L1-norm (e.g., complex_soft_threshold).
        params: A dictionary of parameters containing:
            - 'max_iter' (int): The number of iterations.
            - 'alpha' (float): The regularization strength (α).

    Returns:
        The reconstructed sparse coefficient vector 's'.
    """
    max_iter = params.get('max_iter', 100)
    alpha = params.get('alpha', 0.1)

    # Determine dtype from the measurement vector y to handle real/complex cases
    dtype = y.dtype if np.iscomplexobj(y) else float

    # --- Step 1: Set the step size ---
    # The step size for ISTA must be <= 1/L, where L is the Lipschitz constant
    # of the gradient of the data fidelity term. L = ||A_eff.H * A_eff||_2.
    # We estimate L using power iteration.
    st.write("ISTA: Estimating Lipschitz constant for step size...")
    try:
        L = _estimate_lipschitz(forward_op, adjoint_op, s_len, dtype)
        st.write(f"ISTA: Estimated Lipschitz constant L = {L:.4f}")
    except Exception as e:
        st.warning(
            f"Power iteration for Lipschitz constant failed: {e}. Defaulting to L=1.0, which may affect convergence.")
        L = 1.0

    step_size = 1.0 / L
    threshold = alpha * step_size  # The threshold for the proximal operator

    # --- Step 2: Initialization ---
    # Initialize the sparse coefficient vector s with zeros.
    s = np.zeros(s_len, dtype=dtype)

    # UI placeholders for progress reporting
    status_text = st.empty()
    progress_bar = st.progress(0)

    # --- Step 3: Main ISTA Loop ---
    for i in range(max_iter):
        # The ISTA algorithm consists of two main steps per iteration:
        # a) A standard gradient descent step on the data fidelity term.
        # b) A proximal mapping step that applies the L1 regularization.

        # a) Gradient descent step
        # The gradient of 0.5 * ||A_eff*s - y||^2 is A_eff.H @ (A_eff*s - y)
        residual = forward_op(s) - y
        grad = adjoint_op(residual)
        s_grad_update = s - step_size * grad

        # b) Proximal mapping step (soft-thresholding)
        # This step effectively solves the L1 part of the problem.
        s = prox_op(s_grad_update, threshold)

        # Update UI periodically
        if (i + 1) % 10 == 0 or i == max_iter - 1:
            progress_bar.progress((i + 1) / max_iter)
            status_text.text(f"ISTA Iteration: {i + 1}/{max_iter}")

    status_text.text(f"ISTA reconstruction complete after {max_iter} iterations.")

    return s
