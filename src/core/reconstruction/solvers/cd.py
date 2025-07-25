# src/core/reconstruction/solvers/cd.py

import numpy as np
import streamlit as st
from typing import Callable, Dict, Any


def coordinate_descent_solver(
        y: np.ndarray,
        s_len: int,
        forward_op: Callable[[np.ndarray], np.ndarray],
        adjoint_op: Callable[[np.ndarray], np.ndarray],
        prox_op: Callable[[np.ndarray, float], np.ndarray],
        params: Dict[str, Any]
) -> np.ndarray:
    """
    A Coordinate Descent (CD) solver for the complex LASSO (c-LASSO) problem
    in the synthesis formulation.

    This algorithm iteratively updates one coefficient of the solution vector 's'
    at a time, cycling through all coordinates until convergence. It solves:
    min_s ||forward_op(s) - y||₂² + α * ||s||₁

    NOTE: For non-identity bases (like Wavelets), this implementation can be
    computationally intensive as it needs to compute the effective dictionary
    columns (A @ D_j) on the fly or pre-compute them.

    Args:
        y: The complex-valued measurement vector.
        s_len: The length of the complex sparse coefficient vector 's' to solve for.
        forward_op: The forward operator, lambda s: A @ D(s).
        adjoint_op: The adjoint of the forward operator. Not directly used here but
                    maintained for a consistent solver interface.
        prox_op: The complex proximal operator (e.g., complex_soft_threshold).
        params: A dictionary of parameters containing:
            - 'max_iter' (int): The number of full cycles through all coordinates.
            - 'alpha' (float): The regularization strength (α).

    Returns:
        The reconstructed complex-valued sparse coefficient vector 's'.
    """
    max_iter = params.get('max_iter', 20)
    alpha = params.get('alpha', 0.1)
    dtype = y.dtype if np.iscomplexobj(y) else float

    # --- Initialization ---
    s = np.zeros(s_len, dtype=dtype)
    # Initial residual r = y - A*D*s = y, since s is zero
    residual = y.copy()

    # --- Pre-computation of effective dictionary columns ---
    # This is the computationally expensive part for a dense transform D.
    st.info("CD Solver: Pre-calculating effective dictionary columns for wavelet basis...")
    progress_bar_precompute = st.progress(0)

    A_eff_cols = np.zeros((y.shape[0], s_len), dtype=dtype)
    for j in range(s_len):
        # Create a basis vector for the j-th coefficient
        s_basis_vec = np.zeros(s_len, dtype=dtype)
        s_basis_vec[j] = 1.0
        # The j-th column of the effective dictionary is A @ D(s_basis_vec)
        A_eff_cols[:, j] = forward_op(s_basis_vec)
        if (j + 1) % 100 == 0 or j == s_len - 1:
            progress_bar_precompute.progress((j + 1) / s_len)

    st.info("CD Solver: Pre-computation complete. Starting iterations...")

    col_norms_sq = np.sum(np.abs(A_eff_cols) ** 2, axis=0)
    # Avoid division by zero for null columns
    col_norms_sq[col_norms_sq == 0] = 1.0

    # --- Main Iteration Loop ---
    status_text = st.empty()
    progress_bar_iter = st.progress(0)

    # Outer loop: cycles over the entire set of coordinates
    for i in range(max_iter):
        # Inner loop: iterate through each coordinate j
        for j in range(s_len):
            s_j_old = s[j]

            # Correlate the j-th column of the effective operator with the
            # residual that *includes* the contribution from the old s_j.
            A_eff_j = A_eff_cols[:, j]
            u_j = A_eff_j.conj().T @ (residual + A_eff_j * s_j_old)

            # Apply the scaled proximal operator to get the new coordinate value.
            s[j] = (1.0 / col_norms_sq[j]) * prox_op(u_j, alpha)

            # Update the overall residual efficiently using the change in s_j.
            delta_s_j = s[j] - s_j_old
            if np.abs(delta_s_j) > 1e-12:  # Check for meaningful change
                residual -= A_eff_j * delta_s_j

        # Update UI after each full cycle
        progress_bar_iter.progress((i + 1) / max_iter)
        status_text.text(f"Coordinate Descent Cycle: {i + 1}/{max_iter}")

    status_text.text(f"Coordinate Descent complete after {max_iter} cycles.")
    return s