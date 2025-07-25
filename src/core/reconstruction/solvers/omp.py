# src/core/reconstruction/solvers/omp.py

import numpy as np
import streamlit as st
from typing import Callable, Dict, Any


def omp_solver(
        y: np.ndarray,
        s_len: int,
        forward_op: Callable[[np.ndarray], np.ndarray],
        adjoint_op: Callable[[np.ndarray], np.ndarray],
        prox_op: Callable[[np.ndarray, float], np.ndarray],  # Not used by OMP, but kept for interface consistency
        params: Dict[str, Any]
) -> np.ndarray:
    """
    Solves the L1-regularized problem using Orthogonal Matching Pursuit (OMP).

    This implementation is designed for the synthesis model where the goal is to
    find a sparse coefficient vector 's' such that y ≈ forward_op(s).
    OMP is a greedy algorithm that iteratively selects the most correlated
    atom from the dictionary and then projects the signal onto the space
    spanned by the selected atoms.

    Args:
        y: The complex-valued measurement vector.
        s_len: The length of the sparse coefficient vector 's' to be solved for.
        forward_op: The forward operator, e.g., lambda s: A @ D(s).
        adjoint_op: The adjoint of the forward operator, e.g., lambda r: D.T(A.H @ r).
        prox_op: The proximal operator, not used by OMP but included for a consistent solver interface.
        params: A dictionary of parameters containing:
            - 'max_iter' (int): The number of non-zero coefficients to select
                               (the sparsity level k).
            - 'alpha' (float): Regularization strength, used here as a residual
                               threshold for early stopping.

    Returns:
        The reconstructed complex-valued sparse coefficient vector 's'.
    """
    # In OMP, 'max_iter' is interpreted as the target sparsity (number of non-zeros)
    sparsity_k = params.get('max_iter', 50)
    # A small residual norm can be a secondary stopping criterion
    tol = params.get('alpha', 1e-4)

    # UI placeholders for progress reporting
    status_text = st.empty()
    progress_bar = st.progress(0)

    # --- Initialization ---
    # Residual 'r' starts as the measurement vector itself
    residual = y.copy()
    # The solution vector 's' starts at zero
    s_recon = np.zeros(s_len, dtype=y.dtype)
    # The support set 'S' of selected atom indices starts empty
    support_indices = []
    # Matrix of selected atoms (columns of the effective dictionary A_eff = A@D)
    selected_atoms = np.array([]).reshape(y.shape[0], 0)
    k = 0

    for k in range(sparsity_k):
        # 1. Matching Step: Find the atom most correlated with the current residual
        # This is done by applying the adjoint operator to the residual.
        # The result is a vector of correlations with all atoms.
        correlations = adjoint_op(residual)
        # Find the index of the atom with the maximum absolute correlation
        # We must exclude atoms that are already in the support set
        correlations[support_indices] = 0
        new_index = np.argmax(np.abs(correlations))

        # Check for stagnation
        if new_index in support_indices:
            st.warning("OMP stagnated: No new atom provides significant correlation.")
            break

        # Add the new index to the support set
        support_indices.append(new_index)

        # 2. Orthogonalization Step
        # Construct the newly selected atom (the new column of A_eff)
        # This is done by applying the forward operator to a basis vector
        s_basis_vec = np.zeros(s_len, dtype=y.dtype)
        s_basis_vec[new_index] = 1.0
        new_atom = forward_op(s_basis_vec).reshape(-1, 1)

        # Add the new atom to our matrix of selected atoms
        selected_atoms = np.hstack([selected_atoms, new_atom])

        # Solve the least-squares problem: min_x || y - A_S * x ||^2
        # where A_S is the matrix of selected_atoms.
        # The solution gives the optimal coefficients for the atoms in the support.
        s_k, _, _, _ = np.linalg.lstsq(selected_atoms, y, rcond=None)

        # 3. Update Step
        # Update the full sparse coefficient vector 's' with the new values
        s_recon[support_indices] = s_k

        # Update the residual using the new coefficients
        # r = y - A_eff * s = y - A_S * s_k
        residual = y - forward_op(s_recon)

        # Update UI
        progress_bar.progress((k + 1) / sparsity_k)
        status_text.text(f"OMP Iteration: {k + 1}/{sparsity_k} (Selected atom {new_index})")

        # Check for early stopping based on residual norm
        if np.linalg.norm(residual) < tol:
            st.success(f"OMP converged early at iteration {k + 1}.")
            break

    status_text.text(f"OMP reconstruction complete after {k + 1} iterations.")
    return s_recon