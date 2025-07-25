# src/core/reconstruction/l1_solvers.py

import numpy as np
import pywt
import streamlit as st
from typing import Dict, Any, Tuple, Callable

# Import the individual solver algorithms
from .solvers.ista import ista_solver
from .solvers.fista import fista_solver
from .solvers.cd import coordinate_descent_solver
from .solvers.omp import omp_solver

# Import the required proximal operators for the L1-norm
from .proximal.l1 import soft_threshold


def _get_wavelet_operators(
        grid_shape: Tuple[int, int],
        params: Dict[str, Any],
        dtype: np.dtype
) -> Tuple[Callable, Callable, int]:
    """
    Creates the wavelet synthesis (D) and analysis (D.T) operators.

    This function sets up the forward and inverse wavelet transforms which
    are used to move between the image domain and the sparse coefficient domain.

    Args:
        grid_shape: The (rows, cols) of the image.
        params: Dictionary of parameters containing wavelet_name and wavelet_level.
        dtype: The data type (e.g., complex128) for the operators.

    Returns:
        A tuple containing:
        - synthesis_op (D): Operator to convert sparse coefficients to an image.
        - analysis_op (D.T): Operator to convert an image to sparse coefficients.
        - coeff_len: The total number of coefficients.
    """
    wavelet = params.get('wavelet_name', 'db4')
    level = params.get('wavelet_level', 2)

    # Perform a dummy decomposition to get the structure and length of the coefficients
    dummy_coeffs = pywt.wavedec2(np.zeros(grid_shape, dtype=float), wavelet, level=level)
    coeff_vector, coeff_slices = pywt.coeffs_to_array(dummy_coeffs)
    coeff_len = len(coeff_vector)

    def synthesis_op(s_vec: np.ndarray) -> np.ndarray:
        """D operator: from sparse vector `s` to image `x`."""
        try:
            # Ensure vector has the correct type for the wavelet transform library
            coeffs = pywt.array_to_coeffs(s_vec.astype(float), coeff_slices, output_format='wavedec2')
            return pywt.waverec2(coeffs, wavelet).astype(dtype)
        except ValueError as e:
            st.error(f"Error during wavelet synthesis (waverec2): {e}. Check vector lengths.")
            return np.zeros(grid_shape, dtype=dtype)

    def analysis_op(x_img: np.ndarray) -> np.ndarray:
        """D.T operator: from image `x` to sparse vector `s`."""
        # Wavelet decomposition works on real-valued images
        s_coeffs = pywt.wavedec2(x_img.real, wavelet, level=level)
        s_vec, _ = pywt.coeffs_to_array(s_coeffs)
        return s_vec.astype(dtype)

    return synthesis_op, analysis_op, coeff_len


def _get_pixel_operators(grid_shape: Tuple[int, int], dtype: np.dtype):
    n_pixels = grid_shape[0] * grid_shape[1]
    def synthesis_op(s_vec: np.ndarray) -> np.ndarray:
        return s_vec.reshape(grid_shape).astype(dtype)
    def analysis_op(x_img: np.ndarray) -> np.ndarray:
        return x_img.flatten().astype(dtype)
    return synthesis_op, analysis_op, n_pixels


def solve_l1_wavelet(
        A: np.ndarray, y: np.ndarray, grid_shape: tuple, params: Dict[str, Any]
) -> np.ndarray:
    """
    Main dispatcher for L1-regularized reconstruction in the pixel or wavelet domain.

    If 'Pixel' is selected as the basis, the identity is used and s_len = n_pixels = A.shape[1].
    If 'Wavelet' is selected, s_len may not match A.shape[1] unless A is constructed for the wavelet basis (not currently supported).
    """
    solver_name = params.get('solver', 'FISTA').lower()
    basis = params.get('sparsifying_basis', 'Pixel')
    st.write(f"Solving L1 ({basis} basis, real-valued) with solver: {solver_name.upper()}...")

    if basis == 'Pixel':
        op_D, op_D_inv, s_len = _get_pixel_operators(grid_shape, dtype=float)
    else:
        op_D, op_D_inv, s_len = _get_wavelet_operators(grid_shape, params, dtype=float)

    def forward_op(s_vec: np.ndarray) -> np.ndarray:
        s_vec = s_vec.reshape(-1)
        img_x = op_D(s_vec)
        st.write(f"forward_op: s_vec.shape={s_vec.shape}, img_x.shape={img_x.shape}")
        result = (A @ img_x.flatten()).real
        st.write(f"forward_op: (A @ img_x.flatten()).shape={result.shape}")
        return result

    def adjoint_op(res: np.ndarray) -> np.ndarray:
        st.write(f"adjoint_op: res.shape={res.shape}")
        img_res = (A.conj().T @ res).reshape(grid_shape)
        st.write(f"adjoint_op: img_res.shape={img_res.shape}")
        s_vec = op_D_inv(img_res.real).flatten()
        st.write(f"adjoint_op: s_vec.shape={s_vec.shape}")
        return s_vec

    solver_map = {
        'ista': ista_solver,
        'fista': fista_solver,
        'cd': coordinate_descent_solver,
        'omp': omp_solver
    }
    if solver_name not in solver_map:
        raise NotImplementedError(f"The '{solver_name.upper()}' solver is not implemented.")
    selected_solver = solver_map[solver_name]
    prox_op = soft_threshold

    if np.iscomplexobj(y) or np.iscomplexobj(A):
        st.warning("LASSO: y and/or A are complex-valued. Only the real part will be used for reconstruction. The solution x will be real-valued.")

    st.write(f"A.shape={A.shape}, s_len={s_len}, grid_shape={grid_shape}")
    test_img = op_D(np.zeros(s_len))
    st.write(f"op_D(np.zeros(s_len)).shape={test_img.shape}")
    if A.shape[1] != s_len:
        st.error(f"Shape mismatch: A has {A.shape[1]} columns but the operator expects {s_len} coefficients.\nA.shape={A.shape}, s_len={s_len}. If using a wavelet basis, this is not supported unless A is constructed for the wavelet basis.")
        return np.zeros(grid_shape)
    if test_img.shape != grid_shape:
        st.error(f"Synthesis operator returns shape {test_img.shape}, expected {grid_shape}. Check parameters and grid size.")
        return np.zeros(grid_shape)

    try:
        s_recon = selected_solver(y.real, s_len, forward_op, adjoint_op, prox_op, params)
        x_recon = op_D(s_recon).reshape(grid_shape)
        return np.abs(x_recon)
    except Exception as e:
        st.error(f"LASSO/Wavelet reconstruction failed: {e}")
        return np.zeros(grid_shape)
