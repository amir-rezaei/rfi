# src/dashboard/logic.py

import numpy as np
import streamlit as st
from typing import Dict, Any, Tuple
from pathlib import Path

# --- Simulation Imports ---
# These are required for the simulation tab and are unchanged.
from src.core.simulation.terminal import TerminalConfig
from src.core.simulation.forward import MeasurementGenerator2D
from src.core.simulation.channel import create_target_grid, ChannelModel2D
from src.core.simulation.scene import rasterize_target_on_grid, generate_random_shapes

# --- REIMPLEMENTED Reconstruction Imports ---
# These imports now point to the specific solvers required by the new UI.
from src.core.reconstruction.classical import back_projection
from src.core.reconstruction.l1_solvers import solve_l1_wavelet
from src.core.learning.solvers.nn_solvers import solve_with_nn_prior

# --- Utility Imports ---
from src.utils.device import get_device


def compute_wavenumber_samples(terminal: TerminalConfig, target_pos: tuple) -> tuple:
    """
    Computes the sampled k-space vectors based on the system geometry.
    This function remains unchanged.

    Args:
        terminal: The configured TerminalConfig object.
        target_pos: The (x,y,z) position of the target's center.

    Returns:
        A tuple (kx, ky, kz, k_vectors_flat) containing the flattened
        components and the full 3D k-space vectors.
    """
    k0 = 2 * np.pi / terminal.wavelength
    tx_pos = terminal.get_tx_positions()
    rx_pos = terminal.get_rx_positions()
    target = np.array(target_pos).reshape(1, 1, 3)

    # Unit vectors from each antenna to the target center
    u_tx = target - tx_pos[:, np.newaxis, :]
    u_tx /= np.linalg.norm(u_tx, axis=2, keepdims=True)

    u_rx = target - rx_pos[np.newaxis, :, :]
    u_rx /= np.linalg.norm(u_rx, axis=2, keepdims=True)

    # The bistatic k-space vector is the sum of the incident and scattered wave vectors
    k_vectors = k0 * (u_tx + u_rx)
    k_vectors_flat = k_vectors.reshape(-1, 3)

    return k_vectors_flat[:, 0], k_vectors_flat[:, 1], k_vectors_flat[:, 2], k_vectors_flat


def run_simulation(
        terminal: TerminalConfig,
        target_type: str,
        target_params: Dict[str, Any],
        meas_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Orchestrates the full simulation pipeline based on UI configurations.
    Ensures that the simulation and reconstruction grids are created with consistent physical extents.
    Warns if there is a mismatch in grid extents or resolutions.

    Returns:
        A dictionary containing all simulation artifacts needed for
        reconstruction and analysis, including explicit grid mapping.
    """
    # Handle the special case of generating random shapes for the scene
    if target_type == '2d_shapes' and target_params.get('shape') == 'n_random_shapes':
        n_shapes = target_params.get('n_shapes', 5)
        base_center = target_params.get('center', (0, 0, 1))
        scene_width = meas_config['x_range'][1] - meas_config['x_range'][0]
        target_params['generated_shapes'] = generate_random_shapes(n_shapes, base_center, scene_width * 0.8)

    # Determine the fixed z-depth for the 2D simulation
    if 'generated_shapes' in target_params:
        fixed_z = target_params.get('center', (0, 0, 1))[2]
    elif target_type == 'points':
        fixed_z = target_params.get('positions', [[0, 0, 1]])[0][2]
    else:
        fixed_z = target_params.get('center', (0, 0, 1))[2]

    # Create distinct grids for simulation and reconstruction
    sim_x, sim_y = create_target_grid(meas_config['x_range'], meas_config['y_range'], meas_config['sim_grid'], fixed_z)
    rec_x, rec_y = create_target_grid(meas_config['x_range'], meas_config['y_range'], meas_config['rec_grid'], fixed_z)

    # --- Grid Consistency Checks ---
    sim_extent = (sim_x[0], sim_x[-1], sim_y[0], sim_y[-1])
    rec_extent = (rec_x[0], rec_x[-1], rec_y[0], rec_y[-1])
    if not (np.allclose(sim_extent, rec_extent, atol=1e-6)):
        st.warning(f"Simulation and reconstruction grids have different physical extents!\nSim: {sim_extent}\nRec: {rec_extent}\nThis may cause artifacts or misalignment in reconstruction.")
    if (len(sim_x) < len(rec_x)) or (len(sim_y) < len(rec_y)):
        st.warning("Reconstruction grid is higher resolution than simulation grid. This may cause interpolation artifacts.")

    # Generate ground truth reflectivity maps on both grids
    ground_truth_rec = rasterize_target_on_grid(target_type, target_params, rec_x, rec_y, fixed_z)
    ground_truth_sim = rasterize_target_on_grid(target_type, target_params, sim_x, sim_y, fixed_z)

    # 1. Simulate measurements 'y' using the high-resolution simulation grid
    meas_gen = MeasurementGenerator2D(terminal, sim_x, sim_y, fixed_z)
    y = meas_gen.simulate_measurement(ground_truth_sim, meas_config)

    # 2. Compute the measurement matrix 'A' using the reconstruction grid
    channel_model = ChannelModel2D(terminal, rec_x, rec_y, fixed_z)
    A = channel_model.construct_measurement_matrix()

    return {
        "y": y,
        "A": A,
        "terminal": terminal,
        "ground_truth_sim": ground_truth_sim,
        "ground_truth_rec": ground_truth_rec,
        "grids": {"sim_x": sim_x, "sim_y": sim_y, "rec_x": rec_x, "rec_y": rec_y, "fixed_z": fixed_z},
        "grid_shape": (len(rec_x), len(rec_y)),
        "grid_mapping": {
            "sim_extent": sim_extent,
            "rec_extent": rec_extent,
            "sim_resolution": (len(sim_x), len(sim_y)),
            "rec_resolution": (len(rec_x), len(rec_y)),
        },
        "target_type": target_type,
        "target_params": target_params,
    }


def reconstruct_image(
    measurement_data: Dict[str, Any],
    recon_config: Dict[str, Any]
) -> np.ndarray:
    """
    Centralized, robust reconstruction API for Tab 3 backend.
    Validates parameters, dispatches to the correct solver, and handles errors.
    Checks that the measurement and reconstruction grids are consistent in shape and extent.
    Returns the reconstructed image as a numpy array (2D, real-valued).
    """
    # --- Parameter Validation ---
    required_keys = ["A", "y", "grid_shape", "grid_mapping"]
    for k in required_keys:
        if k not in measurement_data:
            st.error(f"Measurement data missing required key: {k}")
            return np.zeros((32, 32))  # fallback shape
    grid_mapping = measurement_data["grid_mapping"]
    if not np.allclose(grid_mapping["sim_extent"], grid_mapping["rec_extent"], atol=1e-6):
        st.warning(f"Reconstruction grid extent does not match simulation grid extent.\nSim: {grid_mapping['sim_extent']}\nRec: {grid_mapping['rec_extent']}")
    if (grid_mapping["rec_resolution"][0] > grid_mapping["sim_resolution"][0]) or (grid_mapping["rec_resolution"][1] > grid_mapping["sim_resolution"][1]):
        st.warning("Reconstruction grid is higher resolution than simulation grid. This may cause interpolation artifacts.")
    A = measurement_data["A"]
    y = measurement_data["y"]
    grid_shape = measurement_data["grid_shape"]
    method = recon_config.get("method")
    if not method:
        st.error("No reconstruction method specified.")
        return np.zeros(grid_shape)

    # --- Method Dispatch ---
    if method == "Back Projection":
        # No parameters needed
        try:
            return back_projection(y, A, grid_shape)
        except Exception as e:
            st.error(f"Back Projection failed: {e}")
            return np.zeros(grid_shape)

    elif method == "LASSO":
        # Validate LASSO params
        alpha = recon_config.get("alpha", 0.01)
        max_iter = recon_config.get("max_iter", 200)
        wavelet_name = recon_config.get("wavelet_name", "db4")
        wavelet_level = recon_config.get("wavelet_level", 2)
        solver = recon_config.get("solver", "FISTA")
        if alpha <= 0 or max_iter < 1:
            st.error("Invalid LASSO parameters: alpha must be > 0, max_iter >= 1.")
            return np.zeros(grid_shape)
        lasso_params = dict(recon_config)
        lasso_params.update({
            "alpha": alpha,
            "max_iter": max_iter,
            "wavelet_name": wavelet_name,
            "wavelet_level": wavelet_level,
            "solver": solver
        })
        try:
            return solve_l1_wavelet(A, y, grid_shape, lasso_params)
        except Exception as e:
            st.error(f"LASSO reconstruction failed: {e}")
            return np.zeros(grid_shape)

    elif method == "NN-Prior (OAMP)":
        # Validate NN params
        nn_model_path = recon_config.get("nn_model_path")
        latent_dim = recon_config.get("latent_dim", 32)
        nn_iter = recon_config.get("nn_iter", 50)
        mu_penalty = recon_config.get("mu_penalty", 0.5)
        step_size = recon_config.get("step_size", 0.7)
        if not nn_model_path or not Path(nn_model_path).exists():
            st.error("No valid Neural Network Model selected. Please choose a model in the reconstruction settings.")
            return np.zeros(grid_shape)
        nn_params = dict(recon_config)
        nn_params.update({
            "nn_model_path": nn_model_path,
            "latent_dim": latent_dim,
            "nn_iter": nn_iter,
            "mu_penalty": mu_penalty,
            "step_size": step_size,
            "formulation": "Image Space Penalty",
            "solver": "OAMP"
        })
        device = get_device()
        try:
            return solve_with_nn_prior(measurement_data, nn_params, device)
        except Exception as e:
            st.error(f"NN-Prior (OAMP) reconstruction failed: {e}")
            return np.zeros(grid_shape)

    else:
        st.error(f"Unknown reconstruction method specified: {method}")
        return np.zeros(grid_shape)

# Replace the old run_reconstruction with a call to the new API
def run_reconstruction(measurement_data: Dict[str, Any], recon_config: Dict[str, Any]) -> np.ndarray:
    return reconstruct_image(measurement_data, recon_config)
