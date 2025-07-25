# app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import sys

# Add the source directory to the Python path
# This allows us to import from our refactored packages
src_path = str(Path(__file__).resolve().parent / 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# UI components
from dashboard.ui.setup import display_terminal_config, display_target_config
from dashboard.ui.measurement import display_measurement_config
from dashboard.ui.reconstruction import display_reconstruction_config
from dashboard.ui.training import display_training_config

# Dashboard logic (controller)
from dashboard.logic import run_simulation, run_reconstruction, compute_wavenumber_samples

# Plotting functions
from dashboard.plotting import (
    plot_scene_2d, plot_scene_3d, plot_measurements,
    plot_reconstruction, plot_kspace_2d, plot_kspace_3d
)

# Core modules
from core.simulation.terminal import TerminalConfig
from core.learning.training.trainer import train_vae

# Utility functions
from utils.config import get_config_hash, convert_ui_to_terminal_config, get_recon_display_name
from utils.device import get_device

# --- Page Configuration ---
st.set_page_config(page_title="ISAC RF Imaging Dashboard", page_icon="📡", layout="wide")
st.title("📡 ISAC RF Imaging Dashboard")

# --- Session State Initialization ---
# This ensures that data persists across tabs and user interactions.
if 'measurement_data' not in st.session_state:
    st.session_state.measurement_data = None
if 'reconstruction_results' not in st.session_state:
    st.session_state.reconstruction_results = {}
if 'selected_recons' not in st.session_state:
    st.session_state.selected_recons = []

# --- Main Application Tabs ---
tab_setup, tab_measure, tab_recon, tab_train = st.tabs([
    "**1. Setup**",
    "**2. Measurements**",
    "**3. Reconstruction & Analysis**",
    "**4. Train NN Regularizer**"
])

# =============================================================================
# TAB 1: SETUP
# =============================================================================
with tab_setup:
    st.header("🛠️ Configure Simulation Setup")
    setup_left, setup_right = st.columns([1, 1], gap="large")

    with setup_left:
        st.subheader("Terminal & Target")
        terminal_ui_config = display_terminal_config()
        target_type, target_params = display_target_config()
        # Convert the flat UI dict to the nested format required by TerminalConfig
        terminal_config_dict = convert_ui_to_terminal_config(terminal_ui_config)

    with setup_right:
        st.subheader("Live Previews")
        try:
            # Create a TerminalConfig instance from the current UI settings
            terminal = TerminalConfig(**terminal_config_dict)
            rx_pos = terminal.get_rx_positions()
            tx_pos = terminal.get_tx_positions()

            if 'positions' in target_params:
                target_center = target_params.get('positions', [[0, 0, 1]])[0]
            else:
                target_center = target_params.get('center', (0, 0, 1))

            kx, ky, kz, k_vecs = compute_wavenumber_samples(terminal, target_center)

            # UI for view controls
            c1, c2 = st.columns(2)
            view_plane = c1.selectbox("2D View Plane", ["XY", "XZ", "YZ"], key="view_plane_setup")
            elev = c2.slider("3D View Elevation", -90, 90, 30, key="elev_setup", step=1)
            azim = c2.slider("3D View Azimuth", -180, 180, -60, key="azim_setup", step=1)

            # Display plots
            plot_c1, plot_c2 = st.columns(2)
            with plot_c1:
                fig, ax = plt.subplots()
                plot_scene_2d(ax, rx_pos, tx_pos, target_type, target_params, plane=view_plane)
                st.pyplot(fig, use_container_width=True)

                fig, ax = plt.subplots()
                plot_kspace_2d(ax, kx, ky, kz, plane=view_plane)
                st.pyplot(fig, use_container_width=True)
            with plot_c2:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(elev=elev, azim=azim)
                plot_scene_3d(ax, rx_pos, tx_pos, target_type, target_params)
                st.pyplot(fig, use_container_width=True)

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(elev=elev, azim=azim)
                plot_kspace_3d(ax, k_vecs)
                st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate preview. Please check configurations.\nError: {e}")

# =============================================================================
# TAB 2: MEASUREMENTS
# =============================================================================
with tab_measure:
    st.header("🔬 Generate & Analyze Measurements")
    if 'terminal' not in locals():
        st.warning("Please configure the terminal in the 'Setup' tab first.")
    else:
        measurement_config, generate_button = display_measurement_config(terminal, target_params)

        if generate_button:
            with st.spinner("🔬 Simulating channel and generating measurements..."):
                st.session_state.measurement_data = run_simulation(
                    terminal, target_type, target_params, measurement_config
                )
                # Clear old results when new measurements are generated
                st.session_state.reconstruction_results.clear()
                st.session_state.selected_recons.clear()
            st.success("Measurement generation complete! Proceed to the 'Reconstruction' tab.")

        if st.session_state.measurement_data:
            st.markdown("---")
            st.subheader("Measurement Analysis")
            measurements = st.session_state.measurement_data['y']
            term_obj = st.session_state.measurement_data['terminal']
            tx_size = tuple(map(int, term_obj.tx_array_size.split('x')))
            rx_size = tuple(map(int, term_obj.rx_array_size.split('x')))

            selected_tx_index = 0
            if tx_size[0] * tx_size[1] > 1:
                st.caption("Use the selectors below to view measurements from a specific Tx antenna.")
                c1, c2 = st.columns(2)
                tx_x_idx = c1.selectbox("Tx X Index", range(tx_size[1]), key="tx_x_idx")
                tx_y_idx = c2.selectbox("Tx Y Index", range(tx_size[0]), key="tx_y_idx")
                selected_tx_index = tx_y_idx * tx_size[1] + tx_x_idx

            meas_reshaped = measurements.reshape((tx_size[0] * tx_size[1], -1))
            rx_meas_slice = meas_reshaped[selected_tx_index, :].reshape(rx_size)
            fig_meas = plt.figure(figsize=(10, 4))
            plot_measurements(fig_meas, rx_meas_slice)
            st.pyplot(fig_meas, use_container_width=True)

# =============================================================================
# TAB 3: RECONSTRUCTION & ANALYSIS
# =============================================================================
with tab_recon:
    st.header("📊 Reconstruction & Analysis")
    if not st.session_state.measurement_data:
        st.info("Please generate measurements in the 'Measurements' tab to begin.")
    else:
        with st.container(border=True):
            # This now calls your new, reimplemented UI function
            reconstruction_config = display_reconstruction_config()
            if st.button("Reconstruct Image", use_container_width=True, type="primary"):
                reconstruction_config['reconstruct_button'] = True

        if reconstruction_config.pop('reconstruct_button', False):
            method_name = reconstruction_config['method']
            with st.spinner(f"Running {method_name} reconstruction..."):
                # This now calls your new, reimplemented logic function
                recon = run_reconstruction(st.session_state.measurement_data, reconstruction_config)
                recon_hash = get_config_hash(reconstruction_config)
                # This now calls your new, reimplemented display name function
                display_name = get_recon_display_name(reconstruction_config)

                # Calculate Normalized Mean Square Error (NMSE)
                ground_truth = st.session_state.measurement_data['ground_truth_rec']
                # Normalize both images before comparing to handle scale differences
                recon_norm = recon / np.linalg.norm(recon) if np.linalg.norm(recon) > 0 else recon
                gt_norm = ground_truth / np.linalg.norm(ground_truth) if np.linalg.norm(ground_truth) > 0 else ground_truth
                nmse = np.linalg.norm(gt_norm - recon_norm) ** 2

                # Store result
                st.session_state.reconstruction_results[recon_hash] = {
                    'image': recon, 'error': nmse, 'title': display_name
                }
                if recon_hash not in st.session_state.selected_recons:
                    st.session_state.selected_recons.append(recon_hash)
            st.success(f"Added '{display_name}' to results.")

        st.markdown("---")
        st.subheader("Result Comparison")
        if not st.session_state.reconstruction_results:
            st.write("No reconstructions yet. Configure parameters above and click 'Reconstruct Image'.")
        else:
            options = list(st.session_state.reconstruction_results.keys())
            st.session_state.selected_recons = [h for h in st.session_state.selected_recons if h in options]

            selected_hashes = st.multiselect(
                "Select reconstructions to compare",
                options=options,
                format_func=lambda h: st.session_state.reconstruction_results[h]['title'],
                default=st.session_state.selected_recons
            )
            st.session_state.selected_recons = selected_hashes

            apply_smoothing = st.checkbox("Apply Gaussian smoothing to images", value=True)
            if st.button("Clear All Reconstruction Results"):
                st.session_state.reconstruction_results.clear()
                st.session_state.selected_recons.clear()
                st.rerun()

            # Plotting
            num_to_plot = len(selected_hashes) + 1  # +1 for ground truth
            cols = st.columns(min(num_to_plot, 4))

            grids = st.session_state.measurement_data['grids']
            extent = [grids['rec_x'][0], grids['rec_x'][-1], grids['rec_y'][0], grids['rec_y'][-1]]

            # Ground truth plot
            with cols[0]:
                fig, ax = plt.subplots()
                plot_reconstruction(ax, st.session_state.measurement_data['ground_truth_rec'], extent, "Ground Truth",
                                    apply_smoothing)
                st.pyplot(fig, use_container_width=True)

            # Reconstructions plots
            for i, h in enumerate(selected_hashes):
                col_index = (i + 1) % len(cols)
                with cols[col_index]:
                    result = st.session_state.reconstruction_results[h]
                    fig, ax = plt.subplots()
                    plot_reconstruction(ax, result['image'], extent, result['title'], apply_smoothing, result['error'])
                    st.pyplot(fig, use_container_width=True)

# =============================================================================
# TAB 4: TRAIN NN REGULARIZER
# =============================================================================
with tab_train:
    training_config, start_training = display_training_config()

    if start_training:
        device = get_device()
        st.info(f"Starting training on device: {device}")

        # Placeholders for live updates
        progress_bar = st.progress(0.0, "Starting...")
        chart_placeholder = st.empty()
        image_placeholder = st.empty()


        def update_callback(epoch, train_losses, val_losses, originals, recons):
            """This function is passed to the trainer to update the Streamlit UI."""
            progress_bar.progress((epoch + 1) / training_config['epochs'],
                                  f"Epoch {epoch + 1}/{training_config['epochs']}")

            # Update loss chart
            fig_loss, ax_loss = plt.subplots()
            ax_loss.plot(train_losses, label="Train Loss")
            ax_loss.plot(val_losses, label="Validation Loss")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.set_title("Training Progress")
            ax_loss.legend()
            ax_loss.grid(True)
            chart_placeholder.pyplot(fig_loss)
            plt.close(fig_loss)

            # Update image sample chart
            fig_img, axs = plt.subplots(2, originals.shape[0], figsize=(10, 3))
            for i in range(originals.shape[0]):
                axs[0, i].imshow(originals[i][0].detach().cpu().numpy(), cmap="gray")
                axs[0, i].set_title("Orig")
                axs[0, i].axis('off')
                axs[1, i].imshow(recons[i][0].detach().cpu().numpy(), cmap="gray")
                axs[1, i].set_title("Recon")
                axs[1, i].axis('off')
            fig_img.suptitle(f"Reconstructions after Epoch {epoch + 1}")
            image_placeholder.pyplot(fig_img)
            plt.close(fig_img)


        try:
            train_vae(
                **training_config,  # Pass all UI config directly
                device=device,
                update_callback=update_callback
            )
            st.success(f"Training complete! Model saved to {training_config['save_path']}")
        except Exception as e:
            st.error(f"An error occurred during training: {e}")
