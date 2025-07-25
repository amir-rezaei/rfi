# src/dashboard/ui/measurement.py

import streamlit as st
import numpy as np
from typing import Dict, Tuple, Any

# It's better to import the specific assets and core modules needed.
# This assumes the new structure where TerminalConfig is in core.simulation.
from src.core.simulation.terminal import TerminalConfig
from src.dashboard.assets.help_texts import (
    MEASUREMENT_MODEL_GUIDELINE,
    RESOLUTION_GUIDELINE
)

def display_measurement_config(
    terminal: TerminalConfig,
    target_params: Dict[str, Any]
) -> Tuple[Dict[str, Any], bool]:
    """
    Renders the UI for measurement generation parameters and the Generate button.

    This includes settings for the scene dimensions, simulation and reconstruction
    grid resolutions, and channel effects like noise and fading.

    Args:
        terminal: The configured TerminalConfig object, needed for live
                  resolution calculations.
        target_params: The target parameter dictionary, needed for the
                       target's range.

    Returns:
        A tuple containing:
        - A dictionary with all simulation and grid configuration parameters.
        - A boolean indicating if the 'Generate Measurements' button was clicked.
    """
    st.markdown("### 🔬 Generate Channel Measurements")
    st.caption("Configure the scene and channel effects, then click 'Generate Measurements'.")

    with st.expander("How are measurements calculated? (The Math behind y=Ax+n)", expanded=False):
        st.markdown(MEASUREMENT_MODEL_GUIDELINE, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    x_len = c1.number_input(
        "Scene Width (m)", 1.0, 50.0, 2.0, step=0.5,
        help="Size of the imaging scene in the X-direction."
    )
    y_len = c2.number_input(
        "Scene Depth (m)", 1.0, 50.0, 2.0, step=0.5,
        help="Size of the imaging scene in the Y-direction."
    )

    with st.expander("Advanced Grid & Channel Settings"):
        st.markdown("**Grid Resolution**")
        gc1, gc2 = st.columns(2)
        sim_grid = gc1.number_input(
            "Simulation Grid", 32, 512, 64, step=16,
            help="Grid resolution for generating 'ground truth' data (y=Ax). Higher is more accurate but slower."
        )
        rec_grid = gc2.number_input(
            "Reconstruction Grid", 32, 512, 64, step=16,
            help="Grid resolution for the reconstructed image. This determines the size of the matrix A."
        )

        st.markdown("**Channel Effects**")
        nc1, nc2 = st.columns(2)
        enable_noise = nc1.checkbox(
            "Enable Noise", value=True,
            help="Add Additive White Gaussian Noise (AWGN) to the measurements."
        )
        enable_fading = nc2.checkbox(
            "Enable Fading", value=False,
            help="Apply Rayleigh channel fading to the measurements."
        )

        noise_level = 30.0
        fading_variance = 0.1
        if enable_noise:
            noise_level = nc1.number_input("Noise Level (SNR in dB)", -50.0, 50.0, 30.0, step=1.0)
        if enable_fading:
            fading_variance = nc2.number_input(
                "Fading Variance", 0.0, 2.0, 0.1, step=0.05,
                help="Variance of the complex Gaussian variable for Rayleigh fading."
            )

    with st.expander("Live Resolution and Grid Guideline", expanded=False):
        st.markdown(RESOLUTION_GUIDELINE, unsafe_allow_html=True)
        st.subheader("Live Calculation")
        try:
            wavelength = terminal.wavelength
            rx_size_x, _ = map(int, terminal.rx_array_size.split('x'))
            # Use the z-coordinate of the target center as the range
            if 'positions' in target_params:
                range_r = target_params.get('positions', [[0, 0, 1]])[0][2]
            else:
                range_r = target_params.get('center', (0, 0, 1))[2]

            spacing_x_factor = (
                terminal.rx_spacing[0] if terminal.rx_spacing and terminal.rx_spacing[0] is not None else 0.5
            )
            aperture_x = (rx_size_x - 1) * (wavelength * spacing_x_factor) if rx_size_x > 1 else wavelength

            if aperture_x > 0 and range_r > 0:
                res_x = (wavelength * range_r) / aperture_x
                rec_grid_x = (2 * aperture_x * x_len) / (wavelength * range_r)
            else:
                res_x = float('inf')
                rec_grid_x = 0

            st.markdown("Your configuration yields:")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Wavelength (λ):** `{wavelength:.4f}` m")
                st.info(f"**Rx Aperture X (L_x):** `{aperture_x:.3f}` m")
            with col2:
                st.info(f"**Target Range (R):** `{range_r:.2f}` m")
                st.success(f"**Est. Resolution X (Δx):** `{res_x:.4f}` m")

            st.markdown("**Theoretically Recommended Minimum Grid Size (X-axis):**")
            st.warning(f"**`{int(np.ceil(rec_grid_x))}` pixels**")
            st.caption(f"Your currently set 'Reconstruction Grid' is **{rec_grid}x{rec_grid}**.")

        except Exception as e:
            st.error(f"Could not perform live calculation: {e}")

    generate_button = st.button("Generate Measurements", use_container_width=True, type="primary")

    config = {
        "x_range": (-x_len / 2, x_len / 2),
        "y_range": (-y_len / 2, y_len / 2),
        "sim_grid": sim_grid,
        "rec_grid": rec_grid,
        "enable_noise": enable_noise,
        "noise_level": noise_level,
        "enable_fading": enable_fading,
        "fading_variance": fading_variance
    }
    return config, generate_button



