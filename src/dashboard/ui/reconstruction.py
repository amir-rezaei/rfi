# src/dashboard/ui/reconstruction.py

import streamlit as st
import pywt
import os
from typing import Dict, Any, List

# Import the detailed explanation texts for each method.
# These provide context to the user within the UI.
from src.dashboard.assets.help_texts import (
    BP_DETAILS,
    L1_DETAILS,
    NN_DETAILS,
    WAVELET_LEVEL_HELP
)


def _get_available_nn_models(models_dir: str = "data/models") -> List[str]:
    """
    Scans the specified directory for .pth model files and returns a list of their names.
    This is used to populate the model selection dropdown in the UI.

    Args:
        models_dir: The path to the directory containing trained models.

    Returns:
        A list of model filenames, or a message if none are found.
    """
    if not os.path.isdir(models_dir):
        # Display a helpful warning in the UI if the directory doesn't exist.
        st.warning(f"Model directory not found: '{models_dir}'. Create it and place your .pth files inside.")
        return ["<no models found>"]
    try:
        # List all files in the directory that end with the .pth extension.
        files = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
        return files if files else ["<no models found>"]
    except Exception as e:
        st.error(f"Could not read from model directory: {e}")
        return ["<error reading directory>"]


def display_reconstruction_config() -> Dict[str, Any]:
    """
    Renders the UI for reconstruction algorithm selection and parameter configuration.
    Ensures all parameters required by the backend are surfaced, validated, and explained.
    Returns a dictionary containing the selected method and all its configured parameters.
    """
    config: Dict[str, Any] = {}

    # Get grid shape for dynamic parameter ranges
    grid_shape = (64, 64)
    if 'measurement_data' in st.session_state and st.session_state['measurement_data'] is not None:
        grid_shape = st.session_state['measurement_data'].get('grid_shape', (64, 64))

    st.markdown("### ⚙️ Select Reconstruction Algorithm")
    st.caption("Choose an algorithm and configure its parameters. Detailed explanations are in the expanders below.")

    method_options = [
        "Back Projection",
        "LASSO",
        "NN-Prior (OAMP)"
    ]
    method = st.selectbox("Method", method_options, index=1, key="recon_method")
    config["method"] = method
    st.markdown("---")

    if method == "Back Projection":
        st.info("Back Projection is a direct method and has no parameters to configure.")

    elif method == "LASSO":
        st.markdown("##### LASSO Parameters")
        st.info("Solves the L1-regularized inverse problem (LASSO), promoting a sparse solution in the wavelet domain.")
        c1, c2 = st.columns(2)
        config['alpha'] = c1.number_input(
            "Regularization Strength (α)",
            value=0.01, min_value=1e-6, max_value=1e4, format="%.6f",
            help="Controls the sparsity of the solution. Higher alpha forces more wavelet coefficients to zero."
        )
        config['max_iter'] = c2.number_input(
            "Max Iterations",
            min_value=1, max_value=10000, value=200,
            help="Maximum number of iterations for the solver."
        )
        c1, c2 = st.columns(2)
        wavelist = pywt.wavelist(kind='discrete')
        config['wavelet_name'] = c1.selectbox(
            "Wavelet Family",
            wavelist,
            index=wavelist.index('db4') if 'db4' in wavelist else 0,
            help="The wavelet family used for the sparsifying transform."
        )
        try:
            max_level = pywt.dwtn_max_level(grid_shape, config['wavelet_name'])
            config['wavelet_level'] = c2.number_input(
                "Decomposition Level",
                min_value=1, max_value=max(1, max_level), value=2,
                help="Wavelet decomposition level. Higher levels mean coarser features."
            )
        except Exception:
            st.warning("Could not determine max wavelet level for this wavelet. Defaulting to a max of 10.")
            config['wavelet_level'] = c2.number_input("Decomposition Level", 1, 10, 2)
        solver_options = ["ISTA", "FISTA", "CD", "OMP"]
        config['solver'] = st.selectbox(
            "LASSO Solver",
            solver_options,
            index=solver_options.index("FISTA"),
            help="Select the algorithm for solving the LASSO problem. FISTA is usually fastest."
        )

    elif method == "NN-Prior (OAMP)":
        st.markdown("##### NN-Prior (OAMP) Parameters")
        st.info("Uses a pre-trained VAE as a denoiser within the OAMP algorithm (Image Space Penalty formulation). All parameters must match the trained model.")
        c1, c2 = st.columns([2, 1])
        model_dir = c1.text_input("Model Directory", value="data/models")
        def _get_available_nn_models(model_dir):
            try:
                files = os.listdir(model_dir)
                return [f for f in files if f.endswith('.pt') or f.endswith('.pth')]
            except Exception:
                return ["<no models found>"]
        models = _get_available_nn_models(model_dir)
        model_file = c1.selectbox("Select Trained Model", models)
        if model_file and model_file not in ["<no models found>"]:
            config['nn_model_path'] = os.path.join(model_dir, model_file)
        else:
            config['nn_model_path'] = None
        config['latent_dim'] = c2.number_input(
            "Model Latent Dim", min_value=2, max_value=512, value=32,
            help="Must match the latent dimension of the loaded VAE model."
        )
        st.markdown("###### OAMP Solver Settings")
        p1, p2, p3 = st.columns(3)
        config['nn_iter'] = p1.number_input(
            "Max Iterations", min_value=1, max_value=1000, value=50, key="oamp_iter",
            help="Number of OAMP iterations."
        )
        config['mu_penalty'] = p2.number_input(
            "Penalty Weight (μ)", min_value=0.0, max_value=100.0, value=0.5, format="%.4f", key="oamp_mu",
            help="Controls how strictly the solution conforms to the NN's learned manifold."
        )
        config['step_size'] = p3.number_input(
            "Damping Factor", min_value=0.1, max_value=1.0, value=0.7, format="%.2f", key="oamp_damping",
            help="Damping factor for OAMP updates to improve stability."
        )

    return config
