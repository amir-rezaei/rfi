# src/dashboard/ui/training.py

import streamlit as st
from typing import Dict, Any, Tuple

from src.dashboard.assets.help_texts import VAE_TRAINING_EXPLANATION
from src.core.learning.training.losses import losses_dict
import os


def display_training_config() -> Tuple[Dict[str, Any], bool]:
    """
    Renders the UI for configuring and launching a VAE training job.

    Returns:
        A tuple containing:
        - A dictionary of all training configuration parameters.
        - A boolean indicating if the 'Start Training' button was clicked.
    """
    config = {}
    st.header("🧠 Train a VAE Regularizer")

    with st.expander("ℹ️ VAE Model and Training: Detailed Explanation", expanded=False):
        st.markdown(VAE_TRAINING_EXPLANATION, unsafe_allow_html=True)

    # --- Column Layout ---
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Data & I/O")
        config['data_dir'] = st.text_input("Database Directory", value="data/raw/mpeg7")
        config['ext'] = st.selectbox("Image Extension", ["png", "jpg", "jpeg"], index=0)
        config['img_size'] = st.number_input("Image Size", 16, 256, 64, step=8,
                                             help="All images will be resized to this square dimension.")
        config['binarize'] = st.checkbox("Binarize Images", value=False,
                                         help="Convert images to strict 0/1 values based on a threshold.")

        st.subheader("Network Architecture")
        config['latent_dim'] = st.number_input("Latent Dimension", 2, 512, 32, step=1,
                                               help="The size of the VAE's compressed latent vector z.")
        n_layers = st.number_input("Number of Hidden Layers", 1, 8, 4)
        first_hidden = st.number_input("First Hidden Layer Channels", 8, 1024, 32, step=8)
        growth = st.selectbox("Layer Growth Pattern", ["Double", "Constant", "Halve"], index=0,
                              help="How the number of channels changes between layers.")

        if growth == "Double":
            hidden_dims = [first_hidden * (2 ** i) for i in range(n_layers)]
        elif growth == "Constant":
            hidden_dims = [first_hidden] * n_layers
        else:  # Halve
            hidden_dims = [max(8, first_hidden // (2 ** i)) for i in range(n_layers)]
        config['hidden_dims'] = hidden_dims

        config['activation'] = st.selectbox("Activation Function", ["relu", "leakyrelu", "elu", "selu", "tanh", "gelu"],
                                            index=0)
        config['batchnorm'] = st.checkbox("Use BatchNorm", value=True)

    with col2:
        st.subheader("Training Parameters")
        config['epochs'] = st.number_input("Epochs", 1, 1000, 50, step=1)
        config['batch_size'] = st.number_input("Batch Size", 1, 512, 32, step=1)
        config['optimizer_name'] = st.selectbox("Optimizer", ['Adam', 'SGD', 'RMSprop'], index=0)
        config['lr'] = st.number_input("Learning Rate", 1e-6, 1e-1, 1e-4, format="%.5f")
        config['val_split'] = st.slider("Validation Split", 0.01, 0.5, 0.1, step=0.01,
                                        help="Fraction of data to use for validation.")

        st.subheader("VAE Loss Configuration")
        config['loss_fn_name'] = st.selectbox("Loss Function", list(losses_dict.keys()), index=0)
        loss_kwargs = {}
        if config['loss_fn_name'] == "BCE+L1":
            l1_weight = st.number_input("L1 Weight", 0.0, 1.0, 0.1, step=0.01)
            loss_kwargs = {'l1_weight': l1_weight}
        config['loss_kwargs'] = loss_kwargs

        config['beta'] = st.number_input("KL Divergence Weight (β)", 0.0, 10.0, 1.0, format="%.2f",
                                         help="Weight of the KL term in the VAE loss.")
        config['kl_anneal_epochs'] = st.number_input("KL Annealing Epochs", 0, 100, 25,
                                                     help="Epochs to linearly ramp up β from 0 to its final value. 0 to disable.")

    st.subheader("Output")
    default_save_name = f"vae_ld{config['latent_dim']}_ep{config['epochs']}_{config['loss_fn_name']}.pth"
    config['save_path'] = st.text_input("Save Model Path", value=os.path.join("data/models", default_save_name))

    st.markdown("---")
    start_training_button = st.button("Start VAE Training", use_container_width=True, type="primary")

    return config, start_training_button



