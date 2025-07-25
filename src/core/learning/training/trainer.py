# src/core/learning/training/trainer.py

import torch
from torch.utils.data import DataLoader, random_split
from typing import List, Dict, Callable, Optional, Tuple

from ..models.vae import VAE
from .dataset import ImageFolderDataset
from .losses import losses_dict


def train_vae(
        # Data parameters
        data_dir: str,
        img_size: int,
        batch_size: int,
        val_split: float = 0.1,
        ext: str = 'png',
        binarize: bool = False,
        transform: Optional[Callable] = None,
        # Model parameters
        latent_dim: int = 32,
        hidden_dims: List[int] = None,
        activation: str = 'relu',
        batchnorm: bool = True,
        # Training parameters
        epochs: int = 50,
        optimizer_name: str = 'Adam',
        lr: float = 1e-3,
        # VAE-specific parameters
        loss_fn_name: str = 'BCE',
        beta: float = 1.0,
        kl_anneal_epochs: int = 10,
        loss_kwargs: Optional[Dict] = None,
        # System & I/O parameters
        device: torch.device = torch.device('cpu'),
        save_path: str = './vae_model.pth',
        update_callback: Optional[Callable] = None,
) -> Tuple[VAE, List[float], List[float]]:
    """
    Orchestrates the complete training and validation pipeline for a VAE.

    Args:
        data_dir: Path to the directory with training images.
        img_size: The size (height and width) to which images will be resized.
        batch_size: The number of samples per batch.
        val_split: The fraction of the dataset to use for validation.
        ext: The file extension of the images to load.
        binarize: If True, convert images to binary {0, 1} values.
        transform: Optional torchvision transforms to apply to the images.
        latent_dim: The dimensionality of the VAE's latent space.
        hidden_dims: A list of integers specifying the number of channels in each hidden convolutional layer.
        activation: The name of the activation function to use (e.g., 'relu').
        batchnorm: If True, use BatchNorm2d layers in the network.
        epochs: The total number of training epochs.
        optimizer_name: The name of the optimizer to use (e.g., 'Adam').
        lr: The learning rate for the optimizer.
        loss_fn_name: The name of the VAE loss function to use (e.g., 'BCE').
        beta: The final weight of the KL divergence term (for β-VAE).
        kl_anneal_epochs: The number of epochs over which to linearly increase beta from 0 to its final value.
        loss_kwargs: Additional keyword arguments for the loss function (e.g., 'l1_weight').
        device: The torch.device ('cpu' or 'cuda') to train on.
        save_path: The file path where the trained model's state dictionary will be saved.
        update_callback: An optional function to call at the end of each epoch for progress
                         reporting. It receives (epoch, train_losses, val_losses, sample_originals, sample_recons).

    Returns:
        A tuple containing:
        - The trained VAE model.
        - A list of the average training loss for each epoch.
        - A list of the average validation loss for each epoch.
    """
    # 1. Setup Dataset and DataLoaders
    dataset = ImageFolderDataset(data_dir, img_size, transform, ext, binarize)
    train_len = int(len(dataset) * (1 - val_split))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Initialize Model, Optimizer, and Loss Function
    if hidden_dims is None:
        hidden_dims = [32, 64, 128, 256]

    model = VAE(
        img_channels=1,
        img_size=img_size,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        activation=activation,
        batchnorm=batchnorm
    ).to(device)

    opt_class = getattr(torch.optim, optimizer_name)
    optimizer = opt_class(model.parameters(), lr=lr)

    loss_fn = losses_dict[loss_fn_name]
    if loss_kwargs is None:
        loss_kwargs = {}

    train_losses, val_losses = [], []

    # 3. Start Training Loop
    for epoch in range(epochs):
        # Linearly anneal the KL weight (beta) from 0 to its target value
        beta_current = min(beta, beta * (epoch + 1) / kl_anneal_epochs) if kl_anneal_epochs > 0 else beta

        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            recon_batch, mu, logvar = model(batch)

            # Calculate loss
            loss = loss_fn(recon_batch, batch, mu, logvar, beta_current, **loss_kwargs)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = val_batch.to(device)
                recon_val, mu_val, logvar_val = model(val_batch)
                val_loss = loss_fn(recon_val, val_batch, mu_val, logvar_val, beta, **loss_kwargs)
                running_val_loss += val_loss.item()

        avg_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        # --- Progress Reporting ---
        if update_callback:
            # Get a sample batch for visualization
            sample_originals = next(iter(val_loader)).to(device)
            sample_recons, _, _ = model(sample_originals)
            n_show = min(8, sample_originals.shape[0])
            update_callback(
                epoch,
                train_losses,
                val_losses,
                sample_originals[:n_show].cpu(),
                sample_recons[:n_show].cpu()
            )

    # 4. Save the trained model
    torch.save(model.state_dict(), save_path)

    return model, train_losses, val_losses




