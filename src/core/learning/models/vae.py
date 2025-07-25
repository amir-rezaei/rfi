import torch
import torch.nn as nn
from typing import List, Tuple


def _get_activation(name: str) -> nn.Module:
    """Helper function to get an activation layer from its name."""
    if name == 'relu':
        return nn.ReLU()
    elif name == 'leakyrelu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif name == 'elu':
        return nn.ELU(inplace=True)
    elif name == 'selu':
        return nn.SELU(inplace=True)
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'gelu':
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation function: {name}")


class Encoder(nn.Module):
    """
    The Encoder part of the VAE.

    This network takes an input image and maps it to the parameters (mean and
    log-variance) of a distribution in the latent space. It consists of a
    series of convolutional layers followed by fully connected layers.
    """

    def __init__(self, img_channels: int, hidden_dims: List[int], latent_dim: int,
                 img_size: int, activation: str = 'relu', batchnorm: bool = True):
        super().__init__()

        modules = []
        in_channels = img_channels
        # Build convolutional layers
        for h_dim in hidden_dims:
            modules.append(
                nn.Conv2d(in_channels, out_channels=h_dim,
                          kernel_size=3, stride=2, padding=1)
            )
            if batchnorm:
                modules.append(nn.BatchNorm2d(h_dim))
            modules.append(_get_activation(activation))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # Determine the flattened dimension after convolutions dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, img_channels, img_size, img_size)
            self.flatten_dim = self.encoder(dummy_input).flatten().shape[0]

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the encoder.

        Args:
            x: The input image tensor of shape (N, C, H, W).

        Returns:
            A tuple containing:
            - mu (torch.Tensor): The mean of the latent distribution.
            - log_var (torch.Tensor): The log-variance of the latent distribution.
        """
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_logvar(result)
        return mu, log_var


class Decoder(nn.Module):
    """
    The Decoder part of the VAE.

    This network takes a point from the latent space and maps it back to an
    image. It is structured as the reverse of the encoder, using transposed
    convolutional layers to upsample the signal to the original image size.
    """

    def __init__(self, out_channels: int, hidden_dims: List[int], latent_dim: int,
                 final_conv_shape: Tuple[int, int, int], activation: str = 'relu', batchnorm: bool = True):
        super().__init__()
        self.decoder_input_ch, self.decoder_input_h, self.decoder_input_w = final_conv_shape

        # Layer to project latent vector to the shape required by the first deconv layer
        self.fc = nn.Linear(latent_dim, self.decoder_input_ch * self.decoder_input_h * self.decoder_input_w)

        modules = []
        hidden_dims = hidden_dims[::-1]  # Reverse the list for upsampling

        # Build transposed convolutional layers
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i + 1],
                                   kernel_size=3, stride=2, padding=1, output_padding=1)
            )
            if batchnorm:
                modules.append(nn.BatchNorm2d(hidden_dims[i + 1]))
            modules.append(_get_activation(activation))

        self.decoder = nn.Sequential(*modules)

        # Final layer to produce the output image
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1],
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            _get_activation(activation),
            nn.Conv2d(hidden_dims[-1], out_channels=out_channels,
                      kernel_size=3, padding=1),
            nn.Sigmoid()  # Use Sigmoid to ensure output values are in [0, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        result = self.fc(z)
        result = result.view(-1, self.decoder_input_ch, self.decoder_input_h, self.decoder_input_w)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result


class VAE(nn.Module):
    """
    A Convolutional Variational Autoencoder (VAE).

    This model combines an Encoder and a Decoder to learn a generative model of
    data. It learns a mapping from the input data space to a continuous,
    structured latent space and a mapping back to the data space.
    """

    def __init__(self, img_channels: int = 1, hidden_dims: List[int] = None, latent_dim: int = 32,
                 img_size: int = 64, activation: str = 'relu', batchnorm: bool = True):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        self.encoder = Encoder(img_channels, hidden_dims, latent_dim, img_size, activation, batchnorm)

        # Get the shape before the flatten layer in the encoder to pass to the decoder
        with torch.no_grad():
            dummy_input = torch.zeros(1, img_channels, img_size, img_size)
            final_conv_shape = self.encoder.encoder(dummy_input).shape[1:]

        self.decoder = Decoder(img_channels, hidden_dims, latent_dim, final_conv_shape, activation, batchnorm)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        The reparameterization trick.

        Instead of sampling from q(z|x), we sample from a standard normal
        distribution and scale by the learned standard deviation and shift by
        the learned mean. This allows gradients to flow through the sampling process.

        Args:
            mu: The mean from the encoder's output.
            logvar: The log-variance from the encoder's output.

        Returns:
            A sample from the latent space distribution.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the full forward pass of the VAE.

        Args:
            x: Input image tensor of shape (N, C, H, W).

        Returns:
            A tuple containing:
            - recon_x (torch.Tensor): The reconstructed image.
            - mu (torch.Tensor): The latent mean.
            - log_var (torch.Tensor): The latent log-variance.
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        return recon_x, mu, log_var




