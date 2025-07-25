# src/core/simulation/forward.py

import numpy as np
from typing import Dict

from .channel import ChannelModel2D
from .terminal import TerminalConfig


def apply_rayleigh_fading(y: np.ndarray, variance: float) -> np.ndarray:
    """
    Applies complex Rayleigh fading to a signal vector.

    This simulates the effect of a channel with multiple random scattering paths
    not captured in the direct line-of-sight model. It is achieved by
    multiplying the signal by a complex Gaussian random variable. The magnitude
    of this variable follows a Rayleigh distribution.

    Args:
        y: The clean complex signal vector.
        variance: The variance of the underlying complex Gaussian distribution.
                  If variance is zero or negative, the original signal is returned.

    Returns:
        The faded complex signal vector.
    """
    if variance <= 0:
        return y

    # Generate complex Gaussian fading channel coefficients
    # E[|fading|^2] = (var/2) + (var/2) = variance
    fading = (np.random.randn(*y.shape) + 1j * np.random.randn(*y.shape)) * np.sqrt(variance / 2)

    return fading * y


def add_awgn_noise(y: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Adds complex Additive White Gaussian Noise (AWGN) to a signal vector.

    The power of the noise is calculated based on the signal's power and a
    specified Signal-to-Noise Ratio (SNR) in decibels (dB).

    Args:
        y: The complex signal vector.
        snr_db: The desired Signal-to-Noise Ratio in dB.

    Returns:
        The noisy complex signal vector (y + noise).
    """
    # Calculate the average power of the signal
    signal_power = np.mean(np.abs(y) ** 2)

    if signal_power == 0:
        return y  # No signal, so no noise is added relative to it

    # Convert SNR from dB to linear scale and calculate noise power
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # Generate complex Gaussian noise with the calculated power
    # The variance of the complex noise is `noise_power`. Each real/imag component has variance `noise_power / 2`.
    noise = (np.random.randn(*y.shape) + 1j * np.random.randn(*y.shape)) * np.sqrt(noise_power / 2)

    return y + noise


class MeasurementGenerator2D:
    """
    Generates the complex measurement vector 'y' for a 2D scene.

    This class orchestrates the entire forward simulation. It simulates the
    channel response by integrating the reflectivity of the scene with the
    channel model and then applies optional channel impairments like fading
    and noise. This effectively computes y = h*(Ax) + n.
    """

    def __init__(
            self,
            terminal: TerminalConfig,
            sim_x_grid: np.ndarray,
            sim_y_grid: np.ndarray,
            fixed_z: float
    ):
        """
        Initializes the measurement generator.

        Args:
            terminal: The configured terminal object.
            sim_x_grid: The x-coordinates of the high-resolution simulation grid.
            sim_y_grid: The y-coordinates of the high-resolution simulation grid.
            fixed_z: The z-depth of the imaging plane.
        """
        self.terminal = terminal
        self.sim_x_grid = sim_x_grid
        self.sim_y_grid = sim_y_grid
        self.fixed_z = fixed_z
        self.channel_model = ChannelModel2D(terminal, sim_x_grid, sim_y_grid, fixed_z)

    def simulate_measurement(self, reflectivity_map: np.ndarray, sim_params: Dict) -> np.ndarray:
        """
        Simulates the measurement vector y by combining scene reflectivity and channel responses.

        The process follows the linear model: y = sum_pixels(reflectivity * response) + impairments.
        This is a discretized version of the integral in the Lippmann-Schwinger equation.

        Args:
            reflectivity_map: A 2D array representing the target scene's reflectivity.
                              Shape: (len(sim_x_grid), len(sim_y_grid)).
            sim_params: A dictionary with simulation parameters, including:
                - 'enable_noise' (bool): Whether to add AWGN.
                - 'noise_level' (float): The SNR in dB for the noise.
                - 'enable_fading' (bool): Whether to apply Rayleigh fading.
                - 'fading_variance' (float): The variance for the fading channel.

        Returns:
            A 1D complex measurement vector 'y' of shape (N_tx * N_rx,).
        """
        n_tx = self.terminal.get_tx_positions().shape[0]
        n_rx = self.terminal.get_rx_positions().shape[0]
        n_x, n_y = len(self.sim_x_grid), len(self.sim_y_grid)

        # Compute the differential area of a grid cell for integral approximation
        dx = self.sim_x_grid[1] - self.sim_x_grid[0] if n_x > 1 else 1.0
        dy = self.sim_y_grid[1] - self.sim_y_grid[0] if n_y > 1 else 1.0
        cell_area = dx * dy

        # Flatten the reflectivity map for easier iteration
        refl_flat = reflectivity_map.flatten()

        # --- Vectorized Computation ---
        # This loop calculates `Ax`, where `x` is the reflectivity map.
        # It sums the contribution of each pixel to the final measurement vector.
        y = np.zeros((n_tx * n_rx,), dtype=np.complex128)
        for idx in range(len(refl_flat)):
            # Only compute responses for non-zero pixels to save time
            if refl_flat[idx] != 0:
                ix, iy = np.unravel_index(idx, (n_x, n_y))
                x_coord, y_coord = self.sim_x_grid[ix], self.sim_y_grid[iy]

                # Get the channel response for this pixel and scale it by reflectivity and area
                response = self.channel_model.get_response_for_scatterer(x_coord, y_coord)
                y += refl_flat[idx] * cell_area * response

        # --- Apply Channel Impairments ---

        # Apply Rayleigh fading if enabled
        if sim_params.get('enable_fading', False):
            y = apply_rayleigh_fading(y, sim_params.get('fading_variance', 0.1))

        # Add AWGN if enabled
        if sim_params.get('enable_noise', False):
            y = add_awgn_noise(y, sim_params.get('noise_level', 30.0))

        return y




