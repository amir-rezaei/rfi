# src/core/simulation/channel.py

import numpy as np
from typing import Tuple

from .terminal import TerminalConfig

def create_target_grid(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    grid_size: int,
    fixed_z: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates uniformly spaced 1D arrays for a 2D grid's coordinates.

    This grid represents the imaging plane where the target's reflectivity
    will be defined and later reconstructed.

    Args:
        x_range: A tuple (min_x, max_x) defining the grid's extent along the x-axis in meters.
        y_range: A tuple (min_y, max_y) defining the grid's extent along the y-axis in meters.
        grid_size: The number of points along each axis, resulting in a (grid_size x grid_size) grid.
        fixed_z: The constant z-coordinate (depth) of the imaging plane in meters.

    Returns:
        A tuple of two numpy arrays (x_coords, y_coords), representing the
        coordinates along each axis of the grid.
    """
    x_coords = np.linspace(x_range[0], x_range[1], grid_size)
    y_coords = np.linspace(y_range[0], y_range[1], grid_size)
    return x_coords, y_coords


class ChannelModel2D:
    """
    Models the 2D channel response from scatterers to a Tx-Rx antenna array.

    This class is central to the physics of the imaging problem. It implements
    the discretized forward model based on the Born approximation. Its primary
    responsibilities are:
    1. Calculating the channel response vector for a single pixel (scatterer).
    2. Constructing the full measurement matrix 'A', which linearly maps the
       discretized scene reflectivity 'x' to the measurement vector 'y' (y = Ax).
    """

    def __init__(
        self,
        terminal: TerminalConfig,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        fixed_z: float
    ):
        """
        Initializes the 2D channel model.

        Args:
            terminal: A TerminalConfig instance containing the Tx/Rx antenna array configurations.
            x_grid: A 1D numpy array of x-coordinates for the reconstruction grid.
            y_grid: A 1D numpy array of y-coordinates for the reconstruction grid.
            fixed_z: The fixed z-depth of the 2D imaging plane.
        """
        self.terminal = terminal
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.fixed_z = fixed_z

        self.tx_pos = self.terminal.get_tx_positions()  # Shape (N_tx, 3)
        self.rx_pos = self.terminal.get_rx_positions()  # Shape (N_rx, 3)
        self.wavelength = self.terminal.wavelength
        self.k = 2 * np.pi / self.wavelength  # Wavenumber in rad/m

    def get_response_for_scatterer(self, x: float, y: float) -> np.ndarray:
        """
        Computes the channel response vector for a single scatterer.

        This function calculates the complex signal received at all Rx antennas
        from all Tx antennas, assuming a single point scatterer is located at
        (x, y, fixed_z). The formula for each Tx-Rx pair is derived from the
        discretized Lippmann-Schwinger equation under the Born approximation:

        h = exp(-j * k * (d_tx + d_rx)) / (4 * pi * d_tx * d_rx)

        where d_tx and d_rx are the distances from the transmitter and receiver
        to the scatterer, respectively.

        Args:
            x: The x-coordinate of the scatterer.
            y: The y-coordinate of the scatterer.

        Returns:
            A 1D complex numpy array of shape (N_tx * N_rx,) containing the
            channel responses, flattened in column-major order.
        """
        scatter_pos = np.array([x, y, self.fixed_z])

        # Calculate distances from all Tx antennas to the scatterer -> shape (N_tx,)
        d_tx = np.linalg.norm(self.tx_pos - scatter_pos, axis=1)

        # Calculate distances from all Rx antennas to the scatterer -> shape (N_rx,)
        d_rx = np.linalg.norm(self.rx_pos - scatter_pos, axis=1)

        # Use broadcasting to compute total path distance for all Tx-Rx pairs -> shape (N_tx, N_rx)
        total_dist = d_tx[:, np.newaxis] + d_rx[np.newaxis, :]

        # Compute the attenuation factor (d_tx * d_rx) for all pairs
        attenuation = d_tx[:, np.newaxis] * d_rx[np.newaxis, :]
        # Prevent division by zero for co-located antennas/scatterers
        attenuation[attenuation == 0] = 1e-9

        # Calculate the complex phase delay term
        phase_term = np.exp(-1j * self.k * total_dist)

        # Combine phase and attenuation to get the full response matrix
        response_matrix = phase_term / (4 * np.pi * attenuation)

        # Flatten to a 1D vector, matching the measurement vector's structure
        return response_matrix.flatten()

    def construct_measurement_matrix(self) -> np.ndarray:
        """
        Constructs the full channel measurement matrix A.

        This matrix maps the entire flattened reflectivity map (vector x) to the
        measurement vector (y = Ax). Each column of A corresponds to the channel
        response from a single pixel on the grid, scaled by the pixel's area
        to approximate a continuous surface integral.

        Returns:
            A complex numpy array of shape (N_tx * N_rx, n_pixels), where
            n_pixels is len(x_grid) * len(y_grid).
        """
        N_tx = self.tx_pos.shape[0]
        N_rx = self.rx_pos.shape[0]
        n_x = len(self.x_grid)
        n_y = len(self.y_grid)
        n_pixels = n_x * n_y

        A = np.zeros((N_tx * N_rx, n_pixels), dtype=np.complex128)

        # Compute the differential area of a single pixel for scaling
        dx = self.x_grid[1] - self.x_grid[0] if n_x > 1 else 1.0
        dy = self.y_grid[1] - self.y_grid[0] if n_y > 1 else 1.0
        cell_area = dx * dy

        # Iterate through each pixel on the grid to compute its corresponding column in A
        for col_idx, (ix, iy) in enumerate(np.ndindex(n_y, n_x)):
            # Get the coordinates of the current pixel's center
            pixel_x, pixel_y = self.x_grid[ix], self.y_grid[iy]

            # Calculate the response vector for this single pixel
            pixel_response = self.get_response_for_scatterer(pixel_x, pixel_y)

            # Scale by cell area and assign to the appropriate column of A
            A[:, col_idx] = cell_area * pixel_response

        return A




