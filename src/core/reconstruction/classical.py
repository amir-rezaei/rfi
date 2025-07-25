# src/core/reconstruction/classical.py

import numpy as np
from typing import Tuple

def back_projection(
    measurements: np.ndarray,
    channel_matrix: np.ndarray,
    grid_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Performs image reconstruction using the Back-Projection algorithm.

    Back-Projection is a simple and fast method based on the matched filter
    concept. It is equivalent to applying the Hermitian transpose (adjoint) of
    the measurement matrix to the measurement vector:

    $$ \hat{\mathbf{x}} = \mathbf{A}^H \mathbf{y} $$

    While not a true inverse, it correlates the measurements with the expected
    responses from each pixel, effectively "smearing" the data back onto the
    imaging grid.

    Args:
        measurements: The complex measurement vector y.
        channel_matrix: The complex measurement matrix A.
        grid_shape: A tuple (rows, cols) defining the 2D shape of the output.

    Returns:
        The reconstructed reflectivity map as a 2D real-valued array,
        representing the magnitude of the result.
    """
    # Applying the Hermitian transpose of A to y is the core of back-projection.
    reconstruction_vec = channel_matrix.conj().T @ measurements

    # The result is complex; for visualization, we take the absolute value (magnitude)
    # and reshape it from a vector back into a 2D image.
    return np.abs(reconstruction_vec).reshape(grid_shape)
