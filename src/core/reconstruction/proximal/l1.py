# src/core/reconstruction/proximal/l1.py
# src/core/reconstruction/proximal/l1.py

import numpy as np


def soft_threshold(u: np.ndarray, thresh: float) -> np.ndarray:
    """
    The proximal operator for the L1 norm on real values.
    This function applies element-wise soft-thresholding.
    """
    return np.sign(u) * np.maximum(np.abs(u) - thresh, 0)


def complex_soft_threshold(u: np.ndarray, thresh: float) -> np.ndarray:
    """
    Proximal operator (soft threshold) for the L1-norm on complex values.

    This shrinks the magnitude of each element toward zero by `thresh`,
    while preserving the phase. It is the key non-linear step in solvers
    like ISTA and FISTA for complex-valued LASSO problems.

    Args:
        u: Complex-valued input array.
        thresh: Non-negative threshold parameter.

    Returns:
        Thresholded complex-valued array with the same shape as `u`.
    """
    mag = np.abs(u)
    # Where mag is zero, the output is already zero, which avoids division by zero.
    scale = np.where(mag > 0, np.maximum(1.0 - thresh / mag, 0.0), 0.0)
    return u * scale


def complex_soft_threshold_divergence(u: np.ndarray, thresh: float) -> float:
    """
    Calculates the average derivative (divergence) of the complex
    soft-threshold function, required for some AMP-based algorithms.

    It is approximated by the fraction of "active" elements (those not set
    to zero by the thresholding operation).

    Args:
        u: The complex-valued numpy array to which the operator is applied.
        thresh: The threshold value used in the operator.

    Returns:
        A scalar float representing the average divergence.
    """
    if u.size == 0:
        return 0.0

    # The active set are the elements whose magnitude is greater than the threshold.
    active_elements = np.sum(np.abs(u) > thresh)

    # The divergence is the fraction of active elements over the total size.
    return active_elements / u.size
