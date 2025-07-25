# src/utils/device.py

import torch
from typing import Optional

def get_device() -> torch.device:
    """
    Checks for CUDA availability and returns the appropriate torch.device.

    Returns:
        A torch.device object, either 'cuda' if a GPU is available, or 'cpu'.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(data: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Moves a tensor to the specified device.

    If no device is specified, it uses the device returned by get_device().

    Args:
        data: The PyTorch tensor to move.
        device: The target device.

    Returns:
        The tensor on the specified device.
    """
    if device is None:
        device = get_device()
    return data.to(device)

def from_device(data: torch.Tensor) -> torch.Tensor:
    """
    Moves a tensor from any device back to the CPU.

    This is often required before converting a tensor to a NumPy array or
    using it with libraries that do not support CUDA.

    Args:
        data: The PyTorch tensor to move.

    Returns:
        The tensor on the CPU.
    """
    return data.cpu()

