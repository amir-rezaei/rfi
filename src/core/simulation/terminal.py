# src/core/simulation/terminal.py

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class TerminalConfig:
    """
    Configuration and geometry of an ISAC terminal, including Tx/Rx antenna arrays.

    This class defines the terminal's physical properties, such as power and
    frequency, as well as the geometry of its transmitter (Tx) and receiver (Rx)
    antenna arrays. It provides methods to compute the precise 3D global
    coordinates of each antenna element based on position, orientation, offsets,
    and antenna spacing.

    Attributes:
        tx_power: Transmit power in dBm.
        frequency: Carrier frequency in GHz.
        tx_array_size: A string "NxM" defining the Tx antenna array dimensions (e.g., "4x4").
        rx_array_size: A string "NxM" defining the Rx antenna array dimensions.
        position: The 3D base position (x, y, z) of the terminal's coordinate system origin in meters.
        orientation: Tuple of rotation angles (elevation, azimuth, tilt) in degrees,
                     applied in the order: Tilt (Y-axis), Azimuth (Z-axis), Elevation (X-axis).
        tx_offset: A 3D vector for the offset of the Tx array's center relative to the terminal's base position.
        rx_offset: A 3D vector for the offset of the Rx array's center relative to the terminal's base position.
        tx_spacing: Optional tuple (spacing_x, spacing_y) defining antenna element spacing
                    as a multiple of the wavelength. Defaults to half-wavelength if None.
        rx_spacing: Optional tuple for Rx antenna spacing, similar to tx_spacing.
    """
    tx_power: float
    frequency: float
    tx_array_size: str
    rx_array_size: str
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float]
    tx_offset: Tuple[float, float, float]
    rx_offset: Tuple[float, float, float]
    tx_spacing: Optional[Tuple[Optional[float], Optional[float]]] = None
    rx_spacing: Optional[Tuple[Optional[float], Optional[float]]] = None

    def __post_init__(self):
        """Validates the configuration parameters after initialization."""
        self._validate_params()

    def _validate_params(self):
        """Performs validation checks on the terminal parameters."""
        if not (-20 <= self.tx_power <= 30):
            raise ValueError("Tx power must be between -20 and 30 dBm")
        if not (2.0 <= self.frequency <= 300.0):
            raise ValueError("Frequency must be between 2 and 300 GHz")

        for array_size, name in [(self.tx_array_size, "Tx"), (self.rx_array_size, "Rx")]:
            try:
                n_x, n_y = map(int, array_size.split('x'))
                if not (1 <= n_x <= 128 and 1 <= n_y <= 128):
                    raise ValueError()
            except Exception:
                raise ValueError(f"{name} array size must be 'NxM' with values between 1 and 128.")

        el, az, tilt = self.orientation
        if not (-90 <= el <= 90):
            raise ValueError("Elevation must be between -90 and 90 degrees.")
        if not (-180 <= az <= 180):
            raise ValueError("Azimuth must be between -180 and 180 degrees.")
        if not (-90 <= tilt <= 90):
            raise ValueError("Tilt must be between -90 and 90 degrees.")

    @property
    def wavelength(self) -> float:
        """Calculates the signal wavelength in meters from the frequency in GHz."""
        return 3e8 / (self.frequency * 1e9)

    def _compute_array_positions(
        self,
        array_size: str,
        offset: Tuple[float, float, float],
        spacing_factors: Optional[Tuple[Optional[float], Optional[float]]]
    ) -> np.ndarray:
        """
        Computes the 3D global coordinates for each antenna element in an array.

        This method generates a local 2D grid of antenna positions centered at
        the origin, then applies rotation and translation to place it correctly
        in the global coordinate system.

        Args:
            array_size: The "NxM" size string for the array.
            offset: The 3D offset vector for this array from the terminal's base.
            spacing_factors: The spacing multipliers relative to the wavelength.

        Returns:
            A numpy array of shape (N_antennas, 3) containing the global
            (x, y, z) coordinates of each antenna element.
        """
        n_y, n_x = map(int, array_size.split('x'))  # Note: NxM corresponds to (cols, rows)

        # Default to half-wavelength spacing if not provided
        spacing_x_factor = spacing_factors[0] if spacing_factors and spacing_factors[0] is not None else 0.5
        spacing_y_factor = spacing_factors[1] if spacing_factors and spacing_factors[1] is not None else 0.5

        spacing_x = self.wavelength * spacing_x_factor
        spacing_y = self.wavelength * spacing_y_factor

        # Create a 2D grid of local antenna positions centered at the origin
        x_coords = np.linspace(-(n_x - 1) / 2, (n_x - 1) / 2, n_x) * spacing_x
        y_coords = np.linspace(-(n_y - 1) / 2, (n_y - 1) / 2, n_y) * spacing_y
        xx, yy = np.meshgrid(x_coords, y_coords)
        local_positions = np.stack([xx.flatten(), yy.flatten(), np.zeros(n_x * n_y)], axis=1)

        # Convert orientation from degrees to radians for trigonometric functions
        el_rad, az_rad, tilt_rad = np.radians(self.orientation)

        # Define rotation matrices for elevation (X-axis), azimuth (Z-axis), and tilt (Y-axis)
        R_el = np.array([
            [1, 0, 0],
            [0, np.cos(el_rad), -np.sin(el_rad)],
            [0, np.sin(el_rad), np.cos(el_rad)]
        ])
        R_az = np.array([
            [np.cos(az_rad), -np.sin(az_rad), 0],
            [np.sin(az_rad), np.cos(az_rad), 0],
            [0, 0, 1]
        ])
        R_tilt = np.array([
            [np.cos(tilt_rad), 0, np.sin(tilt_rad)],
            [0, 1, 0],
            [-np.sin(tilt_rad), 0, np.cos(tilt_rad)]
        ])

        # Combine rotation matrices. The order (Tilt -> Azimuth -> Elevation) is crucial.
        R = R_tilt @ R_az @ R_el

        # Apply rotation to local positions, then add offset and base position
        # to get global coordinates.
        global_positions = local_positions @ R.T + np.array(offset) + np.array(self.position)

        return global_positions

    def get_tx_positions(self) -> np.ndarray:
        """Computes and returns the 3D global coordinates of all Tx antennas."""
        return self._compute_array_positions(self.tx_array_size, self.tx_offset, self.tx_spacing)

    def get_rx_positions(self) -> np.ndarray:
        """Computes and returns the 3D global coordinates of all Rx antennas."""
        return self._compute_array_positions(self.rx_array_size, self.rx_offset, self.rx_spacing)




