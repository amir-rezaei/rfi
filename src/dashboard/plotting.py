# src/dashboard/plotting.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from typing import Dict, Any

# Import from the new 'core' package structure
from src.core.simulation.scene import get_polygon_vertices


def _set_equal_aspect_3d(ax):
    """Sets aspect ratio of a 3D plot to be equal."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_scene_2d(
        ax: plt.Axes,
        rx_pos: np.ndarray,
        tx_pos: np.ndarray,
        target_type: str,
        target_params: Dict[str, Any],
        plane: str = "XY"
):
    """
    Plots a 2D orthogonal projection of the terminal antennas and target.
    This corrected version properly projects 2D shapes as lines for XZ and YZ views.

    Args:
        ax: The Matplotlib axes object to plot on.
        rx_pos: Array of Rx antenna coordinates, shape (N_rx, 3).
        tx_pos: Array of Tx antenna coordinates, shape (N_tx, 3).
        target_type: The string identifier for the target type.
        target_params: Dictionary of parameters defining the target.
        plane: The 2D plane to project onto ('XY', 'XZ', or 'YZ').
    """
    axis_map = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}
    idx0, idx1 = axis_map.get(plane, (0, 1))

    # Plot antenna positions
    ax.scatter(tx_pos[:, idx0], tx_pos[:, idx1], c='red', marker='*', label='Tx Array', s=40, zorder=10)
    ax.scatter(rx_pos[:, idx0], rx_pos[:, idx1], c='blue', marker='o', label='Rx Array', s=20, zorder=10)

    # Plot target based on its type
    if target_type == "points":
        positions = np.array(target_params.get('positions', []))
        if positions.size > 0:
            ax.scatter(positions[:, idx0], positions[:, idx1], c='green', s=50, label='Target', edgecolors='black')

    elif target_type == "2d_shapes":
        shapes_to_draw = target_params.get('generated_shapes', [])
        if not shapes_to_draw and target_params.get('shape') and target_params.get('shape') != 'n_random_shapes':
            shapes_to_draw = [target_params]

        for i, params in enumerate(shapes_to_draw):
            center = params.get('center', (0, 0, 0))
            # Create 3D vertices for the shape in its local XY plane, then translate
            verts_2d = get_polygon_vertices(params)
            if verts_2d is not None:
                verts_3d = np.c_[verts_2d, np.full(verts_2d.shape[0], 0)]  # Add z=0 locally
                verts_3d[:, 0] -= center[0]  # De-center before re-centering
                verts_3d[:, 1] -= center[1]
                verts_3d += np.array(center)  # Translate to final 3D position

                # Project the 3D vertices onto the chosen 2D plane
                projected_verts = verts_3d[:, [idx0, idx1]]
                patch = Polygon(projected_verts, closed=True, color='green', alpha=0.5,
                                label='Target' if i == 0 else None)
                ax.add_patch(patch)

    elif target_type in ["upload_file", "sketch"] and 'img_array' in target_params:
        center = target_params.get('center', (0, 0, 0))
        scale = target_params.get('scale', 2.0)
        # If viewing top-down (XY), show the image
        if plane == "XY":
            extent = [center[0] - scale / 2, center[0] + scale / 2,
                      center[1] - scale / 2, center[1] + scale / 2]
            ax.imshow(target_params['img_array'].T, extent=extent, origin='lower', cmap='Greens', alpha=0.8)
            ax.plot([], [], color='green', linewidth=5, label='Target')  # Dummy artist for legend
        # If viewing from the side (XZ or YZ), show a line representing the plane
        else:
            p_idx = 0 if plane == "XZ" else 1
            ax.plot([center[p_idx] - scale / 2, center[p_idx] + scale / 2],
                    [center[2], center[2]], 'g-', linewidth=3, label='Target')

    # Final plot adjustments
    ax.set_xlabel(f'{plane[0]} (m)')
    ax.set_ylabel(f'{plane[1]} (m)')
    ax.set_title(f'2D Scene Projection ({plane} Plane)')
    ax.legend()
    ax.grid(True, linestyle='--')
    ax.set_aspect('equal', adjustable='box')


def plot_scene_3d(ax: plt.Axes, rx_pos: np.ndarray, tx_pos: np.ndarray, target_type: str, target_params: dict):
    """
    Plots the full 3D view of terminal antennas and the target.
    This corrected version renders all target types in 3D.
    """
    ax.scatter(tx_pos[:, 0], tx_pos[:, 1], tx_pos[:, 2], c='red', label='Tx Antennas', marker='*')
    ax.scatter(rx_pos[:, 0], rx_pos[:, 1], rx_pos[:, 2], c='blue', label='Rx Antennas', marker='o')

    def draw_shape_3d(params):
        """Helper to draw a single shape in 3D."""
        verts_2d = get_polygon_vertices(params)
        if verts_2d is not None:
            center = params.get('center', (0, 0, 0))
            # Add the Z coordinate to the 2D vertices to place them in the 3D scene
            verts_3d = np.c_[verts_2d[:, 0], verts_2d[:, 1], np.full(verts_2d.shape[0], center[2])]
            collection = Poly3DCollection([verts_3d], alpha=0.5, facecolor='green')
            ax.add_collection3d(collection)

    # Plot target based on its type
    if target_type == "points":
        pts = np.array(target_params.get('positions', []))
        if pts.size > 0:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='green', s=50, label='Target')
    elif target_type == "2d_shapes":
        shapes_to_draw = target_params.get('generated_shapes', [])
        if not shapes_to_draw and target_params.get('shape') and target_params.get('shape') != 'n_random_shapes':
            shapes_to_draw = [target_params]
        for params in shapes_to_draw:
            draw_shape_3d(params)
    elif target_type in ["upload_file", "sketch"] and 'img_array' in target_params:
        img_array = target_params['img_array']
        scale = target_params.get('scale', 2.0)
        center = target_params.get('center', (0, 0, 0))
        # Create a grid for the surface plot
        x_surf = np.linspace(center[0] - scale / 2, center[0] + scale / 2, img_array.shape[1])
        y_surf = np.linspace(center[1] - scale / 2, center[1] + scale / 2, img_array.shape[0])
        xx, yy = np.meshgrid(x_surf, y_surf)
        zz = np.full_like(xx, center[2])
        # Map the image reflectivity to colors and plot as a surface
        face_colors = plt.cm.Greens(img_array)
        ax.plot_surface(xx, yy, zz, facecolors=face_colors, rstride=4, cstride=4, shade=False, label='Target')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title("3D Scene and Terminal Setup")
    # Create a dummy artist for the 3D shape legend
    if target_type != "points":
        ax.add_collection3d(Poly3DCollection([], alpha=0.5, facecolor='green', label='Target'))
    ax.legend()
    _set_equal_aspect_3d(ax)


def plot_measurements(fig: plt.Figure, measurement_slice: np.ndarray):
    """Plots side-by-side measurement magnitude and phase for a selected Tx antenna."""
    fig.clear()
    ax1, ax2 = fig.subplots(1, 2)
    fig.suptitle("Measurements at Rx Array (for one Tx)")

    ny, nx = measurement_slice.T.shape
    xticks = [0, (nx - 1) // 2, nx - 1] if nx > 1 else [0]
    yticks = [0, (ny - 1) // 2, ny - 1] if ny > 1 else [0]

    im1 = ax1.imshow(np.abs(measurement_slice).T, origin='lower', cmap='viridis', aspect='equal')
    fig.colorbar(im1, ax=ax1, label='Magnitude (Linear)')
    ax1.set_title("Magnitude")
    ax1.set_xlabel("Rx Element Index (X)")
    ax1.set_ylabel("Rx Element Index (Y)")
    ax1.set_xticks(xticks)
    ax1.set_yticks(yticks)

    im2 = ax2.imshow(np.angle(measurement_slice).T, origin='lower', cmap='twilight_shifted', aspect='equal')
    fig.colorbar(im2, ax=ax2, label='Phase (Radians)')
    ax2.set_title("Phase")
    ax2.set_xlabel("Rx Element Index (X)")
    ax2.set_ylabel("Rx Element Index (Y)")
    ax2.set_xticks(xticks)
    ax2.set_yticks(yticks)
    plt.tight_layout(rect=[0, 0, 1, 0.95])


def plot_reconstruction(ax: plt.Axes, image: np.ndarray, extent: list, title: str,
                        smooth: bool = True, error: float = None):
    """
    Plots a 2D reconstructed image.

    Args:
        ax: The Matplotlib axes to plot on.
        image: The 2D numpy array of the image to plot.
        extent: The [xmin, xmax, ymin, ymax] boundaries for the plot.
        title: The title for the plot.
        smooth: If True, applies a Gaussian filter to smooth the image for display.
        error: Optional NMSE error to display in the title.
    """
    if image is None or image.size == 0:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        return

    img_to_plot = np.abs(image.T)
    # Normalize for consistent color mapping
    max_val = img_to_plot.max()
    if max_val > 0:
        img_to_plot /= max_val

    if smooth:
        img_to_plot = gaussian_filter(img_to_plot, sigma=0.8)

    ax.imshow(img_to_plot, cmap='viridis', extent=extent, origin='lower', aspect='equal', vmin=0, vmax=1)

    full_title = title
    if error is not None:
        full_title += f"\nNMSE: {error:.4f}"

    ax.set_title(full_title, fontsize=10)
    ax.set_xlabel("X (m)", fontsize=8)
    ax.set_ylabel("Y (m)", fontsize=8)


def plot_kspace_2d(ax: plt.Axes, kx: np.ndarray, ky: np.ndarray, kz: np.ndarray, plane: str):
    """Plots the 2D projection of the wavenumber domain sampling (k-space)."""
    axis_map = {"XY": (kx, ky), "XZ": (kx, kz), "YZ": (ky, kz)}[plane]
    ax.scatter(axis_map[0], axis_map[1], c='purple', alpha=0.6, s=12, edgecolors='none')
    ax.set_xlabel(f'$k_{{{plane[0].lower()}}}$ (rad/m)')
    ax.set_ylabel(f'$k_{{{plane[1].lower()}}}$ (rad/m)')
    ax.set_title(f"Wavenumber Sampling ({plane} plane)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')


def plot_kspace_3d(ax: plt.Axes, k_vectors: np.ndarray):
    """Plots the 3D view of the wavenumber domain sampling."""
    ax.scatter(k_vectors[:, 0], k_vectors[:, 1], k_vectors[:, 2], c='purple', alpha=0.5, s=15, depthshade=True)
    ax.set_xlabel('$k_x$ (rad/m)')
    ax.set_ylabel('$k_y$ (rad/m)')
    ax.set_zlabel('$k_z$ (rad/m)')
    ax.set_title('3D Wavenumber Domain Sampling')
    _set_equal_aspect_3d(ax)



