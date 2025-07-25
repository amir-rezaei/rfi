# src/core/simulation/scene.py

import numpy as np
from matplotlib.path import Path
from skimage.transform import resize
from typing import Dict, List, Optional, Tuple


def rasterize_target_on_grid(
        target_type: str,
        target_params: Dict,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        fixed_z: float
) -> np.ndarray:
    """
    Generates a 2D reflectivity map by rasterizing a target onto a grid.

    This is a key function that translates a high-level description of a target
    (e.g., its shape, position, and parameters) into a discrete grid that can be
    used in the linear forward model y = Ax.

    Args:
        target_type: The category of the target. One of 'points', '2d_shapes',
                     'upload_file', or 'sketch'.
        target_params: A dictionary containing the specific parameters for the
                       target, such as positions, shape type, radius, etc.
        x_grid: A 1D numpy array of the grid's x-coordinates.
        y_grid: A 1D numpy array of the grid's y-coordinates.
        fixed_z: The z-depth of the imaging plane. The function will only
                 rasterize targets located on this plane.

    Returns:
        A 2D numpy array of shape (len(x_grid), len(y_grid)) representing the
        reflectivity map, with values typically between 0 and 1.
    """
    nx, ny = len(x_grid), len(y_grid)
    refl_map = np.zeros((nx, ny), dtype=np.float32)

    if target_type == 'points':
        positions = target_params.get('positions', [])
        for x0, y0, z0 in positions:
            # Only consider points that lie on the imaging plane
            if np.isclose(z0, fixed_z):
                ix = np.argmin(np.abs(x_grid - x0))
                iy = np.argmin(np.abs(y_grid - y0))
                refl_map[ix, iy] = 1.0

    elif target_type == '2d_shapes':
        # Handle both single shapes and lists of generated shapes
        shapes = target_params.get('generated_shapes', [])
        if not shapes and target_params.get('shape') is not None:
            shapes = [target_params]

        for shape_params in shapes:
            # Ensure the shape's center is on the imaging plane
            if np.isclose(shape_params.get('center', (0, 0, 0))[2], fixed_z):
                single_map = _rasterize_single_shape(shape_params, x_grid, y_grid)
                # Combine shapes using a logical OR
                refl_map = np.logical_or(refl_map, single_map)
        refl_map = refl_map.astype(np.float32)

    elif target_type in ['upload_file', 'sketch']:
        img = target_params.get('img_array')
        if img is not None:
            # Resize the input image to match the grid resolution and normalize
            img_resized = resize(img, (nx, ny), mode='reflect', anti_aliasing=True)
            refl_map = np.clip(img_resized, 0, 1).astype(np.float32)

    return refl_map


def _rasterize_single_shape(
        params: Dict,
        x_grid: np.ndarray,
        y_grid: np.ndarray
) -> np.ndarray:
    """
    Helper function to rasterize a single geometric shape onto a boolean mask.

    Supports circles, ellipses, and various polygons.

    Args:
        params: The parameter dictionary for a single shape.
        x_grid: A 1D numpy array of the grid's x-coordinates.
        y_grid: A 1D numpy array of the grid's y-coordinates.

    Returns:
        A boolean numpy array of shape (len(x_grid), len(y_grid)) where True
        indicates the pixel is inside the shape.
    """
    nx, ny = len(x_grid), len(y_grid)
    # Create a meshgrid for vectorized calculations
    xx, yy = np.meshgrid(x_grid, y_grid, indexing='ij')

    shape = params.get('shape', 'circle')
    cx, cy, _ = params.get('center', (0, 0, 0))

    if shape == 'circle':
        r = params.get('radius', 1.0)
        mask = (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) <= r)
        return mask

    elif shape == 'ellipse':
        maj = params.get('major_axis', 1.0) / 2.0
        mino = params.get('minor_axis', 0.5) / 2.0
        rot_rad = np.deg2rad(-params.get('rotation', 0))  # Negative to match common graphics rotation

        # Apply rotation to coordinates
        x_rot = (xx - cx) * np.cos(rot_rad) - (yy - cy) * np.sin(rot_rad)
        y_rot = (xx - cx) * np.sin(rot_rad) + (yy - cy) * np.cos(rot_rad)

        mask = ((x_rot / maj) ** 2 + (y_rot / mino) ** 2) <= 1
        return mask

    else:  # All other shapes are polygon-based
        verts = get_polygon_vertices(params)
        if verts is not None:
            return rasterize_polygon(x_grid, y_grid, verts)

    return np.zeros((nx, ny), dtype=bool)


def rasterize_polygon(
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        vertices: np.ndarray
) -> np.ndarray:
    """
    Rasterizes a polygon using point-in-polygon testing.

    Args:
        x_grid: 1D numpy array of the grid's x-coordinates.
        y_grid: 1D numpy array of the grid's y-coordinates.
        vertices: A numpy array of shape (N, 2) defining the polygon's vertices.

    Returns:
        A boolean mask array of shape (nx, ny) indicating which grid points
        are inside the polygon.
    """
    nx, ny = len(x_grid), len(y_grid)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing='ij')
    # Flatten grid points for efficient processing
    points = np.vstack((xx.ravel(), yy.ravel())).T

    path = Path(vertices)
    mask = path.contains_points(points)

    # Reshape the 1D mask back to the 2D grid shape
    return mask.reshape((nx, ny))


def get_polygon_vertices(params: Dict) -> Optional[np.ndarray]:
    """
    Generates the 2D vertex coordinates for various standard polygonal shapes.

    Supported shapes include 'rectangle', 'regular_polygon', 'star', and 'cross'.
    The vertices are generated around a local origin (0,0) and then rotated
    and translated to their final position.

    Args:
        params: The parameter dictionary for the shape.

    Returns:
        A numpy array of shape (N, 2) containing the vertex coordinates, or
        None if the shape is not recognized.
    """
    shape = params.get('shape', 'unsupported')
    center = params.get('center', (0, 0, 0))
    cx, cy = center[:2]
    rotation_rad = np.deg2rad(params.get('rotation', 0))

    # Rotation matrix
    R = np.array([[np.cos(rotation_rad), -np.sin(rotation_rad)],
                  [np.sin(rotation_rad), np.cos(rotation_rad)]])

    verts = None
    if shape == 'rectangle':
        w, h = params.get('width', 1.0), params.get('height', 1.0)
        w2, h2 = w / 2, h / 2
        verts = np.array([[-w2, -h2], [w2, -h2], [w2, h2], [-w2, h2]])

    elif shape == 'regular_polygon':
        n = params.get('n_sides', 5)
        side_length = params.get('side_length', 1.0)
        # Calculate circumradius from side length
        r = side_length / (2 * np.sin(np.pi / n)) if n > 2 else side_length / 2
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        verts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])

    elif shape == 'star':
        p = params.get('points', 5)
        r_out = params.get('outer_radius', 1.0)
        r_in = params.get('inner_radius', 0.5)
        angles = np.linspace(0, 2 * np.pi, 2 * p, endpoint=False)
        verts = np.empty((2 * p, 2))
        verts[0::2, 0] = r_out * np.cos(angles[0::2])  # Outer points
        verts[0::2, 1] = r_out * np.sin(angles[0::2])
        verts[1::2, 0] = r_in * np.cos(angles[1::2])  # Inner points
        verts[1::2, 1] = r_in * np.sin(angles[1::2])

    elif shape == 'cross':
        l = params.get('arm_length', 1.0)
        w = params.get('arm_width', 0.2)
        l2, w2 = l / 2, w / 2
        verts = np.array([
            [-w2, -l2], [w2, -l2], [w2, -w2], [l2, -w2], [l2, w2],
            [w2, w2], [w2, l2], [-w2, l2], [-w2, w2], [-l2, w2], [-l2, -w2]
        ])

    if verts is None:
        return None

    # Apply rotation and translation to the locally-defined vertices
    return (R @ verts.T).T + np.array([cx, cy])


def generate_random_shapes(
        n_shapes: int,
        base_center: Tuple[float, float, float],
        area_size: float
) -> List[Dict]:
    """
    Generates a list of random shape parameter dictionaries.

    Useful for creating complex, randomized scenes for testing reconstruction algorithms.

    Args:
        n_shapes: The number of random shapes to generate.
        base_center: The (x, y, z) coordinate around which shapes are placed.
        area_size: The size of the square area for random placement.

    Returns:
        A list of shape parameter dictionaries, compatible with other functions
        in this module.
    """
    shapes = []
    shape_types = ['circle', 'ellipse', 'rectangle', 'regular_polygon', 'star', 'cross']

    for _ in range(n_shapes):
        shape_type = np.random.choice(shape_types)
        # Randomly offset the center from the base center
        offset = (np.random.rand(2) - 0.5) * area_size
        center = (base_center[0] + offset[0], base_center[1] + offset[1], base_center[2])
        params = {'shape': shape_type, 'center': center, 'rotation': np.random.uniform(0, 360)}

        if shape_type == 'circle':
            params['radius'] = np.random.uniform(0.2, 0.5)
        elif shape_type == 'ellipse':
            params['major_axis'] = np.random.uniform(0.4, 1.2)
            params['minor_axis'] = np.random.uniform(0.2, params['major_axis'] * 0.8)
        elif shape_type == 'rectangle':
            params['width'] = np.random.uniform(0.3, 1.0)
            params['height'] = np.random.uniform(0.3, 1.0)
        elif shape_type == 'regular_polygon':
            params['n_sides'] = np.random.randint(3, 8)
            params['side_length'] = np.random.uniform(0.2, 0.6)
        elif shape_type == 'star':
            params['points'] = np.random.randint(4, 8)
            params['outer_radius'] = np.random.uniform(0.4, 0.8)
            params['inner_radius'] = np.random.uniform(0.1, params['outer_radius'] * 0.7)
        elif shape_type == 'cross':
            params['arm_length'] = np.random.uniform(0.5, 1.2)
            params['arm_width'] = np.random.uniform(0.1, params['arm_length'] / 3.0)

        shapes.append(params)

    return shapes




