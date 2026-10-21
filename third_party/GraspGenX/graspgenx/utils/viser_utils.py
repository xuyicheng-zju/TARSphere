# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for visualization using viser.
This module provides visualization helpers for meshes, point clouds, grasps, and coordinate frames.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import trimesh
import trimesh.transformations as tra
import viser
import viser.transforms as vtf


def is_rotation_matrix(M, tol=1e-4):
    """Check if matrix M is a valid rotation matrix."""
    tag = False
    I = np.identity(M.shape[0])

    if (np.linalg.norm((np.matmul(M, M.T) - I)) < tol) and (
        np.abs(np.linalg.det(M) - 1) < tol
    ):
        tag = True

    if tag is False:
        print("M @ M.T:\n", np.matmul(M, M.T))
        print("det:", np.linalg.det(M))

    return tag


def get_color_from_score(labels, use_255_scale=False):
    """Convert score labels to RGB colors (red=low, green=high)."""
    scale = 255.0 if use_255_scale else 1.0
    if type(labels) in [np.float32, float]:
        return scale * np.array([1 - labels, labels, 0])
    else:
        score = scale * np.stack(
            [np.ones(labels.shape[0]) - labels, labels, np.zeros(labels.shape[0])],
            axis=1,
        )
        return score.astype(np.int32)


def rgb2hex(rgb: Tuple[int, int, int]) -> str:
    """
    Converts rgb color to hex.

    Args:
        rgb: color in rgb, e.g. (255,0,0)
    """
    return "0x%02x%02x%02x" % (rgb)


def matrix_to_wxyz_position(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 4x4 homogeneous transformation matrix to wxyz quaternion and position.

    Args:
        T: 4x4 homogeneous transformation matrix

    Returns:
        Tuple of (wxyz quaternion, position)
    """
    rotation_matrix = T[:3, :3]
    so3 = vtf.SO3.from_matrix(rotation_matrix)
    wxyz = so3.wxyz
    position = T[:3, 3]
    return wxyz, position


def create_visualizer(clear=True, port: int = 8080) -> viser.ViserServer:
    """
    Create a viser server for visualization.

    Args:
        clear: If True, clear existing scene content
        port: Port number for the viser server (default: 8080)

    Returns:
        viser.ViserServer instance
    """
    print(f"Starting viser server on http://localhost:{port}")
    server = viser.ViserServer(port=port)
    if clear:
        server.scene.reset()
    print(f"Viser server running at http://localhost:{port}")
    return server


def make_frame(
    vis: viser.ViserServer,
    name: str,
    h: float = 0.15,
    radius: float = 0.01,
    o: float = 1.0,
    T: Optional[np.ndarray] = None,
):
    """Add a red-green-blue triad to the Viser visualizer.

    Args:
        vis (viser.ViserServer): the visualizer
        name (string): name for this frame (should be unique)
        h (float): height of frame visualization (axes_length)
        radius (float): radius of frame visualization (axes_radius)
        o (float): opacity (not used in viser frames, kept for API compatibility)
        T (4x4 numpy.array): (optional) transform to apply to this geometry
    """
    if vis is None:
        return

    wxyz = (1.0, 0.0, 0.0, 0.0)
    position = (0.0, 0.0, 0.0)

    if T is not None:
        is_valid = is_rotation_matrix(T[:3, :3])
        if not is_valid:
            raise ValueError("viser_utils: attempted to visualize invalid transform T")
        wxyz, position = matrix_to_wxyz_position(T)

    vis.scene.add_frame(
        name,
        show_axes=True,
        axes_length=h,
        axes_radius=radius,
        wxyz=wxyz,
        position=position,
    )


def visualize_mesh(
    vis: viser.ViserServer,
    name: str,
    mesh: trimesh.Trimesh,
    color: Optional[List[int]] = None,
    transform: Optional[np.ndarray] = None,
):
    """Visualize a mesh in viser.

    Args:
        vis: viser server object
        name: unique name for the mesh
        mesh: trimesh.Trimesh object
        color: RGB color list [R, G, B] in range [0, 255]
        transform: 4x4 homogeneous transform to apply
    """
    if vis is None:
        return None

    if color is None:
        color = np.random.randint(low=0, high=256, size=3).tolist()

    if isinstance(color, np.ndarray):
        color = color.tolist()
    color_tuple = tuple(int(c) for c in color[:3])

    wxyz = (1.0, 0.0, 0.0, 0.0)
    position = (0.0, 0.0, 0.0)

    if transform is not None:
        wxyz, position = matrix_to_wxyz_position(transform)

    return vis.scene.add_mesh_simple(
        name,
        vertices=mesh.vertices.astype(np.float32),
        faces=mesh.faces.astype(np.uint32),
        color=color_tuple,
        wxyz=wxyz,
        position=position,
    )


def visualize_bbox(
    vis: viser.ViserServer,
    name: str,
    dims: np.ndarray,
    T: Optional[np.ndarray] = None,
    color: Optional[List[int]] = None,
):
    """Visualize a bounding box using a wireframe.

    Args:
        vis (viser.ViserServer): the visualizer
        name (string): name for this frame (should be unique)
        dims (array-like): shape (3,), dimensions of the bounding box
        T (4x4 numpy.array): (optional) transform to apply to this geometry
        color: RGB color tuple [R, G, B] in range [0, 255]
    """
    if vis is None:
        return

    if color is None:
        color = [255, 0, 0]

    if isinstance(color, np.ndarray):
        color = color.tolist()
    color_tuple = tuple(int(c) for c in color[:3])

    wxyz = (1.0, 0.0, 0.0, 0.0)
    position = (0.0, 0.0, 0.0)

    if T is not None:
        wxyz, position = matrix_to_wxyz_position(T)

    if isinstance(dims, np.ndarray):
        dims = tuple(float(d) for d in dims)

    vis.scene.add_box(
        name,
        color=color_tuple,
        dimensions=dims,
        wireframe=True,
        wxyz=wxyz,
        position=position,
    )


def visualize_pointcloud(
    vis: viser.ViserServer,
    name: str,
    pc: np.ndarray,
    color: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    size: float = 0.01,
    **kwargs,
):
    """Visualize a point cloud in viser.

    Args:
        vis: viser server object
        name: str
        pc: Nx3 or HxWx3
        color: (optional) same shape as pc[0 - 255] scale or just rgb tuple
        transform: (optional) 4x4 homogeneous transform
        size: point size (default 0.01)
    """
    if vis is None:
        return
    if pc.ndim == 3:
        pc = pc.reshape(-1, pc.shape[-1])

    if pc.shape[-1] != 3:
        pc = pc[:, :3]

    num_points = pc.shape[0]

    if color is not None:
        if isinstance(color, list):
            color = np.array(color)
        color = np.array(color)

        if color.ndim == 3:
            color = color.reshape(-1, color.shape[-1])

        if color.ndim == 1:
            single_color = np.array(color).flatten()[:3]
            color = np.tile(single_color, (num_points, 1))
        elif color.ndim == 2:
            if color.shape[-1] > 3:
                color = color[:, :3]
            if color.shape[0] != num_points:
                if color.shape[0] > num_points:
                    color = color[:num_points]
                else:
                    padding = np.tile(color[-1:], (num_points - color.shape[0], 1))
                    color = np.vstack([color, padding])

        if np.issubdtype(color.dtype, np.floating):
            color = np.clip(color * 255.0, 0, 255).astype(np.uint8)
        else:
            color = np.clip(color, 0, 255).astype(np.uint8)
    else:
        color = np.full((num_points, 3), 255, dtype=np.uint8)

    assert color.shape == (
        num_points,
        3,
    ), f"Color shape {color.shape} doesn't match expected ({num_points}, 3)"

    wxyz = (1.0, 0.0, 0.0, 0.0)
    position = (0.0, 0.0, 0.0)

    if transform is not None:
        wxyz, position = matrix_to_wxyz_position(transform)

    vis.scene.add_point_cloud(
        name,
        points=pc.astype(np.float32),
        colors=color,
        point_size=size,
        wxyz=wxyz,
        position=position,
    )


def visualize_grasp(
    vis: viser.ViserServer,
    name: str,
    transform: np.ndarray,
    color: List[int] = [255, 0, 0],
    gripper_points: Optional[List[np.ndarray]] = None,
    linewidth: float = 1.0,
    **kwargs,
):
    """
    Visualize a grasp using line segments in viser.

    Args:
        vis: viser server object
        name: str, name for this grasp visualization
        transform: 4x4 homogeneous transform for the grasp pose
        color: RGB color list
        gripper_points: List of control point arrays for the gripper visualization
        linewidth: width of the line segments
    """
    if vis is None:
        return

    if gripper_points is None:
        # Default simple gripper visualization (basic parallel jaw)
        gripper_points = [
            np.array(
                [
                    [0, 0, 0, 1],
                    [0, 0, 0.05, 1],
                    [0, 0.04, 0.05, 1],
                ]
            ).T,
            np.array(
                [
                    [0, 0, 0.05, 1],
                    [0, -0.04, 0.05, 1],
                ]
            ).T,
        ]

    if isinstance(color, np.ndarray):
        color = color.tolist()
    color_tuple = tuple(int(c) for c in color[:3])

    wxyz, position = matrix_to_wxyz_position(transform.astype(float))

    for i, grasp_vertex in enumerate(gripper_points):
        points_3d = grasp_vertex[:3, :].T  # Shape: [N, 3]

        num_points = points_3d.shape[0]
        if num_points < 2:
            continue

        segments = np.zeros((num_points - 1, 2, 3), dtype=np.float32)
        for j in range(num_points - 1):
            segments[j, 0, :] = points_3d[j]
            segments[j, 1, :] = points_3d[j + 1]

        vis.scene.add_line_segments(
            f"{name}/{i}",
            points=segments,
            colors=color_tuple,
            line_width=linewidth,
            wxyz=wxyz,
            position=position,
        )


def get_normals_from_mesh(
    mesh: trimesh.Trimesh, contact_pts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get surface normals at contact points on a mesh."""
    from sklearn.neighbors import KDTree

    points_codebook, index = mesh.sample(16000, return_index=True)
    normals_codebook = mesh.face_normals[index]

    contact_radius = 0.005

    tree = KDTree(points_codebook)
    dist, idx = tree.query(contact_pts)
    matched = dist < contact_radius
    idx2 = idx[matched]
    normals = normals_codebook[idx2]
    mask = matched[:, 0]
    return normals, contact_pts[mask], mask


def clear_visualization(vis: viser.ViserServer, name: str = None):
    """Clear visualization objects from scene.

    Args:
        vis: viser server object
        name: if provided, remove only this object; otherwise reset entire scene
    """
    if vis is None:
        return

    if name is not None:
        vis.scene.remove_by_name(name)
    else:
        vis.scene.reset()


def create_gripper_control_points_for_viz(
    width: float, depth: float, height: float = 0.0
) -> List[np.ndarray]:
    """
    Create control points for gripper visualization based on width, depth, and height.
    This follows the x-grasp convention for parallel jaw grippers.

    Args:
        width: gripper opening width (X extent)
        depth: gripper depth/fingertip z position (Z extent)
        height: gripper height (Y extent) - used for mid_point positioning

    Returns:
        List of control point arrays for visualization (each array is shape [N, 4])
    """
    hw = width / 2  # half width
    hh = height / 2  # half height

    # Create canonical control points
    # Format: left fingertip, right fingertip, mid-finger points, base
    left_tip = np.array([-hw, 0, depth, 1])
    right_tip = np.array([hw, 0, depth, 1])
    left_mid = np.array([-hw, 0, depth * 0.6, 1])
    right_mid = np.array([hw, 0, depth * 0.6, 1])
    base = np.array([0, 0, 0, 1])

    # Mid point at fingertip depth, offset by half height in Y
    mid_point = np.array([0, -hh, depth, 1])

    # Create visualization line path:
    # left_mid -> left_tip -> mid_point -> base -> mid_point -> right_tip -> right_mid
    vis_points = np.array(
        [left_mid, left_tip, mid_point, base, mid_point, right_tip, right_mid]
    )

    return [vis_points]


def generate_control_points_from_sweep_volume(sweep_volume: Dict) -> List[np.ndarray]:
    """Generate control points from sweep volume."""
    sv_extents = sweep_volume["extents"]
    f = sweep_volume["offset"][2]

    w = sv_extents[0]
    d = sv_extents[1]
    h = sv_extents[2]

    control_points = np.array(
        [
            [w / 2, 0, h / 2 + f, 1],
            [w / 2, 0, -h / 2 + f, 1],
            [0, 0, -h / 2 + f, 1],
            [0, 0, 0, 1],
            [0, 0, -h / 2 + f, 1],
            [-w / 2, 0, -h / 2 + f, 1],
            [-w / 2, 0, h / 2 + f, 1],
        ]
    )
    control_points = [
        control_points,
    ]  # Not sure why this is needed, but it is.
    return control_points


def visualize_x_grasp(
    vis: viser.ViserServer,
    name: str,
    transform: np.ndarray,
    color: List[int] = [255, 0, 0],
    gripper_info=None,
    width: float = None,
    depth: float = None,
    height: float = 0.0,
    sweep_volume: Dict = None,
    linewidth: float = 2.0,
    **kwargs,
):
    """
    Visualize a grasp using XGripperInfo control points or width/depth/height parameters.

    Args:
        vis: viser server object
        name: str, name for this grasp visualization
        transform: 4x4 homogeneous transform for the grasp pose
        color: RGB color list
        gripper_info: XGripperInfo object with control_points_visualization (optional)
        width: gripper width (used if gripper_info not provided)
        depth: gripper depth (used if gripper_info not provided)
        height: gripper height (used if gripper_info not provided)
        linewidth: width of the line segments
    """
    if vis is None:
        return []

    # Get control points — prefer sweep_volume for accurate extremities
    if (
        gripper_info is not None
        and hasattr(gripper_info, "sweep_volume")
        and gripper_info.sweep_volume is not None
    ):
        # gripper_info.sweep_volume is [extents(3), offset(3)]
        sv = gripper_info.sweep_volume
        sv_dict = {"extents": sv[:3].tolist(), "offset": sv[3:].tolist()}
        grasp_vertices = generate_control_points_from_sweep_volume(sv_dict)
    elif sweep_volume is not None:
        grasp_vertices = generate_control_points_from_sweep_volume(sweep_volume)
    elif width is not None and depth is not None:
        grasp_vertices = create_gripper_control_points_for_viz(width, depth, height)
    else:
        # Default fallback
        grasp_vertices = create_gripper_control_points_for_viz(0.08, 0.05, 0.0)

    if isinstance(color, np.ndarray):
        color = color.tolist()
    color_tuple = tuple(int(c) for c in color[:3])

    wxyz, position = matrix_to_wxyz_position(transform.astype(float))

    handles = []
    for i, ctrl_pts in enumerate(grasp_vertices):
        # Convert to numpy array
        ctrl_pts = np.array(ctrl_pts, dtype=np.float32)

        # Handle different input shapes
        if ctrl_pts.ndim == 1:
            # Single point, skip
            continue

        # Ensure shape is [N, 4] (N points with homogeneous coords)
        if ctrl_pts.shape[0] == 4 and ctrl_pts.shape[1] != 4:
            # Shape is [4, N], transpose it
            ctrl_pts = ctrl_pts.T

        points_3d = ctrl_pts[:, :3]  # Shape: [N, 3]

        num_points = points_3d.shape[0]
        if num_points < 2:
            continue

        # Create line segments
        segments = np.zeros((num_points - 1, 2, 3), dtype=np.float32)
        for j in range(num_points - 1):
            segments[j, 0, :] = points_3d[j]
            segments[j, 1, :] = points_3d[j + 1]

        handle = vis.scene.add_line_segments(
            f"{name}/{i}",
            points=segments,
            colors=color_tuple,
            line_width=linewidth,
            wxyz=wxyz,
            position=position,
        )
        handles.append(handle)

    return handles
