# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pyrender
import torch
import trimesh
import trimesh.transformations as tra
from scipy.ndimage import convolve

from graspgenx.dataset.dataset_utils import ObjectGraspDataset
from graspgenx.dataset.exceptions import DataLoaderError
from graspgenx.utils.logging_config import get_logger

logger = get_logger(__name__)

NOISE_PARAMS = {
    "kin_noi": {"std_range": [0.25, 0.75], "thresh_range": [100, 1000]},
    "gau_noi": {"gaussian_std_range": [0.001, 0.002]},
    "dp_ell": {
        "ellipse_dropout_mean": 10,
        "ellipse_gamma_shape": 5.0,
        "ellipse_gamma_scale": 1.0,
    },
    "noi_xyz": {
        "gaussian_scale_range": [0.0, 0.005],
        "gp_rescale_factor_range": [6, 20],
    },
    "wrist_cali_err": {"low": -0.005, "high": 0.005},
    "rgb_color_jitter": {
        "brightness": 0.1,
        "contrast": 0.1,
        "saturation": 0.1,
        "hue": 0.1,
    },
}


def fov_and_size_to_intrinsics(fov, img_size, device="cpu"):
    img_h, img_w = img_size
    fx = img_w / (2 * math.tan(math.radians(fov) / 2))
    fy = img_h / (2 * math.tan(math.radians(fov) / 2))

    # intrinsics = torch.tensor(
    #     [[fx, 0, img_h / 2], [0, fy, img_w / 2], [0, 0, 1]],
    #     dtype=torch.float,
    #     device=device,
    # )
    intrinsics = np.array([[fx, 0, img_h / 2], [0, fy, img_w / 2], [0, 0, 1]])
    return intrinsics


def depth2points(
    depth: np.array,
    fx: int,
    fy: int,
    cx: int,
    cy: int,
    xmap: np.array = None,
    ymap: np.array = None,
    rgb: np.array = None,
    seg: np.array = None,
    mask: np.arange = None,
) -> Dict:
    """Compute point cloud from a depth image.

    Args:
        depth (np.array): 2D depth image.
        fx (int): Focal length x.
        fy (int): Focal length y.
        cx (int): Camera principal point x.
        cy (int): Camera principal point y.
        xmap (np.array, optional): precomputed meshgrid map X. Defaults to None.
        ymap (np.array, optional): precomputed meshgrid map Y. Defaults to None.
        rgb (np.array, optional): RGB imgae for colored point cloud. Defaults to None.
        seg (np.array, optional): Segmentation image corresponding to the rgb image Defaults to None.
        mask (np.arange, optional): Binary mask. 1 to keep. Defaults to None.

    Returns:
        Dict: Output is a dictionary with fields `xyz`, `rgb`, and `index`,
        where `xyz` and `rgb` are Nx3 numpy arrays and `index` is a Nx1 array
        to indicate which point is valid.
    """
    """convert depth image to point cloud"""
    if rgb is not None:
        assert rgb.shape[0] == depth.shape[0] and rgb.shape[1] == depth.shape[1]
    if xmap is not None:
        assert xmap.shape[0] == depth.shape[0] and xmap.shape[1] == depth.shape[1]
    if ymap is not None:
        assert ymap.shape[0] == depth.shape[0] and ymap.shape[1] == depth.shape[1]

    im_height, im_width = depth.shape[0], depth.shape[1]

    if xmap is None or ymap is None:
        ww = np.linspace(0, im_width - 1, im_width)
        hh = np.linspace(0, im_height - 1, im_height)
        xmap, ymap = np.meshgrid(ww, hh)

    pt2 = depth
    pt0 = (xmap - cx) * pt2 / fx
    pt1 = (ymap - cy) * pt2 / fy

    mask_depth = np.ma.getmaskarray(np.ma.masked_greater(pt2, 0))
    if mask is None:
        mask = mask_depth
    else:
        mask_semantic = np.ma.getmaskarray(np.ma.masked_equal(mask, 1))
        mask = mask_depth * mask_semantic

    index = mask.flatten().nonzero()[0]

    pt2_valid = pt2.flatten()[:, np.newaxis].astype(np.float32)
    pt0_valid = pt0.flatten()[:, np.newaxis].astype(np.float32)
    pt1_valid = pt1.flatten()[:, np.newaxis].astype(np.float32)
    pc_xyz = np.concatenate((pt0_valid, pt1_valid, pt2_valid), axis=1)
    if rgb is not None:
        r = rgb[:, :, 0].flatten()[:, np.newaxis]
        g = rgb[:, :, 1].flatten()[:, np.newaxis]
        b = rgb[:, :, 2].flatten()[:, np.newaxis]
        pc_rgb = np.concatenate((r, g, b), axis=1)
    else:
        pc_rgb = None

    if seg is not None:
        pc_seg = seg.flatten()[:, np.newaxis]
    else:
        pc_seg = None

    return {"xyz": pc_xyz, "rgb": pc_rgb, "seg": pc_seg, "index": index}


def compute_camera_pose(center, distance, azimuth, elevation):
    cam_tf = tra.euler_matrix(np.pi / 2, 0, 0).dot(tra.euler_matrix(0, np.pi / 2, 0))
    pose = tra.euler_matrix(0, -elevation, azimuth) @ tra.translation_matrix(
        [distance, 0, 0]
    )
    pose = tra.translation_matrix(center) @ pose
    pose_opengl = pose @ cam_tf
    pose_opencv = pose_opengl.copy()
    pose_opencv[:, 1:3] *= -1.0
    # pose_opengl = pose_opencv @ T_opencv_to_opengl_format
    return pose_opencv, pose_opengl


def sample_camera_pose(
    lookat_point=[0, 0, 0],
    azimuth_lims=[-0.2, 0.2],
    elevation_lims=[0.6, 1.0],
    radius_lims=[1.5, 2.0],
    num_cameras=1,
):
    camera_poses = []

    assert len(lookat_point) == 3
    assert len(azimuth_lims) == 2
    assert len(elevation_lims) == 2
    assert len(radius_lims) == 2
    assert num_cameras > 0

    camera_poses_opengl = []
    camera_poses_opencv = []

    for _ in range(num_cameras):
        target = lookat_point
        radius = np.random.uniform(*radius_lims)
        elevation = np.random.uniform(*elevation_lims)
        azimuth = np.random.uniform(*azimuth_lims)
        pose_opencv, pose_opengl = compute_camera_pose(
            target, radius, azimuth, elevation
        )
        camera_poses_opengl.append(pose_opengl)
        camera_poses_opencv.append(pose_opencv)

    return camera_poses_opengl, camera_poses_opencv


class Renderer:
    def __init__(self):
        self._scene = pyrender.Scene()
        self._node_dict = {}
        self._camera_intr = None
        self._camera_node = None
        self._light_node = None
        self._renderer = None

    def create_camera(self, intrinsics, image_size, znear=0.04, zfar=100.0):
        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        cx = intrinsics["cx"]
        cy = intrinsics["cy"]
        cam = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear, zfar)
        self._camera_intr = intrinsics
        self._camera_node = pyrender.Node(camera=cam, matrix=np.eye(4))
        self._scene.add_node(self._camera_node)
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=4.0)
        self._light_node = pyrender.Node(light=light, matrix=np.eye(4))
        self._scene.add_node(self._light_node)
        self._renderer = pyrender.OffscreenRenderer(
            viewport_width=image_size[0],
            viewport_height=image_size[1],
            # point_size=5.0,
        )

    def set_camera_pose(self, cam_pose):
        if self._camera_node is None:
            raise ValueError("call create camera first!")
        self._scene.set_pose(self._camera_node, cam_pose)
        self._scene.set_pose(self._light_node, cam_pose)

    def get_camera_pose(self):
        if self._camera_node is None:
            raise ValueError("call create camera first!")
        return self._camera_node.matrix

    def render_rgbd(self, depth_only=True):
        if depth_only:
            depth = self._renderer.render(self._scene, pyrender.RenderFlags.DEPTH_ONLY)
            color = None
        else:
            color, depth = self._renderer.render(self._scene)
        return color, depth

    def render_segmentation(self, full_depth=None):
        if full_depth is None:
            _, full_depth = self.render_rgbd(depth_only=True)

        self.hide_objects()
        output = np.zeros(full_depth.shape, dtype=np.uint8)
        for i, obj_name in enumerate(self._node_dict):
            self._node_dict[obj_name].mesh.is_visible = True
            _, depth = self.render_rgbd(depth_only=True)
            mask = np.logical_and(
                (np.abs(depth.data - full_depth) < 1e-6),
                np.abs(full_depth) > 0,
            )
            if np.any(output[mask] != 0):
                raise ValueError("wrong label")
            output[mask] = i + 1
            self._node_dict[obj_name].mesh.is_visible = False
        self.show_objects()

        return output, ["BACKGROUND"] + list(self._node_dict.keys())

    def render_all(self):
        rgb, depth = self.render_rgbd()
        seg, labels = self.render_segmentation(depth)

        return rgb, depth, seg, labels

    def add_object(self, name, mesh, pose=None):
        if pose is None:
            pose = np.eye(4, dtype=np.float32)

        node = pyrender.Node(
            name=name,
            mesh=pyrender.Mesh.from_trimesh(mesh, smooth=False),
            matrix=pose,
        )
        self._node_dict[name] = node
        self._scene.add_node(node)

    def set_object_pose(self, name, pose):
        self._scene.set_pose(self._node_dict[name], pose)

    def has_object(self, name):
        return name in self._node_dict

    def remove_object(self, name):
        self._scene.remove_node(self._node_dict[name])
        del self._node_dict[name]

    def show_objects(self, names=None):
        for name, node in self._node_dict.items():
            if names is None or name in names:
                node.mesh.is_visible = True

    def toggle_wireframe(self, names=None):
        for name, node in self._node_dict.items():
            if names is None or name in names:
                node.mesh.primitives[0].material.wireframe ^= True

    def hide_objects(self, names=None):
        for name, node in self._node_dict.items():
            if names is None or name in names:
                node.mesh.is_visible = False

    def reset(self):
        for name in self._node_dict:
            self._scene.remove_node(self._node_dict[name])
        self._node_dict = {}


def add_gaussian_noise_to_depth(
    depth_img: np.ndarray, noise_params: Dict
) -> np.ndarray:
    """Add Gaussian noise to depth image.

    Applies zero-mean Gaussian noise with random standard deviation.
    Used for simulating depth sensor noise.

    Args:
        depth_img: HxW depth image
        noise_params: Dictionary with 'gaussian_std_range' parameter

    Returns:
        np.ndarray: Noisy depth image
    """
    assert isinstance(depth_img, np.ndarray)
    std = np.random.uniform(
        noise_params["gaussian_std_range"][0], noise_params["gaussian_std_range"][1]
    )
    rng = np.random.default_rng()
    noise = rng.standard_normal(size=depth_img.shape, dtype=np.float32) * std
    depth_img += noise
    return depth_img


def add_depth_noise(depth: np.array) -> np.array:

    depth = add_gaussian_noise_to_depth(depth, NOISE_PARAMS["gau_noi"])

    depth_mask_after_noise = np.isinf(depth)
    depth[depth_mask_after_noise] = 0
    return depth


def add_noise_to_random_object_region(
    segmentation_image, noise_level=0.05, region_ratio=0.5
):
    """
    Adds noise near the edges of the object (class 1) in a randomly chosen quadrant.

    Parameters:
    - segmentation_image (numpy array, shape (H, W)): The input segmentation image.
    - noise_level (float): Fraction of object edge pixels (class 1) to perturb.
    - region_ratio (float): Fraction of the object bounding box height/width for selecting the region.

    Returns:
    - noisy_image (numpy array, shape (H, W)): The noisy segmentation image.
    """
    import random

    noisy_image = segmentation_image.copy()
    H, W = segmentation_image.shape

    # Define edge detection kernel
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

    # Detect edges (but keep only class 1 edges)
    edge_map = np.abs(
        convolve((segmentation_image == 1).astype(np.float32), kernel, mode="constant")
    )
    object_edge_pixels = np.argwhere(edge_map > 0)  # Edge pixels of class 1

    if len(object_edge_pixels) == 0:
        return noisy_image  # No object edges found, return original image

    # Find bounding box of the object (class 1)
    obj_pixels = np.argwhere(segmentation_image == 1)
    min_y, min_x = obj_pixels.min(axis=0)
    max_y, max_x = obj_pixels.max(axis=0)

    # Define the region sizes based on the object bounding box
    region_H = int((max_y - min_y) * region_ratio)
    region_W = int((max_x - min_x) * region_ratio)

    # Randomly select a region to apply noise
    selected_region = random.choice(
        ["top-right", "top-left", "bottom-right", "bottom-left"]
    )

    if selected_region == "top-right":
        selected_edges = np.array(
            [
                p
                for p in object_edge_pixels
                if p[0] < min_y + region_H and p[1] > max_x - region_W
            ]
        )
    elif selected_region == "top-left":
        selected_edges = np.array(
            [
                p
                for p in object_edge_pixels
                if p[0] < min_y + region_H and p[1] < min_x + region_W
            ]
        )
    elif selected_region == "bottom-right":
        selected_edges = np.array(
            [
                p
                for p in object_edge_pixels
                if p[0] > max_y - region_H and p[1] > max_x - region_W
            ]
        )
    elif selected_region == "bottom-left":
        selected_edges = np.array(
            [
                p
                for p in object_edge_pixels
                if p[0] > max_y - region_H and p[1] < min_x + region_W
            ]
        )

    if len(selected_edges) == 0:
        return noisy_image  # No edge pixels in the chosen region

    # Determine number of edge pixels to modify
    num_noisy_pixels = int(noise_level * len(selected_edges))
    if num_noisy_pixels == 0:
        return noisy_image  # Avoid modifying too few pixels

    # Select a random subset of edge pixels in the chosen region
    noisy_indices = np.random.choice(
        len(selected_edges), num_noisy_pixels, replace=False
    )

    # Apply noise to selected edge pixels
    for idx in noisy_indices:
        y, x = selected_edges[idx]
        noisy_image[y, x] = np.random.choice(
            [0, 2]
        )  # Object (1) flips to background (0) or support surface (2)

    return noisy_image


def render_images_given_scene(
    scene_info: Dict, camera_intrinsics: np.array, camera_poses: List, image_size: int
) -> Tuple[np.array, np.array, np.array]:
    # TODO: Make rendering step more efficient
    r = Renderer()
    r.create_camera(camera_intrinsics, image_size, znear=0.04, zfar=100)

    for s in scene_info:
        r.add_object(s["name"], s["mesh"], s["pose"])

    colors = []
    depths = []
    segs = []
    labels = []
    for cam_pose in camera_poses:
        r.set_camera_pose(cam_pose)
        color, depth, seg, labels = r.render_all()
        depth = add_depth_noise(depth)
        colors.append(color)
        depths.append(depth)
        segs.append(seg)
        labels.append(labels)

    colors, depths, segs = np.array(colors), np.array(depths), np.array(segs)
    return colors, depths, segs, labels


def add_edge_noise(segmentation_image, noise_level=0.40):
    """
    Adds noise near the internal edges of objects in a segmentation image while avoiding image borders.

    Parameters:
    - segmentation_image (numpy array, shape (H, W)): The input segmentation image.
    - noise_level (float): Fraction of internal edge pixels to perturb.

    Returns:
    - noisy_image (numpy array, shape (H, W)): The noisy segmentation image.
    """
    noisy_image = segmentation_image.copy()
    H, W = segmentation_image.shape

    # Define a simple 3x3 kernel to detect edges
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

    # Apply convolution to detect edges (nonzero values indicate edges)
    edge_map = np.abs(
        convolve(segmentation_image.astype(np.float32), kernel, mode="constant")
    )

    # Get edge pixel coordinates, but exclude image borders
    edge_pixels = np.argwhere(
        (edge_map > 0)
        & (np.arange(H)[:, None] > 0)
        & (np.arange(H)[:, None] < H - 1)  # Avoid first & last row
        & (np.arange(W) > 0)
        & (np.arange(W) < W - 1)
    )  # Avoid first & last column

    num_edge_pixels = len(edge_pixels)
    if num_edge_pixels == 0:
        return noisy_image  # No change if no internal edges detected

    # Determine number of edge pixels to modify
    num_noisy_pixels = int(noise_level * num_edge_pixels)
    if num_noisy_pixels == 0:
        return noisy_image  # Avoid modifying too few pixels

    # Select a random subset of internal edge pixels
    noisy_indices = np.random.choice(num_edge_pixels, num_noisy_pixels, replace=False)

    # Apply noise to selected internal edge pixels
    for idx in noisy_indices:
        y, x = edge_pixels[idx]
        current_label = segmentation_image[y, x]

        if current_label == 1:
            # Object (1) can flip to background (0) or support surface (2)
            new_label = np.random.choice([0, 2])
        elif current_label == 2:
            # Support surface (2) can flip to object (1)
            new_label = 1
        else:
            continue  # Background (0) remains unchanged

        noisy_image[y, x] = new_label

    return noisy_image


def render_point_cloud_from_object(
    object_grasp_data: ObjectGraspDataset,
    num_points: int,
    prob_object_only: float = 0.75,
) -> Tuple[Union[np.array, None], DataLoaderError]:

    object_only = np.random.random() < prob_object_only
    scene_info = []
    T_move_to_obj_frame = tra.translation_matrix(
        -object_grasp_data.object_mesh.centroid
    )
    if object_only:
        camera_pose_randomization_params = {
            "azimuth": [np.radians(-180), np.radians(180)],
            "elevation": [np.radians(-180), np.radians(180)],
            "radius": [0.30, 0.60],
        }
        camera_look_at_point = np.array([0.0, 0.0, 0.0])
        scene_info.append(
            {
                "name": "obj",
                "mesh": object_grasp_data.object_mesh,
                "pose": T_move_to_obj_frame,
            }
        )
    else:
        # Places object on a table, adds noise to the instance segmentation

        camera_pose_randomization_params = {
            "azimuth": [np.radians(-180), np.radians(180)],
            "elevation": [np.radians(70), np.radians(110)],
            "radius": [0.30, 0.60],
        }

        import scene_synthesizer
        from scene_synthesizer import procedural_assets as pa

        obj_id = "obj"

        scene = scene_synthesizer.Scene()
        table_asset = pa.TableAsset(10.0, 10.4, 0.1)
        scene.add_object(
            asset=table_asset,
            obj_id="table",
            transform=np.eye(4),
        )
        scene.label_support(label="support", erosion_distance=4.0)

        obj_asset = scene_synthesizer.assets.MeshAsset(
            fname=object_grasp_data.object_asset_path,
            scale=object_grasp_data.object_scale,
            origin=("center", "center", "bottom"),
        )
        scene.place_object(
            obj_id=obj_id,
            obj_asset=obj_asset,
            support_id="support",
        )

        original_asset_origin = scene.graph.transforms.children[obj_id][0]
        T_original_assert_origin_to_world = scene.get_transform(original_asset_origin)

        scene_info.append(
            {"name": "obj", "mesh": object_grasp_data.object_mesh, "pose": np.eye(4)}
        )
        camera_look_at_point = np.array([0.0, 0.0, 0.0])
        scene_info.append(
            {
                "name": "support",
                "mesh": table_asset.mesh(),
                "pose": tra.inverse_matrix(T_original_assert_origin_to_world),
            }
        )

    image_size = [256, 256]
    fov = 60
    azimuth = camera_pose_randomization_params["azimuth"]
    elevation = camera_pose_randomization_params["elevation"]
    radius = camera_pose_randomization_params["radius"]
    num_cameras = 1

    camera_poses_opengl, camera_poses_opencv = sample_camera_pose(
        camera_look_at_point,
        azimuth,
        elevation,
        radius,
        num_cameras,
    )

    camera_intrinsics_raw = fov_and_size_to_intrinsics(
        fov, (image_size[0], image_size[1]), device="cuda"
    )
    camera_intrinsics = {
        "fx": int(camera_intrinsics_raw[0][0].item()),
        "fy": int(camera_intrinsics_raw[1][1].item()),
        "cx": int(camera_intrinsics_raw[0][2].item()),
        "cy": int(camera_intrinsics_raw[1][2].item()),
    }
    colors, depths, segs, labels = render_images_given_scene(
        scene_info, camera_intrinsics, camera_poses_opengl, image_size
    )

    debug = False

    if debug:
        vis = create_visualizer()
        for s in scene_info:
            visualize_mesh(
                vis,
                f'renderer/{s["name"]}/object_mesh',
                s["mesh"],
                transform=s["pose"],
                color=[169, 169, 169],
            )

    OBJ_SEG_ID = 1
    assert len(camera_poses_opencv) == 1  # This is a decent assumption for now
    for i, cam_pose in enumerate(camera_poses_opencv):

        depth = depths[i]
        pts = depth2points(
            depth=depth,
            fx=camera_intrinsics["fx"],
            fy=camera_intrinsics["fy"],
            cx=camera_intrinsics["cx"],
            cy=camera_intrinsics["cy"],
        )
        xyz = pts["xyz"]
        mask = pts["index"]
        xyz = xyz[mask]
        seg = segs[i]
        seg = add_edge_noise(seg, noise_level=np.random.uniform(low=0.2, high=0.6))
        seg_mask = seg.flatten()[:, np.newaxis][mask] == OBJ_SEG_ID
        xyz = xyz[seg_mask.reshape(-1)]
        xyz = tra.transform_points(xyz, cam_pose)
        if debug:
            make_frame(vis, f"renderer/poses/camera/{i}", T=cam_pose)
            visualize_pointcloud(
                vis, f"renderer/scene_pc/{i}", xyz, color=[10, 250, 10], size=0.003
            )

    if debug:
        input()
    init_num_pts = xyz.shape[0]
    if init_num_pts < 150:
        # Meaningless point cloud, too few pts to give any context
        return None, DataLoaderError.RENDERING_ERROR_POINT_CLOUD_VERY_FEW_POINTS
    xyz_mask = np.random.randint(0, xyz.shape[0], num_points)
    xyz = xyz[xyz_mask]  # bring back number of points to self.num_points
    if object_only:
        xyz = tra.transform_points(
            xyz, tra.inverse_matrix(T_move_to_obj_frame)
        )  # Move back to object frame
    return xyz, DataLoaderError.RENDERING_SUCCESS


def render_pc(
    object_asset_data: ObjectGraspDataset, num_points: int, mesh_mode: bool = True
) -> Tuple[Dict, DataLoaderError]:
    xyz = None
    max_partial_pc_size = 0.075
    rendering_warning = None
    rendering_detail = None

    if mesh_mode:
        # Sample points from mesh surface
        object_mesh = object_asset_data.object_mesh.copy()
        xyz, _ = trimesh.sample.sample_surface(object_mesh, num_points)
        xyz = np.array(xyz)
    else:
        # Sample partial point cloud
        try:
            xyz, error_code = render_point_cloud_from_object(
                object_asset_data, num_points
            )
        except Exception as e:
            logger.error(
                f"Encoutered exception when rendering point cloud of object: {str(e)}"
            )
            xyz = None
            return {"invalid": True}, DataLoaderError.RENDERING_ERROR

        if error_code == DataLoaderError.RENDERING_ERROR_POINT_CLOUD_VERY_FEW_POINTS:
            return {
                "invalid": True
            }, DataLoaderError.RENDERING_ERROR_POINT_CLOUD_VERY_FEW_POINTS

        bounds = xyz.max(axis=0) - xyz.min(axis=0)
        if bounds.max() < max_partial_pc_size:
            logger.warning(
                f"Partial point cloud is small (bounds [x,y,z]: {bounds}), "
                f"keeping datapoint anyway"
            )
            rendering_warning = DataLoaderError.RENDERING_ERROR_POINT_CLOUD_TOO_SMALL
            rendering_detail = {
                "bounds": bounds.tolist(),
                "max_bound": float(bounds.max()),
                "threshold": max_partial_pc_size,
                "num_points": int(xyz.shape[0]),
            }

    T_move_to_pc_mean = tra.translation_matrix(-xyz.mean(axis=0))
    xyz = tra.transform_points(xyz, T_move_to_pc_mean)
    xyz = torch.from_numpy(xyz).float()
    status = (
        rendering_warning
        if rendering_warning is not None
        else DataLoaderError.RENDERING_SUCCESS
    )
    outputs = {
        "T_move_to_pc_mean": T_move_to_pc_mean,
        "points": xyz,
        "invalid": False,
        "rendering_detail": rendering_detail,
    }
    return outputs, status
