# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from logging.handlers import QueueHandler
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import trimesh
import trimesh.transformations as tra
import yaml
from h5py._hl.group import Group
from trimesh.collision import CollisionManager


def log_worker(log_queue, log_file=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if log_file is None:
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(log_file)
    logger.addHandler(handler)
    while True:
        record = log_queue.get()
        if record is None:
            break
        logger.handle(record)


def get_logger(name, log_queue):
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    handler = QueueHandler(log_queue)
    formatter = logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    )
    handler.setFormatter(formatter)
    log.addHandler(handler)
    return log


def get_timestamp():
    import datetime

    now = datetime.datetime.now()
    year = "{:02d}".format(now.year)
    month = "{:02d}".format(now.month)
    day = "{:02d}".format(now.day)
    hour = "{:02d}".format(now.hour)
    minute = "{:02d}".format(now.minute)
    day_month_year = "{}-{}-{}-{}-{}".format(year, month, day, hour, minute)
    return day_month_year


def check_collision(scene_mesh, object_mesh, transforms):
    """Args:
    scene_mesh: trimesh.Trimesh
    object_mesh: trimesh.Trimesh
    transforms: list of 4x4 np.array
    """
    scene_manager = CollisionManager()
    scene_manager.add_object("object", scene_mesh)
    obj_manager = CollisionManager()
    obj_manager.add_object("object", object_mesh)
    collision = []
    for tr in transforms:
        obj_manager.set_transform("object", tr)
        collision.append(scene_manager.in_collision_other(obj_manager))
    return np.array(collision).astype("bool")


def write_to_h5(key, src, dst):
    if isinstance(src, dict):
        group = dst.create_group(key)
        for k, v in src.items():
            write_to_h5(k, v, group)
    else:
        dst.create_dataset(key, data=src)


def is_empty(data):
    return data == h5py.Empty("f")


def load_h5_handle_empty_case(data):
    if is_empty(data):
        return None
    else:
        return data[...]


def write_info(group: Group, info: dict):
    """Writes dictionary info to h5 file, given h5 object handle.

    Args:
        group: hdf5 group
        info: data to write into hdf5
    """
    for key, data in info.items():
        if isinstance(data, dict):
            subgroup = group.create_group(key)
            write_info(subgroup, data)
        else:

            dtype = None
            if isinstance(data, np.ndarray):
                if str(data.dtype) == "float64":
                    dtype = "float32"
                if str(data.dtype) == "int64":
                    dtype = "int16"
                if str(data.dtype) == "uint8":
                    dtype = "uint8"
            if isinstance(data, float):
                dtype = "float32"
            if isinstance(data, int):
                dtype = "uint8"
            if dtype is None:
                if data is None:
                    group.create_dataset(key, data=h5py.Empty("f"))
                else:
                    group.create_dataset(key, data=data)
            else:
                group.create_dataset(key, data=data, dtype=dtype)


def pose_as_dict(p):
    trans = tra.translation_from_matrix(p).tolist()
    rpy = tra.euler_from_matrix(p)
    return {
        "p": {"x": trans[0], "y": trans[1], "z": trans[2]},
        "r": {"roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2]},
    }


def create_scene(robot_pose, object_asset):
    return {
        "actors": [
            {
                "asset": "robot",
                "pose": pose_as_dict(robot_pose),
            },
            {
                "asset": object_asset,
                "pose": pose_as_dict(np.eye(4)),
            },
        ]
    }


def create_object_asset(file_name, scale, actor_name, asset_root, resolution):
    return {
        "actor_name": actor_name,
        "collision_group": 0,
        "collision_filter": 2,
        "asset_root": asset_root,
        "file": file_name,
        "scale": scale,
        "textured": False,
        "options": {
            "fix_base_link": False,
            "flip_visual_attachments": False,
            "thickness": 0.003,
            "armature": 0.002,
            "vhacd_enabled": True,
            "vhacd_params": {
                "resolution": resolution,
                "concavity": 0.00001,
                "plane_downsampling": 16,
                "convex_hull_downsampling": 16,
                "alpha": 0.15,
                "beta": 0.15,
                "pca": 0,
                "max_num_vertices_per_ch": 128,
                "min_volume_per_ch": 0.0001,
            },
        },
        "rigid_shape_properties": {
            "friction": 2.0,
        },
        "rigid_body_properties": {
            "mass": 0.2,
        },
    }


def create_robot_asset():
    return {
        "actor_name": "robot",
        "asset_root": "assets/",
        "file": "urdf/franka_description/robots/franka_panda_gripper_spherical_dof_acronym.urdf",
        "controllable": True,
        "collision_group": 0,
        "collision_filter": 1,
        "default_config": [0, 0, 0, 0, 0, 0, 0.04, 0.04],
        "options": {
            "fix_base_link": True,
            "flip_visual_attachments": True,
            "armature": 0.01,
        },
        "dof_properties": {
            "driveMode": [1, 1, 1, 1, 1, 1, 2, 2],
            "stiffness": [1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 0, 0],
            "damping": [50, 50, 50, 50, 50, 50, 800, 800],
            "effort": 1400.0,
        },
    }


def save_to_isaac_grasp_format(
    grasps: np.ndarray, confidences: np.ndarray, output_path: str
):
    """Saves grasps and confidences to output file, in a Isaac grasp format

    The Isaac Grasp format is detailed here:
    https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/grasp_editor.html#what-is-an-isaac-grasp-file

    grasps (np.array): Grasp poses, 4x4 homogenous matrices
    confidences (np.array): Confidences, predicted by the network, for each grasp above
    output_path (str): Path to the yaml file to save
    """
    data = {"format": "isaac_grasp", "format_version": 1.0, "grasps": {}}
    confidences = confidences.tolist()
    assert len(grasps) == len(confidences)

    for i, (g, c) in enumerate(zip(grasps, confidences)):
        xyz = g[:3, 3].tolist()
        q = tra.quaternion_from_matrix(g[:3, :3])
        qw = float(q[0])
        qxyz = q[1:].tolist()

        data["grasps"][f"grasp_{i}"] = {
            "confidence": c,
            "position": xyz,
            "orientation": {
                "w": qw,
                "xyz": qxyz,
            },
        }

    # don't write out if there are no grasps or if there is no output_path
    if len(grasps) > 0:
        if output_path is not None:
            with open(output_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    return data


def load_from_isaac_grasp_format(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads grasps and confidences from output file, which is written in the Isaac grasp format

    The Isaac Grasp format is detailed here:
    https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/grasp_editor.html#what-is-an-isaac-grasp-file

    output_path (str): Path to the yaml file

    returns:
        grasps (np.array): Grasp poses, 4x4 homogenous matrices
        confidences (np.array): Confidences, predicted by the network, for each grasp above
    """
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    grasps = data["grasps"]
    confidences = [g["confidence"] for _, g in grasps.items()]

    positions = [tra.translation_matrix(g["position"]) for _, g in grasps.items()]
    rotations = [
        tra.quaternion_matrix([g["orientation"]["w"]] + g["orientation"]["xyz"])
        for _, g in grasps.items()
    ]
    grasps = [p @ r for p, r in zip(positions, rotations)]

    assert len(grasps) == len(confidences)

    grasps = np.array(grasps)
    confidences = np.array(confidences)
    return grasps, confidences


from yourdfpy.urdf import URDF


def load_urdf_scene(urdf_path: str) -> URDF:
    """Loads urdf scene given path."""
    import yourdfpy

    scene = yourdfpy.URDF.load(
        urdf_path,
        build_scene_graph=True,
        load_meshes=True,
        build_collision_scene_graph=True,
        load_collision_meshes=True,
        force_mesh=False,
        force_collision_mesh=False,
    )
    return scene
