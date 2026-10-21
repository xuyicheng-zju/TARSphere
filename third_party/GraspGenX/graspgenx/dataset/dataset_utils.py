#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions for data preprocessing.
"""

import glob
import io
import json
import logging
import os
import time
from typing import Dict, Tuple

import h5py
import imageio
import numpy as np
import scipy
import torch
import torch.nn.functional as F
import trimesh
import trimesh.transformations as tra
from tqdm import tqdm

from graspgenx.utils.logging_config import get_logger

logger = get_logger(__name__)

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

from graspgenx.dataset.eval_utils import (
    is_empty,
    load_h5_handle_empty_case,
    write_info,
    write_to_h5,
)
from graspgenx.dataset.exceptions import DataLoaderError
from graspgenx.dataset.webdataset_utils import GraspWebDatasetReader
from graspgenx.robot import GripperInfo

try:
    import cv2
except:
    pass


class GraspJsonDatasetReader:
    """Class to efficiently read grasps data from JSON files in a regular directory structure."""

    def __init__(self, grasp_root_dir: str):
        """
        Initialize the reader with grasp root directory and load/create the UUID to JSON path mapping.

        Args:
            grasp_root_dir (str): Path to directory containing grasp JSON files
        """
        self.grasp_root_dir = grasp_root_dir
        self.map_uuid_to_path = {}

        # Check if mapping file exists
        mapping_file_path = os.path.join(grasp_root_dir, "map_uuid_to_path.json")

        if os.path.exists(mapping_file_path):
            logger.info(
                f"Loading existing UUID to JSON path mapping from {mapping_file_path}"
            )
            with open(mapping_file_path, "r") as f:
                self.map_uuid_to_path = json.load(f)
        else:
            logger.info(
                f"Creating new UUID to JSON path mapping for Grasp Dataset at {grasp_root_dir}. One time effort."
            )
            self._create_uuid_to_path_mapping()
            # Save the mapping
            with open(mapping_file_path, "w") as f:
                json.dump(self.map_uuid_to_path, f, indent=2)
            logger.info(f"Saved UUID to JSON path mapping to {mapping_file_path}")

    def _create_uuid_to_path_mapping(self):
        """Create mapping from object UUIDs to JSON file paths by scanning all JSON files."""
        # Get all JSON files, excluding mapping files
        json_files = glob.glob(os.path.join(self.grasp_root_dir, "*.json"))
        json_files = [f for f in json_files if not f.endswith("map_uuid_to_path.json")]

        logger.info(f"Scanning {len(json_files)} JSON files to create UUID mapping")

        for json_file in tqdm(json_files, desc="Creating UUID mapping"):
            try:
                with open(json_file, "r") as f:
                    grasps_dict = json.load(f)

                # Extract file path from the grasp data according to the schema
                if "object" in grasps_dict and "file" in grasps_dict["object"]:
                    file_path = grasps_dict["object"]["file"]
                    # Use the file path as the object_id (UUID)
                    # Save only the basename of the JSON file
                    self.map_uuid_to_path[file_path] = os.path.basename(json_file)
                else:
                    logger.warning(
                        f"JSON file {json_file} does not contain expected 'object.file' field"
                    )

            except Exception as e:
                logger.warning(f"Error processing JSON file {json_file}: {e}")
                continue

        logger.info(f"Created mapping for {len(self.map_uuid_to_path)} objects")

    def read_grasps_by_uuid(self, object_id: str) -> Union[Dict, None]:
        """
        Read grasps data for a specific object UUID.

        Args:
            object_id (str): UUID/path of the object to retrieve

        Returns:
            Union[Dict, None]: Dictionary containing the grasps data if found, None otherwise
        """

        if object_id not in self.map_uuid_to_path:
            logger.debug(f"Object ID {object_id} not found in UUID mapping")
            return None

        json_file_basename = self.map_uuid_to_path[object_id]
        json_file_path = os.path.join(self.grasp_root_dir, json_file_basename)

        try:
            with open(json_file_path, "r") as f:
                grasps_dict = json.load(f)
            return grasps_dict
        except Exception as e:
            logger.error(f"Error loading grasps from {json_file_path}: {e}")
            return None


def sample_points(xyz, num_points):
    num_replica = num_points // xyz.shape[0]
    num_remain = num_points % xyz.shape[0]
    pt_idx = torch.randperm(xyz.shape[0])
    pt_idx = torch.cat([pt_idx for _ in range(num_replica)] + [pt_idx[:num_remain]])
    return pt_idx


def load_from_json(json_path) -> dict:
    if os.path.exists(json_path):
        return json.load(open(json_path, "r"))
    else:
        return {}


@dataclass
class ObjectGraspDataset:
    object_mesh: trimesh.base.Trimesh
    positive_grasps: np.ndarray
    contacts: np.ndarray
    object_asset_path: str
    object_scale: float = 1.0
    negative_grasps: np.ndarray = None
    positive_grasps_onpolicy: np.ndarray = None
    negative_grasps_onpolicy: np.ndarray = None

    def export_to_dict(self) -> Dict:
        return {
            "object_mesh": None,
            "positive_grasps": self.positive_grasps,
            "contacts": self.contacts,
            "object_asset_path": self.object_asset_path,
            "object_scale": self.object_scale,
            "negative_grasps": self.negative_grasps,
            "positive_grasps_onpolicy": self.positive_grasps_onpolicy,
            "negative_grasps_onpolicy": self.negative_grasps_onpolicy,
        }

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            object_mesh=None,
            positive_grasps=load_h5_handle_empty_case(data["positive_grasps"]),
            contacts=load_h5_handle_empty_case(data["contacts"]),
            object_asset_path=data["object_asset_path"][...].item().decode("utf-8"),
            object_scale=data["object_scale"][...].item(),
            negative_grasps=load_h5_handle_empty_case(data["negative_grasps"]),
            positive_grasps_onpolicy=load_h5_handle_empty_case(
                data["positive_grasps_onpolicy"]
            ),
            negative_grasps_onpolicy=load_h5_handle_empty_case(
                data["negative_grasps_onpolicy"]
            ),
        )


def filter_grasps_by_point_cloud_visibility(
    grasps: np.ndarray,
    pointcloud: Union[np.ndarray, torch.Tensor],
    transform_from_base_link_to_tool_tcp: np.ndarray,
    radius: float = 0.03,
) -> Union[np.ndarray, None]:
    """
    Grasps are assumed to be in the point cloud frame.

    Removes grasps are in the self-occluded regions (not visible from current camera pose) of the point cloud
    """
    num_grasps_initial = grasps.shape[0]

    if num_grasps_initial == 0:
        return None

    grasps_tool_tip_locations = np.array(
        [g @ transform_from_base_link_to_tool_tcp for g in grasps]
    )

    grasps_tool_tip_locations = torch.from_numpy(
        grasps_tool_tip_locations
    ).float()  # (N1, 3)
    if type(pointcloud) == np.ndarray:
        pointcloud = torch.from_numpy(pointcloud).float()  # (N2, 3)

    norm = 1
    num_nneighbors = 5
    dist_matrix = torch.cdist(
        grasps_tool_tip_locations[:, :3, 3], pointcloud, p=norm
    )  # (N1, N2)

    nn_dists, idxs = torch.topk(dist_matrix, num_nneighbors, dim=1, largest=False)
    mask = nn_dists.mean(1) < radius

    if mask.sum().item() > 0:
        # print(f"Filtered out {num_grasps_initial - mask.sum().item()}/{num_grasps_initial} grasps based on point cloud visibility")
        mask = mask.numpy()
    else:
        mask = None
    return mask


class GraspGenDatasetCache(dict):
    def __init__(self):
        self._cache = {}

    def __getitem__(self, key: str) -> Tuple[ObjectGraspDataset, dict]:
        return self._cache[key]

    def __setitem__(self, key: str, value: Tuple[ObjectGraspDataset, dict]):
        self._cache[key] = value

    def __len__(self):
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    @classmethod
    def load_from_h5_file(cls, path_to_h5_file: str):
        t0 = time.time()
        cache = cls()
        h5_file = h5py.File(path_to_h5_file, "r")

        for key_h5 in tqdm(
            h5_file.keys(), desc=f"Loading cache from H5 file: {path_to_h5_file}"
        ):

            h5_obj = h5_file[key_h5]
            renderings = []
            object_grasp_data = ObjectGraspDataset.from_dict(h5_obj["grasp_data"])
            if key_h5.find("____") >= 0:
                key = key_h5.replace("____", "/")
            else:
                key = key_h5
            for i in h5_obj["renderings"].keys():
                rendering_data_h5 = h5_obj["renderings"][i]
                rendering_data_dict = {}
                for bool_key in ["mesh_mode", "load_contact_batch", "invalid"]:
                    rendering_data_dict[bool_key] = (
                        rendering_data_h5[bool_key][...].astype(np.bool_).item()
                    )
                rendering_data_dict["points"] = rendering_data_h5["points"][...]
                rendering_data_dict["T_move_to_pc_mean"] = rendering_data_h5[
                    "T_move_to_pc_mean"
                ][...]
                rendering_data_dict["positive_grasps"] = rendering_data_h5[
                    "positive_grasps"
                ][...]
                renderings.append(rendering_data_dict)

            cache[key] = (object_grasp_data, renderings)
        logger.info(f"Loading cache from {path_to_h5_file} took {time.time() - t0}(s)")
        return cache

    def save_to_h5_file(self, path_to_h5_file: str):
        t0 = time.time()
        logger.info(f"Deleting old cache at {path_to_h5_file}")

        if os.path.exists(path_to_h5_file):
            os.system(f"rm {path_to_h5_file}")  # For safety

        output_file = h5py.File(path_to_h5_file, "a")
        for key, (object_grasp_data, rendering_output) in tqdm(
            self._cache.items(), desc=f"Saving cache to H5 file: {path_to_h5_file}"
        ):

            if key.find("/") >= 0:
                key_h5 = key.replace(
                    "/", "____"
                )  # This is a hack to avoid path problems in H5 files, if the key is already a path
            else:
                key_h5 = key
            grp = output_file.create_group(key_h5)
            object_grasp_data_dict = object_grasp_data.export_to_dict()
            grp_key = grp.create_group("grasp_data")
            write_info(grp_key, object_grasp_data_dict)
            grp_key = grp.create_group("renderings")
            for i, rendering_data in enumerate(rendering_output):
                grp_key_i = grp_key.create_group(f"{i}")
                write_info(grp_key_i, rendering_data)
        output_file.close()
        logger.info(f"Saving cache to {path_to_h5_file} took {time.time() - t0}(s)")


def compute_emd_data(
    ogd: ObjectGraspDataset, num_grasps: int = 1000, num_samples: int = 5
):
    """
    Compute EMD data for a given object grasp dataset
    """
    from graspgenx.metrics import compute_pose_emd

    data = {}
    if ogd.positive_grasps is not None:
        emds = [
            compute_pose_emd(
                ogd.positive_grasps[
                    torch.randint(0, ogd.positive_grasps.shape[0], (num_grasps,))
                ],
                ogd.positive_grasps[
                    torch.randint(0, ogd.positive_grasps.shape[0], (num_grasps,))
                ],
            )
            for _ in range(num_samples)
        ]
        data["off-off-pos"] = np.mean(emds)

    if ogd.negative_grasps is not None:
        emds = [
            compute_pose_emd(
                ogd.negative_grasps[
                    torch.randint(0, ogd.negative_grasps.shape[0], (num_grasps,))
                ],
                ogd.negative_grasps[
                    torch.randint(0, ogd.negative_grasps.shape[0], (num_grasps,))
                ],
            )
            for _ in range(num_samples)
        ]
        data["off-off-neg"] = np.mean(emds)

    try:
        if ogd.positive_grasps_onpolicy is not None:
            if ogd.positive_grasps_onpolicy.shape[0] > 0:
                emds = [
                    compute_pose_emd(
                        ogd.positive_grasps_onpolicy[
                            torch.randint(
                                0, ogd.positive_grasps_onpolicy.shape[0], (num_grasps,)
                            )
                        ],
                        ogd.positive_grasps_onpolicy[
                            torch.randint(
                                0, ogd.positive_grasps_onpolicy.shape[0], (num_grasps,)
                            )
                        ],
                    )
                    for _ in range(num_samples)
                ]
            data["on-on-pos"] = np.mean(emds)

        if ogd.negative_grasps_onpolicy is not None:
            if ogd.negative_grasps_onpolicy.shape[0] > 0:
                emds = [
                    compute_pose_emd(
                        ogd.negative_grasps_onpolicy[
                            torch.randint(
                                0, ogd.negative_grasps_onpolicy.shape[0], (num_grasps,)
                            )
                        ],
                        ogd.negative_grasps_onpolicy[
                            torch.randint(
                                0, ogd.negative_grasps_onpolicy.shape[0], (num_grasps,)
                            )
                        ],
                    )
                    for _ in range(num_samples)
                ]
            data["on-on-neg"] = np.mean(emds)

        if ogd.positive_grasps is not None and ogd.positive_grasps_onpolicy is not None:
            if (
                ogd.positive_grasps.shape[0] > 0
                and ogd.positive_grasps_onpolicy.shape[0] > 0
            ):
                emds = [
                    compute_pose_emd(
                        ogd.positive_grasps[
                            torch.randint(
                                0, ogd.positive_grasps.shape[0], (num_grasps,)
                            )
                        ],
                        ogd.positive_grasps_onpolicy[
                            torch.randint(
                                0, ogd.positive_grasps_onpolicy.shape[0], (num_grasps,)
                            )
                        ],
                    )
                    for _ in range(num_samples)
                ]
            data["off-on-pos"] = np.mean(emds)

        if ogd.negative_grasps is not None and ogd.negative_grasps_onpolicy is not None:
            if (
                ogd.negative_grasps.shape[0] > 0
                and ogd.negative_grasps_onpolicy.shape[0] > 0
            ):
                emds = [
                    compute_pose_emd(
                        ogd.negative_grasps[
                            torch.randint(
                                0, ogd.negative_grasps.shape[0], (num_grasps,)
                            )
                        ],
                        ogd.negative_grasps_onpolicy[
                            torch.randint(
                                0, ogd.negative_grasps_onpolicy.shape[0], (num_grasps,)
                            )
                        ],
                    )
                    for _ in range(num_samples)
                ]
            data["off-on-neg"] = np.mean(emds)
    except:
        from IPython import embed

        embed()

    return data


def load_object_grasp_data(
    key,
    object_root_dir,
    grasp_root_dir,
    dataset_version="v2",
    load_discriminator_dataset=False,
    gripper_info=None,
    onpolicy_dataset_dir=None,
    onpolicy_dataset_h5_path=None,
    onpolicy_json_path=None,
    onpolicy_data_found=False,
    grasp_dataset_reader: Union[GraspWebDatasetReader, GraspJsonDatasetReader] = None,
) -> Tuple[DataLoaderError, Union[ObjectGraspDataset, None]]:

    if grasp_dataset_reader is not None:
        grasp_root_dir = None

    if onpolicy_dataset_dir is not None:
        if not onpolicy_data_found:
            onpolicy_json_path = None
            onpolicy_dataset_h5_path = None
            onpolicy_dataset_dir = None
    error_code, object_grasp_data = DataLoaderError.SUCCESS, None

    if dataset_version == "v1":
        error_code, object_grasp_data = load_object_grasp_acronym(
            key,
            object_root_dir,
            grasp_root_dir,
            load_discriminator_dataset=load_discriminator_dataset,
            onpolicy_dataset_dir=onpolicy_dataset_dir,
            onpolicy_dataset_h5_path=onpolicy_dataset_h5_path,
            onpolicy_json_path=onpolicy_json_path,
        )
    elif dataset_version == "v2":
        error_code, object_grasp_data = load_object_grasp_datapoint_objaverse(
            key,
            object_root_dir,
            grasp_root_dir,
            load_discriminator_dataset=load_discriminator_dataset,
            onpolicy_dataset_dir=onpolicy_dataset_dir,
            onpolicy_dataset_h5_path=onpolicy_dataset_h5_path,
            onpolicy_json_path=onpolicy_json_path,
            grasp_dataset_reader=grasp_dataset_reader,
        )
    else:
        raise NotImplementedError(
            f"Version {dataset_version} of object dataset {object_root_dir} not implemented"
        )

    if object_grasp_data is not None:
        offset_transform = gripper_info.offset_transform
        object_grasp_data.positive_grasps = np.array(
            [g @ offset_transform for g in object_grasp_data.positive_grasps]
        )
        if object_grasp_data.negative_grasps is not None:
            object_grasp_data.negative_grasps = np.array(
                [g @ offset_transform for g in object_grasp_data.negative_grasps]
            )
        if object_grasp_data.negative_grasps_onpolicy is not None:
            object_grasp_data.negative_grasps_onpolicy = np.array(
                [
                    g @ offset_transform
                    for g in object_grasp_data.negative_grasps_onpolicy
                ]
            )
        if object_grasp_data.positive_grasps_onpolicy is not None:
            object_grasp_data.positive_grasps_onpolicy = np.array(
                [
                    g @ offset_transform
                    for g in object_grasp_data.positive_grasps_onpolicy
                ]
            )

        if object_grasp_data.contacts is not None:
            pass

    return error_code, object_grasp_data


def get_json_file_given_object_id(json_files: List[str], object_id_key: str) -> Dict:
    """This func is needed as there 1-2 discrepancies in objects."""
    object_id_dict = {}
    # Brute force
    for json_path in json_files:
        # json_path_object_id = json_path.split('/')[-2].split('_')[-1]
        # json_path_object_id = load_from_json(json_path)['object']['file'].split('_')[-1].split('.')[0][1:]
        json_path_object_id = load_from_json(json_path)["object"]["file"]
        if os.path.basename(json_path_object_id) == os.path.basename(object_id_key):
            object_id_dict[json_path_object_id] = {
                "json_path": json_path,
                "object_id_key": object_id_key,
            }
            break
    return object_id_dict


def load_onpolicy_dataset(
    key: str,
    onpolicy_dataset_dir: str,
    onpolicy_dataset_h5_path: str,
    onpolicy_json_path: str,
    dataset_version: str = "v4",
) -> Tuple[np.ndarray, np.ndarray]:
    logger.info(f"Loading onpolicy dataset for {key}: {onpolicy_json_path}")

    h5 = h5py.File(onpolicy_dataset_h5_path, "r")
    h5_obj = h5["objects"][key]

    json_path = onpolicy_json_path

    pred_grasps = h5_obj["pred_grasps"][...]
    gt_grasps = h5_obj["gt_grasps"][
        ...
    ]  # TODO: This is not real ground truth. Replace this...
    scores = h5_obj["confidence"][...]
    collision = h5_obj["collision"][...]
    mask_not_colliding = np.logical_not(collision)
    num_grasps_attempted_inference = mask_not_colliding.sum()
    data = json.load(open(json_path, "rb"))
    try:
        num_grasps_attempted_igg = len(data["grasps"]["transforms"])
    except:
        logger.error(f"ONPOLICY: Error in opening file {key}: {onpolicy_json_path}")
        return None, None
    # pred_grasps2 = np.array(data['grasps']['transforms'])
    mask_eval_success = np.array(data["grasps"]["object_in_gripper"])
    num_successful_attempts = np.sum(mask_eval_success)
    success_result = np.zeros(len(scores))

    try:
        # print(f"num_grasps_attempted_inference: {num_grasps_attempted_inference}, num_grasps_attempted_igg: {num_grasps_attempted_igg}")
        assert (
            num_grasps_attempted_inference == num_grasps_attempted_igg
        )  # Sanity check if the object id in h5 files (inference) is the same as the index in the evaluated json files (IGG)
    except:
        logger.error(
            f"ONPOLICY: Number of attempted grasps in h5 file (predicted) and Issac Sim does not match... {num_grasps_attempted_inference} vs. {num_grasps_attempted_igg} for object {key}."
        )
        return None, None

    success_mask = np.where(mask_not_colliding)[0][np.where(mask_eval_success)[0]]
    success_result[success_mask] = 1.0
    positive_grasps_onpolicy = pred_grasps[success_result.astype(np.bool_)]
    negative_grasps_onpolicy = pred_grasps[~success_result.astype(np.bool_)]
    logger.info(
        f"ONPOLICY: num pos {len(positive_grasps_onpolicy)}, num neg: {len(negative_grasps_onpolicy)}"
    )
    return positive_grasps_onpolicy, negative_grasps_onpolicy


def load_object_grasp_acronym(
    key: str,
    object_root_dir: str,
    grasp_root_dir: str,
    load_discriminator_dataset: bool = False,
    onpolicy_dataset_dir: str = None,
    onpolicy_dataset_h5_path: str = None,
    onpolicy_json_path: str = None,
    min_pos_grasps_gen: int = 5,
    min_neg_grasps_dis: int = 5,
    min_pos_grasps_dis: int = 5,
) -> ObjectGraspDataset:
    """
    For a given datapoint key, returns the following:
    - Object mesh, loaded and scaled
    - Grasps
    - Contact points [optional]
    """

    positive_grasps_onpolicy = None
    negative_grasps_onpolicy = None
    if onpolicy_dataset_h5_path is not None:
        positive_grasps_onpolicy, negative_grasps_onpolicy = load_onpolicy_dataset(
            key, onpolicy_dataset_dir, onpolicy_dataset_h5_path, onpolicy_json_path
        )

    object_name = key.split("/")[-1]
    cat, model, scale = object_name.split("_")
    scale = float(scale)
    asset_path = os.path.join(object_root_dir, f"meshes/{cat}/{model}.obj")

    # Load grasps
    # t0 = time.time()
    success_og = np.load(f"{grasp_root_dir}/grasp_eval/{object_name}.npy")
    grasp_data = np.load(f"{grasp_root_dir}/contacts/{object_name}.npz")
    # print(f"Loading graspdata took {time.time() - t0}s") # This is minimal 0.0006771087646484375s
    success = success_og[grasp_data["grasp_ids"]] & grasp_data["successful"].astype(
        "bool"
    )
    negative_grasps = None
    try:
        grasps = grasp_data["grasp_transform"]
        positive_grasps = grasps[success]

        if load_discriminator_dataset:

            not_success = np.logical_not(
                success_og[grasp_data["grasp_ids"]]
            ) & np.logical_not(grasp_data["successful"].astype("bool"))
            negative_grasps = grasps[not_success]

            if positive_grasps.shape[0] < min_pos_grasps_dis:
                return (
                    DataLoaderError.INSUFFICIENT_GRASPS_FOR_DISCRIMINATOR_DATASET,
                    None,
                )

            if negative_grasps is None:
                return (
                    DataLoaderError.INSUFFICIENT_GRASPS_FOR_DISCRIMINATOR_DATASET,
                    None,
                )

            if negative_grasps.shape[0] < min_neg_grasps_dis:
                return (
                    DataLoaderError.INSUFFICIENT_GRASPS_FOR_DISCRIMINATOR_DATASET,
                    None,
                )

    except:
        logger.error("ERROR: grasp_transform.npy is corrupted for ", object_name)
        return DataLoaderError.GRASPS_FILE_LOAD_ERROR, None

    grasp_ids = grasp_data["grasp_ids"][success]

    try:
        contacts = grasp_data["contact_points"][success]
    except:
        logger.error(f"Contact not loaded for key {key}")
        return DataLoaderError.GRASPS_FILE_LOAD_ERROR, None

    offsets = np.linalg.norm(contacts[:, 1] - contacts[:, 0], axis=1)
    valid = offsets > 1e-3
    positive_grasps = positive_grasps[valid]
    grasp_ids = grasp_ids[valid]
    contacts = contacts[valid]

    if not load_discriminator_dataset:
        # Only do this for the generator
        if positive_grasps.shape[0] < min_pos_grasps_gen:
            # print(f"ERROR: Skipping object {object_name} since there are too few grasps ({positive_grasps.shape[0]})")
            # print(f"object_name:{object_name} num pos:{len(positive_grasps)}, neg:{len(negative_grasps)} grasps")
            return DataLoaderError.INSUFFICIENT_GRASPS_FOR_GENERATOR_DATASET, None

    # t0 = time.time()
    obj_mesh = trimesh.load(asset_path)
    obj_mesh.apply_scale(scale)

    return DataLoaderError.SUCCESS, ObjectGraspDataset(
        obj_mesh,
        positive_grasps,
        contacts,
        asset_path,
        scale,
        negative_grasps=negative_grasps,
        positive_grasps_onpolicy=positive_grasps_onpolicy,
        negative_grasps_onpolicy=negative_grasps_onpolicy,
    )


def load_grasp_data(
    object_id: str,
    grasp_dataset_reader: Union[GraspWebDatasetReader, GraspJsonDatasetReader] = None,
) -> Tuple[DataLoaderError, Union[Dict, None]]:
    """
    Loads grasp data from either a webdataset, json dataset reader, or a regular folder.
    """
    assert grasp_dataset_reader is not None, "Must provide grasp_dataset_reader"

    grasps_dict = grasp_dataset_reader.read_grasps_by_uuid(object_id)
    if grasps_dict is None:
        logger.info(f"No grasp found for object {object_id} using dataset reader")
        return DataLoaderError.GRASPS_FILE_NOT_FOUND, None
    return DataLoaderError.SUCCESS, grasps_dict


def load_object_grasp_datapoint_objaverse(
    object_id: str,
    object_root_dir: str = None,
    grasp_root_dir: str = None,
    load_discriminator_dataset: bool = False,
    onpolicy_dataset_dir: str = None,
    onpolicy_dataset_h5_path: str = None,
    onpolicy_json_path: str = None,
    grasp_dataset_reader: Union[GraspWebDatasetReader, GraspJsonDatasetReader] = None,
    min_pos_grasps_gen: int = 5,
    min_neg_grasps_dis: int = 5,
    min_pos_grasps_dis: int = 5,
) -> Tuple[DataLoaderError, Union[ObjectGraspDataset, None]]:
    """

    Args:
        object_root_dir: Root directory of the object dataset
        grasp_root_dir: Root directory of the grasp dataset
        object_id: Key of the object to load
        load_discriminator_dataset: Whether to load the discriminator dataset
        onpolicy_dataset_dir: Directory of the onpolicy dataset
        onpolicy_dataset_h5_path: Path to the onpolicy dataset h5 file
        onpolicy_json_path: Path to the onpolicy dataset json file
        grasp_dataset_reader: Dataset reader to use if the grasp dataset is stored in a webdataset or json format
        min_pos_grasps_gen: Minimum number of positive grasps for the generator dataset
        min_neg_grasps_dis: Minimum number of negative grasps for the discriminator dataset
        min_pos_grasps_dis: Minimum number of positive grasps for the discriminator dataset
    """

    positive_grasps_onpolicy = None
    negative_grasps_onpolicy = None
    if onpolicy_dataset_h5_path is not None:
        positive_grasps_onpolicy, negative_grasps_onpolicy = load_onpolicy_dataset(
            object_id,
            onpolicy_dataset_dir,
            onpolicy_dataset_h5_path,
            onpolicy_json_path,
        )

    error_code, grasps_dict = load_grasp_data(object_id, grasp_dataset_reader)

    if error_code != DataLoaderError.SUCCESS:
        return error_code, grasps_dict

    object_file = os.path.join(object_root_dir, object_id)
    if not os.path.exists(object_file):
        # Load object paths
        uuid_object_paths_file = os.path.join(
            object_root_dir, "map_uuid_to_path_simplified.json"
        )
        if not os.path.exists(uuid_object_paths_file):
            uuid_object_paths_file = os.path.join(
                object_root_dir, "map_uuid_to_path.json"
            )
        if not os.path.exists(uuid_object_paths_file):
            raise FileNotFoundError(
                f"Could not find mapping file at {uuid_object_paths_file}"
            )
        try:
            uuid_to_path = json.load(open(uuid_object_paths_file))
            if object_id not in uuid_to_path:
                return DataLoaderError.UUID_NOT_FOUND_IN_MAPPING, None
            object_file = uuid_to_path[object_id]
            object_file = os.path.join(object_root_dir, object_file)
            if not os.path.exists(object_file):
                logger.error(f"Object mesh not found, at {object_file}")
                return DataLoaderError.OBJECT_MESH_NOT_FOUND, None

        except Exception as e:
            return DataLoaderError.UUID_MAPPING_LOAD_ERROR, None

    object_scale = grasps_dict["object"]["scale"]
    grasps = grasps_dict["grasps"]

    grasp_poses = np.array(grasps["transforms"])
    grasp_mask = np.array(grasps["object_in_gripper"])

    positive_grasps = grasp_poses[grasp_mask]
    negative_grasps = None
    contacts = None
    not_success = np.logical_not(grasp_mask)
    negative_grasps = grasp_poses[not_success]

    if not load_discriminator_dataset:
        # Only do this for the generator
        if positive_grasps.shape[0] < min_pos_grasps_gen:
            logger.debug(
                f"Skipping object {object_id} since there are too few grasps: num pos:{len(positive_grasps)}, neg:{len(negative_grasps) if negative_grasps is not None else 0} grasps"
            )
            return DataLoaderError.INSUFFICIENT_GRASPS_FOR_GENERATOR_DATASET, None

    if "contact_locations" in grasps:
        contacts = np.array(grasps["contact_locations"])
        contacts = contacts[grasp_mask]

        mask_grasp_with_contacts = np.isnan(contacts)
        mask_grasp_with_contacts = np.any(mask_grasp_with_contacts, axis=1)
        if contacts.shape[-1] == 3:
            mask_grasp_with_contacts = np.any(mask_grasp_with_contacts, axis=1)
        mask_grasp_with_contacts = np.logical_not(mask_grasp_with_contacts)

        if mask_grasp_with_contacts.sum() == 0:
            logger.debug(
                f"No grasp with contacts found for object {object_id}. Number of positive grasps: {positive_grasps.shape[0]}"
            )
            return DataLoaderError.GRASPS_HAVE_INVALID_CONTACT_POINTS, None

        positive_grasps = positive_grasps[mask_grasp_with_contacts]
        contacts = contacts[mask_grasp_with_contacts]
    try:
        object_mesh = trimesh.load(object_file)

        if type(object_mesh) == trimesh.Scene:
            object_mesh = object_mesh.dump(concatenate=True)

        object_mesh.apply_scale(object_scale)
    except:
        logger.debug(f"Unable to load object mesh at {object_file}")
        return DataLoaderError.OBJECT_MESH_LOAD_ERROR, None

    # HACK - Only works for single cup suction gripper
    if contacts is None:
        cp = np.array([[0.0, 0, 0]])
        if len(positive_grasps) > 0:
            contacts = np.vstack([tra.transform_points(cp, g) for g in positive_grasps])

    return DataLoaderError.SUCCESS, ObjectGraspDataset(
        object_mesh,
        positive_grasps,
        contacts,
        object_file,
        object_scale,
        negative_grasps=negative_grasps,
        positive_grasps_onpolicy=positive_grasps_onpolicy,
        negative_grasps_onpolicy=negative_grasps_onpolicy,
    )


def dump_object_list(objects, json_file_path):
    json_file = open(json_file_path, "w")
    json.dump(objects, json_file)
    json_file.close()


def get_rotation_augmentation(stratified_sampling: bool = True) -> np.ndarray:
    """
    Applies stratified sampling for rotation augmentation.

    Somestimes the object/point cloud are axis aligned. Random sampling over the rotation space has a very low chance of
    hitting these axis-aligned orientations. This function applies stratified sampling to ensure some rotations are sampled accordingly
    """

    # Stratified Sampling for rotation augmentation
    p = np.random.random()
    if p < 0.25:  # 25%
        base_rotation = np.eye(4)
    elif p < 0.50:  # 25%
        base_rotation = tra.euler_matrix(np.pi / 2, 0, 0)
    elif p < 0.75:  # 25%
        base_rotation = tra.euler_matrix(0, np.pi / 2, 0)
    else:  # 25%
        base_rotation = tra.euler_matrix(0, 0, np.pi / 2)

    rotation = np.random.uniform(-np.pi, np.pi, 3)
    p = np.random.random()
    if not stratified_sampling:
        rotation = tra.euler_matrix(rotation[0], rotation[1], rotation[2])
    elif p < 0.25:  # 25%
        rotation = np.eye(4)
    elif p < 0.30:  # 5%
        rotation = tra.euler_matrix(rotation[0], 0, 0)
    elif p < 0.35:  # 5%
        rotation = tra.euler_matrix(0, rotation[1], 0)
    elif p < 0.40:  # 5%
        rotation = tra.euler_matrix(0, 0, rotation[2])
    else:  # 60%
        rotation = tra.euler_matrix(rotation[0], rotation[1], rotation[2])
    rotation = rotation @ base_rotation
    return rotation
