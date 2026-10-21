# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import webdataset as wds
from tqdm import tqdm

from graspgenx.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_uuid_list(uuid_list_path: str) -> list[str]:
    """
    Load a list of UUIDs from a JSON or text file.

    Args:
        uuid_list_path (str): Path to the UUID list file

    Returns:
        list[str]: List of UUIDs
    """
    if not os.path.exists(uuid_list_path):
        raise FileNotFoundError(f"UUID list file not found: {uuid_list_path}")
    if uuid_list_path.endswith(".json"):
        with open(uuid_list_path, "r") as f:
            uuids = json.load(f)
        if type(uuids) == list:
            return uuids
        elif type(uuids) == dict:
            return list(uuids.keys())
        else:
            raise ValueError(f"UUID list is not a list or dict: {uuids}")
    elif uuid_list_path.endswith(".txt"):
        with open(uuid_list_path, "r") as f:
            uuids = [line.strip() for line in f.readlines()]
    else:
        raise ValueError(f"Unsupported file format: {uuid_list_path}")
    return uuids


def convert_to_webdataset(
    root_dir: str, num_shards: int = 8, output_dir: str = None
) -> None:
    """
    Convert dataset to WebDataset format and create an index file.

    Args:
        root_dir (str): Path to root directory containing object folders
        num_shards (int): Number of shards to create (default: 8)
    """
    # Create output directory for shards
    if output_dir is None:
        output_dir = os.path.join(root_dir, "webdataset_shards")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize UUID to shard index mapping
    uuid_to_shard = {}

    # Get all folders matching the pattern <integer_id>_<uuid>
    object_folders = glob.glob(os.path.join(root_dir, "*_*"))

    # Calculate items per shard
    items_per_shard = len(object_folders) // num_shards

    # Create WebDataset shards
    for shard_idx in range(num_shards):
        shard_name = f"{output_dir}/shard_{shard_idx:03d}.tar"
        start_idx = shard_idx * items_per_shard
        end_idx = (
            start_idx + items_per_shard
            if shard_idx < num_shards - 1
            else len(object_folders)
        )

        logger.info(f"Creating shard {shard_idx + 1}/{num_shards}")

        with wds.TarWriter(shard_name) as sink:
            # Process folders for this shard
            for folder in tqdm(object_folders[start_idx:end_idx]):
                # Get integer_id and uuid from folder name
                folder_name = os.path.basename(folder)

                integer_id, uuid = folder_name.split("_", 1)

                # Store shard location for this UUID
                uuid_to_shard[uuid] = shard_idx

                # Read grasps data
                grasp_file = os.path.join(folder, "grasps.json")
                with open(grasp_file, "r") as f:
                    grasps_data = json.load(f)

                # Create sample dictionary using uuid as key
                sample = {
                    "__key__": uuid,
                    "integer_id": integer_id,
                    "grasps.json": json.dumps(grasps_data),
                }

                # Write sample to shard
                sink.write(sample)

    # Save the UUID to shard mapping
    index_path = os.path.join(output_dir, "uuid_index.json")
    with open(index_path, "w") as f:
        json.dump(uuid_to_shard, f)


def is_webdataset(dataset_path: str) -> bool:
    """
    Check if directory has tar files.

    Args:
        directory (str): Path to directory to check

    Returns:
        bool: True if directory contains at least one .tar file, False otherwise
    """
    # Check if directory exists
    if not os.path.isdir(dataset_path):
        return False

    # Look for .json files
    tar_files = glob.glob(os.path.join(dataset_path, "*.tar"))
    return len(tar_files) > 0


class GraspWebDatasetReader:
    """Class to efficiently read grasps data using a pre-loaded index."""

    def __init__(self, dataset_path: str):
        """
        Initialize the reader with dataset path and load the index.

        Args:
            dataset_path (str): Path to directory containing WebDataset shards
        """
        self.dataset_path = dataset_path
        self.shards_dir = self.dataset_path

        # Load the UUID index
        index_path = os.path.join(self.shards_dir, "uuid_index.json")
        with open(index_path, "r") as f:
            self.uuid_index = json.load(f)

        # Cache for open datasets
        self.shard_datasets = {}

    def read_grasps_by_uuid(self, object_uuid: str) -> Optional[Dict]:
        """
        Read grasps data for a specific object UUID using the index.

        Args:
            object_uuid (str): UUID of the object to retrieve

        Returns:
            Optional[Dict]: Dictionary containing the grasps data if found, None otherwise
        """
        if object_uuid not in self.uuid_index:
            return None

        shard_idx = self.uuid_index[object_uuid]

        # Get or create dataset for this shard
        if shard_idx not in self.shard_datasets:
            shard_path = f"{self.shards_dir}/shard_{shard_idx:03d}.tar"
            self.shard_datasets[shard_idx] = wds.WebDataset(shard_path)

        dataset = self.shard_datasets[shard_idx]

        # Search for the UUID in the specific shard
        for sample in dataset:
            if sample["__key__"] == object_uuid:
                return json.loads(sample["grasps.json"])

        return None
