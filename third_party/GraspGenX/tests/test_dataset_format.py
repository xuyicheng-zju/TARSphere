import pytest
import json
import os
import numpy as np
from pathlib import Path
from graspgenx.dataset.webdataset_utils import is_webdataset, GraspWebDatasetReader
from graspgenx.dataset.dataset_utils import GraspJsonDatasetReader


def test_dataset_format_convention():
    """Validate dataset follows the GraspGenX dataset convention.

    This test requires actual dataset mounts and is skipped otherwise.
    """
    object_dataset_path = "/object_dataset"
    grasp_dataset_path = "/grasp_dataset"

    if not os.path.exists(object_dataset_path):
        pytest.skip(f"Object dataset not mounted at {object_dataset_path}")

    if not os.path.exists(grasp_dataset_path):
        pytest.skip(f"Grasp dataset not mounted at {grasp_dataset_path}")

    splits_path = os.path.join(grasp_dataset_path, "splits", "franka_panda")
    train_split_path = os.path.join(splits_path, "train.txt")
    valid_split_path = os.path.join(splits_path, "valid.txt")

    assert os.path.exists(splits_path), f"Splits directory not found at {splits_path}"
    assert os.path.exists(train_split_path), f"train.txt not found at {train_split_path}"
    assert os.path.exists(valid_split_path), f"valid.txt not found at {valid_split_path}"

    with open(train_split_path, "r") as f:
        train_objects = [line.strip() for line in f.readlines() if line.strip()]
    with open(valid_split_path, "r") as f:
        valid_objects = [line.strip() for line in f.readlines() if line.strip()]

    assert len(train_objects) > 0, "train.txt is empty"
    assert len(valid_objects) > 0, "valid.txt is empty"

    grasp_data_path = os.path.join(grasp_dataset_path, "grasp_data", "franka_panda")
    assert os.path.exists(grasp_data_path), f"Grasp data directory not found at {grasp_data_path}"

    if is_webdataset(grasp_data_path):
        tar_shards = list(Path(grasp_data_path).glob("shard_*.tar"))
        uuid_index_path = os.path.join(grasp_data_path, "uuid_index.json")

        assert len(tar_shards) > 0, f"No tar shards found in {grasp_data_path}"
        assert os.path.exists(uuid_index_path), f"uuid_index.json not found"

        with open(uuid_index_path, "r") as f:
            uuid_index = json.load(f)
        assert isinstance(uuid_index, dict)
        assert len(uuid_index) > 0

        grasp_dataset_reader = GraspWebDatasetReader(grasp_data_path)
    else:
        json_files = list(Path(grasp_data_path).glob("*.json"))
        assert len(json_files) > 0, f"No JSON files found in {grasp_data_path}"

        grasp_dataset_reader = GraspJsonDatasetReader(grasp_data_path)

    sample_objects = train_objects[:3] + valid_objects[:3]

    for obj_id in sample_objects:
        try:
            grasps_dict = grasp_dataset_reader.read_grasps_by_uuid(obj_id)

            if grasps_dict is None:
                continue

            assert "object" in grasps_dict, f"Missing 'object' key for {obj_id}"
            assert "grasps" in grasps_dict, f"Missing 'grasps' key for {obj_id}"

            object_section = grasps_dict["object"]
            assert "file" in object_section, f"Missing 'file' key for {obj_id}"
            assert "scale" in object_section, f"Missing 'scale' key for {obj_id}"

            grasps_section = grasps_dict["grasps"]
            assert "transforms" in grasps_section, f"Missing 'transforms' key for {obj_id}"
            assert "object_in_gripper" in grasps_section, f"Missing 'object_in_gripper' key for {obj_id}"

            transforms = np.array(grasps_section["transforms"])
            grasp_mask = np.array(grasps_section["object_in_gripper"])

            assert transforms.ndim == 3, f"Transforms should be 3D for {obj_id}"
            assert transforms.shape[1:] == (4, 4), f"Transforms should be 4x4 for {obj_id}"
            assert len(grasp_mask) == len(transforms), f"Mask length mismatch for {obj_id}"

            scale = object_section["scale"]
            assert isinstance(scale, (int, float)) and scale > 0, f"Scale should be positive for {obj_id}"

        except Exception as e:
            pytest.fail(f"Error processing object {obj_id}: {e}")
