# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum


@dataclass
class ErrorInfo:
    code: int
    description: str


class DataLoaderError(Enum):
    # Success
    SUCCESS = ErrorInfo(0, "Datapoint loaded successfully")

    # Data loading errors (100-199)
    GRASPS_FILE_NOT_FOUND = ErrorInfo(100, "No grasp file located for object")
    GRASPS_FILE_NOT_FOUND_OFFLINE = ErrorInfo(
        114, "No offline grasp file, but on-policy data available"
    )
    GRASPS_FILE_NOT_FOUND_ONLINE = ErrorInfo(
        115, "No on-policy grasp data, but offline grasps available"
    )
    GRASPS_FILE_NOT_FOUND_BOTH = ErrorInfo(
        116, "No offline or on-policy grasp data found"
    )
    OBJECT_SCALE_NOT_FOUND = ErrorInfo(
        117, "Unable to find object scale from any source"
    )
    GRASPS_FILE_LOAD_ERROR = ErrorInfo(101, "Unable to load grasp file successfully")
    INSUFFICIENT_GRASPS_FOR_GENERATOR_DATASET = ErrorInfo(
        102, "Not enough grasps provided (minimum 5 required)"
    )
    INSUFFICIENT_GRASPS_FOR_DISCRIMINATOR_DATASET = ErrorInfo(
        103, "Not enough grasps provided (minimum 5 required)"
    )
    GRASPS_HAVE_INVALID_CONTACT_POINTS = ErrorInfo(
        104, "Grasps have invalid contact points"
    )
    OBJECT_FILE_MAPPING_ERROR = ErrorInfo(
        105, "Object file was not mapped properly to grasp dataset"
    )
    OBJECT_FILE_NOT_FOUND = ErrorInfo(106, "Object file was not found")
    OBJECT_MESH_LOAD_ERROR = ErrorInfo(107, "Unable to load mesh")
    AMBIGUOUS_LOAD_ERROR = ErrorInfo(108, "Not sure why this error occurred.")
    GRASP_POINT_CLOUD_CORRESPONDENCE_ERROR = ErrorInfo(
        109, "Unable to find valids grasps for visible point cloud"
    )
    OBJECT_MESH_NOT_FOUND = ErrorInfo(110, "Object mesh not found")
    UUID_MAPPING_NOT_FOUND = ErrorInfo(111, "UUID mapping not found")
    UUID_NOT_FOUND_IN_MAPPING = ErrorInfo(112, "UUID not found in mapping")
    UUID_MAPPING_LOAD_ERROR = ErrorInfo(113, "Error loading UUID mapping")

    # Rendering errors (200-299)
    RENDERING_SUCCESS = ErrorInfo(200, "Rendering successful")
    RENDERING_ERROR = ErrorInfo(201, "Error during rendering")
    RENDERING_ERROR_POINT_CLOUD_TOO_SMALL = ErrorInfo(202, "Point cloud is too small")
    RENDERING_ERROR_POINT_CLOUD_VERY_FEW_POINTS = ErrorInfo(
        203, "Point cloud is too small"
    )
    RENDERING_NO_GRASPS_IN_VISIBLE_POINT_CLOUD = ErrorInfo(
        110, "No grasps are in the visible point cloud"
    )

    # Rendering errors (300-399)

    @property
    def code(self) -> int:
        return self.value.code

    @property
    def description(self) -> str:
        return self.value.description

    def is_success(self) -> bool:
        return self == ErrorCode.SUCCESS

    def is_error(self) -> bool:
        return not self.is_success()
