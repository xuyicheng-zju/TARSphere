"""Resolve a known-positive grasp reference inside one real GRScene."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.bench.grounding.schema import GroundedScene
from tarsphere.bench.spatial.physical_task.schema import (
    GraspAction,
)
from tarsphere.harness.perception.backends.gt_bev import (
    OfficialOccupancyBevMap,
)
from tarsphere.harness.planner.planner import erode_navigable_mask
from tarsphere.assets import tarsphere_assets_root
from tarsphere.infra.control.physical_contracts import (
    G1_NAVIGATION_FOOTPRINT_RADIUS_M,
)


Point3 = Tuple[float, float, float]
GRASP_REFERENCE_OBJECT_ID = "tarsphere_grasp_reference"
GRASP_REFERENCE_PRIM_PATH = "/World/env_0/TARSPHERE_GraspReference"
GRASP_REFERENCE_EDGE_LENGTH_M = 0.12
GRASP_REFERENCE_CENTER_HEIGHT_M = 0.70
GRASP_REFERENCE_OBSERVATION_DISTANCE_M = 1.60
# The independently certified G1 reference task reaches the same 7 cm-wide,
# 0.70 m-high object from a 0.32 m centre distance.  Keep calibration inside
# that proven right-hand IK workspace so a calibration failure diagnoses the
# grasp stack instead of an unreachable fixture.
GRASP_REFERENCE_GRASP_DISTANCE_M = 0.32
GRASP_REFERENCE_ROUTE_LENGTH_M = 0.65
GRASP_REFERENCE_LOCAL_UNIT_IN_METERS = 0.01
EMPTY_GRASP_SCENE_ID = "empty_grasp_scene"


class StrictModel(BaseModel):
    """Immutable current-release grasp-system schema base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GraspReferenceObject(StrictModel):
    """Known geometry spawned only to calibrate the shared grasp stack."""

    object_id: str = Field(
        default=GRASP_REFERENCE_OBJECT_ID,
        min_length=1,
        description="Stable identifier of the known-positive reference object.",
    )
    prim_path: str = Field(
        default=GRASP_REFERENCE_PRIM_PATH,
        min_length=1,
        description="Dedicated USD prim created for the reference object.",
    )
    center_xyz: Point3 = Field(
        description="Reference-object centre in scene world metres.",
    )
    edge_length_m: float = Field(
        default=GRASP_REFERENCE_EDGE_LENGTH_M,
        gt=0.0,
        description="Known physical edge length of the reference cube.",
    )
    local_unit_in_meters: float = Field(
        default=GRASP_REFERENCE_LOCAL_UNIT_IN_METERS,
        gt=0.0,
        description=(
            "Authored local-unit scale, matched to GRScene manipulation "
            "objects so the same FixedJoint transform path is exercised."
        ),
    )

    @model_validator(mode="after")
    def validate_geometry(self) -> "GraspReferenceObject":
        if (
            self.object_id != GRASP_REFERENCE_OBJECT_ID
            or self.prim_path != GRASP_REFERENCE_PRIM_PATH
            or not all(math.isfinite(value) for value in self.center_xyz)
            or not math.isclose(
                self.center_xyz[2],
                GRASP_REFERENCE_CENTER_HEIGHT_M,
                abs_tol=1e-9,
            )
            or not math.isclose(
                self.edge_length_m,
                GRASP_REFERENCE_EDGE_LENGTH_M,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(
                "grasp reference world geometry is frozen by the calibration contract"
            )
        return self

    @property
    def bounds_min_xyz(self) -> Point3:
        """Return the known world-space minimum cube corner."""
        half = self.edge_length_m / 2.0
        return (
            self.center_xyz[0] - half,
            self.center_xyz[1] - half,
            self.center_xyz[2] - half,
        )

    @property
    def bounds_max_xyz(self) -> Point3:
        """Return the known world-space maximum cube corner."""
        half = self.edge_length_m / 2.0
        return (
            self.center_xyz[0] + half,
            self.center_xyz[1] + half,
            self.center_xyz[2] + half,
        )


class GraspSystemTrial(StrictModel):
    """One known-positive trial of G1 IK, contact, attach, and retraction."""

    scene_id: str = Field(
        min_length=1,
        description="Real GRScene in which the grasp stack is calibrated.",
    )
    scene_server_id: str = Field(
        min_length=1,
        description="Exact Isaac process that must execute this trial.",
    )
    scene_server_git_sha: str = Field(
        min_length=1,
        description="Repository revision loaded by that Isaac process.",
    )
    reference_object: GraspReferenceObject = Field(
        description="Known object whose geometry cannot explain a failed grasp.",
    )
    grasp: GraspAction = Field(
        description="Target-facing stance and hand target used by the normal stack.",
    )

    @model_validator(mode="after")
    def validate_trial(self) -> "GraspSystemTrial":
        if self.grasp.target_object_id != self.reference_object.object_id:
            raise ValueError("grasp-system trial bindings are inconsistent")
        return self

    @property
    def robot_start_xy(self) -> Tuple[float, float]:
        """Return the fixed near-field stance that isolates arm motion."""
        return self.grasp.robot_stance_xy

    @property
    def robot_start_yaw_rad(self) -> float:
        """Return the authored target-facing yaw for the reference object."""
        return self.grasp.robot_yaw_rad


def resolve_grasp_system_trial(
    scene: GroundedScene,
    *,
    scene_server_id: str,
    scene_server_git_sha: str,
    assets_root: Optional[Path | str] = None,
) -> GraspSystemTrial:
    """Place the fixed reference rig in deterministic open scene geometry."""
    root = (
        Path(assets_root).expanduser().resolve()
        if assets_root is not None
        else tarsphere_assets_root()
    )
    occupancy_path = (root / scene.map.occupancy).resolve()
    occupancy = OfficialOccupancyBevMap(occupancy_path=occupancy_path)
    navigation = occupancy.navigable_mask()
    pixels_per_meter = float(occupancy.pixels_per_meter)
    robot_cells = math.ceil(G1_NAVIGATION_FOOTPRINT_RADIUS_M * pixels_per_meter)
    reference_cells = math.ceil(
        (GRASP_REFERENCE_EDGE_LENGTH_M * math.sqrt(2.0) / 2.0 + 0.05) * pixels_per_meter
    )
    robot_clear = erode_navigable_mask(navigation, robot_cells)
    reference_clear = erode_navigable_mask(
        navigation,
        reference_cells,
    )
    _, _, grasp_xy, target_xy = _reference_layout(
        occupancy,
        robot_clear=robot_clear,
        reference_clear=reference_clear,
    )
    yaw = math.atan2(
        target_xy[1] - grasp_xy[1],
        target_xy[0] - grasp_xy[0],
    )
    reference = GraspReferenceObject(
        center_xyz=(
            target_xy[0],
            target_xy[1],
            GRASP_REFERENCE_CENTER_HEIGHT_M,
        )
    )
    return GraspSystemTrial(
        scene_id=scene.scene_id,
        scene_server_id=str(scene_server_id),
        scene_server_git_sha=str(scene_server_git_sha),
        reference_object=reference,
        grasp=GraspAction(
            target_object_id=reference.object_id,
            navigation_approach_xy=grasp_xy,
            robot_stance_xy=grasp_xy,
            robot_yaw_rad=yaw,
            retreat_xy=grasp_xy,
        ),
    )


def resolve_empty_grasp_trial(
    navigation_scene: GroundedScene,
    *,
    scene_server_id: str,
    scene_server_git_sha: str,
    assets_root: Optional[Path | str] = None,
) -> GraspSystemTrial:
    """Resolve the same grasp rig with metric geometry for ``empty.usd``."""
    scene_trial = resolve_grasp_system_trial(
        navigation_scene,
        scene_server_id=scene_server_id,
        scene_server_git_sha=scene_server_git_sha,
        assets_root=assets_root,
    )
    reference = GraspReferenceObject(
        center_xyz=scene_trial.reference_object.center_xyz,
        local_unit_in_meters=1.0,
    )
    return GraspSystemTrial(
        scene_id=EMPTY_GRASP_SCENE_ID,
        scene_server_id=str(scene_server_id),
        scene_server_git_sha=str(scene_server_git_sha),
        reference_object=reference,
        grasp=scene_trial.grasp,
    )


def _reference_layout(
    occupancy: OfficialOccupancyBevMap,
    *,
    robot_clear: np.ndarray,
    reference_clear: np.ndarray,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    """Find visible and graspable stances on one collision-free line."""
    cells = np.argwhere(robot_clear)
    if cells.size == 0:
        raise ValueError("scene has no G1-clear area for grasp-system calibration")
    centre = np.asarray(robot_clear.shape, dtype=float) / 2.0
    ordering = np.argsort(np.sum((cells - centre) ** 2, axis=1))
    directions = tuple(
        (
            math.cos(2.0 * math.pi * index / 16),
            math.sin(2.0 * math.pi * index / 16),
        )
        for index in range(16)
    )
    for cell_index in ordering[: min(len(ordering), 8192)]:
        row, column = (int(value) for value in cells[int(cell_index)])
        observation = occupancy.cell_to_world(row, column)
        for direction in directions:
            start_desired = (
                observation[0] - direction[0] * GRASP_REFERENCE_ROUTE_LENGTH_M,
                observation[1] - direction[1] * GRASP_REFERENCE_ROUTE_LENGTH_M,
            )
            target = (
                observation[0] + direction[0] * GRASP_REFERENCE_OBSERVATION_DISTANCE_M,
                observation[1] + direction[1] * GRASP_REFERENCE_OBSERVATION_DISTANCE_M,
            )
            grasp = (
                target[0] - direction[0] * GRASP_REFERENCE_GRASP_DISTANCE_M,
                target[1] - direction[1] * GRASP_REFERENCE_GRASP_DISTANCE_M,
            )
            start_cell = occupancy.world_to_cell(
                np.asarray([start_desired], dtype=float)
            )[0]
            grasp_cell = occupancy.world_to_cell(np.asarray([grasp], dtype=float))[0]
            target_cell = occupancy.world_to_cell(np.asarray([target], dtype=float))[0]
            start_row, start_column = (
                int(start_cell[0]),
                int(start_cell[1]),
            )
            grasp_row, grasp_column = (
                int(grasp_cell[0]),
                int(grasp_cell[1]),
            )
            target_row, target_column = (
                int(target_cell[0]),
                int(target_cell[1]),
            )
            if not (
                _inside(robot_clear, start_row, start_column)
                and robot_clear[start_row, start_column]
                and _inside(robot_clear, grasp_row, grasp_column)
                and robot_clear[grasp_row, grasp_column]
                and _inside(reference_clear, target_row, target_column)
                and reference_clear[target_row, target_column]
                and _straight_path_is_clear(
                    occupancy,
                    robot_clear,
                    observation,
                    grasp,
                )
            ):
                continue
            start = occupancy.cell_to_world(start_row, start_column)
            grasp = occupancy.cell_to_world(grasp_row, grasp_column)
            if math.dist(start, observation) < 0.10:
                continue
            return (
                (float(observation[0]), float(observation[1])),
                (float(start[0]), float(start[1])),
                (float(grasp[0]), float(grasp[1])),
                (float(target[0]), float(target[1])),
            )
    raise ValueError(
        "scene has no open reference layout satisfying the frozen grasp rig"
    )


def _straight_path_is_clear(
    occupancy: OfficialOccupancyBevMap,
    robot_clear: np.ndarray,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
) -> bool:
    """Require the direct observation-to-grasp motion to remain navigable."""
    sample_count = max(
        2,
        math.ceil(
            math.dist(start_xy, end_xy) * float(occupancy.pixels_per_meter) * 2.0
        ),
    )
    samples = np.linspace(
        np.asarray(start_xy, dtype=float),
        np.asarray(end_xy, dtype=float),
        sample_count,
    )
    for row, column in occupancy.world_to_cell(samples):
        row = int(row)
        column = int(column)
        if not _inside(robot_clear, row, column) or not robot_clear[row, column]:
            return False
    return True


def _inside(mask: np.ndarray, row: int, column: int) -> bool:
    return 0 <= row < mask.shape[0] and 0 <= column < mask.shape[1]


__all__ = [
    "EMPTY_GRASP_SCENE_ID",
    "GRASP_REFERENCE_OBJECT_ID",
    "GRASP_REFERENCE_PRIM_PATH",
    "GraspReferenceObject",
    "GraspSystemTrial",
    "resolve_empty_grasp_trial",
    "resolve_grasp_system_trial",
]
