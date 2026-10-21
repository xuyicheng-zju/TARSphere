"""Compile repeatable empty-table certificates for real graspable assets."""

from __future__ import annotations

import math
import re
from typing import Tuple

from pydantic import Field, model_validator

from tarsphere.bench.certification.grasp_motion import (
    GRASP_MOTION_TABLE_OBJECT_ID,
    GRASP_MOTION_TABLE_PRIM,
    GRASP_MOTION_TARGET_PRIM,
    GraspMotionTrial,
)
from tarsphere.bench.compilation.episode import (
    yaw_quaternion,
)
from tarsphere.bench.compilation.schema import EpisodeObject, StrictModel
from tarsphere.bench.grounding.schema import GroundedScene, SceneMap
from tarsphere.bench.schema import (
    ActionSpec,
    TARSphereEpisode,
    RobotSpec,
    RuntimeSpec,
    SceneSpec,
    DiagnosticCheck,
    DiagnosticSpec,
)
from tarsphere.infra.episode.human_actor_runtime import HumanActorRuntime


Point3 = Tuple[float, float, float]
_TRIAL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_MAX_DECISION_STEPS = 12
_MINIMUM_TABLE_EDGE_INSET_M = 0.12


class GraspAssetCandidate(StrictModel):
    """One asset admitted by the deterministic geometry screen."""

    asset_key: str = Field(
        min_length=1,
        description="Stable category/asset-id key in the GRScenes object library.",
    )
    category: str = Field(
        min_length=1,
        description="Ordinary object category used to name the visible target.",
    )
    asset_path: str = Field(
        min_length=1,
        description="Repository asset-relative USD file loaded for this trial.",
    )
    asset_meters_per_unit: float = Field(
        gt=0.0,
        description="Metres represented by one coordinate unit in the asset USD.",
    )
    local_bounds_min_xyz_m: Point3 = Field(
        description="Minimum asset-local AABB corner after conversion to metres.",
    )
    local_bounds_max_xyz_m: Point3 = Field(
        description="Maximum asset-local AABB corner after conversion to metres.",
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> "GraspAssetCandidate":
        values = (*self.local_bounds_min_xyz_m, *self.local_bounds_max_xyz_m)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("grasp asset bounds must be finite")
        if any(
            self.local_bounds_min_xyz_m[axis] >= self.local_bounds_max_xyz_m[axis]
            for axis in range(3)
        ):
            raise ValueError("grasp asset bounds must have positive extent")
        return self

    @property
    def size_xyz_m(self) -> Point3:
        """Return the measured asset dimensions in metres."""

        return tuple(
            self.local_bounds_max_xyz_m[axis] - self.local_bounds_min_xyz_m[axis]
            for axis in range(3)
        )  # type: ignore[return-value]


class GraspAssetCertificationTrial(StrictModel):
    """Exact placement of one asset for a live right-hand grasp certificate."""

    candidate: GraspAssetCandidate = Field(
        description="Measured real asset exercised by this certificate.",
    )
    target_object_id: str = Field(
        min_length=1,
        description="Episode-local target identity used by runtime evidence.",
    )
    runtime_prim: str = Field(
        pattern=r"^/World/.+",
        description="Canonical live prim replaced by the selected asset.",
    )
    target_translation_xyz_m: Point3 = Field(
        description="World translation placing the asset centre over the table.",
    )
    target_bounds_min_xyz_m: Point3 = Field(
        description="Expected live world AABB minimum after placement.",
    )
    target_bounds_max_xyz_m: Point3 = Field(
        description="Expected live world AABB maximum after placement.",
    )

    @model_validator(mode="after")
    def validate_world_geometry(self) -> "GraspAssetCertificationTrial":
        expected_min = tuple(
            self.candidate.local_bounds_min_xyz_m[axis]
            + self.target_translation_xyz_m[axis]
            for axis in range(3)
        )
        expected_max = tuple(
            self.candidate.local_bounds_max_xyz_m[axis]
            + self.target_translation_xyz_m[axis]
            for axis in range(3)
        )
        if any(
            not math.isclose(
                self.target_bounds_min_xyz_m[axis],
                expected_min[axis],
                abs_tol=1e-9,
            )
            or not math.isclose(
                self.target_bounds_max_xyz_m[axis],
                expected_max[axis],
                abs_tol=1e-9,
            )
            for axis in range(3)
        ):
            raise ValueError("grasp asset world bounds disagree with its placement")
        return self


class CompiledGraspAssetWorld(StrictModel):
    """Private facts for one real-asset grasp certification episode."""

    scene_id: str = Field(
        min_length=1,
        description="Shared simple-table scene used for every asset trial.",
    )
    scene_map: SceneMap = Field(
        description="Reviewed map retained by the standard episode contract.",
    )
    grasp_motion_trial: GraspMotionTrial = Field(
        description="Target, support, and fixed Robot stance for attachment setup.",
    )
    grasp_asset_certification: GraspAssetCertificationTrial = Field(
        description="Exact asset replacement and hand exercised in this run.",
    )
    objects: Tuple[EpisodeObject, ...] = Field(
        min_length=2,
        max_length=2,
        description="Selected pickable followed by the unchanged support table.",
    )


def place_candidate_on_table(
    candidate: GraspAssetCandidate,
    *,
    base_trial: GraspMotionTrial,
) -> GraspAssetCertificationTrial:
    """Centre one measured asset over the table with its lowest point supported."""

    support_min = base_trial.source_support_bounds_min_xyz
    support_max = base_trial.source_support_bounds_max_xyz
    table_top = base_trial.source_support_bounds_max_xyz[2]
    local_min = candidate.local_bounds_min_xyz_m
    local_max = candidate.local_bounds_max_xyz_m
    local_center_xy = (
        (local_min[0] + local_max[0]) / 2.0,
        (local_min[1] + local_max[1]) / 2.0,
    )
    half_size_x = (local_max[0] - local_min[0]) / 2.0
    target_x = max(
        base_trial.target_position_xyz[0],
        support_min[0] + _MINIMUM_TABLE_EDGE_INSET_M + half_size_x,
    )
    target_y = (support_min[1] + support_max[1]) / 2.0
    translation = (
        target_x - local_center_xy[0],
        target_y - local_center_xy[1],
        table_top - local_min[2],
    )
    world_min = tuple(local_min[axis] + translation[axis] for axis in range(3))
    world_max = tuple(local_max[axis] + translation[axis] for axis in range(3))
    object_id = f"grasp_asset/{candidate.asset_key}"
    return GraspAssetCertificationTrial(
        candidate=candidate,
        target_object_id=object_id,
        runtime_prim=GRASP_MOTION_TARGET_PRIM,
        target_translation_xyz_m=translation,
        target_bounds_min_xyz_m=world_min,
        target_bounds_max_xyz_m=world_max,
    )


def compile_grasp_asset_episode(
    episode_id: str,
    *,
    certification: GraspAssetCertificationTrial,
    base_trial: GraspMotionTrial,
    navigation_scene: GroundedScene,
) -> TARSphereEpisode:
    """Compile one no-VLM grasp-and-loaded-carry certificate on the table."""

    if not _TRIAL_ID_PATTERN.fullmatch(episode_id):
        raise ValueError("grasp asset episode_id must be lowercase snake_case")
    target_center = tuple(
        (
            certification.target_bounds_min_xyz_m[axis]
            + certification.target_bounds_max_xyz_m[axis]
        )
        / 2.0
        for axis in range(3)
    )
    target_shift_xy = (
        target_center[0] - base_trial.target_position_xyz[0],
        target_center[1] - base_trial.target_position_xyz[1],
    )
    grasp = base_trial.grasp.model_copy(
        update={
            "target_object_id": certification.target_object_id,
            "robot_stance_xy": (
                base_trial.grasp.robot_stance_xy[0] + target_shift_xy[0],
                base_trial.grasp.robot_stance_xy[1] + target_shift_xy[1],
            ),
        }
    )
    motion_trial = GraspMotionTrial.model_validate(
        {
            **base_trial.model_dump(mode="python"),
            "target_object_id": certification.target_object_id,
            "target_position_xyz": target_center,
            "target_bounds_min_xyz": certification.target_bounds_min_xyz_m,
            "target_bounds_max_xyz": certification.target_bounds_max_xyz_m,
            "grasp": grasp.model_dump(mode="python"),
        }
    )
    target = EpisodeObject(
        object_id=certification.target_object_id,
        name=certification.candidate.category,
        category=certification.candidate.category,
        room_id="grasp_calibration",
        position_xyz=target_center,
        bounds_min_xyz=certification.target_bounds_min_xyz_m,
        bounds_max_xyz=certification.target_bounds_max_xyz_m,
        runtime_prim=certification.runtime_prim,
        interaction="pickable",
    )
    table = EpisodeObject(
        object_id=GRASP_MOTION_TABLE_OBJECT_ID,
        name="simple wooden table",
        category="table",
        room_id="grasp_calibration",
        position_xyz=base_trial.source_support_position_xyz,
        bounds_min_xyz=base_trial.source_support_bounds_min_xyz,
        bounds_max_xyz=base_trial.source_support_bounds_max_xyz,
        runtime_prim=GRASP_MOTION_TABLE_PRIM,
    )
    world = CompiledGraspAssetWorld(
        scene_id=base_trial.scene_id,
        scene_map=navigation_scene.map,
        grasp_motion_trial=motion_trial,
        grasp_asset_certification=certification,
        objects=(target, table),
    )
    stance = base_trial.grasp.robot_stance_xy
    episode = TARSphereEpisode(
        episode_id=episode_id,
        title=(
            "G1 Right-Hand "
            f"{certification.candidate.category.title()} Grasp Certificate"
        ),
        description=(
            "A deterministic current-view certificate for real contact, Dex3 "
            "closure, cuRobo lift, transport-ready retention, and measured "
            "loaded-body motion."
        ),
        system_prompt=(
            f"Pick up the {certification.candidate.category} on the table with "
            "the right hand."
        ),
        scene=SceneSpec(
            asset_path=base_trial.scene_asset_path,
            scale=(1.0, 1.0, 1.0),
        ),
        robot=RobotSpec(
            id=0,
            name="robot",
            model="G1Robot",
            start_position=(float(stance[0]), float(stance[1]), 0.8),
            start_orientation=yaw_quaternion(base_trial.grasp.robot_yaw_rad),
        ),
        humans=[],
        agent_action_space=ActionSpec(
            cognitive=["tilt_camera"],
            communicative=["report_done"],
            physical=["grasp_object", "move_body"],
        ),
        runtime=RuntimeSpec(
            max_decision_steps=_MAX_DECISION_STEPS,
        ),
        diagnostics=DiagnosticSpec(
            description=(
                "The right hand physically retains the asset without a "
                "constraint through the complete backward-turn-forward "
                "loaded-body sequence."
            ),
            checks=[
                DiagnosticCheck(
                    type="robot_holding_object",
                    params={"object_id": certification.target_object_id},
                ),
                DiagnosticCheck(type="agent_reported_done"),
            ],
        ),
        world_knowledge=world.model_dump(mode="python", exclude_none=True),
        metadata={
            "validation_scope": "grasp_asset_certification",
            "asset_key": certification.candidate.asset_key,
        },
    )
    HumanActorRuntime(episode)
    return episode


__all__ = [
    "CompiledGraspAssetWorld",
    "GraspAssetCandidate",
    "GraspAssetCertificationTrial",
    "compile_grasp_asset_episode",
    "place_candidate_on_table",
]
