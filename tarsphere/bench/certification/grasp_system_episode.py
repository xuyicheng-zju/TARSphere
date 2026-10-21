"""Compile the known-positive grasp-system trial into one Isaac episode."""

from __future__ import annotations

import re
from typing import Any, Tuple

from pydantic import Field

from tarsphere.bench.certification.grasp_system import (
    EMPTY_GRASP_SCENE_ID,
    GRASP_REFERENCE_OBJECT_ID,
    GraspSystemTrial,
)
from tarsphere.bench.compilation.episode import (
    agent_action_space,
    yaw_quaternion,
)
from tarsphere.bench.compilation.schema import (
    EpisodeObject,
    StrictModel,
)
from tarsphere.bench.grounding.schema import (
    GroundedScene,
    SceneMap,
    SceneSources,
)
from tarsphere.bench.schema import (
    TARSphereEpisode,
    RobotSpec,
    RuntimeSpec,
    SceneSpec,
)
from tarsphere.infra.episode.human_actor_runtime import HumanActorRuntime


_TRIAL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_GRASP_SYSTEM_MAX_DECISION_STEPS = 60
EMPTY_GRASP_SCENE_ASSET = "scenes/empty.usd"


class CompiledGraspSystemWorld(StrictModel):
    """Private runtime facts for one known-positive grasp-stack trial."""

    scene_id: str = Field(
        min_length=1,
        description="Real GRScene containing the calibration rig.",
    )
    scene_sources: SceneSources = Field(
        description="Canonical files for the loaded GRScene.",
    )
    scene_map: SceneMap = Field(
        description="Canonical occupancy and visual-map references.",
    )
    grasp_system_trial: GraspSystemTrial = Field(
        description="Frozen reference geometry and target-facing Robot stance.",
    )
    objects: Tuple[EpisodeObject, ...] = Field(
        min_length=1,
        max_length=1,
        description="Known reference object exposed to runtime and evaluation.",
    )


class CompiledEmptyGraspWorld(StrictModel):
    """Private runtime facts for the empty-scene G1 grasp demonstration."""

    scene_id: str = Field(
        min_length=1,
        description="Stable identifier for the empty calibration scene.",
    )
    scene_map: SceneMap = Field(
        description="Open calibration area selected from a real occupancy map.",
    )
    occupancy_source_scene_id: str = Field(
        min_length=1,
        description="Grounded scene whose open occupancy cell placed the rig.",
    )
    grasp_system_trial: GraspSystemTrial = Field(
        description="Reference geometry and fixed target-facing Robot stance.",
    )
    objects: Tuple[EpisodeObject, ...] = Field(
        min_length=1,
        max_length=1,
        description="Single reference object present in the empty scene.",
    )


def compile_grasp_system_episode(
    trial_id: str,
    *,
    trial: GraspSystemTrial,
    scene: GroundedScene,
) -> TARSphereEpisode:
    """Compile the reference trial through the normal Golden grasp path."""
    if not _TRIAL_ID_PATTERN.fullmatch(trial_id):
        raise ValueError(
            "trial_id must be lowercase snake_case beginning with a letter; "
            f"got {trial_id!r}"
        )
    if trial.scene_id != scene.scene_id:
        raise ValueError(
            "grasp-system trial and grounded scene must identify one scene"
        )
    target = _reference_object(trial)
    world = _world_with_physical_task(
        CompiledGraspSystemWorld(
            scene_id=scene.scene_id,
            scene_sources=scene.sources,
            scene_map=scene.map,
            grasp_system_trial=trial,
            objects=(target,),
        ).model_dump(mode="python", exclude_none=True),
        trial=trial,
    )
    return _compile_episode(
        trial_id,
        title="G1 Grasp System Calibration",
        trial=trial,
        scene=SceneSpec(
            asset_path=scene.sources.scene_usd,
            scale=scene.scene_scale,
        ),
        world=world,
    )


def compile_empty_grasp_episode(
    trial_id: str,
    *,
    trial: GraspSystemTrial,
    navigation_scene: GroundedScene,
) -> TARSphereEpisode:
    """Place G1 and one reference cube in the repository's empty scene."""
    if trial.scene_id != EMPTY_GRASP_SCENE_ID:
        raise ValueError(
            "empty-scene grasp trial must use the canonical scene identifier"
        )
    if not (abs(trial.reference_object.local_unit_in_meters - 1.0) <= 1e-12):
        raise ValueError(
            "empty-scene reference geometry must be authored directly in metres"
        )
    target = _reference_object(trial)
    world = _world_with_physical_task(
        CompiledEmptyGraspWorld(
            scene_id=EMPTY_GRASP_SCENE_ID,
            scene_map=navigation_scene.map,
            occupancy_source_scene_id=navigation_scene.scene_id,
            grasp_system_trial=trial,
            objects=(target,),
        ).model_dump(mode="python", exclude_none=True),
        trial=trial,
    )
    return _compile_episode(
        trial_id,
        title="G1 Empty-Scene Grasp Motion",
        trial=trial,
        scene=SceneSpec(
            asset_path=EMPTY_GRASP_SCENE_ASSET,
            scale=(1.0, 1.0, 1.0),
        ),
        world=world,
    )


def _physical_task(trial: GraspSystemTrial) -> dict[str, Any]:
    """Bind the reference pickable to the runtime manipulation contract."""

    reference = trial.reference_object
    if trial.grasp.target_object_id != reference.object_id:
        raise ValueError(
            "grasp-system trial grasp.target_object_id must match "
            "reference_object.object_id"
        )
    return {
        "target_object_id": reference.object_id,
        "grasp": trial.grasp.model_dump(mode="python"),
    }


def _world_with_physical_task(
    world: dict[str, Any],
    *,
    trial: GraspSystemTrial,
) -> dict[str, Any]:
    """Attach the shared physical-task block expected by G1 attachment setup."""

    updated = dict(world)
    updated["physical_task"] = _physical_task(trial)
    return updated


def _reference_object(trial: GraspSystemTrial) -> EpisodeObject:
    reference = trial.reference_object
    return EpisodeObject(
        object_id=reference.object_id,
        name="grasp reference",
        category="grasp_reference",
        room_id="grasp_system_calibration",
        position_xyz=reference.center_xyz,
        bounds_min_xyz=reference.bounds_min_xyz,
        bounds_max_xyz=reference.bounds_max_xyz,
        runtime_prim=reference.prim_path,
        interaction="pickable",
    )


def _compile_episode(
    trial_id: str,
    *,
    title: str,
    trial: GraspSystemTrial,
    scene: SceneSpec,
    world: dict,
) -> TARSphereEpisode:
    if not _TRIAL_ID_PATTERN.fullmatch(trial_id):
        raise ValueError(
            "trial_id must be lowercase snake_case beginning with a letter; "
            f"got {trial_id!r}"
        )
    robot_start = trial.robot_start_xy
    episode = TARSphereEpisode(
        episode_id=trial_id,
        title=title,
        scene=scene,
        robot=RobotSpec(
            id=0,
            name="robot",
            model="G1Robot",
            start_position=(
                float(robot_start[0]),
                float(robot_start[1]),
                0.8,
            ),
            start_orientation=yaw_quaternion(trial.robot_start_yaw_rad),
        ),
        humans=[],
        agent_action_space=agent_action_space(),
        runtime=RuntimeSpec(
            max_decision_steps=_GRASP_SYSTEM_MAX_DECISION_STEPS,
        ),
        world_knowledge=world,
    )
    HumanActorRuntime(episode)
    _assert_manipulation_contract(episode)
    return episode


def _assert_manipulation_contract(episode: TARSphereEpisode) -> None:
    """Fail at compile time when the episode cannot arm G1 attachment."""

    from tarsphere.infra.manipulation.g1_attachment import (
        manipulation_objects_from_episode,
    )

    objects = manipulation_objects_from_episode(episode)
    if not objects:
        raise ValueError(
            "compiled grasp-system episode has no physical-task manipulation "
            "targets; world_knowledge.physical_task.target_object_id is "
            "required."
        )
    pickables = [item for item in objects if item.pickable]
    if len(pickables) != 1:
        raise ValueError(
            "compiled grasp-system episode must expose exactly one pickable "
            f"manipulation target; found {len(pickables)}."
        )
    if pickables[0].object_id != GRASP_REFERENCE_OBJECT_ID:
        raise ValueError(
            "compiled grasp-system episode must name the reference cube as "
            f"the pickable target; found {pickables[0].object_id!r}."
        )


__all__ = [
    "CompiledEmptyGraspWorld",
    "CompiledGraspSystemWorld",
    "EMPTY_GRASP_SCENE_ASSET",
    "EMPTY_GRASP_SCENE_ID",
    "compile_empty_grasp_episode",
    "compile_grasp_system_episode",
]
