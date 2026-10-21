"""Compile one resolved physical trial into a Human-free runtime episode."""

from __future__ import annotations

import re
from typing import Tuple

from pydantic import Field

from tarsphere.bench.certification.physical_trial import PhysicalTrial
from tarsphere.bench.compilation.episode import (
    agent_action_space,
    yaw_quaternion,
)
from tarsphere.bench.compilation.objects import (
    compile_episode_objects,
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
_PHYSICAL_TRIAL_MAX_DECISION_STEPS = 160


class CompiledPhysicalTrialWorld(StrictModel):
    """Benchmark-private facts for one Human-free physical certification run."""

    scene_id: str = Field(
        min_length=1,
        description="Grounded scene in which the physical trial executes.",
    )
    scene_sources: SceneSources = Field(
        description="Canonical scene files from which physical facts originate.",
    )
    scene_map: SceneMap = Field(
        description="Canonical occupancy and visual-map references.",
    )
    physical_trial: PhysicalTrial = Field(
        description=(
            "Explicit geometry choice, physical route, and static navigation "
            "evidence under rollout certification."
        ),
    )
    objects: Tuple[EpisodeObject, ...] = Field(
        min_length=1,
        description=(
            "Target, initial support, and destination objects required by the "
            "physical trial."
        ),
    )


class PhysicalTrialCompilationError(ValueError):
    """Raised when a resolved trial cannot form a valid runtime episode."""


def compile_physical_trial_episode(
    trial_id: str,
    *,
    trial: PhysicalTrial,
    scene: GroundedScene,
) -> TARSphereEpisode:
    """Compile a Human-free rollout without inventing semantic milestones."""
    if not _TRIAL_ID_PATTERN.fullmatch(trial_id):
        raise ValueError(
            "trial_id must be lowercase snake_case beginning with a letter; "
            f"got {trial_id!r}"
        )
    if trial.scene_id != scene.scene_id:
        raise ValueError(
            "physical trial and grounded scene must identify the same scene"
        )

    objects_by_id = {item.object_id: item for item in scene.objects}
    try:
        objects_by_id[trial.grasp.target_object_id]
        objects_by_id[trial.place.destination_object_id]
    except KeyError as exc:
        raise PhysicalTrialCompilationError(
            f"Physical trial references an unknown object: {exc.args[0]!r}"
        ) from exc
    try:
        objects = compile_episode_objects(
            scene,
            (
                trial.grasp.target_object_id,
                trial.source_support_object_id,
                trial.place.destination_object_id,
            ),
        )
    except ValueError as exc:
        raise PhysicalTrialCompilationError(str(exc)) from exc

    world = CompiledPhysicalTrialWorld(
        scene_id=scene.scene_id,
        scene_sources=scene.sources,
        scene_map=scene.map,
        physical_trial=trial,
        objects=objects,
    )
    robot_start = trial.navigate_to_target.points[0].position_xy
    episode = TARSphereEpisode(
        episode_id=trial_id,
        title=f"Physical Trial — {trial_id.replace('_', ' ').title()}",
        scene=SceneSpec(
            asset_path=scene.sources.scene_usd,
            scale=scene.scene_scale,
        ),
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
            max_decision_steps=_PHYSICAL_TRIAL_MAX_DECISION_STEPS,
        ),
        world_knowledge=world.model_dump(
            mode="python",
            exclude_none=True,
        ),
    )
    HumanActorRuntime(episode)
    return episode


__all__ = [
    "CompiledPhysicalTrialWorld",
    "PhysicalTrialCompilationError",
    "compile_physical_trial_episode",
]
