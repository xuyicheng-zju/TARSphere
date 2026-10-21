"""Compile the real-asset G1 grasp-motion calibration episode."""

from __future__ import annotations

import re
from typing import Tuple

from pydantic import Field

from tarsphere.bench.certification.grasp_motion import (
    GRASP_MOTION_TABLE_OBJECT_ID,
    GRASP_MOTION_TABLE_PRIM,
    GRASP_MOTION_TARGET_OBJECT_ID,
    GRASP_MOTION_TARGET_PRIM,
    GraspMotionTrial,
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
    GroundedObject,
    GroundedScene,
    SceneMap,
)
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


_TRIAL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_GRASP_MOTION_MAX_DECISION_STEPS = 80
_TABLE_GRASP_SKILL_MAX_DECISION_STEPS = 12
_TABLE_GRASP_COMPONENT_MAX_DECISION_STEPS = 32


class CompiledGraspMotionWorld(StrictModel):
    """Private runtime facts for the real-asset grasp-motion calibration."""

    scene_id: str = Field(
        min_length=1,
        description="Simple table scene loaded by the calibration.",
    )
    scene_map: SceneMap = Field(
        description="Reviewed occupancy map used to certify both routes.",
    )
    grasp_motion_trial: GraspMotionTrial = Field(
        description="Exact navigation, grasp, retraction, and return program.",
    )
    objects: Tuple[EpisodeObject, ...] = Field(
        min_length=2,
        max_length=2,
        description="Real bottle target and the simple supporting table.",
    )


def compile_grasp_motion_episode(
    trial_id: str,
    *,
    trial: GraspMotionTrial,
    navigation_scene: GroundedScene,
) -> TARSphereEpisode:
    """Compile one Human-free real-bottle Stage-B calibration."""
    if not _TRIAL_ID_PATTERN.fullmatch(trial_id):
        raise ValueError(
            "trial_id must be lowercase snake_case beginning with a letter; "
            f"got {trial_id!r}"
        )
    world = _compile_world(trial, navigation_scene)
    robot_start = trial.robot_start_xy
    episode = TARSphereEpisode(
        episode_id=trial_id,
        title="G1 Real-Bottle Grasp Motion",
        scene=SceneSpec(
            asset_path=trial.scene_asset_path,
            scale=(1.0, 1.0, 1.0),
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
            start_orientation=yaw_quaternion(
                trial.robot_start_yaw_rad,
            ),
        ),
        humans=[],
        agent_action_space=agent_action_space(),
        runtime=RuntimeSpec(
            max_decision_steps=_GRASP_MOTION_MAX_DECISION_STEPS,
        ),
        diagnostics=DiagnosticSpec(
            description=(
                "The G1 returns from the table and reports completion; the "
                "Stage-B certificate independently proves physical grasp and "
                "unconstrained friction-only transport."
            ),
            checks=[
                DiagnosticCheck(
                    type="robot_near_position",
                    params={
                        "position": [
                            float(trial.return_from_target.points[-1].position_xy[0]),
                            float(trial.return_from_target.points[-1].position_xy[1]),
                            0.8,
                        ],
                        "distance": 0.35,
                    },
                ),
                DiagnosticCheck(type="agent_reported_done"),
            ],
        ),
        world_knowledge=world.model_dump(mode="python", exclude_none=True),
    )
    HumanActorRuntime(episode)
    return episode


def compile_table_grasp_component_episode(
    trial_id: str,
    *,
    trial: GraspMotionTrial,
    navigation_scene: GroundedScene,
) -> TARSphereEpisode:
    """Compile the Human-free approach, one-call grasp, and transport component.

    The Robot starts at the certified route origin rather than at the grasp
    stance. The evaluated Agent must use ego evidence to approach the table,
    ground the bottle from its current view, and invoke the shared physical
    grasp executor. No world coordinate or authored stance is public.
    """
    if not _TRIAL_ID_PATTERN.fullmatch(trial_id):
        raise ValueError(
            "trial_id must be lowercase snake_case beginning with a letter; "
            f"got {trial_id!r}"
        )
    world = _compile_world(trial, navigation_scene)
    robot_start = trial.robot_start_xy
    target = world.objects[0]
    episode = TARSphereEpisode(
        episode_id=trial_id,
        title="G1 Grasp-Skill and Transport Component Validation",
        description=(
            "Component diagnostic: model-controlled G1 approaches a visible "
            "table, picks up a real bottle in one grasp skill, then carries it "
            "away with ordinary locomotion. Humans and NavMesh are absent."
        ),
        system_prompt=(
            "Pick up the small dark bottle from the simple wooden table, then "
            "carry it back into the open starting area. Approach until the "
            "object is plausibly within arm reach. Once the whole target is "
            "visible and near, call grasp_object rather than trying to infer "
            "the remaining centimetres from RGB; the skill uses synchronized "
            "depth to enforce metric reachability. Use camera tilt only if "
            "the target is not fully visible, then call grasp_object. That "
            "skill performs current SAM3/depth/GraspGenX grounding and first "
            "plans pelvis, waist, and right-arm motion only to one closing "
            "pose. After measured contact and retention, a separate plan "
            "lifts the object and restores the walking posture with closed "
            "fingers. After feedback confirms "
            "transport_ready, use ordinary body motion to carry the bottle "
            "away from the table. Report completion only after the return."
        ),
        scene=SceneSpec(
            asset_path=trial.scene_asset_path,
            scale=(1.0, 1.0, 1.0),
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
        agent_action_space=ActionSpec(
            cognitive=["wait", "tilt_camera"],
            communicative=["report_done"],
            physical=["move_body", "grasp_object"],
        ),
        runtime=RuntimeSpec(
            max_decision_steps=_TABLE_GRASP_COMPONENT_MAX_DECISION_STEPS,
        ),
        diagnostics=DiagnosticSpec(
            description=(
                "The real bottle remains physically held without a constraint "
                "while ordinary locomotion carries it back from the table."
            ),
            checks=[
                DiagnosticCheck(
                    type="robot_holding_object",
                    params={"object_id": target.object_id},
                ),
                DiagnosticCheck(
                    type="robot_near_position",
                    params={
                        "position": [
                            float(trial.return_from_target.points[-1].position_xy[0]),
                            float(trial.return_from_target.points[-1].position_xy[1]),
                            0.8,
                        ],
                        "distance": 0.35,
                    },
                ),
                DiagnosticCheck(type="agent_reported_done"),
            ],
        ),
        world_knowledge=world.model_dump(mode="python", exclude_none=True),
        metadata={"validation_scope": "grasp_skill_transport_component"},
    )
    HumanActorRuntime(episode)
    return episode


def compile_table_grasp_transport_episode(
    trial_id: str,
    *,
    trial: GraspMotionTrial,
    navigation_scene: GroundedScene,
) -> TARSphereEpisode:
    """Compile contact grasp followed by ordinary friction-only transport.

    The Robot starts at the same independently resolved stance as the grasp
    skill gate. Removing the approach leg isolates whether a live Dex3 contact
    survives arm retraction, standing, and ordinary locomotion without an
    object-to-hand constraint.
    """
    if not _TRIAL_ID_PATTERN.fullmatch(trial_id):
        raise ValueError(
            "trial_id must be lowercase snake_case beginning with a letter; "
            f"got {trial_id!r}"
        )
    world = _compile_world(trial, navigation_scene)
    target = world.objects[0]
    stance = trial.grasp.robot_stance_xy
    transport_goal = trial.grasp.retreat_xy
    episode = TARSphereEpisode(
        episode_id=trial_id,
        title="G1 Friction-Only Grasp Transport Validation",
        description=(
            "The model-controlled G1 grasps a real bottle with native Dex3 "
            "contact, restores its walking posture, and carries the bottle "
            "away using the ordinary locomotion policy."
        ),
        system_prompt=(
            "Pick up the small dark bottle on the wooden table and carry it "
            "backward into the open area. You already face the bottle from a "
            "plausible arm-reach stance. If the whole bottle is visible, call "
            "grasp_object; synchronized depth "
            "will verify its exact reach. After feedback confirms "
            "transport_ready, use ordinary backward body motion to carry it "
            "clear of the table. Do not release it. Report completion only "
            "after you have transported it away."
        ),
        scene=SceneSpec(
            asset_path=trial.scene_asset_path,
            scale=(1.0, 1.0, 1.0),
        ),
        robot=RobotSpec(
            id=0,
            name="robot",
            model="G1Robot",
            start_position=(float(stance[0]), float(stance[1]), 0.8),
            start_orientation=yaw_quaternion(trial.grasp.robot_yaw_rad),
        ),
        humans=[],
        agent_action_space=ActionSpec(
            cognitive=["wait", "tilt_camera"],
            communicative=["report_done"],
            physical=["move_body", "grasp_object"],
        ),
        runtime=RuntimeSpec(
            max_decision_steps=None,
            max_physics_steps=None,
        ),
        diagnostics=DiagnosticSpec(
            description=(
                "The Robot carries the dynamically simulated bottle away "
                "while live hand-object contact remains and no attachment "
                "constraint exists."
            ),
            checks=[
                DiagnosticCheck(
                    type="robot_holding_object",
                    params={"object_id": target.object_id},
                ),
                DiagnosticCheck(
                    type="robot_near_position",
                    params={
                        "position": [
                            float(transport_goal[0]),
                            float(transport_goal[1]),
                            0.8,
                        ],
                        "distance": 0.20,
                    },
                ),
                DiagnosticCheck(type="agent_reported_done"),
            ],
        ),
        world_knowledge=world.model_dump(mode="python", exclude_none=True),
        metadata={"validation_scope": "friction_only_grasp_transport"},
    )
    HumanActorRuntime(episode)
    return episode


def compile_table_grasp_skill_episode(
    trial_id: str,
    *,
    trial: GraspMotionTrial,
    navigation_scene: GroundedScene,
) -> TARSphereEpisode:
    """Compile the current-view grasp skill at a certified body stance.

    This is the manipulation Stage-A gate.  It deliberately removes base
    navigation so a failure can only come from current-view grounding, grasp
    generation, arm reach, hand closure, or physical retention.  The target
    pose remains private; the evaluated Agent still has to name what it sees
    and invoke the same public ``grasp_object`` tool used in full episodes.
    """
    if not _TRIAL_ID_PATTERN.fullmatch(trial_id):
        raise ValueError(
            "trial_id must be lowercase snake_case beginning with a letter; "
            f"got {trial_id!r}"
        )
    world = _compile_world(trial, navigation_scene)
    target = world.objects[0]
    stance = trial.grasp.robot_stance_xy
    episode = TARSphereEpisode(
        episode_id=trial_id,
        title="G1 Current-View Grasp Skill Validation",
        description=(
            "Stage-A diagnostic for the complete current-view grasp skill. "
            "The Robot begins at an independently certified body stance; no "
            "base-navigation decision is available."
        ),
        system_prompt=(
            "Pick up the bottle on the wooden table. You "
            "are already standing at a certified arm-reach stance. Use the "
            "calibrated observation pose: first call tilt_camera with "
            "pitch_deg=-30 exactly once, then call grasp_object with "
            'target="small dark bottle" without changing the camera or target phrase. '
            "The grasp skill performs current SAM3/depth/GraspGenX grounding "
            "and first plans only the motion to one right-hand closing pose. "
            "After measured contact and retention, a separate plan lifts the "
            "object and restores the walking posture with closed fingers. Report "
            "completion only "
            "after feedback confirms transport_ready."
        ),
        scene=SceneSpec(
            asset_path=trial.scene_asset_path,
            scale=(1.0, 1.0, 1.0),
        ),
        robot=RobotSpec(
            id=0,
            name="robot",
            model="G1Robot",
            start_position=(float(stance[0]), float(stance[1]), 0.8),
            start_orientation=yaw_quaternion(trial.grasp.robot_yaw_rad),
        ),
        humans=[],
        agent_action_space=ActionSpec(
            cognitive=["wait", "tilt_camera"],
            communicative=["report_done"],
            physical=["grasp_object"],
        ),
        runtime=RuntimeSpec(
            max_decision_steps=_TABLE_GRASP_SKILL_MAX_DECISION_STEPS,
        ),
        diagnostics=DiagnosticSpec(
            description=(
                "The current-view grasp skill physically retains the real "
                "bottle and returns the Robot to a transport-ready posture."
            ),
            checks=[
                DiagnosticCheck(
                    type="robot_holding_object",
                    params={"object_id": target.object_id},
                ),
                DiagnosticCheck(type="agent_reported_done"),
            ],
        ),
        world_knowledge=world.model_dump(mode="python", exclude_none=True),
        metadata={"validation_scope": "current_view_grasp_skill"},
    )
    HumanActorRuntime(episode)
    return episode


def _compile_world(
    trial: GraspMotionTrial,
    navigation_scene: GroundedScene,
) -> CompiledGraspMotionWorld:
    target = _target_object(trial)
    table = EpisodeObject(
        object_id=GRASP_MOTION_TABLE_OBJECT_ID,
        name="simple wooden table",
        category="table",
        room_id="grasp_calibration",
        position_xyz=trial.source_support_position_xyz,
        bounds_min_xyz=trial.source_support_bounds_min_xyz,
        bounds_max_xyz=trial.source_support_bounds_max_xyz,
        runtime_prim=GRASP_MOTION_TABLE_PRIM,
    )
    return CompiledGraspMotionWorld(
        scene_id=trial.scene_id,
        scene_map=navigation_scene.map,
        grasp_motion_trial=trial,
        objects=(_episode_object(target), table),
    )


def _target_object(trial: GraspMotionTrial) -> GroundedObject:
    return GroundedObject(
        object_id=GRASP_MOTION_TARGET_OBJECT_ID,
        category="bottle",
        room_id="grasp_calibration",
        position_xyz=trial.target_position_xyz,
        bounds_min_xyz=trial.target_bounds_min_xyz,
        bounds_max_xyz=trial.target_bounds_max_xyz,
        caption="small dark cylindrical bottle with a narrow neck",
        source_prim=None,
        runtime_prim=GRASP_MOTION_TARGET_PRIM,
        interaction="pickable",
    )


def _episode_object(target: GroundedObject) -> EpisodeObject:
    return EpisodeObject(
        object_id=target.object_id,
        name="small dark bottle",
        category=target.category,
        room_id=target.room_id,
        position_xyz=target.position_xyz,
        bounds_min_xyz=target.bounds_min_xyz,
        bounds_max_xyz=target.bounds_max_xyz,
        runtime_prim=str(target.runtime_prim),
        interaction=target.interaction,
    )


__all__ = [
    "CompiledGraspMotionWorld",
    "compile_grasp_motion_episode",
    "compile_table_grasp_component_episode",
    "compile_table_grasp_skill_episode",
    "compile_table_grasp_transport_episode",
]
