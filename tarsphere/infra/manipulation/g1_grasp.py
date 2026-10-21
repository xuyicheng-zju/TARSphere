"""GraspGenX-to-cuRoboV2 motion planning for G1 with native Dex3 hands.

GraspGenX supplies camera-frame closing poses. cuRoboV2 selects the first
reachable pose and produces a collision-aware pregrasp followed by a short
linear approach to the closing pose. Only after Isaac confirms physical contact
does a second, independent request lift the object off the source support.
Standing/waist restoration preserves the lifted arms in Isaac. Planning
runs in a clean subprocess because Isaac Sim and cuRobo
intentionally ship incompatible Warp runtimes. This module has no endpoint-IK
or object-specific fallback.

Isaac/PhysX collision is mesh geometry, not axis-aligned bounding boxes.
No motion planner in this repository may inject filled scene AABB cuboids: a
filled AABB occupies empty volume (between table legs, above a thin tabletop)
and rejects motions that never hit live colliders. Closing-reach planning
carries the grasp target primitive and the actual support contact-layer mesh.
Carry planning carries neither.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from tarsphere.embodiment.dex3 import (
    DEX3_GRASP_JOINT_POSITIONS,
    DEX3_JOINT_NAMES,
    DEX3_OPEN_JOINT_POSITIONS,
    G1_MANIPULATION_HAND,
)
from tarsphere.infra.manipulation.g1_arm_ik import (
    G1_LEFT_ARM_LOWER,
    G1_LEFT_ARM_JOINT_NAMES,
    G1_LEFT_ARM_UPPER,
    G1_RIGHT_ARM_LOWER,
    G1_RIGHT_ARM_JOINT_NAMES,
    G1_RIGHT_ARM_UPPER,
)
from tarsphere.infra.manipulation.graspgenx import (
    GraspPoseCandidate,
    UNITREE_G1_PREGRASP_STANDOFF_M,
    grasp_pose_payload,
)
from tarsphere.infra.control.motion_planning import (
    SCENE_AABB_PLANNER_OBSTACLES_FORBIDDEN as G1_SCENE_AABB_OBSTACLES_FORBIDDEN,
    SupportSurfaceObstacle,
    TargetObjectObstacle,
    forbid_scene_aabb_planner_obstacles as forbid_g1_scene_aabb_obstacles,
    require_support_surface_obstacle,
    require_target_object_obstacle,
)
from tarsphere.infra.control.physical_contracts import (
    G1_ARM_TRAJECTORY_TRACKING_TOLERANCE_RAD,
    G1_GRASP_CARRY_RETRACTION_M,
    G1_GRASP_CARRY_WORLD_LIFT_M,
    G1_MANIPULATION_WAIST_PITCH_LIMIT_RAD,
    G1_MANIPULATION_WAIST_YAW_LIMIT_RAD,
    G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M,
    G1_STANDING_HEIGHT_COMMAND_M,
    TARGET_SCREENING_MAX_GRASP_SPAN_M,
)
from tarsphere.infra.simulation.demo_gpu import resolve_demo_gpu_mapping


CUROBO_V2_VERSION = "0.8.0"
G1_GRASP_PLANNING_TIMEOUT_S = 600.0
# Progressive stage-timing evidence written by the isolated worker next to its
# result payload. Each line is one completed stage, flushed immediately, so a
# timed-out worker still leaves the exact boundary where wall clock was spent.
G1_GRASP_WORKER_STAGES_FILENAME = "stages.jsonl"
G1_CUROBO_IK_SEEDS = 64
G1_CUROBO_TRAJECTORY_SEEDS = 8
# A large GraspGenX result is first tested as one batched cuRobo IK call with
# one independent problem per exact candidate.  This is not a learned or
# certified distance gate: every candidate receives the same IK seed count and
# retry budget as the original serial loop against the live state/collision
# world. Every failed row skips its redundant serial ``plan_pose`` call; every
# successful row still enters the original full trajectory pipeline.
G1_GRASP_BATCH_IK_PREFLIGHT_MIN_CANDIDATES = 8
G1_GRASP_BATCH_IK_PREFLIGHT_ATTEMPTS = 5
# Keep the IK-only working set bounded on a 24 GB device. MotionPlanner pads a
# smaller request to its configured maximum, so this cap is also the actual
# per-call batch width; the full candidate list is covered in chunks.
G1_GRASP_BATCH_IK_PREFLIGHT_MAX_BATCH_SIZE = 16
# Measured PD joints can overshoot a commanded limit by a fraction of a
# degree. cuRobo needs a start inside its limits; a genuine mapping or
# unit failure is off by radians, not hundredths.
G1_CUROBO_START_STATE_LIMIT_TOLERANCE_RAD = 0.02
# cuRobo's graph and trajectory validators require revolute starts to lie
# strictly inside their hard bounds. Project a measured saturation one
# milliradian inward; this is below the G1 tracking tolerance but avoids
# representing a physically valid hard-stop posture as an invalid graph node.
G1_CUROBO_START_STATE_LIMIT_MARGIN_RAD = 0.001
G1_GRASP_REACH_PROBE_CANDIDATE_LIMIT = 5
G1_GRASP_REACH_PROBE_COARSE_STEP_M = 0.05
G1_GRASP_REACH_PROBE_COARSE_MAX_M = 0.40
G1_GRASP_REACH_PROBE_FINE_STEP_M = 0.01


# cuRobo keeps an internal fractional margin away from every hard joint limit.
# Failed worker requests stay under one named temp root, not the process-wide
# temp directory, and only the most recent failures are retained.
G1_GRASP_PRESERVED_WORKER_DIRNAME = "tarsphere_g1_grasp_preserved"
G1_GRASP_PRESERVED_WORKER_RETENTION = 20
G1_CUROBO_GPU_LOCK_PREFIX = "tarsphere_g1_curobo_gpu"
# cuRobo's synthetic root places the nominal G1 pelvis 0.8 m above its
# planning origin. Runtime body-height commands are represented as changes
# around this live reference rather than as a second robot-base pose.
G1_CUROBO_REFERENCE_PELVIS_HEIGHT_M = 0.8
G1_PELVIS_HEIGHT_JOINT = "base_j_z"
# cuRobo tool_frames is ``{hand}_hand_palm_link``. GraspGenX gripper poses
# are converted through UNITREE_G1_GRIPPER_FROM_DEX3_PALM before the
# GoalToolPose is built, so the IK goal is that link origin. Palm-error
# reporting samples the same origin; there is no extra TCP offset.
G1_CUROBO_PALM_TOOL_POINT_IN_PALM_LINK_M = (0.0, 0.0, 0.0)


def g1_curobo_pelvis_model_height_from_body_height_m(body_height_m: float) -> float:
    """Map pelvis-above-stance height onto cuRobo's synthetic ``base_j_z``.

    The YAML default places the pelvis at 0.8 m. Standing (0.74 m) sits on
    that default; every other height is the same metric displacement. Live
    measurements may sit a few centimetres outside the certified command
    window and still need this mapping so the planning root is not rebuilt
    from a command the pelvis never reached.
    """

    height = float(body_height_m)
    if not math.isfinite(height):
        raise ValueError(
            "G1 pelvis-above-stance height must be finite to map onto "
            f"cuRobo base_j_z; got {body_height_m!r}."
        )
    return G1_CUROBO_REFERENCE_PELVIS_HEIGHT_M + height - G1_STANDING_HEIGHT_COMMAND_M


def g1_curobo_pelvis_model_height_m(command_height_m: float) -> float:
    """Map a live G1 height command onto cuRobo's synthetic ``base_j_z``.

    Commands must stay inside the certified [0.46, 0.74] m window. Measured
    pelvis height uses :func:`g1_curobo_pelvis_model_height_from_body_height_m`
    so a tracking error cannot abort the live root.
    """

    command = float(command_height_m)
    if not (
        G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M <= command <= G1_STANDING_HEIGHT_COMMAND_M
    ):
        raise ValueError(
            "G1 pelvis-height command must lie inside the certified "
            f"[{G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M}, "
            f"{G1_STANDING_HEIGHT_COMMAND_M}] m range; got {command}."
        )
    return g1_curobo_pelvis_model_height_from_body_height_m(command)


def palm_goal_root_to_pelvis_m(
    palm_goal_root_m: Sequence[float],
    *,
    pelvis_model_height_m: float,
) -> tuple[float, float, float]:
    """Express one cuRobo palm goal in the live pelvis frame.

    ``robot_root_from_camera`` maps camera poses into the cuRobo kinematic
    root fixed to the ground-projected stance. The pelvis sits directly above
    that root on the purely vertical ``base_j_z`` prismatic joint with no
    lateral or forward offset, so only the vertical component changes.
    """

    if len(palm_goal_root_m) != 3:
        raise ValueError("palm_goal_root_m must contain three coordinates.")
    x, y, z = (float(value) for value in palm_goal_root_m)
    height = float(pelvis_model_height_m)
    if not all(math.isfinite(value) for value in (x, y, z, height)):
        raise ValueError("palm_goal_root_m and pelvis_model_height_m must be finite.")
    return (x, y, z - height)


def g1_carry_palm_goal_offset_pelvis_root_m(
    *,
    waist_yaw_rad: float,
    waist_pitch_rad: float,
    retraction_m: float = G1_GRASP_CARRY_RETRACTION_M,
    world_lift_m: float = G1_GRASP_CARRY_WORLD_LIFT_M,
) -> tuple[float, float, float]:
    """Map carry retraction onto the cuRobo pelvis-root frame.

    Pull backward along the live torso -X axis, transformed through the
    measured waist yaw then pitch (waist roll is locked to zero). Additional
    lift stays along world +Z rather than rotating with the torso.
    """

    yaw = float(waist_yaw_rad)
    pitch = float(waist_pitch_rad)
    retract = float(retraction_m)
    lift = float(world_lift_m)
    if not all(math.isfinite(value) for value in (yaw, pitch, retract, lift)):
        raise ValueError(
            "G1 carry palm-goal offset requires finite waist yaw and pitch, "
            "retraction, and world-lift values."
        )
    if retract < 0.0 or lift <= 0.0:
        raise ValueError(
            "G1 carry palm-goal offset requires non-negative retraction and "
            "positive world lift."
        )
    sin_pitch = math.sin(pitch)
    cos_pitch = math.cos(pitch)
    return (
        -retract * cos_pitch * math.cos(yaw),
        -retract * cos_pitch * math.sin(yaw),
        retract * sin_pitch + lift,
    )


G1_WAIST_JOINT_NAMES = ("waist_yaw_joint", "waist_pitch_joint")
# Grasp/cuRobo hold roll at 0, matching the certified FixedJoint composition.
G1_FIXED_WAIST_JOINT_POSITIONS = {
    "waist_roll_joint": 0.0,
}
G1_MANIPULATION_WAIST_LIMITS_RAD = {
    "waist_yaw_joint": G1_MANIPULATION_WAIST_YAW_LIMIT_RAD,
    "waist_pitch_joint": G1_MANIPULATION_WAIST_PITCH_LIMIT_RAD,
}
# The shoulder-roll housings are second neighbours of the torso in the URDF
# (torso -> shoulder-pitch -> shoulder-roll). Their fitted collision spheres
# overlap at the locomotion policy's physically valid neutral arm pose even
# though the rigid meshes do not. Treat this symmetric mechanical assembly as
# one adjacent chain; all non-adjacent upper-body pairs remain collision-checked.
G1_ASSEMBLY_COLLISION_IGNORES = {
    "torso_link": (
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
    ),
}
# cuRobo obstacle name for the grasp target. The closing reach loads this
# cuboid together with the support contact mesh; carry planning loads neither.
G1_GRASP_TARGET_OBSTACLE_NAME = "grasp_target"
# cuRobo obstacle name for the authored source-support contact mesh. The closing
# reach loads this cuboid together with the grasp target so a palm or forearm
# trajectory that would press through the desk is rejected at planning time.
G1_SUPPORT_SURFACE_OBSTACLE_NAME = "support_surface"
# A grasp ends with the fingers enclosing the target, so the distal Dex3 links
# must be free to occupy it. The palm, the wrist and the forearm must not: a
# planned pose that puts them inside a kinematically pinned target cannot be
# reached, and the drive stalls against it.
G1_GRASP_TARGET_ENCLOSING_LINKS = {
    hand: (
        f"{hand}_hand_thumb_1_link",
        f"{hand}_hand_thumb_2_link",
        f"{hand}_hand_middle_0_link",
        f"{hand}_hand_middle_1_link",
        f"{hand}_hand_index_0_link",
        f"{hand}_hand_index_1_link",
    )
    for hand in ("left", "right")
}
# cuRobo can disable collision spheres only per robot link, not per
# (robot-link, world-obstacle) pair.  The links above therefore have to be
# disabled while solving against the grasp target, otherwise the fingers
# cannot enclose it.  That also makes those links invisible to the support
# slab during the solve.  Every successful result is consequently checked a
# second time below with all link spheres restored against the support contact mesh.
G1_SUPPORT_SURFACE_CLEARANCE_CONSTRAINT = "support_surface_collision"
G1_ARM_JOINT_NAMES = {
    "left": G1_LEFT_ARM_JOINT_NAMES,
    "right": G1_RIGHT_ARM_JOINT_NAMES,
}
G1_CUROBO_JOINT_NAMES = (
    (G1_PELVIS_HEIGHT_JOINT,)
    + G1_WAIST_JOINT_NAMES
    + G1_LEFT_ARM_JOINT_NAMES
    + G1_RIGHT_ARM_JOINT_NAMES
    + DEX3_JOINT_NAMES["left"]
    + DEX3_JOINT_NAMES["right"]
)
G1_MANIPULATION_JOINT_NAMES = {
    hand: (
        *G1_WAIST_JOINT_NAMES,
        *G1_ARM_JOINT_NAMES[hand],
    )
    for hand in ("left", "right")
}
G1_MANIPULATION_COORDINATE_NAMES = {
    hand: (
        "pelvis_height_command_m",
        *G1_WAIST_JOINT_NAMES,
        *G1_ARM_JOINT_NAMES[hand],
    )
    for hand in ("left", "right")
}


def g1_manipulation_waypoint_length(hand: str) -> int:
    """Return the manipulation waypoint dimension for one hand."""
    return len(G1_MANIPULATION_COORDINATE_NAMES[hand])


def g1_manipulation_waypoint_value(
    waypoint: Sequence[float],
    hand: str,
    coordinate_name: str,
) -> float:
    """Read one named manipulation coordinate from a planner waypoint."""
    names = G1_MANIPULATION_COORDINATE_NAMES[hand]
    values = tuple(float(value) for value in waypoint)
    if len(values) != len(names):
        raise ValueError(
            "G1 manipulation waypoint length does not match its hand: "
            f"expected={len(names)}, actual={len(values)}."
        )
    try:
        index = names.index(coordinate_name)
    except ValueError as exc:
        raise ValueError(
            f"Unknown G1 manipulation coordinate {coordinate_name!r} for {hand!r}."
        ) from exc
    return values[index]


def g1_manipulation_waypoint_arm_values(
    waypoint: Sequence[float],
    hand: str,
) -> tuple[float, ...]:
    """Return the seven arm joints for one hand from a manipulation waypoint."""
    return tuple(
        g1_manipulation_waypoint_value(waypoint, hand, name)
        for name in G1_ARM_JOINT_NAMES[hand]
    )


def g1_manipulation_waypoint_with_arm_values(
    waypoint: Sequence[float],
    hand: str,
    arm_values: Sequence[float],
) -> tuple[float, ...]:
    """Copy one waypoint, replacing only the seven arm joints."""
    names = G1_MANIPULATION_COORDINATE_NAMES[hand]
    arm_names = G1_ARM_JOINT_NAMES[hand]
    values = tuple(float(value) for value in waypoint)
    arm = tuple(float(value) for value in arm_values)
    if len(values) != len(names):
        raise ValueError(
            "G1 manipulation waypoint length does not match its hand: "
            f"expected={len(names)}, actual={len(values)}."
        )
    if len(arm) != len(arm_names):
        raise ValueError(
            "G1 manipulation waypoint arm replacement needs seven joints: "
            f"got {len(arm)}."
        )
    if not all(math.isfinite(value) for value in (*values, *arm)):
        raise ValueError(
            "G1 manipulation waypoint arm replacement requires finite values."
        )
    replaced = list(values)
    for coordinate_name, value in zip(arm_names, arm, strict=True):
        replaced[names.index(coordinate_name)] = value
    return tuple(replaced)


def g1_locomotion_zero_manipulation_waypoint(
    hand: str,
    *,
    pelvis_height_command_m: float,
) -> tuple[float, ...]:
    """Build a manipulation waypoint with zero waist and arm joints."""
    return (
        float(pelvis_height_command_m),
        *(0.0 for _ in G1_WAIST_JOINT_NAMES),
        *(0.0 for _ in G1_ARM_JOINT_NAMES[hand]),
    )


G1_GRASP_PLANNING_PHASES = ("pregrasp", "grasp", "carry_preflight")
G1_GRASP_PLANNING_OUTCOMES = (
    "ik_unreachable",
    "trajectory_infeasible",
    "near_previous_retention_failure",
)


@dataclass(frozen=True)
class G1GraspCandidateFailure:
    """Why one visual grasp candidate failed for one hand and one phase.

    ``pose_index`` identifies the GraspGenX proposal only inside the private
    planner request. ``hand`` and ``phase`` tell the Agent which embodiment
    branch failed. The optional errors and constraints are measurements cuRobo
    actually returned; they remain absent when cuRobo reports no feasible IK
    seed and therefore has no trajectory result to diagnose.
    """

    pose_index: int
    hand: str
    phase: str
    outcome: str
    minimum_position_error_m: float | None = None
    minimum_orientation_error_rad: float | None = None
    minimum_cspace_error: float | None = None
    any_trajectory_feasible: bool | None = None
    constraint_violations: tuple[str, ...] = ()
    palm_goal_pelvis_m: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.pose_index, bool) or self.pose_index < 0:
            raise ValueError("G1 failed grasp pose_index must be non-negative.")
        if self.hand != G1_MANIPULATION_HAND:
            raise ValueError("G1 failed grasp evidence must use the right hand.")
        if self.phase not in G1_GRASP_PLANNING_PHASES:
            raise ValueError(
                "G1 failed grasp phase must be pregrasp, grasp or carry_preflight."
            )
        if self.outcome not in G1_GRASP_PLANNING_OUTCOMES:
            raise ValueError(
                "G1 failed grasp outcome must be ik_unreachable, "
                "trajectory_infeasible, or near_previous_retention_failure."
            )
        for name in (
            "minimum_position_error_m",
            "minimum_orientation_error_rad",
            "minimum_cspace_error",
        ):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(float(value)) or float(value) < 0.0
            ):
                raise ValueError(f"G1 {name} must be finite and non-negative.")
        if self.any_trajectory_feasible is not None and not isinstance(
            self.any_trajectory_feasible,
            bool,
        ):
            raise TypeError("G1 any_trajectory_feasible must be Boolean or null.")
        if any(not str(item).strip() for item in self.constraint_violations):
            raise ValueError("G1 constraint violations must be non-empty strings.")
        if self.palm_goal_pelvis_m is not None:
            values = tuple(float(value) for value in self.palm_goal_pelvis_m)
            if len(values) != 3 or not all(math.isfinite(value) for value in values):
                raise ValueError(
                    "G1 palm_goal_pelvis_m must contain three finite values."
                )
            object.__setattr__(self, "palm_goal_pelvis_m", values)

    def as_dict(self) -> dict[str, Any]:
        """Serialize one failure across the isolated worker boundary."""

        return {
            "pose_index": self.pose_index,
            "hand": self.hand,
            "phase": self.phase,
            "outcome": self.outcome,
            "minimum_position_error_m": self.minimum_position_error_m,
            "minimum_orientation_error_rad": (self.minimum_orientation_error_rad),
            "minimum_cspace_error": self.minimum_cspace_error,
            "any_trajectory_feasible": self.any_trajectory_feasible,
            "constraint_violations": list(self.constraint_violations),
            "palm_goal_pelvis_m": (
                None
                if self.palm_goal_pelvis_m is None
                else [float(value) for value in self.palm_goal_pelvis_m]
            ),
        }

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
    ) -> "G1GraspCandidateFailure":
        """Validate one failure returned by the isolated worker."""

        if not isinstance(payload, Mapping):
            raise TypeError("G1 grasp candidate failure must be a mapping.")

        def optional_float(name: str) -> float | None:
            value = payload.get(name)
            return None if value is None else float(value)

        violations = payload.get("constraint_violations")
        if not isinstance(violations, Sequence) or isinstance(
            violations,
            (str, bytes),
        ):
            raise TypeError("G1 constraint_violations must be a sequence of strings.")
        feasible = payload.get("any_trajectory_feasible")
        palm_goal = payload.get("palm_goal_pelvis_m")
        palm_goal_pelvis_m = None
        if palm_goal is not None:
            if (
                not isinstance(palm_goal, Sequence)
                or isinstance(
                    palm_goal,
                    (str, bytes),
                )
                or len(palm_goal) != 3
            ):
                raise TypeError("G1 palm_goal_pelvis_m must be a length-3 sequence.")
            palm_goal_pelvis_m = tuple(float(value) for value in palm_goal)
        return cls(
            pose_index=int(payload["pose_index"]),
            hand=str(payload["hand"]),
            phase=str(payload["phase"]),
            outcome=str(payload["outcome"]),
            minimum_position_error_m=optional_float("minimum_position_error_m"),
            minimum_orientation_error_rad=optional_float(
                "minimum_orientation_error_rad"
            ),
            minimum_cspace_error=optional_float("minimum_cspace_error"),
            any_trajectory_feasible=(None if feasible is None else feasible),
            constraint_violations=tuple(str(item) for item in violations),
            palm_goal_pelvis_m=palm_goal_pelvis_m,
        )


@dataclass(frozen=True)
class G1GraspReachCorrection:
    """Body-forward reach correction measured by live cuRobo re-planning."""

    status: str
    basis: str
    body_forward_m: float | None = None
    approach: dict[str, Any] | None = None
    sample_count: int | None = None
    minimum_body_forward_m: float | None = None
    maximum_body_forward_m: float | None = None
    probe_budget_m: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in (
            "measured",
            "not_found_within_budget",
            "unavailable",
        ):
            raise ValueError(
                "G1 grasp reach correction status must be measured, "
                "not_found_within_budget, or unavailable."
            )
        if not str(self.basis).strip():
            raise ValueError("G1 grasp reach correction basis must be non-empty.")
        if self.status == "measured":
            if (
                self.body_forward_m is None
                or not math.isfinite(float(self.body_forward_m))
                or float(self.body_forward_m) < 0.0
            ):
                raise ValueError(
                    "Measured G1 grasp reach correction requires a finite "
                    "non-negative body_forward_m."
                )
            if not isinstance(self.approach, Mapping):
                raise TypeError(
                    "Measured G1 grasp reach correction requires an approach "
                    "descriptor."
                )
            if (
                self.sample_count is None
                or int(self.sample_count) != self.sample_count
                or int(self.sample_count) < 1
            ):
                raise ValueError(
                    "Measured G1 grasp reach correction requires a positive "
                    "sample_count."
                )
            if (
                self.minimum_body_forward_m is None
                or not math.isfinite(float(self.minimum_body_forward_m))
                or float(self.minimum_body_forward_m) < 0.0
            ):
                raise ValueError(
                    "Measured G1 grasp reach correction requires a finite "
                    "non-negative minimum_body_forward_m."
                )
            if (
                self.maximum_body_forward_m is None
                or not math.isfinite(float(self.maximum_body_forward_m))
                or float(self.maximum_body_forward_m) < 0.0
            ):
                raise ValueError(
                    "Measured G1 grasp reach correction requires a finite "
                    "non-negative maximum_body_forward_m."
                )
            if not (
                float(self.minimum_body_forward_m)
                <= float(self.body_forward_m)
                <= float(self.maximum_body_forward_m)
            ):
                raise ValueError(
                    "Measured body_forward_m must lie between "
                    "minimum_body_forward_m and maximum_body_forward_m."
                )
        elif (
            self.body_forward_m is not None
            or self.approach is not None
            or self.sample_count is not None
            or self.minimum_body_forward_m is not None
            or self.maximum_body_forward_m is not None
        ):
            raise ValueError(
                "Non-measured G1 grasp reach correction must omit "
                "body_forward_m, approach, and measured sample statistics."
            )
        if self.status == "not_found_within_budget":
            if (
                self.probe_budget_m is None
                or not math.isfinite(float(self.probe_budget_m))
                or float(self.probe_budget_m) <= 0.0
            ):
                raise ValueError(
                    "A budget-exhausted reach correction requires a positive "
                    "probe_budget_m."
                )
        elif self.probe_budget_m is not None:
            raise ValueError(
                "probe_budget_m is allowed only for not_found_within_budget."
            )
        if self.status == "unavailable":
            if not str(self.reason or "").strip():
                raise ValueError("An unavailable reach correction requires a reason.")
        elif self.reason is not None:
            raise ValueError("reason is allowed only for unavailable.")

    @classmethod
    def measured(
        cls,
        body_forward_m: float,
        approach: Mapping[str, Any],
        *,
        sample_count: int,
        minimum_body_forward_m: float,
        maximum_body_forward_m: float,
    ) -> "G1GraspReachCorrection":
        distance = float(body_forward_m)
        return cls(
            status="measured",
            body_forward_m=distance,
            basis=(
                "cuRobo re-planned failed candidate palm poses, each "
                "translated toward the robot along body-forward in "
                f"{G1_GRASP_REACH_PROBE_COARSE_STEP_M:.2f} m steps refined to "
                f"{G1_GRASP_REACH_PROBE_FINE_STEP_M:.2f} m; the distance "
                "reported is the median of those per-candidate translations."
            ),
            approach=dict(approach),
            sample_count=int(sample_count),
            minimum_body_forward_m=float(minimum_body_forward_m),
            maximum_body_forward_m=float(maximum_body_forward_m),
        )

    @classmethod
    def not_found_within_budget(cls) -> "G1GraspReachCorrection":
        budget = float(G1_GRASP_REACH_PROBE_COARSE_MAX_M)
        return cls(
            status="not_found_within_budget",
            basis=(
                "cuRobo re-planned failed palm poses after body-forward "
                f"translation up to {budget:.2f} m; none became solvable "
                "within that probe budget."
            ),
            probe_budget_m=budget,
        )

    @classmethod
    def unavailable(cls, *, reason: str) -> "G1GraspReachCorrection":
        return cls(
            status="unavailable",
            basis=(
                "cuRobo could not be re-planned against the failed palm "
                "poses, so no body-forward translation was measured."
            ),
            reason=str(reason).strip(),
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "basis": self.basis,
        }
        if self.body_forward_m is not None:
            payload["body_forward_m"] = float(self.body_forward_m)
        if self.approach is not None:
            payload["approach"] = dict(self.approach)
        if self.sample_count is not None:
            payload["sample_count"] = int(self.sample_count)
        if self.minimum_body_forward_m is not None:
            payload["minimum_body_forward_m"] = float(self.minimum_body_forward_m)
        if self.maximum_body_forward_m is not None:
            payload["maximum_body_forward_m"] = float(self.maximum_body_forward_m)
        if self.probe_budget_m is not None:
            payload["probe_budget_m"] = float(self.probe_budget_m)
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | None,
    ) -> "G1GraspReachCorrection | None":
        if payload is None:
            return None
        if not isinstance(payload, Mapping):
            raise TypeError("G1 grasp reach correction must be a mapping.")
        return cls(
            status=str(payload["status"]),
            basis=str(payload["basis"]),
            body_forward_m=(
                None
                if payload.get("body_forward_m") is None
                else float(payload["body_forward_m"])
            ),
            approach=(
                None if payload.get("approach") is None else dict(payload["approach"])
            ),
            sample_count=(
                None
                if payload.get("sample_count") is None
                else int(payload["sample_count"])
            ),
            minimum_body_forward_m=(
                None
                if payload.get("minimum_body_forward_m") is None
                else float(payload["minimum_body_forward_m"])
            ),
            maximum_body_forward_m=(
                None
                if payload.get("maximum_body_forward_m") is None
                else float(payload["maximum_body_forward_m"])
            ),
            probe_budget_m=(
                None
                if payload.get("probe_budget_m") is None
                else float(payload["probe_budget_m"])
            ),
            reason=(None if payload.get("reason") is None else str(payload["reason"])),
        )


class G1GraspPlanningInfeasible(RuntimeError):
    """No candidate has a complete collision-free manipulation plan."""

    def __init__(
        self,
        failures: Sequence[G1GraspCandidateFailure],
        *,
        reach_correction: G1GraspReachCorrection | None = None,
    ) -> None:
        rejected = tuple(failures)
        if not rejected or any(
            not isinstance(item, G1GraspCandidateFailure) for item in rejected
        ):
            raise ValueError(
                "An infeasible G1 grasp requires typed candidate failures."
            )
        if reach_correction is not None and not isinstance(
            reach_correction,
            G1GraspReachCorrection,
        ):
            raise TypeError(
                "G1 grasp reach correction must be a G1GraspReachCorrection."
            )
        self.failures = rejected
        self.reach_correction = reach_correction
        counts: dict[tuple[str, str, str], int] = {}
        for item in rejected:
            key = (item.hand, item.phase, item.outcome)
            counts[key] = counts.get(key, 0) + 1
        summary = ", ".join(
            f"{hand}:{phase}:{outcome}={count}"
            for (hand, phase, outcome), count in sorted(counts.items())
        )
        super().__init__("cuRoboV2 found no reachable G1 closing-pose plan: " + summary)


class G1CarryPlanningInfeasible(RuntimeError):
    """A valid carry solve failed at IK or collision-aware trajectory search."""

    def __init__(self, message: str, *, outcome: str) -> None:
        if outcome not in ("ik_unreachable", "trajectory_infeasible"):
            raise ValueError("Carry failure requires a typed IK or trajectory outcome.")
        super().__init__(message)
        self.outcome = outcome


class G1GraspWorkerFailure(RuntimeError):
    """The isolated planner process failed to return a valid result."""

    def __init__(self, message: str, *, returncode: int | None = None) -> None:
        super().__init__(message)
        self.returncode = returncode


class G1GraspWorkerTimeout(G1GraspWorkerFailure):
    """The isolated planner exceeded its bounded wall-clock lifetime."""

    def __init__(
        self,
        *,
        timeout_s: float,
        preserved_request_dir: Path | None = None,
        completed_stages: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        stages = tuple(completed_stages)
        stage_summary = (
            ", ".join(
                f"{record.get('stage')}={float(record.get('duration_s', 0.0)):.1f}s"
                for record in stages
            )
            or "no completed stages"
        )
        preserved = (
            f" Request preserved at {preserved_request_dir}."
            if preserved_request_dir is not None
            else ""
        )
        super().__init__(
            f"cuRoboV2 G1 worker exceeded its {timeout_s:g} s timeout. "
            f"Completed stages: {stage_summary}.{preserved}"
        )
        self.timeout_s = timeout_s
        self.preserved_request_dir = preserved_request_dir
        self.completed_stages = stages


@dataclass(frozen=True)
class G1PelvisHeightCoordinates:
    """Bind a fixed measured planning height to the unchanged control command.

    A locomotion height command is not a kinematically actuated prismatic
    joint. Partial-body plans keep the measured pelvis fixed; explicit body
    skills own posture changes and subsequent plans use their measured result.
    """

    # Current synthetic-joint position measured in the cuRobo model, in metres.
    current_model_height_m: float
    # Current pelvis-height command tracked by the G1 controller, in metres.
    current_command_height_m: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.current_model_height_m):
            raise ValueError("Current cuRobo pelvis height must be finite.")
        if not (
            G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M
            <= self.current_command_height_m
            <= G1_STANDING_HEIGHT_COMMAND_M
        ):
            raise ValueError(
                "Current G1 pelvis-height command must lie inside the "
                "certified crouch-to-stand range."
            )

    def command_height_m(self, model_height_m: float) -> float:
        """Reject vertical planning motion and preserve the existing command."""
        model_height = float(model_height_m)
        if not math.isfinite(model_height) or not math.isclose(
            model_height,
            self.current_model_height_m,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                "Partial-body planning must preserve the measured pelvis height."
            )
        return self.current_command_height_m


@dataclass(frozen=True)
class G1GraspPlan:
    """One cuRoboV2 trajectory to a GraspGenX closing pose."""

    # Index in the score-ranked GraspGenX candidate sequence.
    selected_pose_index: int
    # GraspGenX confidence attached to the selected pose.
    selected_confidence: float
    # Native Dex3 hand whose seven arm joints execute this plan.
    hand: str
    # Centre of the generated open-finger sweep volume in robot-root metres.
    grasp_center_robot_m: tuple[float, float, float]
    # Ordered meanings of each value in the reach trajectory.
    coordinate_names: tuple[str, ...]
    # Self-collision-aware motion from the live state to the closing pose.
    reach_positions: tuple[tuple[float, ...], ...]
    # Seconds between consecutive reach samples.
    trajectory_dt_s: float

    def __post_init__(self) -> None:
        if self.selected_pose_index < 0:
            raise ValueError("A G1 grasp plan needs a selected pose index.")
        if not math.isfinite(self.selected_confidence):
            raise ValueError("A selected GraspGenX confidence must be finite.")
        if self.hand != G1_MANIPULATION_HAND:
            raise ValueError("A G1 grasp plan must use the right hand.")
        center = np.asarray(self.grasp_center_robot_m, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("G1 grasp centre must contain three finite metres.")
        expected_coordinates = G1_MANIPULATION_COORDINATE_NAMES[self.hand]
        if self.coordinate_names != expected_coordinates:
            raise ValueError(
                "G1 trajectory coordinates do not match its selected hand: "
                f"expected={expected_coordinates!r}, "
                f"actual={self.coordinate_names!r}."
            )
        if not self.reach_positions:
            raise ValueError("G1 grasp reach trajectory is empty.")
        if any(
            len(waypoint) != len(expected_coordinates)
            for waypoint in self.reach_positions
        ):
            raise ValueError(
                "G1 grasp reach trajectory must contain pelvis height, "
                "waist yaw, waist pitch, and seven arm joints."
            )
        if not all(
            math.isfinite(float(value))
            for waypoint in self.reach_positions
            for value in waypoint
        ):
            raise ValueError("G1 grasp reach trajectory must be finite.")
        invalid_heights = [
            float(waypoint[0])
            for waypoint in self.reach_positions
            if not (
                G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M
                <= float(waypoint[0])
                <= G1_STANDING_HEIGHT_COMMAND_M
            )
        ]
        if invalid_heights:
            raise ValueError(
                "G1 grasp reach trajectory contains pelvis-height commands "
                f"outside the certified range: {invalid_heights[:3]!r}."
            )
        if any(
            abs(float(p[0]) - float(self.reach_positions[0][0])) > 1e-6
            for p in self.reach_positions
        ):
            raise ValueError(
                "G1 grasp trajectory must keep the pelvis-height command fixed."
            )
        if not math.isfinite(self.trajectory_dt_s) or self.trajectory_dt_s <= 0.0:
            raise ValueError("G1 trajectory_dt_s must be finite and positive.")


@dataclass(frozen=True)
class G1CarryPlan:
    """Post-contact cuRobo lift that clears the source support.

    Walking-arm restoration is not part of this trajectory. Isaac restores the
    locomotion upper body after the lift while keeping the finger closure.
    """

    hand: str
    coordinate_names: tuple[str, ...]
    positions: tuple[tuple[float, ...], ...]
    trajectory_dt_s: float
    retraction_step_count: int

    def __post_init__(self) -> None:
        if self.hand != G1_MANIPULATION_HAND:
            raise ValueError("A G1 carry plan must use the right hand.")
        expected = G1_MANIPULATION_COORDINATE_NAMES[self.hand]
        if self.coordinate_names != expected:
            raise ValueError(
                "G1 carry coordinates do not match the carrying hand: "
                f"expected={expected!r}, actual={self.coordinate_names!r}."
            )
        if not self.positions:
            raise ValueError("A G1 carry trajectory cannot be empty.")
        if not 1 <= self.retraction_step_count <= len(self.positions):
            raise ValueError(
                "G1 carry retraction_step_count must reference a non-empty "
                "retraction prefix of the trajectory."
            )
        if self.retraction_step_count != len(self.positions):
            raise ValueError(
                "A G1 carry plan contains only the lift off the source "
                "support; upper-body restoration is executed in Isaac."
            )
        if any(len(position) != len(expected) for position in self.positions):
            raise ValueError(
                "Every G1 carry waypoint must contain pelvis height, waist "
                "yaw, waist pitch, and seven arm joints."
            )
        if not all(
            math.isfinite(float(value))
            for position in self.positions
            for value in position
        ):
            raise ValueError("G1 carry trajectory must be finite.")
        if any(
            not (
                G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M
                <= float(position[0])
                <= G1_STANDING_HEIGHT_COMMAND_M
            )
            for position in self.positions
        ):
            raise ValueError(
                "G1 carry trajectory contains an invalid pelvis-height command."
            )
        if not math.isfinite(self.trajectory_dt_s) or self.trajectory_dt_s <= 0.0:
            raise ValueError("G1 carry trajectory_dt_s must be positive.")
        if any(
            abs(float(p[0]) - float(self.positions[0][0])) > 1e-6
            for p in self.positions
        ):
            raise ValueError(
                "G1 carry trajectory must keep the pelvis-height command fixed."
            )


def _planning_start_joint_positions(
    current_joint_positions: Mapping[str, float],
    joint_names: Sequence[str],
) -> list[float]:
    """Read the cuRobo start vector from the projected live-state mapping."""

    return [float(current_joint_positions[name]) for name in joint_names]


def _apply_curobo_start_state_limits(
    current_joint_positions: dict[str, float],
    *,
    joint_names: Sequence[str],
    limits: Any,
    tolerance_rad: float = G1_CUROBO_START_STATE_LIMIT_TOLERANCE_RAD,
) -> dict[str, float]:
    """Project measured saturation into cuRobo's strict feasible interior."""

    invalid_start = []
    for index, name in enumerate(joint_names):
        value = float(current_joint_positions[name])
        lower = float(limits[0, index])
        upper = float(limits[1, index])
        tolerance = float(tolerance_rad)
        projected_lower = lower + G1_CUROBO_START_STATE_LIMIT_MARGIN_RAD
        projected_upper = upper - G1_CUROBO_START_STATE_LIMIT_MARGIN_RAD
        if projected_lower > projected_upper:
            raise RuntimeError(
                f"cuRobo joint {name!r} has no feasible interior after "
                "applying the start-state margin."
            )
        if value < lower or value > upper:
            in_tolerance = (
                value >= lower - tolerance
                if value < lower
                else value <= upper + tolerance
            )
            if not in_tolerance:
                invalid_start.append(
                    f"{name}={value:.5f} outside [{lower:.5f},{upper:.5f}]"
                )
                continue
        projected = min(projected_upper, max(projected_lower, value))
        if projected != value:
            adjustment = abs(projected - value)
            current_joint_positions[name] = projected
            print(
                f"[G1-GRASP] projected {name} start by "
                f"{adjustment:.5f} rad "
                f"into [{lower:.5f},{upper:.5f}]",
                flush=True,
            )
    if invalid_start:
        raise RuntimeError(
            "Live G1 manipulation state violates the cuRobo joint "
            f"contract: {invalid_start!r}."
        )
    return current_joint_positions


class G1GraspPlanner:
    """Create one two-stage grasp approach and one post-contact carry plan."""

    def __init__(
        self,
        *,
        carry_lift_distance_m: float = G1_GRASP_CARRY_WORLD_LIFT_M,
        planning_stage_recorder: Callable[[str, float], None] | None = None,
    ) -> None:
        # Applied only by ``plan_carry`` after live contact is established.
        self.carry_lift_distance_m = float(carry_lift_distance_m)
        if (
            not math.isfinite(self.carry_lift_distance_m)
            or self.carry_lift_distance_m <= 0.0
        ):
            raise ValueError("G1 carry_lift_distance_m must be finite and positive.")
        # Receives one completed stage name and its wall-clock duration.
        # The isolated worker records these progressively so a timed-out
        # process still shows whether cuRobo import, planner construction,
        # warm-up, pregrasp, or closing approach consumed the budget.
        self.planning_stage_recorder = planning_stage_recorder

    def _record_planning_stage(self, stage: str, duration_s: float) -> None:
        if self.planning_stage_recorder is not None:
            self.planning_stage_recorder(stage, float(duration_s))

    def plan(
        self,
        *,
        candidates: Sequence[GraspPoseCandidate],
        robot_root_from_camera: np.ndarray,
        current_joint_positions: Mapping[str, float],
        current_body_height_command_m: float,
        hand: str,
        grasp_target_object_id: str,
        source_support_object_id: str,
        target_obstacle: TargetObjectObstacle,
        support_obstacle: SupportSurfaceObstacle,
        exclude_pose_indices: Sequence[int] = (),
    ) -> G1GraspPlan:
        """Return the first score-ranked closing pose with a valid reach.

        ``grasp_target_object_id`` is the object the runtime decided to grasp
        and ``target_obstacle`` is the bounding primitive measured for it. They
        come from different layers, so the planner refuses to proceed unless
        they name the same object. ``source_support_object_id`` and
        ``support_obstacle`` obey the same contract for the resting surface.
        Without the thin slab a backhand reach can plan a path through the desk.
        """

        _require_curobo_v2()
        transform = _rigid_transform(robot_root_from_camera)
        proposals = tuple(candidates)
        if not proposals:
            raise ValueError("G1 grasp planning requires GraspGenX poses.")
        normalized_hand = str(hand).strip().lower()
        if normalized_hand != G1_MANIPULATION_HAND:
            raise ValueError("G1 grasp planning supports only the right hand.")
        target = _require_graspable_target_obstacle(
            target_obstacle,
            grasp_target_object_id=grasp_target_object_id,
        )
        support = require_support_surface_obstacle(
            support_obstacle,
            support_object_id=source_support_object_id,
        )
        excluded = {int(index) for index in exclude_pose_indices}
        if any(index < 0 for index in excluded):
            raise ValueError("G1 grasp exclude_pose_indices must be non-negative.")
        missing = [
            name
            for name in G1_CUROBO_JOINT_NAMES
            if name not in current_joint_positions
        ]
        if missing:
            raise ValueError(f"G1 cuRobo current joint state is missing {missing!r}.")
        current_joint_positions = {
            str(name): float(value) for name, value in current_joint_positions.items()
        }

        ranked_proposals = sorted(
            enumerate(proposals),
            key=lambda item: (-float(item[1].confidence), item[0]),
        )
        # Wrist roll is left to cuRobo. The Dex3 pinch closes thumb against the
        # index/middle pair, which share one retention role, so rolling the palm
        # about its approach axis produces a mechanically equivalent grasp that
        # only reachability can rule out.
        failures: list[G1GraspCandidateFailure] = []
        plannable_proposals: list[tuple[int, GraspPoseCandidate]] = []
        for pose_index, candidate in ranked_proposals:
            if pose_index in excluded:
                failures.append(
                    G1GraspCandidateFailure(
                        pose_index=pose_index,
                        hand=normalized_hand,
                        phase="grasp",
                        outcome="near_previous_retention_failure",
                    )
                )
                continue
            plannable_proposals.append((pose_index, candidate))
        if not plannable_proposals:
            raise G1GraspPlanningInfeasible(failures)

        batch_reachable_pose_indices: set[int] | None = None
        if len(plannable_proposals) >= G1_GRASP_BATCH_IK_PREFLIGHT_MIN_CANDIDATES:
            batch_planner = self._create_hand_planner(
                hand=normalized_hand,
                current_joint_positions=current_joint_positions,
                current_body_height_command_m=current_body_height_command_m,
                target_obstacle=target,
                support_obstacle=support,
                source_support_object_id=source_support_object_id,
                max_batch_size=min(
                    len(plannable_proposals),
                    G1_GRASP_BATCH_IK_PREFLIGHT_MAX_BATCH_SIZE,
                ),
                warmup=False,
            )
            try:
                batch_reachable_pose_indices = self._batch_reachable_pose_indices(
                    planner=batch_planner,
                    plannable_proposals=plannable_proposals,
                    hand=normalized_hand,
                    robot_root_from_camera=transform,
                    current_joint_positions=current_joint_positions,
                )
            finally:
                del batch_planner
                _release_curobo_cuda_memory()

        # The ordinary score-ranked planner stays batch-size one. cuRobo pads
        # smaller requests to ``max_batch_size``; reusing the preflight planner
        # here would silently multiply every serial trajopt and correction
        # probe by the candidate count.
        planner = self._create_hand_planner(
            hand=normalized_hand,
            current_joint_positions=current_joint_positions,
            current_body_height_command_m=current_body_height_command_m,
            target_obstacle=target,
            support_obstacle=support,
            source_support_object_id=source_support_object_id,
        )
        try:
            reach_correction: G1GraspReachCorrection | None = None
            pelvis_model_height_m = float(
                current_joint_positions[G1_PELVIS_HEIGHT_JOINT]
            )
            controllability_fallback: G1GraspPlan | None = None
            for pose_index, candidate in plannable_proposals:
                if (
                    batch_reachable_pose_indices is not None
                    and pose_index not in batch_reachable_pose_indices
                ):
                    palm = transform @ candidate.palm_pose_camera(hand=normalized_hand)
                    failures.append(
                        G1GraspCandidateFailure(
                            pose_index=pose_index,
                            hand=normalized_hand,
                            phase="grasp",
                            outcome="ik_unreachable",
                            palm_goal_pelvis_m=palm_goal_root_to_pelvis_m(
                                tuple(float(value) for value in palm[:3, 3]),
                                pelvis_model_height_m=pelvis_model_height_m,
                            ),
                        )
                    )
                    continue
                planned = self._plan_candidate(
                    planner=planner,
                    candidate=candidate,
                    pose_index=pose_index,
                    hand=normalized_hand,
                    robot_root_from_camera=transform,
                    current_joint_positions=current_joint_positions,
                    current_body_height_command_m=(current_body_height_command_m),
                    support_obstacle=support,
                )
                if isinstance(planned, G1GraspPlan):
                    preflight_started = time.monotonic()
                    try:
                        carry_plan = self._preflight_carry(
                            planner=planner,
                            planned=planned,
                            current_joint_positions=current_joint_positions,
                            current_body_height_command_m=current_body_height_command_m,
                        )
                    except G1CarryPlanningInfeasible as exc:
                        print(
                            f"[G1-GRASP] pose={pose_index} carry preflight rejected: {exc}",
                            flush=True,
                        )
                        failures.append(
                            G1GraspCandidateFailure(
                                pose_index=pose_index,
                                hand=normalized_hand,
                                phase="carry_preflight",
                                outcome=exc.outcome,
                            )
                        )
                        continue
                    finally:
                        self._record_planning_stage(
                            f"carry_preflight:pose_{pose_index}",
                            time.monotonic() - preflight_started,
                        )
                    terminal_margin = g1_carry_terminal_joint_limit_margin_rad(
                        carry_plan,
                    )
                    if terminal_margin >= G1_ARM_TRAJECTORY_TRACKING_TOLERANCE_RAD:
                        return planned
                    if controllability_fallback is None:
                        controllability_fallback = planned
                    print(
                        "[G1-GRASP] "
                        f"pose={pose_index} carry terminal has only "
                        f"{terminal_margin:.4f} rad hard-limit margin; "
                        "continuing score order for a tracking-robust pose.",
                        flush=True,
                    )
                    continue
                failures.append(planned)
            if controllability_fallback is not None:
                # Do not shrink the graspable asset set. If every otherwise
                # feasible candidate is near a hard stop, preserve the
                # historical highest-confidence result and let the unchanged
                # measured tracking gate decide its execution.
                return controllability_fallback
            correction_started = time.monotonic()
            try:
                reach_correction = self._measure_reach_correction(
                    planner=planner,
                    failures=failures,
                    plannable_proposals=plannable_proposals,
                    robot_root_from_camera=transform,
                    current_joint_positions=current_joint_positions,
                    hand=normalized_hand,
                )
            finally:
                self._record_planning_stage(
                    "reach_correction_probe",
                    time.monotonic() - correction_started,
                )
        finally:
            del planner
            _release_curobo_cuda_memory()
        raise G1GraspPlanningInfeasible(
            failures,
            reach_correction=reach_correction,
        )

    def _preflight_carry(
        self,
        *,
        planner: Any,
        planned: G1GraspPlan,
        current_joint_positions: Mapping[str, float],
        current_body_height_command_m: float,
    ) -> G1CarryPlan:
        """Reject a closing solution with no predicted post-contact lift.

        Nominal closed fingers are a planning assumption, not contact evidence.
        This trajectory is discarded; physics still replans after real contact
        from the measured hand, waist, pelvis and palm pose.
        """
        import torch
        from curobo.types import JointState

        predicted, command_height = _predicted_closing_joint_state(
            planned=planned,
            current_joint_positions=current_joint_positions,
            current_body_height_command_m=current_body_height_command_m,
        )
        names = list(G1_MANIPULATION_JOINT_NAMES[planned.hand])
        state = JointState.from_position(
            torch.tensor(
                [[predicted[name] for name in names]],
                device="cuda",
                dtype=torch.float32,
            ),
            joint_names=names,
        )
        with torch.no_grad():
            kinematics = planner.compute_kinematics(state)
        if kinematics.tool_poses is None:
            raise RuntimeError("Carry preflight requires predicted closing-palm FK.")
        palm = kinematics.tool_poses.get_link_pose(f"{planned.hand}_hand_palm_link")
        closing_palm = palm.get_matrix().detach().cpu().numpy().reshape(-1, 4, 4)
        if len(closing_palm) != 1:
            raise RuntimeError("Carry preflight requires exactly one closing state.")
        return self.plan_carry(
            hand=planned.hand,
            current_joint_positions=predicted,
            current_body_height_command_m=command_height,
            closing_palm_pose_robot=closing_palm[0],
        )

    def _batch_reachable_pose_indices(
        self,
        *,
        planner: Any,
        plannable_proposals: Sequence[tuple[int, GraspPoseCandidate]],
        hand: str,
        robot_root_from_camera: np.ndarray,
        current_joint_positions: Mapping[str, float],
    ) -> set[int]:
        """Return every exact closing pose that passes independent batched IK.

        Each successful row deliberately falls through to the existing score-
        ranked, support-aware trajectory loop unchanged. Each failed row skips
        only a redundant serial ``plan_pose`` call. Every row gets the same
        seed count and attempts that ``MotionPlanner.plan_pose`` gives one
        serial goal; this changes scheduling, not search budget.
        """

        import torch
        from curobo.types import GoalToolPose, JointState

        if not plannable_proposals:
            raise ValueError("G1 batched IK preflight requires candidates.")
        palms = [
            robot_root_from_camera @ candidate.palm_pose_camera(hand=hand)
            for _pose_index, candidate in plannable_proposals
        ]
        reachable: set[int] = set()
        started = time.monotonic()
        planner.disable_link_collision(list(G1_GRASP_TARGET_ENCLOSING_LINKS[hand]))
        try:
            batch_size = int(planner.ik_solver.config.max_batch_size)
            for chunk_start in range(0, len(palms), batch_size):
                chunk = palms[chunk_start : chunk_start + batch_size]
                real_count = len(chunk)
                # cuRobo 0.8 keeps the first batch-sized goal buffer and its
                # JointState copy path does not consistently pad a shorter
                # later batch. Duplicate the final real goal so every call has
                # the configured width, then discard those padded result rows.
                if len(chunk) < batch_size:
                    chunk = [
                        *chunk,
                        *([chunk[-1]] * (batch_size - len(chunk))),
                    ]
                goal = GoalToolPose(
                    tool_frames=planner.tool_frames,
                    position=torch.tensor(
                        [
                            tuple(float(value) for value in palm[:3, 3])
                            for palm in chunk
                        ],
                        device="cuda",
                        dtype=torch.float32,
                    ).reshape(len(chunk), 1, 1, 1, 3),
                    quaternion=torch.tensor(
                        [
                            tuple(
                                float(value)
                                for value in _rotation_to_quaternion_wxyz(palm[:3, :3])
                            )
                            for palm in chunk
                        ],
                        device="cuda",
                        dtype=torch.float32,
                    ).reshape(len(chunk), 1, 1, 1, 4),
                )
                current = JointState.from_position(
                    torch.tensor(
                        [
                            _planning_start_joint_positions(
                                current_joint_positions,
                                G1_MANIPULATION_JOINT_NAMES[hand],
                            )
                            for _palm in chunk
                        ],
                        device="cuda",
                        dtype=torch.float32,
                    ),
                    joint_names=list(G1_MANIPULATION_JOINT_NAMES[hand]),
                )
                for _attempt in range(G1_GRASP_BATCH_IK_PREFLIGHT_ATTEMPTS):
                    result = planner.ik_solver.solve_pose(
                        goal,
                        current_state=current,
                        return_seeds=1,
                    )
                    success = getattr(result, "success", None)
                    if success is None or not success.numel():
                        continue
                    row_success = (
                        success.reshape(batch_size, -1)
                        .any(dim=1)
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    for local_index, succeeded in enumerate(row_success[:real_count]):
                        if succeeded:
                            reachable.add(
                                plannable_proposals[chunk_start + local_index][0]
                            )
                    if len(reachable) == len(plannable_proposals):
                        return reachable
            return reachable
        finally:
            planner.enable_link_collision(list(G1_GRASP_TARGET_ENCLOSING_LINKS[hand]))
            self._record_planning_stage(
                "closing_batch_ik_preflight",
                time.monotonic() - started,
            )

    def _create_hand_planner(
        self,
        *,
        hand: str,
        current_joint_positions: Mapping[str, float],
        current_body_height_command_m: float,
        target_obstacle: TargetObjectObstacle | None,
        support_obstacle: SupportSurfaceObstacle | None,
        source_support_object_id: str | None = None,
        selected_hand_joint_positions: Sequence[float] | None = None,
        max_batch_size: int = 1,
        warmup: bool = True,
    ) -> Any:
        """Plan one arm and waist yaw/pitch at the measured fixed pelvis.

        The closing reach loads the grasp target and the support contact mesh, the
        only world colliders this planner carries. The post-contact carry
        passes ``None`` for both: by then the target is retained by the hand,
        so it is no longer an independent body in the way, and adding the slab
        would reject the known-good start state with fingers at the surface.
        """

        import_started = time.monotonic()
        from curobo.content import get_robot_configs_path
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo._src.types.robot import RobotCfg
        from curobo._src.util_file import load_yaml

        self._record_planning_stage(
            "curobo_import",
            time.monotonic() - import_started,
        )

        # plan() and plan_carry() reuse this mapping for the cuRobo start
        # JointState. Keep it mutable so the later limit clamp is visible
        # to every downstream consumer.
        if not isinstance(current_joint_positions, dict):
            current_joint_positions = dict(current_joint_positions)
        construction_started = time.monotonic()
        source = Path(get_robot_configs_path()) / "unitree_g1.yml"
        configuration = copy.deepcopy(load_yaml(str(source))["kinematics"])
        other_hand = "left" if hand == "right" else "right"
        upper_links = [
            name
            for name in configuration["collision_link_names"]
            if name
            in {
                "pelvis",
                "pelvis_contour_link",
                "waist_yaw_link",
                "waist_roll_link",
                "torso_link",
                "head_link",
                "logo_link",
            }
            or any(
                marker in name for marker in ("shoulder", "elbow", "wrist", "_hand_")
            )
        ]
        locked_positions = {
            name: float(current_joint_positions[name])
            for name in (G1_ARM_JOINT_NAMES[other_hand] + DEX3_JOINT_NAMES[other_hand])
        }
        locked_positions.update(G1_FIXED_WAIST_JOINT_POSITIONS)
        selected_fingers = tuple(
            DEX3_OPEN_JOINT_POSITIONS[hand]
            if selected_hand_joint_positions is None
            else selected_hand_joint_positions
        )
        if len(selected_fingers) != len(DEX3_JOINT_NAMES[hand]) or not all(
            math.isfinite(float(value)) for value in selected_fingers
        ):
            raise ValueError(
                "G1 selected-hand planning state must contain seven finite "
                "Dex3 joint positions."
            )
        locked_positions.update(
            dict(
                zip(
                    DEX3_JOINT_NAMES[hand],
                    selected_fingers,
                    strict=True,
                )
            )
        )
        pelvis_coordinates = G1PelvisHeightCoordinates(
            current_model_height_m=float(
                current_joint_positions[G1_PELVIS_HEIGHT_JOINT]
            ),
            current_command_height_m=float(current_body_height_command_m),
        )
        locked_positions[G1_PELVIS_HEIGHT_JOINT] = (
            pelvis_coordinates.current_model_height_m
        )
        print(
            "[G1-GRASP] cuRobo pelvis fixed at measured height: "
            f"command_m={current_body_height_command_m:.3f} "
            f"model_start_m="
            f"{current_joint_positions[G1_PELVIS_HEIGHT_JOINT]:.3f}",
            flush=True,
        )
        locked_positions.update(
            {
                "base_j_x": 0.0,
                "base_j_y": 0.0,
                "base_j_xtheta": 0.0,
                "base_j_ytheta": 0.0,
                "base_j_ztheta": 0.0,
            }
        )
        configuration["base_link"] = "base_link"
        configuration["tool_frames"] = [f"{hand}_hand_palm_link"]
        configuration["collision_link_names"] = upper_links
        configuration["collision_spheres"] = {
            name: spheres
            for name, spheres in configuration["collision_spheres"].items()
            if name in upper_links
        }
        configuration["self_collision_ignore"] = {
            name: [other for other in ignored if other in upper_links]
            for name, ignored in configuration["self_collision_ignore"].items()
            if name in upper_links
        }
        for link, neighbours in G1_ASSEMBLY_COLLISION_IGNORES.items():
            ignored = configuration["self_collision_ignore"].setdefault(link, [])
            ignored.extend(
                neighbour
                for neighbour in neighbours
                if neighbour in upper_links and neighbour not in ignored
            )
        configuration["self_collision_buffer"] = {
            name: value
            for name, value in configuration["self_collision_buffer"].items()
            if name in upper_links
        }
        configuration["lock_joints"] = locked_positions
        # The null-space posture is the live robot state. This makes waist
        # yaw and pitch available for reachability without attracting
        # the solver toward a static asset pose unrelated to the current body.
        cspace = configuration["cspace"]
        cspace_names = tuple(str(name) for name in cspace["joint_names"])
        default_positions = list(cspace["default_joint_position"])
        for name, value in current_joint_positions.items():
            if name in cspace_names:
                default_positions[cspace_names.index(name)] = float(value)
        cspace["default_joint_position"] = default_positions
        robot_configuration = RobotCfg.create({"kinematics": configuration})
        kinematics = robot_configuration.kinematics.kinematics_config
        for name, absolute_limit in G1_MANIPULATION_WAIST_LIMITS_RAD.items():
            waist_index = kinematics.joint_names.index(name)
            kinematics.joint_limits.position[0, waist_index] = max(
                float(kinematics.joint_limits.position[0, waist_index]),
                -float(absolute_limit),
            )
            kinematics.joint_limits.position[1, waist_index] = min(
                float(kinematics.joint_limits.position[1, waist_index]),
                float(absolute_limit),
            )
        planner_config = MotionPlannerCfg.create(
            robot=robot_configuration,
            # Without a collision cache cuRobo builds no scene collision
            # checker at all, so update_world would have nowhere to put the
            # cuboids and the reach would silently be self-collision only.
            scene_model=None,
            collision_cache=(
                None if target_obstacle is None else {"cuboid": 1, "mesh": 1}
            ),
            num_ik_seeds=G1_CUROBO_IK_SEEDS,
            num_trajopt_seeds=G1_CUROBO_TRAJECTORY_SEEDS,
            max_batch_size=max(1, int(max_batch_size)),
            # Each worker solves a short, heterogeneous sequence of pose and
            # joint-space goals.  cuRobo CUDA graphs require one frozen tensor
            # shape and this build cannot reset them when the phase changes.
            # The worker is already isolated from Isaac, so eager execution is
            # the correct deterministic mode here.
            use_cuda_graph=False,
        )
        trajectory_config = planner_config.trajopt_solver_config
        transition_config = (
            trajectory_config.core_cfg.metrics_rollout_config.transition_model_cfg
        )
        trajectory_config.interpolation_buffer_size = _curobo_interpolation_buffer_size(
            horizon=transition_config.horizon,
            maximum_dt=trajectory_config.maximum_trajectory_dt,
            interpolation_dt=trajectory_config.interpolation_dt,
        )
        planner = MotionPlanner(planner_config)
        self._record_planning_stage(
            "planner_construction",
            time.monotonic() - construction_started,
        )
        if target_obstacle is not None:
            if support_obstacle is None:
                raise RuntimeError(
                    "Closing-reach planning requires the support-surface mesh; "
                    "without it a backhand reach can pass through the desk."
                )
            if source_support_object_id is None:
                raise RuntimeError(
                    "Closing-reach planner construction requires the authored "
                    "source-support object id so the support mesh identity can "
                    "be checked against the runtime binding."
                )
            _load_world_obstacles(
                planner,
                target=target_obstacle,
                support=support_obstacle,
                support_object_id=source_support_object_id,
            )
        elif support_obstacle is not None:
            raise RuntimeError(
                "Carry planning carries no world obstacles; the support mesh "
                "would reject the known-good start with fingers at the surface."
            )
        if warmup:
            warmup_started = time.monotonic()
            planner.warmup(enable_graph=False, num_warmup_iterations=2)
            self._record_planning_stage(
                "planner_warmup",
                time.monotonic() - warmup_started,
            )
        else:
            self._record_planning_stage("planner_warmup_skipped_ik_only", 0.0)
        actual_joints = tuple(planner.kinematics.joint_names)
        expected_joints = G1_MANIPULATION_JOINT_NAMES[hand]
        if actual_joints != expected_joints:
            raise RuntimeError(
                "cuRoboV2 did not reduce G1 to the requested partial-body "
                "planning coordinates: "
                f"expected={expected_joints!r}, actual={actual_joints!r}."
            )
        limits = planner.kinematics.kinematics_config.joint_limits.position
        _apply_curobo_start_state_limits(
            current_joint_positions,
            joint_names=actual_joints,
            limits=limits,
        )
        return planner

    def _plan_candidate(
        self,
        *,
        planner: Any,
        candidate: GraspPoseCandidate,
        pose_index: int,
        hand: str,
        robot_root_from_camera: np.ndarray,
        current_joint_positions: Mapping[str, float],
        current_body_height_command_m: float,
        support_obstacle: SupportSurfaceObstacle,
    ) -> G1GraspPlan | G1GraspCandidateFailure:
        """Reach an open pregrasp, then enter linearly; verify both support paths."""

        import torch
        from curobo.types import JointState

        palm = robot_root_from_camera @ candidate.palm_pose_camera(hand=hand)
        pregrasp_palm = robot_root_from_camera @ candidate.palm_pose_camera(
            hand=hand, standoff_m=UNITREE_G1_PREGRASP_STANDOFF_M
        )
        current = JointState.from_position(
            torch.tensor(
                [
                    _planning_start_joint_positions(
                        current_joint_positions,
                        G1_MANIPULATION_JOINT_NAMES[hand],
                    )
                ],
                device="cuda",
                dtype=torch.float32,
            ),
            joint_names=list(G1_MANIPULATION_JOINT_NAMES[hand]),
        )
        results = []
        for phase, phase_palm in (("pregrasp", pregrasp_palm), ("grasp", palm)):
            reach_started = time.monotonic()
            result = self._plan_pose_phase(
                planner=planner,
                palm_pose_robot=phase_palm,
                start=current,
                disabled_collision_links=(
                    G1_GRASP_TARGET_ENCLOSING_LINKS[hand] if phase == "grasp" else ()
                ),
                # GraspGenX +Z maps to right Dex3 palm +Y, not palm +Z.
                linear_approach_axis="y" if phase == "grasp" else None,
            )
            self._record_planning_stage(
                f"{phase}_reach:pose_{pose_index}",
                time.monotonic() - reach_started,
            )
            palm_goal = palm_goal_root_to_pelvis_m(
                tuple(float(value) for value in phase_palm[:3, 3]),
                pelvis_model_height_m=float(
                    current_joint_positions[G1_PELVIS_HEIGHT_JOINT]
                ),
            )
            if not _motion_succeeded(result):
                return _grasp_candidate_failure(
                    pose_index=pose_index,
                    hand=hand,
                    phase=phase,
                    result=result,
                    palm_goal_pelvis_m=palm_goal,
                )
            support_clearance_m = _minimum_support_clearance_for_trajectory(
                planner=planner,
                trajectory=result.interpolated_trajectory,
                last_timestep=result.interpolated_last_tstep,
                support_obstacle=support_obstacle,
            )
            if support_clearance_m <= 0.0:
                print(
                    "[G1-GRASP] restored robot collision sphere intersects support mesh: "
                    f"phase={phase} pose={pose_index} clearance_m={support_clearance_m:.5f}.",
                    flush=True,
                )
                return G1GraspCandidateFailure(
                    pose_index=pose_index,
                    hand=hand,
                    phase=phase,
                    outcome="trajectory_infeasible",
                    any_trajectory_feasible=False,
                    constraint_violations=(G1_SUPPORT_SURFACE_CLEARANCE_CONSTRAINT,),
                    palm_goal_pelvis_m=palm_goal,
                )
            results.append(result)
            if phase == "pregrasp":
                trajectory = result.interpolated_trajectory
                names = list(trajectory.joint_names)
                positions = trajectory.position.reshape(-1, len(names))
                length = int(result.interpolated_last_tstep.reshape(-1)[0].item())
                if not 1 <= length <= positions.shape[0]:
                    raise RuntimeError("cuRobo returned an invalid pregrasp endpoint.")
                current = JointState.from_position(
                    positions[length - 1 : length], joint_names=names
                )
                current = current.reorder(list(G1_MANIPULATION_JOINT_NAMES[hand]))

        trajectory_dt_s = float(planner.trajopt_solver.config.interpolation_dt)
        pelvis_coordinates = G1PelvisHeightCoordinates(
            current_model_height_m=float(
                current_joint_positions[G1_PELVIS_HEIGHT_JOINT]
            ),
            current_command_height_m=float(current_body_height_command_m),
        )
        grasp_center = (
            robot_root_from_camera[:3, :3] @ candidate.sweep_center_camera_m
            + robot_root_from_camera[:3, 3]
        )
        reach_positions = tuple(
            waypoint
            for result in results
            for waypoint in _control_trajectory(
                result.interpolated_trajectory,
                result.interpolated_last_tstep,
                hand=hand,
                pelvis_coordinates=pelvis_coordinates,
            )
        )
        return G1GraspPlan(
            selected_pose_index=pose_index,
            selected_confidence=float(candidate.confidence),
            hand=hand,
            grasp_center_robot_m=tuple(float(value) for value in grasp_center),
            coordinate_names=G1_MANIPULATION_COORDINATE_NAMES[hand],
            reach_positions=reach_positions,
            trajectory_dt_s=trajectory_dt_s,
        )

    @staticmethod
    def _plan_pose_phase(
        *,
        planner: Any,
        palm_pose_robot: np.ndarray,
        start: Any,
        disabled_collision_links: Sequence[str] = (),
        linear_approach_axis: str | None = None,
    ) -> Any:
        """Plan one phase while keeping cuRobo's public state contract exact."""

        import torch
        from curobo.types import GoalToolPose
        from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

        palm = _rigid_transform(palm_pose_robot)
        goal = GoalToolPose(
            tool_frames=planner.tool_frames,
            position=torch.tensor(
                palm[:3, 3],
                device="cuda",
                dtype=torch.float32,
            ).reshape(1, 1, 1, 1, 3),
            quaternion=torch.tensor(
                _rotation_to_quaternion_wxyz(palm[:3, :3]),
                device="cuda",
                dtype=torch.float32,
            ).reshape(1, 1, 1, 1, 4),
        )
        try:
            if linear_approach_axis is not None:
                criteria = ToolPoseCriteria.linear_motion(
                    axis=linear_approach_axis,
                    non_terminal_scale=1.0,
                    project_distance_to_goal=True,
                )
                planner.update_tool_pose_criteria(
                    {frame: criteria for frame in planner.tool_frames}
                )
            if disabled_collision_links:
                planner.disable_link_collision(list(disabled_collision_links))
            return planner.plan_pose(goal, start)
        finally:
            try:
                if disabled_collision_links:
                    planner.enable_link_collision(list(disabled_collision_links))
            finally:
                if linear_approach_axis is not None:
                    planner.update_tool_pose_criteria(
                        {frame: ToolPoseCriteria() for frame in planner.tool_frames}
                    )

    def _measure_reach_correction(
        self,
        *,
        planner: Any,
        failures: Sequence[G1GraspCandidateFailure],
        plannable_proposals: Sequence[tuple[int, GraspPoseCandidate]],
        robot_root_from_camera: np.ndarray,
        current_joint_positions: Mapping[str, float],
        hand: str,
    ) -> G1GraspReachCorrection | None:
        """Measure the median body-forward move that makes failed poses solvable."""

        attempted = [
            item
            for item in failures
            if (
                item.phase == "grasp"
                and item.outcome != "near_previous_retention_failure"
                and item.palm_goal_pelvis_m is not None
            )
        ]
        if not attempted:
            return None
        proposal_by_index = dict(plannable_proposals)
        ordered_failures = sorted(
            attempted,
            key=lambda item: (
                -float(proposal_by_index[item.pose_index].confidence),
                item.pose_index,
            ),
        )[:G1_GRASP_REACH_PROBE_CANDIDATE_LIMIT]
        try:
            import torch
            from curobo.types import JointState

            start = JointState.from_position(
                torch.tensor(
                    [
                        _planning_start_joint_positions(
                            current_joint_positions,
                            G1_MANIPULATION_JOINT_NAMES[hand],
                        )
                    ],
                    device="cuda",
                    dtype=torch.float32,
                ),
                joint_names=list(G1_MANIPULATION_JOINT_NAMES[hand]),
            )
        except (ImportError, RuntimeError) as exc:
            return G1GraspReachCorrection.unavailable(reason=str(exc))

        measured_samples: list[tuple[float, dict[str, Any]]] = []
        for failure in ordered_failures:
            candidate = proposal_by_index.get(failure.pose_index)
            if candidate is None or failure.palm_goal_pelvis_m is None:
                continue
            palm = robot_root_from_camera @ candidate.palm_pose_camera(hand=hand)
            try:
                distance = _probe_minimum_body_forward_correction(
                    planner,
                    palm_pose_robot=palm,
                    start=start,
                    plan_pose_phase=self._plan_pose_phase,
                    disabled_collision_links=(G1_GRASP_TARGET_ENCLOSING_LINKS[hand]),
                )
            except (ImportError, RuntimeError) as exc:
                return G1GraspReachCorrection.unavailable(reason=str(exc))
            if distance is None:
                continue
            measured_samples.append(
                (
                    float(distance),
                    _public_grasp_approach_descriptor(
                        palm_pose_robot=palm,
                        palm_goal_pelvis_m=failure.palm_goal_pelvis_m,
                    ),
                )
            )
        if not measured_samples:
            return G1GraspReachCorrection.not_found_within_budget()
        measured_samples.sort(key=lambda item: item[0])
        selected_distance, selected_approach = measured_samples[
            len(measured_samples) // 2
        ]
        distances = [item[0] for item in measured_samples]
        return G1GraspReachCorrection.measured(
            selected_distance,
            selected_approach,
            sample_count=len(measured_samples),
            minimum_body_forward_m=distances[0],
            maximum_body_forward_m=distances[-1],
        )

    def plan_carry(
        self,
        *,
        hand: str,
        current_joint_positions: Mapping[str, float],
        current_body_height_command_m: float,
        closing_palm_pose_robot: np.ndarray,
    ) -> G1CarryPlan:
        """Lift the retained object clear of the source support.

        Candidate selection probes this solver from a predicted closing state.
        After real contact, execution solves again from the measured state;
        the predicted trajectory is never executed as contact evidence. The
        lift preserves the closing orientation until the object clears its
        support. Isaac then restores the standing locomotion upper body.
        """

        _require_curobo_v2()
        normalized_hand = str(hand).strip().lower()
        if normalized_hand != G1_MANIPULATION_HAND:
            raise ValueError("G1 carry planning supports only the right hand.")
        missing = [
            name
            for name in G1_CUROBO_JOINT_NAMES
            if name not in current_joint_positions
        ]
        if missing:
            raise ValueError(f"G1 carry start is missing joints: {missing!r}.")
        current_joint_positions = {
            str(name): float(value) for name, value in current_joint_positions.items()
        }
        planner = self._create_hand_planner(
            hand=normalized_hand,
            current_joint_positions=current_joint_positions,
            current_body_height_command_m=current_body_height_command_m,
            # The target is retained by the hand from here on, so it is no
            # longer an independent body the arm has to avoid. The support
            # slab is omitted for the same reason: the retraction starts with
            # fingers at the surface.
            target_obstacle=None,
            support_obstacle=None,
            selected_hand_joint_positions=tuple(
                float(current_joint_positions[name])
                for name in DEX3_JOINT_NAMES[normalized_hand]
            ),
        )
        try:
            import torch
            from curobo.types import JointState

            names = G1_MANIPULATION_JOINT_NAMES[normalized_hand]
            start = JointState.from_position(
                torch.tensor(
                    [
                        _planning_start_joint_positions(
                            current_joint_positions,
                            names,
                        )
                    ],
                    device="cuda",
                    dtype=torch.float32,
                ),
                joint_names=list(names),
            )
            closing_palm = _rigid_transform(closing_palm_pose_robot)
            carry_palm = np.array(closing_palm, copy=True)
            carry_offset = g1_carry_palm_goal_offset_pelvis_root_m(
                waist_yaw_rad=float(current_joint_positions["waist_yaw_joint"]),
                waist_pitch_rad=float(current_joint_positions["waist_pitch_joint"]),
                retraction_m=G1_GRASP_CARRY_RETRACTION_M,
                world_lift_m=self.carry_lift_distance_m,
            )
            carry_palm[:3, 3] += np.asarray(carry_offset)
            contact_links = [
                f"{normalized_hand}_hand_thumb_2_link",
                f"{normalized_hand}_hand_middle_1_link",
                f"{normalized_hand}_hand_index_1_link",
            ]
            retraction_started = time.monotonic()
            retraction_result = self._plan_pose_phase(
                planner=planner,
                palm_pose_robot=carry_palm,
                start=start,
                disabled_collision_links=contact_links,
            )
            self._record_planning_stage(
                "carry_retraction_reach",
                time.monotonic() - retraction_started,
            )
            if not _motion_succeeded(retraction_result):
                raise G1CarryPlanningInfeasible(
                    "cuRoboV2 found no self-collision-free support-clearance lift "
                    "from the measured retained pose: "
                    f"{_motion_failure_status(retraction_result)}.",
                    outcome=(
                        "ik_unreachable"
                        if retraction_result is None
                        else "trajectory_infeasible"
                    ),
                )
            pelvis_coordinates = G1PelvisHeightCoordinates(
                current_model_height_m=float(
                    current_joint_positions[G1_PELVIS_HEIGHT_JOINT]
                ),
                current_command_height_m=float(current_body_height_command_m),
            )
            retraction_positions = _control_trajectory(
                retraction_result.interpolated_trajectory,
                retraction_result.interpolated_last_tstep,
                hand=normalized_hand,
                pelvis_coordinates=pelvis_coordinates,
            )
            return G1CarryPlan(
                hand=normalized_hand,
                coordinate_names=G1_MANIPULATION_COORDINATE_NAMES[normalized_hand],
                positions=tuple(retraction_positions),
                trajectory_dt_s=float(planner.trajopt_solver.config.interpolation_dt),
                retraction_step_count=len(retraction_positions),
            )
        finally:
            del planner
            _release_curobo_cuda_memory()


class G1GraspPlannerClient:
    """Run cuRoboV2 outside Isaac's process and validate its JSON result."""

    def __init__(
        self,
        *,
        python_executable: Path | None = None,
        worker_script: Path | None = None,
        timeout_s: float = G1_GRASP_PLANNING_TIMEOUT_S,
    ) -> None:
        # Python environment containing the pinned cuRoboV2 checkout.
        self.python_executable = (
            Path(python_executable or sys.executable).expanduser().resolve()
        )
        repository_root = Path(__file__).resolve().parents[3]
        # Small process entry point; it never imports Isaac or its Warp build.
        self.worker_script = (
            Path(
                worker_script
                or repository_root / "tarsphere/infra/manipulation/worker.py"
            )
            .expanduser()
            .resolve()
        )
        self.timeout_s = float(timeout_s)
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError("G1 grasp worker timeout must be finite and positive.")
        if not self.python_executable.is_file():
            raise FileNotFoundError(
                f"cuRoboV2 Python executable not found: {self.python_executable}"
            )
        if not self.worker_script.is_file():
            raise FileNotFoundError(
                f"G1 cuRoboV2 worker not found: {self.worker_script}"
            )

    def plan(
        self,
        *,
        candidates: Sequence[GraspPoseCandidate],
        robot_root_from_camera: np.ndarray,
        current_joint_positions: Mapping[str, float],
        current_body_height_command_m: float,
        hand: str,
        grasp_target_object_id: str,
        source_support_object_id: str,
        target_obstacle: TargetObjectObstacle,
        support_obstacle: SupportSurfaceObstacle,
        exclude_pose_indices: Sequence[int] = (),
    ) -> G1GraspPlan:
        """Serialize one request and return its validated closing reach."""

        request = {
            "operation": "grasp",
            "candidates": [grasp_pose_payload(candidate) for candidate in candidates],
            "robot_root_from_camera": np.asarray(
                robot_root_from_camera,
                dtype=np.float64,
            ).tolist(),
            "current_joint_positions": {
                str(name): float(value)
                for name, value in current_joint_positions.items()
            },
            "hand": str(hand),
            "current_body_height_command_m": float(current_body_height_command_m),
            "grasp_target_object_id": str(grasp_target_object_id),
            "source_support_object_id": str(source_support_object_id),
            "target_obstacle": target_object_obstacle_payload(target_obstacle),
            "support_surface_obstacle": support_surface_obstacle_payload(
                support_obstacle
            ),
            "exclude_pose_indices": [int(index) for index in exclude_pose_indices],
        }
        payload = self._run_worker(request)
        return grasp_planning_result_from_payload(payload)

    def plan_carry(
        self,
        *,
        hand: str,
        current_joint_positions: Mapping[str, float],
        current_body_height_command_m: float,
        closing_palm_pose_robot: np.ndarray,
    ) -> G1CarryPlan:
        """Plan the independent post-contact lift off the source support."""

        payload = self._run_worker(
            {
                "operation": "carry",
                "hand": str(hand),
                "current_joint_positions": {
                    str(name): float(value)
                    for name, value in current_joint_positions.items()
                },
                "current_body_height_command_m": float(current_body_height_command_m),
                "closing_palm_pose_robot": np.asarray(
                    closing_palm_pose_robot,
                    dtype=np.float64,
                ).tolist(),
            },
        )
        return carry_planning_result_from_payload(payload)

    def _run_worker(
        self,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Run one typed planning request in the isolated cuRobo process.

        Successful requests are removed. Every infeasible grasp or carry
        search, timeout, or worker failure retains its exact request and
        output under the bounded diagnostics root for standalone reproduction.
        """

        directory: Path | None = None
        preserve_directory = False
        root: Path | None = None
        try:
            root = _preserved_worker_root()
            directory = Path(
                tempfile.mkdtemp(
                    prefix="tarsphere_g1_curobo_",
                    dir=str(root),
                )
            )
            request_path = directory / "request.json"
            result_path = directory / "result.json"
            stages_path = directory / G1_GRASP_WORKER_STAGES_FILENAME
            request_path.write_text(
                json.dumps(request, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            command = [
                str(self.python_executable),
                str(self.worker_script),
                "--request",
                str(request_path),
                "--result",
                str(result_path),
            ]
            worker_environment = self._clean_environment()
            worker_gpu = worker_environment.get("CUDA_VISIBLE_DEVICES", "default")
            # Explicit demo isolation exposes a physical GPU as CUDA ordinal
            # zero through its UUID. Keep the same physical-index lock used by
            # unmasked workers so both launch modes serialize on one card.
            lock_gpu = (
                worker_environment["TARSPHERE_KIT_ACTIVE_GPU"]
                if worker_environment.get("TARSPHERE_DEMO_GPU_UUID")
                else worker_gpu
            )
            lock_key = (
                "".join(
                    character if character.isalnum() else "_" for character in lock_gpu
                ).strip("_")
                or "default"
            )
            lock_path = Path(tempfile.gettempdir()) / (
                f"{G1_CUROBO_GPU_LOCK_PREFIX}_{lock_key}.lock"
            )
            try:
                with lock_path.open("a+", encoding="utf-8") as gpu_lock:
                    print(
                        f"[G1-GRASP] Waiting for cuRobo GPU {lock_gpu} lock "
                        f"(CUDA_VISIBLE_DEVICES={worker_gpu}).",
                        flush=True,
                    )
                    fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_EX)
                    try:
                        print(
                            f"[G1-GRASP] Acquired cuRobo GPU {lock_gpu} lock "
                            f"(CUDA_VISIBLE_DEVICES={worker_gpu}).",
                            flush=True,
                        )
                        completed = subprocess.run(
                            command,
                            capture_output=True,
                            text=True,
                            env=worker_environment,
                            timeout=self.timeout_s,
                        )
                    finally:
                        fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_UN)
            except subprocess.TimeoutExpired as exc:
                preserve_directory = True
                _preserve_worker_output(directory, exc.stdout, exc.stderr)
                raise G1GraspWorkerTimeout(
                    timeout_s=self.timeout_s,
                    preserved_request_dir=directory,
                    completed_stages=_read_worker_stage_timings(stages_path),
                ) from exc
            if completed.returncode != 0:
                preserve_directory = True
                _preserve_worker_output(
                    directory,
                    completed.stdout,
                    completed.stderr,
                )
                detail = (completed.stderr or completed.stdout or "").strip()
                raise G1GraspWorkerFailure(
                    "cuRoboV2 G1 worker failed outside Isaac: "
                    f"{detail[-2000:] or 'no diagnostic output'} "
                    f"Request preserved at {directory}.",
                    returncode=completed.returncode,
                )
            if not result_path.is_file():
                preserve_directory = True
                _preserve_worker_output(
                    directory,
                    completed.stdout,
                    completed.stderr,
                )
                raise G1GraspWorkerFailure(
                    "cuRoboV2 G1 worker exited without a result payload. "
                    f"Request preserved at {directory}.",
                    returncode=completed.returncode,
                )
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                preserve_directory = True
                _preserve_worker_output(
                    directory,
                    completed.stdout,
                    completed.stderr,
                )
                raise G1GraspWorkerFailure(
                    "cuRoboV2 G1 worker returned unreadable JSON. "
                    f"Request preserved at {directory}.",
                    returncode=completed.returncode,
                ) from exc
            if not isinstance(payload, dict):
                preserve_directory = True
                _preserve_worker_output(
                    directory,
                    completed.stdout,
                    completed.stderr,
                )
                raise G1GraspWorkerFailure(
                    "G1 worker result must be a JSON object. "
                    f"Request preserved at {directory}."
                )
            for line in (completed.stdout or "").splitlines():
                if line.startswith("[G1-SUPPORT]"):
                    print(line, flush=True)
            if payload.get("status") == "infeasible":
                preserve_directory = True
                _preserve_worker_output(
                    directory,
                    completed.stdout,
                    completed.stderr,
                )
                print(
                    f"[G1-GRASP] Infeasible {request['operation']} request preserved at {directory}.",
                    flush=True,
                )
                message = payload.get("message")
                payload = dict(payload)
                payload["message"] = (
                    f"{message} Request preserved at {directory}."
                    if isinstance(message, str) and message.strip()
                    else f"Request preserved at {directory}."
                )
                return payload
            return payload
        finally:
            if directory is not None and directory.exists() and not preserve_directory:
                shutil.rmtree(directory, ignore_errors=True)
            if preserve_directory and root is not None:
                _bound_preserved_worker_directories(root)

    def _clean_environment(self) -> dict[str, str]:
        """Remove Isaac's Python and Warp paths from the cuRoboV2 worker."""

        environment_root = self.python_executable.parent.parent
        repository_root = Path(__file__).resolve().parents[3]
        environment = {
            "HOME": os.environ.get("HOME", str(Path.home())),
            "PATH": ":".join(
                (
                    str(self.python_executable.parent),
                    "/usr/local/cuda/bin",
                    "/usr/bin",
                    "/bin",
                )
            ),
            # Bind the worker to this frozen checkout's cuRobo source too,
            # rather than a globally editable install from another run.
            "PYTHONPATH": os.pathsep.join(
                (str(repository_root), str(repository_root / "third_party" / "curobo"))
            ),
            "PYTHONNOUSERSITE": "1",
            "LD_LIBRARY_PATH": ":".join(
                (
                    str(environment_root / "lib"),
                    "/usr/local/cuda/lib64",
                    "/usr/lib/x86_64-linux-gnu",
                )
            ),
            "CUDA_HOME": "/usr/local/cuda",
        }
        worker_cuda = os.environ.get(
            "TARSPHERE_WORKER_CUDA_VISIBLE_DEVICES",
            "",
        ).strip()
        demo_mapping = resolve_demo_gpu_mapping()
        if demo_mapping is not None:
            physical_gpu = str(demo_mapping["physical_gpu"])
            gpu_uuid = demo_mapping["uuid"]
            if worker_cuda not in ("", physical_gpu, gpu_uuid):
                raise RuntimeError(
                    "The isolated demo cuRobo GPU must match the verified "
                    "Isaac physical GPU; TARSPHERE_WORKER_CUDA_VISIBLE_DEVICES "
                    "must be its physical index or the same full GPU UUID."
                )
            # CUDA enumerates this one verified UUID as cuda:0. These explicit
            # identity fields also preserve physical lock ownership; no Isaac
            # Python/Warp paths are inherited by the clean worker.
            environment["CUDA_VISIBLE_DEVICES"] = gpu_uuid
            environment["TARSPHERE_DEMO_GPU_UUID"] = gpu_uuid
            environment["TARSPHERE_KIT_ACTIVE_GPU"] = physical_gpu
            return environment
        isaac_gpu = os.environ.get("TARSPHERE_KIT_ACTIVE_GPU", "").strip()
        if isaac_gpu and worker_cuda and worker_cuda != isaac_gpu:
            raise RuntimeError(
                "The Isaac physical GPU and isolated cuRobo GPU must match; "
                "set TARSPHERE_WORKER_CUDA_VISIBLE_DEVICES to "
                "TARSPHERE_KIT_ACTIVE_GPU for this evaluation lane."
            )
        # Scene-server launches also pin Kit without going through the formal
        # scheduler. They must not silently fall back to the shared GPU lock.
        worker_cuda = worker_cuda or isaac_gpu
        if worker_cuda:
            environment["CUDA_VISIBLE_DEVICES"] = worker_cuda
        elif "CUDA_VISIBLE_DEVICES" in os.environ:
            environment["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
        return environment


def plan_g1_grasp(
    candidates: Sequence[GraspPoseCandidate],
    *,
    robot_root_from_camera: np.ndarray,
    current_joint_positions: Mapping[str, float],
    current_body_height_command_m: float,
    grasp_target_object_id: str,
    source_support_object_id: str,
    target_obstacle: TargetObjectObstacle,
    support_obstacle: SupportSurfaceObstacle,
    hand: str = G1_MANIPULATION_HAND,
    planner_client: G1GraspPlannerClient | None = None,
    exclude_pose_indices: Sequence[int] = (),
) -> G1GraspPlan:
    """Plan one closing reach for one ``grasp_object`` action."""

    if not candidates:
        raise ValueError("G1 grasp planning requires GraspGenX poses.")
    client = planner_client or G1GraspPlannerClient()
    return client.plan(
        candidates=candidates,
        robot_root_from_camera=robot_root_from_camera,
        current_joint_positions=current_joint_positions,
        current_body_height_command_m=current_body_height_command_m,
        hand=hand,
        grasp_target_object_id=grasp_target_object_id,
        source_support_object_id=source_support_object_id,
        target_obstacle=target_obstacle,
        support_obstacle=support_obstacle,
        exclude_pose_indices=exclude_pose_indices,
    )


def grasp_plan_payload(plan: G1GraspPlan) -> dict[str, Any]:
    """Serialize one G1 plan without CUDA tensors or Isaac objects."""

    return {
        "selected_pose_index": plan.selected_pose_index,
        "selected_confidence": plan.selected_confidence,
        "hand": plan.hand,
        "grasp_center_robot_m": plan.grasp_center_robot_m,
        "coordinate_names": plan.coordinate_names,
        "reach_positions": plan.reach_positions,
        "trajectory_dt_s": plan.trajectory_dt_s,
    }


def grasp_plan_from_payload(payload: Mapping[str, Any]) -> G1GraspPlan:
    """Validate the worker payload before it enters Isaac execution."""

    def trajectory(name: str) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(float(value) for value in waypoint) for waypoint in payload[name]
        )

    return G1GraspPlan(
        selected_pose_index=int(payload["selected_pose_index"]),
        selected_confidence=float(payload["selected_confidence"]),
        hand=str(payload["hand"]),
        grasp_center_robot_m=tuple(
            float(value) for value in payload["grasp_center_robot_m"]
        ),
        coordinate_names=tuple(str(name) for name in payload["coordinate_names"]),
        reach_positions=trajectory("reach_positions"),
        trajectory_dt_s=float(payload["trajectory_dt_s"]),
    )


def carry_plan_payload(plan: G1CarryPlan) -> dict[str, Any]:
    """Serialize the independent post-contact carry trajectory."""

    return {
        "hand": plan.hand,
        "coordinate_names": plan.coordinate_names,
        "positions": plan.positions,
        "trajectory_dt_s": plan.trajectory_dt_s,
        "retraction_step_count": plan.retraction_step_count,
    }


def carry_plan_from_payload(payload: Mapping[str, Any]) -> G1CarryPlan:
    """Validate one carry trajectory returned by the cuRobo worker."""

    positions = tuple(
        tuple(float(value) for value in position) for position in payload["positions"]
    )
    retraction_step_count = payload.get("retraction_step_count")
    if retraction_step_count is None:
        raise ValueError("A G1 carry payload must include retraction_step_count.")
    return G1CarryPlan(
        hand=str(payload["hand"]),
        coordinate_names=tuple(str(name) for name in payload["coordinate_names"]),
        positions=positions,
        trajectory_dt_s=float(payload["trajectory_dt_s"]),
        retraction_step_count=int(retraction_step_count),
    )


def carry_planning_success_payload(plan: G1CarryPlan) -> dict[str, Any]:
    """Serialize one successful post-contact carry plan."""

    return {"status": "ok", "plan": carry_plan_payload(plan)}


def carry_planning_infeasible_payload(
    failure: G1CarryPlanningInfeasible,
) -> dict[str, str]:
    """Serialize a valid carry search with no collision-free result."""

    detail = str(failure).strip()
    if not detail:
        raise ValueError("An infeasible carry result requires a diagnostic.")
    return {"status": "infeasible", "message": detail, "outcome": failure.outcome}


def carry_planning_result_from_payload(
    payload: Mapping[str, Any],
) -> G1CarryPlan:
    """Parse the typed result of an independent carry-planning request."""

    status = payload.get("status")
    if status == "infeasible":
        message = payload.get("message")
        if not isinstance(message, str) or not message.strip():
            raise G1GraspWorkerFailure(
                "G1 worker infeasible carry result needs a diagnostic."
            )
        outcome = payload.get("outcome")
        if outcome not in ("ik_unreachable", "trajectory_infeasible"):
            raise G1GraspWorkerFailure("G1 worker carry failure needs a typed outcome.")
        raise G1CarryPlanningInfeasible(message.strip(), outcome=outcome)
    if status != "ok":
        raise G1GraspWorkerFailure(
            f"G1 worker returned unknown carry status: {status!r}."
        )
    try:
        return carry_plan_from_payload(payload["plan"])
    except (KeyError, TypeError, ValueError) as exc:
        raise G1GraspWorkerFailure("G1 worker returned an invalid carry plan.") from exc


def grasp_planning_success_payload(
    plan: G1GraspPlan,
) -> dict[str, Any]:
    """Serialize one closing reach for the worker boundary."""

    if not isinstance(plan, G1GraspPlan):
        raise TypeError("G1 worker output must contain one grasp plan.")
    return {"status": "ok", "plan": grasp_plan_payload(plan)}


def grasp_planning_infeasible_payload(
    failures: Sequence[G1GraspCandidateFailure],
    *,
    reach_correction: G1GraspReachCorrection | None = None,
) -> dict[str, Any]:
    """Serialize every measured candidate failure from a valid search."""

    rejected = tuple(failures)
    if not rejected:
        raise ValueError("An infeasible grasp result requires failures.")
    payload: dict[str, Any] = {
        "status": "infeasible",
        "failures": [failure.as_dict() for failure in rejected],
    }
    if reach_correction is not None:
        payload["reach_correction"] = reach_correction.as_dict()
    return payload


def grasp_planning_result_from_payload(
    payload: Mapping[str, Any],
) -> G1GraspPlan:
    """Parse the current typed result from the isolated worker."""

    if not isinstance(payload, Mapping):
        raise G1GraspWorkerFailure("G1 worker result must be a JSON object.")
    status = payload.get("status")
    if status == "infeasible":
        failures = payload.get("failures")
        if (
            not isinstance(failures, Sequence)
            or isinstance(
                failures,
                (str, bytes),
            )
            or not failures
        ):
            raise G1GraspWorkerFailure(
                "G1 worker infeasible result requires candidate failures."
            )
        try:
            rejected = tuple(
                G1GraspCandidateFailure.from_mapping(item) for item in failures
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise G1GraspWorkerFailure(
                "G1 worker returned an invalid candidate failure."
            ) from exc
        reach_correction = G1GraspReachCorrection.from_mapping(
            payload.get("reach_correction"),
        )
        raise G1GraspPlanningInfeasible(
            rejected,
            reach_correction=reach_correction,
        )
    if status != "ok":
        raise G1GraspWorkerFailure(
            f"G1 worker returned unknown result status: {status!r}."
        )

    try:
        return grasp_plan_from_payload(payload["plan"])
    except (KeyError, TypeError, ValueError) as exc:
        raise G1GraspWorkerFailure("G1 worker returned an invalid grasp plan.") from exc


def _preserved_worker_root() -> Path:
    """Return the single temp root that holds preserved failed requests."""

    root = Path(tempfile.gettempdir()) / G1_GRASP_PRESERVED_WORKER_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def _bound_preserved_worker_directories(root: Path) -> None:
    """Keep only the most recent preserved worker directories under ``root``."""

    children = [
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("tarsphere_g1_curobo_")
    ]
    children.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for stale in children[G1_GRASP_PRESERVED_WORKER_RETENTION:]:
        shutil.rmtree(stale, ignore_errors=True)


def _preserve_worker_output(
    directory: Path,
    stdout: Any,
    stderr: Any,
) -> None:
    """Keep captured worker streams beside the preserved failed request."""

    for name, stream in (("stdout.txt", stdout), ("stderr.txt", stderr)):
        if not stream:
            continue
        text = (
            stream.decode("utf-8", errors="replace")
            if isinstance(stream, bytes)
            else str(stream)
        )
        try:
            (directory / name).write_text(text, encoding="utf-8")
        except OSError:
            pass


def _read_worker_stage_timings(
    stages_path: Path,
) -> tuple[dict[str, Any], ...]:
    """Read the progressive stage records a worker completed before dying."""

    if not stages_path.is_file():
        return ()
    records: list[dict[str, Any]] = []
    try:
        for line in stages_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                records.append(record)
    except (OSError, json.JSONDecodeError):
        return tuple(records)
    return tuple(records)


def target_object_obstacle_in_planning_root(
    *,
    object_id: str,
    object_bbox: Sequence[Sequence[float]],
    robot_root_T_world: np.ndarray,
) -> TargetObjectObstacle:
    """Express the live target's world bounding box in the planner root frame.

    ``object_bbox`` is the measured world-axis-aligned box as ``(min, max)``.
    Its edge lengths are preserved and its world orientation is carried through
    as a quaternion, so the planner receives the measured box rather than an
    enclosing box refitted to the rotated planner axes.
    """

    bounds = np.asarray(object_bbox, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        raise ValueError(
            "A grasp-target bounding box must be a finite (min, max) pair of "
            f"world coordinates; got shape {bounds.shape}."
        )
    extents = bounds[1] - bounds[0]
    if np.any(extents <= 0.0):
        raise ValueError(
            f"Grasp target {object_id!r} has a degenerate live bounding box: "
            f"min={bounds[0].tolist()!r} max={bounds[1].tolist()!r}."
        )
    root_from_world = _rigid_transform(robot_root_T_world)
    centre_world = 0.5 * (bounds[0] + bounds[1])
    centre_root = root_from_world[:3, :3] @ centre_world + root_from_world[:3, 3]
    return TargetObjectObstacle(
        object_id=str(object_id),
        center_robot_m=tuple(float(value) for value in centre_root),
        extents_m=tuple(float(value) for value in extents),
        orientation_robot_wxyz=_rotation_to_quaternion_wxyz(root_from_world[:3, :3]),
    )


def support_surface_obstacle_in_planning_root(
    *,
    support_object_id: str,
    vertices_world_m: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    robot_root_T_world: np.ndarray,
) -> SupportSurfaceObstacle:
    """Rigidly transform the live closed contact mesh; never fit a box."""
    vertices = np.asarray(vertices_world_m, dtype=np.float64)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not np.all(np.isfinite(vertices))
    ):
        raise ValueError("Support contact mesh requires finite world vertices")
    root_from_world = _rigid_transform(robot_root_T_world)
    vertices_root = vertices @ root_from_world[:3, :3].T + root_from_world[:3, 3]
    return SupportSurfaceObstacle(
        object_id=str(support_object_id),
        vertices_robot_m=tuple(tuple(map(float, row)) for row in vertices_root),
        faces=tuple(tuple(row) for row in faces),
        normal_robot=tuple(map(float, root_from_world[:3, 2])),
    )


def target_object_obstacle_payload(
    obstacle: TargetObjectObstacle,
) -> dict[str, Any]:
    """Serialize the grasp-target obstacle for the isolated planning worker."""

    if not isinstance(obstacle, TargetObjectObstacle):
        raise TypeError(
            "A G1 grasp request carries the target obstacle as a "
            f"TargetObjectObstacle, not {type(obstacle).__name__}."
        )
    return {
        "object_id": str(obstacle.object_id),
        "center_robot_m": [float(value) for value in obstacle.center_robot_m],
        "extents_m": [float(value) for value in obstacle.extents_m],
        "orientation_robot_wxyz": [
            float(value) for value in obstacle.orientation_robot_wxyz
        ],
    }


def target_object_obstacle_from_payload(
    payload: Mapping[str, Any],
) -> TargetObjectObstacle:
    """Rebuild the grasp-target obstacle inside the planning worker."""

    return TargetObjectObstacle(
        object_id=str(payload["object_id"]),
        center_robot_m=tuple(float(value) for value in payload["center_robot_m"]),
        extents_m=tuple(float(value) for value in payload["extents_m"]),
        orientation_robot_wxyz=tuple(
            float(value) for value in payload["orientation_robot_wxyz"]
        ),
    )


def support_surface_obstacle_payload(
    obstacle: SupportSurfaceObstacle,
) -> dict[str, Any]:
    """Serialize the contact mesh for the isolated planning worker."""

    if not isinstance(obstacle, SupportSurfaceObstacle):
        raise TypeError(
            "A G1 grasp request carries the support mesh as a "
            f"SupportSurfaceObstacle, not {type(obstacle).__name__}."
        )
    return {
        "object_id": str(obstacle.object_id),
        "vertices_robot_m": [list(row) for row in obstacle.vertices_robot_m],
        "faces": [[int(index) for index in row] for row in obstacle.faces],
        "normal_robot": list(obstacle.normal_robot),
    }


def support_surface_obstacle_from_payload(
    payload: Mapping[str, Any],
) -> SupportSurfaceObstacle:
    """Rebuild only the current mesh contract; old cuboid payloads fail."""
    if set(payload) != {"object_id", "vertices_robot_m", "faces", "normal_robot"}:
        raise ValueError(
            "Support obstacle payload requires only the current contact mesh fields"
        )
    return SupportSurfaceObstacle(
        object_id=str(payload["object_id"]),
        vertices_robot_m=tuple(
            tuple(map(float, row)) for row in payload["vertices_robot_m"]
        ),
        faces=tuple(tuple(row) for row in payload["faces"]),
        normal_robot=tuple(map(float, payload["normal_robot"])),
    )


def _cuboid_from_obstacle(name: str, obstacle: TargetObjectObstacle):
    """Build one named cuRobo cuboid from a planner-root primitive."""

    from curobo.scene import Cuboid

    return Cuboid(
        name=name,
        dims=[float(value) for value in obstacle.extents_m],
        pose=[
            *(float(value) for value in obstacle.center_robot_m),
            *(float(value) for value in obstacle.orientation_robot_wxyz),
        ],
    )


def _load_world_obstacles(
    planner: Any,
    *,
    target: TargetObjectObstacle,
    support: SupportSurfaceObstacle,
    support_object_id: str,
) -> None:
    """Give the planner the grasp target and contact mesh together.

    ``update_world`` replaces the scene, so both obstacles must enter in one
    call; a second call would drop the first.
    """

    from curobo.scene import Mesh, Scene

    require_support_surface_obstacle(
        support,
        support_object_id=support_object_id,
    )
    planner.update_world(
        Scene(
            cuboid=[_cuboid_from_obstacle(G1_GRASP_TARGET_OBSTACLE_NAME, target)],
            mesh=[
                Mesh(
                    name=G1_SUPPORT_SURFACE_OBSTACLE_NAME,
                    vertices=[list(row) for row in support.vertices_robot_m],
                    faces=[[int(index) for index in row] for row in support.faces],
                    pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                )
            ],
        )
    )
    # Raw signed distance, not the collision checker's activated cost, is
    # required after restoring all hand links. Reuse its BVH across candidates.
    import trimesh
    from curobo._src.geom.sphere_fit.wp_mesh_query import WarpMeshQuery

    planner.tarsphere_support_distance_query = WarpMeshQuery(
        trimesh.Trimesh(
            vertices=support.vertices_robot_m, faces=support.faces, process=False
        ),
        planner.device_cfg.device,
    )
    planner.tarsphere_support_obstacle = support
    loaded = planner.scene_collision_checker.get_obstacle_names()
    missing = [
        name
        for name in (
            G1_GRASP_TARGET_OBSTACLE_NAME,
            G1_SUPPORT_SURFACE_OBSTACLE_NAME,
        )
        if name not in loaded
    ]
    if missing:
        raise RuntimeError(
            "cuRoboV2 accepted the closing-reach world obstacles but did not "
            f"load {missing!r}: obstacles={sorted(loaded)!r}. Planning without "
            "both obstacles would drive the palm into the target or through the "
            "desk."
        )


def _require_graspable_target_obstacle(
    obstacle: Any,
    *,
    grasp_target_object_id: str,
) -> TargetObjectObstacle:
    """Return the target obstacle, refusing anything the Dex3 cannot grasp.

    The identity check is the contract; this size check is the guard that keeps
    a support or a furniture bounding box from entering the planner under a
    target-shaped label. A body the mounted hand cannot span is not a grasp
    target, and its bounding box would occupy empty volume.
    """

    target = require_target_object_obstacle(
        obstacle,
        grasp_target_object_id=grasp_target_object_id,
    )
    span = target.horizontal_span_m
    if span > TARGET_SCREENING_MAX_GRASP_SPAN_M:
        raise RuntimeError(
            f"{G1_SCENE_AABB_OBSTACLES_FORBIDDEN} Obstacle "
            f"{target.object_id!r} spans {span:.4f} m horizontally, beyond the "
            f"{TARGET_SCREENING_MAX_GRASP_SPAN_M:.4f} m the mounted Dex3 can "
            "grasp, so it is not a grasp target."
        )
    return target


def _curobo_interpolation_buffer_size(
    *,
    horizon: int,
    maximum_dt: float,
    interpolation_dt: float,
) -> int:
    """Bound padding, not trajectory length, from cuRobo's resolved time limits.

    cuRobo clamps optimized dt before interpolation. Its B-spline knot dt
    includes the substeps already counted in this dense transition horizon.
    Allow one extra output sample per dense interval for integer rounding,
    plus the final sample. This conservatively covers the complete trajectory
    without relying on dynamic buffer growth (disabled in the pinned build).
    The 1,000-sample floor covers the current 81-step horizon with margin;
    larger configured horizons/time limits grow the allocation, never truncate.
    """

    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or horizon < 2
        or not math.isfinite(maximum_dt)
        or not math.isfinite(interpolation_dt)
        or maximum_dt <= 0
        or interpolation_dt <= 0
    ):
        raise ValueError(
            "cuRobo interpolation requires a horizon >= 2 and positive finite dt."
        )
    return max(1000, math.ceil((horizon - 1) * (maximum_dt / interpolation_dt + 1)) + 1)


def _release_curobo_cuda_memory() -> None:
    """Release the isolated worker's planner and cached CUDA memory."""

    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _predicted_closing_joint_state(
    *,
    planned: G1GraspPlan,
    current_joint_positions: Mapping[str, float],
    current_body_height_command_m: float,
) -> tuple[dict[str, float], float]:
    """Map a controller waypoint back to model joints for carry-only preflight."""
    predicted = dict(current_joint_positions)
    endpoint = planned.reach_positions[-1]
    if not math.isclose(
        float(endpoint[0]),
        float(current_body_height_command_m),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "Closing preflight cannot assume an unmeasured pelvis-height change."
        )
    for name, value in zip(
        G1_MANIPULATION_JOINT_NAMES[planned.hand], endpoint[1:], strict=True
    ):
        predicted[name] = float(value)
    predicted.update(
        zip(
            DEX3_JOINT_NAMES[planned.hand],
            DEX3_GRASP_JOINT_POSITIONS[planned.hand],
            strict=True,
        )
    )
    return predicted, float(endpoint[0])


def g1_carry_terminal_joint_limit_margin_rad(plan: G1CarryPlan) -> float:
    """Return the smallest hard-limit margin after support-clearance lift.

    The endpoint is the candidate-specific support-clearance lift; neutral
    locomotion restoration is executed afterwards by the simulator controller.
    """

    terminal = dict(
        zip(
            plan.coordinate_names,
            plan.positions[plan.retraction_step_count - 1],
            strict=True,
        )
    )
    waist_margins = [
        G1_MANIPULATION_WAIST_LIMITS_RAD[name] - abs(float(terminal[name]))
        for name in G1_WAIST_JOINT_NAMES
    ]
    if plan.hand == "right":
        arm_names = G1_RIGHT_ARM_JOINT_NAMES
        lower, upper = G1_RIGHT_ARM_LOWER, G1_RIGHT_ARM_UPPER
    else:
        arm_names = G1_LEFT_ARM_JOINT_NAMES
        lower, upper = G1_LEFT_ARM_LOWER, G1_LEFT_ARM_UPPER
    arm_margins = [
        min(float(terminal[name]) - float(low), float(high) - float(terminal[name]))
        for name, low, high in zip(arm_names, lower, upper, strict=True)
    ]
    return min(*waist_margins, *arm_margins)


def _control_trajectory(
    trajectory: Any,
    last_timestep: Any,
    *,
    hand: str,
    pelvis_coordinates: G1PelvisHeightCoordinates,
) -> tuple[tuple[float, ...], ...]:
    if trajectory is None or last_timestep is None:
        raise RuntimeError("cuRoboV2 returned an incomplete G1 trajectory.")
    positions = trajectory.position.detach().cpu().numpy()
    positions = positions.reshape(-1, positions.shape[-1])
    valid_length = int(last_timestep.reshape(-1)[0].item())
    if not 1 <= valid_length <= positions.shape[0]:
        raise RuntimeError(
            "cuRoboV2 returned an invalid G1 trajectory length: "
            f"{valid_length} for buffer {positions.shape[0]}."
        )
    names = tuple(str(name) for name in trajectory.joint_names)
    indices = []
    # cuRobo expands locked joints in its result; parse those names rather
    # than reshaping the result using the smaller active planning dimension.
    for name in (G1_PELVIS_HEIGHT_JOINT, *G1_MANIPULATION_JOINT_NAMES[hand]):
        if name not in names:
            raise RuntimeError(
                f"cuRoboV2 trajectory is missing G1 manipulation coordinate {name!r}."
            )
        indices.append(names.index(name))
    output = []
    for row in positions[:valid_length]:
        values = [float(row[index]) for index in indices]
        values[0] = pelvis_coordinates.command_height_m(values[0])
        output.append(tuple(values))
    return tuple(output)


def _minimum_support_clearance_for_trajectory(
    *,
    planner: Any,
    trajectory: Any,
    last_timestep: Any,
    support_obstacle: SupportSurfaceObstacle,
) -> float:
    """Check a target-aware plan against the support with all links enabled."""

    if trajectory is None or last_timestep is None:
        raise RuntimeError(
            "cuRoboV2 returned no trajectory for support-clearance validation."
        )
    import torch
    from curobo.types import JointState

    positions = trajectory.position
    positions = positions.reshape(-1, positions.shape[-1])
    valid_length = int(last_timestep.reshape(-1)[0].item())
    if not 1 <= valid_length <= positions.shape[0]:
        raise RuntimeError(
            "cuRoboV2 returned an invalid support-validation trajectory "
            f"length: {valid_length} for buffer {positions.shape[0]}."
        )
    names = [str(name) for name in trajectory.joint_names]
    expected = list(planner.kinematics.joint_names)
    state = JointState.from_position(
        positions[:valid_length].unsqueeze(0),
        joint_names=names,
    )
    if names != expected:
        state = state.reorder(expected)
    with torch.no_grad():
        kinematics = planner.compute_kinematics(state)
    if kinematics.robot_spheres is None:
        raise RuntimeError("cuRoboV2 returned no robot spheres for support validation.")
    if planner.tarsphere_support_obstacle != support_obstacle:
        raise RuntimeError("Support distance query is not bound to this planning mesh")
    spheres = kinematics.robot_spheres.reshape(-1, 4)
    if (
        not len(spheres)
        or not bool(torch.isfinite(spheres).all())
        or bool((spheres[:, 3] < 0.0).any())
    ):
        raise RuntimeError(
            "All restored robot collision spheres must be finite with nonnegative radii"
        )
    distances, _ = planner.tarsphere_support_distance_query.query_sdf(
        spheres[:, :3].to(dtype=torch.float32)
    )
    clearance = distances - spheres[:, 3]
    if not bool(torch.isfinite(clearance).all()):
        raise RuntimeError("Non-finite signed support clearance")
    return float(clearance.min().item())


def _translate_palm_body_forward_toward_robot(
    palm_pose_robot: np.ndarray,
    translation_m: float,
) -> np.ndarray:
    """Move one palm goal toward the robot along body-forward (+X)."""

    translated = np.array(palm_pose_robot, copy=True)
    translated[0, 3] -= float(translation_m)
    return translated


def _probe_minimum_body_forward_correction(
    planner: Any,
    *,
    palm_pose_robot: np.ndarray,
    start: Any,
    plan_pose_phase: Callable[..., Any],
    disabled_collision_links: Sequence[str],
) -> float | None:
    """Return the smallest body-forward translation that makes one pose solvable."""

    coarse_step = G1_GRASP_REACH_PROBE_COARSE_STEP_M
    coarse_max = G1_GRASP_REACH_PROBE_COARSE_MAX_M
    fine_step = G1_GRASP_REACH_PROBE_FINE_STEP_M

    def succeeds_at(distance_m: float) -> bool:
        translated = _translate_palm_body_forward_toward_robot(
            palm_pose_robot,
            distance_m,
        )
        result = plan_pose_phase(
            planner=planner,
            palm_pose_robot=translated,
            start=start,
            disabled_collision_links=disabled_collision_links,
        )
        return _motion_succeeded(result)

    if succeeds_at(0.0):
        return 0.0

    coarse_steps = int(round(coarse_max / coarse_step))
    previous_failure_m = 0.0
    first_success_m: float | None = None
    for step_index in range(1, coarse_steps + 1):
        distance_m = step_index * coarse_step
        if succeeds_at(distance_m):
            first_success_m = distance_m
            break
        previous_failure_m = distance_m

    if first_success_m is None:
        return None

    minimum_m = first_success_m
    fine_steps = int(round((first_success_m - previous_failure_m) / fine_step))
    for step_index in range(1, fine_steps):
        probe_m = previous_failure_m + step_index * fine_step
        if succeeds_at(probe_m):
            minimum_m = probe_m
            break
    return minimum_m


def _public_grasp_approach_descriptor(
    *,
    palm_pose_robot: np.ndarray,
    palm_goal_pelvis_m: tuple[float, float, float],
) -> dict[str, Any]:
    """Describe one failed grasp approach without private pose indices."""

    approach_axis = np.asarray(palm_pose_robot, dtype=np.float64)[:3, 2]
    downward = float(-approach_axis[2])
    kind = "top_down" if downward > 0.7 else "side"
    forward_m, lateral_m, vertical_m = palm_goal_pelvis_m
    return {
        "kind": kind,
        "palm_goal_pelvis_m": {
            "forward_m": round(float(forward_m), 2),
            "lateral_m": round(float(lateral_m), 2),
            "vertical_m": round(float(vertical_m), 2),
        },
    }


def _motion_succeeded(result: Any) -> bool:
    """Return whether one cuRoboV2 pose-planning phase has a valid result."""

    return bool(
        result is not None
        and result.success is not None
        and result.success.any().item()
    )


def _minimum_tensor_value(value: Any) -> float | None:
    """Return one finite non-negative solver minimum when it exists."""

    if value is None or not value.numel():
        return None
    minimum = float(value.min().item())
    return minimum if math.isfinite(minimum) and minimum >= 0.0 else None


def _motion_failure_measurements(result: Any) -> dict[str, Any]:
    """Extract only diagnostics actually returned by one cuRobo result.

    cuRoboV2 returns ``None`` after every IK seed fails. In that case the only
    supported conclusion is ``ik_unreachable``; inventing a position error or
    collision cause would mislead the Agent. A non-null failed trajectory owns
    error tensors, feasibility, and named constraint violations.
    """

    if result is None:
        return {
            "outcome": "ik_unreachable",
            "minimum_position_error_m": None,
            "minimum_orientation_error_rad": None,
            "minimum_cspace_error": None,
            "any_trajectory_feasible": None,
            "constraint_violations": (),
        }
    position_error = getattr(result, "position_error", None)
    rotation_error = getattr(result, "rotation_error", None)
    cspace_error = getattr(result, "cspace_error", None)
    feasible = getattr(result, "feasible", None)
    any_feasible = (
        bool(feasible.any().item())
        if feasible is not None and feasible.numel()
        else None
    )
    violations: list[str] = []
    for label, metrics in (
        ("trajectory", getattr(result, "metrics", None)),
        ("interpolated", getattr(result, "interpolated_metrics", None)),
    ):
        if metrics is None:
            continue
        costs = metrics.costs_and_constraints
        for collection in (
            costs.constraints,
            costs.hybrid_costs_constraints,
        ):
            for name, values in zip(
                collection.names,
                collection.values,
                strict=True,
            ):
                maximum = float(values.max().item())
                if maximum > 0.0:
                    violations.append(f"{label}:{name}={maximum:.4g}")
    return {
        "outcome": "trajectory_infeasible",
        "minimum_position_error_m": _minimum_tensor_value(position_error),
        "minimum_orientation_error_rad": _minimum_tensor_value(rotation_error),
        "minimum_cspace_error": _minimum_tensor_value(cspace_error),
        "any_trajectory_feasible": any_feasible,
        "constraint_violations": tuple(violations),
    }


def _grasp_candidate_failure(
    *,
    pose_index: int,
    hand: str,
    phase: str,
    result: Any,
    palm_goal_pelvis_m: tuple[float, float, float] | None = None,
) -> G1GraspCandidateFailure:
    """Bind one cuRobo phase result to its visual candidate and hand."""

    return G1GraspCandidateFailure(
        pose_index=pose_index,
        hand=hand,
        phase=phase,
        palm_goal_pelvis_m=palm_goal_pelvis_m,
        **_motion_failure_measurements(result),
    )


def _motion_failure_status(result: Any) -> str:
    """Format private carry-planning diagnostics for fail-loud logs."""

    measured = _motion_failure_measurements(result)
    if measured["outcome"] == "ik_unreachable":
        return "ik_unreachable"
    violation_summary = "|".join(measured["constraint_violations"][:4]) or "unknown"

    def display(name: str) -> float:
        value = measured[name]
        return math.nan if value is None else float(value)

    return (
        "trajectory_infeasible("
        f"position_error={display('minimum_position_error_m'):.4f},"
        f"rotation_error={display('minimum_orientation_error_rad'):.4f},"
        f"cspace_error={display('minimum_cspace_error'):.4f},"
        f"any_feasible={measured['any_trajectory_feasible']},"
        f"violations={violation_summary})"
    )


def _summarize_palm_goals_pelvis_m(
    failures: Sequence[G1GraspCandidateFailure],
) -> dict[str, Any] | None:
    """Compact min/max envelope of measured palm goals in the pelvis frame."""

    goals = [
        item.palm_goal_pelvis_m
        for item in failures
        if item.palm_goal_pelvis_m is not None
    ]
    if not goals:
        return None
    forwards = [goal[0] for goal in goals]
    laterals = [goal[1] for goal in goals]
    verticals = [goal[2] for goal in goals]
    return {
        "forward_m": [min(forwards), max(forwards)],
        "lateral_m": [min(laterals), max(laterals)],
        "vertical_m": [min(verticals), max(verticals)],
        "sample_count": len(goals),
    }


def summarize_grasp_reachability(
    failures: Sequence[G1GraspCandidateFailure],
    *,
    reach_correction: G1GraspReachCorrection | None = None,
) -> dict[str, Any]:
    """Project private candidate failures into concise public evidence.

    Individual GraspGenX pose indices are controller internals. The Agent
    needs to know which hand was searched, which motion phase failed, whether
    the cause was absent IK, self-collision, trajectory infeasibility, or a
    closing-centre gate exclusion before planning, and where cuRobo was asked
    to place the palm relative to the pelvis. Raw solver costs remain private.
    When present, ``reach_correction`` carries a body-forward distance measured
    by live cuRobo re-planning at refusal time; it never exposes world
    coordinates or private pose indices.
    """

    rejected = tuple(failures)
    if not rejected:
        raise ValueError("Grasp reachability requires at least one failure.")
    if any(item.hand != G1_MANIPULATION_HAND for item in rejected):
        raise ValueError("Public G1 reachability may contain only right-hand failures.")
    failures_by_phase: dict[str, dict[str, int]] = {}
    for item in rejected:
        phase = failures_by_phase.setdefault(item.phase, {})
        if item.outcome == "near_previous_retention_failure":
            reason = "closing_centre_excluded"
        elif item.outcome == "trajectory_infeasible":
            reason = (
                "self_collision"
                if any(
                    "self_collision" in violation
                    for violation in item.constraint_violations
                )
                else "trajectory_infeasible"
            )
        else:
            reason = "no_ik_solution"
        phase[reason] = phase.get(reason, 0) + 1
    summary: dict[str, Any] = {
        "candidate_count": len({item.pose_index for item in rejected}),
        "failures": {
            phase: dict(sorted(outcomes.items()))
            for phase, outcomes in sorted(failures_by_phase.items())
        },
    }
    palm_goals = _summarize_palm_goals_pelvis_m(rejected)
    if palm_goals is not None:
        summary["palm_goal_pelvis_m"] = palm_goals
    if reach_correction is not None:
        summary["reach_correction"] = reach_correction.as_dict()
    return summary


def _format_pelvis_goal_range(
    label: str,
    values: Sequence[float],
    *,
    signed_hint: str,
) -> str:
    """Render one measured palm-goal axis span in the pelvis frame."""

    low = float(min(values))
    high = float(max(values))
    if abs(low - high) <= 1e-9:
        return f"{label}={low:+.2f} m ({signed_hint})"
    return f"{label}={low:+.2f}..{high:+.2f} m ({signed_hint})"


def format_reach_envelope_refusal_message(
    *,
    target_observation: Mapping[str, Any],
    right_hand: Mapping[str, Any],
) -> str:
    """Build the public reach-envelope refusal the Agent reads on the next step.

    The message carries only quantities already measured at refusal time:
    ego-camera target observation, cuRobo palm-goal summaries, failure counts,
    and any body-forward reach-correction probe. It never points the model at
    structured evidence it must look up elsewhere.
    """

    if not isinstance(target_observation, Mapping):
        raise TypeError("target_observation must be a mapping.")
    if not isinstance(right_hand, Mapping):
        raise TypeError("right_hand reachability evidence must be a mapping.")

    horizontal_distance_m = float(target_observation["horizontal_distance_m"])
    height_from_camera_m = float(target_observation["height_from_camera_m"])
    bearing_deg = float(target_observation["bearing_deg"])
    bearing_rad = math.radians(bearing_deg)
    target_forward_m = horizontal_distance_m * math.cos(bearing_rad)
    target_left_m = horizontal_distance_m * math.sin(bearing_rad)

    parts = [
        "The right-hand-only grasp skill is not reachable from the current "
        "body pose. The observed target is at horizontal distance "
        f"{horizontal_distance_m:.2f} m, height "
        f"{height_from_camera_m:+.2f} m from the ego camera, and bearing "
        f"{bearing_deg:+.1f}° (positive bearing is left), which is about "
        f"{target_forward_m:+.2f} m forward and {target_left_m:+.2f} m left "
        "relative to the ego heading.",
    ]

    candidate_count = int(right_hand["candidate_count"])
    failures = right_hand.get("failures") or {}
    if not isinstance(failures, Mapping):
        raise TypeError("right_hand.failures must be a mapping when present.")
    failure_phrases: list[str] = []
    for phase, outcomes in sorted(failures.items()):
        if not isinstance(outcomes, Mapping):
            raise TypeError("right_hand failure outcomes must be mappings.")
        for reason, count in sorted(outcomes.items()):
            failure_phrases.append(f"{phase}:{reason}={int(count)}")
    candidate_noun = "candidate" if candidate_count == 1 else "candidates"
    parts.append(
        f"cuRobo searched {candidate_count} closing-pose {candidate_noun} "
        f"and every plan failed ({', '.join(failure_phrases)})."
    )

    palm_goal = right_hand.get("palm_goal_pelvis_m")
    if palm_goal is not None:
        if not isinstance(palm_goal, Mapping):
            raise TypeError(
                "right_hand.palm_goal_pelvis_m must be a mapping when present."
            )
        forward = palm_goal["forward_m"]
        lateral = palm_goal["lateral_m"]
        vertical = palm_goal["vertical_m"]
        parts.append(
            "The measured palm goals relative to the pelvis spanned "
            + _format_pelvis_goal_range(
                "forward",
                (float(forward[0]), float(forward[1])),
                signed_hint="positive forward is ahead of the pelvis",
            )
            + ", "
            + _format_pelvis_goal_range(
                "lateral",
                (float(lateral[0]), float(lateral[1])),
                signed_hint="positive lateral is left",
            )
            + ", and "
            + _format_pelvis_goal_range(
                "vertical",
                (float(vertical[0]), float(vertical[1])),
                signed_hint="positive vertical is up",
            )
            + "."
        )

    reach_correction = right_hand.get("reach_correction")
    if reach_correction is not None:
        if not isinstance(reach_correction, Mapping):
            raise TypeError(
                "right_hand.reach_correction must be a mapping when present."
            )
        status = str(reach_correction.get("status") or "")
        if status == "measured":
            body_forward_m = float(reach_correction["body_forward_m"])
            approach = reach_correction.get("approach")
            if not isinstance(approach, Mapping):
                raise TypeError(
                    "Measured reach_correction requires an approach mapping."
                )
            kind = str(approach.get("kind") or "unknown")
            parts.append(
                "A body-forward probe of those failed palm goals found "
                f"solvability only after translating candidates about "
                f"{body_forward_m:.2f} m toward the robot along body-forward "
                f"({kind} approach)."
            )
        elif status == "not_found_within_budget":
            budget_m = float(reach_correction["probe_budget_m"])
            parts.append(
                "No body-forward reach correction was found within the "
                f"planner probe budget of {budget_m:.2f} m."
            )
        elif status == "unavailable":
            reason = str(reach_correction.get("reason") or "").strip()
            parts.append(
                f"Body-forward reach correction could not be measured ({reason})."
            )
        else:
            raise ValueError(
                f"right_hand.reach_correction has an unknown status: {status!r}."
            )

    parts.append(
        "Change stance, posture, or view before retrying; solvability depends "
        "on stance, height, bearing, and approach."
    )
    return " ".join(parts)


def _require_curobo_v2() -> None:
    import curobo
    from packaging.version import InvalidVersion, Version

    version = str(getattr(curobo, "__version__", ""))
    try:
        supported = Version(version).release == (0, 8, 0)
    except InvalidVersion:
        supported = False
    if not supported:
        raise RuntimeError(
            f"TARSphere requires cuRobo {CUROBO_V2_VERSION}; "
            f"found {version or 'an unversioned installation'}."
        )


def _rigid_transform(value: np.ndarray) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("robot_root_from_camera must be a finite 4x4 transform.")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("robot_root_from_camera must use homogeneous coordinates.")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError("robot_root_from_camera rotation must be orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3):
        raise ValueError("robot_root_from_camera rotation must be right-handed.")
    return transform


def _rotation_to_quaternion_wxyz(
    rotation: np.ndarray,
) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)


def _require_finite_vector(
    name: str,
    values: Sequence[float],
    size: int,
) -> tuple[float, ...]:
    """Return one finite vector or fail; diagnostic math has no placeholders."""

    try:
        out = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain {size} finite values, got {values!r}."
        ) from exc
    if len(out) != size or not all(math.isfinite(value) for value in out):
        raise ValueError(f"{name} must contain {size} finite values, got {values!r}.")
    return out


def homogeneous_transform_point(
    frame_from_world: np.ndarray,
    point_world_m: Sequence[float],
) -> tuple[float, float, float]:
    """Map one world point through a 4x4 world-to-frame transform."""

    matrix = np.asarray(frame_from_world, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(
            "frame_from_world must be a finite 4x4 transform, got "
            f"shape={getattr(matrix, 'shape', None)!r}."
        )
    point = _require_finite_vector("point_world_m", point_world_m, 3)
    mapped = matrix @ np.asarray((*point, 1.0), dtype=np.float64)
    if mapped.shape != (4,) or not np.all(np.isfinite(mapped)):
        raise ValueError(
            f"frame_from_world produced a non-finite point, got {mapped!r}."
        )
    return (float(mapped[0]), float(mapped[1]), float(mapped[2]))


def quaternion_wxyz_roll_pitch_yaw_rad(
    quaternion_wxyz: Sequence[float],
) -> tuple[float, float, float]:
    """Return intrinsic XYZ roll/pitch/yaw for a scalar-first world orientation."""

    quaternion = _require_finite_vector(
        "quaternion_wxyz",
        quaternion_wxyz,
        4,
    )
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-8:
        raise ValueError("quaternion_wxyz has zero norm.")
    w, x, y, z = (value / norm for value in quaternion)
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    sin_pitch = 2.0 * (w * y - z * x)
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    pitch_argument = min(1.0, max(-1.0, sin_pitch))
    return (
        math.atan2(sin_roll, cos_roll),
        math.asin(pitch_argument),
        math.atan2(sin_yaw, cos_yaw),
    )


def _wrapped_angle_delta_rad(measured_rad: float, captured_rad: float) -> float:
    delta = float(measured_rad) - float(captured_rad)
    return math.atan2(math.sin(delta), math.cos(delta))


def pelvis_world_pose_from_g1_planning_root(
    robot_root_T_world: np.ndarray,
    *,
    commanded_body_height_m: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Recover the pelvis pose that built one cuRobo planning-root transform.

    Runtime builds the root by taking the live pelvis world-to-frame pose and
    adding the synthetic ``base_j_z`` height to that translation. Inverting
    that offset recovers the articulated pelvis, not the USD robot prim.
    ``commanded_body_height_m`` is the pelvis-above-stance height that was
    added when the root was built (the live measurement, not a later command).
    """

    root = np.asarray(robot_root_T_world, dtype=np.float64)
    if root.shape != (4, 4) or not np.all(np.isfinite(root)):
        raise ValueError(
            "robot_root_T_world must be a finite 4x4 transform, got "
            f"shape={getattr(root, 'shape', None)!r}."
        )
    height = g1_curobo_pelvis_model_height_from_body_height_m(
        commanded_body_height_m,
    )
    rotation_root_from_world = root[:3, :3]
    translation_pelvis = root[:3, 3] - np.asarray(
        [0.0, 0.0, height],
        dtype=np.float64,
    )
    rotation_world_from_pelvis = rotation_root_from_world.T
    pelvis_position = -rotation_world_from_pelvis @ translation_pelvis
    if not np.all(np.isfinite(pelvis_position)):
        raise ValueError(
            f"recovered pelvis position is not finite: {pelvis_position!r}."
        )
    orientation = _rotation_to_quaternion_wxyz(rotation_world_from_pelvis)
    return (
        (
            float(pelvis_position[0]),
            float(pelvis_position[1]),
            float(pelvis_position[2]),
        ),
        orientation,
    )


def g1_closing_reach_deflection_metrics(
    *,
    commanded_waist_yaw_rad: float,
    commanded_waist_pitch_rad: float,
    measured_waist_yaw_rad: float,
    measured_waist_pitch_rad: float,
    captured_pelvis_position_world_m: Sequence[float],
    captured_pelvis_orientation_world_wxyz: Sequence[float],
    live_pelvis_position_world_m: Sequence[float],
    live_pelvis_orientation_world_wxyz: Sequence[float],
    requested_palm_world_m: Sequence[float],
    requested_palm_in_planning_root_m: Sequence[float],
    reached_palm_world_m: Sequence[float],
    live_planning_root_T_world: np.ndarray,
    commanded_arm_rad: Sequence[float],
    measured_arm_rad: Sequence[float],
    commanded_body_height_m: float,
    measured_body_height_m: float,
) -> dict[str, float]:
    """Attribute closing-reach palm error to waist, pelvis, or the arm.

    Waist, arm, and body-height errors are commanded minus measured, matching
    the existing terminal-arm signed-error convention. ``palm_err_in_live_root_m``
    compares the reached palm in the *live* planning root against the palm pose
    cuRobo was asked for in the captured root. Small in-root error with large
    world error means the frame moved; large in-root error means the chain
    missed.
    """

    waist = {
        "waist_yaw_cmd": commanded_waist_yaw_rad,
        "waist_yaw_meas": measured_waist_yaw_rad,
        "waist_pitch_cmd": commanded_waist_pitch_rad,
        "waist_pitch_meas": measured_waist_pitch_rad,
    }
    for name, value in waist.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value!r}.")
    captured_position = _require_finite_vector(
        "captured_pelvis_position_world_m",
        captured_pelvis_position_world_m,
        3,
    )
    live_position = _require_finite_vector(
        "live_pelvis_position_world_m",
        live_pelvis_position_world_m,
        3,
    )
    captured_rpy = quaternion_wxyz_roll_pitch_yaw_rad(
        captured_pelvis_orientation_world_wxyz,
    )
    live_rpy = quaternion_wxyz_roll_pitch_yaw_rad(
        live_pelvis_orientation_world_wxyz,
    )
    requested_world = _require_finite_vector(
        "requested_palm_world_m",
        requested_palm_world_m,
        3,
    )
    requested_root = _require_finite_vector(
        "requested_palm_in_planning_root_m",
        requested_palm_in_planning_root_m,
        3,
    )
    reached_world = _require_finite_vector(
        "reached_palm_world_m",
        reached_palm_world_m,
        3,
    )
    commanded_arm = _require_finite_vector(
        "commanded_arm_rad",
        commanded_arm_rad,
        7,
    )
    measured_arm = _require_finite_vector(
        "measured_arm_rad",
        measured_arm_rad,
        7,
    )
    if not isinstance(commanded_body_height_m, (int, float)) or not math.isfinite(
        float(commanded_body_height_m)
    ):
        raise ValueError(
            f"commanded_body_height_m must be finite, got {commanded_body_height_m!r}."
        )
    if not isinstance(measured_body_height_m, (int, float)) or not math.isfinite(
        float(measured_body_height_m)
    ):
        raise ValueError(
            f"measured_body_height_m must be finite, got {measured_body_height_m!r}."
        )
    reached_in_live_root = homogeneous_transform_point(
        live_planning_root_T_world,
        reached_world,
    )
    return {
        "waist_yaw_cmd": float(commanded_waist_yaw_rad),
        "waist_yaw_meas": float(measured_waist_yaw_rad),
        "waist_yaw_err": (
            float(commanded_waist_yaw_rad) - float(measured_waist_yaw_rad)
        ),
        "waist_pitch_cmd": float(commanded_waist_pitch_rad),
        "waist_pitch_meas": float(measured_waist_pitch_rad),
        "waist_pitch_err": (
            float(commanded_waist_pitch_rad) - float(measured_waist_pitch_rad)
        ),
        "pelvis_dz_m": live_position[2] - captured_position[2],
        "pelvis_pitch_delta_rad": _wrapped_angle_delta_rad(
            live_rpy[1],
            captured_rpy[1],
        ),
        "pelvis_roll_delta_rad": _wrapped_angle_delta_rad(
            live_rpy[0],
            captured_rpy[0],
        ),
        "pelvis_xy_delta_m": math.hypot(
            live_position[0] - captured_position[0],
            live_position[1] - captured_position[1],
        ),
        "body_height_cmd": float(commanded_body_height_m),
        "body_height_meas": float(measured_body_height_m),
        "body_height_err": (
            float(commanded_body_height_m) - float(measured_body_height_m)
        ),
        "palm_err_world_m": math.dist(reached_world, requested_world),
        "palm_err_in_live_root_m": math.dist(
            reached_in_live_root,
            requested_root,
        ),
        "max_arm_joint_err_rad": max(
            abs(commanded - measured)
            for commanded, measured in zip(
                commanded_arm,
                measured_arm,
                strict=True,
            )
        ),
    }


def format_g1_deflection_probe_line(
    *,
    at: str,
    metrics: Mapping[str, float],
) -> str:
    """Format one ``[G1-DEFLECT]`` line from already-sampled metrics."""

    checkpoint = str(at).strip()
    if not checkpoint:
        raise ValueError("deflection probe checkpoint name must be non-empty.")
    required = (
        "waist_yaw_cmd",
        "waist_yaw_meas",
        "waist_yaw_err",
        "waist_pitch_cmd",
        "waist_pitch_meas",
        "waist_pitch_err",
        "pelvis_dz_m",
        "body_height_cmd",
        "body_height_meas",
        "body_height_err",
        "pelvis_pitch_delta_rad",
        "pelvis_roll_delta_rad",
        "pelvis_xy_delta_m",
        "palm_err_world_m",
        "palm_err_in_live_root_m",
        "max_arm_joint_err_rad",
    )
    missing = [name for name in required if name not in metrics]
    if missing:
        raise ValueError(f"deflection probe metrics are missing fields: {missing!r}.")
    for name in required:
        value = metrics[name]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(
                f"deflection probe metric {name} must be finite, got {value!r}."
            )
    return (
        "[G1-DEFLECT] "
        f"at={checkpoint} "
        f"waist_yaw_cmd={float(metrics['waist_yaw_cmd']):.4f} "
        f"waist_yaw_meas={float(metrics['waist_yaw_meas']):.4f} "
        f"waist_yaw_err={float(metrics['waist_yaw_err']):.4f} "
        f"waist_pitch_cmd={float(metrics['waist_pitch_cmd']):.4f} "
        f"waist_pitch_meas={float(metrics['waist_pitch_meas']):.4f} "
        f"waist_pitch_err={float(metrics['waist_pitch_err']):.4f} "
        f"pelvis_dz_m={float(metrics['pelvis_dz_m']):.4f} "
        f"body_height_cmd={float(metrics['body_height_cmd']):.4f} "
        f"body_height_meas={float(metrics['body_height_meas']):.4f} "
        f"body_height_err={float(metrics['body_height_err']):.4f} "
        f"pelvis_pitch_delta_rad={float(metrics['pelvis_pitch_delta_rad']):.4f} "
        f"pelvis_roll_delta_rad={float(metrics['pelvis_roll_delta_rad']):.4f} "
        f"pelvis_xy_delta_m={float(metrics['pelvis_xy_delta_m']):.4f} "
        f"palm_err_world_m={float(metrics['palm_err_world_m']):.4f} "
        f"palm_err_in_live_root_m={float(metrics['palm_err_in_live_root_m']):.4f} "
        f"max_arm_joint_err_rad={float(metrics['max_arm_joint_err_rad']):.4f}"
    )


__all__ = [
    "CUROBO_V2_VERSION",
    "G1_ARM_JOINT_NAMES",
    "G1_CUROBO_JOINT_NAMES",
    "G1_CUROBO_REFERENCE_PELVIS_HEIGHT_M",
    "G1_CUROBO_PALM_TOOL_POINT_IN_PALM_LINK_M",
    "g1_curobo_pelvis_model_height_from_body_height_m",
    "g1_curobo_pelvis_model_height_m",
    "g1_carry_palm_goal_offset_pelvis_root_m",
    "g1_closing_reach_deflection_metrics",
    "format_g1_deflection_probe_line",
    "homogeneous_transform_point",
    "pelvis_world_pose_from_g1_planning_root",
    "quaternion_wxyz_roll_pitch_yaw_rad",
    "palm_goal_root_to_pelvis_m",
    "G1_MANIPULATION_WAIST_LIMITS_RAD",
    "G1_MANIPULATION_COORDINATE_NAMES",
    "G1_WAIST_JOINT_NAMES",
    "g1_locomotion_zero_manipulation_waypoint",
    "g1_manipulation_waypoint_arm_values",
    "g1_manipulation_waypoint_with_arm_values",
    "g1_manipulation_waypoint_length",
    "g1_manipulation_waypoint_value",
    "G1_GRASP_PLANNING_TIMEOUT_S",
    "G1_GRASP_WORKER_STAGES_FILENAME",
    "G1_SCENE_AABB_OBSTACLES_FORBIDDEN",
    "G1_GRASP_TARGET_ENCLOSING_LINKS",
    "G1_GRASP_TARGET_OBSTACLE_NAME",
    "G1_SUPPORT_SURFACE_OBSTACLE_NAME",
    "SupportSurfaceObstacle",
    "TargetObjectObstacle",
    "forbid_g1_scene_aabb_obstacles",
    "support_surface_obstacle_from_payload",
    "support_surface_obstacle_in_planning_root",
    "support_surface_obstacle_payload",
    "target_object_obstacle_from_payload",
    "target_object_obstacle_in_planning_root",
    "target_object_obstacle_payload",
    "G1GraspCandidateFailure",
    "G1GraspReachCorrection",
    "G1GraspPlanningInfeasible",
    "G1CarryPlanningInfeasible",
    "G1GraspWorkerFailure",
    "G1GraspWorkerTimeout",
    "G1PelvisHeightCoordinates",
    "G1GraspPlan",
    "G1CarryPlan",
    "G1GraspPlanner",
    "G1GraspPlannerClient",
    "carry_plan_from_payload",
    "carry_plan_payload",
    "carry_planning_infeasible_payload",
    "carry_planning_result_from_payload",
    "carry_planning_success_payload",
    "grasp_plan_from_payload",
    "grasp_plan_payload",
    "grasp_planning_infeasible_payload",
    "grasp_planning_result_from_payload",
    "grasp_planning_success_payload",
    "summarize_grasp_reachability",
    "format_reach_envelope_refusal_message",
    "plan_g1_grasp",
]
