"""TARSphere/Isaac Sim backend for the episode runner."""

from __future__ import annotations

import json
import os
import math
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, cast

import numpy as np

from tarsphere.geometry import (
    BBox3D,
    HUMAN_BODY_HEIGHT_M,
    HUMAN_BODY_RADIUS_M,
    fov_to_intrinsics,
    human_bbox3d,
    observed_volume_bbox,
    observed_volume_evidence,
)
from tarsphere.embodiment.dex3 import G1_MANIPULATION_HAND
from tarsphere.infra.manipulation.grasp_catalog import object_class_of
from tarsphere.bench.schema import DiagnosticCheck, TARSphereEpisode
from tarsphere.infra.perception.rgbd_utils import compose_rgbd_image
from tarsphere.infra.perception.sensor_conditions import (
    condition_for_episode,
    condition_sensor_frame,
)
from tarsphere.infra.control.human_body import (
    HumanBodyResult,
    HumanBodyState,
)
from tarsphere.infra.episode.human_actor_runtime import resolve_human_action_point
from tarsphere.infra.episode.errors import RobotFallError
from tarsphere.infra.manipulation.agent_grasp import (
    closing_palm_residual_m,
    GRASP_GROUNDING_MEMORY,
    GRASP_GROUNDING_SAM3,
    frame_from_world_pose,
    GraspFailureCode,
    GraspNotPossibleHere,
    grasp_target_observation,
    match_grounded_pickable,
    nearest_grounded_pickable,
    sam3_gt_aabb_discrepancy,
)
from tarsphere.infra.manipulation.graspgenx import (
    UNITREE_G1_SWEEP_VOLUME,
    grasp_pose_from_payload,
)
from tarsphere.infra.manipulation.g1_grasp import (
    G1_ARM_JOINT_NAMES,
    G1_CUROBO_JOINT_NAMES,
    G1_CUROBO_PALM_TOOL_POINT_IN_PALM_LINK_M,
    G1CarryPlanningInfeasible,
    G1GraspPlanningInfeasible,
    G1GraspPlan,
    G1GraspPlannerClient,
    g1_curobo_pelvis_model_height_from_body_height_m,
    g1_manipulation_waypoint_arm_values,
    g1_manipulation_waypoint_length,
    g1_manipulation_waypoint_value,
    g1_manipulation_waypoint_with_arm_values,
    g1_closing_reach_deflection_metrics,
    format_g1_deflection_probe_line,
    homogeneous_transform_point,
    pelvis_world_pose_from_g1_planning_root,
    grasp_plan_from_payload,
    grasp_plan_payload,
    plan_g1_grasp,
    summarize_grasp_reachability,
    format_reach_envelope_refusal_message,
    support_surface_obstacle_in_planning_root,
    target_object_obstacle_in_planning_root,
)
from tarsphere.infra.control.motion_planning import (
    resample_control_trajectory,
)
from tarsphere.harness.harness import (
    AgentHarnessConfig,
)
from tarsphere.infra.control.internnav import (
    DUALVLN_DEPTH_MAX_M,
    DUALVLN_HORIZONTAL_FOV_DEG,
    DUALVLN_LOOK_DOWN_EXTRA_DEG,
    DUALVLN_LOOK_UP_EXTRA_DEG,
    DUALVLN_MAX_INNER_STEPS,
    DUALVLN_RGB_HEIGHT,
    DUALVLN_RGB_WIDTH,
    DualVLNObservation,
    build_internnav_backend,
    internav_backend_available,
    official_norm_depth,
)
from tarsphere.contracts.human_interaction import (
    HUMAN_COMMUNICATION_RADIUS_M,
)
from tarsphere.infra.control.physical_contracts import (
    CAMERA_GAZE_TILT_LIMIT_DEG,
    G1_ARM_IK_SOLVER_TOLERANCE_M,
    G1_END_EFFECTOR_DELTA_MAX_ROTATION_RAD,
    G1_END_EFFECTOR_DELTA_MAX_TRANSLATION_M,
    G1_END_EFFECTOR_DELTA_POSITION_TOLERANCE_M,
    G1_END_EFFECTOR_GRIPPER_TRACKING_TOLERANCE,
    G1_ARM_TRAJECTORY_TRACKING_TOLERANCE_RAD,
    G1_BODY_POSTURE_ACTIONS,
    G1_CARRY_LOCOMOTION_UPPER_BODY_INTERMEDIATE_FRACTIONS,
    G1_CARRY_WAIST_RESTORE_SETTLE_TOLERANCE_RAD,
    G1_GRASP_REACH_TRACKING_TOLERANCE_RAD,
    G1_MAXIMUM_CROUCH_HEIGHT_COMMAND_M,
    G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M,
    G1_GRASP_CONTACT_GAP_M,
    G1_GRASP_POSE_EXCLUSION_RADIUS_M,
    G1_GRASP_SUPPORT_SETTLE_MAX_HORIZONTAL_DISPLACEMENT_M,
    G1_GRASP_SUPPORT_SETTLE_MAX_VERTICAL_DROP_M,
    G1_GRASP_SUPPORT_SETTLE_PHYSICS_STEPS,
    G1_PRE_RELEASE_CLEARANCE_M,
    G1_REACH_SETTLE_CONVERGENCE_M,
    G1_REACH_SETTLE_MAX_PHYSICS_STEPS,
    G1_REACH_SETTLE_PHYSICS_STEPS,
    G1_REACH_CORRECTION_MAX_JOINT_DELTA_RAD,
    G1_REACH_CORRECTION_ORIENTATION_TOLERANCE_RAD,
    G1_REACH_CORRECTION_MAX_M,
    G1_REACH_CORRECTION_MAX_STEP_M,
    G1_REACH_CORRECTION_MAX_PASSES,
    G1_REACH_CORRECTION_MIN_PROGRESS_M,
    G1_REACH_CORRECTION_RAMP_PHYSICS_STEPS,
    G1_REACH_CORRECTION_TOLERANCE_M,
    G1_PRE_RELEASE_HORIZONTAL_TOLERANCE_M,
    G1_PRE_RELEASE_SUPPORT_CONTACT_TOLERANCE_M,
    G1_PELVIS_HEIGHT_TRACKING_TOLERANCE_M,
    G1_INITIAL_POSE_TOLERANCE_M,
    G1_STANDING_BODY_TOP_M,
    G1_STANDING_HEIGHT_COMMAND_M,
    HUMAN_GOTO_ARRIVAL_TOLERANCE_M,
    HUMAN_ANIMATION_KIT_STEP_SIZE_S,
    HUMAN_ANIMATION_PUMP_EVERY_TICKS,
    HUMAN_INITIAL_POSE_TOLERANCE_M,
    HUMAN_NAVMESH_AGENT_RADIUS_M,
    LIDAR_FLOOR_TOLERANCE_M,
    LIDAR_MAX_RANGE_M,
    LIDAR_ORIGIN_HEIGHT_M,
    SIMULATION_PHYSICS_HZ,
    g1_body_posture_from_height_command,
    g1_body_surface_offset_m,
    g1_control_smooth_ramp_samples,
    require_standing_g1_locomotion,
)
from tarsphere.infra.perception.lidar import (
    beam_azimuths_deg,
    beam_elevations_deg,
    build_lidar,
)
from tarsphere.infra.control.anim_people_executor import AnimPeopleBodyExecutor
from tarsphere.infra.control.g1_balance_control import (
    G1BodyDynamics,
    measure_g1_body_dynamics,
)
from tarsphere.infra.control.robot_body_executor import RobotBodyExecutor
from tarsphere.infra.control.relative_pose import BodyPose2D
from tarsphere.infra.perception.continuous_ego_video import (
    ContinuousEgoVideoArtifacts,
    ContinuousEgoVideoRecorder,
)
from tarsphere.infra.perception.room_observer import (
    ROOM_OBSERVER_RECORDER_ROOT,
    ROOM_OBSERVER_NATIVE_RENDER_WARMUP_LIMIT,
    RoomObserverRecorder,
    episode_route_points_xy,
)
from tarsphere.infra.episode.episode_asset_adapter import (
    EpisodeAssetAdapter,
    adaptation_from_episode,
)
from tarsphere.infra.simulation.physical_scene_preparation import (
    PhysicalScenePreparer,
    preparation_from_episode,
    validate_live_manipulation_target_pose,
)
from tarsphere.infra.simulation.scene_spatial_lighting import (
    scene_lighting_repair_for_asset,
)
from tarsphere.infra.simulation.scene_rendering import (
    apply_evaluation_scene_rendering,
)
from tarsphere.assets import (
    configure_simulator_assets,
    tarsphere_asset_path,
    resolve_tarsphere_asset,
)


# PhysX contact reporting can be empty at one public-action boundary while a
# friction-held target is still inside the closed Dex3 hand.  Reuse the
# controller's existing one-tenth-second settle window before applying the
# unchanged live-contact gate again.  Keeping this executor-only avoids
# changing the production G1 geometry/contact profile: no footprint or
# acceptance threshold is altered.
_G1_CARRY_CONTACT_CONFIRMATION_PHYSICS_STEPS = G1_REACH_SETTLE_PHYSICS_STEPS


VIEWER_MODES = {"third_person", "first_person", "free"}
_PRIVATE_MAPPING_LOCOMOTION_ACTIONS = frozenset(
    {
        "move_body",
        "move_forward",
        "move_backward",
        "move_left",
        "move_right",
        "turn_left",
        "turn_right",
        "precise_rotate",
        "precise_approach",
        "rotate_to",
        "follow_path",
    }
)


#: AnimPeople characters face heading + 90 deg in character-yaw space; the
#: initial standing pose and the dict initializations must apply the same
#: offset the walk/sit command paths already compensate for.
from tarsphere.infra.control.physical_contracts import HUMAN_CHARACTER_YAW_OFFSET_RAD


class RenderMode(str, Enum):
    """When native Kit rendering runs between decision boundaries.

    ``evaluation`` (default) withholds TARSphere video-cadence renders
    inside a decision window. Agent-facing RGB still comes only from the
    decision snapshot; online-navigation diagnostics may independently render
    private fixed-rate mapping frames. An explicitly enabled continuous recorder
    also retains native cadence without exposing those frames to the policy.
    ``demo`` keeps the configured cadence
    so recorded video is smooth. Neither private mapping nor demo frames may
    change model-visible history or trace observations.
    """

    EVALUATION = "evaluation"
    DEMO = "demo"


class G1CarryPostureTrackingError(RuntimeError):
    """A loaded G1 transition missed its measured locomotion posture."""


class G1ArmTrackingError(G1CarryPostureTrackingError):
    """A planned G1 manipulation segment missed its measured joint contract."""


class G1WaistTrackingError(G1CarryPostureTrackingError):
    """A loaded G1 transition failed to restore the measured waist posture."""


class RobotPoseUnavailableError(RuntimeError):
    """The live robot pose required by spatial evaluation is unavailable."""


class _G1SupportUnstable(Exception):
    """Finish failed-grasp cleanup before publishing the next ego frame."""


DEPTH_VALID_RATIO_THRESHOLD = 0.05
RGB_VISIBLE_MAX_THRESHOLD = 20.0
RGB_VISIBLE_STD_THRESHOLD = 1.0
_SIMULATOR_REP_CAMERA_PATCHED = False
# The extra millimetre is only a floating-point/mesh-boundary comparison
# margin; it must not be used by the planner as additional reach.
_G1_PLACEMENT_NUMERICAL_MARGIN_M = 0.001
# The initial support reach remains position-only for workspace coverage.
# Residual refinements preserve their live warm-start hand orientation, and
# decreasing gains keep that bounded payload correction contractive near the
# support envelope.
_G1_PLACEMENT_REFINEMENT_GAINS = (0.50, 0.35, 0.25, 0.15, 0.10)
# A held object's hand-relative offset can leave an otherwise valid arm solve
# outside the support envelope.  Close that world-XY residual with the same
# official G1 body-velocity controller used by precise stance approach.  The
# loop is live-geometry driven and bounded; it is not another navigation
# backend and it never changes the authored support target.
_G1_PLACEMENT_BASE_MIN_SPEED_MPS = 0.04
_G1_PLACEMENT_BASE_MAX_SPEED_MPS = 0.20


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return int(default)
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return bool(default)
    return value in {"1", "true", "yes", "on"}


def _read_scene_asset_unit_in_meters(scene_usd: Path) -> float:
    """Read how many metres one authored coordinate unit represents."""
    from pxr import Usd, UsdGeom

    source_stage = Usd.Stage.Open(
        str(scene_usd),
        load=Usd.Stage.LoadNone,
    )
    if source_stage is None:
        raise RuntimeError(f"Cannot open scene asset metadata: {scene_usd}")
    unit_in_meters = float(UsdGeom.GetStageMetersPerUnit(source_stage))
    if not math.isfinite(unit_in_meters) or unit_in_meters <= 0.0:
        raise RuntimeError(
            f"Scene asset has an invalid metersPerUnit value: "
            f"{scene_usd} -> {unit_in_meters!r}"
        )
    if "GRScenes-100" in scene_usd.parts and not math.isclose(
        unit_in_meters, 0.01, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(
            "GRScenes-100 assets require 1 authored unit = 0.01 m; "
            f"{scene_usd} declares {unit_in_meters:g} m."
        )
    return unit_in_meters


def _require_metric_scene_scale(
    scene_scale: Sequence[float],
    *,
    asset_unit_in_meters: float,
) -> None:
    """Require the scene reference to map authored coordinates into metres."""
    values = tuple(float(value) for value in scene_scale)
    expected = (float(asset_unit_in_meters),) * 3
    if len(values) != 3 or any(
        not math.isclose(
            value,
            expected_value,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for value, expected_value in zip(values, expected)
    ):
        raise RuntimeError(
            "Episode scene.scale must map the authored scene into the "
            "benchmark metre-based world: "
            f"expected={expected!r}, received={values!r}."
        )


def multiply_quaternions_wxyz(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float, float]:
    """Return left ⊗ right for scalar-first ``(w, x, y, z)`` quaternions."""

    lw, lx, ly, lz = (float(value) for value in left)
    rw, rx, ry, rz = (float(value) for value in right)
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def rotate_x_quaternion_wxyz(pitch_deg: float) -> tuple[float, float, float, float]:
    """Local RotateX quaternion, matching the camera's negative-down pitch."""

    half = math.radians(float(pitch_deg)) * 0.5
    return (math.cos(half), math.sin(half), 0.0, 0.0)


def camera_local_orientation_wxyz(
    rest_wxyz: Sequence[float],
    pitch_deg: float,
) -> tuple[float, float, float, float]:
    """Compose the camera's rest orientation with the commanded local pitch."""

    return multiply_quaternions_wxyz(
        rest_wxyz,
        rotate_x_quaternion_wxyz(pitch_deg),
    )


_CAMERA_SIDECAR_PITCH_OP = "xformOp:rotateX:hcaCameraPitch"


def _compiled_occupancy_path(episode: TARSphereEpisode) -> Path:
    """Resolve the official occupancy declared by the compiled scene."""
    world = episode.world_knowledge or {}
    scene_map = world.get("scene_map")
    if not isinstance(scene_map, dict):
        raise RuntimeError("Runtime navigation requires world_knowledge.scene_map.")
    occupancy = str(scene_map.get("occupancy") or "").strip()
    if not occupancy:
        raise RuntimeError(
            "Runtime navigation requires world_knowledge.scene_map.occupancy."
        )
    path = Path(resolve_tarsphere_asset(occupancy))
    if not path.is_file():
        raise FileNotFoundError(f"Compiled scene occupancy is missing: {path}")
    return path


def _rollout_memory_diagnostic(label: str, physics_steps: int) -> None:
    """Emit opt-in native RSS markers around late-rollout simulator calls."""
    if not _bool_env("TARSPHERE_ROLLOUT_MEMORY_DIAGNOSTICS", False):
        return
    threshold = _int_env("TARSPHERE_ROLLOUT_MEMORY_DIAGNOSTICS_MIN_STEP", 31_000)
    if int(physics_steps) < threshold:
        return
    rss_kib = None
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss_kib = int(line.split()[1])
                break
    except (OSError, ValueError, IndexError):
        pass
    print(
        f"[MEMTRACE] {label} physics_steps={int(physics_steps)} rss_kib={rss_kib}",
        flush=True,
    )


ROBOT_CAMERA_MIN_WARMUP_STEPS = max(
    0,
    _int_env("TARSPHERE_ROBOT_CAMERA_MIN_WARMUP_STEPS", 80),
)
ROBOT_CAMERA_MAX_WARMUP_STEPS = max(
    ROBOT_CAMERA_MIN_WARMUP_STEPS + 1,
    _int_env("TARSPHERE_ROBOT_CAMERA_MAX_WARMUP_STEPS", 360),
)
# Soft reset keeps the Replicator SDG pipeline; only re-settle loco + confirm
# RGBD. Default far below the cold-start min so bake→/run_episode and later
# episodes do not replay the full 80-step gate.
ROBOT_CAMERA_SOFT_MIN_WARMUP_STEPS = max(
    0,
    _int_env("TARSPHERE_SOFT_RESET_CAMERA_MIN_WARMUP_STEPS", 12),
)
# Bake primes the persistent RTX/Replicator camera once.  This is deliberately
# shorter than an ordinary cold episode gate: the first soft reset will verify
# the already-live render product again after spawning Humans.
ROBOT_CAMERA_PREPARE_MIN_WARMUP_STEPS = max(
    0,
    _int_env("TARSPHERE_BAKE_PREPARE_CAMERA_MIN_WARMUP_STEPS", 16),
)
# The evaluated current ego observation is a native 640x480 frame. The same
# RGB-D render product feeds model perception and optional ego recording, so
# neither path resizes or creates a second camera. Horizontal FOV remains the
# independently authored optical contract; vertical aperture follows this 4:3
# render product through ROBOT_EGO_CAMERA_ASPECT_RATIO below.
ROBOT_EGO_CAMERA_DEFAULT_RESOLUTION = (640, 480)
ROBOT_EGO_CAMERA_RESOLUTION = (
    max(
        64,
        _int_env(
            "TARSPHERE_ROBOT_CAMERA_WIDTH",
            ROBOT_EGO_CAMERA_DEFAULT_RESOLUTION[0],
        ),
    ),
    max(
        64,
        _int_env(
            "TARSPHERE_ROBOT_CAMERA_HEIGHT",
            ROBOT_EGO_CAMERA_DEFAULT_RESOLUTION[1],
        ),
    ),
)
HUMAN_EGO_CAMERA_RESOLUTION = (1280, 720)
EGO_CAMERA_ASPECT_RATIO = (
    HUMAN_EGO_CAMERA_RESOLUTION[0] / HUMAN_EGO_CAMERA_RESOLUTION[1]
)
# The robot resolution is env-overridable, so its vertical aperture must follow
# its own render product. Borrowing the human ratio would leave the rendered
# aspect and the camera aperture disagreeing, while fov_to_intrinsics still
# assumes square pixels — every depth reprojection would then be silently off.
ROBOT_EGO_CAMERA_ASPECT_RATIO = (
    ROBOT_EGO_CAMERA_RESOLUTION[0] / ROBOT_EGO_CAMERA_RESOLUTION[1]
)
HUMAN_EGO_CAMERA_HORIZONTAL_FOV_DEG = _float_env(
    "TARSPHERE_HUMAN_EGO_CAMERA_FOV_DEG", 90.0
)
HUMAN_EGO_EYE_HEIGHT_METERS = 1.55
HUMAN_EGO_FORWARD_OFFSET_METERS = 0.18
# Isaac People runtime yaw is reported in the navigation/world +X convention,
# while the rendered medical character's visual front is opposite the earlier
# +Y assumption.  Use -π/2 so the ego camera looks out of the character's face
# instead of behind the head.
HUMAN_EGO_CAMERA_YAW_OFFSET_RAD = -math.pi / 2

# Keep robot and human ego cameras aligned for comparable observations while
# avoiding the earlier fisheye-like 120-150 degree debug view.
ROBOT_CAMERA_HORIZONTAL_FOV_DEG = _float_env(
    "TARSPHERE_ROBOT_CAMERA_FOV_DEG",
    100.0,
)
# Robot ego cameras use asset/profile-authored orientation. Do not apply a
# global pitch correction: it corrupts G1 and H1 differently depending on mount
# frame.  Per-robot fixed mount calibration lives in RobotProfile.
# Height of the ego camera above the robot base link, exposed to the agent so a
# BEV memory module can world-Z filter obstacles.  Override for calibration.
ROBOT_CAMERA_HEIGHT_M = _float_env("TARSPHERE_ROBOT_CAMERA_HEIGHT_M", 1.0)

HUMAN_COLLISION_RADIUS_METERS = HUMAN_BODY_RADIUS_M
HUMAN_COLLISION_HEIGHT_METERS = HUMAN_BODY_HEIGHT_M
ANIM_PEOPLE_NAVMESH_MARGIN_METERS = 4.0
ANIM_PEOPLE_NAVMESH_MIN_SIZE_METERS = 8.0
ANIM_PEOPLE_NAVMESH_HEIGHT_METERS = 3.0
ANIM_PEOPLE_NAVMESH_FLOOR_OFFSET_METERS = 0.5  # NavMeshVolume bottom extends below z=0
ANIM_PEOPLE_NAVMESH_AGENT_MIN_HEIGHT_METERS = 1.0
ANIM_PEOPLE_NAVMESH_AGENT_MIN_RADIUS_METERS = (
    0.2  # was 0.1; coarser voxel → faster bake
)
ANIM_PEOPLE_NAVMESH_AGENT_MAX_RADIUS_METERS = 0.5
ANIM_PEOPLE_NAVMESH_AGENT_MAX_STEP_HEIGHT_METERS = 0.4
ANIM_PEOPLE_NAVMESH_AGENT_MIN_ISLAND_RADIUS_METERS = 0.0
ANIM_PEOPLE_NAVMESH_AGENT_MAX_FLOOR_SLOPE_DEGREES = 45.0
ANIM_PEOPLE_NAVMESH_EXCLUDE_RIGID_BODIES = _bool_env(
    "TARSPHERE_NAVMESH_EXCLUDE_RIGID_BODIES",
    False,
)
TARSPHERE_ISAAC_DEBUG = _bool_env("TARSPHERE_ISAAC_DEBUG", False)


ANIM_PEOPLE_NAVMESH_VOLUME_PATH = "/World/NavMeshVolume"
ANIM_PEOPLE_NAVMESH_RENDER_ONLY_PATH_PARTS = (
    "__default_setting",
    "HDR_Sphere",
    "Sky",
    "sky",
)

# Default character used when episode YAML does not specify one.
# Must be a character name available under assets/characters/IsaacPeople/.
DEFAULT_ISAAC_CHARACTER = "F_Medical_01"


def prepare_depth_frame(
    depth: Any,
    image_shape: Tuple[int, int],
    *,
    valid_ratio_threshold: float = DEPTH_VALID_RATIO_THRESHOLD,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return a finite positive depth frame and public quality diagnostics."""
    height, width = image_shape
    empty = np.zeros((height, width), dtype=np.float32)
    if depth is None:
        return empty, {"depth_status": "missing", "depth_valid_ratio": 0.0}

    depth_array = np.asarray(depth.cpu() if hasattr(depth, "cpu") else depth)
    depth_array = np.squeeze(depth_array)
    if depth_array.size == 0:
        return empty, {"depth_status": "empty", "depth_valid_ratio": 0.0}
    if depth_array.shape != (height, width):
        return empty, {
            "depth_status": "invalid_shape",
            "depth_valid_ratio": 0.0,
            "depth_shape": list(depth_array.shape),
        }

    depth_array = depth_array.astype(np.float32, copy=False)
    valid_mask = np.isfinite(depth_array) & (depth_array > 0)
    valid_ratio = float(np.count_nonzero(valid_mask) / depth_array.size)
    clean_depth = np.where(valid_mask, depth_array, 0.0).astype(np.float32)

    if valid_ratio == 0.0:
        status = "empty"
    elif valid_ratio < valid_ratio_threshold:
        status = "low_confidence"
    else:
        status = "valid"
    return clean_depth, {"depth_status": status, "depth_valid_ratio": valid_ratio}


def rgb_frame_stats(rgb: Any) -> Dict[str, float]:
    """Return compact visibility diagnostics for a robot RGB frame."""
    arr = np.asarray(rgb)
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def rgb_frame_is_visible(rgb: Any) -> bool:
    """A real ego frame must contain RGB signal, including valid dark scenes.

    Mean luminance is deliberately not a validity condition: a camera facing a
    dark wall or plant can have a low global mean while retaining substantial
    contrast. Requiring both a non-trivial peak and variation rejects empty or
    near-constant buffers without treating scene illumination as sensor health.
    """
    stats = rgb_frame_stats(rgb)
    return (
        stats["max"] >= RGB_VISIBLE_MAX_THRESHOLD
        and stats["std"] >= RGB_VISIBLE_STD_THRESHOLD
    )


def rgbd_frame_is_observable(rgb: Any, *, depth_status: str) -> bool:
    """Accept a rendered RGB-D view without mistaking texture for liveness.

    Contrast remains useful during cold-start warmup, where an uninitialised
    RTX buffer must not become the first Agent observation. Once the camera is
    live, however, a close view of a plain tabletop or wall can be almost
    constant or darker than the warmup threshold while its aligned depth is
    fully valid. Such a frame is an observation, not a sensor failure. Require
    a finite, nonzero RGB payload in that case; do not apply an illumination
    threshold to an already-live camera. Cold-start warmup remains stricter.
    """
    values = np.asarray(rgb)
    if (
        values.ndim != 3
        or values.shape[-1] != 3
        or values.size == 0
        or not np.all(np.isfinite(values))
        or np.any(values < 0)
    ):
        return False
    stats = rgb_frame_stats(rgb)
    return rgb_frame_is_visible(rgb) or (
        stats["max"] > 0.0 and str(depth_status) in {"valid", "low_confidence"}
    )


def _validated_closing_centre(centre: Any) -> np.ndarray:
    """Return the world closing centre a failed pose is excluded by."""

    if centre is None:
        raise RuntimeError(
            "a compiled G1 grasp must report closing_volume_center_world so "
            "a pose that loses retention can be excluded by geometry."
        )
    validated = np.asarray(centre, dtype=np.float64)
    if validated.shape != (3,) or not np.all(np.isfinite(validated)):
        raise RuntimeError(
            f"closing_volume_center_world must be a finite world point; got {centre!r}."
        )
    return validated


def _planned_closing_centre_world(
    *,
    camera_from_gripper: Any,
    world_T_camera: np.ndarray,
) -> np.ndarray:
    """World position where a GraspGenX pose would close its fingers."""

    world_T_gripper = world_T_camera @ np.asarray(
        camera_from_gripper,
        dtype=np.float64,
    )
    return (
        world_T_gripper[:3, :3]
        @ np.asarray(UNITREE_G1_SWEEP_VOLUME["offset_mid"], dtype=np.float64)
        + world_T_gripper[:3, 3]
    )


def _planned_closing_centre_gap_m(
    *,
    camera_from_gripper: Any,
    world_T_camera: np.ndarray,
    object_bbox: tuple[Sequence[float], Sequence[float]],
) -> float:
    """Distance from the GraspGenX closing centre to the live target AABB."""

    center = _planned_closing_centre_world(
        camera_from_gripper=camera_from_gripper,
        world_T_camera=world_T_camera,
    )
    lower, upper = object_bbox
    squared = 0.0
    for axis in range(3):
        delta = max(
            float(lower[axis]) - float(center[axis]),
            0.0,
            float(center[axis]) - float(upper[axis]),
        )
        squared += delta * delta
    return math.sqrt(squared)


def _immutable_transform(matrix: Any) -> tuple[tuple[float, ...], ...]:
    """Freeze a 4x4 transform so a snapshot cannot be mutated in place."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (4, 4) or not np.all(np.isfinite(values)):
        raise RuntimeError(
            f"Ego snapshot requires a finite 4x4 transform, got shape={values.shape!r}."
        )
    return tuple(tuple(float(value) for value in row) for row in values)


def _mutable_transform(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64)


def ego_capture_filename(*, physics_step: int, capture_id: int) -> str:
    """Name one RGB-D capture without collisions inside a physics tick."""

    return f"step_{int(physics_step):06d}_capture_{int(capture_id):06d}_rgbd.png"


@dataclass(frozen=True)
class EgoObservationSnapshot:
    """One stamped ego measurement: RGB-D, extrinsics, and base pose.

    Every field is taken at the same instant. ``observe()`` may only publish
    world geometry copied from this object; it must not mix in a later USD
    read. ``capture_id`` increases once per capture. ``physics_step`` is
    ``IsaacAutonomousEnv._physics_steps`` at that capture.
    """

    capture_id: int
    physics_step: int
    rgb: Any
    depth: Any
    visibility_depth: Any
    fov_deg: float
    width: int
    height: int
    camera_path: str | None
    world_T_cam: tuple[tuple[float, ...], ...]
    robot_position: tuple[float, ...]
    robot_orientation: Any
    robot_yaw_rad: float
    robot_root_T_world: tuple[tuple[float, ...], ...]
    camera_height_m: float
    camera_pitch_deg: float
    camera_gaze_tilt_deg: float
    body_posture: str
    depth_status: str
    visibility_depth_status: str
    rgbd_png: Any = None
    image_path: str | None = None
    bounding_box_2d_tight: Any = None
    depth_valid_ratio: float | None = None
    depth_shape: Any = None


class IsaacAutonomousEnv:
    """Real-scene backend that exposes RGBD but keeps simulator truth hidden."""

    #: Set by ``_attach_current_stage_for_navmesh`` so the post-bake code knows
    #: whether the TARSphere/PhysX handles need post-bake repair.
    _navmesh_attach_performed: bool = False

    def __init__(
        self,
        episode: TARSphereEpisode,
        *,
        headless: bool = False,
        image_dir: Optional[str] = None,
        viewer_mode: str = "third_person",
        enable_ego_sensors: bool = True,
        enable_streaming: bool | None = None,
        enable_human_ego_cameras: bool = True,
        room_observer_view_names: tuple[str, ...] = (),
        room_observer_resolution: tuple[int, int] = (640, 360),
        room_observer_stride: int = 1,
        room_observer_max_frames: int | None = None,
        room_observer_gate_presentation_updates: bool = False,
        continuous_ego_video: bool = False,
        continuous_ego_video_stride: int = 8,
        continuous_ego_video_output_dir: str | None = None,
        continuous_ego_video_slug: str | None = None,
        agent_harness_config: AgentHarnessConfig | None = None,
        render_mode: RenderMode | str = RenderMode.EVALUATION,
        human_motion_backend: str = "kinematic",
    ):
        if viewer_mode not in VIEWER_MODES:
            raise ValueError(
                f'Unknown viewer mode "{viewer_mode}". Expected one of {sorted(VIEWER_MODES)}.'
            )
        self.episode = episode
        if human_motion_backend not in {"kinematic", "animation"}:
            raise ValueError("human_motion_backend must be kinematic or animation")
        self.human_motion_backend = human_motion_backend
        # Immutable bake identity. ``self.episode`` changes on soft reset, so
        # certification must never use it to describe the scene already open
        # in Isaac.
        self._loaded_scene_asset_path = str(episode.scene.asset_path)
        self._loaded_scene_usd_path: Path | None = None
        self.agent_harness_config = agent_harness_config or AgentHarnessConfig()
        self.render_mode = RenderMode(render_mode)
        if (
            room_observer_gate_presentation_updates
            and self.render_mode is not RenderMode.DEMO
        ):
            raise ValueError(
                "Presentation RenderProduct update gating requires explicit demo mode."
            )
        if (
            room_observer_gate_presentation_updates
            and "presentation_ego" not in room_observer_view_names
        ):
            raise ValueError(
                "Presentation RenderProduct update gating requires presentation_ego."
            )
        self.headless = headless
        self.viewer_mode = viewer_mode
        self.enable_ego_sensors = bool(enable_ego_sensors)
        # Direct teleoperation callers retain the historical headless stream
        # unless they opt out explicitly. Batch SceneServer callers pass False:
        # WebRTC otherwise owns an unnecessary second viewport/render pipeline.
        self.enable_streaming = (
            self.headless if enable_streaming is None else bool(enable_streaming)
        )
        self.enable_human_ego_cameras = bool(enable_human_ego_cameras)
        self.room_observer_view_names = tuple(room_observer_view_names)
        self.room_observer_resolution = (
            int(room_observer_resolution[0]),
            int(room_observer_resolution[1]),
        )
        self.room_observer_stride = int(room_observer_stride)
        self.room_observer_max_frames = room_observer_max_frames
        #: Human id the room observer frames instead of the robot.  ``None`` on
        #: every evaluation path; set only by the Human-animation verification
        #: runner.  See ``_live_room_observer_pose``.
        self.room_observer_focus_human: str | None = None
        self._room_observer = (
            RoomObserverRecorder(
                view_names=self.room_observer_view_names,
                resolution=self.room_observer_resolution,
                capture_stride=self.room_observer_stride,
                max_frames_per_view=self.room_observer_max_frames,
                gate_presentation_updates=room_observer_gate_presentation_updates,
            )
            if self.room_observer_view_names
            else None
        )
        self._last_room_observer_paths: Dict[str, str] = {}
        self._continuous_ego_video_enabled = bool(continuous_ego_video)
        self._continuous_ego_video_output_dir = (
            Path(continuous_ego_video_output_dir)
            if continuous_ego_video_output_dir
            else None
        )
        self._continuous_ego_video_slug = (
            str(continuous_ego_video_slug).strip()
            if continuous_ego_video_slug
            else None
        )
        self._continuous_ego_video = ContinuousEgoVideoRecorder(
            capture_stride=int(continuous_ego_video_stride),
            physics_hz=float(SIMULATION_PHYSICS_HZ),
            resolution=ROBOT_EGO_CAMERA_RESOLUTION,
        )
        self.image_dir = Path(image_dir) if image_dir else None
        if self.image_dir is not None:
            self.image_dir.mkdir(parents=True, exist_ok=True)
        self._env = None
        self._obs: Dict[str, Any] = {}
        self._last_utterance: Optional[Dict[str, Any]] = None
        self._interaction_feedback = ""
        self._accepted_handovers: Dict[str, str] = {}
        self._last_handover_claim: Optional[Dict[str, Any]] = None
        self._physics_steps = 0
        # Populated by ``_configure_human_animation_clock`` once characters
        # spawn.  Stays None on the scene-prepare-only path, where there are no
        # characters to animate.
        self._animation_clock: Optional[Dict[str, Any]] = None
        self._ego_snapshot: EgoObservationSnapshot | None = None
        self._next_ego_capture_id = 0
        self._render_count = 0
        self._physics_step_limit: int | None = None
        self._abort_requested: Callable[[], bool] = lambda: False
        self._robot_body = RobotBodyExecutor()
        self._g1_attachment: Any = None
        self._episode_asset_adapter: Optional[EpisodeAssetAdapter] = None
        self._physical_scene_preparer: Optional[PhysicalScenePreparer] = None
        self._scene_asset_unit_in_meters: float | None = None
        self._last_g1_arm_ik: Optional[Dict[str, Any]] = None
        self._g1_carried_hand: Optional[str] = None
        self._g1_carry_finger_positions: Optional[tuple[float, ...]] = None
        self._g1_body_height_command_m = G1_STANDING_HEIGHT_COMMAND_M
        self._g1_closing_reach_deflection_capture: Dict[str, Any] | None = None
        self._g1_excluded_grasp_centres: list[np.ndarray] = []
        self._grasp_grounding_by_target: Dict[str, Dict[str, Any]] = {}
        self._grasp_grounding_attempts: list[Dict[str, Any]] = []
        self._active_grasp_grounding: Dict[str, Any] | None = None
        self._runtime_grasp_backend: Any = None
        self._robot_contact_monitor: Any = None
        self._human_body = self._create_human_body_executor()
        self._human_positions: Dict[str, list] = {
            human.id: list(human.start_position)
            for human in getattr(episode, "humans", [])
        }
        # Character-style yaw (heading + AnimPeople 90 deg offset), matching
        # the runtime write-back at state sync; see the initial-yaw block in
        # _prepare_human_runtime for the compensation.
        self._human_yaws: Dict[str, float] = {
            human.id: (
                float(getattr(human, "start_yaw", 0.0) or 0.0)
                + HUMAN_CHARACTER_YAW_OFFSET_RAD
                if float(getattr(human, "start_yaw", 0.0) or 0.0) != 0.0
                else 0.0
            )
            for human in getattr(episode, "humans", [])
        }
        self._human_action_log: list[Dict[str, Any]] = []
        self._human_ego_cameras: Dict[str, Any] = {}
        self._human_ego_camera_paths: Dict[str, str] = {}
        self._human_ego_camera_errors: Dict[str, str] = {}
        self._human_ego_camera_poses: Dict[str, Dict[str, Any]] = {}
        self._human_collision_proxy_paths: Dict[str, str] = {}
        self._human_collision_errors: Dict[str, str] = {}
        self._robot_camera_mount_log_emitted = False
        # The agent's commanded ego-camera gaze tilt in degrees, negative down,
        # absolute against the profile's mount calibration.  It is the one part
        # of the camera's orientation the agent owns, it persists across decision
        # steps until changed, and it returns to 0 on reset because a new episode
        # starts from the mounted zero.  See ``set_camera_gaze_tilt_deg``.
        self._camera_gaze_tilt_deg = 0.0
        # Official DualVLN Habitat LOOKDOWN extra on the shared ego camera.
        # It is non-zero only inside one navigate call and is restored before
        # control returns to the evaluated Agent.
        self._dualvln_look_down_extra_deg = 0.0
        # Local camera orientation after Isaac XFormPrim init, before TARSPHERE
        # bakes mount+gaze into ``xformOp:orient``.  Sidecar RotateX ops are
        # dropped the moment the parent link moves, so rest ⊗ Rx(pitch) is
        # rewritten onto the Fabric-tracked orient every physics tick.
        self._camera_rest_local_quat: tuple[float, float, float, float] | None = None
        self._camera_gaze_persist_suspended = False
        self._static_scene_bodies_prepared = False
        # LiDAR uses live PhysX colliders. The audit belongs to the loaded
        # stage and is reset whenever a new episode may rebuild that stage.
        self._lidar_collider_audit: Dict[str, Any] | None = None
        self._navmesh_baked = False  # Track if NavMesh has been baked this session
        self._navmesh: Any = None
        self._decision_render_state: tuple[int, int, bool] | None = None
        self._private_mapping_observer: Callable[..., Dict[str, Any]] | None = None
        self._private_mapping_action_space: list[str] = []
        self._private_mapping_last_physics_step = 0
        self._private_mapping_frame_index = 0
        self._private_mapping_stats: Dict[str, Any] = {
            "captures": 0,
            "integrated_frames": 0,
            "render_seconds": 0.0,
            "fusion_seconds": 0.0,
            "artifact_seconds": 0.0,
            "mapping_callback_seconds": 0.0,
            "observed_area_gain_m2": 0.0,
        }
        # When True, reset() skips AnimPeople NavMesh bake (H1 / --no-navmesh).
        self._skip_navmesh = False
        # Pose history ring buffers for the Runtime Director (robot + NPC trajectories).
        self._robot_pose_history: list[list[float]] = []
        self._human_pose_history: Dict[str, list[list[float]]] = {
            hid: [] for hid in self._human_positions
        }
        self._decision_step: int = 0
        # Robot spawn baked into the env config at _create_env time.  Episodes
        # loaded later via load_episode() may use a different start on the same
        # scene; reset() teleports the robot from this configured pose to the
        # current episode's start so one baked NavMesh serves them all.
        self._configured_robot_start: Optional[tuple] = None
        self._configured_robot_orientation: Optional[tuple] = None
        # Set by ``load_episode`` once the TARSphere env already exists.  The
        # next ``reset()`` then performs a *soft reset* — it keeps the loaded
        # scene, the baked NavMesh, the robot articulation, and the Replicator
        # camera pipeline, and only resets physics + the robot's pose + the human
        # actors.  This is what lets a later episode on the SAME scene reuse one
        # bake: TARSphere's native episode advance reloads ``/World/env_0``,
        # which corrupts the SDG/Og graph beyond recovery, so we bypass it.
        self._pending_soft_reset = False

    def configure_agent_harness(self, config: AgentHarnessConfig) -> None:
        """Install the typed run configuration before episode execution."""

        if not isinstance(config, AgentHarnessConfig):
            raise TypeError("IsaacAutonomousEnv requires an AgentHarnessConfig")
        self.agent_harness_config = config

    def configure_private_mapping_observer(
        self,
        *,
        observer: Callable[..., Dict[str, Any]] | None,
        action_space: Sequence[str],
    ) -> None:
        """Bind the Agent's private occupancy sink for this episode.

        The environment owns sensor timing; the Agent owns map state.  This
        narrow callback crosses that boundary without invoking policy,
        perception, or memory code.
        """

        if observer is not None and not callable(observer):
            raise TypeError("private mapping observer must be callable or None")
        self._private_mapping_observer = observer
        self._private_mapping_action_space = [str(name) for name in action_space]
        # The reset observation has already been fused at this exact physics
        # time. Start the motion clock there so a locomotion action does not
        # pay for an almost-identical render on its first 240 Hz tick.
        snapshot = getattr(self, "_ego_snapshot", None)
        if observer is not None and snapshot is not None:
            self._private_mapping_last_physics_step = int(snapshot.physics_step)

    def _create_human_body_executor(self):
        """Select the explicit Human motion backend for this environment."""
        if self.human_motion_backend == "kinematic":
            from tarsphere.infra.control.kinematic_human_executor import (
                KinematicHumanExecutor,
                UsdKinematicCharacters,
            )

            return KinematicHumanExecutor(
                characters=UsdKinematicCharacters(
                    default_character=DEFAULT_ISAAC_CHARACTER
                ),
                route_resolver=self._resolve_kinematic_human_route,
                pose_sync=self._sync_human_body_state,
            )
        return AnimPeopleBodyExecutor(
            pose_sync=self._sync_human_body_state,
            default_character=DEFAULT_ISAAC_CHARACTER,
            capture_native_motion_trace=_bool_env(
                "TARSPHERE_DEMO_NATIVE_MOTION_TRACE", False
            ),
        )

    def _resolve_kinematic_human_route(self, human_id, start, target, action):
        """Use scene A* through authored Human stops without endpoint snapping."""
        from tarsphere.harness.perception.backends.gt_bev import OfficialOccupancyBevMap
        from tarsphere.harness.planner.planner import Planner
        from tarsphere.infra.control.kinematic_human_route import resolve_human_route

        if action.get("route_source") == "live_robot_aware_astar":
            route = [
                [float(value) for value in point]
                for point in action.get("waypoints") or ()
            ]
            if not route or math.dist(route[-1], target) > 1e-6:
                raise ValueError("Live Robot-aware Human route must end at its target.")
            return route
        planner = Planner(
            OfficialOccupancyBevMap(
                occupancy_path=_compiled_occupancy_path(self.episode)
            ),
            clearance_radius_m=HUMAN_COLLISION_RADIUS_METERS,
            # Authoring certifies these static Human routes with a 30 s A*
            # budget. The generic 5 s online-navigation budget can expire
            # under full Isaac concurrency and report a reachable route as
            # "unreachable", invalidating an otherwise valid episode.
            time_limit_s=30.0,
        )
        return resolve_human_route(
            planner,
            start=start,
            target=target,
            waypoints=action.get("waypoints"),
        )

    def plan_spatial_blocker_clearance(self, human_id: str) -> dict[str, Any]:
        """Plan the blocker's live exit around the current Robot position."""

        from tarsphere.harness.perception.backends.gt_bev import (
            OfficialOccupancyBevMap,
        )
        from tarsphere.contracts.task_profile import SpatialBlockingProfile
        from tarsphere.infra.control.spatial_blocker_clearance import (
            plan_spatial_blocker_clearance,
        )

        profile = getattr(self.episode.evaluation, "task_profile", None)
        if not isinstance(profile, SpatialBlockingProfile):
            raise RuntimeError(
                "Dynamic blocker clearance requires a spatial_blocking episode."
            )
        if str(human_id) != profile.blocking_human_id:
            raise ValueError(f"{human_id!r} is not the configured blocking Human.")
        robot_position = self._robot_position()
        blocker_position = self.get_human_position(human_id)
        if robot_position is None:
            raise RuntimeError("Dynamic blocker clearance requires a live Robot pose.")
        # Use the current animation-synced poses. Authored starts become stale
        # as soon as another Human moves; omitting them can select an exit
        # inside another person's body or restore a Robot route through them.
        other_human_positions = [
            self.get_human_position(other_id)
            for other_id in self._human_positions
            if str(other_id) != str(human_id)
        ]
        route = (self.episode.world_knowledge or {}).get("episode_route") or {}
        navigate = route.get("navigate_to_target") or {}
        points = list(navigate.get("points") or ())
        by_id = {
            str(point.get("point_id") or ""): point.get("position_xy")
            for point in points
        }
        contact = by_id.get("passage_contact")
        goal = points[-1].get("position_xy") if points else None
        if contact is None or goal is None:
            raise RuntimeError(
                "Spatial blocking runtime requires passage_contact and task "
                "target route evidence."
            )
        plan = plan_spatial_blocker_clearance(
            OfficialOccupancyBevMap(
                occupancy_path=_compiled_occupancy_path(self.episode)
            ),
            blocker_position=blocker_position,
            robot_position=robot_position,
            passage_contact_xy=contact,
            task_goal_xy=goal,
            other_human_positions=other_human_positions,
        )
        return plan.as_dict()

    def load_episode(
        self,
        episode: TARSphereEpisode,
        *,
        reuse_navmesh: bool = True,
    ) -> None:
        """Swap in a new episode without reloading the scene.

        This is the core of the persistent-worker fast path: the TARSphere
        ``Env`` (with its loaded GRScene USD) and the baked NavMesh are kept, so
        a follow-up ``reset()`` skips the ~160 s bake and only refreshes runtime
        handles (~26 s).

        Constraints (fail-loud):
        - When ``reuse_navmesh`` is True and an env is already loaded, the new
          episode must reference the **same scene** as the loaded one.  A
          different scene needs a fresh env (call ``close()`` and construct a new
          one).  A different **robot start is allowed** on the same scene:
          ``reset()`` teleports the robot to the new start, so one bake serves
          every episode on that scene.
        - When ``reuse_navmesh`` is False, the next ``reset()`` performs a full
          rebake (e.g. the worker detected a scene change and recreated the env
          but kept the wrapper).

        The human position/yaw caches are rebuilt from the new episode; camera,
        collision-proxy, and animation handles are refreshed on the next
        ``reset()``.
        """
        if reuse_navmesh and self._env is not None:
            loaded_scene = self._loaded_scene_asset_path
            if episode.scene.asset_path != loaded_scene:
                raise ValueError(
                    "load_episode scene mismatch: env was created for "
                    f'"{loaded_scene}" but the new episode uses '
                    f'"{episode.scene.asset_path}". Call close() and create a '
                    "new IsaacAutonomousEnv to switch scenes."
                )

        if reuse_navmesh and self._navmesh_baked:
            self.validate_human_navmesh_contract(episode)

        # If the TARSphere env is already live (same scene), arm a soft reset:
        # the next reset() keeps the scene/robot/NavMesh/camera pipeline and only
        # resets physics + the robot pose + humans.  This avoids TARSphere's
        # episode advance, which reloads /World/env_0 and corrupts the SDG graph.
        if self._env is not None:
            self._pending_soft_reset = True

        self._close_robot_contact_monitor()
        if self._g1_attachment is not None:
            restored = self._g1_attachment.restore_initial_state()
            if restored:
                print(
                    f"[INFO] Restored manipulation objects for soft reset: {restored}",
                    flush=True,
                )
        self._close_g1_attachment()
        self._restore_episode_asset_adaptation()
        self._restore_physical_scene_preparation()

        self.episode = episode
        self._human_positions = {
            human.id: list(human.start_position)
            for human in getattr(episode, "humans", [])
        }
        self._human_yaws = {
            human.id: (
                float(getattr(human, "start_yaw", 0.0) or 0.0)
                + HUMAN_CHARACTER_YAW_OFFSET_RAD
                if float(getattr(human, "start_yaw", 0.0) or 0.0) != 0.0
                else 0.0
            )
            for human in getattr(episode, "humans", [])
        }
        # Clear per-episode transient state so the next reset rebuilds it.
        self._human_action_log.clear()
        self._last_utterance = None
        self._interaction_feedback = ""
        self._accepted_handovers = {}
        self._last_handover_claim = None
        self._physics_steps = 0
        self._render_count = 0
        self._next_ego_capture_id = 0
        self._private_mapping_last_physics_step = 0
        self._private_mapping_frame_index = 0
        self._private_mapping_stats = {
            "captures": 0,
            "integrated_frames": 0,
            "render_seconds": 0.0,
            "fusion_seconds": 0.0,
            "artifact_seconds": 0.0,
            "mapping_callback_seconds": 0.0,
            "observed_area_gain_m2": 0.0,
        }
        self._invalidate_ego_snapshot()
        self._last_g1_arm_ik = None
        self._g1_carried_hand = None
        self._g1_carry_finger_positions = None
        self._g1_body_height_command_m = G1_STANDING_HEIGHT_COMMAND_M
        self._g1_closing_reach_deflection_capture = None
        self._g1_excluded_grasp_centres = []
        self._grasp_grounding_by_target = {}
        self._grasp_grounding_attempts = []
        self._active_grasp_grounding = None
        self._lidar_collider_audit = None
        if not reuse_navmesh:
            self._navmesh_baked = False
            self._navmesh = None

    @property
    def loaded_scene_asset_path(self) -> str:
        """Authored asset path used to create the persistent Isaac env."""

        return self._loaded_scene_asset_path

    @property
    def loaded_scene_usd_path(self) -> Path | None:
        """Concrete USD file actually opened after runtime material resolution."""

        return self._loaded_scene_usd_path

    def reset(
        self,
        *,
        scene_prepare_only: bool = False,
        skip_navmesh: bool = False,
    ) -> Dict[str, Any]:
        """Reset for an episode, or prepare the persistent scene for later serve.

        ``scene_prepare_only=True`` (``SceneServer.bake_scene`` / ``--bake-only``)
        creates the env, bakes NavMesh (unless ``skip_navmesh``), and snapshots a
        standing G1. It also pays the cold RTX/RGBD initialization so the
        persistent server owns a live render product. Human spawn and
        human-ego cameras remain deferred to the first ``/run_episode`` soft
        reset, whose camera verification is intentionally short.

        ``skip_navmesh=True`` skips AnimPeople NavMesh bake (stationary
        humans / no human ``goto``). Caller must fail-loud earlier if the episode
        needs locomotion; ``_navmesh_baked`` stays False. The flag is sticky on
        this env for the process lifetime (later ``reset()`` without the kwarg
        does not re-enable baking).
        """
        t_reset_start = time.monotonic()
        # Sensor perturbations use this episode-local sequence. Reset it even
        # when the same persistent simulator process serves multiple episodes
        # or repeats one episode after an infrastructure retry.
        self._next_ego_capture_id = 0
        # Reset warmup and episode spawn teleports are not Agent execution.
        self._execution_path = None
        if scene_prepare_only and self._pending_soft_reset:
            raise RuntimeError(
                "scene_prepare_only cannot combine with a pending soft reset."
            )
        # Sticky process policy: once skipped, stay skipped (scene server
        # --no-navmesh). Soft resets from AutonomousEpisodeRunner call reset()
        # with defaults and must not re-arm a multi-minute bake.
        if skip_navmesh:
            self._skip_navmesh = True

        if self._env is None:
            self._env = self._create_env()
        self._install_camera_gaze_persistence()

        # A repeated reset of the same wrapper must be as independent as a
        # load_episode soft reset. Remove the previous family edit before the
        # scene/world reset and rebuild it from the current compiled contract.
        self._restore_episode_asset_adaptation()
        self._restore_physical_scene_preparation()

        t0 = time.monotonic()
        self._enable_anim_people_extensions()
        # Do not bother unloading isaacsim.sensors.physics: it does not work.
        # A 2026-08-17 reproducer disabled it right after SimulationApp init
        # (extension manager reported enabled_now=False) and still died with
        # the identical libisaacsim.sensors.physics.plugin.so!+0x17276 frame,
        # under carbOnPluginStartup. Carb keeps the native plugin loaded and
        # its PhysX contact-report callback subscribed after the Kit extension
        # is disabled, so the crashing consumer survives the teardown.
        #
        # That run also refutes the 2026-08-15 note this replaces: the monitor
        # subscribes through omni.physx, not through this extension, and it
        # logged 300 reports / ~2600 pairs per action with the extension off.
        # Contact reporting does not depend on the plugin.
        self._close_robot_contact_monitor()
        self._cleanup_human_ego_cameras()
        # A soft reset (armed by load_episode on an already-live env) keeps the
        # scene/robot/NavMesh/camera pipeline alive.  The first episode takes
        # the full TARSphere reset; later episodes use a zero-velocity step
        # plus explicit robot/human pose reset below.  Do not call
        # TARSphere/Isaac ``world.reset(soft=True)`` here: in practice it
        # invokes task.cleanup() and can remove robot registry entries while
        # the persistent scene is supposed to remain live.
        soft = self._pending_soft_reset
        self._pending_soft_reset = False
        zero_speed = {"move_by_speed": [0.0, 0.0, 0.0]}
        self._camera_gaze_persist_suspended = True
        try:
            if soft:
                # Restore the standing articulation BEFORE any physics step.
                # Stepping while the robot is already fallen (common after bake→idle
                # with a live timeline and no loco command) digs the policy into a
                # bad state; teleport alone cannot recover joint/controller memory.
                self._restore_robot_episode_reset_state()
                # Restore/apply the complete episode placement before the first
                # live tick; source transforms may be in centimetres and can
                # put a pickable at the robot spawn until preparation runs.
                self._apply_physical_scene_preparation()
                self._obs = self._soft_reset_keep_scene()
            else:
                self._obs, _ = self._env.reset()
        finally:
            self._camera_gaze_persist_suspended = False
        self._apply_episode_lighting()
        self._align_physics_scene_timesteps(expected_hz=SIMULATION_PHYSICS_HZ)
        self._prepare_static_scene_bodies()
        t_env_reset = time.monotonic() - t0
        t0_phase = time.monotonic()

        # Camera gaze is episode state.  A soft reset preserves the live USD
        # camera prim, so reset the logical command first and then rewrite the
        # same calibrated mount pose on both cold and soft paths.  Otherwise a
        # previous episode's final gaze silently becomes the next episode's
        # initial observation while memory reports zero.
        self._camera_gaze_tilt_deg = 0.0
        # A cold TARSphere reset rebuilds the camera XFormPrim; recapture
        # rest from that prim.  Soft reset keeps the live camera, so the rest
        # orientation captured earlier remains the asset pose to compose with.
        if not soft:
            self._camera_rest_local_quat = None
        if self.enable_ego_sensors:
            self._dualvln_look_down_extra_deg = 0.0
            self._configure_robot_ego_camera()
            self._validate_dualvln_ego_camera_contract()
        self._physics_steps = 0
        self._render_count = 0
        self._invalidate_ego_snapshot()
        self._last_utterance = None
        self._accepted_handovers = {}
        self._last_handover_claim = None
        # Remove stale proxy/camera prims from the previous episode before the
        # new episode's humans are spawned — otherwise a persistent worker leaks
        # one set of collision/ego prims per episode when human ids change.
        self._cleanup_human_collision_proxies()
        self._human_collision_proxy_paths.clear()
        self._human_collision_errors.clear()
        self._human_action_log.clear()
        # Timeline must be playing before NavMesh bake — the C++ baker
        # requires an active timeline to properly process stage updates.
        self._start_human_animation_timeline()

        # omni.anim.graph.core builds its CharacterManager only from the
        # AnimationGraph stage-update node's *attach* callback, and only when an
        # AnimationGraph prim is already on the stage.  Authoring Biped_Setup
        # here -- before the NavMesh path's stage attach -- is what makes that
        # attach initialize the manager; author it at spawn time instead and the
        # manager is never built, so ag.get_character() answers None forever and
        # every character's BehaviorScript refuses to initialize.
        if not soft and isinstance(self._human_body, AnimPeopleBodyExecutor):
            self._human_body.author_character_animation_graph()

        # Official AnimPeople GoTo requires a baked NavMesh before spawning.
        # Skipped on a soft reset: the NavMesh and SDG pipeline persist from the
        # first episode (no scene reload, no PhysX re-attach), so no refresh is
        # needed and a rebake must never happen here.
        # ``skip_navmesh`` (stationary Humans / --no-navmesh): never bake; leave
        # ``_navmesh_baked`` False so GoTo cannot be mistaken for ready.
        t_camera_graph = time.monotonic() - t0_phase
        t0 = time.monotonic()
        skip_nav = bool(
            self.human_motion_backend == "kinematic"
            or getattr(self, "_skip_navmesh", False)
            or skip_navmesh
        )
        if soft:
            pass
        elif skip_nav:
            print(
                "[INFO] Skipping AnimPeople NavMesh bake "
                "(skip_navmesh=True / --no-navmesh).",
                flush=True,
            )
            if isinstance(self._human_body, AnimPeopleBodyExecutor):
                self._prepare_anim_people_character_runtime()
        elif isinstance(self._human_body, AnimPeopleBodyExecutor):
            self._configure_anim_people_navigation()
            # The bake path already attached the stage (with the AnimationGraph
            # now authored above) and refreshed the TARSphere runtime, so the
            # CharacterManager exists and spawning is unblocked.
            self._human_body.mark_character_runtime_prepared()
        t_navmesh = time.monotonic() - t0

        if not scene_prepare_only and self._navmesh_baked:
            self.validate_human_navmesh_contract(self.episode)

        # Episode assets are tabletop-scale and therefore do not participate
        # in Human NavMesh baking. Apply them after the bake/reattach boundary,
        # then require measured settling before spawning actors or configuring
        # manipulation/evaluation handles.
        # Time asset preparation separately from NavMesh baking and camera warmup.
        t_assets = 0.0
        t_grasp_asset = 0.0
        if not scene_prepare_only:
            t0_phase = time.monotonic()
            self._apply_and_settle_episode_assets(zero_speed)
            t_assets = time.monotonic() - t0_phase

            # Asset certification keeps the scene, table, Robot pose, camera,
            # and grasp executor fixed while replacing only the selected
            # object.  The replacement must exist before attachment handles
            # and the first Agent-ready camera frame are created.
            from tarsphere.infra.manipulation.grasp_asset_certification import (
                synchronize_grasp_asset,
            )

            t0_phase = time.monotonic()
            synchronize_grasp_asset(self.episode)
            if not soft:
                self._apply_physical_scene_preparation()
            import omni.usd

            live_stage = omni.usd.get_context().get_stage()
            target_pose_diagnostic = validate_live_manipulation_target_pose(
                live_stage,
                self.episode,
            )
            if target_pose_diagnostic is not None:
                print(
                    "[INFO] Live manipulation target matches episode fact: "
                    f"{target_pose_diagnostic}",
                    flush=True,
                )
            t_grasp_asset = time.monotonic() - t0_phase

        from tarsphere.infra.manipulation.grasp_reference import (
            synchronize_grasp_reference,
        )

        # Grasp-system calibration needs the reference prim before attachment
        # handles bind to it. Idempotent and a no-op for ordinary episodes.
        grasp_reference_changed = synchronize_grasp_reference(self.episode)

        # Task pickables become dynamic rigid bodies here, once, as part of
        # episode bring-up. Furniture stays static. Grasp execution never
        # retoggles this identity. Camera warmup then advances PhysX, so
        # authored mesh overlap seats before the first Agent observation.
        t0_phase = time.monotonic()
        if not scene_prepare_only:
            self._configure_g1_attachment()
        t_g1_attach = time.monotonic() - t0_phase

        # Teleport the robot to this episode's start AFTER the NavMesh refresh,
        # which re-initializes the articulation handles (PhysX re-attach).  For a
        # persistent worker this is what lets episodes with different robot
        # starts share one baked NavMesh.  No-op when the start matches the
        # configured spawn (first episode / same-start episodes).
        # NOTE: a timeline stop/start around the human recycle was tried and
        # rejected -- it clears omni.anim.people C++ state but irrecoverably
        # corrupts the Replicator SDG/Og graph (Invalid object in Py_Graph),
        # which the post-NavMesh recovery cannot rebuild.  So the soft path
        # keeps the timeline live and relies on (a) the per-character GoTo
        # dedupe in AnimPeopleBodyExecutor (prevents the degenerate-path
        # NavigationPath::Advance crash on a recycled character) plus (b)
        # despawn/respawn only differing humans.  Independence is validated
        # empirically per scene-group.
        t0_phase = time.monotonic()
        robot_start_teleported = self._apply_robot_start_pose()
        if robot_start_teleported:
            self._obs, _, _, _, _ = self._env.step(action=zero_speed)
            self._physics_steps += 1
        t_start_pose = time.monotonic() - t0_phase
        t0_phase = time.monotonic()
        if scene_prepare_only:
            print(
                "[INFO] Scene prepare: deferring human spawn to first episode.",
                flush=True,
            )
        else:
            self._human_body.reset(self.episode.humans)
            if self.human_motion_backend == "animation":
                self._human_body.set_physics_step(self._physics_steps)
                self._human_body.step(0.0)
                native_trace = getattr(
                    self._human_body, "poll_native_motion_trace", None
                )
                if callable(native_trace):
                    native_trace(observation_boundary="after_actor_reset")
            # The characters exist now, so give them a working animation clock
            # before any command is dispatched.  Verified not to disturb the
            # physics dt, which robot dynamics run on.
            self._animation_clock = (
                self._configure_human_animation_clock()
                if self.human_motion_backend == "animation"
                else None
            )
            if self._animation_clock is not None:
                self._assert_physics_dt_unchanged(self._animation_clock)
            # Apply initial yaw from episode spec to the body executor's state
            # so the character model and ego camera face the correct direction.
            # AnimPeople character yaw is offset from the world heading by
            # +90 deg (see AnimPeopleBodyExecutor.CHARACTER_FACING_YAW_OFFSET_DEG,
            # measured on walk and sit command paths); the initial standing
            # pose needs the same compensation or the character faces 90 deg
            # away from the authored direction.
            facing_offset_rad = HUMAN_CHARACTER_YAW_OFFSET_RAD
            for human in getattr(self.episode, "humans", []):
                if self.human_motion_backend == "kinematic":
                    continue
                start_yaw = float(getattr(human, "start_yaw", 0.0) or 0.0)
                if start_yaw == 0.0:
                    continue
                body_state = self._human_body._states.get(human.id)
                if body_state is not None:
                    body_state.yaw_rad = start_yaw + facing_offset_rad
                    self._human_yaws[human.id] = start_yaw
        t_human_spawn = time.monotonic() - t0_phase

        # Warm up physics until the camera returns actual image data.
        # GRScenes is a large scene; the first render may take many physics
        # steps before the camera pipeline produces non-empty tensors.
        # After NavMesh refresh, Replicator annotators may be detached and
        # need a few physics steps to re-attach — catch those transient errors.
        # Keep warmup bounded. GRScene/RTX can return visible but unconverged
        # frames during the first few render intervals; do not expose those to
        # the agent or recorder on a cold start.
        if soft:
            min_camera_warmup = ROBOT_CAMERA_SOFT_MIN_WARMUP_STEPS
            warmup_mode = "soft"
        elif scene_prepare_only:
            min_camera_warmup = ROBOT_CAMERA_PREPARE_MIN_WARMUP_STEPS
            warmup_mode = "prepare"
        else:
            min_camera_warmup = ROBOT_CAMERA_MIN_WARMUP_STEPS
            warmup_mode = "cold"
        t0 = time.monotonic()
        if self.enable_ego_sensors:
            camera_ready = False
            last_camera_status: Dict[str, Any] = {"status": "not_checked"}
            for i in range(ROBOT_CAMERA_MAX_WARMUP_STEPS):
                try:
                    self._obs, _, _, _, _ = self._env.step(action=zero_speed)
                except (KeyError, RuntimeError, Exception) as warmup_exc:
                    # Camera annotator may be detached after PhysX re-attach.
                    # Pump kit updates to let Replicator re-attach, then retry.
                    exc_name = type(warmup_exc).__name__
                    if i < 5 or i % 20 == 0:
                        print(
                            f"[WARN] Warmup step {i} error ({exc_name}: "
                            f"{warmup_exc}), pumping kit updates.",
                            flush=True,
                        )
                    self._pump_kit_updates(count=5)
                    continue
                self._physics_steps += 1
                camera = self._obs.get("sensors", {}).get("camera", {})
                rgba = camera.get("rgba")
                depth_raw = camera.get("depth")
                if rgba is None:
                    last_camera_status = {
                        "status": "missing_rgba",
                        "camera_keys": sorted(str(key) for key in camera.keys()),
                    }
                    continue
                rgba_arr = np.asarray(rgba.cpu() if hasattr(rgba, "cpu") else rgba)
                if rgba_arr.size == 0:
                    last_camera_status = {"status": "empty_rgba"}
                    continue
                rgb = (
                    rgba_arr[0, :, :, :3] if rgba_arr.ndim == 4 else rgba_arr[:, :, :3]
                )
                depth_frame, depth_info = prepare_depth_frame(depth_raw, rgb.shape[:2])
                del depth_frame
                rgb_stats = rgb_frame_stats(rgb)
                last_camera_status = {
                    "status": "checked",
                    "warmup_step": i + 1,
                    "rgb_stats": rgb_stats,
                    **depth_info,
                }
                if (
                    i + 1 >= min_camera_warmup
                    and rgb_frame_is_visible(rgb)
                    and depth_info.get("depth_status") == "valid"
                ):
                    print(
                        f"[INFO] Camera RGBD available after {i + 1} warmup steps "
                        f"(min={min_camera_warmup}, mode={warmup_mode}) "
                        f"rgb_stats={rgb_stats} depth_valid_ratio="
                        f"{depth_info.get('depth_valid_ratio', 0.0):.3f}",
                        flush=True,
                    )
                    camera_ready = True
                    break
            if not camera_ready:
                raise RuntimeError(
                    "Robot ego camera did not produce a visible RGBD frame during "
                    f"warmup after {ROBOT_CAMERA_MAX_WARMUP_STEPS} steps "
                    f"(min={min_camera_warmup}, mode={warmup_mode}); "
                    f"last_camera_status={last_camera_status}"
                )
        t_warmup = time.monotonic() - t0

        t0_phase = time.monotonic()
        if not scene_prepare_only:
            # Read the initialized animation graph before collision proxies or
            # task-event conditions consume Human world poses. A broken biped
            # payload can leave the visible parent correctly placed while its
            # runtime transform reports [0, 0]; synchronising that value would
            # create an invisible collision obstacle at the scene origin.
            self._human_body.step(0.0)
            self._assert_human_initial_poses()

        # Snapshot the canonical baseline only after the cold reset has reached
        # a demonstrably upright, policy-stabilized state. A soft reset restores
        # that immutable baseline and may teleport it to another episode start;
        # it must never overwrite the baseline with a slightly tipped body.
        self._assert_robot_initial_pose()
        self._assert_robot_upright(
            action_name=f"{warmup_mode}_reset_warmup",
            env_action=zero_speed,
        )
        if not soft:
            self._save_robot_episode_reset_state()
        t_settle = time.monotonic() - t0_phase

        if grasp_reference_changed and self.enable_ego_sensors:
            self._synchronize_robot_camera_after_scene_edit()

        t0 = time.monotonic()
        if scene_prepare_only:
            # Humans + ego cams are created on the first soft reset.
            self._configure_viewer()
        else:
            self._sync_all_human_prims()
            self._setup_human_collision_proxies()
            if self.enable_human_ego_cameras:
                self._setup_human_ego_cameras()
                self._warmup_human_ego_cameras()
            self._configure_viewer()
            self._configure_robot_contact_monitor()
        t_human = time.monotonic() - t0

        t_reset_total = time.monotonic() - t_reset_start
        # ``rest`` is every line of reset() no timer covers.  Run 8 reported
        # total=1456.4s against 351.7s of named phases, i.e. the interesting
        # number was the one nobody printed; keep it visible so that cannot
        # happen again.
        t_named = (
            t_env_reset
            + t_camera_graph
            + t_navmesh
            + t_assets
            + t_grasp_asset
            + t_g1_attach
            + t_start_pose
            + t_human_spawn
            + t_warmup
            + t_settle
            + t_human
        )
        print(
            f"[PERF] reset: env_reset={t_env_reset:.1f}s  "
            f"camera_graph={t_camera_graph:.1f}s  navmesh={t_navmesh:.1f}s  "
            f"assets={t_assets:.1f}s  grasp_asset={t_grasp_asset:.1f}s  "
            f"g1_attach={t_g1_attach:.1f}s  start_pose={t_start_pose:.1f}s  "
            f"human_spawn={t_human_spawn:.1f}s  "
            f"warmup={t_warmup:.1f}s  settle={t_settle:.1f}s  "
            f"human={t_human:.1f}s  rest={t_reset_total - t_named:.1f}s  "
            f"total={t_reset_total:.1f}s"
            f"{'  (scene_prepare_only)' if scene_prepare_only else ''}",
            flush=True,
        )
        observation = self.observe()
        if not scene_prepare_only:
            self._start_execution_measurements()
            self._start_room_observer_episode()
            self._start_continuous_ego_video()
        return observation

    @staticmethod
    def _synchronize_robot_camera_after_scene_edit() -> None:
        """Render one deterministic frame after adding the grasp fixture."""
        import omni.replicator.core as rep

        rep.orchestrator.step(
            rt_subframes=1,
            delta_time=0.0,
            pause_timeline=False,
            wait_for_render=True,
        )

    def _soft_reset_keep_scene(self) -> Dict[str, Any]:
        """Refresh observation in place, keeping scene/NavMesh/SDG pipeline.

        Used by the persistent worker for a later episode on the SAME scene.
        TARSphere's native episode advance reloads ``/World/env_0``, which
        destroys the Replicator SDG/Og graph the camera relies on, and recovering
        it is fragile.  Isaac ``world.reset(soft=True)`` also calls
        TARSphere task cleanup and can try to remove an already-unregistered
        robot from the scene registry.  Keep the scene alive instead: issue one
        zero-velocity step to stop active controllers and fetch a fresh
        observation. Caller must restore standing articulation state BEFORE
        this step so a fallen idle robot is not stepped while collapsed.
        ``_apply_robot_start_pose`` (called next in ``reset()``) then places the
        robot at this episode's start.
        """
        runner = getattr(self._env, "runner", None)
        world = getattr(runner, "_world", None)
        if world is None:
            raise RuntimeError("Cannot soft-reset: TARSphere World is unavailable.")
        print(
            "[INFO] Soft reset: keeping scene/NavMesh/SDG, skipping "
            "TARSphere world.reset to preserve scene registry.",
            flush=True,
        )
        # One zero-velocity step produces a fresh observation after restore.
        obs, _, _, _, _ = self._env.step(action={"move_by_speed": [0.0, 0.0, 0.0]})
        return obs

    def _save_robot_episode_reset_state(self) -> None:
        for robot in self._live_robots():
            save = getattr(robot, "save_episode_reset_state", None)
            if not callable(save):
                raise RuntimeError(
                    f"{type(robot).__name__} does not implement "
                    "save_episode_reset_state()."
                )
            save()

    def _restore_robot_episode_reset_state(self) -> None:
        for robot in self._live_robots():
            restore = getattr(robot, "restore_episode_reset_state", None)
            if not callable(restore):
                raise RuntimeError(
                    f"{type(robot).__name__} does not implement "
                    "restore_episode_reset_state()."
                )
            restore()

    def _live_robots(self) -> list[Any]:
        runner = getattr(self._env, "runner", None)
        tasks = getattr(runner, "current_tasks", {}) or {}
        robots = [
            robot
            for task in tasks.values()
            for robot in (getattr(task, "robots", {}) or {}).values()
        ]
        if not robots:
            raise RuntimeError("Persistent reset cannot resolve the live G1 robot.")
        return robots

    @staticmethod
    def _align_physics_scene_timesteps(*, expected_hz: float) -> None:
        """Make every live USD PhysicsScene step at the G1 control rate.

        GRScene references can carry their own ``PhysicsScene`` in addition to
        TARSphere's world scene. ``omni.anim.people`` subscribes to all live
        scenes and PhysX does not support per-scene stepping; mixed authored
        rates delay step notifications and break the 240 Hz G1 policy cadence.
        Aligning the scenes before character spawn keeps robot and Human
        simulation on one clock and removes that ambiguous runtime state.
        """
        if expected_hz <= 0:
            raise ValueError(f"expected_hz must be positive, got {expected_hz}")
        import omni.usd
        from pxr import PhysxSchema, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError(
                "Cannot align PhysicsScene timesteps: stage unavailable."
            )

        scenes = []
        failures = []
        for prim in stage.TraverseAll():
            if not prim.IsA(UsdPhysics.Scene):
                continue
            scene = PhysxSchema.PhysxSceneAPI.Apply(prim)
            attr = scene.GetTimeStepsPerSecondAttr()
            previous = attr.Get() if attr else None
            try:
                if not attr:
                    attr = scene.CreateTimeStepsPerSecondAttr()
                attr.Set(float(expected_hz))
                scenes.append((str(prim.GetPath()), previous))
            except Exception as exc:
                failures.append(f"{prim.GetPath()}: {exc}")

        if not scenes:
            raise RuntimeError("No live USD PhysicsScene found after env reset.")
        if failures:
            raise RuntimeError(
                "Failed to align all PhysicsScene timesteps: "
                + "; ".join(failures[:10])
            )
        print(
            "[INFO] PhysicsScene clocks aligned: "
            f"target_hz={float(expected_hz):.1f}, scenes={scenes}.",
            flush=True,
        )

    def _prepare_static_scene_bodies(self) -> None:
        """Make immutable GRScene geometry static while preserving task objects.

        GRScene authors hundreds of room and furniture meshes as rigid bodies,
        often with invalid mass properties. Keeping those meshes kinematic
        still makes PhysX instantiate every body on the first simulation step.
        The immutable scene needs collision geometry, not rigid-body dynamics,
        so its rigid bodies are disabled and become static colliders.

        A manipulation target is the sole exception. Real pickable objects also
        live below ``/World/env_0/scene`` and must remain dynamic for grasp,
        carry, release, and settling. Their exact prim paths come from the
        episode's manipulation contract; no semantic or path guessing occurs.
        """
        if self._static_scene_bodies_prepared:
            return
        import omni.usd
        from pxr import Usd

        from tarsphere.infra.manipulation.g1_attachment import (
            manipulation_objects_from_episode,
        )

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError(
                "Cannot prepare static scene bodies: USD stage unavailable."
            )
        scene_root = stage.GetPrimAtPath("/World/env_0/scene")
        if not scene_root or not scene_root.IsValid():
            raise RuntimeError(
                "Cannot prepare static scene bodies: /World/env_0/scene is missing."
            )

        dynamic_object_paths = tuple(
            obj.prim_path.rstrip("/")
            for obj in manipulation_objects_from_episode(self.episode)
            if obj.pickable
        )

        made_static = 0
        preserved_dynamic = 0
        for prim in Usd.PrimRange.AllPrims(scene_root):
            attr = prim.GetAttribute("physics:rigidBodyEnabled")
            enabled = bool(attr.Get()) if attr and attr.IsValid() else False
            if not enabled:
                continue
            prim_path = str(prim.GetPath())
            if any(
                prim_path == path or prim_path.startswith(path + "/")
                for path in dynamic_object_paths
            ):
                preserved_dynamic += 1
                continue
            if not attr.Set(False):
                raise RuntimeError(
                    f"Cannot disable rigid-body dynamics for {prim_path}."
                )
            made_static += 1

        self._static_scene_bodies_prepared = True
        print(
            "[INFO] Immutable scene bodies prepared as static colliders: "
            f"made_static={made_static}, "
            f"preserved_task_bodies={preserved_dynamic}.",
            flush=True,
        )

    def observe(self) -> Dict[str, Any]:
        # ``text`` carries only what is true of THIS frame. How to read an
        # RGBD side-by-side view is a standing fact about the robot's body and
        # lives in the system preamble; repeating it here would make every
        # entry in the agent's observation history start with the same 170
        # characters of boilerplate and crowd out the transient notes that are
        # the whole point of keeping that history.
        notes: list[str] = []
        if self._interaction_feedback:
            notes.append(self._interaction_feedback)
            self._interaction_feedback = ""
        snapshot = self._require_ego_snapshot()
        result = self._observation_from_ego_snapshot(snapshot)
        if result.get("depth_status") != "valid" and snapshot.rgb is not None:
            notes.append(
                "Depth signal is low-confidence in this frame; rely more "
                "on RGB, motion feedback, and future observations."
            )
        result["text"] = " ".join(notes)
        rpos = result["pose"].get("position")
        if rpos is not None:
            self._robot_pose_history.append([float(rpos[0]), float(rpos[1])])
            if len(self._robot_pose_history) > 200:
                self._robot_pose_history.pop(0)
        for hid in self._human_positions:
            hpos = self._human_positions.get(hid)
            if hpos is not None:
                hist = self._human_pose_history.setdefault(hid, [])
                hist.append([float(hpos[0]), float(hpos[1])])
                if len(hist) > 200:
                    hist.pop(0)
        result["proprio"] = self._robot_proprio()
        lidar = self._capture_lidar()
        if lidar is None:
            raise RuntimeError(
                "LiDAR is enabled but live PhysX geometry or the robot "
                "pose is unavailable. Missing distance evidence cannot "
                "be represented as open space."
            )
        result["lidar"] = lidar
        return result

    def _invalidate_ego_snapshot(self) -> None:
        """Drop the stamped snapshot after a measured-world mutation.

        ``_require_ego_snapshot`` recaptures when ``physics_step`` disagrees.
        Mutations that do not advance physics — camera aim, authored pickable
        xforms, teleports — must clear the snapshot explicitly. The physics-step
        check remains the backstop for ordinary simulation advance.
        """

        self._ego_snapshot = None

    def _record_render(self) -> None:
        """Count one actual Kit/Replicator render for episode stats."""

        self._render_count = int(getattr(self, "_render_count", 0)) + 1

    def _install_native_render_accounting(self, env: Any) -> None:
        """Count TARSphere runner frames after each physics step."""

        original_step = env.step

        def counted_step(*args: Any, **kwargs: Any):
            result = original_step(*args, **kwargs)
            runner = getattr(env, "runner", None)
            if runner is not None and bool(getattr(runner, "_render", False)):
                self._record_render()
            return result

        env.step = counted_step

    def _require_ego_snapshot(self) -> EgoObservationSnapshot:
        """Return the snapshot for the current physics step, capturing if needed."""

        snapshot = getattr(self, "_ego_snapshot", None)
        if snapshot is None or int(snapshot.physics_step) != int(self._physics_steps):
            self._ego_snapshot = self._capture_ego_snapshot()
        return self._ego_snapshot

    def _observation_from_ego_snapshot(
        self,
        snapshot: EgoObservationSnapshot,
    ) -> Dict[str, Any]:
        """Publish one observation copied entirely from a stamped snapshot."""

        camera_params = {
            "fov_deg": float(snapshot.fov_deg),
            "width": int(snapshot.width),
            "height": int(snapshot.height),
            "source": "robot_ego",
            "camera_path": snapshot.camera_path,
            "world_T_cam": [list(row) for row in snapshot.world_T_cam],
            "capture_id": int(snapshot.capture_id),
            "depth_capture_id": int(snapshot.capture_id),
            "physics_step": int(snapshot.physics_step),
        }
        result: Dict[str, Any] = {
            "pose": {
                "position": list(snapshot.robot_position),
                "orientation": snapshot.robot_orientation,
                "yaw_rad": float(snapshot.robot_yaw_rad),
                "camera_height_m": float(snapshot.camera_height_m),
                "camera_pitch_deg": float(snapshot.camera_pitch_deg),
                "camera_gaze_tilt_deg": float(snapshot.camera_gaze_tilt_deg),
                "body_posture": snapshot.body_posture,
            },
            "camera_params": camera_params,
            "camera_source": "robot_ego",
            "camera_path": snapshot.camera_path,
            "depth_status": snapshot.depth_status,
        }
        proprioception = self._eef_proprioception_summary()
        if proprioception is not None:
            result["proprioception"] = proprioception
        if snapshot.rgb is not None:
            result["rgb"] = snapshot.rgb
        if snapshot.depth is not None:
            result["depth"] = snapshot.depth
        if snapshot.rgbd_png is not None:
            result["rgbd_png"] = snapshot.rgbd_png
        if snapshot.image_path is not None:
            result["image_path"] = snapshot.image_path
        if snapshot.bounding_box_2d_tight is not None:
            result["bounding_box_2d_tight"] = snapshot.bounding_box_2d_tight
        if snapshot.depth_valid_ratio is not None:
            result["depth_valid_ratio"] = float(snapshot.depth_valid_ratio)
        if snapshot.depth_shape is not None:
            result["depth_shape"] = snapshot.depth_shape
        if snapshot.rgb is not None:
            result["views"] = {
                "ego": {
                    "rgb": snapshot.rgb,
                    "depth": snapshot.depth,
                    "camera_params": dict(camera_params),
                },
            }
        return result

    def _capture_ego_snapshot(
        self,
        *,
        physics_step: int | None = None,
        capture_observer: bool = True,
        record_continuous: bool = True,
        write_image: bool = True,
    ) -> EgoObservationSnapshot:
        """Render once, then read RGB-D, camera extrinsics, and base pose.

        Nothing in this method may step physics. The render uses
        ``delta_time=0.0`` so Kit/Replicator refresh the current USD state
        without advancing PhysX. Extrinsics and the pelvis/base pose are
        read immediately after that annotator read.
        """

        capture_physics_step = (
            int(self._physics_steps) if physics_step is None else int(physics_step)
        )
        captured: Dict[str, Any] = {}
        if self.enable_ego_sensors:
            self._force_one_ego_render(capture_observer=capture_observer)
            captured = self._read_ego_rgbd_after_render()
        camera_path = self._agent_camera_path()
        if not camera_path:
            raise RuntimeError(
                "Cannot capture an ego snapshot: the robot has no camera path "
                f"at physics step {self._physics_steps}."
            )
        world_T_cam = self._robot_camera_world_T_optical(camera_path)
        position, orientation, yaw_rad = self._read_robot_base_pose_now()
        robot_root_T_world = self._planning_root_at_capture(
            position=position,
            orientation=orientation,
        )
        raw_rgb = captured.get("rgb")
        raw_depth = captured.get("depth")
        self._next_ego_capture_id = int(getattr(self, "_next_ego_capture_id", 0)) + 1
        capture_id = int(self._next_ego_capture_id)
        if raw_rgb is not None:
            rgb, depth = condition_sensor_frame(
                raw_rgb,
                raw_depth,
                condition=condition_for_episode(self.episode),
                episode_id=str(self.episode.episode_id),
                episode_capture_index=capture_id,
            )
        else:
            rgb, depth = None, raw_depth
        if rgb is not None and getattr(rgb, "ndim", 0) >= 2:
            width = int(rgb.shape[1])
            height = int(rgb.shape[0])
        else:
            width, height = ROBOT_EGO_CAMERA_RESOLUTION
        policy_png = compose_rgbd_image(rgb, depth)
        image_path = None
        if self.image_dir is not None and write_image and policy_png:
            image_path = self.image_dir / ego_capture_filename(
                physics_step=capture_physics_step,
                capture_id=capture_id,
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(policy_png)
        snapshot = EgoObservationSnapshot(
            capture_id=capture_id,
            physics_step=capture_physics_step,
            rgb=rgb,
            depth=depth,
            visibility_depth=raw_depth,
            fov_deg=float(ROBOT_CAMERA_HORIZONTAL_FOV_DEG),
            width=width,
            height=height,
            camera_path=str(camera_path),
            world_T_cam=_immutable_transform(world_T_cam),
            robot_position=tuple(float(value) for value in position),
            robot_orientation=orientation,
            robot_yaw_rad=float(yaw_rad),
            robot_root_T_world=_immutable_transform(robot_root_T_world),
            camera_height_m=float(ROBOT_CAMERA_HEIGHT_M),
            camera_pitch_deg=float(self._robot_camera_total_pitch_deg()),
            camera_gaze_tilt_deg=float(self.camera_gaze_tilt_deg()),
            body_posture=g1_body_posture_from_height_command(
                self._g1_body_height_command()
            ),
            depth_status=str(captured.get("depth_status") or ""),
            visibility_depth_status=str(captured.get("depth_status") or ""),
            rgbd_png=policy_png,
            image_path=(str(image_path) if image_path is not None else None),
            bounding_box_2d_tight=captured.get("bounding_box_2d_tight"),
            depth_valid_ratio=(
                None
                if captured.get("depth_valid_ratio") is None
                else float(
                    np.mean(np.isfinite(depth) & (depth > 0.0))
                    if depth is not None
                    else captured["depth_valid_ratio"]
                )
            ),
            depth_shape=(
                tuple(depth.shape) if depth is not None else captured.get("depth_shape")
            ),
        )
        if record_continuous:
            self._record_continuous_ego_snapshot(snapshot)
        return snapshot

    def _record_continuous_ego_snapshot(
        self,
        snapshot: EgoObservationSnapshot,
    ) -> None:
        """Append an exact evaluated ego snapshot when recording is active."""

        recorder = getattr(self, "_continuous_ego_video", None)
        if recorder is None or not recorder.active:
            return
        recorder.capture(
            physics_step=int(snapshot.physics_step),
            rgba=snapshot.rgb,
            position=snapshot.robot_position,
            yaw_rad=float(snapshot.robot_yaw_rad),
        )

    def capture_initial_presentation_frame(self) -> dict[str, int]:
        """Capture all opened presentation streams at the current physics tick.

        Recording starts after reset, which may already have sampled this tick
        to PNG. The explicit initial capture owns a fresh video sample epoch.
        Kit animation and PhysX time must remain unchanged during bounded RGB
        warmup; actual stream capture, not renderer invocation, proves success.
        """
        observer = getattr(self, "_room_observer", None)
        if (
            self._env is None
            or observer is None
            or self.image_dir is None
            or not observer.accepts_captures(self.image_dir)
        ):
            raise RuntimeError(
                "Initial presentation capture requires a live episode and observer."
            )
        physics_step = int(self._physics_steps)
        captures = observer.video_capture_steps
        if set(captures) != set(observer.view_names):
            raise RuntimeError(
                "Open every requested observer video before initial presentation capture."
            )
        if all(tick == physics_step for tick in captures.values()):
            return captures
        if any(tick is not None for tick in captures.values()):
            raise RuntimeError(
                "Initial presentation streams have partial or stale captures."
            )
        import omni.kit.loop._loop as omni_loop
        import omni.timeline

        loop = omni_loop.acquire_loop_interface()
        timeline = omni.timeline.get_timeline_interface()
        previous_mode = loop.get_manual_mode()
        previous_step = float(loop.get_manual_step_size())
        timeline_time = float(timeline.get_current_time())
        if not math.isfinite(previous_step) or not math.isfinite(timeline_time):
            raise RuntimeError(
                "Initial presentation capture requires readable finite native clocks."
            )
        try:
            loop.set_manual_mode(True)
            loop.set_manual_step_size(0.0)
            for _ in range(ROOM_OBSERVER_NATIVE_RENDER_WARMUP_LIMIT):
                self._force_one_ego_render(initial_presentation_frame=True)
                if int(self._physics_steps) != physics_step or not math.isclose(
                    float(timeline.get_current_time()),
                    timeline_time,
                    abs_tol=1e-9,
                    rel_tol=0.0,
                ):
                    raise RuntimeError(
                        "Initial presentation rendering advanced physics or Kit timeline time."
                    )
                captures = observer.video_capture_steps
                if set(captures) != set(observer.view_names):
                    raise RuntimeError(
                        "Observer streams changed during initial presentation capture."
                    )
                if all(tick == physics_step for tick in captures.values()):
                    return captures
                if any(tick is not None for tick in captures.values()):
                    raise RuntimeError(
                        "Initial presentation rendering only captured some requested views."
                    )
            raise RuntimeError(
                "Initial presentation streams produced no complete RGB frame within "
                f"{ROOM_OBSERVER_NATIVE_RENDER_WARMUP_LIMIT} zero-time renders."
            )
        finally:
            loop.set_manual_step_size(previous_step)
            loop.set_manual_mode(previous_mode)

    def _force_one_ego_render(
        self,
        *,
        capture_observer: bool = True,
        initial_presentation_frame: bool = False,
    ) -> None:
        """Render the current USD state once without advancing PhysX."""

        if self._env is None:
            return
        capture_observer = bool(capture_observer) and (
            (
                initial_presentation_frame
                or getattr(self, "render_mode", RenderMode.EVALUATION)
                is RenderMode.EVALUATION
            )
            and getattr(self, "_room_observer", None) is not None
            and getattr(self, "image_dir", None) is not None
            and self._room_observer.accepts_captures(self.image_dir)
        )
        if capture_observer:
            position, yaw_rad = self._live_room_observer_pose()
            prepare = (
                self._room_observer.prepare_initial_video_capture
                if initial_presentation_frame
                else self._room_observer.prepare_capture
            )
            prepare(
                image_dir=self.image_dir,
                physics_step=int(self._physics_steps),
                position=position,
                yaw_rad=yaw_rad,
            )
        try:
            import omni.replicator.core as rep

            rep.orchestrator.step(
                rt_subframes=1,
                delta_time=0.0,
                pause_timeline=False,
                wait_for_render=True,
            )
        except Exception as exc:
            if capture_observer:
                self._room_observer.cancel_capture()
            raise RuntimeError(
                "Cannot render a synchronized ego frame without stepping "
                f"physics at step {self._physics_steps}: {exc}"
            ) from exc
        self._poll_human_animation_graph_after_render()
        self._record_render()
        if capture_observer:
            self._last_room_observer_paths = self._room_observer.finish_capture()

    def _poll_human_animation_graph_after_render(self) -> None:
        """Observe zero-time graph edges separately from command completion."""
        if getattr(self, "human_motion_backend", "kinematic") != "animation":
            return
        poll = getattr(getattr(self, "_human_body", None), "poll_animation_graph", None)
        if callable(poll):
            poll(observation_boundary="after_native_render")

    def _read_ego_rgbd_after_render(self) -> Dict[str, Any]:
        """Read annotators immediately after ``_force_one_ego_render``."""

        camera = self._robot_camera_sensor()
        if camera is not None:
            data = camera.get_data()
            if not isinstance(self._obs, dict):
                self._obs = {}
            sensors = self._obs.setdefault("sensors", {})
            if not isinstance(sensors, dict):
                sensors = {}
                self._obs["sensors"] = sensors
            sensors["camera"] = data
        return self._capture_rgbd()

    def _read_robot_base_pose_now(
        self,
    ) -> tuple[tuple[float, ...], Any, float]:
        """Read the robot base pose in the same instant as the ego frame."""

        if self._env is not None:
            tasks = tuple(self._env.runner.current_tasks.values())
            robots = tuple(tasks[0].robots.values()) if len(tasks) == 1 else ()
            if len(robots) == 1:
                get_pose = getattr(robots[0], "get_pose", None)
                if callable(get_pose):
                    position, orientation = get_pose()
                    position_values = np.asarray(
                        position,
                        dtype=np.float64,
                    ).reshape(-1)
                    quaternion = np.asarray(
                        orientation,
                        dtype=np.float64,
                    ).reshape(-1)
                    if (
                        position_values.size == 3
                        and quaternion.size == 4
                        and np.all(np.isfinite(position_values))
                        and np.all(np.isfinite(quaternion))
                    ):
                        w, x, y, z = (float(value) for value in quaternion.tolist())
                        yaw_rad = math.atan2(
                            2.0 * (w * z + x * y),
                            1.0 - 2.0 * (y * y + z * z),
                        )
                        return (
                            tuple(float(value) for value in position_values),
                            [w, x, y, z],
                            float(yaw_rad),
                        )
        position = self._robot_position()
        if position is None or len(position) < 3:
            raise RuntimeError(
                "Cannot capture an ego snapshot: robot base position is "
                f"unavailable at physics step {self._physics_steps}."
            )
        orientation = self._obs.get("orientation")
        if hasattr(orientation, "cpu"):
            orientation = orientation.cpu().numpy()
        if orientation is not None:
            quat = np.asarray(orientation, dtype=float).reshape(-1)
            if quat.size == 4 and np.all(np.isfinite(quat)):
                orientation = [float(value) for value in quat.tolist()]
        return (
            tuple(float(value) for value in position[:3]),
            orientation,
            float(self._robot_yaw()),
        )

    def _planning_root_at_capture(
        self,
        *,
        position: Sequence[float],
        orientation: Any,
    ) -> np.ndarray:
        """Planning-root transform captured with the ego frame, never later."""

        if self._env is not None:
            # Falling back to the base-pose frame here would move the planning
            # root by far more than G1_ARM_IK_SOLVER_TOLERANCE_M, and the miss
            # would look like a solver error rather than a frame swap. With a
            # live environment the lookup failing means the scene is broken.
            try:
                self._runtime_g1_robot()
            except RuntimeError as exc:
                raise RuntimeError(
                    "Cannot capture the G1 planning root from a live "
                    f"environment at physics step {self._physics_steps}: "
                    f"{exc}"
                ) from exc
            return self._g1_planning_root_from_world()
        if orientation is None:
            raise RuntimeError(
                "Cannot capture a planning root without a robot orientation "
                f"at physics step {self._physics_steps}."
            )
        return np.asarray(
            frame_from_world_pose(
                frame_position_world_m=np.asarray(position, dtype=np.float64),
                frame_orientation_world_wxyz=np.asarray(
                    orientation,
                    dtype=np.float64,
                ),
            ),
            dtype=np.float64,
        )

    def _planning_root_for_camera_measurement(
        self,
        *,
        world_T_camera: np.ndarray,
        capture_id: Any,
    ) -> np.ndarray:
        """Return the planning root from the snapshot that produced the camera."""

        snapshot = getattr(self, "_ego_snapshot", None)
        if snapshot is None:
            raise RuntimeError(
                "agent_grasp cannot compose robot_root_from_camera without "
                "an ego snapshot from the same capture as world_T_camera."
            )
        if capture_id is None:
            raise RuntimeError(
                "agent_grasp world_T_camera is missing capture_id; the "
                "planning root cannot be paired with that extrinsics matrix."
            )
        if int(capture_id) != int(snapshot.capture_id):
            raise RuntimeError(
                "agent_grasp world_T_camera capture_id="
                f"{int(capture_id)} does not match the current ego snapshot "
                f"capture_id={int(snapshot.capture_id)} "
                f"(snapshot physics_step={int(snapshot.physics_step)}, "
                f"env physics_step={int(self._physics_steps)})."
            )
        snapshot_T = _mutable_transform(snapshot.world_T_cam)
        if not np.allclose(snapshot_T, world_T_camera, atol=1e-8, rtol=0.0):
            raise RuntimeError(
                "agent_grasp world_T_camera does not match the snapshot "
                f"for capture_id={int(capture_id)}."
            )
        return _mutable_transform(snapshot.robot_root_T_world)

    def _begin_g1_closing_reach_deflection_capture(
        self,
        *,
        robot_root_T_world: np.ndarray,
        commanded_body_height_m: float,
    ) -> None:
        """Record the planning-root pelvis for this closing-reach decision."""

        pelvis_position, pelvis_orientation = pelvis_world_pose_from_g1_planning_root(
            robot_root_T_world,
            commanded_body_height_m=self._g1_measured_body_height_m(
                self._live_g1_pelvis_world_pose()[0],
            ),
        )
        self._g1_closing_reach_deflection_capture = {
            "pelvis_position_world_m": pelvis_position,
            "pelvis_orientation_world_wxyz": pelvis_orientation,
            "commanded_body_height_m": float(commanded_body_height_m),
        }

    def _complete_g1_closing_reach_deflection_capture(
        self,
        *,
        hand: str,
        terminal_waypoint: Sequence[float],
        requested_palm_world_m: Sequence[float],
        robot_root_T_world: np.ndarray,
    ) -> None:
        """Fill commanded waist/palm from the plan chosen for this decision."""

        capture = getattr(self, "_g1_closing_reach_deflection_capture", None)
        if not isinstance(capture, dict) or not capture:
            raise RuntimeError(
                "Cannot complete a closing-reach deflection capture before "
                "the planning root for this decision was recorded."
            )
        requested_palm = tuple(float(value) for value in requested_palm_world_m)
        if len(requested_palm) != 3 or not all(
            math.isfinite(value) for value in requested_palm
        ):
            raise RuntimeError(
                "Closing-reach deflection capture requires a finite requested "
                f"palm world position, got {requested_palm_world_m!r}."
            )
        capture["hand"] = str(hand)
        capture["commanded_waist_yaw_rad"] = g1_manipulation_waypoint_value(
            terminal_waypoint,
            hand,
            "waist_yaw_joint",
        )
        capture["commanded_waist_pitch_rad"] = g1_manipulation_waypoint_value(
            terminal_waypoint,
            hand,
            "waist_pitch_joint",
        )
        capture["commanded_arm_rad"] = g1_manipulation_waypoint_arm_values(
            terminal_waypoint,
            hand,
        )
        capture["requested_palm_world_m"] = requested_palm
        capture["requested_palm_in_planning_root_m"] = homogeneous_transform_point(
            robot_root_T_world,
            requested_palm,
        )
        self._g1_closing_reach_deflection_capture = capture

    def _g1_closing_reach_deflection_capture_or_raise(self) -> Dict[str, Any]:
        """Return the current decision's complete capture, or fail loud."""

        capture = getattr(self, "_g1_closing_reach_deflection_capture", None)
        required = (
            "pelvis_position_world_m",
            "pelvis_orientation_world_wxyz",
            "commanded_body_height_m",
            "commanded_waist_yaw_rad",
            "commanded_waist_pitch_rad",
            "commanded_arm_rad",
            "requested_palm_world_m",
            "requested_palm_in_planning_root_m",
            "hand",
        )
        if not isinstance(capture, dict):
            raise RuntimeError(
                "Closing-reach deflection probe has no capture for this "
                "decision; the planning root was never recorded."
            )
        missing = [name for name in required if name not in capture]
        if missing:
            raise RuntimeError(
                "Closing-reach deflection capture is incomplete; missing "
                f"{missing!r}. Commanded waist and requested palm come from "
                "the plan chosen after the planning root was captured."
            )
        return capture

    def _live_g1_pelvis_world_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Read the live articulated pelvis pose used as the planning root."""

        pelvis_position, pelvis_orientation = (
            self._runtime_g1_robot().get_robot_base().get_pose()
        )
        for value_name, value in (
            ("position", pelvis_position),
            ("orientation", pelvis_orientation),
        ):
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            if value_name == "position":
                pelvis_position = value
            else:
                pelvis_orientation = value
        position = tuple(
            float(value)
            for value in np.asarray(pelvis_position, dtype=np.float64).reshape(-1)
        )
        orientation = tuple(
            float(value)
            for value in np.asarray(
                pelvis_orientation,
                dtype=np.float64,
            ).reshape(-1)
        )
        if (
            len(position) != 3
            or len(orientation) != 4
            or not all(math.isfinite(value) for value in (*position, *orientation))
        ):
            raise RuntimeError(
                "Live G1 pelvis pose is missing or non-finite: "
                f"position={position!r}, orientation={orientation!r}."
            )
        return position, orientation

    def _g1_relative_pose_measurement(self) -> BodyPose2D:
        """Read the live PhysX-owned pelvis pose for base feedback.

        The decision-boundary observation cache is intentionally not used:
        every measurement comes directly from the articulation after a
        physics tick. This function never writes a root transform, so native
        contact remains free to block, deflect, or topple the robot.
        """

        position, orientation = self._live_g1_pelvis_world_pose()
        quaternion = np.asarray(orientation, dtype=np.float64)
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-12:
            raise RuntimeError("Live G1 pelvis orientation has zero quaternion norm.")
        w, x, y, z = (quaternion / norm).tolist()
        yaw_rad = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        if not math.isfinite(yaw_rad):
            raise RuntimeError("Live G1 pelvis orientation produced a non-finite yaw.")
        return BodyPose2D(
            x_m=float(position[0]),
            y_m=float(position[1]),
            yaw_rad=float(yaw_rad),
        )

    def _print_g1_closing_reach_deflection(self, at: str) -> None:
        """Sample live waist/pelvis/palm now and print one diagnostic line."""

        capture = self._g1_closing_reach_deflection_capture_or_raise()
        hand = str(capture["hand"])
        live_position, live_orientation = self._live_g1_pelvis_world_pose()
        joint_state = self._g1_curobo_joint_positions()
        if "waist_yaw_joint" not in joint_state:
            raise RuntimeError(
                "Live G1 articulation has no waist_yaw_joint measurement."
            )
        if "waist_pitch_joint" not in joint_state:
            raise RuntimeError(
                "Live G1 articulation has no waist_pitch_joint measurement."
            )
        measured_arm = self._arm_joint_positions(hand)
        if measured_arm is None:
            raise RuntimeError(
                f"Cannot sample the live {hand} arm for the deflection probe at {at!r}."
            )
        reached_palm = self._robot_frame_point_in_world(
            np.asarray(
                G1_CUROBO_PALM_TOOL_POINT_IN_PALM_LINK_M,
                dtype=np.float64,
            ),
            frame_name=f"{hand}_hand_palm_link",
        )
        metrics = g1_closing_reach_deflection_metrics(
            commanded_waist_yaw_rad=float(capture["commanded_waist_yaw_rad"]),
            commanded_waist_pitch_rad=float(capture["commanded_waist_pitch_rad"]),
            measured_waist_yaw_rad=float(joint_state["waist_yaw_joint"]),
            measured_waist_pitch_rad=float(joint_state["waist_pitch_joint"]),
            captured_pelvis_position_world_m=capture["pelvis_position_world_m"],
            captured_pelvis_orientation_world_wxyz=capture[
                "pelvis_orientation_world_wxyz"
            ],
            live_pelvis_position_world_m=live_position,
            live_pelvis_orientation_world_wxyz=live_orientation,
            requested_palm_world_m=capture["requested_palm_world_m"],
            requested_palm_in_planning_root_m=capture[
                "requested_palm_in_planning_root_m"
            ],
            reached_palm_world_m=reached_palm,
            live_planning_root_T_world=self._g1_planning_root_from_world(),
            commanded_arm_rad=capture["commanded_arm_rad"],
            measured_arm_rad=measured_arm,
            commanded_body_height_m=self._g1_body_height_command(),
            measured_body_height_m=self._g1_measured_body_height_m(
                live_position,
            ),
        )
        print(format_g1_deflection_probe_line(at=at, metrics=metrics), flush=True)

    def increment_decision_step(self) -> int:
        """Called by the runner after each decision step."""
        self._decision_step += 1
        return self._decision_step

    def set_abort_checker(self, checker: Callable[[], bool]) -> None:
        """Install the active scene-server request's cancellation signal."""
        if not callable(checker):
            raise TypeError("Isaac abort checker must be callable.")
        self._abort_requested = checker

    def set_physics_step_limit(self, limit: int | None) -> None:
        """Install the optional episode-wide simulated-time budget."""

        if limit is not None and int(limit) <= 0:
            raise ValueError("Isaac physics-step limit must be positive or null.")
        self._physics_step_limit = None if limit is None else int(limit)

    def _remaining_physics_steps(self) -> int | None:
        """Return unspent global physics ticks, or ``None`` when unbounded."""

        if self._physics_step_limit is None:
            return None
        return max(0, self._physics_step_limit - self._physics_steps)

    def get_human_observations(
        self,
        human_ids: Optional[list[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Return per-human ego observations for human policies.

        Humans are animated Isaac People characters.  The body executor polls
        their stage pose after ``omni.anim.people`` updates the skeleton and
        NavMesh locomotion; the ego camera is synchronized to that observed
        pose at eye height.
        """
        ids = human_ids or list(self._human_positions)
        if self._env is not None and self.enable_human_ego_cameras:
            self._setup_human_ego_cameras()
        robot_position = self._robot_position()
        observations: Dict[str, Dict[str, Any]] = {}
        for human_id in ids:
            human_position = self._human_positions.get(human_id)
            can_communicate = self.can_communicate(human_id)
            distance = None
            if robot_position is not None and human_position is not None:
                distance = float(
                    np.linalg.norm(
                        np.asarray(robot_position[:2]) - np.asarray(human_position[:2])
                    )
                )
            if distance is None:
                robot_relation = "robot pose is unavailable"
            elif can_communicate:
                robot_relation = "robot is within speaking range"
            else:
                robot_relation = "robot is outside speaking range"

            observation: Dict[str, Any] = {
                "text": f"Human ego observation: {robot_relation}.",
                "position": list(human_position)
                if human_position is not None
                else None,
                "yaw_rad": self._human_yaws.get(human_id, 0.0),
                "robot_can_communicate": can_communicate,
                "robot_relation": robot_relation,
                "ego_view": (
                    self._capture_human_ego_view(human_id)
                    if self.enable_human_ego_cameras
                    else {"status": "disabled"}
                ),
                "last_utterance": self._last_utterance,
            }
            body_state = self._human_body.get_state(human_id)
            if body_state is not None:
                observation.update(body_state.to_observation())
                if body_state.metadata:
                    observation["body_metadata"] = dict(body_state.metadata)
            ego_camera_pose = self._human_ego_camera_poses.get(human_id)
            if ego_camera_pose:
                observation["ego_camera_pose"] = dict(ego_camera_pose)
            if distance is not None:
                # This value gates Human events, not just display text. Keep
                # the same precision as the evaluator's live XY predicate.
                observation["robot_distance_m"] = distance
            observations[human_id] = observation
        return observations

    def get_world_snapshot(
        self,
        human_ids: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """Return benchmark-private world state for human actors/evaluator.

        This is deliberately not part of robot observation. It gives the human
        actor runtime the oracle state needed to trigger scripted benchmark
        events such as "robot enters operator desk".
        """
        _rollout_memory_diagnostic("snapshot.begin", self._physics_steps)
        ids = human_ids or list(self._human_positions)
        robot_position = self._robot_position()
        humans: Dict[str, Dict[str, Any]] = {}
        for human_id in ids:
            state: Dict[str, Any] = {}
            human_position = self._human_positions.get(human_id)
            if human_position is not None:
                state["position"] = list(human_position)
            state["yaw_rad"] = self._human_yaws.get(human_id, 0.0)
            body_state = self._human_body.get_state(human_id)
            if body_state is not None:
                state.update(body_state.to_observation())
            if robot_position is not None and human_position is not None:
                state["robot_distance_m"] = float(
                    np.linalg.norm(
                        np.asarray(robot_position[:2]) - np.asarray(human_position[:2])
                    )
                )
            humans[human_id] = state
        snapshot = {
            "step": self._physics_steps,
            "robot": {
                "position": list(robot_position)
                if robot_position is not None
                else None,
                "yaw_rad": float(self._robot_yaw()),
            },
            "humans": humans,
            "last_utterance": self._last_utterance,
            "accepted_handovers": dict(getattr(self, "_accepted_handovers", {})),
        }
        claim = getattr(self, "_last_handover_claim", None)
        if isinstance(claim, dict):
            private = claim.get("private")
            snapshot["handover"] = dict(private if isinstance(private, dict) else claim)
        if self._g1_attachment is not None:
            # Golden certification is allowed benchmark-private geometry, but
            # its pre-grasp completion predicate must use the same *live*
            # torso frame as execution-time IK.  Export an affine world-to-
            # torso transform rather than approximating it from root XY/yaw;
            # the official locomotion policy can leave the articulated torso
            # rolled, pitched, and offset from that root pose.
            snapshot["robot"]["torso_world_to_frame"] = (
                self._world_to_robot_frame_affine("torso_link")
            )
            _rollout_memory_diagnostic(
                "snapshot.before_manipulation", self._physics_steps
            )
            manipulation = self._g1_attachment.snapshot()
            if self._last_g1_arm_ik is not None:
                # Private certification evidence: identify the arm solution
                # that produced the current live hand geometry. This snapshot
                # is name-gated to the Golden oracle by the harness and is
                # never part of the evaluated Agent observation.
                manipulation["last_arm_ik"] = {
                    "hand": str(self._last_g1_arm_ik["hand"]),
                    "target_position_world": list(
                        self._last_g1_arm_ik["target_position_world"]
                    ),
                    "position_error_m": float(self._last_g1_arm_ik["position_error_m"]),
                }
            snapshot["manipulation"] = manipulation
            _rollout_memory_diagnostic(
                "snapshot.after_manipulation", self._physics_steps
            )
        if self._robot_contact_monitor is not None:
            snapshot["execution_safety"] = self._robot_contact_monitor.snapshot()
            snapshot["grasp_grounding_summary"] = self._grasp_grounding_summary()
        _rollout_memory_diagnostic("snapshot.end", self._physics_steps)
        return snapshot

    def get_human_position(self, human_id: str) -> list[float]:
        """Return one Human's live simulated world position.

        Reads the pose last synced from the animation runtime, not the
        authored spawn coordinate. Evaluators may read it directly; an Agent
        boundary may admit it only after ``is_human_visible`` succeeds for
        the same current ego RGB-D snapshot. Raises if that Human is absent
        or has no live pose yet.
        """
        key = str(human_id)
        if key not in self._human_positions:
            raise KeyError(
                f"No live position for human {key!r}; known ids="
                f"{sorted(self._human_positions)}"
            )
        position = self._human_positions[key]
        if position is None:
            raise RuntimeError(f"Human {key!r} has no live simulated position.")
        return list(position)

    @property
    def last_utterance(self) -> Optional[Dict[str, Any]]:
        return self._last_utterance

    @property
    def physics_steps(self) -> int:
        return int(self._physics_steps)

    def _start_execution_measurements(self) -> None:
        from tarsphere.infra.control.execution_measurements import (
            PhysicsPathAccumulator,
        )

        position, _ = self._live_g1_pelvis_world_pose()
        self._execution_path = PhysicsPathAccumulator(
            position=position,
            starting_physics_step=self._physics_steps,
        )

    def execution_measurements(self) -> Dict[str, Any]:
        """Private measured execution interval; never a control or scoring gate."""
        path = getattr(self, "_execution_path", None)
        measurements = (
            path.snapshot(total_physics_steps=self._physics_steps)
            if path is not None
            else {
                "status": "unavailable",
                "source": None,
                "missing_evidence": ["native_execution_sampling_not_started"],
                "physics_steps": None,
                "path_length_m": None,
            }
        )
        monitor = getattr(self, "_robot_contact_monitor", None)
        measurements["execution_safety"] = (
            monitor.snapshot() if monitor is not None else None
        )
        measurements["human_motion_backend"] = self.human_motion_backend
        return measurements

    @property
    def render_count(self) -> int:
        return int(self._render_count)

    def handover_was_accepted(self, *, human_id: str, object_id: str) -> bool:
        """Return accepted spoken-handover state, never physical Human hold."""

        accepted = getattr(self, "_accepted_handovers", {})
        return accepted.get(str(object_id)) == str(human_id)

    def evaluate_object_localization(
        self,
        *,
        object_id: str,
        projected_world_position: Sequence[float],
        tolerance_m: float,
    ) -> Dict[str, Any]:
        """Compare a SAM3/depth world projection with live evaluator GT.

        This measures perception after the Agent chose ``locate_object``. It
        never gates, redirects, or shortens manipulation execution.
        """

        point = np.asarray(projected_world_position, dtype=float).reshape(-1)
        tolerance = float(tolerance_m)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(
                "projected_world_position must contain three finite values"
            )
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance_m must be finite and positive")
        if self._g1_attachment is None:
            return {
                "object_id": str(object_id),
                "projected_world_position": point.tolist(),
                "tolerance_m": tolerance,
                "matched": False,
                "reason": "manipulation_state_unavailable",
            }
        state = self._g1_attachment.snapshot().get("objects", {}).get(str(object_id))
        if not isinstance(state, dict):
            return {
                "object_id": str(object_id),
                "projected_world_position": point.tolist(),
                "tolerance_m": tolerance,
                "matched": False,
                "reason": "object_not_present",
            }
        lower = np.asarray(state.get("bbox_min"), dtype=float).reshape(-1)
        upper = np.asarray(state.get("bbox_max"), dtype=float).reshape(-1)
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper))
            or np.any(lower > upper)
        ):
            raise ValueError(f"object {object_id!r} has invalid live AABB geometry")
        gt_center = (lower + upper) * 0.5
        error = float(np.linalg.norm(point - gt_center))
        return {
            "object_id": str(object_id),
            "projected_world_position": point.tolist(),
            "gt_center_world_position": gt_center.tolist(),
            "center_error_m": error,
            "tolerance_m": tolerance,
            "matched": error <= tolerance,
        }

    def _apply_handover_object(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate one spoken handover claim against live pose and hold state."""

        from tarsphere.infra.manipulation.handover import evaluate_handover_claim

        recipient = decision.get("recipient")
        if not isinstance(recipient, str):
            recipient = ""
        if decision.get("_handover_resolution_checked") is True:
            resolved_recipient = decision.get("_resolved_handover_human_id")
        else:
            resolved_recipient = self.resolve_handover_human_id(recipient)
        canonical_recipient = (
            str(resolved_recipient) if resolved_recipient is not None else ""
        )
        robot_position = self._robot_position()
        if robot_position is None or len(robot_position) < 2:
            raise RuntimeError("handover_object requires a live robot position.")
        held = self._currently_held_object()
        held_object_id = str(held["object_id"]) if held is not None else None
        humans = self._handover_human_identities()
        live_positions = {
            human.id: self.get_human_position(human.id) for human in humans
        }
        verdict = evaluate_handover_claim(
            target=canonical_recipient,
            humans=humans,
            held_object_id=held_object_id,
            object_transport_ready=bool(held and held["transport_ready"]),
            robot_xy=robot_position[:2],
            human_positions=live_positions,
            visible_human_ids={
                human.id for human in humans if self.is_human_visible(human.id)
            },
            accepted_handovers=self._accepted_handovers,
        )
        if verdict.accepted:
            if held is None or verdict.object_id is None:
                raise RuntimeError("Accepted handover is missing the live held object.")
            if not verdict.resolved_human_id:
                raise RuntimeError(
                    "Accepted handover is missing the resolved human id."
                )
            human_id = str(verdict.resolved_human_id)
            self._accepted_handovers[verdict.object_id] = human_id
        speaker = self._handover_speaker_name(verdict.resolved_human_id)
        if speaker:
            self.set_human_speech(verdict.feedback, speaker=speaker)
        self.set_interaction_feedback(verdict.feedback)
        claim = {
            "feedback": verdict.feedback,
            "private": verdict.as_private_record(),
        }
        self._last_handover_claim = claim
        return claim

    def resolve_handover_human_id(self, recipient: str) -> str | None:
        """Resolve one public name or description to an unambiguous Human id."""

        from tarsphere.infra.manipulation.handover import resolve_handover_target

        return resolve_handover_target(
            str(recipient),
            self._handover_human_identities(),
        )

    def _currently_held_object(self) -> dict[str, Any] | None:
        """Use the selected retention contract for the live handover claim."""

        attachment = self._g1_attachment
        if attachment is None:
            return None
        held: list[dict[str, Any]] = []
        for object_id in attachment.physical_holds:
            state = attachment.physical_hold_state(object_id=str(object_id))
            if state.get("held"):
                held.append(state)
        if len(held) > 1:
            raise RuntimeError(
                f"handover_object requires exactly one held object; found {held!r}."
            )
        return held[0] if held else None

    def _handover_human_identities(self):
        from tarsphere.infra.manipulation.handover import HandoverHuman

        identities = []
        for human in getattr(self.episode, "humans", []) or []:
            asset = str(
                getattr(human, "asset", None)
                or getattr(human, "character_name", None)
                or ""
            )
            descriptions: list[str] = []
            if asset:
                from tarsphere.humans.character_catalog import character_record

                # Without a catalog entry the human keeps no caption, job or
                # appearance, so every descriptive route is silently lost and
                # a valid claim like "the woman in scrubs" comes back
                # unresolved. That is an authoring gap, not an agent error.
                try:
                    record = character_record(asset)
                except KeyError as exc:
                    raise RuntimeError(
                        f"Human {getattr(human, 'human_id', '?')!r} uses asset "
                        f"{asset!r}, which has no character_catalog entry, so "
                        "handover targeting cannot describe them."
                    ) from exc
                descriptions.extend(
                    str(value)
                    for value in (
                        record.get("caption"),
                        None
                        if getattr(human, "public_job", "").strip()
                        else record.get("job"),
                        record.get("isaac_category"),
                        *(record.get("appearance_attributes_gt") or ()),
                    )
                    if value
                )
            descriptions.extend(
                str(value)
                for value in (
                    getattr(human, "public_description", ""),
                    getattr(human, "public_job", ""),
                    *(getattr(human, "appearance_attributes_gt", ()) or ()),
                )
                if str(value).strip()
            )
            identities.append(
                HandoverHuman(
                    id=str(human.id),
                    name=str(getattr(human, "name", "") or ""),
                    character_name=str(getattr(human, "character_name", "") or ""),
                    asset=asset,
                    role=str(getattr(human, "role", "") or ""),
                    descriptions=tuple(descriptions),
                )
            )
        return identities

    def _handover_speaker_name(self, human_id: str | None) -> str:
        if not human_id:
            return ""
        for human in getattr(self.episode, "humans", []) or []:
            if str(human.id) == str(human_id):
                return str(getattr(human, "name", "") or human.id)
        return str(human_id)

    def _world_to_robot_frame_affine(self, frame_name: str) -> Dict[str, Any]:
        """Serialize an exact USD world-to-robot-frame affine transform.

        ``Gf.Matrix4d`` uses conventions which should not leak into the
        benchmark-private oracle.  Transforming the world origin and three
        basis points here produces the unambiguous column-affine form
        ``point_frame = offset + linear @ point_world``.
        """
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        root_path = self._robot_root_prim_path()
        if stage is None or not root_path:
            raise RuntimeError(
                f"Cannot resolve live robot frame {frame_name!r} without a stage/root."
            )
        frame_path = f"{root_path}/{frame_name}"
        prim = stage.GetPrimAtPath(frame_path)
        if prim is None or not prim.IsValid():
            raise RuntimeError(f"Live robot frame is missing: {frame_path}")
        frame_from_world = (
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).GetInverse()
        )

        def transformed(point: tuple[float, float, float]) -> np.ndarray:
            value = frame_from_world.Transform(Gf.Vec3d(*point))
            return np.asarray(
                [float(value[0]), float(value[1]), float(value[2])],
                dtype=np.float64,
            )

        offset = transformed((0.0, 0.0, 0.0))
        columns = [
            transformed(axis) - offset
            for axis in (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        ]
        return {
            "frame_name": str(frame_name),
            "offset": offset.tolist(),
            "linear": np.column_stack(columns).tolist(),
        }

    def can_communicate(self, human_name: str) -> bool:
        position = self._robot_position()
        human_position = self._human_positions.get(human_name)
        if position is None or human_position is None:
            return False
        in_range = (
            float(
                np.linalg.norm(
                    np.asarray(position[:2]) - np.asarray(human_position[:2])
                )
            )
            <= HUMAN_COMMUNICATION_RADIUS_M
        )
        return in_range and self.is_human_visible(human_name)

    def is_human_visible(self, human_name: str) -> bool:
        """Return current ego RGB-D support for the named Human body volume."""

        human_position = self._human_positions.get(human_name)
        if human_position is None:
            return False
        snapshot = self._require_ego_snapshot()
        if (
            snapshot.visibility_depth is None
            or snapshot.visibility_depth_status
            not in {
                "valid",
                "low_confidence",
            }
        ):
            return False
        intrinsics = fov_to_intrinsics(
            float(snapshot.fov_deg),
            int(snapshot.width),
            int(snapshot.height),
        )
        human_spec = next(
            (
                human
                for human in self.episode.humans
                if str(human.id) == str(human_name)
            ),
            None,
        )
        bounds_min = getattr(human_spec, "scene_person_bounds_min_m", None)
        bounds_max = getattr(human_spec, "scene_person_bounds_max_m", None)
        body_bounds = (
            BBox3D(
                min=tuple(map(float, bounds_min)),
                max=tuple(map(float, bounds_max)),
            )
            if bounds_min is not None and bounds_max is not None
            else human_bbox3d(human_position)
        )
        visible_box = observed_volume_bbox(
            body_bounds,
            np.asarray(snapshot.visibility_depth),
            intrinsics,
            np.asarray(snapshot.world_T_cam, dtype=np.float64),
            tolerance_m=0.05,
            min_pixels=8,
        )
        return visible_box is not None

    def evaluate_visual_object_visibility(
        self,
        *,
        object_ids: tuple[str, ...],
    ) -> Dict[str, Any]:
        """Return current RGB-D evidence for exact episode-owned objects.

        Depth pixels must lift into each object's authored or live world AABB,
        so a projected box hidden behind other geometry cannot pass.  This is
        evaluator-owned evidence and does not depend on a policy tool call.
        """

        required = tuple(dict.fromkeys(str(value) for value in object_ids))
        world = getattr(self.episode, "world_knowledge", None) or {}
        records = {
            str(item.get("object_id")): item
            for item in (world.get("objects") or ())
            if isinstance(item, dict) and item.get("object_id") is not None
        }
        unknown = [value for value in required if value not in records]
        if unknown:
            raise ValueError(
                f"visual milestone references unknown episode objects: {unknown}"
            )

        snapshot = self._require_ego_snapshot()
        if (
            snapshot.visibility_depth is None
            or snapshot.visibility_depth_status
            not in {
                "valid",
                "low_confidence",
            }
        ):
            return {
                "matched": False,
                "evidence_source": "evaluator_rgbd_object_volume",
                "required_object_ids": list(required),
                "visible_object_ids": [],
                "boxes_xyxy": {},
                "supported_pixel_counts": {},
                "minimum_supported_pixels": 256,
                "minimum_bbox_size_px": [6, 6],
                "depth_status": str(snapshot.visibility_depth_status),
            }
        intrinsics = fov_to_intrinsics(
            float(snapshot.fov_deg),
            int(snapshot.width),
            int(snapshot.height),
        )
        boxes: Dict[str, list[float]] = {}
        supported_pixel_counts: Dict[str, int] = {}
        for object_id in required:
            record = records[object_id]
            lower = record.get("bounds_min_xyz")
            upper = record.get("bounds_max_xyz")
            runtime_prim = str(record.get("runtime_prim") or "").strip()
            if runtime_prim:
                try:
                    lower, upper = self.live_scene_object_bounds(runtime_prim)
                except (RuntimeError, ValueError) as exc:
                    raise RuntimeError(
                        f"visual evidence for runtime object {object_id!r} "
                        f"cannot read live bounds from {runtime_prim!r}"
                    ) from exc
            if not (
                isinstance(lower, (list, tuple))
                and isinstance(upper, (list, tuple))
                and len(lower) == 3
                and len(upper) == 3
            ):
                raise ValueError(
                    f"object {object_id!r} has no usable 3-D visibility bounds"
                )
            visibility = observed_volume_evidence(
                BBox3D(
                    tuple(float(value) for value in lower),
                    tuple(float(value) for value in upper),
                ),
                np.asarray(snapshot.visibility_depth),
                intrinsics,
                np.asarray(snapshot.world_T_cam, dtype=np.float64),
                tolerance_m=0.04,
                min_pixels=256,
                min_bbox_width_px=6,
                min_bbox_height_px=6,
            )
            if visibility is not None:
                boxes[object_id] = [float(value) for value in visibility.bbox_xyxy]
                supported_pixel_counts[object_id] = int(
                    visibility.supported_pixel_count
                )

        visible_ids = [value for value in required if value in boxes]
        return {
            "matched": len(visible_ids) == len(required),
            "evidence_source": "evaluator_rgbd_object_volume",
            "required_object_ids": list(required),
            "visible_object_ids": visible_ids,
            "boxes_xyxy": boxes,
            "supported_pixel_counts": supported_pixel_counts,
            "minimum_supported_pixels": 256,
            "minimum_bbox_size_px": [6, 6],
            "depth_status": str(snapshot.visibility_depth_status),
        }

    def set_interaction_feedback(self, message: str):
        self._interaction_feedback = message

    def step(
        self, decision: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        from tarsphere.infra.manipulation.g1_attachment import (
            attachment_command_from_decision,
        )

        _rollout_memory_diagnostic("step.begin", self._physics_steps)
        action = str(decision.get("action") or "")
        if (
            action == "stand_up"
            and self._g1_carried_hand is not None
            and (self._last_g1_arm_ik is not None)
        ):
            return self._execute_physical_carry_stand_up(decision)
        self._reject_crouched_g1_locomotion(action)
        if action == "navigate":
            if "controller_action" in decision:
                raise ValueError(
                    "navigate is one VLM skill; DualVLN inner commands "
                    "must not arrive as a stuffed controller_action."
                )
            return self._execute_navigate_skill(decision)
        self._remember_g1_body_posture(decision)
        self._apply_camera_gaze_tilt_decision(decision)
        if action == "handover_object":
            self._apply_handover_object(decision)
        try:
            decision = self._prepare_evaluated_grasp_grounding(decision)
            decision = self._compile_agent_grasp_decision(decision)
        except GraspNotPossibleHere as refusal:
            # Where the robot stands is the agent's decision, so an arm that
            # cannot reach from there is a measurement of that decision. The step
            # reports what was measured and the episode continues; the agent may
            # move and try again.
            print(f"[G1-GRASP] refused: {refusal}", flush=True)
            return (
                self.observe(),
                False,
                self._finish_evaluated_grasp_grounding(
                    {
                        "executed": False,
                        "feedback": str(refusal),
                        "grasp_failure": refusal.as_feedback(),
                        "grasp_reachability": dict(refusal.reachability),
                        "physics_steps": self._physics_steps,
                        "robot_stability": self._robot_stability(),
                    },
                    succeeded=False,
                ),
            )
        decision = self._compile_move_end_effector_decision(decision)
        compiled_metadata = decision.get("controller_metadata")
        if isinstance(compiled_metadata, dict) and isinstance(
            compiled_metadata.get("g1_grasp_plan"), dict
        ):
            observation, done, info = self._execute_agent_grasp_plan(decision)
            succeeded = not info.get("grasp_failure") and bool(info.get("executed"))
            return (
                observation,
                done,
                self._finish_evaluated_grasp_grounding(
                    info,
                    succeeded=succeeded,
                ),
            )
        try:
            decision = self._resolve_g1_arm_ik_decision(decision)
        except GraspNotPossibleHere as refusal:
            if action != "move_end_effector":
                raise
            print(f"[G1-EEF] refused: {refusal}", flush=True)
            return (
                self.observe(),
                False,
                {
                    "executed": False,
                    "feedback": str(refusal),
                    "end_effector_motion": {
                        "executed": False,
                        "failure": refusal.as_feedback(),
                    },
                    "physics_steps": self._physics_steps,
                    "robot_stability": self._robot_stability(),
                    "private_online_mapping": dict(self._private_mapping_stats),
                },
            )
        decision = self._include_dex3_controller_action(decision)
        transition = attachment_command_from_decision(decision)
        attachment_feedback = ""
        attachment_executed = True
        grasp_failure: Dict[str, Any] | None = None
        placement_release_report: Dict[str, Any] | None = None
        if transition is not None and self._g1_attachment is None:
            raise RuntimeError(
                "G1 attachment action requested, but this episode has no "
                "physical task target."
            )
        command = self._robot_body.command_for_decision(decision)
        _rollout_memory_diagnostic(
            f"step.command_ready.{command.action_name}", self._physics_steps
        )
        early_attachment: Dict[str, Any] | None = None
        late_attachment_error: RuntimeError | None = None
        manipulation_stance_held = False
        resolved_metadata = decision.get("controller_metadata")
        placement_window = bool(
            transition is not None
            and transition[0] == "release"
            and isinstance(resolved_metadata, dict)
            and isinstance(resolved_metadata.get("g1_place_ik_result"), dict)
        )
        grasp_reach_window = bool(
            self._g1_attachment is not None
            and isinstance(resolved_metadata, dict)
            and resolved_metadata.get("g1_pin_base_for_arm_reach", False)
            and isinstance(resolved_metadata.get("g1_arm_ik_result"), dict)
        )

        def _attempt_attachment(hand: str) -> Dict[str, Any]:
            agent_grasp = (
                resolved_metadata.get("agent_grasp")
                if isinstance(resolved_metadata, dict)
                else None
            )
            object_id = (
                str(agent_grasp.get("object_id") or "")
                if isinstance(agent_grasp, dict)
                else ""
            )
            if not object_id:
                raise RuntimeError(
                    "Physical grasp requires the target selected by current "
                    "RGB-D grounding."
                )
            return self._g1_attachment.begin_physical_grasp(
                hand,
                object_id=object_id,
                closing_axis_world=list(agent_grasp.get("closing_axis_world") or []),
                finger_joint_positions=(self._dex3_joint_positions(hand)),
            )

        def _hold_manipulation_stance() -> None:
            nonlocal manipulation_stance_held
            if manipulation_stance_held:
                return
            self._g1_attachment.hold_manipulation_stance()
            manipulation_stance_held = True

        if (
            transition is not None
            and transition[0] == "attach"
            or placement_window
            or grasp_reach_window
        ):
            _hold_manipulation_stance()

        def _release_manipulation_stance() -> None:
            nonlocal manipulation_stance_held
            if not manipulation_stance_held:
                return
            self._g1_attachment.release_manipulation_stance()
            manipulation_stance_held = False

        arm_pose_completion_report = (
            (decision.get("controller_metadata") or {}).get("g1_arm_ik_result")
            if isinstance(decision.get("controller_metadata"), dict)
            else None
        )
        # ``prepare_grasp`` must converge to its refined arm command before the
        # oracle may inspect the result. Once native PhysX reports the exact
        # hand–object contact, however, that contact is the authoritative
        # attachment gate. Requiring the arm to keep converging after contact
        # asks it to penetrate the collider and turns valid contact into a
        # false failure.
        finish_arm_pose = bool(
            isinstance(decision.get("controller_metadata"), dict)
            and decision["controller_metadata"].get("g1_arm_ik_finish_on_target", False)
        )
        arm_pose_reached = False

        def _observe_grasp_arm_completion() -> bool:
            if (
                transition is None
                or transition[0] != "attach"
                or self._g1_attachment is None
            ):
                return False
            if finish_arm_pose and not arm_pose_reached:
                if not _finish_when_arm_pose_reached():
                    return False
            # A friction grasp needs the complete authored finger-closing
            # horizon. Stopping on the first transient contact weakens the
            # normal force and turns an apparent early success into a drop on
            # lift. The final-tick gate below measures the closed hand.
            return False

        def _finish_when_arm_pose_reached() -> bool:
            nonlocal arm_pose_reached
            if not finish_arm_pose or not isinstance(arm_pose_completion_report, dict):
                return False
            hand = str(arm_pose_completion_report.get("hand") or "")
            target_joints = np.asarray(
                arm_pose_completion_report.get("joint_positions"),
                dtype=np.float64,
            ).reshape(-1)
            current_joints = self._arm_joint_positions(hand)
            if current_joints is None or target_joints.size != len(current_joints):
                return False
            max_error = float(
                np.max(
                    np.abs(np.asarray(current_joints, dtype=np.float64) - target_joints)
                )
            )
            tolerance = float(
                decision["controller_metadata"].get(
                    "g1_arm_ik_finish_tolerance_rad", 0.08
                )
            )
            if max_error > tolerance:
                return False
            if action == "move_end_effector":
                from tarsphere.embodiment.dex3 import (
                    dex3_continuous_closure,
                )

                raw_delta = decision.get("delta")
                if not isinstance(raw_delta, (list, tuple)) or len(raw_delta) != 7:
                    raise RuntimeError(
                        "move_end_effector lost its seven-coordinate command."
                    )
                measured_fingers = self._dex3_joint_positions(G1_MANIPULATION_HAND)
                measured_closure = dex3_continuous_closure(
                    G1_MANIPULATION_HAND,
                    measured_fingers,
                )
                closure_error = abs(measured_closure - float(raw_delta[6]))
                if closure_error > G1_END_EFFECTOR_GRIPPER_TRACKING_TOLERANCE:
                    return False
            arm_pose_reached = True
            print(
                "[G1-IK] physical arm target reached: "
                f"hand={hand} max_joint_error_rad={max_error:.4f} "
                f"tolerance_rad={tolerance:.4f}",
                flush=True,
            )
            return True

        action_start_physics_step = self._physics_steps
        try:
            result = self._robot_body.step_simulation(
                env=self._env,
                action_name=command.action_name,
                env_action=command.env_action,
                physics_step_budget=getattr(command, "physics_step_budget", None),
                step_human_bodies=self._step_human_bodies,
                pump_kit_updates=self._pump_kit_updates,
                pump_motion_window=self._pump_human_motion_window,
                global_tick=self._physics_steps,
                sample_body_dynamics=self._g1_body_dynamics,
                relative_pose_target=command.relative_pose_target,
                relative_pose_profile=command.relative_pose_profile,
                sample_body_pose=(
                    self._g1_relative_pose_measurement
                    if command.relative_pose_target is not None
                    else None
                ),
                physics_step_limit=decision.get("navigation_physics_step_limit"),
                begin_decision_window=self._begin_decision_render_window,
                prepare_final_frame=self._prepare_final_decision_frame,
                end_decision_window=self._end_decision_render_window,
                diagnose_physics_tick=self._on_physics_tick,
                observe_motion_tick=self._capture_private_mapping_tick,
                finish_when=(
                    _observe_grasp_arm_completion
                    if transition is not None and transition[0] == "attach"
                    else _finish_when_arm_pose_reached
                    if finish_arm_pose
                    else None
                ),
                abort_requested=self._abort_requested,
            )
            # ``finish_when`` is evaluated after each non-terminal tick.  If
            # the hand enters the gate on the final tick, confirm physical
            # contact before releasing the grasp-window base fixture.
            if (
                transition is not None
                and transition[0] == "attach"
                and result.physics_steps > 0
                and early_attachment is None
            ):
                try:
                    if finish_arm_pose and not arm_pose_reached:
                        raise RuntimeError(
                            "G1 grasp arm did not physically converge before "
                            "the authored grasp horizon; contact retention was not established."
                        )
                    early_attachment = _attempt_attachment(transition[1])
                    self._g1_carried_hand = str(early_attachment["hand"])
                except RuntimeError as exc:
                    late_attachment_error = exc
        finally:
            # Grasp follows the official flow and resumes whole-body control as
            # soon as contact retention is established. Placement remains pinned until
            # the object joint is released below.
            if not placement_window:
                _release_manipulation_stance()
        try:
            if result.observation:
                self._obs = result.observation
            if bool(result.balance_control.get("intervened", False)):
                trigger_action_tick = int(result.balance_control["trigger_action_tick"])
                result.balance_control["trigger_physics_step"] = (
                    action_start_physics_step + trigger_action_tick + 1
                )
            self._physics_steps += result.physics_steps
            trace_position = self._robot_position()
            trace_stability = self._robot_stability()
            print(
                "[TRACE] G1 action result: "
                f"action={command.action_name} physics_steps={self._physics_steps} "
                f"position={None if trace_position is None else [round(float(v), 4) for v in trace_position]} "
                f"yaw_rad={self._robot_yaw():.4f} stability={trace_stability}",
                flush=True,
            )
            self._assert_robot_upright(
                action_name=command.action_name,
                env_action=command.env_action,
            )
            if (
                transition is not None
                and transition[0] == "attach"
                and result.physics_steps <= 0
            ):
                attachment_executed = False
                attachment_feedback = (
                    "G1 attachment was not attempted because the gripper command "
                    "advanced zero physics steps."
                )
            elif transition is not None and transition[0] == "attach":
                try:
                    if late_attachment_error is not None:
                        raise late_attachment_error
                    attached = early_attachment
                    if attached is None:
                        raise RuntimeError(
                            "G1 grasp window ended without an attachment result."
                        )
                    self._g1_carried_hand = str(attached["hand"])
                    object_id = str(attached["object_id"])
                    if attached.get("retention") == "finite_coulomb_friction":
                        physical_hold = self._g1_attachment.physical_hold_state(
                            object_id=object_id
                        )
                        if not physical_hold.get("held"):
                            raise RuntimeError(
                                "Dex3 contact was not retained after closure: "
                                f"{physical_hold!r}"
                            )
                        print(
                            "[G1-GRASP] friction contact established: "
                            f"object={object_id} hand={self._g1_carried_hand} "
                            "constraint_joint_created=false "
                            f"contact_links="
                            f"{[item['hand_link'] for item in attached['physx_contacts']]} "
                            f"aabb_surface_gap_m="
                            f"{float(attached['aabb_surface_gap_m']):.4f}",
                            flush=True,
                        )
                    else:
                        raise RuntimeError(
                            "Grasp retention must use finite Coulomb friction."
                        )
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"G1 grasp retention failed (fail-loud): {exc}"
                    ) from exc
            if transition is not None and transition[0] == "release":
                # Execute any compiled placement arm trajectory before breaking the
                # active retention. This lets one legal ``place_object`` decision mean
                # "move the held object to the requested support pose, then
                # release" without exposing an internal prepare-place action in
                # the episode contract.
                try:
                    if placement_window:
                        placement_report = resolved_metadata["g1_place_ik_result"]
                        if not arm_pose_reached:
                            # Joint tracking is only an internal early-stop
                            # optimization.  The task contract is the held
                            # object's live support error, so a compliant arm
                            # that exhausts this bounded horizon must continue
                            # into the same geometry/refinement gate used by
                            # every subsequent residual correction.
                            errors = self._g1_pre_release_placement_errors(
                                placement_report
                            )
                            print(
                                "[G1-PLACE] initial arm transition exhausted "
                                "its joint-space horizon; deferring to live "
                                "object geometry: "
                                f"xy_error_m={float(errors['xy_error_m']):.4f} "
                                "support_gap_error_m="
                                f"{float(errors['support_gap_error_m']):.4f}.",
                                flush=True,
                            )
                        for correction_gain in _G1_PLACEMENT_REFINEMENT_GAINS:
                            if self._g1_pre_release_placement_within_tolerance(
                                placement_report,
                            ):
                                break
                            errors = self._g1_pre_release_placement_errors(
                                placement_report
                            )
                            horizontal_limit = (
                                G1_PRE_RELEASE_HORIZONTAL_TOLERANCE_M
                                + _G1_PLACEMENT_NUMERICAL_MARGIN_M
                            )
                            if float(errors["xy_error_m"]) > horizontal_limit:
                                # The placement fixture serves the arm IK
                                # window. Temporarily release it for one
                                # measured body-velocity correction, then
                                # restore it before the next arm solve.
                                _release_manipulation_stance()
                                try:
                                    placement_report = self._refine_g1_pre_release_base(
                                        placement_report
                                    )
                                finally:
                                    _hold_manipulation_stance()
                                if self._g1_pre_release_placement_within_tolerance(
                                    placement_report,
                                ):
                                    break
                            placement_report = self._refine_g1_pre_release_placement(
                                decision,
                                correction_gain=correction_gain,
                            )
                            resolved_metadata["g1_place_ik_result"] = placement_report
                        # An arm correction can close the vertical error while
                        # slightly moving an offset payload in XY. Finish with
                        # one measured base-only correction when height is
                        # already valid; never ask the arm to undo that motion.
                        final_errors = self._g1_pre_release_placement_errors(
                            placement_report
                        )
                        horizontal_limit = G1_PRE_RELEASE_HORIZONTAL_TOLERANCE_M
                        support_gap_limit = G1_PRE_RELEASE_SUPPORT_CONTACT_TOLERANCE_M
                        if (
                            float(final_errors["support_gap_error_m"])
                            <= support_gap_limit
                            and float(final_errors["xy_error_m"]) > horizontal_limit
                        ):
                            _release_manipulation_stance()
                            try:
                                placement_report = self._refine_g1_pre_release_base(
                                    placement_report
                                )
                            finally:
                                _hold_manipulation_stance()
                        self._assert_g1_pre_release_placement(placement_report)
                    pre_release_evidence = (
                        self._g1_pre_release_evidence(
                            placement_report,
                            decision_step=int(decision.get("action_step", 0) or 0),
                            controller_metadata=resolved_metadata,
                        )
                        if placement_window
                        else None
                    )
                    released = self._g1_attachment.release(transition[1])
                    hand_opening = self._execute_g1_hand_opening(
                        transition[1],
                        decision_step=int(decision.get("action_step", 0) or 0),
                    )
                    released_object_id = str(released.get("object_id") or "")
                    released_state = None
                    if released_object_id:
                        released_state = (
                            self._g1_attachment.snapshot()
                            .get("objects", {})
                            .get(released_object_id)
                        )
                    released_state_valid = self._g1_released_state_valid(released_state)
                    if released_object_id and not released_state_valid:
                        raise RuntimeError(
                            "Dex3 opened but the target still satisfies the "
                            "active retention contract."
                        )
                    placement_release_report = {
                        **released,
                        "retention_mode": "physical",
                        "pre_release": pre_release_evidence,
                        "hand_opening": hand_opening,
                    }
                    self._g1_carried_hand = None
                    self._g1_carry_finger_positions = None
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"G1 placement release failed (fail-loud): {exc}"
                    ) from exc
            carry_contact_confirmation = (
                self._confirm_g1_carry_contact()
                if self._g1_attachment is not None
                else None
            )
            dropped_objects = (
                self._g1_attachment.detect_dropped_objects()
                if self._g1_attachment is not None
                else []
            )
            dropped_ids: list[str] = []
            if dropped_objects:
                self._g1_carried_hand = None
                self._g1_carry_finger_positions = None
                dropped_ids = [
                    str(state.get("object_id") or "") for state in dropped_objects
                ]
                if any(not object_id for object_id in dropped_ids):
                    raise RuntimeError(
                        "A dropped grasp target is missing its object id."
                    )
                drop_feedback = (
                    "The physically grasped object lost live hand contact "
                    f"during this action and is no longer held: {dropped_ids}."
                )
                attachment_feedback = " ".join(
                    part for part in (attachment_feedback, drop_feedback) if part
                )
                print(f"[G1-GRASP] {drop_feedback}", flush=True)
            physical_contacts = (
                self._robot_contact_monitor.drain_contacts()
                if self._robot_contact_monitor is not None
                else ()
            )
            human_safety_violations = (
                self._robot_contact_monitor.drain_human_safety_violations()
                if self._robot_contact_monitor is not None
                else ()
            )
            robot_position = self._robot_position()
            physical_contact_details = (
                self._robot_contact_monitor.public_details(
                    physical_contacts,
                    robot_position=robot_position,
                    robot_yaw_rad=self._robot_yaw(),
                )
                if self._robot_contact_monitor is not None
                and physical_contacts
                and robot_position is not None
                else []
            )
            contact_feedback = (
                self._robot_contact_monitor.feedback(
                    physical_contacts,
                    robot_position=robot_position,
                    robot_yaw_rad=self._robot_yaw(),
                )
                if self._robot_contact_monitor is not None
                and physical_contacts
                and robot_position is not None
                else ""
            )
            human_safety_feedback = (
                "Unsafe physical contact with "
                + ", ".join(
                    sorted(
                        {
                            str(violation.human_id)
                            for violation in human_safety_violations
                        }
                    )
                )
                + " was measured. Separate from the Human before continuing."
                if human_safety_violations
                else ""
            )
            execution_feedback = " ".join(
                part
                for part in (
                    command.feedback,
                    result.balance_feedback,
                    (
                        "Measured body-pose error: position "
                        f"{float(result.relative_pose_control['final_translation_error_m']):.3f} m; "
                        "yaw "
                        f"{float(result.relative_pose_control['final_yaw_error_deg']):.2f} deg."
                        if result.relative_pose_control
                        and bool(result.relative_pose_control.get("timed_out"))
                        else ""
                    ),
                    contact_feedback,
                    human_safety_feedback,
                    attachment_feedback,
                )
                if part
            )
            end_effector_motion = None
            if action == "move_end_effector":
                end_effector_motion = self._move_end_effector_report(resolved_metadata)
                execution_feedback = " ".join(
                    part
                    for part in (
                        execution_feedback,
                        str(end_effector_motion["feedback"]),
                    )
                    if part
                )
            # The claim outlives its step so later snapshots can report what
            # the human holds, so it must only speak for the step that made it.
            # Without this guard it overwrites every later step's feedback and
            # forces `executed`, for the whole remainder of the episode.
            handover_claim = (
                self._last_handover_claim if action == "handover_object" else None
            )
            if handover_claim is not None:
                execution_feedback = str(handover_claim["feedback"])
            if execution_feedback:
                self._interaction_feedback = execution_feedback
            _rollout_memory_diagnostic("step.before_observe", self._physics_steps)
            observation = self.observe()
            _rollout_memory_diagnostic("step.after_observe", self._physics_steps)
            info = {
                "executed": command.executed and attachment_executed,
                "feedback": execution_feedback,
                "physical_contact": bool(physical_contacts),
                "physical_contact_details": physical_contact_details,
                "native_contacts": [contact.as_dict() for contact in physical_contacts],
                "retention_contacts": (
                    self._robot_contact_monitor.snapshot()["retention_contacts"]
                    if self._robot_contact_monitor is not None
                    else []
                ),
                "safety_violation": (
                    {
                        "type": "unsafe_human_contact",
                        "events": [
                            violation.as_dict() for violation in human_safety_violations
                        ],
                    }
                    if human_safety_violations
                    else None
                ),
                "balance_control": dict(result.balance_control),
                "relative_pose_control": dict(result.relative_pose_control),
                "travelled_distance_m": result.travelled_distance_m,
                "navigation_motion_samples": result.navigation_motion_samples,
                "post_motion_settle_ticks": result.post_motion_settle_ticks,
                "grasp_failure": grasp_failure,
                # A carried target that leaves the hand ends the task. The
                # runner records this action, then terminates before another
                # policy decision can reason from stale holding state.
                "dropped_objects": dropped_ids,
                "carry_contact_confirmation": carry_contact_confirmation,
                "placement_release": placement_release_report,
                # Public outcome is explicit; the private evidence report is
                # not itself interpreted as a Boolean by the Harness.
                "placed": placement_release_report is not None,
                "physics_steps": self._physics_steps,
                "robot_stability": self._robot_stability(),
                # Motion-time RGB-D samples are evaluator-private diagnostics.
                # Expose their cumulative counters to the private heartbeat;
                # build_agent_feedback deliberately does not publish them to
                # the evaluated policy.
                "private_online_mapping": dict(self._private_mapping_stats),
            }
            if end_effector_motion is not None:
                info["end_effector_motion"] = end_effector_motion
            if handover_claim is not None:
                info["handover"] = dict(handover_claim["private"])
                info["executed"] = True
            return (
                observation,
                bool(result.terminated or result.truncated),
                info,
            )
        finally:
            _release_manipulation_stance()

    def _execute_navigate_skill(
        self,
        decision: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        """Run DualVLN's official inner loop through the ``move_body`` primitive."""

        raw = decision.get("instruction")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("navigate requires `instruction`.")
        instruction = raw.strip()
        if not internav_backend_available():
            raise RuntimeError(
                "navigate reached Isaac without a DualVLN backend. "
                "Set TARSPHERE_INTERNNAV_HOST/PORT."
            )
        backend = build_internnav_backend()
        backend.reset()
        self._reset_dualvln_camera_tilt()
        inner: list[Dict[str, Any]] = []
        try:
            observation, done, info = self._run_dualvln_inner_loop(
                instruction,
                backend,
                inner,
            )
        finally:
            self._reset_dualvln_camera_tilt()
        return observation, done, info

    def _run_dualvln_inner_loop(
        self,
        instruction: str,
        backend: Any,
        inner: list[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        stop_reason = "max_inner_steps"
        human_safety_violations: list[Any] = []
        for index in range(DUALVLN_MAX_INNER_STEPS):
            if self._abort_requested():
                stop_reason = "abort"
                break
            observation = self._capture_dualvln_observation(instruction)
            command = backend.step(observation)
            inner.append(self._dualvln_inner_step_record(index, observation, command))
            self._dump_dualvln_nav_frame(index, observation)
            if command.status == "stop":
                stop_reason = "dualvln_stop"
                break
            self._apply_navigate_command(command)
            if self._robot_contact_monitor is not None:
                human_safety_violations = (
                    self._robot_contact_monitor.drain_human_safety_violations()
                )
                if human_safety_violations:
                    stop_reason = "human_contact"
                    break
        # The inner policy may stop while LOOKDOWN is active. Restore the
        # evaluated Agent's normal gaze before producing its next observation.
        self._reset_dualvln_camera_tilt()
        observation = self.observe()
        human_ids = sorted(
            {str(violation.human_id) for violation in human_safety_violations}
        )
        feedback = (
            f"DualVLN finished {instruction!r} after {len(inner)} inner "
            f"steps ({stop_reason})."
        )
        if human_ids:
            feedback = (
                f"{feedback} Unsafe physical contact with "
                f"{', '.join(human_ids)} was measured."
            )
        info = {
            "executed": True,
            "feedback": feedback,
            "physics_steps": self._physics_steps,
            "robot_stability": self._robot_stability(),
            "navigate_skill": {
                "backend": backend.name,
                "instruction": instruction,
                "inner_steps": inner,
                "stop_reason": stop_reason,
            },
        }
        if human_safety_violations:
            info["safety_violation"] = {
                "type": "unsafe_human_contact",
                "human_ids": human_ids,
            }
        return observation, False, info

    def _navigate_body_decision(self, command: Any) -> Dict[str, Any]:
        """Return the ``move_body`` or wait decision DualVLN reuses."""

        if command.status in ("hold", "look_down", "look_up"):
            return {"action": "wait"}
        if command.status != "continue" or command.delta is None:
            raise ValueError(
                "DualVLN continue must carry a body-frame delta; "
                f"got status={command.status!r} delta={command.delta!r}."
            )
        return {
            "action": "move_body",
            "delta": [float(value) for value in command.delta],
        }

    def _dualvln_inner_step_record(
        self,
        index: int,
        observation: DualVLNObservation,
        command: Any,
    ) -> Dict[str, Any]:
        """Record one DualVLN decision with the live G1 pose and RGB stats."""

        position, _orientation, yaw = self._read_robot_base_pose_now()
        action_id: Any = None
        raw = command.raw_action
        if isinstance(raw, (list, tuple)) and raw:
            payload = raw[0] if isinstance(raw[0], dict) else raw
            if isinstance(payload, dict):
                current = payload.get("action")
                while isinstance(current, (list, tuple)) and len(current) == 1:
                    current = current[0]
                action_id = current
        return {
            "inner_step": index,
            "status": command.status,
            "raw_action": command.raw_action,
            "action_id": action_id,
            "xy": [float(value) for value in position[:3]],
            "yaw": float(yaw),
            "camera_tilt_extra_deg": float(self._dualvln_look_down_extra_deg),
            "camera_pitch_deg": float(self._dualvln_camera_pitch_deg()),
            "rgb_mean": float(np.mean(observation.rgb)),
            "rgb_std": float(np.std(observation.rgb)),
            "depth_mean": float(np.mean(observation.depth)),
        }

    def _dump_dualvln_nav_frame(
        self,
        index: int,
        observation: DualVLNObservation,
    ) -> None:
        """Write the shared ego RGB-D used by the DualVLN inner loop."""

        if self.image_dir is None:
            return
        inner_dir = Path(self.image_dir) / "navigate_inner"
        inner_dir.mkdir(parents=True, exist_ok=True)
        depth = np.asarray(observation.depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0] * DUALVLN_DEPTH_MAX_M
        path = inner_dir / f"inner_{index:04d}_rgbd.png"
        path.write_bytes(compose_rgbd_image(observation.rgb, depth))

    def _apply_navigate_command(self, command: Any) -> None:
        """Apply one DualVLN spatial delta through the ``move_body`` primitive."""

        if command.status == "look_down":
            self._apply_dualvln_look_down()
        elif command.status == "look_up":
            self._apply_dualvln_look_up()
        decision = self._navigate_body_decision(command)
        decision = self._resolve_g1_arm_ik_decision(decision)
        body_command = self._robot_body.command_for_decision(decision)
        result = self._robot_body.step_simulation(
            env=self._env,
            action_name=body_command.action_name,
            env_action=body_command.env_action,
            physics_step_budget=getattr(
                body_command,
                "physics_step_budget",
                None,
            ),
            step_human_bodies=self._step_human_bodies,
            pump_kit_updates=self._pump_kit_updates,
            pump_motion_window=self._pump_human_motion_window,
            global_tick=self._physics_steps,
            sample_body_dynamics=self._g1_body_dynamics,
            relative_pose_target=body_command.relative_pose_target,
            relative_pose_profile=body_command.relative_pose_profile,
            sample_body_pose=(
                self._g1_relative_pose_measurement
                if body_command.relative_pose_target is not None
                else None
            ),
            begin_decision_window=self._begin_decision_render_window,
            prepare_final_frame=self._prepare_final_decision_frame,
            end_decision_window=self._end_decision_render_window,
            diagnose_physics_tick=self._on_physics_tick,
            observe_motion_tick=self._capture_private_mapping_tick,
            abort_requested=self._abort_requested,
        )
        if result.observation:
            self._obs = result.observation
        self._physics_steps += result.physics_steps
        self._assert_robot_upright(
            action_name=body_command.action_name,
            env_action=body_command.env_action,
        )

    def _capture_dualvln_observation(
        self,
        instruction: str,
    ) -> DualVLNObservation:
        """Capture DualVLN RGB-D from the evaluated G1 ego sensor."""

        if not self.enable_ego_sensors:
            raise RuntimeError(
                "DualVLN navigate requires the 640x480 G1 ego RGB-D camera."
            )
        self._force_one_ego_render()
        camera = self._robot_camera_sensor()
        if camera is None:
            raise RuntimeError(
                "DualVLN navigate cannot find the evaluated G1 ego sensor "
                'named "camera" at d435_link/Camera.'
            )
        data = camera.get_data()
        rgba = data.get("rgba") if isinstance(data, dict) else None
        depth = data.get("depth") if isinstance(data, dict) else None
        if rgba is None:
            raise RuntimeError(
                "DualVLN ego camera returned no RGB frame at physics step "
                f"{self._physics_steps}."
            )
        rgba = np.asarray(rgba.cpu() if hasattr(rgba, "cpu") else rgba)
        rgb = rgba[0, :, :, :3] if rgba.ndim == 4 else rgba[:, :, :3]
        if np.issubdtype(rgb.dtype, np.integer):
            rgb = np.clip(rgb, 0, 255).astype(np.uint8, copy=False)
        elif rgb.dtype != np.uint8:
            raise RuntimeError(
                "DualVLN ego camera RGB must be uint8; official "
                f"InternVLAN1_Policy calls Image.fromarray(rgb). Got {rgb.dtype}."
            )
        if rgb.shape[0] != DUALVLN_RGB_HEIGHT or rgb.shape[1] != DUALVLN_RGB_WIDTH:
            raise RuntimeError(
                "DualVLN ego camera must be "
                f"{DUALVLN_RGB_WIDTH}x{DUALVLN_RGB_HEIGHT}; "
                f"got {rgb.shape[1]}x{rgb.shape[0]}."
            )
        depth_m, depth_info = prepare_depth_frame(
            depth,
            (DUALVLN_RGB_HEIGHT, DUALVLN_RGB_WIDTH),
        )
        if str(depth_info.get("depth_status") or "") not in {
            "valid",
            "low_confidence",
        }:
            raise RuntimeError(
                "DualVLN ego camera depth is unusable: "
                f"{depth_info!r} at physics step {self._physics_steps}."
            )
        position, orientation, _yaw = self._read_robot_base_pose_now()
        if not isinstance(orientation, (list, tuple)) or len(orientation) != 4:
            raise RuntimeError(
                "DualVLN observation needs a finite robot quaternion; "
                f"got {orientation!r}."
            )
        w, x, y, z = (float(value) for value in orientation)
        return DualVLNObservation(
            rgb=np.ascontiguousarray(rgb),
            depth=official_norm_depth(depth_m),
            instruction=instruction,
            globalgps=(
                float(position[0]),
                float(position[1]),
                float(position[2]),
            ),
            globalrotation=(x, y, z, w),
        )

    def _assert_g1_pre_release_placement(
        self,
        placement_report: Dict[str, Any],
    ) -> None:
        """Fail loud unless the held payload is physically ready to set down."""

        errors = self._g1_pre_release_placement_errors(placement_report)
        xy_error = float(errors["xy_error_m"])
        support_gap_error = float(errors["support_gap_error_m"])
        if (
            xy_error > G1_PRE_RELEASE_HORIZONTAL_TOLERANCE_M
            or support_gap_error > G1_PRE_RELEASE_SUPPORT_CONTACT_TOLERANCE_M
            or not bool(errors["support_point_below_object"])
        ):
            raise RuntimeError(
                "G1 placement did not reach the physical set-down envelope: "
                f"xy_error_m={xy_error:.4f}, "
                f"horizontal_tolerance_m="
                f"{G1_PRE_RELEASE_HORIZONTAL_TOLERANCE_M:.4f}, "
                f"support_gap_error_m={support_gap_error:.4f}, "
                f"support_gap_tolerance_m="
                f"{G1_PRE_RELEASE_SUPPORT_CONTACT_TOLERANCE_M:.4f}, "
                f"support_point_below_object="
                f"{bool(errors['support_point_below_object'])}."
            )

    def _g1_pre_release_evidence(
        self,
        placement_report: Dict[str, Any],
        *,
        decision_step: int,
        controller_metadata: Any,
    ) -> Dict[str, Any]:
        """Capture the held payload at the verified support pose.

        ``place_object`` is one macro-action: the arm reaches the support and
        the fingers open within the same decision.  Consequently, the previous
        decision snapshot is not release evidence.  This record is sampled
        after the live support gate and immediately before retention is ended.
        """

        if self._g1_attachment is None:
            raise RuntimeError("Pre-release evidence requires an attachment executor.")
        geometry = self._g1_pre_release_placement_geometry(placement_report)
        object_id = str(geometry["object_id"])
        destination_object_id = self._placement_destination_object_id(
            controller_metadata
        )
        state = self._g1_attachment.snapshot().get("objects", {}).get(object_id)
        if not isinstance(state, dict):
            raise RuntimeError(f"Pre-release evidence is missing target {object_id!r}.")
        physical_hold = state.get("physical_hold")
        retained = bool(
            state.get("physically_held") is True
            and state.get("held_by_hand") in {"left", "right"}
            and state.get("attachment_joint_present") is False
            and isinstance(physical_hold, dict)
            and physical_hold.get("held") is True
            and physical_hold.get("retention") == "finite_coulomb_friction"
            and bool(physical_hold.get("physx_contacts"))
        )
        if not retained:
            raise RuntimeError(
                f"Target {object_id!r} lost physical retention before "
                "the support gate completed."
            )
        destination_state = (
            self._g1_attachment.snapshot().get("objects", {}).get(destination_object_id)
        )
        if not isinstance(destination_state, dict):
            raise RuntimeError(
                "Pre-release evidence is missing destination "
                f"{destination_object_id!r}."
            )
        monitor = self._g1_attachment.grasp_contact_monitor
        if monitor is None:
            raise RuntimeError(
                "Pre-release support evidence requires a live PhysX contact monitor."
            )
        support_contact = monitor.support_contact(
            object_id=object_id,
            destination_object_id=destination_object_id,
        )
        if support_contact is None:
            raise RuntimeError(
                f"Target {object_id!r} has no live PhysX contact with "
                f"destination {destination_object_id!r}."
            )
        return {
            "decision_step": int(decision_step),
            "physics_step": int(self._physics_steps),
            "object_id": object_id,
            "destination_object_id": destination_object_id,
            "retention_mode": "physical",
            "hand": state.get("held_by_hand") or state.get("hand"),
            "retention": {
                "object_id": object_id,
                "attached": state.get("attached"),
                "attachment_joint_present": state.get("attachment_joint_present"),
                "attachment_joint_valid": state.get("attachment_joint_valid"),
                "hand": state.get("hand"),
                "physically_held": state.get("physically_held"),
                "held_by_hand": state.get("held_by_hand"),
                "physical_hold": dict(physical_hold or {}),
                "live_hand_contact_count": len(
                    (physical_hold or {}).get("physx_contacts") or ()
                ),
            },
            "target_bbox_min": [float(value) for value in geometry["bbox_min"]],
            "target_bbox_max": [float(value) for value in geometry["bbox_max"]],
            "support_surface_world": [
                float(value) for value in geometry["support_surface_world"]
            ],
            "destination_bbox_min": [
                float(value) for value in destination_state["bbox_min"]
            ],
            "destination_bbox_max": [
                float(value) for value in destination_state["bbox_max"]
            ],
            "support_contact": support_contact.as_dict(),
        }

    def _placement_destination_object_id(
        self,
        controller_metadata: Any,
    ) -> str:
        """Resolve the task destination from private compiled metadata."""

        if isinstance(controller_metadata, dict):
            certification = controller_metadata.get("certification_place_binding")
            if isinstance(certification, dict):
                value = str(certification.get("destination_object_id") or "").strip()
                if value:
                    return value
        world = getattr(self.episode, "world_knowledge", None) or {}
        task = world.get("physical_task") if isinstance(world, dict) else None
        value = str(
            (task or {}).get("final_destination_object_id")
            if isinstance(task, dict)
            else ""
        ).strip()
        if not value:
            raise RuntimeError(
                "place_object requires the compiled physical-task destination."
            )
        return value

    @staticmethod
    def _g1_released_state_valid(
        state: Any,
    ) -> bool:
        """Return whether one target no longer satisfies its retention mode."""

        if not isinstance(state, dict):
            return False
        unconstrained = bool(
            state.get("attached") is False
            and state.get("attachment_joint_present") is False
            and state.get("attachment_joint_valid") is False
        )
        return bool(
            unconstrained
            and state.get("physically_held") is False
            and state.get("held_by_hand") not in {"left", "right"}
        )

    def _g1_pre_release_placement_within_tolerance(
        self,
        placement_report: Dict[str, Any],
    ) -> bool:
        errors = self._g1_pre_release_placement_errors(placement_report)
        return (
            float(errors["xy_error_m"]) <= G1_PRE_RELEASE_HORIZONTAL_TOLERANCE_M
            and float(errors["support_gap_error_m"])
            <= G1_PRE_RELEASE_SUPPORT_CONTACT_TOLERANCE_M
            and bool(errors["support_point_below_object"])
        )

    def _g1_pre_release_placement_errors(
        self,
        placement_report: Dict[str, Any],
    ) -> Dict[str, float]:
        """Measure the held object's live support error without releasing it."""
        geometry = self._g1_pre_release_placement_geometry(placement_report)
        support = geometry["support_surface_world"]
        lower = geometry["bbox_min"]
        upper = geometry["bbox_max"]
        center_xy = geometry["bbox_center_xy"]
        xy_error = float(np.linalg.norm(center_xy - support[:2]))
        support_gap_error = abs(float(lower[2]) - float(support[2]))
        support_point_below_object = bool(
            float(lower[0]) <= float(support[0]) <= float(upper[0])
            and float(lower[1]) <= float(support[1]) <= float(upper[1])
        )
        print(
            "[G1-PLACE] pre-release live geometry: "
            f"object={geometry['object_id']} xy_error_m={xy_error:.4f} "
            f"support_gap_error_m={support_gap_error:.4f} "
            f"support_point_below_object={support_point_below_object} "
            f"bbox_min={[round(float(v), 4) for v in lower]} "
            f"bbox_max={[round(float(v), 4) for v in upper]}",
            flush=True,
        )
        return {
            "xy_error_m": xy_error,
            "support_gap_error_m": support_gap_error,
            "support_point_below_object": support_point_below_object,
        }

    def _g1_pre_release_placement_geometry(
        self,
        placement_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return the authoritative live payload/support geometry."""
        if self._g1_attachment is None:
            raise RuntimeError(
                "Placement verification requires an attachment executor."
            )
        object_id = str(placement_report.get("object_id") or "")
        support = np.asarray(
            placement_report.get("support_surface_world"), dtype=np.float64
        ).reshape(-1)
        clearance = float(placement_report.get("clearance_m", 0.0))
        objects = self._g1_attachment.snapshot().get("objects") or {}
        state = objects.get(object_id) if isinstance(objects, dict) else None
        if not isinstance(state, dict) or support.size != 3:
            raise RuntimeError(
                f"Placement verification is missing live geometry for {object_id!r}."
            )
        lower = np.asarray(state.get("bbox_min"), dtype=np.float64).reshape(-1)
        upper = np.asarray(state.get("bbox_max"), dtype=np.float64).reshape(-1)
        if lower.size != 3 or upper.size != 3:
            raise RuntimeError(
                f"Placement verification found an invalid bbox for {object_id!r}."
            )
        center_xy = 0.5 * (lower[:2] + upper[:2])
        return {
            "object_id": object_id,
            "support_surface_world": support,
            "clearance_m": clearance,
            "bbox_min": lower,
            "bbox_max": upper,
            "bbox_center_xy": center_xy,
        }

    def _refine_g1_pre_release_base(
        self,
        placement_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Close one live payload XY residual with ordinary G1 body velocity.

        Moving the base translates the attached hand/object relation without
        asking an already extended arm to absorb the entire world-space
        residual. The caller owns the temporary manipulation fixture and
        re-pins the pelvis immediately after this bounded physical horizon.
        The official gait is standing-only, so this correction is refused
        while the pelvis is crouched.
        """
        self._reject_crouched_g1_locomotion("precise_approach")
        if self._last_g1_arm_ik is None:
            raise RuntimeError(
                "G1 placement base refinement requires a live arm hold command."
            )
        upper_command = np.asarray(
            self._last_g1_arm_ik.get("upper_body_command"),
            dtype=np.float64,
        ).reshape(-1)
        if upper_command.size != 15 or not np.all(np.isfinite(upper_command)):
            raise RuntimeError(
                "G1 placement base refinement requires a finite 15-D arm hold."
            )

        geometry = self._g1_pre_release_placement_geometry(placement_report)
        support_xy = np.asarray(
            geometry["support_surface_world"][:2],
            dtype=np.float64,
        )
        center_xy = np.asarray(geometry["bbox_center_xy"], dtype=np.float64)
        residual_world = support_xy - center_xy
        distance = float(np.linalg.norm(residual_world))
        if distance <= 1e-9:
            return placement_report

        speed = min(
            _G1_PLACEMENT_BASE_MAX_SPEED_MPS,
            max(_G1_PLACEMENT_BASE_MIN_SPEED_MPS, distance),
        )
        direction_world = residual_world / distance
        yaw = self._robot_yaw()
        forward = speed * (
            math.cos(yaw) * direction_world[0] + math.sin(yaw) * direction_world[1]
        )
        lateral = speed * (
            -math.sin(yaw) * direction_world[0] + math.cos(yaw) * direction_world[1]
        )
        decision = {
            "action": "precise_approach",
            "controller_action": {
                "move_by_speed": [
                    float(forward),
                    float(lateral),
                    0.0,
                    0.74,
                    [float(value) for value in upper_command],
                ],
            },
            "controller_metadata": {
                "placement_base_refinement": {
                    "object_id": str(geometry["object_id"]),
                    "residual_world_xy": residual_world.tolist(),
                    "speed_mps": speed,
                },
            },
        }
        decision = self._inject_g1_carry_pose(decision)
        command = self._robot_body.command_for_decision(decision)
        before = distance
        print(
            "[G1-PLACE] applying one live payload/base XY refinement: "
            f"residual_world_xy="
            f"{[round(float(v), 4) for v in residual_world]} "
            f"speed_mps={speed:.3f}.",
            flush=True,
        )
        result = self._robot_body.step_simulation(
            env=self._env,
            action_name=command.action_name,
            env_action=command.env_action,
            physics_step_budget=getattr(command, "physics_step_budget", None),
            step_human_bodies=self._step_human_bodies,
            pump_kit_updates=self._pump_kit_updates,
            pump_motion_window=self._pump_human_motion_window,
            global_tick=self._physics_steps,
            sample_body_dynamics=self._g1_body_dynamics,
            diagnose_physics_tick=self._on_physics_tick,
            abort_requested=self._abort_requested,
        )
        if result.observation:
            self._obs = result.observation
        self._physics_steps += result.physics_steps
        self._assert_robot_upright(
            action_name=command.action_name,
            env_action=command.env_action,
        )
        after = float(
            self._g1_pre_release_placement_errors(placement_report)["xy_error_m"]
        )
        print(
            "[G1-PLACE] payload/base XY refinement result: "
            f"before_m={before:.4f} after_m={after:.4f}.",
            flush=True,
        )
        return placement_report

    def _refine_g1_pre_release_placement(
        self,
        decision: Dict[str, Any],
        *,
        correction_gain: float = 0.5,
    ) -> Dict[str, Any]:
        """Apply one live-geometry residual correction before object release.

        The first IK pass solves from the pre-motion hand/object transform.  A
        compliant articulation can settle a few centimetres away from that
        kinematic prediction even after its joints converge.  Recompiling the
        same public placement primitive from the *live* held bbox closes that
        residual; it neither changes the episode target nor consumes another
        agent decision.
        """
        refinement_decision = dict(decision)
        source_metadata = decision.get("controller_metadata")
        refinement_metadata = (
            dict(source_metadata) if isinstance(source_metadata, dict) else {}
        )
        place_spec = refinement_metadata.get("g1_place_ik")
        if not isinstance(place_spec, dict):
            raise RuntimeError(
                "G1 placement refinement is missing the placement contract."
            )
        place_spec = dict(place_spec)
        gain = float(correction_gain)
        if not 0.0 < gain <= 1.0:
            raise ValueError("G1 placement correction gain must be in (0, 1].")
        # The nominal kinematic chain only proposes the next arm target.
        # Callers decrease this gain across bounded live-feedback attempts so
        # the payload settles near the support envelope.
        place_spec["correction_gain"] = gain
        refinement_metadata["g1_place_ik"] = place_spec
        refinement_decision["controller_metadata"] = refinement_metadata
        refined = self._resolve_g1_arm_ik_decision(refinement_decision)
        refined = self._include_dex3_controller_action(refined)
        metadata = refined.get("controller_metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError(
                "G1 placement refinement did not produce controller metadata."
            )
        arm_report = metadata.get("g1_arm_ik_result")
        placement_report = metadata.get("g1_place_ik_result")
        if not isinstance(arm_report, dict) or not isinstance(placement_report, dict):
            raise RuntimeError(
                "G1 placement refinement did not produce live IK geometry."
            )

        hand = str(arm_report.get("hand") or "")
        target_joints = np.asarray(
            arm_report.get("joint_positions"),
            dtype=np.float64,
        ).reshape(-1)
        joint_tolerance = float(metadata.get("g1_arm_ik_finish_tolerance_rad", 0.08))

        def _refinement_reached() -> bool:
            current = self._arm_joint_positions(hand)
            if current is None or target_joints.size != len(current):
                return False
            return (
                float(
                    np.max(
                        np.abs(np.asarray(current, dtype=np.float64) - target_joints)
                    )
                )
                <= joint_tolerance
            )

        command = self._robot_body.command_for_decision(refined)
        print(
            "[G1-PLACE] applying one closed-loop live-geometry refinement: "
            f"correction_gain={gain:.2f}.",
            flush=True,
        )
        result = self._robot_body.step_simulation(
            env=self._env,
            action_name=command.action_name,
            env_action=command.env_action,
            physics_step_budget=getattr(command, "physics_step_budget", None),
            step_human_bodies=self._step_human_bodies,
            pump_kit_updates=self._pump_kit_updates,
            pump_motion_window=self._pump_human_motion_window,
            global_tick=self._physics_steps,
            sample_body_dynamics=self._g1_body_dynamics,
            diagnose_physics_tick=self._on_physics_tick,
            finish_when=_refinement_reached,
            abort_requested=self._abort_requested,
        )
        if result.observation:
            self._obs = result.observation
        self._physics_steps += result.physics_steps
        self._assert_robot_upright(
            action_name=command.action_name,
            env_action=command.env_action,
        )
        if not _refinement_reached():
            # Joint convergence is an internal early-stop optimization.  The
            # public placement contract is the held object's live support
            # error, measured by the caller after every bounded correction.
            # A compliant arm can exhaust this short horizon while still
            # moving the object into tolerance; failing here would bypass the
            # authoritative geometry gate and create a false negative.
            print(
                "[G1-PLACE] refinement exhausted its joint-space horizon; "
                "deferring success/failure to live object geometry.",
                flush=True,
            )
        return placement_report

    def _grasp_grounding_target_key(self, object_id: str) -> str:
        """Stable per-object key for one evaluated grasp retry sequence.

        The key is the live pickable id, not the agent's free-text target
        phrase, so a rephrasing of the same object continues the same
        attempt sequence.
        """

        return str(object_id or "").strip()

    def _live_pickable_bounds(
        self,
    ) -> Dict[str, Tuple[Sequence[float], Sequence[float]]]:
        """Live AABBs of pickables that are not already held."""

        if self._g1_attachment is None:
            return {}
        live = self._g1_attachment.snapshot()["objects"]
        held = set(self._g1_attachment.attachment_reports)
        return {
            object_id: (state["bbox_min"], state["bbox_max"])
            for object_id, state in live.items()
            if object_id in self._g1_attachment.pickable_ids and object_id not in held
        }

    def _grasp_grounding_summary(self) -> Dict[str, Any]:
        """Rollout-level record of which mask produced each grasp attempt."""

        attempts = [
            dict(record) for record in getattr(self, "_grasp_grounding_attempts", ())
        ]
        used_memory = any(
            record.get("grounding_used") == GRASP_GROUNDING_MEMORY
            for record in attempts
        )
        success_used_memory = any(
            record.get("grounding_used") == GRASP_GROUNDING_MEMORY
            and record.get("outcome") == "success"
            for record in attempts
        )
        return {
            "used_memory_grounding": bool(used_memory),
            "success_used_memory_grounding": bool(success_used_memory),
            "attempts": attempts,
        }

    def _measured_grasp_target_displacement_m(self, object_id: str) -> float | None:
        """Return live horizontal displacement for one authorized grasp target."""

        if not object_id or self._g1_attachment is None:
            return None
        report = self._g1_attachment.grasp_target_displacement_report(
            object_id=str(object_id),
        )
        if not isinstance(report, dict):
            return None
        raw = report.get("horizontal_displacement_m", report.get("displacement_m"))
        if raw is None:
            return None
        displacement = float(raw)
        if not math.isfinite(displacement):
            raise RuntimeError(
                "grasp target displacement must be finite when measured."
            )
        return displacement

    def _with_grasp_grounding_displacement(
        self,
        info: Dict[str, Any],
        *,
        object_id: str,
        horizontal_displacement_m: float | None = None,
    ) -> Dict[str, Any]:
        """Attach agent-legible target displacement for grounding invalidation."""

        measured = horizontal_displacement_m
        if measured is None:
            measured = self._measured_grasp_target_displacement_m(object_id)
        if measured is None:
            return dict(info)
        payload = dict(info)
        payload["grounding_displacement_m"] = float(measured)
        return payload

    def _evaluated_grasp_measurement(
        self,
        decision: Mapping[str, Any],
    ) -> Dict[str, Any] | None:
        """Return the model's current-view grasp payload, or None if absent."""

        if str(decision.get("action") or "") != "grasp_object":
            return None
        metadata = decision.get("controller_metadata")
        if not isinstance(metadata, dict):
            return None
        if isinstance(metadata.get("g1_arm_ik"), dict) or bool(
            metadata.get("g1_arm_ik_hold", False)
        ):
            return None
        agent_grasp = metadata.get("agent_grasp")
        if not isinstance(agent_grasp, dict):
            return None
        if str(agent_grasp.get("grounding_mode") or "") == "oracle_gt_aabb":
            return None
        return agent_grasp

    def _prepare_evaluated_grasp_grounding(
        self,
        decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Record evaluator-private grounding provenance without rewriting poses.

        The evaluated agent must arrive with SAM3 grounding from the current
        synchronized camera frame. This hook only measures that grounding
        against the live pickable AABB for honest reporting.
        """

        self._active_grasp_grounding = None
        agent_grasp = self._evaluated_grasp_measurement(decision)
        if agent_grasp is None:
            return decision
        target_center = np.asarray(
            agent_grasp.get("target_center_world_m"),
            dtype=np.float64,
        ).reshape(-1)
        if target_center.size != 3 or not np.all(np.isfinite(target_center)):
            raise ValueError(
                "agent_grasp target_center_world_m must contain three finite "
                "current-view depth coordinates"
            )
        candidate_bounds = self._live_pickable_bounds()
        object_id = ""
        live_bbox = None
        if candidate_bounds:
            object_id, _ = nearest_grounded_pickable(
                grounded_world_xy=target_center[:2],
                candidate_bounds=candidate_bounds,
            )
            live_bbox = candidate_bounds[object_id]
        target_key = self._grasp_grounding_target_key(object_id)
        if not target_key:
            return decision
        by_target = getattr(self, "_grasp_grounding_by_target", None)
        if by_target is None:
            self._grasp_grounding_by_target = {}
            by_target = self._grasp_grounding_by_target
        state = by_target.setdefault(
            target_key,
            {
                "failed": False,
                "attempt_count": 0,
                "object_id": target_key,
            },
        )
        attempt_index = int(state["attempt_count"]) + 1
        grounding_used = str(agent_grasp.get("grounding_mode") or GRASP_GROUNDING_SAM3)
        if grounding_used not in {
            GRASP_GROUNDING_SAM3,
            GRASP_GROUNDING_MEMORY,
        }:
            raise ValueError(
                f"evaluated grasp grounding_mode must be sam3, "
                f"or memory; "
                f"got {grounding_used!r}."
            )
        discrepancy: Dict[str, Any] = {}
        if live_bbox is not None and grounding_used == GRASP_GROUNDING_SAM3:
            discrepancy = sam3_gt_aabb_discrepancy(
                sam3_center_world_m=target_center,
                bbox_min=live_bbox[0],
                bbox_max=live_bbox[1],
            )
        record = {
            "attempt_index": attempt_index,
            "grounding_used": grounding_used,
            "target": str(decision.get("target") or ""),
            "object_id": object_id,
            **discrepancy,
        }
        remembered_step = agent_grasp.get("remembered_action_step")
        if remembered_step is not None:
            record["remembered_action_step"] = int(remembered_step)
        print(
            "[G1-GRASP] grounding discrepancy: "
            f"target={record['target']!r} object={object_id} "
            f"attempt={attempt_index} grounding={grounding_used} "
            f"sam3_center={discrepancy.get('sam3_target_center_world_m')} "
            f"gt_aabb_center={discrepancy.get('gt_aabb_center_world_m')} "
            f"distance_m={float(discrepancy.get('center_distance_m') or 0.0):.4f} "
            f"inside={discrepancy.get('sam3_center_inside_gt_aabb')} "
            f"signed_gap_m={float(discrepancy.get('signed_surface_gap_m') or 0.0):.4f}",
            flush=True,
        )
        self._active_grasp_grounding = record
        return decision

    def _finish_evaluated_grasp_grounding(
        self,
        info: Dict[str, Any],
        *,
        succeeded: bool,
    ) -> Dict[str, Any]:
        """Attach the attempt record to execution info without changing gates."""

        record = getattr(self, "_active_grasp_grounding", None)
        if not isinstance(record, dict):
            return info
        finished = dict(record)
        finished["outcome"] = "success" if succeeded else "failure"
        if not succeeded:
            failure = info.get("grasp_failure") or {}
            if isinstance(failure, dict):
                finished["failure_code"] = str(failure.get("code") or "")
                evidence = failure.get("evidence") or {}
                if isinstance(evidence, dict):
                    detail = evidence.get("evaluator_detail")
                    if detail:
                        finished["evaluator_detail"] = str(detail)
                    bbox = evidence.get("projected_bbox_2d")
                    if bbox is not None:
                        finished["projected_bbox_2d"] = [int(value) for value in bbox]
        attempts = getattr(self, "_grasp_grounding_attempts", None)
        if attempts is None:
            self._grasp_grounding_attempts = []
            attempts = self._grasp_grounding_attempts
        attempts.append(finished)
        target_key = self._grasp_grounding_target_key(
            str(finished.get("object_id") or ""),
        )
        by_target = getattr(self, "_grasp_grounding_by_target", None)
        if by_target is None:
            self._grasp_grounding_by_target = {}
            by_target = self._grasp_grounding_by_target
        if target_key:
            state = by_target.setdefault(
                target_key,
                {
                    "failed": False,
                    "attempt_count": 0,
                    "object_id": target_key,
                },
            )
            state["attempt_count"] = int(finished["attempt_index"])
            state["failed"] = not succeeded
            state["object_id"] = target_key
        summary = self._grasp_grounding_summary()
        annotated = dict(info)
        annotated["grasp_grounding"] = dict(finished)
        annotated["grasp_grounding_summary"] = summary
        # grasp_contact_details is evaluator-private in the public step
        # trace; Agent feedback never receives it. That is the channel that
        # can carry GT AABB numbers without leaking them to the model.
        contact_details = list(annotated.get("grasp_contact_details") or [])
        contact_details.append(
            {
                "record_type": "grasp_grounding",
                **finished,
                "grasp_grounding_summary": summary,
            }
        )
        annotated["grasp_contact_details"] = contact_details
        failure = annotated.get("grasp_failure")
        if isinstance(failure, dict):
            evidence = dict(failure.get("evidence") or {})
            evidence["grasp_grounding"] = dict(finished)
            evidence["grasp_grounding_summary"] = summary
            annotated["grasp_failure"] = {**failure, "evidence": evidence}
        self._active_grasp_grounding = None
        return annotated

    def _compile_agent_grasp_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Turn one ``grasp_object`` call into an executed reach and a closure.

        The Agent's tool call contains SAM3-conditioned GraspGenX poses from
        the synchronized current RGB plus tool-private depth. The skill first
        plans only the right-hand motion to one closing pose. After the live
        hand reaches that pose, Isaac closes the fingers and requires physical
        contact. Only a retained grasp authorizes the independent lift and
        walking-posture plan. All internal stages consume physics time but
        remain one public Agent decision.

        The oracle reaches the same primitive from the other side: it authors its
        arm target from benchmark-private geometry and carries it in
        ``controller_metadata`` already. Such a decision passes through
        untouched. An evaluated grasp without current-view proposals is invalid;
        cached ``locate_object`` coordinates and canonical object poses are not
        substitute grasp geometry.

        Returns the public decision with one private closing-reach plan. The
        dedicated executor, rather than the ordinary attachment path, owns its
        contact, retention, and post-contact carry stages.

        Raises:
            GraspNotPossibleHere: the object cannot be taken from where the robot
                is standing. This is a measurement of the agent's stance, so the
                caller reports it as step feedback rather than failing the run.
        """
        if str(decision.get("action") or "") != "grasp_object":
            return decision
        metadata = decision.get("controller_metadata")
        carries_own_reach = isinstance(metadata, dict) and (
            isinstance(metadata.get("g1_arm_ik"), dict)
            or bool(metadata.get("g1_arm_ik_hold", False))
        )
        if carries_own_reach:
            if isinstance(metadata.get("agent_grasp"), dict):
                raise ValueError(
                    "An evaluated grasp cannot provide both current-view "
                    "GraspGenX evidence and a benchmark-authored arm target."
                )
            return decision
        agent_measurement = (
            metadata.get("agent_grasp") if isinstance(metadata, dict) else None
        )
        if not isinstance(agent_measurement, dict):
            raise ValueError(
                "evaluated grasp_object requires synchronized SAM3 + depth + "
                "GraspGenX evidence from the current tool call"
            )
        # The formal tool surface is deliberately task-agnostic, so a model
        # may legally call grasp_object in an episode with no authorized
        # pickable (or after every pickable is already held).  That is an
        # unsuccessful action choice, not broken simulator infrastructure.
        # Reuse the ordinary target-resolution refusal below so the model sees
        # one measured failure and can continue the same rollout.
        candidate_bounds = self._live_pickable_bounds()
        target_center_world = np.asarray(
            agent_measurement.get("target_center_world_m"),
            dtype=np.float64,
        ).reshape(-1)
        if target_center_world.size != 3 or not np.all(
            np.isfinite(target_center_world)
        ):
            raise ValueError(
                "agent_grasp target_center_world_m must contain three finite "
                "current-view depth coordinates"
            )
        target_center_camera = np.asarray(
            agent_measurement.get("target_center_camera_m"),
            dtype=np.float64,
        ).reshape(-1)
        if target_center_camera.size != 3 or not np.all(
            np.isfinite(target_center_camera)
        ):
            raise ValueError(
                "agent_grasp target_center_camera_m must contain three finite "
                "current-view depth coordinates"
            )
        object_id, match_gap = match_grounded_pickable(
            grounded_world_xy=target_center_world[:2],
            candidate_bounds=candidate_bounds,
        )
        payloads = agent_measurement.get("grasp_pose_candidates")
        if not isinstance(payloads, list) or not payloads:
            raise ValueError(
                "agent_grasp requires at least one serialized GraspGenX pose"
            )
        candidates = tuple(grasp_pose_from_payload(payload) for payload in payloads)
        world_T_camera = np.asarray(
            agent_measurement.get("world_T_camera"),
            dtype=np.float64,
        )
        if world_T_camera.shape != (4, 4) or not np.all(np.isfinite(world_T_camera)):
            raise ValueError(
                "agent_grasp world_T_camera must be a finite 4x4 transform"
            )
        robot_root_T_world = self._planning_root_for_camera_measurement(
            world_T_camera=world_T_camera,
            capture_id=agent_measurement.get("capture_id"),
        )
        self._begin_g1_closing_reach_deflection_capture(
            robot_root_T_world=robot_root_T_world,
            commanded_body_height_m=self._g1_body_height_command(),
        )
        requested_hand = str(decision.get("hand") or "").lower()
        if requested_hand != G1_MANIPULATION_HAND:
            raise ValueError(
                "evaluated grasp_object must use the certified right hand; "
                f"got {requested_hand!r}."
            )
        current_joint_positions = self._g1_curobo_joint_positions()
        current_body_height_command_m = self._g1_body_height_command()
        # Retention gates the executed grasp on the distance from the pose's
        # closing-volume centre to the live target. That distance is fixed once
        # the pose is chosen, so a pose failing it can only waste a reach.
        # Score-ranked confidence and cuRobo reachability say nothing about it,
        # so it is measured here and decides which poses may be planned at all.
        live_bbox = candidate_bounds[object_id]
        centre_gaps = [
            _planned_closing_centre_gap_m(
                camera_from_gripper=candidate.camera_from_gripper,
                world_T_camera=world_T_camera,
                object_bbox=live_bbox,
            )
            for candidate in candidates
        ]
        # G1_GRASP_CONTACT_GAP_M is the acceptance tolerance, not a selection
        # target. The executed arm carries a repeatable ~0.01 m offset from the
        # requested palm, so a pose whose centre merely grazes the tolerance
        # lands outside the object once executed. Prefer poses whose centre is
        # strictly inside the measured target, where the object's own half
        # width absorbs that offset, and fall back to the tolerance band only
        # when no interior pose can be reached.
        interior_exclude = {
            pose_index for pose_index, gap in enumerate(centre_gaps) if gap > 0.0
        }
        geometric_exclude = {
            pose_index
            for pose_index, gap in enumerate(centre_gaps)
            if gap > G1_GRASP_CONTACT_GAP_M
        }
        # Exclusions carry over from earlier attempts, but the proposal list
        # does not: each decision re-runs GraspGenX on a fresh depth frame, so
        # a remembered index would name an unrelated pose. Match on where each
        # pose closes in the world instead, which is stable while the target is.
        already_excluded = {
            pose_index
            for pose_index, candidate in enumerate(candidates)
            if self._grasp_pose_centre_is_excluded(
                _planned_closing_centre_world(
                    camera_from_gripper=candidate.camera_from_gripper,
                    world_T_camera=world_T_camera,
                )
            )
        }
        excluded = already_excluded | geometric_exclude
        interior_excluded = already_excluded | interior_exclude
        print(
            "[G1-GRASP] closing-centre gate: "
            f"{len(candidates) - len(interior_exclude)} poses inside the live "
            f"target, {len(candidates) - len(geometric_exclude)} within "
            f"{G1_GRASP_CONTACT_GAP_M:.4f} m of it, out of {len(candidates)} "
            f"(best={min(centre_gaps):.4f} m).",
            flush=True,
        )
        from tarsphere.infra.simulation.support_surface import live_support_surface_mesh

        support_id = self._g1_source_support_object_id()
        supports = [
            item for item in self._g1_attachment.objects if item.object_id == support_id
        ]
        if len(supports) != 1:
            raise RuntimeError(
                f"Grasp requires one live source support: {support_id!r}"
            )
        monitor = self._g1_attachment.grasp_contact_monitor
        contact = (
            monitor.support_contact(
                object_id=object_id,
                destination_object_id=support_id,
            )
            if monitor is not None
            else None
        )
        print(
            "[G1-GRASP] live support geometry: "
            + json.dumps(
                {
                    "target_object_id": object_id,
                    "support_object_id": support_id,
                    "target_bounds_xyz": np.asarray(live_bbox).tolist(),
                    "native_contact": contact.as_dict()
                    if contact is not None
                    else None,
                }
            ),
            flush=True,
        )
        support_vertices, support_faces = live_support_surface_mesh(
            self._g1_attachment.stage,
            supports[0].prim_path,
            target_bounds_xyz=live_bbox,
            target_support_separation_m=contact.separation_m
            if contact is not None
            else None,
        )
        support_obstacle = support_surface_obstacle_in_planning_root(
            support_object_id=support_id,
            vertices_world_m=support_vertices,
            faces=support_faces,
            robot_root_T_world=robot_root_T_world,
        )

        def _plan_closing_reach(
            exclude_pose_indices: tuple[int, ...],
            *,
            raise_on_failure: bool,
        ) -> G1GraspPlan | None:
            """Plan one closing reach with the given pose exclusions."""

            common = {
                "candidates": candidates,
                "robot_root_from_camera": (robot_root_T_world @ world_T_camera),
                "current_joint_positions": current_joint_positions,
                "current_body_height_command_m": (current_body_height_command_m),
                "hand": G1_MANIPULATION_HAND,
                "grasp_target_object_id": object_id,
                "source_support_object_id": self._g1_source_support_object_id(),
                "target_obstacle": target_object_obstacle_in_planning_root(
                    object_id=object_id,
                    object_bbox=live_bbox,
                    robot_root_T_world=robot_root_T_world,
                ),
                "support_obstacle": support_obstacle,
                "exclude_pose_indices": exclude_pose_indices,
            }
            try:
                plan = plan_g1_grasp(**common)
            except G1GraspPlanningInfeasible as error:
                if raise_on_failure:
                    raise
                # The interior pass plans most of the candidates and its
                # breakdown is the only record of why they failed; the
                # fallback pass that follows reports a different exclusion
                # set and cannot stand in for it.
                print(
                    f"[G1-GRASP] interior-centre pass found no reach: {error}",
                    flush=True,
                )
                return None
            pelvis = float(plan.reach_positions[-1][0])
            print(
                "[G1-GRASP] pelvis planning produced the plan: "
                f"pelvis_height_command_m={pelvis:.3f}",
                flush=True,
            )
            return plan

        interior_plan = None
        if len(interior_excluded) < len(candidates):
            interior_plan = _plan_closing_reach(
                tuple(sorted(interior_excluded)),
                raise_on_failure=False,
            )
            if interior_plan is not None:
                print(
                    "[G1-GRASP] selected a pose whose closing centre is "
                    "inside the live target.",
                    flush=True,
                )
        if interior_plan is None and len(excluded) >= len(candidates):
            raise GraspNotPossibleHere(
                GraspFailureCode.CONTACT_MISS,
                "No generated grasp pose closes on the target from this "
                "stance: every GraspGenX closing centre lies outside the "
                "measured object, so no reach can establish contact "
                "retention. Change the body stance or viewing distance "
                "before retrying.",
                evidence={
                    "object_id": object_id,
                    "hand": requested_hand,
                    "graspgenx_pose_count": len(candidates),
                    "best_closing_centre_gap_m": float(min(centre_gaps)),
                    "maximum_closing_centre_gap_m": float(G1_GRASP_CONTACT_GAP_M),
                },
            )
        try:
            plan = (
                interior_plan
                if interior_plan is not None
                else (
                    _plan_closing_reach(
                        tuple(sorted(excluded)),
                        raise_on_failure=True,
                    )
                )
            )
        except G1GraspPlanningInfeasible as exc:
            target_observation = grasp_target_observation(
                target_center_camera_m=target_center_camera,
                world_T_camera=world_T_camera,
            )
            right_hand_summary = summarize_grasp_reachability(
                exc.failures,
                reach_correction=exc.reach_correction,
            )
            reachability = {
                "reachable": False,
                "stage": "motion_planning",
                "target_observation": target_observation,
                "right_hand": right_hand_summary,
            }
            print(
                "[G1-GRASP] Closing-pose reach was infeasible from the current "
                f"body pose: {exc}",
                flush=True,
            )
            raise GraspNotPossibleHere(
                GraspFailureCode.REACH_ENVELOPE,
                format_reach_envelope_refusal_message(
                    target_observation=target_observation,
                    right_hand=right_hand_summary,
                ),
                evidence={
                    "hand": requested_hand,
                    "candidate_count": len(candidates),
                    "target_observation": target_observation,
                },
                reachability=reachability,
            ) from exc
        camera_rotation_world = world_T_camera[:3, :3]
        selected = candidates[plan.selected_pose_index]
        world_T_gripper = world_T_camera @ np.asarray(
            selected.camera_from_gripper,
            dtype=np.float64,
        )
        grasp_center_world = (
            camera_rotation_world @ selected.sweep_center_camera_m
            + world_T_camera[:3, 3]
        )
        closing_volume_center_world = (
            world_T_gripper[:3, :3]
            @ np.asarray(
                UNITREE_G1_SWEEP_VOLUME["offset_mid"],
                dtype=np.float64,
            )
            + world_T_gripper[:3, 3]
        )
        closing_palm_world = world_T_camera @ selected.palm_pose_camera(hand=plan.hand)
        grasp_evidence = {
            "object_id": object_id,
            "object_class": object_class_of(object_id),
            "target": str(decision.get("target") or ""),
            "match_gap_m": float(match_gap),
            "semantic_score": float(agent_measurement.get("semantic_score", 0.0)),
            "bbox_2d": list(agent_measurement.get("bbox_2d") or []),
            "graspgenx_pose_count": len(candidates),
            "selected_pose_index": int(plan.selected_pose_index),
            "selected_pose_confidence": float(plan.selected_confidence),
            "hand": plan.hand,
            "grasp_position_world": grasp_center_world.tolist(),
            "closing_volume_center_world": (closing_volume_center_world.tolist()),
            "closing_volume_axes_world": (world_T_gripper[:3, :3].T.tolist()),
            "closing_volume_extents_m": np.asarray(
                UNITREE_G1_SWEEP_VOLUME["extents_mid"],
                dtype=np.float64,
            ).tolist(),
            "closing_palm_position_world": (closing_palm_world[:3, 3].tolist()),
            "closing_palm_rotation_world": (closing_palm_world[:3, :3].tolist()),
            "closing_axis_world": (
                camera_rotation_world @ selected.closing_axis_camera
            ).tolist(),
        }
        self._complete_g1_closing_reach_deflection_capture(
            hand=plan.hand,
            terminal_waypoint=plan.reach_positions[-1],
            requested_palm_world_m=tuple(
                float(value) for value in closing_palm_world[:3, 3]
            ),
            robot_root_T_world=robot_root_T_world,
        )
        print(
            "[G1-GRASP] compiling one right-hand closing reach: "
            f"target={decision.get('target')!r} object={object_id} "
            f"match_gap_m={match_gap:.3f} "
            f"pose_index={plan.selected_pose_index}",
            flush=True,
        )

        resolved_decision = dict(decision)
        resolved_decision["hand"] = requested_hand
        resolved_decision["controller_metadata"] = {
            "g1_grasp_plan": grasp_plan_payload(plan),
            "agent_grasp": grasp_evidence,
            "grounding_mode": "current_rgb_with_private_depth",
        }
        return resolved_decision

    def _grasp_pose_centre_is_excluded(self, centre: Any) -> bool:
        """True when a pose closes where an earlier attempt lost retention."""

        candidate = np.asarray(centre, dtype=np.float64)
        return any(
            float(np.linalg.norm(candidate - excluded))
            <= G1_GRASP_POSE_EXCLUSION_RADIUS_M
            for excluded in getattr(self, "_g1_excluded_grasp_centres", ())
        )

    def _exclude_grasp_pose_centre(self, centre: np.ndarray) -> None:
        """Retire the closing position whose grasp failed to retain."""

        if not hasattr(self, "_g1_excluded_grasp_centres"):
            self._g1_excluded_grasp_centres = []
        if not self._grasp_pose_centre_is_excluded(centre):
            self._g1_excluded_grasp_centres.append(centre)

    def _eef_proprioception_summary(self) -> Dict[str, Any] | None:
        """EEF 7D pose for the model's proprioception channel.

        The certified right hand's palm tool frame as position [x, y, z]
        plus unit quaternion [w, qx, qy, qz], both body-relative — the
        standard end-effector state a `move_end_effector` decision reasons
        over. Holding state is attached from the measured grasp ledger.
        Any read failure returns None so the observation stays usable.
        """
        try:
            hand = G1_MANIPULATION_HAND
            frame = f"{hand}_hand_palm_link"
            affine = self._world_to_robot_frame_affine(frame)
            linear = np.asarray(affine["linear"], dtype=np.float64)
            offset = np.asarray(affine["offset"], dtype=np.float64)
            if linear.shape != (3, 3) or offset.shape != (3,):
                return None
            # Body-frame rotation vector in radians — the exact
            # representation and unit the move_end_effector delta uses for
            # [drx, dry, drz] (norm-bounded axis-angle).
            angle = math.acos(
                max(-1.0, min(1.0, (float(np.trace(linear)) - 1.0) / 2.0))
            )
            if angle < 1e-6:
                rotation_vector = np.zeros(3)
            elif abs(math.pi - angle) < 1e-3:
                axis = np.asarray(
                    [
                        linear[0, 0] + 1.0,
                        linear[1, 1] + 1.0,
                        linear[2, 2] + 1.0,
                    ]
                )
                norm = float(np.linalg.norm(axis))
                if norm < 1e-9:
                    axis = np.asarray([1.0, 0.0, 0.0])
                    norm = 1.0
                rotation_vector = axis / norm * angle
            else:
                axis = np.asarray(
                    [
                        linear[2, 1] - linear[1, 2],
                        linear[0, 2] - linear[2, 0],
                        linear[1, 0] - linear[0, 1],
                    ]
                )
                rotation_vector = axis / (2.0 * math.sin(angle)) * angle
            # Gripper closure on the same 0 open -> 1 closed scale the
            # move_end_effector delta's last element uses.
            fingers = self._dex3_joint_positions(hand)
            max_closure = getattr(self, "_dex3_max_joint_rad", None)
            if not fingers:
                gripper = None
            else:
                closure = float(np.mean(fingers))
                if max_closure:
                    gripper = round(min(1.0, max(0.0, closure / max_closure)), 3)
                else:
                    gripper = round(min(1.0, max(0.0, closure / 1.2)), 3)
            state = {
                "eef_position_body_m": [round(float(v), 3) for v in offset],
                "eef_rotation_body_rad": [round(float(v), 3) for v in rotation_vector],
            }
            if gripper is not None:
                state["eef_gripper_closure"] = gripper
            return state
        except Exception:
            return None

    def _measured_g1_palm_position_world(self, hand: str) -> np.ndarray:
        """Return the live cuRobo palm tool point in world coordinates."""

        return self._robot_frame_point_in_world(
            np.asarray(
                G1_CUROBO_PALM_TOOL_POINT_IN_PALM_LINK_M,
                dtype=np.float64,
            ),
            frame_name=f"{hand}_hand_palm_link",
        )

    def _g1_reach_palm_correction_support_floor_m(
        self,
        object_id: str,
    ) -> float:
        """Return the live target bottom height used for support-slab planning."""

        candidate_bounds = self._live_pickable_bounds()
        if object_id not in candidate_bounds:
            raise RuntimeError(
                "G1 reach palm correction requires live pickable bounds for "
                f"object={object_id!r}; got {sorted(candidate_bounds)}."
            )
        target_bottom_height_m = float(candidate_bounds[object_id][0][2])
        if not math.isfinite(target_bottom_height_m):
            raise RuntimeError(
                "G1 reach palm correction requires a finite target bottom "
                f"height; got {target_bottom_height_m!r} for object="
                f"{object_id!r}."
            )
        return target_bottom_height_m

    def _g1_reach_settle_until_converged(
        self,
        *,
        closing_waypoint: tuple[float, ...],
        hand: str,
        other_arm: Sequence[float],
        open_fingers: Sequence[float],
        object_id: str,
        phase_reports: list[Dict[str, Any]],
        initial_palm_world: np.ndarray,
    ) -> tuple[Dict[str, Any], np.ndarray, int]:
        """Hold one waypoint until palm motion and arm tracking settle."""

        reach_report = {
            "palm_after_world_m": initial_palm_world.tolist(),
            "maximum_terminal_arm_error_rad": 0.0,
            "phase": "reach_settle",
        }
        previous_palm = np.asarray(initial_palm_world, dtype=np.float64)
        reach_settle_ticks = 0
        while reach_settle_ticks < G1_REACH_SETTLE_MAX_PHYSICS_STEPS:
            remaining = self._remaining_physics_steps()
            if remaining is not None and remaining < G1_REACH_SETTLE_PHYSICS_STEPS:
                break
            settle_report = self._execute_g1_control_samples(
                phase="reach_settle",
                samples=((closing_waypoint,) * G1_REACH_SETTLE_PHYSICS_STEPS),
                hand=hand,
                other_arm_start=other_arm,
                other_arm_end=other_arm,
                finger_start=open_fingers,
                finger_end=open_fingers,
                contact_target_id=object_id,
            )
            phase_reports.append(settle_report)
            reach_report = settle_report
            reach_settle_ticks += G1_REACH_SETTLE_PHYSICS_STEPS
            reached_palm = np.asarray(
                settle_report["palm_after_world_m"],
                dtype=np.float64,
            )
            motion = float(np.linalg.norm(reached_palm - previous_palm))
            previous_palm = reached_palm
            if (
                motion <= G1_REACH_SETTLE_CONVERGENCE_M
                and float(settle_report["maximum_terminal_arm_error_rad"])
                <= G1_GRASP_REACH_TRACKING_TOLERANCE_RAD
            ):
                break
        return reach_report, previous_palm, reach_settle_ticks

    def _run_g1_reach_palm_correction(
        self,
        *,
        hand: str,
        closing_waypoint: tuple[float, ...],
        planned_palm_world: Sequence[float],
        object_id: str,
        other_arm: Sequence[float],
        open_fingers: Sequence[float],
        phase_reports: list[Dict[str, Any]],
        reach_report: Dict[str, Any],
        current_palm_world: np.ndarray,
    ) -> tuple[tuple[float, ...], Dict[str, Any], np.ndarray, Dict[str, Any]]:
        """Close the measured-versus-planned palm gap before finger closure."""

        from tarsphere.infra.manipulation.g1_arm_ik import (
            G1ArmSide,
            actuator_command_from_measured_ik_delta,
            clamp_palm_correction_step_torso,
            hand_position_in_torso,
            hand_transform_in_torso,
            shifted_hand_pose_ik_target_torso,
            solve_hand_pose,
        )

        side = cast(G1ArmSide, hand)
        goal_world = np.asarray(planned_palm_world, dtype=np.float64).reshape(-1)
        if goal_world.size != 3 or not np.all(np.isfinite(goal_world)):
            raise RuntimeError(
                "G1 reach palm correction requires a finite planned palm goal."
            )
        correction_record: Dict[str, Any] = {}
        palm_world = np.asarray(current_palm_world, dtype=np.float64)
        initial_residual_m = float(np.linalg.norm(palm_world - goal_world))
        correction_record["initial_residual_m"] = initial_residual_m
        passes_used = 0
        stop_reason = "max_passes"

        if initial_residual_m <= G1_REACH_CORRECTION_TOLERANCE_M:
            correction_record["final_residual_m"] = initial_residual_m
            correction_record["passes_used"] = passes_used
            correction_record["stop_reason"] = "initial_within_tolerance"
            return closing_waypoint, reach_report, palm_world, correction_record

        target_bottom_height_m = self._g1_reach_palm_correction_support_floor_m(
            object_id,
        )
        # Anchor orientation in the original planned model posture.  The live
        # torso transform below already carries every pelvis/waist outcome into
        # the position feedback; keeping body commands fixed avoids turning a
        # contact correction into a second balance policy.
        planned_palm_rotation_torso = hand_transform_in_torso(
            g1_manipulation_waypoint_arm_values(closing_waypoint, hand),
            hand=side,
        )[:3, :3]

        min_pass_ticks = (
            G1_REACH_CORRECTION_RAMP_PHYSICS_STEPS + G1_REACH_SETTLE_PHYSICS_STEPS
        )

        for pass_index in range(G1_REACH_CORRECTION_MAX_PASSES):
            remaining = self._remaining_physics_steps()
            if remaining is not None and remaining < min_pass_ticks:
                stop_reason = "budget"
                break

            palm_world = self._measured_g1_palm_position_world(hand)
            residual_m = float(np.linalg.norm(palm_world - goal_world))
            if residual_m <= G1_REACH_CORRECTION_TOLERANCE_M:
                stop_reason = "converged"
                break
            if residual_m > G1_REACH_CORRECTION_MAX_M:
                print(
                    "[G1-GRASP] reach palm correction skipped: residual "
                    f"{residual_m:.4f} m exceeds "
                    f"{G1_REACH_CORRECTION_MAX_M:.4f} m.",
                    flush=True,
                )
                stop_reason = "residual_exceeds_max"
                break

            residual_before_pass_m = residual_m
            q_meas = self._arm_joint_positions(hand)
            if q_meas is None:
                raise RuntimeError(
                    f"Cannot sample the live {hand} arm for reach palm correction."
                )
            palm_meas_torso = self._world_point_in_robot_frame(
                palm_world,
                frame_name="torso_link",
            )
            # Recompute both points in the same live torso snapshot every pass.
            # This is the full downstream effect of pelvis and waist feedback,
            # without issuing a new command to either balance-controlled axis.
            goal_torso = self._world_point_in_robot_frame(
                goal_world,
                frame_name="torso_link",
            )
            model_palm_torso = hand_position_in_torso(q_meas, hand=side)
            shifted_target_torso = shifted_hand_pose_ik_target_torso(
                measured_palm_torso=palm_meas_torso,
                model_palm_torso=model_palm_torso,
                planned_palm_torso=goal_torso,
            )
            ik_target_torso = clamp_palm_correction_step_torso(
                model_palm_torso=model_palm_torso,
                shifted_target_torso=shifted_target_torso,
                max_step_m=G1_REACH_CORRECTION_MAX_STEP_M,
            )
            ik_result = solve_hand_pose(
                ik_target_torso,
                hand=side,
                warm_start=q_meas,
                position_tolerance_m=G1_REACH_CORRECTION_TOLERANCE_M,
                solver_tolerance_m=G1_ARM_IK_SOLVER_TOLERANCE_M,
                reference_from_palm=G1_CUROBO_PALM_TOOL_POINT_IN_PALM_LINK_M,
                target_rotation_torso=planned_palm_rotation_torso,
                orientation_tolerance_rad=(
                    G1_REACH_CORRECTION_ORIENTATION_TOLERANCE_RAD
                ),
            )
            q_new = tuple(float(value) for value in ik_result.joint_positions)
            joint_deltas = np.abs(np.asarray(q_new) - np.asarray(q_meas))
            joint_delta = float(np.max(joint_deltas))
            if (
                not ik_result.converged
                or joint_delta > G1_REACH_CORRECTION_MAX_JOINT_DELTA_RAD
            ):
                print(
                    "[G1-GRASP] reach palm correction rejected on pass "
                    f"{pass_index + 1}: converged={ik_result.converged}, "
                    f"joint_delta_rad={joint_delta:.4f}, "
                    f"position_error_m={ik_result.position_error_m:.4f}.",
                    flush=True,
                )
                stop_reason = "ik_rejected"
                break

            predicted_palm_torso = hand_position_in_torso(q_new, hand=side)
            # The safety gate must predict the physical palm, not the same
            # biased analytic palm that the offset shift is compensating.
            predicted_physical_palm_torso = (
                predicted_palm_torso + palm_meas_torso - model_palm_torso
            )
            predicted_palm_world = self._robot_frame_point_in_world(
                predicted_physical_palm_torso,
                frame_name="torso_link",
            )
            if float(predicted_palm_world[2]) < target_bottom_height_m:
                print(
                    "[G1-GRASP] reach palm correction rejected: corrected "
                    f"palm height {predicted_palm_world[2]:.4f} m would fall "
                    f"below support floor {target_bottom_height_m:.4f} m.",
                    flush=True,
                )
                stop_reason = "support_height"
                break

            # ``q_new`` is the desired *measured* state because the solve was
            # warm-started from ``q_meas``.  The finite-gain upper-body
            # controller already needs a biased command to hold q_meas, so
            # replacing that command with q_new moves the live arm in the
            # wrong direction.  Preserve the existing command bias and apply
            # only the live-state IK delta.
            corrected_command = actuator_command_from_measured_ik_delta(
                previous_command=g1_manipulation_waypoint_arm_values(
                    closing_waypoint,
                    hand,
                ),
                measured_joints=q_meas,
                desired_measured_joints=q_new,
                hand=side,
            )
            corrected_waypoint = g1_manipulation_waypoint_with_arm_values(
                closing_waypoint,
                hand,
                corrected_command,
            )
            ramp_samples = g1_control_smooth_ramp_samples(
                closing_waypoint,
                corrected_waypoint,
                ticks=G1_REACH_CORRECTION_RAMP_PHYSICS_STEPS,
            )
            ramp_report = self._execute_g1_control_samples(
                phase="reach_correction",
                samples=ramp_samples,
                hand=hand,
                other_arm_start=other_arm,
                other_arm_end=other_arm,
                finger_start=open_fingers,
                finger_end=open_fingers,
                contact_target_id=object_id,
            )
            phase_reports.append(ramp_report)
            closing_waypoint = corrected_waypoint
            passes_used += 1
            reach_report, palm_world, settle_ticks = (
                self._g1_reach_settle_until_converged(
                    closing_waypoint=closing_waypoint,
                    hand=hand,
                    other_arm=other_arm,
                    open_fingers=open_fingers,
                    object_id=object_id,
                    phase_reports=phase_reports,
                    initial_palm_world=np.asarray(
                        ramp_report["palm_after_world_m"],
                        dtype=np.float64,
                    ),
                )
            )
            residual_m = float(np.linalg.norm(palm_world - goal_world))
            print(
                "[G1-GRASP] reach palm correction pass "
                f"{passes_used}: residual_m={residual_m:.4f}, "
                f"joint_delta_rad={joint_delta:.4f} after "
                f"{settle_ticks} settle ticks.",
                flush=True,
            )
            if residual_m <= G1_REACH_CORRECTION_TOLERANCE_M:
                stop_reason = "converged"
                break
            improvement_m = residual_before_pass_m - residual_m
            if improvement_m < G1_REACH_CORRECTION_MIN_PROGRESS_M:
                stop_reason = "no_progress"
                break
        else:
            if stop_reason == "max_passes":
                final_check = float(np.linalg.norm(palm_world - goal_world))
                if final_check <= G1_REACH_CORRECTION_TOLERANCE_M:
                    stop_reason = "converged"

        final_residual_m = float(np.linalg.norm(palm_world - goal_world))
        correction_record["final_residual_m"] = final_residual_m
        correction_record["passes_used"] = passes_used
        correction_record["stop_reason"] = stop_reason
        print(
            "[G1-GRASP] reach palm correction finished: "
            f"initial={initial_residual_m:.4f} m, final={final_residual_m:.4f} m, "
            f"passes={passes_used}, stop={stop_reason}.",
            flush=True,
        )
        return closing_waypoint, reach_report, palm_world, correction_record

    def _execute_agent_grasp_plan(
        self,
        decision: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        """Reach one closing pose, then independently form the carry posture."""

        from tarsphere.embodiment.dex3 import (
            DEX3_CLOSURE_PHYSICS_STEPS,
            DEX3_CONTACT_SETTLE_PHYSICS_STEPS,
            DEX3_GRASP_JOINT_POSITIONS,
            DEX3_OPEN_JOINT_POSITIONS,
        )

        metadata = decision.get("controller_metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError("A compiled G1 grasp requires controller metadata.")
        plan = grasp_plan_from_payload(metadata["g1_grasp_plan"])
        evidence = metadata.get("agent_grasp")
        if not isinstance(evidence, dict):
            raise RuntimeError("A compiled G1 grasp requires grounding evidence.")
        object_id = str(evidence.get("object_id") or "")
        hand = str(plan.hand)
        if not object_id or str(evidence.get("hand") or "") != hand:
            raise RuntimeError(
                "G1 grasp plan and grounding evidence disagree about the "
                f"target/hand: object={object_id!r}, plan_hand={hand!r}, "
                f"evidence_hand={evidence.get('hand')!r}."
            )

        other_hand = "left" if hand == "right" else "right"
        other_arm = self._arm_joint_positions(other_hand)
        if other_arm is None:
            raise RuntimeError(
                f"Cannot preserve the live {other_hand} arm during G1 grasp."
            )
        open_fingers = DEX3_OPEN_JOINT_POSITIONS[hand]
        closed_fingers = DEX3_GRASP_JOINT_POSITIONS[hand]
        phase_reports: list[Dict[str, Any]] = []
        reach_samples = resample_control_trajectory(
            plan.reach_positions,
            waypoint_interval_s=plan.trajectory_dt_s,
            control_frequency_hz=SIMULATION_PHYSICS_HZ,
        )
        total_ticks = len(reach_samples)
        total_ticks += int(G1_GRASP_SUPPORT_SETTLE_PHYSICS_STEPS)
        total_ticks += DEX3_CLOSURE_PHYSICS_STEPS + DEX3_CONTACT_SETTLE_PHYSICS_STEPS
        remaining = self._remaining_physics_steps()
        if remaining is not None and remaining < total_ticks:
            raise RuntimeError(
                "The G1 reach-and-close stage needs "
                f"{total_ticks} physics ticks but only {remaining} remain in "
                "the episode."
            )
        attached: Dict[str, Any] | None = None
        physical_retention: Dict[str, Any] | None = None
        retention_failure: RuntimeError | None = None
        held_target_for_approach = False
        retained_hold = False
        support_unstable_info: Dict[str, Any] | None = None
        # Resolved here rather than in the cleanup below, where raising on a
        # malformed value would mask whichever exception is already unwinding.
        failed_pose_centre = _validated_closing_centre(
            evidence.get("closing_volume_center_world")
        )
        restore_body_height_m = self._g1_body_height_command()
        manipulation_stance = self._g1_attachment.hold_manipulation_stance()
        self._begin_decision_render_window(total_ticks)
        activated_grasp_target = False
        try:
            self._g1_attachment.activate_grasp_target(object_id=object_id)
            # From here the pickable is dynamic. Anything that raises before
            # the hold is retained must still refreeze it, or a later
            # grasp_object is refused for a body this attempt left unfrozen.
            activated_grasp_target = True
            support_settle_report = self._settle_dynamic_grasp_target(
                object_id=object_id,
            )
            phase_reports.append(
                {
                    "phase": "support_settle",
                    **support_settle_report,
                }
            )
            if (
                support_settle_report["horizontal_displacement_m"]
                > G1_GRASP_SUPPORT_SETTLE_MAX_HORIZONTAL_DISPLACEMENT_M
                or support_settle_report["vertical_displacement_m"]
                < -G1_GRASP_SUPPORT_SETTLE_MAX_VERTICAL_DROP_M
            ):
                self._g1_attachment.restore_grasp_target(object_id=object_id)
                message = (
                    "The grasp target became unstable on its support once "
                    "dynamics began: "
                    f"object={object_id!r}, "
                    "horizontal_displacement_m="
                    f"{support_settle_report['horizontal_displacement_m']:.4f}, "
                    "vertical_displacement_m="
                    f"{support_settle_report['vertical_displacement_m']:.4f}, "
                    "maximum_horizontal_m="
                    f"{G1_GRASP_SUPPORT_SETTLE_MAX_HORIZONTAL_DISPLACEMENT_M:.3f}, "
                    "maximum_vertical_drop_m="
                    f"{G1_GRASP_SUPPORT_SETTLE_MAX_VERTICAL_DROP_M:.3f}."
                )
                print(f"[G1-GRASP] {message}", flush=True)
                self._interaction_feedback = message
                support_unstable_info = self._with_grasp_grounding_displacement(
                    {
                        "executed": False,
                        "feedback": message,
                        "grasp_failure": {
                            "code": GraspFailureCode.SUPPORT_UNSTABLE,
                            "message": message,
                            "evidence": {
                                "object_id": object_id,
                                "hand": hand,
                                "phase_reports": phase_reports,
                                "support_settle": support_settle_report,
                            },
                        },
                        "grasp_skill": self._g1_failed_grasp_skill(
                            object_id=object_id,
                            hand=hand,
                            phase_reports=phase_reports,
                        ),
                    },
                    object_id=object_id,
                    horizontal_displacement_m=float(
                        support_settle_report["horizontal_displacement_m"]
                    ),
                )
                # A return expression evaluates observe() before finally.
                # Failed-grasp cleanup then advances physics and invalidates
                # that frame, so the next GraspGenX tool can carry a stale
                # capture_id. Finish cleanup first and observe below.
                raise _G1SupportUnstable
            approach_hold = self._g1_attachment.hold_grasp_target(
                object_id=object_id,
            )
            held_target_for_approach = True
            phase_reports.append(
                {
                    "phase": "approach_hold",
                    **approach_hold,
                }
            )
            closing_waypoint = plan.reach_positions[-1]
            reach_report = self._execute_g1_control_samples(
                phase="grasp_reach",
                samples=reach_samples,
                hand=hand,
                other_arm_start=other_arm,
                other_arm_end=other_arm,
                finger_start=open_fingers,
                finger_end=open_fingers,
                contact_target_id=object_id,
            )
            phase_reports.append(reach_report)
            self._print_g1_closing_reach_deflection("reach_after_body_height")
            # Hold the terminal waypoint until the arm has actually arrived:
            # the reach ends on waypoint exhaustion, and the residual left at
            # that moment is several times the contact gap once projected to
            # the palm.
            reach_settle_ticks = 0
            previous_palm = np.asarray(
                reach_report["palm_after_world_m"],
                dtype=np.float64,
            )
            while reach_settle_ticks < G1_REACH_SETTLE_MAX_PHYSICS_STEPS:
                remaining = self._remaining_physics_steps()
                if remaining is not None and remaining < G1_REACH_SETTLE_PHYSICS_STEPS:
                    break
                settle_report = self._execute_g1_control_samples(
                    phase="reach_settle",
                    samples=((closing_waypoint,) * G1_REACH_SETTLE_PHYSICS_STEPS),
                    hand=hand,
                    other_arm_start=other_arm,
                    other_arm_end=other_arm,
                    finger_start=open_fingers,
                    finger_end=open_fingers,
                    contact_target_id=object_id,
                )
                phase_reports.append(settle_report)
                # The tracking gate below must see the arm this loop settled,
                # not the end-of-trajectory residual it exists to remove, so
                # adopt the report before the convergence break can skip it.
                reach_report = settle_report
                reach_settle_ticks += G1_REACH_SETTLE_PHYSICS_STEPS
                reached_palm = np.asarray(
                    settle_report["palm_after_world_m"],
                    dtype=np.float64,
                )
                motion = float(np.linalg.norm(reached_palm - previous_palm))
                previous_palm = reached_palm
                # Wrist yaw spins the hand about the palm origin, so a settled
                # palm position says nothing about whether that joint arrived.
                # Leaving on palm motion alone hands the joint-level tracking
                # gate below an arm that was never given time to converge.
                if (
                    motion <= G1_REACH_SETTLE_CONVERGENCE_M
                    and float(settle_report["maximum_terminal_arm_error_rad"])
                    <= G1_GRASP_REACH_TRACKING_TOLERANCE_RAD
                ):
                    break
            requested_palm = evidence.get("closing_palm_position_world")
            if requested_palm is not None:
                palm_error = float(
                    np.linalg.norm(
                        previous_palm - np.asarray(requested_palm, dtype=np.float64)
                    )
                )
                print(
                    "[G1-GRASP] closing palm requested="
                    f"{[round(float(v), 4) for v in requested_palm]} "
                    f"reached={[round(float(v), 4) for v in previous_palm]} "
                    f"error_m={palm_error:.4f} after "
                    f"{reach_settle_ticks} settle ticks.",
                    flush=True,
                )
                closing_waypoint, reach_report, previous_palm, reach_correction = (
                    self._run_g1_reach_palm_correction(
                        hand=hand,
                        closing_waypoint=closing_waypoint,
                        planned_palm_world=requested_palm,
                        object_id=object_id,
                        other_arm=other_arm,
                        open_fingers=open_fingers,
                        phase_reports=phase_reports,
                        reach_report=reach_report,
                        current_palm_world=previous_palm,
                    )
                )
                evidence["reach_palm_correction"] = reach_correction
            self._print_g1_closing_reach_deflection("reach_settle")
            # A missed reach can reflect contact, execution tracking or a
            # planner/runtime geometry mismatch; this gate alone cannot name
            # the cause. Preserve the miss as diagnostic evidence, then let
            # measured Dex3 contact, force closure and retention decide whether
            # this grasp is physically valid. The carry/transport tracking
            # checks below stay terminal because the object is already retained
            # there and later evidence depends on the executed trajectory.
            phase_reports.append(
                {
                    "phase": "closing_alignment",
                    "palm_residual_m": closing_palm_residual_m(
                        requested_palm,
                        previous_palm,
                    ),
                    "acceptance_basis": (
                        "subsequent_measured_contact_force_closure_and_retention"
                    ),
                }
            )
            reach_tracking_warning = self._g1_reach_tracking_warning(
                reach_report,
            )
            if reach_tracking_warning is not None:
                phase_reports.append(reach_tracking_warning)
            holding_fingers: tuple[float, ...] | None = None
            if retention_failure is None:
                closure_report = self._execute_g1_control_samples(
                    phase="closure",
                    samples=((closing_waypoint,) * DEX3_CLOSURE_PHYSICS_STEPS),
                    hand=hand,
                    other_arm_start=other_arm,
                    other_arm_end=other_arm,
                    finger_start=open_fingers,
                    finger_end=closed_fingers,
                    contact_target_id=object_id,
                )
                phase_reports.append(closure_report)
                holding_fingers = tuple(
                    float(value) for value in closure_report["finger_command_after_rad"]
                )
                wrap = self._g1_attachment.closure_wrap_report(
                    hand=hand,
                    object_id=object_id,
                    closing_axis_world=list(
                        (evidence.get("closing_volume_axes_world") or [[]])[0]
                        if evidence.get("closing_volume_axes_world")
                        else []
                    ),
                )
                phase_reports.append(
                    {
                        "phase": "closure_confirmation",
                        **wrap,
                    }
                )
                if not wrap.get("wrapped"):
                    retention_failure = RuntimeError(
                        "Dex3 closure did not wrap the kinematic target: "
                        f"object={object_id!r}, "
                        "contact_established="
                        f"{wrap.get('contact_established')}, "
                        "force_closure_established="
                        f"{wrap.get('force_closure_established')}, "
                        f"contact_links={wrap.get('contact_links')!r}, "
                        "first_role_contact="
                        f"{wrap.get('first_role_contact')}, "
                        "second_role_contact="
                        f"{wrap.get('second_role_contact')}, "
                        "best_contact_span_m="
                        f"{wrap.get('best_contact_span_m')}, "
                        "best_normal_dot="
                        f"{wrap.get('best_normal_dot')}, "
                        "best_minimum_closing_axis_normal_projection="
                        f"{wrap.get('best_minimum_closing_axis_normal_projection')}, "
                        "best_first_normal_projection="
                        f"{wrap.get('best_first_normal_projection')}, "
                        "best_second_normal_projection="
                        f"{wrap.get('best_second_normal_projection')}, "
                        "best_first_point_projection_m="
                        f"{wrap.get('best_first_point_projection_m')}, "
                        "best_second_point_projection_m="
                        f"{wrap.get('best_second_point_projection_m')}, "
                        "best_failed_gate_names="
                        f"{wrap.get('best_failed_gate_names')!r}, "
                        "best_span_deficit_m="
                        f"{wrap.get('best_span_deficit_m')}, "
                        "best_projection_deficit="
                        f"{wrap.get('best_projection_deficit')}, "
                        "best_normal_dot_deficit="
                        f"{wrap.get('best_normal_dot_deficit')}, "
                        "best_first_normal_impulse_ns="
                        f"{wrap.get('best_first_normal_impulse_ns')}, "
                        "best_second_normal_impulse_ns="
                        f"{wrap.get('best_second_normal_impulse_ns')}, "
                        "target_remained_kinematic=true, "
                        "displaced_during_approach=false."
                    )
            if retention_failure is None:
                self._g1_attachment.activate_grasp_target(object_id=object_id)
                release_state = self._g1_attachment.grasp_target_displacement_report(
                    object_id=object_id,
                )
                phase_reports.append(
                    {
                        "phase": "dynamic_release",
                        "object_id": object_id,
                        "position": [
                            float(value)
                            for value in release_state["current_position_world"]
                        ],
                        "kinematic_enabled": bool(release_state["kinematic_enabled"]),
                    }
                )
                phase_reports.extend(
                    self._settle_g1_dynamic_grasp(
                        object_id=object_id,
                        hand=hand,
                        closing_waypoint=closing_waypoint,
                        other_arm=other_arm,
                        holding_fingers=holding_fingers,
                        previous_fingers=closure_report[
                            "finger_joint_positions_after_rad"
                        ],
                        closing_axis_world=list(
                            evidence["closing_volume_axes_world"][0]
                        ),
                    )
                )
                self._print_g1_closing_reach_deflection("dex3_closure_settle")

            if retention_failure is None:
                try:
                    attached = self._begin_agent_grasp_retention(
                        hand=hand,
                        object_id=object_id,
                        closing_volume_center_world=list(
                            evidence.get("closing_volume_center_world") or []
                        ),
                        closing_volume_axes_world=list(
                            evidence.get("closing_volume_axes_world") or []
                        ),
                        closing_volume_extents_m=list(
                            evidence.get("closing_volume_extents_m") or []
                        ),
                    )
                except RuntimeError as exc:
                    retention_failure = exc
            if retention_failure is None:
                self._g1_carried_hand = hand
                self._g1_carry_finger_positions = holding_fingers
                # The second cuRobo request begins only after live contact and
                # closure are established. It owns lift off the source support.
                # The measured lifted pose seeds a gradual, finger-closed
                # restoration to the official locomotion upper body below.
                measured_palm_robot = (
                    self._g1_planning_root_from_world()
                    @ self._robot_frame_pose_in_world(f"{hand}_hand_palm_link")
                )
                carry_plan = G1GraspPlannerClient().plan_carry(
                    hand=hand,
                    current_joint_positions=self._g1_curobo_joint_positions(),
                    current_body_height_command_m=(self._g1_body_height_command()),
                    closing_palm_pose_robot=measured_palm_robot,
                )
                retraction_samples = resample_control_trajectory(
                    carry_plan.positions[: carry_plan.retraction_step_count],
                    waypoint_interval_s=carry_plan.trajectory_dt_s,
                    control_frequency_hz=SIMULATION_PHYSICS_HZ,
                )
                carry_retract_report = self._execute_g1_control_samples(
                    phase="carry_retraction",
                    samples=retraction_samples,
                    hand=hand,
                    other_arm_start=other_arm,
                    other_arm_end=other_arm,
                    finger_start=holding_fingers,
                    finger_end=holding_fingers,
                    prepare_final_frame=True,
                )
                phase_reports.append(carry_retract_report)
                self._require_g1_arm_tracking(carry_retract_report)
                retracted_waypoint = (
                    float(carry_retract_report["pelvis_height_command_m"]),
                    float(
                        carry_retract_report.get(
                            "measured_waist_yaw_rad",
                            carry_retract_report["waist_yaw_rad"],
                        )
                    ),
                    float(
                        carry_retract_report.get(
                            "measured_waist_pitch_rad",
                            carry_retract_report["waist_pitch_rad"],
                        )
                    ),
                    *(
                        float(value)
                        for value in carry_retract_report["measured_arm_after_rad"]
                    ),
                )
                try:
                    physical_retention = (
                        self._g1_attachment.require_contact_physical_grasp(
                            object_id=object_id,
                        )
                    )
                except RuntimeError as exc:
                    retention_failure = exc
            if retention_failure is None:
                measured_other = self._arm_joint_positions(other_hand)
                if measured_other is None:
                    raise RuntimeError(
                        f"Cannot restore locomotion from the live {other_hand} "
                        "arm after G1 retraction."
                    )
                try:
                    walking_waypoint, restore_reports = (
                        self._restore_g1_locomotion_upper_body_after_grasp(
                            start_waypoint=retracted_waypoint,
                            hand=hand,
                            object_id=object_id,
                            other_arm_start=measured_other,
                            holding_fingers=holding_fingers,
                            require_arm_tracking=not True,
                        )
                    )
                except G1CarryPostureTrackingError as exc:
                    retention_failure = GraspNotPossibleHere(
                        GraspFailureCode.REACH_ALIGNMENT_FAILED,
                        "G1 carry transition could not restore its locomotion "
                        f"posture: {exc}",
                        evidence={"phase": "carry_transition"},
                    )
                else:
                    locomotion_other_arm = (0.0,) * 7
                    phase_reports.extend(restore_reports)
                    try:
                        physical_retention = (
                            self._g1_attachment.require_contact_physical_grasp(
                                object_id=object_id,
                            )
                        )
                    except RuntimeError as exc:
                        retention_failure = exc
            if retention_failure is None:
                # Navigation starts only after the pelvis, waist and both arms
                # have returned to the ordinary standing locomotion posture.
                measured_palm = self._robot_frame_point_in_world(
                    np.zeros(3, dtype=np.float64),
                    frame_name=f"{hand}_hand_palm_link",
                )
                upper_body_command = self._g1_upper_body_command(
                    walking_waypoint,
                    hand=hand,
                    other_arm=locomotion_other_arm,
                )
                self._last_g1_arm_ik = {
                    "hand": hand,
                    "target_position_world": measured_palm.tolist(),
                    "position_error_m": 0.0,
                    "joint_positions": list(
                        g1_manipulation_waypoint_arm_values(
                            walking_waypoint,
                            hand,
                        )
                    ),
                    "upper_body_command": upper_body_command,
                }
                self._g1_body_posture = (
                    "standing"
                    if walking_waypoint[0] >= G1_STANDING_HEIGHT_COMMAND_M - 0.02
                    else "crouched"
                )
                self._g1_body_height_command_m = float(walking_waypoint[0])
                self._publish_g1_body_height_command()
                raw_transport_report = self._execute_g1_control_samples(
                    phase="transport_settle",
                    samples=(walking_waypoint,) * DEX3_CONTACT_SETTLE_PHYSICS_STEPS,
                    hand=hand,
                    other_arm_start=locomotion_other_arm,
                    other_arm_end=locomotion_other_arm,
                    finger_start=holding_fingers,
                    finger_end=holding_fingers,
                )
                transport_report = (
                    IsaacAutonomousEnv._g1_arm_tracking_against_measured_target(
                        raw_transport_report,
                        target_arm=(0.0,) * 7,
                        actuator_command=(
                            g1_manipulation_waypoint_arm_values(
                                walking_waypoint,
                                hand,
                            )
                        ),
                    )
                )
                phase_reports.append(transport_report)
                pass
                final_palm = self._robot_frame_point_in_world(
                    np.zeros(3, dtype=np.float64),
                    frame_name=f"{hand}_hand_palm_link",
                )
                self._last_g1_arm_ik.update(
                    {
                        "target_position_world": final_palm.tolist(),
                        "position_error_m": 0.0,
                    }
                )
                try:
                    physical_retention = (
                        self._g1_attachment.require_contact_physical_grasp(
                            object_id=object_id,
                        )
                    )
                except RuntimeError as exc:
                    retention_failure = exc
            if retention_failure is None:
                arms_retracted, standing = self._require_g1_neutral_transport_posture(
                    walking_waypoint,
                    phase_reports[-1],
                )
                retained_hold = True
        except _G1SupportUnstable:
            pass
        except (
            G1CarryPlanningInfeasible,
            G1CarryPostureTrackingError,
        ) as exc:
            # Closing contact is not task completion: the selected grasp must
            # still admit a support-clearance lift and a measured transport
            # posture. Those two failures depend on the grasp/stance chosen by
            # the Agent and are therefore an ordinary failed grasp attempt,
            # not broken evaluation infrastructure. Worker crashes, missing
            # joints, malformed reports, and simulator failures deliberately
            # remain uncaught and invalidate the run.
            retention_failure = GraspNotPossibleHere(
                GraspFailureCode.REACH_ALIGNMENT_FAILED,
                "The grasp made contact but could not reach a safe transport "
                f"posture from this grasp and stance: {exc}",
                evidence={
                    "phase": "post_contact_carry",
                    "failure_type": type(exc).__name__,
                    **(
                        {"planner_outcome": str(exc.outcome)}
                        if isinstance(exc, G1CarryPlanningInfeasible)
                        else {}
                    ),
                },
            )
        finally:
            try:
                if not retained_hold and (
                    activated_grasp_target
                    or held_target_for_approach
                    or retention_failure is not None
                ):
                    if retention_failure is not None:
                        self._exclude_grasp_pose_centre(failed_pose_centre)
                    self._restore_failed_g1_grasp_approach(
                        hand=hand,
                        object_id=object_id,
                        other_arm=other_arm,
                        restore_body_height_m=restore_body_height_m,
                    )
            finally:
                self._end_decision_render_window()
                self._g1_attachment.release_manipulation_stance()

        if support_unstable_info is not None:
            support_unstable_info["physics_steps"] = self._physics_steps
            support_unstable_info["robot_stability"] = self._robot_stability()
            return self.observe(), False, support_unstable_info

        if retention_failure is not None:
            if self._g1_carried_hand == hand:
                self._g1_carried_hand = None
                self._g1_carry_finger_positions = None
            if isinstance(retention_failure, GraspNotPossibleHere):
                failure_code = retention_failure.code
            else:
                failure_code = (
                    GraspFailureCode.RETENTION_LOST
                    if attached is not None
                    else GraspFailureCode.CONTACT_MISS
                )
            failure_stage = (
                "lost retention after measured contact"
                if attached is not None
                else "did not establish measured contact retention"
            )
            message = f"The grasp {failure_stage}: {retention_failure}"
            print(f"[G1-GRASP] {message}", flush=True)
            self._interaction_feedback = message
            return (
                self.observe(),
                False,
                self._with_grasp_grounding_displacement(
                    {
                        "executed": False,
                        "feedback": message,
                        "grasp_failure": {
                            "code": failure_code,
                            "message": message,
                            "evidence": {
                                "object_id": object_id,
                                "hand": hand,
                                "phase_reports": phase_reports,
                                "retention_report": attached,
                                "failure_details": (
                                    retention_failure.evidence
                                    if isinstance(
                                        retention_failure, GraspNotPossibleHere
                                    )
                                    else {}
                                ),
                                "final_physical_hold": (
                                    self._g1_attachment.physical_hold_state(
                                        object_id=object_id
                                    )
                                    if object_id in self._g1_attachment.physical_holds
                                    else None
                                ),
                            },
                        },
                        "grasp_skill": self._g1_failed_grasp_skill(
                            object_id=object_id,
                            hand=hand,
                            phase_reports=phase_reports,
                        ),
                        "grasp_contact_details": list(
                            (attached or {}).get("physx_contacts") or []
                        ),
                        "physics_steps": self._physics_steps,
                        "robot_stability": self._robot_stability(),
                    },
                    object_id=object_id,
                ),
            )
        if attached is None:
            raise RuntimeError(
                "G1 grasp retention completed without an attachment report."
            )
        self._g1_excluded_grasp_centres = []

        # The command that first produced force closure is the unique carry
        # hand pose. Replacing it with a generic fully-closed preset would
        # preserve a FixedJoint while visibly driving fingers through the item.
        self._g1_carry_finger_positions = holding_fingers

        final_waypoint = walking_waypoint
        final_report = phase_reports[-1]
        completion = {
            "object_id": object_id,
            "hand": hand,
            "planner": "curobo_v2",
            "planning_coordinates": list(plan.coordinate_names),
            "manipulation_stance": manipulation_stance,
            "phases": phase_reports,
            # ``phases`` is agent-forbidden; ``phase_reports`` survives into
            # heartbeat and persisted trace sanitization for diagnosis.
            "phase_reports": list(phase_reports),
            "retention": attached.get("retention"),
            "physical_retention": physical_retention,
            "arms_retracted": arms_retracted,
            "standing": standing,
            "carry_arm_target_rad": [0.0] * 7,
            "carry_arm_command_rad": list(
                g1_manipulation_waypoint_arm_values(final_waypoint, hand)
            ),
            "carry_arm_measured_rad": list(final_report["measured_arm_after_rad"]),
            "idle_arm_measured_rad": list(final_report["measured_other_arm_after_rad"]),
            "maximum_carry_arm_error_rad": float(
                final_report["maximum_terminal_arm_error_rad"]
            ),
            "maximum_idle_arm_error_rad": float(
                final_report["maximum_terminal_other_arm_error_rad"]
            ),
            # The hard-failure scorer, the grasp-skill certification and the
            # agent prompt all read this as evidence the object left its
            # support. A FixedJoint retains it by construction; a friction hold
            # only does so if PhysX measured the support clearance, so under
            # physical attachment this reports the measurement, never a claim.
            "transport_ready": (
                bool((physical_retention or {}).get("transport_ready"))
                and arms_retracted
                and standing
            ),
        }
        print(
            "[G1-GRASP] cuRobo skill complete: "
            f"object={object_id} hand={hand} "
            f"planned_carry_height_m={final_waypoint[0]:.3f} "
            f"transport_height_m={G1_STANDING_HEIGHT_COMMAND_M:.3f} "
            f"waist_yaw_rad={final_waypoint[1]:.3f} "
            f"arms_retracted={str(arms_retracted).lower()} "
            f"standing={str(standing).lower()} "
            f"transport_ready={str(completion['transport_ready']).lower()} "
            "support_clearance_m="
            f"{float((physical_retention or {}).get('support_clearance_m', float('nan'))):.4f}",
            flush=True,
        )
        return (
            self.observe(),
            False,
            {
                "executed": True,
                "feedback": "",
                # Navigation/body contact is a separate public feedback channel.
                # Dex3 contact evidence remains private manipulation evidence so
                # memory cannot mistake finger contact for locomotion contact.
                "physical_contact": False,
                "physical_contact_details": [],
                "grasp_contact_details": list(attached.get("physx_contacts") or []),
                "balance_control": {"intervened": False},
                "post_motion_settle_ticks": 0,
                "grasp_failure": None,
                "grasp_skill": completion,
                "physics_steps": self._physics_steps,
                "robot_stability": self._robot_stability(),
            },
        )

    def _settle_g1_dynamic_grasp(
        self,
        *,
        object_id,
        hand,
        closing_waypoint,
        other_arm,
        holding_fingers,
        previous_fingers,
        closing_axis_world,
    ):
        """Hold the closed hand until joints AND native dynamic forces settle.

        Kinematic wrap is not dynamic force closure. This reads the same
        current-impulse predicate as retention without creating a hold early.
        The existing settle budget and final physical-retention gate remain.
        """
        from tarsphere.embodiment.dex3 import (
            DEX3_CLOSURE_SETTLE_CONVERGENCE_RAD,
            DEX3_CLOSURE_SETTLE_MAX_PHYSICS_STEPS,
            DEX3_CONTACT_SETTLE_PHYSICS_STEPS,
        )

        reports, settled_ticks = [], 0
        previous = np.asarray(previous_fingers, dtype=float)
        while settled_ticks < DEX3_CLOSURE_SETTLE_MAX_PHYSICS_STEPS:
            remaining = self._remaining_physics_steps()
            if remaining is not None and remaining < DEX3_CONTACT_SETTLE_PHYSICS_STEPS:
                break
            report = self._execute_g1_control_samples(
                phase="contact_settle",
                samples=(closing_waypoint,) * DEX3_CONTACT_SETTLE_PHYSICS_STEPS,
                hand=hand,
                other_arm_start=other_arm,
                other_arm_end=other_arm,
                finger_start=holding_fingers,
                finger_end=holding_fingers,
                contact_target_id=object_id,
            )
            current = np.asarray(
                report["finger_joint_positions_after_rad"], dtype=float
            )
            if not np.isfinite(current).all() or current.shape != previous.shape:
                raise RuntimeError(
                    "Dynamic grasp settling requires finite measured finger joints."
                )
            motion = float(np.max(np.abs(current - previous)))
            previous = current
            force_pair = self._g1_attachment.measured_force_closure_pair(
                hand=hand,
                object_id=object_id,
                closing_axis_world=closing_axis_world,
            )
            settled_ticks += DEX3_CONTACT_SETTLE_PHYSICS_STEPS
            settled = bool(
                motion <= DEX3_CLOSURE_SETTLE_CONVERGENCE_RAD and force_pair is not None
            )
            report.update(
                {
                    "dynamic_force_closure_established": force_pair is not None,
                    "closure_joint_motion_rad": motion,
                    "dynamic_closure_settled": settled,
                    "dynamic_closure_settle_ticks": settled_ticks,
                }
            )
            reports.append(report)
            if settled:
                break
        print(
            f"[G1-GRASP] dynamic closure: {settled_ticks} settle ticks; "
            f"joint-and-force convergence={bool(reports and reports[-1]['dynamic_closure_settled'])}.",
            flush=True,
        )
        return reports

    def _g1_source_support_object_id(self) -> str:
        """Resolve the authored support from the initialized physical task."""
        if self._g1_attachment is None:
            raise RuntimeError(
                "G1 source support requires initialized manipulation bodies"
            )
        supports = {
            item.source_support_object_id
            for item in self._g1_attachment.objects
            if item.pickable and item.source_support_object_id
        }
        if len(supports) != 1:
            raise RuntimeError("G1 grasp requires one authored source support")
        return next(iter(supports))

    def _restore_g1_locomotion_upper_body_after_grasp(
        self,
        *,
        start_waypoint: Sequence[float],
        hand: str,
        other_arm_start: Sequence[float],
        holding_fingers: Sequence[float],
        object_id: str = "",
        require_arm_tracking: bool = True,
    ) -> tuple[tuple[float, ...], list[Dict[str, Any]]]:
        """Return the G1 upper body to its ordinary standing carry posture.

        Close stances pick deep-crouch plus backward waist lean. Zeroing a
        negative waist pitch rotates the torso forward and sweeps the shoulder
        downward even when every arm joint is frozen. Standing first raises the
        whole kinematic chain clear of the support. Waist pitch is then restored
        with both arms frozen, followed by the measured gradual interpolation
        of waist yaw and both arms to locomotion zero. Dex3 closure and the
        existing attachment remain active throughout.
        """

        _ = object_id

        ticks = max(1, int(self._robot_body.carry_transition_physics_steps))
        stand_ticks = max(1, int(self._robot_body.body_height_transition_physics_steps))
        other_start = tuple(float(value) for value in other_arm_start)
        if len(other_start) != 7 or not all(
            math.isfinite(value) for value in other_start
        ):
            raise RuntimeError(
                "G1 locomotion interpolation requires a finite opposite-arm start."
            )
        start = tuple(float(value) for value in start_waypoint)
        reports: list[Dict[str, Any]] = []
        idle_arm = other_start
        # Standing is a pure vertical translation: it buys support clearance
        # before waist un-pitch or arm descent. Backward waist lean at close
        # stances would otherwise sweep the held object down against the support
        # while still crouched; walking zeros would do the same to the arm.
        if not math.isclose(
            float(start[0]),
            G1_STANDING_HEIGHT_COMMAND_M,
            abs_tol=0.02,
        ):
            standing_start = (G1_STANDING_HEIGHT_COMMAND_M, *start[1:])
            stand_report = self._execute_g1_control_samples(
                phase="carry_stand_up",
                samples=g1_control_smooth_ramp_samples(
                    start,
                    standing_start,
                    ticks=stand_ticks,
                ),
                hand=hand,
                other_arm_start=idle_arm,
                other_arm_end=idle_arm,
                finger_start=holding_fingers,
                finger_end=holding_fingers,
                prepare_final_frame=True,
            )
            reports.append(stand_report)
            print(
                "[G1-CONTROL] "
                "phase=carry_stand_up "
                f"maximum_terminal_arm_error_rad="
                f"{float(stand_report['maximum_terminal_arm_error_rad']):.4f} "
                "(arm held at the lifted pose before neutral restore)",
                flush=True,
            )
            if require_arm_tracking:
                self._require_g1_arm_tracking(stand_report)
            start = standing_start
        # Restore either sign of waist pitch only after standing. Grasp
        # retraction commonly turns a forward reach into about -0.13 rad of
        # backward lean. The gradual restore
        # freezes both arms, pelvis height and finger closure, and the caller
        # immediately rechecks live friction retention before locomotion.
        start = self._restore_g1_waist_pitch_after_grasp(
            start_waypoint=start,
            hand=hand,
            other_arm=idle_arm,
            holding_fingers=holding_fingers,
            require_arm_tracking=require_arm_tracking,
            ticks=ticks,
            reports=reports,
        )

        from tarsphere.infra.manipulation.g1_arm_ik import (
            actuator_command_from_measured_ik_delta,
        )

        segment_start = start
        loaded_arm_start = tuple(
            float(value) for value in g1_manipulation_waypoint_arm_values(start, hand)
        )
        measured_arm = loaded_arm_start
        actuator_arm = loaded_arm_start
        start_yaw = g1_manipulation_waypoint_value(
            start,
            hand,
            "waist_yaw_joint",
        )
        for fraction in (
            *G1_CARRY_LOCOMOTION_UPPER_BODY_INTERMEDIATE_FRACTIONS,
            0.0,
        ):
            target_arm = tuple(value * float(fraction) for value in loaded_arm_start)
            physical_waypoint = (
                float(start[0]),
                start_yaw * float(fraction),
                0.0,
                *target_arm,
            )
            actuator_arm = actuator_command_from_measured_ik_delta(
                previous_command=actuator_arm,
                measured_joints=measured_arm,
                desired_measured_joints=target_arm,
                hand=cast(Any, hand),
            )
            waypoint = g1_manipulation_waypoint_with_arm_values(
                physical_waypoint,
                hand,
                actuator_arm,
            )
            next_other_arm = tuple(value * float(fraction) for value in other_start)
            raw_report = self._execute_g1_control_samples(
                phase=("carry_upright" if fraction > 0.0 else "carry_ready"),
                samples=g1_control_smooth_ramp_samples(
                    segment_start,
                    waypoint,
                    ticks=ticks,
                ),
                hand=hand,
                other_arm_start=idle_arm,
                other_arm_end=next_other_arm,
                finger_start=holding_fingers,
                finger_end=holding_fingers,
                prepare_final_frame=True,
            )
            report = IsaacAutonomousEnv._g1_arm_tracking_against_measured_target(
                raw_report,
                target_arm=target_arm,
                actuator_command=actuator_arm,
            )
            reports.append(report)
            # A loaded PD controller can require a small command bias to reach
            # the desired measured joint state. Preserve the previous bias and
            # close only the live measured residual before advancing to the
            # next quarter-step. The physical target and acceptance tolerance
            # remain unchanged.
            settled_ticks = 0
            while (
                math.isfinite(float(report["maximum_terminal_arm_error_rad"]))
                and float(report["maximum_terminal_arm_error_rad"])
                > G1_CARRY_WAIST_RESTORE_SETTLE_TOLERANCE_RAD
                and settled_ticks < G1_REACH_SETTLE_MAX_PHYSICS_STEPS
            ):
                remaining = self._remaining_physics_steps()
                if remaining is not None and remaining < G1_REACH_SETTLE_PHYSICS_STEPS:
                    break
                actuator_arm = actuator_command_from_measured_ik_delta(
                    previous_command=actuator_arm,
                    measured_joints=report["measured_arm_after_rad"],
                    desired_measured_joints=target_arm,
                    hand=cast(Any, hand),
                )
                corrected_waypoint = g1_manipulation_waypoint_with_arm_values(
                    physical_waypoint,
                    hand,
                    actuator_arm,
                )
                raw_report = self._execute_g1_control_samples(
                    phase="carry_arm_settle",
                    samples=g1_control_smooth_ramp_samples(
                        waypoint,
                        corrected_waypoint,
                        ticks=G1_REACH_SETTLE_PHYSICS_STEPS,
                    ),
                    hand=hand,
                    other_arm_start=next_other_arm,
                    other_arm_end=next_other_arm,
                    finger_start=holding_fingers,
                    finger_end=holding_fingers,
                    prepare_final_frame=True,
                )
                report = IsaacAutonomousEnv._g1_arm_tracking_against_measured_target(
                    raw_report,
                    target_arm=target_arm,
                    actuator_command=actuator_arm,
                )
                reports.append(report)
                settled_ticks += G1_REACH_SETTLE_PHYSICS_STEPS
                waypoint = corrected_waypoint
            self._require_g1_arm_tracking(report)
            segment_start = waypoint
            measured_arm = tuple(
                float(value) for value in report["measured_arm_after_rad"]
            )
            idle_arm = next_other_arm

        final_report = reports[-1]
        settled_ticks = 0
        while (
            self._g1_waist_tracking_error_rad(final_report)
            > G1_CARRY_WAIST_RESTORE_SETTLE_TOLERANCE_RAD
            and settled_ticks < G1_REACH_SETTLE_MAX_PHYSICS_STEPS
        ):
            remaining = self._remaining_physics_steps()
            if remaining is not None and remaining < G1_REACH_SETTLE_PHYSICS_STEPS:
                break
            raw_report = self._execute_g1_control_samples(
                phase="carry_posture_settle",
                samples=(segment_start,) * G1_REACH_SETTLE_PHYSICS_STEPS,
                hand=hand,
                other_arm_start=idle_arm,
                other_arm_end=idle_arm,
                finger_start=holding_fingers,
                finger_end=holding_fingers,
                prepare_final_frame=True,
            )
            final_report = IsaacAutonomousEnv._g1_arm_tracking_against_measured_target(
                raw_report,
                target_arm=(0.0,) * 7,
                actuator_command=actuator_arm,
            )
            reports.append(final_report)
            self._require_g1_arm_tracking(final_report)
            settled_ticks += G1_REACH_SETTLE_PHYSICS_STEPS
        self._require_g1_waist_tracking(final_report)
        return segment_start, reports

    def _restore_g1_waist_pitch_after_grasp(
        self,
        *,
        start_waypoint: Sequence[float],
        hand: str,
        other_arm: Sequence[float],
        holding_fingers: Sequence[float],
        require_arm_tracking: bool = True,
        ticks: int,
        reports: list[Dict[str, Any]],
    ) -> tuple[float, ...]:
        """Ramp measured waist pitch to zero while arms and pelvis stay frozen.

        Pacing matches the existing carry-prepare cadence: one remaining-
        fraction ramp of ``carry_transition_physics_steps`` per official
        intermediate fraction, plus a terminal ramp to zero. A single jump to
        zero would command the full remaining angle in one PD step and risk
        breaking friction retention. Pelvis height is taken from
        ``start_waypoint`` (standing after the stand-up phase) and held fixed
        for every ramp. The phase picks up the measured post-retraction waist
        rather than replanning ``carry_retraction``.
        """

        from tarsphere.infra.manipulation.g1_arm_ik import (
            actuator_command_from_measured_ik_delta,
        )

        expected_length = g1_manipulation_waypoint_length(hand)
        start = tuple(float(value) for value in start_waypoint)
        if len(start) != expected_length or not all(
            math.isfinite(value) for value in start
        ):
            raise RuntimeError(
                "G1 waist restore requires a finite "
                f"{expected_length}-D measured start waypoint."
            )
        start_pitch = g1_manipulation_waypoint_value(
            start,
            hand,
            "waist_pitch_joint",
        )
        if math.isclose(start_pitch, 0.0, abs_tol=0.02):
            return start
        frozen_arm = g1_manipulation_waypoint_arm_values(start, hand)
        frozen_pelvis = start[0]
        frozen_yaw = g1_manipulation_waypoint_value(
            start,
            hand,
            "waist_yaw_joint",
        )
        fractions = (
            *G1_CARRY_LOCOMOTION_UPPER_BODY_INTERMEDIATE_FRACTIONS,
            0.0,
        )
        segment_start = start
        arm_actuator_command = frozen_arm
        for fraction in fractions:
            waypoint = (
                frozen_pelvis,
                frozen_yaw,
                start_pitch * float(fraction),
                *arm_actuator_command,
            )
            raw_report = self._execute_g1_control_samples(
                phase="carry_waist_restore",
                samples=g1_control_smooth_ramp_samples(
                    segment_start,
                    waypoint,
                    ticks=ticks,
                ),
                hand=hand,
                other_arm_start=other_arm,
                other_arm_end=other_arm,
                finger_start=holding_fingers,
                finger_end=holding_fingers,
                prepare_final_frame=True,
            )
            report = IsaacAutonomousEnv._g1_arm_tracking_against_measured_target(
                raw_report,
                target_arm=frozen_arm,
                actuator_command=arm_actuator_command,
            )
            reports.append(report)
            # A ramp ending is not evidence that the PD-controlled arm has
            # arrived.  Correct the actuator command by the remaining live
            # state error while keeping ``frozen_arm`` as the physical target;
            # otherwise finite steady-state command bias can make repeated
            # holds converge to the wrong joint state forever.
            settled_ticks = 0
            while (
                math.isfinite(float(report["maximum_terminal_arm_error_rad"]))
                and float(report["maximum_terminal_arm_error_rad"])
                > G1_CARRY_WAIST_RESTORE_SETTLE_TOLERANCE_RAD
                and settled_ticks < G1_REACH_SETTLE_MAX_PHYSICS_STEPS
            ):
                remaining = self._remaining_physics_steps()
                if remaining is not None and remaining < G1_REACH_SETTLE_PHYSICS_STEPS:
                    break
                try:
                    arm_actuator_command = actuator_command_from_measured_ik_delta(
                        previous_command=arm_actuator_command,
                        measured_joints=report["measured_arm_after_rad"],
                        desired_measured_joints=frozen_arm,
                        hand=cast(Any, hand),
                    )
                except ValueError as exc:
                    # A target close to an actuator limit can make the
                    # feed-forward bias correction itself inadmissible even
                    # though the last valid command only needs more physical
                    # settling time.  Keep issuing that exact valid command
                    # for the same bounded settle window and retain the
                    # unchanged measured-state acceptance gate below.  If the
                    # arm does not converge naturally, the ordinary maximum
                    # settle budget still ends in the strict tracking error.
                    report["carry_tracking_correction_error"] = str(exc)
                corrected_waypoint = (
                    frozen_pelvis,
                    frozen_yaw,
                    start_pitch * float(fraction),
                    *arm_actuator_command,
                )
                raw_report = self._execute_g1_control_samples(
                    phase="carry_waist_settle",
                    samples=g1_control_smooth_ramp_samples(
                        waypoint,
                        corrected_waypoint,
                        ticks=G1_REACH_SETTLE_PHYSICS_STEPS,
                    ),
                    hand=hand,
                    other_arm_start=other_arm,
                    other_arm_end=other_arm,
                    finger_start=holding_fingers,
                    finger_end=holding_fingers,
                    prepare_final_frame=True,
                )
                report = IsaacAutonomousEnv._g1_arm_tracking_against_measured_target(
                    raw_report,
                    target_arm=frozen_arm,
                    actuator_command=arm_actuator_command,
                )
                reports.append(report)
                settled_ticks += G1_REACH_SETTLE_PHYSICS_STEPS
                waypoint = corrected_waypoint
            print(
                "[G1-CONTROL] "
                f"phase=carry_waist_restore remaining_fraction={fraction:.2f} "
                f"maximum_terminal_arm_error_rad="
                f"{float(report['maximum_terminal_arm_error_rad']):.4f} "
                "(arm joints and pelvis height held at post-stand values)",
                flush=True,
            )
            if require_arm_tracking:
                self._require_g1_arm_tracking(report)
            segment_start = waypoint
        measured_arm = tuple(
            float(value) for value in reports[-1]["measured_arm_after_rad"]
        )
        if len(measured_arm) != 7 or not all(
            math.isfinite(value) for value in measured_arm
        ):
            raise RuntimeError(
                "G1 waist restore lost the measured carrying-arm joints."
            )
        return (
            float(reports[-1]["pelvis_height_command_m"]),
            float(reports[-1]["waist_yaw_rad"]),
            0.0,
            *measured_arm,
        )

    def _execute_g1_control_samples(
        self,
        *,
        phase: str,
        samples: Sequence[Sequence[float]],
        hand: str,
        other_arm_start: Sequence[float],
        other_arm_end: Sequence[float],
        finger_start: Sequence[float],
        finger_end: Sequence[float],
        prepare_final_frame: bool = False,
        contact_target_id: str | None = None,
        finish_when: Callable[[], bool] | None = None,
    ) -> Dict[str, Any]:
        """Follow one partial-body trajectory and measure physical tracking.

        The idle arm interpolates from ``other_arm_start`` to ``other_arm_end``
        on the same per-tick fraction already used for the Dex3 fingers.
        """

        measured_arm_before = self._arm_joint_positions(hand)
        if measured_arm_before is None:
            raise RuntimeError(f"G1 {phase} cannot measure the live {hand} arm state.")
        other_hand = "left" if str(hand).lower() == "right" else "right"
        measured_other_arm_before = self._arm_joint_positions(other_hand)
        if measured_other_arm_before is None:
            raise RuntimeError(
                f"G1 {phase} cannot measure the live {other_hand} arm state."
            )
        palm_before = self._robot_frame_point_in_world(
            np.zeros(3, dtype=np.float64),
            frame_name=f"{hand}_hand_palm_link",
        )
        carried_object_id = next(
            (
                object_id
                for object_id, carried_hand in getattr(
                    self._g1_attachment, "physical_holds", {}
                ).items()
                if carried_hand == hand
            ),
            None,
        )
        tracked_object_id = carried_object_id or contact_target_id
        object_before = None
        object_bottom_before_m: float | None = None
        tracked_item = None
        if tracked_object_id is not None:
            tracked_item = next(
                (
                    candidate
                    for candidate in self._g1_attachment.objects
                    if candidate.object_id == tracked_object_id
                ),
                None,
            )
        if tracked_item is not None:
            object_bbox_before = np.asarray(
                self._g1_attachment._live_pickable_bbox(tracked_item),
                dtype=np.float64,
            )
            object_before = object_bbox_before.mean(axis=0)
            object_bottom_before_m = float(object_bbox_before[0][2])

        commands = tuple(
            self._g1_manipulation_action(
                waypoint,
                hand=hand,
                other_arm=tuple(
                    float(start) + (float(end) - float(start)) * fraction
                    for start, end in zip(
                        other_arm_start,
                        other_arm_end,
                        strict=True,
                    )
                ),
                finger_positions=tuple(
                    float(start) + (float(end) - float(start)) * fraction
                    for start, end in zip(finger_start, finger_end, strict=True)
                ),
            )
            for fraction, waypoint in (
                ((index + 1) / len(samples), sample)
                for index, sample in enumerate(samples)
            )
        )
        if not commands:
            raise RuntimeError(f"G1 {phase} control trajectory is empty.")
        remaining = self._remaining_physics_steps()
        if remaining is not None and remaining < len(commands):
            raise RuntimeError(
                f"G1 {phase} needs {len(commands)} physics ticks but only "
                f"{remaining} remain in the episode."
            )
        target_contact_observed = False

        def diagnose_control_tick(timing: str, step_index: int) -> None:
            nonlocal target_contact_observed
            self._on_physics_tick(timing, step_index)
            if timing == "after" and contact_target_id is not None:
                target_contact_observed = bool(
                    target_contact_observed
                    or self._g1_attachment._grasp_contacts(
                        hand=hand,
                        object_id=contact_target_id,
                    )
                )

        result = self._robot_body.step_simulation(
            env=self._env,
            action_name="follow_joint_trajectory",
            env_action=commands[0],
            physics_step_budget=len(commands),
            control_for_tick=lambda tick: commands[tick],
            step_human_bodies=self._step_human_bodies,
            pump_kit_updates=self._pump_kit_updates,
            pump_motion_window=self._pump_human_motion_window,
            global_tick=self._physics_steps,
            diagnose_physics_tick=diagnose_control_tick,
            finish_when=finish_when,
            prepare_final_frame=(
                self._prepare_final_decision_frame if prepare_final_frame else None
            ),
            abort_requested=self._abort_requested,
        )
        if result.observation:
            self._obs = result.observation
        self._physics_steps += result.physics_steps
        self._g1_body_height_command_m = float(samples[-1][0])
        self._publish_g1_body_height_command()
        self._assert_robot_upright(
            action_name=f"g1_grasp_{phase}",
            env_action=commands[-1],
        )
        if result.terminated or result.truncated:
            raise RuntimeError(f"Isaac terminated during the G1 {phase} trajectory.")
        measured_arm_after = self._arm_joint_positions(hand)
        if measured_arm_after is None:
            raise RuntimeError(f"G1 {phase} lost the live {hand} arm measurement.")
        measured_other_arm_after = self._arm_joint_positions(other_hand)
        if measured_other_arm_after is None:
            raise RuntimeError(
                f"G1 {phase} lost the live {other_hand} arm measurement."
            )
        target_arm = np.asarray(
            g1_manipulation_waypoint_arm_values(samples[-1], hand),
            dtype=np.float64,
        )
        measured_arm = np.asarray(measured_arm_after, dtype=np.float64)
        arm_joint_names = list(G1_ARM_JOINT_NAMES[str(hand).lower()])
        terminal_arm_signed_error_rad = (target_arm - measured_arm).tolist()
        worst_index = int(np.argmax(np.abs(target_arm - measured_arm)))
        joint_state = self._g1_curobo_joint_positions()
        palm_after = self._robot_frame_point_in_world(
            np.zeros(3, dtype=np.float64),
            frame_name=f"{hand}_hand_palm_link",
        )
        palm_delta = palm_after - palm_before
        commanded_waist_yaw_rad = g1_manipulation_waypoint_value(
            samples[-1],
            hand,
            "waist_yaw_joint",
        )
        commanded_waist_pitch_rad = g1_manipulation_waypoint_value(
            samples[-1],
            hand,
            "waist_pitch_joint",
        )
        measured_waist_yaw_rad = float(joint_state["waist_yaw_joint"])
        measured_waist_pitch_rad = float(joint_state["waist_pitch_joint"])
        if not all(
            math.isfinite(value)
            for value in (
                commanded_waist_yaw_rad,
                commanded_waist_pitch_rad,
                measured_waist_yaw_rad,
                measured_waist_pitch_rad,
            )
        ):
            raise RuntimeError(
                f"G1 {phase} lost a finite waist command or measurement: "
                f"commanded_yaw={commanded_waist_yaw_rad!r}, "
                f"measured_yaw={measured_waist_yaw_rad!r}, "
                f"commanded_pitch={commanded_waist_pitch_rad!r}, "
                f"measured_pitch={measured_waist_pitch_rad!r}."
            )
        report = {
            "phase": phase,
            "phase_name": phase,
            "physics_steps": result.physics_steps,
            "pelvis_height_command_m": float(samples[-1][0]),
            "waist_yaw_rad": commanded_waist_yaw_rad,
            "waist_pitch_rad": commanded_waist_pitch_rad,
            "measured_waist_yaw_rad": measured_waist_yaw_rad,
            "measured_waist_pitch_rad": measured_waist_pitch_rad,
            "waist_yaw_error_rad": (commanded_waist_yaw_rad - measured_waist_yaw_rad),
            "waist_pitch_error_rad": (
                commanded_waist_pitch_rad - measured_waist_pitch_rad
            ),
            "arm_joint_names": arm_joint_names,
            "terminal_arm_target_rad": target_arm.tolist(),
            "measured_arm_before_rad": list(measured_arm_before),
            "measured_arm_after_rad": list(measured_arm_after),
            "other_arm_joint_names": list(G1_ARM_JOINT_NAMES[other_hand]),
            "other_arm_target_rad": [float(value) for value in other_arm_end],
            "measured_other_arm_before_rad": list(measured_other_arm_before),
            "measured_other_arm_after_rad": list(measured_other_arm_after),
            "maximum_terminal_other_arm_error_rad": float(
                np.max(
                    np.abs(
                        np.asarray(measured_other_arm_after, dtype=np.float64)
                        - np.asarray(other_arm_end, dtype=np.float64)
                    )
                )
            ),
            "terminal_arm_signed_error_rad": terminal_arm_signed_error_rad,
            "worst_arm_joint_name": arm_joint_names[worst_index],
            "worst_arm_joint_signed_error_rad": float(
                terminal_arm_signed_error_rad[worst_index]
            ),
            "maximum_terminal_arm_error_rad": float(
                np.max(np.abs(measured_arm - target_arm))
            ),
            "commanded_arm_delta_l2_rad": float(
                np.linalg.norm(
                    target_arm - np.asarray(measured_arm_before, dtype=np.float64)
                )
            ),
            "palm_before_world_m": palm_before.tolist(),
            "palm_after_world_m": palm_after.tolist(),
            "palm_delta_xyz_m": palm_delta.tolist(),
            "palm_displacement_m": float(np.linalg.norm(palm_delta)),
            "finger_joint_positions_after_rad": self._dex3_joint_positions(hand),
            "target_contact_observed": target_contact_observed,
        }
        from tarsphere.embodiment.dex3 import dex3_controller_name

        finger_command = result.terminal_controller_action.get(
            dex3_controller_name(hand)
        )
        if not (
            isinstance(finger_command, (list, tuple))
            and len(finger_command) == 1
            and isinstance(finger_command[0], (list, tuple))
            and len(finger_command[0]) == 7
        ):
            raise RuntimeError(f"G1 {phase} lost its terminal Dex3 command.")
        report["finger_command_after_rad"] = [
            float(value) for value in finger_command[0]
        ]
        if str(hand).lower() == "right":
            report.update(
                self._g1_right_wrist_yaw_control_snapshot(
                    waypoint=samples[-1],
                    hand=hand,
                    other_arm=tuple(float(value) for value in other_arm_end),
                )
            )
        if tracked_object_id is not None and object_before is not None:
            assert tracked_item is not None
            object_bbox_after = np.asarray(
                self._g1_attachment._live_pickable_bbox(tracked_item),
                dtype=np.float64,
            )
            object_after = object_bbox_after.mean(axis=0)
            object_bottom_after_m = float(object_bbox_after[0][2])
            object_delta = object_after - object_before
            physical_hold = (
                self._g1_attachment.physical_hold_state(
                    object_id=tracked_object_id,
                )
                if tracked_object_id in self._g1_attachment.physical_holds
                else None
            )
            support_clearance_m = (
                float(physical_hold["support_clearance_m"])
                if physical_hold is not None
                else None
            )
            report.update(
                {
                    "object_id": tracked_object_id,
                    "object_before_world_m": object_before.tolist(),
                    "object_after_world_m": object_after.tolist(),
                    "object_delta_xyz_m": object_delta.tolist(),
                    "object_displacement_m": float(np.linalg.norm(object_delta)),
                    "relative_motion_error_m": float(
                        np.linalg.norm(object_delta - palm_delta)
                    ),
                    **self._g1_lift_phase_evidence(
                        palm_before=palm_before,
                        palm_after=palm_after,
                        object_before=object_before,
                        object_after=object_after,
                        object_bottom_before_m=object_bottom_before_m,
                        object_bottom_after_m=object_bottom_after_m,
                        support_clearance_m=support_clearance_m,
                    ),
                }
            )
            if physical_hold is not None:
                report["physical_hold"] = physical_hold
        return report

    @staticmethod
    def _g1_lift_phase_evidence(
        *,
        palm_before: np.ndarray,
        palm_after: np.ndarray,
        object_before: np.ndarray | None,
        object_after: np.ndarray | None,
        object_bottom_before_m: float | None,
        object_bottom_after_m: float | None,
        support_clearance_m: float | None,
    ) -> Dict[str, Any]:
        """Summarize per-phase lift evidence for post-grasp diagnosis."""

        palm_delta = palm_after - palm_before
        evidence: Dict[str, Any] = {
            "palm_z_m": float(palm_after[2]),
            "palm_delta_horizontal_m": float(np.linalg.norm(palm_delta[:2])),
            "palm_delta_vertical_m": float(palm_delta[2]),
        }
        if object_after is not None:
            evidence["object_z_m"] = float(object_after[2])
        if object_bottom_after_m is not None:
            evidence["object_bottom_z_m"] = float(object_bottom_after_m)
        if object_before is not None and object_after is not None:
            object_delta = object_after - object_before
            evidence.update(
                {
                    "object_delta_horizontal_m": float(
                        np.linalg.norm(object_delta[:2])
                    ),
                    "object_delta_vertical_m": float(object_delta[2]),
                }
            )
        if object_bottom_before_m is not None and object_bottom_after_m is not None:
            evidence["object_bottom_delta_m"] = float(
                object_bottom_after_m - object_bottom_before_m
            )
        if support_clearance_m is not None:
            evidence["support_clearance_m"] = float(support_clearance_m)
        return evidence

    @staticmethod
    def _g1_failed_grasp_skill(
        *,
        object_id: str,
        hand: str,
        phase_reports: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Expose partial phase evidence on failed grasps for persistence."""

        return {
            "object_id": object_id,
            "hand": hand,
            "planner": "curobo_v2",
            # ``phases`` is agent-forbidden; ``phase_reports`` survives into
            # heartbeat and persisted trace sanitization.
            "phase_reports": list(phase_reports),
            "transport_ready": False,
            "standing": False,
            "arms_retracted": False,
        }

    @staticmethod
    def _g1_arm_tracking_against_measured_target(
        report: Mapping[str, Any],
        *,
        target_arm: Sequence[float],
        actuator_command: Sequence[float],
    ) -> Dict[str, Any]:
        """Measure carry tracking against the physical target, not PD bias."""

        measured = np.asarray(
            report.get("measured_arm_after_rad"),
            dtype=np.float64,
        ).reshape(-1)
        target = np.asarray(target_arm, dtype=np.float64).reshape(-1)
        command = np.asarray(
            actuator_command,
            dtype=np.float64,
        ).reshape(-1)
        if measured.size != 7 or target.size != 7 or command.size != 7:
            raise RuntimeError(
                "G1 carry tracking requires seven measured, target, and "
                "actuator joint values."
            )
        signed_error = target - measured
        finite = np.isfinite(signed_error)
        if np.all(finite):
            worst_index = int(np.argmax(np.abs(signed_error)))
            maximum_error = float(abs(signed_error[worst_index]))
        else:
            worst_index = int(np.flatnonzero(~finite)[0])
            maximum_error = float("nan")
        names = list(
            report.get("arm_joint_names")
            or [f"arm_joint_{index}" for index in range(7)]
        )
        if len(names) != 7:
            raise RuntimeError("G1 carry tracking requires seven arm joint names.")
        adjusted = dict(report)
        adjusted.update(
            {
                "carry_tracking_target_rad": target.tolist(),
                "carry_actuator_command_rad": command.tolist(),
                "terminal_arm_target_rad": target.tolist(),
                "terminal_arm_signed_error_rad": signed_error.tolist(),
                "maximum_terminal_arm_error_rad": maximum_error,
                "worst_arm_joint_name": str(names[worst_index]),
                "worst_arm_joint_signed_error_rad": float(signed_error[worst_index]),
            }
        )
        return adjusted

    @staticmethod
    def _require_g1_arm_tracking(
        report: Mapping[str, Any],
        *,
        tolerance_rad: float = G1_ARM_TRAJECTORY_TRACKING_TOLERANCE_RAD,
    ) -> None:
        """Reject a manipulation phase whose measured arm missed its target."""

        phase = str(report.get("phase") or "unknown")
        error = float(report["maximum_terminal_arm_error_rad"])
        worst_joint = str(report["worst_arm_joint_name"])
        worst_signed_error = float(report["worst_arm_joint_signed_error_rad"])
        print(
            "[G1-CONTROL] "
            f"phase={phase} maximum_terminal_arm_error_rad={error:.4f} "
            f"worst_joint={worst_joint} "
            f"worst_signed_error_rad={worst_signed_error:.4f}",
            flush=True,
        )
        if not math.isfinite(error) or error > tolerance_rad:
            per_joint = report.get("terminal_arm_signed_error_rad") or []
            joint_names = report.get("arm_joint_names") or []
            joint_errors = ", ".join(
                f"{name}={float(signed):+.4f}"
                for name, signed in zip(joint_names, per_joint, strict=True)
            )
            raise G1ArmTrackingError(
                f"G1 {phase} arm tracking error {error:.4f} rad exceeds the "
                f"{tolerance_rad:.4f} rad controller contract: "
                f"worst_joint={worst_joint} "
                f"worst_signed_error_rad={worst_signed_error:+.4f}, "
                f"per_joint_signed_error_rad=[{joint_errors}]."
            )

    @staticmethod
    def _require_g1_other_arm_tracking(
        report: Mapping[str, Any],
        *,
        tolerance_rad: float = G1_ARM_TRAJECTORY_TRACKING_TOLERANCE_RAD,
    ) -> None:
        """Reject transport until the unloaded arm reaches its target."""

        phase = str(report.get("phase") or "unknown")
        error = float(report["maximum_terminal_other_arm_error_rad"])
        if not math.isfinite(error) or error > tolerance_rad:
            raise G1ArmTrackingError(
                f"G1 {phase} unloaded-arm tracking error {error:.4f} rad "
                f"exceeds the {tolerance_rad:.4f} rad controller contract."
            )

    @staticmethod
    def _g1_waist_tracking_error_rad(report: Mapping[str, Any]) -> float:
        """Return the largest measured waist error for a control endpoint."""

        errors = (
            float(report["waist_yaw_error_rad"]),
            float(report["waist_pitch_error_rad"]),
        )
        if not all(math.isfinite(value) for value in errors):
            return float("nan")
        return max(abs(value) for value in errors)

    @staticmethod
    def _require_g1_neutral_transport_posture(
        waypoint: Sequence[float],
        report: Mapping[str, Any],
    ) -> tuple[bool, bool]:
        """Confirm the final measured carry pose before retaining the grasp."""

        arm_error = float(report["maximum_terminal_arm_error_rad"])
        other_arm_error = float(report["maximum_terminal_other_arm_error_rad"])
        waist_error = IsaacAutonomousEnv._g1_waist_tracking_error_rad(report)
        body_height = float(waypoint[0])
        arms_retracted = (
            math.isfinite(arm_error)
            and math.isfinite(other_arm_error)
            and arm_error <= G1_ARM_TRAJECTORY_TRACKING_TOLERANCE_RAD
            and other_arm_error <= G1_ARM_TRAJECTORY_TRACKING_TOLERANCE_RAD
        )
        standing = (
            math.isfinite(body_height)
            and math.isfinite(waist_error)
            and body_height >= G1_STANDING_HEIGHT_COMMAND_M - 0.02
            and waist_error <= G1_CARRY_WAIST_RESTORE_SETTLE_TOLERANCE_RAD
        )
        if not arms_retracted or not standing:
            raise G1CarryPostureTrackingError(
                "G1 grasp did not reach a measured neutral transport posture: "
                f"arm_error_rad={arm_error:.4f}, "
                f"other_arm_error_rad={other_arm_error:.4f}, "
                f"waist_error_rad={waist_error:.4f}, "
                f"body_height_command_m={body_height:.4f}."
            )
        return arms_retracted, standing

    @staticmethod
    def _require_g1_waist_tracking(report: Mapping[str, Any]) -> None:
        """Reject loaded locomotion until the measured waist is upright."""

        error = IsaacAutonomousEnv._g1_waist_tracking_error_rad(report)
        if (
            not math.isfinite(error)
            or error > G1_CARRY_WAIST_RESTORE_SETTLE_TOLERANCE_RAD
        ):
            raise G1WaistTrackingError(
                "G1 loaded locomotion waist tracking error "
                f"{error:.4f} rad exceeds the "
                f"{G1_CARRY_WAIST_RESTORE_SETTLE_TOLERANCE_RAD:.4f} rad contract: "
                "commanded_yaw="
                f"{float(report['waist_yaw_rad']):+.4f}, "
                "measured_yaw="
                f"{float(report['measured_waist_yaw_rad']):+.4f}, "
                "commanded_pitch="
                f"{float(report['waist_pitch_rad']):+.4f}, "
                "measured_pitch="
                f"{float(report['measured_waist_pitch_rad']):+.4f}."
            )

    @staticmethod
    def _g1_reach_tracking_warning(
        report: Mapping[str, Any],
    ) -> Dict[str, Any] | None:
        """Record reach tracking misses without pre-empting physical closure."""

        try:
            IsaacAutonomousEnv._require_g1_arm_tracking(
                report,
                tolerance_rad=G1_GRASP_REACH_TRACKING_TOLERANCE_RAD,
            )
        except RuntimeError as exc:
            return {
                "phase": "closing_alignment_tracking_warning",
                "source_phase": str(report.get("phase") or "unknown"),
                "message": str(exc),
                "terminal": False,
                "acceptance_basis": (
                    "subsequent_measured_contact_force_closure_and_retention"
                ),
            }
        return None

    def _execute_physical_carry_stand_up(
        self,
        decision: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        """Raise the pelvis while preserving the compact friction-grasp pose."""

        from tarsphere.infra.manipulation.g1_arm_ik import G1_UPPER_COMMAND_SCALE

        hand = str(self._g1_carried_hand or "")
        if hand not in {"left", "right"}:
            raise RuntimeError(
                "Physical carry stand_up requires an active carried hand."
            )
        if self._g1_attachment is None or self._last_g1_arm_ik is None:
            raise RuntimeError(
                "Physical carry stand_up requires an active attachment executor."
            )
        if self._g1_carry_finger_positions is None:
            raise RuntimeError(
                "Physical carry stand_up requires the active finger closure."
            )
        object_id = next(
            (
                str(item_id)
                for item_id, carried_hand in self._g1_attachment.physical_holds.items()
                if carried_hand == hand
            ),
            "",
        )
        if not object_id:
            raise RuntimeError(
                "Physical carry stand_up could not resolve the held object."
            )
        upper_command = self._last_g1_arm_ik.get("upper_body_command")
        if not isinstance(upper_command, (list, tuple)) or len(upper_command) != 15:
            raise RuntimeError(
                "Physical carry stand_up requires the cached 15-D "
                "upper-body command from the successful grasp."
            )
        waist_yaw_rad = float(upper_command[0]) * G1_UPPER_COMMAND_SCALE
        joint_state = self._g1_curobo_joint_positions()
        waist_pitch_rad = float(joint_state["waist_pitch_joint"])
        arm_joint_positions = tuple(
            float(value) for value in self._last_g1_arm_ik.get("joint_positions") or ()
        )
        if len(arm_joint_positions) != 7:
            raise RuntimeError(
                "Physical carry stand_up requires seven cached arm joints."
            )
        start_pelvis_m = self._g1_body_height_command()
        if math.isclose(
            start_pelvis_m,
            G1_STANDING_HEIGHT_COMMAND_M,
            abs_tol=0.02,
        ):
            self._g1_body_height_command_m = G1_STANDING_HEIGHT_COMMAND_M
            self._publish_g1_body_height_command()
            self._g1_body_posture = "standing"
            return (
                self.observe(),
                False,
                {
                    "executed": True,
                    "feedback": "",
                    "physics_steps": self._physics_steps,
                    "robot_stability": self._robot_stability(),
                },
            )
        start_waypoint = (
            start_pelvis_m,
            waist_yaw_rad,
            waist_pitch_rad,
            *arm_joint_positions,
        )
        end_waypoint = (
            G1_STANDING_HEIGHT_COMMAND_M,
            waist_yaw_rad,
            waist_pitch_rad,
            *arm_joint_positions,
        )
        stand_samples = resample_control_trajectory(
            (start_waypoint, end_waypoint),
            waypoint_interval_s=1.0 / SIMULATION_PHYSICS_HZ,
            control_frequency_hz=SIMULATION_PHYSICS_HZ,
        )
        other_arm = (0.0,) * 7
        holding_fingers = tuple(
            float(value) for value in self._g1_carry_finger_positions
        )
        self._begin_decision_render_window(len(stand_samples))
        self._g1_attachment.hold_manipulation_stance()
        try:
            stand_report = self._execute_g1_control_samples(
                phase="carry_stand_up",
                samples=stand_samples,
                hand=hand,
                other_arm_start=other_arm,
                other_arm_end=other_arm,
                finger_start=holding_fingers,
                finger_end=holding_fingers,
                prepare_final_frame=True,
            )
            print(
                "[G1-CONTROL] "
                "phase=carry_stand_up "
                f"maximum_terminal_arm_error_rad="
                f"{float(stand_report['maximum_terminal_arm_error_rad']):.4f} "
                "(physical retention keeps compact arm during stand_up)",
                flush=True,
            )
            self._g1_attachment.require_contact_physical_grasp(
                object_id=object_id,
            )
        finally:
            self._g1_attachment.release_manipulation_stance()
            self._end_decision_render_window()
        measured_palm = self._robot_frame_point_in_world(
            np.zeros(3, dtype=np.float64),
            frame_name=f"{hand}_hand_palm_link",
        )
        end_waypoint = (
            float(stand_report["pelvis_height_command_m"]),
            float(stand_report["waist_yaw_rad"]),
            float(stand_report["waist_pitch_rad"]),
            *(float(value) for value in stand_report["measured_arm_after_rad"]),
        )
        self._last_g1_arm_ik = {
            "hand": hand,
            "target_position_world": measured_palm.tolist(),
            "position_error_m": 0.0,
            "joint_positions": list(
                g1_manipulation_waypoint_arm_values(end_waypoint, hand)
            ),
            "upper_body_command": self._g1_upper_body_command(
                end_waypoint,
                hand=hand,
                other_arm=other_arm,
            ),
        }
        self._g1_body_height_command_m = G1_STANDING_HEIGHT_COMMAND_M
        self._publish_g1_body_height_command()
        self._g1_body_posture = "standing"
        dropped_objects = self._g1_attachment.detect_dropped_objects()
        dropped_ids = [str(state.get("object_id") or "") for state in dropped_objects]
        feedback = ""
        if dropped_ids:
            self._g1_carried_hand = None
            self._g1_carry_finger_positions = None
            feedback = (
                "The physically grasped object lost live hand contact "
                f"during this action and is no longer held: {dropped_ids}."
            )
            print(f"[G1-GRASP] {feedback}", flush=True)
        return (
            self.observe(),
            False,
            {
                "executed": True,
                "feedback": feedback,
                "dropped_objects": dropped_ids,
                "physics_steps": self._physics_steps,
                "robot_stability": self._robot_stability(),
            },
        )

    def _confirm_g1_carry_contact(self) -> Dict[str, Any] | None:
        """Physically confirm a boundary-time contact miss before a drop.

        Successful carry actions return without advancing physics.  On the
        exceptional no-contact sample, the robot holds its *measured* pelvis,
        waist, and both arm poses and reissues only the already-active Dex3
        closure for a short fixed window.  The ordinary finite-friction hold
        predicate remains the sole verdict after that window.
        """

        hand = str(self._g1_carried_hand or "")
        if hand not in {"left", "right"}:
            return None
        if self._g1_carry_finger_positions is None:
            raise RuntimeError(
                "G1 carry contact confirmation requires the active finger "
                "closure command."
            )
        object_ids = [
            str(object_id)
            for object_id, carried_hand in self._g1_attachment.physical_holds.items()
            if carried_hand == hand
        ]
        if len(object_ids) != 1:
            raise RuntimeError(
                "G1 carry contact confirmation requires exactly one physical "
                f"hold for the {hand} hand; got {object_ids!r}."
            )
        object_id = object_ids[0]
        before = self._g1_attachment.physical_hold_state(
            object_id=object_id,
        )
        if before.get("held"):
            return None

        selected_arm = self._arm_joint_positions(hand)
        other_hand = "left" if hand == "right" else "right"
        other_arm = self._arm_joint_positions(other_hand)
        if selected_arm is None or other_arm is None:
            raise RuntimeError(
                "G1 carry contact confirmation requires measured positions "
                "for both arms."
            )
        joint_state = self._g1_curobo_joint_positions()
        waypoint = (
            self._g1_body_height_command(),
            float(joint_state["waist_yaw_joint"]),
            float(joint_state["waist_pitch_joint"]),
            *(float(value) for value in selected_arm),
        )
        measured_fingers = tuple(self._dex3_joint_positions(hand))
        holding_fingers = tuple(
            float(value) for value in self._g1_carry_finger_positions
        )
        self._begin_decision_render_window(_G1_CARRY_CONTACT_CONFIRMATION_PHYSICS_STEPS)
        try:
            control_report = self._execute_g1_control_samples(
                phase="carry_contact_confirmation",
                samples=(waypoint,) * _G1_CARRY_CONTACT_CONFIRMATION_PHYSICS_STEPS,
                hand=hand,
                other_arm_start=other_arm,
                other_arm_end=other_arm,
                finger_start=measured_fingers,
                finger_end=holding_fingers,
                prepare_final_frame=True,
                contact_target_id=object_id,
            )
        finally:
            self._end_decision_render_window()
        after = self._g1_attachment.physical_hold_state(
            object_id=object_id,
        )
        recovered = bool(after.get("held"))
        print(
            "[G1-GRASP] carry contact confirmation: "
            f"object={object_id} hand={hand} "
            f"physics_steps={_G1_CARRY_CONTACT_CONFIRMATION_PHYSICS_STEPS} "
            f"recovered={str(recovered).lower()}",
            flush=True,
        )
        return {
            "triggered": True,
            "object_id": object_id,
            "hand": hand,
            "physics_steps": _G1_CARRY_CONTACT_CONFIRMATION_PHYSICS_STEPS,
            "contact_observed_during_window": bool(
                control_report.get("target_contact_observed")
            ),
            "held_after_window": recovered,
        }

    def _g1_manipulation_action(
        self,
        waypoint: Sequence[float],
        *,
        hand: str,
        other_arm: Sequence[float],
        finger_positions: Sequence[float],
    ) -> Dict[str, Any]:
        """Map one planner waypoint onto official G1 and Dex3 controllers."""

        from tarsphere.embodiment.dex3 import (
            dex3_controller_name,
            waist_pitch_controller_name,
        )

        expected_length = g1_manipulation_waypoint_length(hand)
        values = tuple(float(value) for value in waypoint)
        if len(values) != expected_length:
            raise ValueError(
                "A G1 manipulation waypoint must contain pelvis height, "
                "waist yaw, waist pitch, and seven arm joints."
            )
        waist_pitch_rad = g1_manipulation_waypoint_value(
            values,
            hand,
            "waist_pitch_joint",
        )
        return {
            "move_by_speed": [
                0.0,
                0.0,
                0.0,
                values[0],
                self._g1_upper_body_command(
                    values,
                    hand=hand,
                    other_arm=other_arm,
                ),
            ],
            waist_pitch_controller_name(): [[waist_pitch_rad]],
            dex3_controller_name(hand): [list(finger_positions)],
        }

    @staticmethod
    def _g1_upper_body_command(
        waypoint: Sequence[float],
        *,
        hand: str,
        other_arm: Sequence[float],
    ) -> list[float]:
        """Encode waist and bilateral arms in G1 policy coordinates."""

        from tarsphere.infra.manipulation.g1_arm_ik import G1_UPPER_COMMAND_SCALE
        from tarsphere.infra.manipulation.g1_grasp import (
            g1_manipulation_waypoint_arm_values,
            g1_manipulation_waypoint_length,
            g1_manipulation_waypoint_value,
        )

        values = tuple(float(value) for value in waypoint)
        other_arm = tuple(float(value) for value in other_arm)
        expected_length = g1_manipulation_waypoint_length(hand)
        if len(values) != expected_length or len(other_arm) != 7:
            raise ValueError("G1 upper-body control requires two seven-joint arms.")
        selected_arm = g1_manipulation_waypoint_arm_values(values, hand)
        left_arm = selected_arm if hand == "left" else other_arm
        right_arm = selected_arm if hand == "right" else other_arm
        return [
            g1_manipulation_waypoint_value(values, hand, "waist_yaw_joint")
            / G1_UPPER_COMMAND_SCALE,
            *(value / G1_UPPER_COMMAND_SCALE for value in left_arm),
            *(value / G1_UPPER_COMMAND_SCALE for value in right_arm),
        ]

    def _g1_right_wrist_yaw_control_snapshot(
        self,
        *,
        waypoint: Sequence[float],
        hand: str,
        other_arm: Sequence[float],
    ) -> Dict[str, float]:
        """Compare planned, commanded, applied, and measured wrist yaw."""

        from tarsphere.embodiment.controllers.g1_upper_body_hold import (
            _G1_RIGHT_WRIST_YAW_UPPER_INDEX,
        )
        from tarsphere.embodiment.robots.g1 import G1_LOCOMOTION_JOINT_NAMES
        from tarsphere.infra.manipulation.g1_arm_ik import (
            G1_RIGHT_ARM_JOINT_NAMES,
            G1_UPPER_COMMAND_SCALE,
        )
        from tarsphere.infra.manipulation.g1_grasp import g1_manipulation_waypoint_value

        planned_rad = float(
            g1_manipulation_waypoint_value(
                waypoint,
                hand,
                "right_wrist_yaw_joint",
            )
        )
        upper_command = self._g1_upper_body_command(
            waypoint,
            hand=hand,
            other_arm=other_arm,
        )
        commanded_policy = float(upper_command[_G1_RIGHT_WRIST_YAW_UPPER_INDEX])
        commanded_rad = commanded_policy * G1_UPPER_COMMAND_SCALE
        applied_rad = float("nan")
        applied_effort_nm = float("nan")
        robot = self._live_robots()[0]
        controller = robot.controllers.get("move_by_speed")
        applied = getattr(controller, "applied_joint_positions", None)
        if applied is not None:
            wrist_index = G1_LOCOMOTION_JOINT_NAMES.index("right_wrist_yaw_joint")
            applied_rad = float(np.asarray(applied, dtype=np.float64)[wrist_index])
        articulation = getattr(robot, "articulation", None)
        if articulation is not None:
            dof_index = articulation.get_dof_index("right_wrist_yaw_joint")
            inner = getattr(articulation, "_articulation", articulation)
            efforts = None
            if hasattr(inner, "get_measured_joint_efforts"):
                try:
                    efforts = inner.get_measured_joint_efforts(
                        joint_indices=np.asarray([int(dof_index)], dtype=np.int64),
                    )
                except TypeError:
                    efforts = inner.get_measured_joint_efforts()
            if efforts is not None:
                values = np.asarray(
                    efforts.cpu() if hasattr(efforts, "cpu") else efforts
                ).reshape(-1)
                if values.size:
                    applied_effort_nm = float(values[-1])
        measured_rad = float("nan")
        measured = self._arm_joint_positions(hand)
        if measured is not None:
            measured_rad = float(
                measured[G1_RIGHT_ARM_JOINT_NAMES.index("right_wrist_yaw_joint")]
            )
        return {
            "right_wrist_yaw_planned_rad": planned_rad,
            "right_wrist_yaw_commanded_rad": commanded_rad,
            "right_wrist_yaw_applied_rad": applied_rad,
            "right_wrist_yaw_measured_rad": measured_rad,
            "right_wrist_yaw_applied_effort_nm": applied_effort_nm,
            "right_wrist_yaw_commanded_minus_applied_rad": (
                commanded_rad - applied_rad
            ),
        }

    def _begin_agent_grasp_retention(
        self,
        *,
        hand: str,
        object_id: str,
        closing_volume_center_world: list[float],
        closing_volume_axes_world: list[list[float]],
        closing_volume_extents_m: list[float],
    ) -> Dict[str, Any]:
        """Apply the configured retention contract after physical closure."""

        finger_positions = self._dex3_joint_positions(hand)
        return self._g1_attachment.begin_physical_grasp(
            hand,
            object_id=object_id,
            closing_axis_world=list(closing_volume_axes_world[0]),
            finger_joint_positions=finger_positions,
        )

    def _execute_g1_grasp_reach(
        self,
        ik_spec: Dict[str, Any],
        *,
        hand: str,
        stage: str,
        gripper_preset: str = "open",
    ) -> Dict[str, Any]:
        """Execute one generated arm stage and report its measured outcome."""
        reach_decision = self._resolve_g1_arm_ik_decision(
            {
                "action": "set_g1_gripper_preset",
                "preset": gripper_preset,
                "hand": hand,
                "controller_metadata": {
                    "g1_arm_ik": dict(ik_spec),
                    "g1_arm_ik_finish_on_target": True,
                    # Same predicate the certified grasp reach converges on: at the
                    # default 8e-2 rad the accumulated link error can leave the
                    # closing centre centimetres from the object.
                    "g1_arm_ik_finish_tolerance_rad": 0.02,
                },
            }
        )
        reach_decision = self._include_dex3_controller_action(reach_decision)
        report = reach_decision["controller_metadata"]["g1_arm_ik_result"]
        target_joints = np.asarray(
            report["joint_positions"],
            dtype=np.float64,
        ).reshape(-1)
        tolerance = float(
            reach_decision["controller_metadata"]["g1_arm_ik_finish_tolerance_rad"]
        )

        def _reached() -> bool:
            current = self._arm_joint_positions(str(report["hand"]))
            if current is None or target_joints.size != len(current):
                return False
            return (
                float(
                    np.max(
                        np.abs(np.asarray(current, dtype=np.float64) - target_joints)
                    )
                )
                <= tolerance
            )

        command = self._robot_body.command_for_decision(reach_decision)
        result = self._robot_body.step_simulation(
            env=self._env,
            action_name=command.action_name,
            env_action=command.env_action,
            physics_step_budget=getattr(command, "physics_step_budget", None),
            step_human_bodies=self._step_human_bodies,
            pump_kit_updates=self._pump_kit_updates,
            pump_motion_window=self._pump_human_motion_window,
            global_tick=self._physics_steps,
            sample_body_dynamics=self._g1_body_dynamics,
            diagnose_physics_tick=self._on_physics_tick,
            finish_when=_reached,
            abort_requested=self._abort_requested,
        )
        if result.observation:
            self._obs = result.observation
        self._physics_steps += result.physics_steps
        self._assert_robot_upright(
            action_name=command.action_name,
            env_action=command.env_action,
        )
        report = {
            **report,
            "joints_converged": _reached(),
            "physics_steps": result.physics_steps,
        }
        print(
            f"[G1-GRASP] {stage} reach: "
            f"hand={report['hand']} "
            f"target_world={report['target_position_world']} "
            f"ik_error_m={float(report['position_error_m']):.4f} "
            f"physics_steps={result.physics_steps} "
            f"joints_converged={report['joints_converged']}",
            flush=True,
        )
        return report

    def _execute_g1_internal_posture_transition(
        self,
        decision: Dict[str, Any],
    ) -> None:
        """Execute one hidden carry/height transition inside a grasp skill."""

        resolved = self._resolve_g1_arm_ik_decision(decision)
        command = self._robot_body.command_for_decision(resolved)
        result = self._robot_body.step_simulation(
            env=self._env,
            action_name=command.action_name,
            env_action=command.env_action,
            physics_step_budget=getattr(command, "physics_step_budget", None),
            step_human_bodies=self._step_human_bodies,
            pump_kit_updates=self._pump_kit_updates,
            pump_motion_window=self._pump_human_motion_window,
            global_tick=self._physics_steps,
            sample_body_dynamics=self._g1_body_dynamics,
            diagnose_physics_tick=self._on_physics_tick,
            abort_requested=self._abort_requested,
        )
        if result.observation:
            self._obs = result.observation
        self._physics_steps += result.physics_steps
        self._assert_robot_upright(
            action_name=command.action_name,
            env_action=command.env_action,
        )

    def _resolve_g1_arm_ik_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Compile a world-space gripper target into official G1 commands.

        This is an execution primitive, not oracle cognition.  Oracle targets
        come from benchmark-private asset geometry; evaluated agents may call
        the same primitive with an ego-estimated 3-D target.  In both cases the
        base must already face the target and the live FixedJoint distance gate
        remains authoritative.
        """
        metadata = decision.get("controller_metadata")
        if not isinstance(metadata, dict):
            return self._inject_g1_carry_pose(decision)
        remaining_fraction = metadata.get("g1_locomotion_upper_body_remaining_fraction")
        if remaining_fraction is not None:
            fraction = float(remaining_fraction)
            if not 0.0 < fraction < 1.0:
                raise ValueError(
                    "G1 locomotion upper-body interpolation fraction must "
                    "lie strictly between zero and one."
                )
            if self._last_g1_arm_ik is None:
                raise RuntimeError(
                    "G1 locomotion upper-body interpolation requested before "
                    "a successful grasp IK command."
                )
            grasp_command = np.asarray(
                self._last_g1_arm_ik.get("upper_body_command"),
                dtype=np.float64,
            ).reshape(-1)
            if grasp_command.size != 15 or not np.all(np.isfinite(grasp_command)):
                raise RuntimeError(
                    "Successful G1 grasp IK must retain a finite 15-D "
                    "upper-body command for locomotion interpolation."
                )
            upper_command = [float(value) for value in fraction * grasp_command]
            resolved = dict(decision)
            resolved["controller_action"] = {
                "move_by_speed": [
                    0.0,
                    0.0,
                    0.0,
                    self._g1_body_height_command(),
                    upper_command,
                ],
            }
            resolved["controller_backend"] = "move_by_speed"
            resolved_metadata = dict(metadata)
            resolved_metadata["upper_body_command"] = upper_command
            resolved["controller_metadata"] = resolved_metadata
            # Preserve _last_g1_arm_ik until the explicit all-zero terminal
            # phase clears it; every fraction is measured from the identical
            # live grasp solution rather than compounding numeric error.
            return resolved
        if bool(metadata.get("g1_reset_to_locomotion_upper_body", False)):
            move = (decision.get("controller_action") or {}).get("move_by_speed")
            if (
                not isinstance(move, (list, tuple))
                or len(move) < 5
                or not isinstance(move[4], (list, tuple))
                or len(move[4]) != 15
                or any(abs(float(value)) > 1e-12 for value in move[4])
            ):
                raise ValueError(
                    "G1 locomotion upper-body reset requires the official "
                    "15-D zero target."
                )
            move = list(move)
            body_height = self._g1_body_height_command()
            if not math.isclose(float(move[3]), body_height, abs_tol=1e-12):
                resolved = dict(decision)
                controller_action = dict(decision["controller_action"])
                move[3] = body_height
                controller_action["move_by_speed"] = move
                resolved["controller_action"] = controller_action
            else:
                resolved = decision
            # Do not let the generic attached-object hold path replace this
            # explicit reset with the preceding grasp IK.  Clearing the
            # transient IK report also makes subsequent nested path commands
            # inherit the zero target cached by TARSphereG1MoveBySpeedController.
            self._last_g1_arm_ik = None
            return resolved
        ik_spec = metadata.get("g1_arm_ik")
        place_spec = metadata.get("g1_place_ik")
        hold_last = bool(metadata.get("g1_arm_ik_hold", False))
        if (
            not isinstance(ik_spec, dict)
            and not isinstance(place_spec, dict)
            and not hold_last
        ):
            return self._inject_g1_carry_pose(decision)

        if isinstance(place_spec, dict):
            if self._g1_attachment is None:
                raise RuntimeError(
                    "G1 placement IK requires an active attachment executor."
                )
            surface = np.asarray(
                place_spec.get("support_surface_world"), dtype=np.float64
            ).reshape(-1)
            if surface.size != 3 or not np.all(np.isfinite(surface)):
                raise ValueError(
                    "g1_place_ik support_surface_world must contain three finite values."
                )
            placement_report = self._g1_attachment.placement_hand_target(
                hand=str(place_spec.get("hand") or "auto").lower(),
                object_id=str(place_spec.get("object_id") or ""),
                support_surface_world=[float(value) for value in surface],
                clearance_m=G1_PRE_RELEASE_CLEARANCE_M,
                correction_gain=float(place_spec.get("correction_gain", 1.0)),
            )
            placement_report["horizontal_tolerance_m"] = (
                G1_PRE_RELEASE_HORIZONTAL_TOLERANCE_M
            )
            placement_report["support_gap_tolerance_m"] = (
                G1_PRE_RELEASE_SUPPORT_CONTACT_TOLERANCE_M
            )
            ik_spec = {
                "hand": str(placement_report["hand"]),
                "target_position_world": list(
                    placement_report["target_position_world"]
                ),
                # The robot must face the placement site.  The compensated
                # hand target can be laterally offset because the FixedJoint
                # preserves the live hand/object transform; it is not the
                # semantic facing reference.
                "facing_reference_position_world": [float(value) for value in surface],
                "position_tolerance_m": (G1_PRE_RELEASE_HORIZONTAL_TOLERANCE_M),
                "posture": str(place_spec.get("posture") or "front_place"),
                "max_facing_error_deg": float(
                    place_spec.get("max_facing_error_deg", 30.0)
                ),
            }
            metadata = dict(metadata)
            metadata["g1_place_ik_result"] = placement_report
            metadata["g1_arm_ik_finish_on_target"] = True
            # At 8e-2 rad the sum of seven link errors can still displace the
            # held object well beyond the 10 cm placement gate.  This stricter
            # joint predicate lets the action-specific arm horizon settle.
            metadata["g1_arm_ik_finish_tolerance_rad"] = 0.02

        if hold_last:
            if self._last_g1_arm_ik is None:
                raise RuntimeError(
                    "G1 arm IK hold requested before a successful IK reach."
                )
            upper_command = list(self._last_g1_arm_ik["upper_body_command"])
            resolved = dict(decision)
            resolved_metadata = dict(metadata)
            resolved_metadata["g1_arm_ik_result"] = dict(self._last_g1_arm_ik)
            resolved["controller_metadata"] = resolved_metadata
            resolved["controller_action"] = {
                "move_by_speed": [
                    0.0,
                    0.0,
                    0.0,
                    self._g1_body_height_command(),
                    upper_command,
                ],
            }
            resolved["controller_backend"] = "move_by_speed"
            return resolved

        requested_hand = str(ik_spec.get("hand") or "auto").lower()
        if requested_hand == "carried":
            if self._g1_carried_hand not in {"left", "right"}:
                raise RuntimeError(
                    "G1 carried-hand IK requested without an active attachment."
                )
            requested_hand = str(self._g1_carried_hand)
        if requested_hand not in {"auto", "left", "right"}:
            raise ValueError(
                "Official G1 IK hand must be auto, carried, left, or right; "
                f"got {requested_hand!r}."
            )
        target_world = np.asarray(
            ik_spec.get("target_position_world"), dtype=np.float64
        ).reshape(-1)
        if target_world.size != 3 or not np.all(np.isfinite(target_world)):
            raise ValueError(
                "g1_arm_ik target_position_world must contain three finite values."
            )

        robot_position = self._robot_position()
        if robot_position is None:
            raise RuntimeError("G1 arm IK cannot read the live robot position.")
        facing_reference = np.asarray(
            ik_spec.get("facing_reference_position_world", target_world),
            dtype=np.float64,
        ).reshape(-1)
        if facing_reference.size != 3 or not np.all(np.isfinite(facing_reference)):
            raise ValueError(
                "G1 arm IK facing reference must contain three finite values."
            )
        target_bearing = math.atan2(
            float(facing_reference[1]) - float(robot_position[1]),
            float(facing_reference[0]) - float(robot_position[0]),
        )
        facing_error = (target_bearing - self._robot_yaw() + math.pi) % (
            2.0 * math.pi
        ) - math.pi
        max_facing_error = math.radians(
            float(ik_spec.get("max_facing_error_deg", 30.0))
        )
        if abs(facing_error) > max_facing_error:
            raise GraspNotPossibleHere(
                GraspFailureCode.STANCE_FACING,
                "G1 arm IK rejected a non-frontal grasp: "
                f"facing_error_deg={math.degrees(facing_error):.1f}, "
                f"limit_deg={math.degrees(max_facing_error):.1f}.",
                evidence={
                    "facing_error_deg": math.degrees(facing_error),
                    "maximum_facing_error_deg": math.degrees(max_facing_error),
                },
            )

        measured_target_torso = self._world_point_in_robot_frame(
            target_world, frame_name="torso_link"
        )
        target_torso = np.asarray(
            ik_spec.get(
                "solver_target_position_torso",
                measured_target_torso,
            ),
            dtype=np.float64,
        ).reshape(-1)
        if target_torso.size != 3 or not np.all(np.isfinite(target_torso)):
            raise ValueError(
                "g1_arm_ik solver target must contain three finite values."
            )
        from tarsphere.embodiment.dex3 import (
            DEX3_GRASP_CENTER_FROM_PALM_M,
        )
        from tarsphere.infra.manipulation.g1_arm_ik import (
            frontal_grasp_palm_rotation_in_torso,
            grasp_palm_rotation_in_torso,
            hand_reference_position_in_torso,
            solve_hand_pose,
        )

        tolerance = float(ik_spec.get("position_tolerance_m", 0.06))
        align_grasp_center = bool(ik_spec.get("align_grasp_center", False))
        reference_from_palm = (
            DEX3_GRASP_CENTER_FROM_PALM_M if align_grasp_center else (0.0, 0.0, 0.0)
        )
        # A caller that authored how the object is approached says so, and the
        # palm is built from those axes: a top-down rim grasp and a frontal body
        # grasp need different rotations of the same hand. A caller that says
        # nothing gets the certified frontal palm, which is what every promoted
        # oracle trajectory was solved with.
        approach_axis_world = ik_spec.get("approach_axis_world")
        closing_axis_world = ik_spec.get("closing_axis_world")
        palm_rotation_world = ik_spec.get("palm_rotation_world")
        if (approach_axis_world is None) != (closing_axis_world is None):
            raise ValueError(
                "g1_arm_ik takes the approach and closing axes together or "
                "neither; one alone does not determine a palm rotation."
            )
        if palm_rotation_world is not None:
            if approach_axis_world is not None:
                raise ValueError(
                    "g1_arm_ik accepts either palm_rotation_world or grasp "
                    "approach/closing axes, not both."
                )
            world_rotation = np.asarray(
                palm_rotation_world,
                dtype=np.float64,
            )
            if (
                world_rotation.shape != (3, 3)
                or not np.all(np.isfinite(world_rotation))
                or not np.allclose(
                    world_rotation.T @ world_rotation,
                    np.eye(3),
                    atol=2e-3,
                )
                or not np.isclose(
                    np.linalg.det(world_rotation),
                    1.0,
                    atol=2e-3,
                )
            ):
                raise ValueError(
                    "g1_arm_ik palm_rotation_world must be a finite "
                    "right-handed 3x3 rotation."
                )
            world_to_torso = np.asarray(
                self._world_to_robot_frame_affine("torso_link")["linear"],
                dtype=np.float64,
            )
            target_rotation_torso = world_to_torso @ world_rotation
        elif not align_grasp_center:
            target_rotation_torso = None
        elif approach_axis_world is None:
            target_rotation_torso = frontal_grasp_palm_rotation_in_torso(target_torso)
        else:
            target_rotation_torso = grasp_palm_rotation_in_torso(
                approach_axis_torso=self._world_direction_in_robot_frame(
                    approach_axis_world,
                    frame_name="torso_link",
                ),
                closing_axis_torso=self._world_direction_in_robot_frame(
                    closing_axis_world,
                    frame_name="torso_link",
                ),
            )
        solver_rotation = ik_spec.get("solver_target_rotation_torso")
        if solver_rotation is not None:
            target_rotation_torso = np.asarray(
                solver_rotation,
                dtype=np.float64,
            )
            if (
                target_rotation_torso.shape != (3, 3)
                or not np.all(np.isfinite(target_rotation_torso))
                or not np.allclose(
                    target_rotation_torso.T @ target_rotation_torso,
                    np.eye(3),
                    atol=2e-3,
                )
                or not np.isclose(
                    np.linalg.det(target_rotation_torso),
                    1.0,
                    atol=2e-3,
                )
            ):
                raise ValueError(
                    "g1_arm_ik solver rotation must be a finite "
                    "right-handed 3x3 rotation."
                )
        candidate_hands = (
            ("left", "right") if requested_hand == "auto" else (requested_hand,)
        )
        warm_starts = {
            hand: self._arm_joint_positions(hand) for hand in candidate_hands
        }
        candidates = [
            solve_hand_pose(
                target_torso,
                hand=hand,
                warm_start=warm_starts[hand],
                position_tolerance_m=tolerance,
                # Task tolerance decides whether the target belongs to the
                # reachable envelope. Numerical tolerance decides how closely
                # an eligible solution is refined before it is commanded.
                # Conflating them made an 8 cm grasp envelope terminate IK up
                # to 8 cm from its hand target, so native contact depended on
                # incidental locomotion drift.
                solver_tolerance_m=min(
                    tolerance,
                    G1_ARM_IK_SOLVER_TOLERANCE_M,
                ),
                reference_from_palm=reference_from_palm,
                target_rotation_torso=target_rotation_torso,
            )
            for hand in candidate_hands
        ]
        converged = [candidate for candidate in candidates if candidate.converged]
        correction_step = (
            isinstance(place_spec, dict) and "correction_gain" in place_spec
        )
        projected = []
        if not converged and correction_step:
            # A residual target can lie just outside the arm workspace even
            # though the best bounded joint solution makes useful progress.
            # Execute that projected step only when its predicted improvement
            # exceeds numerical solver precision, then compile the next step
            # from the live payload bbox. Initial reaches are never relaxed,
            # and final acceptance remains the physical 10 cm geometry gate.
            for candidate in candidates:
                warm_start = warm_starts.get(candidate.hand)
                if warm_start is None:
                    continue
                current_hand = hand_reference_position_in_torso(
                    warm_start,
                    hand=candidate.hand,
                    reference_from_palm=reference_from_palm,
                )
                current_error = float(np.linalg.norm(target_torso - current_hand))
                if (
                    float(candidate.position_error_m) + G1_ARM_IK_SOLVER_TOLERANCE_M
                    < current_error
                ):
                    projected.append(candidate)
        eligible = converged or projected
        if not eligible:
            errors = {
                candidate.hand: round(float(candidate.position_error_m), 4)
                for candidate in candidates
            }
            orientation_errors = {
                candidate.hand: round(
                    math.degrees(float(candidate.orientation_error_rad)),
                    1,
                )
                for candidate in candidates
            }
            raise GraspNotPossibleHere(
                GraspFailureCode.REACH_ENVELOPE,
                "Neither eligible official G1 arm reached the requested manipulation "
                f"envelope: target_torso={target_torso.tolist()!r}, "
                f"errors_m={errors!r}, "
                f"orientation_errors_deg={orientation_errors!r}, "
                f"tolerance_m={tolerance:.3f}.",
                evidence={
                    "position_errors_m": errors,
                    "orientation_errors_deg": orientation_errors,
                    "position_tolerance_m": float(tolerance),
                },
            )
        preferred_for_side = "left" if float(target_torso[1]) >= 0.0 else "right"
        result = min(
            eligible,
            key=lambda candidate: (
                float(candidate.position_error_m),
                candidate.hand != preferred_for_side,
            ),
        )
        balance_contralateral = bool(ik_spec.get("balance_contralateral_arm", False))
        upper_command = result.upper_body_command(
            balance_contralateral=balance_contralateral
        )
        report = {
            "hand": result.hand,
            "target_position_world": [float(value) for value in target_world],
            "facing_reference_position_world": [
                float(value) for value in facing_reference
            ],
            "target_position_torso": list(result.target_position_torso),
            "requested_target_position_torso": [
                float(value) for value in measured_target_torso
            ],
            "predicted_hand_position_torso": list(result.hand_position_torso),
            "predicted_reference_position_torso": list(
                result.reference_position_torso or result.hand_position_torso
            ),
            "reference_from_palm": list(result.reference_from_palm),
            "align_grasp_center": align_grasp_center,
            # Present only when the caller authored an approach. Absent means
            # the certified frontal palm was used, which is a different thing
            # from a frontal approach that was spelled out.
            "approach_axis_world": (
                None
                if approach_axis_world is None
                else [float(value) for value in approach_axis_world]
            ),
            "closing_axis_world": (
                None
                if closing_axis_world is None
                else [float(value) for value in closing_axis_world]
            ),
            "position_error_m": float(result.position_error_m),
            "orientation_error_deg": math.degrees(float(result.orientation_error_rad)),
            "joint_positions": list(result.joint_positions),
            "upper_body_command": list(upper_command),
            "facing_error_deg": math.degrees(facing_error),
            "iterations": int(result.iterations),
            "workspace_projected": not bool(result.converged),
            "posture": str(ik_spec.get("posture") or "front_auto"),
            "balance_contralateral_arm": balance_contralateral,
        }
        self._last_g1_arm_ik = report
        print(
            "[G1-IK] "
            f"hand={result.hand} "
            f"target_world={report['target_position_world']} "
            f"target_torso={[round(v, 4) for v in result.target_position_torso]} "
            f"error_m={result.position_error_m:.4f} "
            f"workspace_projected={report['workspace_projected']} "
            f"facing_error_deg={math.degrees(facing_error):.1f} "
            f"iterations={result.iterations}",
            flush=True,
        )
        resolved = dict(decision)
        resolved_metadata = dict(metadata)
        skill_action = resolved_metadata.get("skill_action")
        if isinstance(skill_action, dict):
            skill_action = dict(skill_action)
            skill_action["hand"] = result.hand
            resolved_metadata["skill_action"] = skill_action
        resolved_metadata["g1_arm_ik_result"] = report
        resolved["controller_metadata"] = resolved_metadata
        if "hand" in resolved:
            resolved["hand"] = result.hand
        if str(decision.get("action") or "") == "move_end_effector":
            from tarsphere.embodiment.dex3 import arm_controller_name

            if result.hand != G1_MANIPULATION_HAND:
                raise RuntimeError(
                    "move_end_effector must resolve to the certified right arm."
                )
            resolved["controller_action"] = {
                arm_controller_name(result.hand): [list(result.joint_positions)],
            }
            resolved["controller_backend"] = "right_arm_joint_controller"
        else:
            resolved["controller_action"] = {
                "move_by_speed": [
                    0.0,
                    0.0,
                    0.0,
                    self._g1_body_height_command(),
                    upper_command,
                ],
            }
            resolved["controller_backend"] = "move_by_speed"
        return resolved

    def _compile_move_end_effector_decision(
        self,
        decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve a public body-frame 7D delta against the live right palm."""

        if str(decision.get("action") or "") != "move_end_effector":
            return decision
        raw_delta = decision.get("delta")
        if not isinstance(raw_delta, (list, tuple)) or len(raw_delta) != 7:
            raise ValueError(
                "move_end_effector requires delta=[dx, dy, dz, drx, dry, drz, gripper]."
            )
        delta = np.asarray(raw_delta, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(delta)):
            raise ValueError("move_end_effector delta must be finite.")
        if np.any(np.abs(delta[:3]) > G1_END_EFFECTOR_DELTA_MAX_TRANSLATION_M + 1e-12):
            raise ValueError("move_end_effector translation exceeds its contract.")
        if (
            float(np.linalg.norm(delta[3:6]))
            > G1_END_EFFECTOR_DELTA_MAX_ROTATION_RAD + 1e-12
        ):
            raise ValueError("move_end_effector rotation exceeds its contract.")
        if not 0.0 <= float(delta[6]) <= 1.0:
            raise ValueError("move_end_effector gripper must lie in [0, 1].")

        from tarsphere.infra.manipulation.g1_arm_ik import (
            hand_transform_in_torso,
            rotation_matrix_from_vector,
        )

        hand = G1_MANIPULATION_HAND
        palm_frame = f"{hand}_hand_palm_link"
        start_palm_world = self._robot_frame_pose_in_world(palm_frame)
        yaw = float(self._robot_yaw())
        body_rotation_world = np.asarray(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        target_palm_world = np.array(start_palm_world, copy=True)
        target_palm_world[:3, 3] += body_rotation_world @ delta[:3]
        delta_rotation_world = (
            body_rotation_world
            @ rotation_matrix_from_vector(delta[3:6])
            @ body_rotation_world.T
        )
        target_palm_world[:3, :3] = delta_rotation_world @ start_palm_world[:3, :3]
        arm_joints = self._arm_joint_positions(hand)
        if arm_joints is None:
            raise RuntimeError(
                "move_end_effector cannot read the live right-arm joints."
            )
        model_start_palm_torso = hand_transform_in_torso(
            arm_joints,
            hand=hand,
        )
        torso_affine = self._world_to_robot_frame_affine("torso_link")
        world_to_torso = np.asarray(
            torso_affine["linear"],
            dtype=np.float64,
        )
        torso_offset = np.asarray(
            torso_affine["offset"],
            dtype=np.float64,
        ).reshape(-1)
        if (
            world_to_torso.shape != (3, 3)
            or torso_offset.size != 3
            or not np.all(np.isfinite(world_to_torso))
            or not np.all(np.isfinite(torso_offset))
            or not np.allclose(
                world_to_torso.T @ world_to_torso,
                np.eye(3),
                atol=1e-5,
            )
        ):
            raise RuntimeError(
                "move_end_effector requires a rigid live torso transform."
            )
        torso_from_world = np.eye(4, dtype=np.float64)
        torso_from_world[:3, :3] = world_to_torso
        torso_from_world[:3, 3] = torso_offset
        measured_start_palm_torso = torso_from_world @ start_palm_world
        measured_target_palm_torso = torso_from_world @ target_palm_world
        # The live USD palm and the analytic arm chain have a small fixed frame
        # discrepancy.  Map the desired live pose through the transform that
        # makes the current measured pose exactly equal to the current model
        # pose.  This preserves a zero delta and correctly conjugates rotations;
        # adding a delta directly to model FK only corrected translation and
        # caused live orientation to overshoot.
        model_from_measured = model_start_palm_torso @ np.linalg.inv(
            measured_start_palm_torso
        )
        solver_target_palm_torso = model_from_measured @ measured_target_palm_torso
        robot_position = self._robot_position()
        if robot_position is None:
            raise RuntimeError("move_end_effector cannot read the live robot position.")
        facing_reference = [
            float(robot_position[0]) + math.cos(yaw),
            float(robot_position[1]) + math.sin(yaw),
            float(robot_position[2]),
        ]
        metadata = decision.get("controller_metadata")
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        metadata.update(
            {
                "skill_action": {
                    "action": "move_end_effector",
                    "delta": delta.tolist(),
                    "hand": hand,
                },
                "g1_arm_ik": {
                    "hand": hand,
                    "target_position_world": target_palm_world[:3, 3].tolist(),
                    "palm_rotation_world": target_palm_world[:3, :3].tolist(),
                    "solver_target_position_torso": (
                        solver_target_palm_torso[:3, 3].tolist()
                    ),
                    "solver_target_rotation_torso": (
                        solver_target_palm_torso[:3, :3].tolist()
                    ),
                    "facing_reference_position_world": facing_reference,
                    "position_tolerance_m": (
                        G1_END_EFFECTOR_DELTA_POSITION_TOLERANCE_M
                    ),
                    "posture": "local_delta",
                },
                "g1_arm_ik_finish_on_target": True,
                "g1_arm_ik_finish_tolerance_rad": 0.02,
                "g1_pin_base_for_arm_reach": True,
                "move_end_effector": {
                    "requested_delta": delta.tolist(),
                    "start_palm_world": start_palm_world.tolist(),
                    "target_palm_world": target_palm_world.tolist(),
                    "body_rotation_world": body_rotation_world.tolist(),
                },
            }
        )
        compiled = dict(decision)
        compiled["hand"] = hand
        compiled["controller_metadata"] = metadata
        return compiled

    def _move_end_effector_report(
        self,
        metadata: Any,
    ) -> Dict[str, Any]:
        """Measure the achieved public 7D delta after its controller window."""

        if not isinstance(metadata, dict):
            raise RuntimeError("move_end_effector lost its controller metadata.")
        contract = metadata.get("move_end_effector")
        if not isinstance(contract, dict):
            raise RuntimeError("move_end_effector lost its execution contract.")
        requested = np.asarray(
            contract.get("requested_delta"),
            dtype=np.float64,
        ).reshape(-1)
        start = np.asarray(
            contract.get("start_palm_world"),
            dtype=np.float64,
        )
        target = np.asarray(
            contract.get("target_palm_world"),
            dtype=np.float64,
        )
        body_rotation = np.asarray(
            contract.get("body_rotation_world"),
            dtype=np.float64,
        )
        if (
            requested.size != 7
            or start.shape != (4, 4)
            or target.shape != (4, 4)
            or body_rotation.shape != (3, 3)
            or not all(
                np.all(np.isfinite(value))
                for value in (
                    requested,
                    start,
                    target,
                    body_rotation,
                )
            )
        ):
            raise RuntimeError("move_end_effector execution contract is malformed.")

        from tarsphere.embodiment.dex3 import dex3_continuous_closure
        from tarsphere.infra.manipulation.g1_arm_ik import rotation_vector_from_matrix

        hand = G1_MANIPULATION_HAND
        reached = self._robot_frame_pose_in_world(f"{hand}_hand_palm_link")
        achieved_translation = body_rotation.T @ (reached[:3, 3] - start[:3, 3])
        achieved_rotation_matrix = (
            body_rotation.T @ reached[:3, :3] @ start[:3, :3].T @ body_rotation
        )
        achieved_rotation = rotation_vector_from_matrix(achieved_rotation_matrix)
        measured_closure = dex3_continuous_closure(
            hand,
            self._dex3_joint_positions(hand),
        )
        achieved = np.concatenate(
            (
                achieved_translation,
                achieved_rotation,
                np.asarray([measured_closure]),
            )
        )
        position_residual = float(np.linalg.norm(target[:3, 3] - reached[:3, 3]))
        orientation_residual = float(
            np.linalg.norm(
                rotation_vector_from_matrix(target[:3, :3] @ reached[:3, :3].T)
            )
        )
        gripper_residual = abs(measured_closure - float(requested[6]))
        return {
            "executed": True,
            "hand": hand,
            "requested_delta": requested.tolist(),
            "achieved_delta": achieved.tolist(),
            "position_residual_m": position_residual,
            "orientation_residual_rad": orientation_residual,
            "gripper_residual": gripper_residual,
            "feedback": (
                "Right-palm delta executed: position residual "
                f"{position_residual:.3f} m, orientation residual "
                f"{orientation_residual:.3f} rad, measured gripper "
                f"{measured_closure:.2f}. Actual body-frame delta: "
                f"{[round(float(value), 4) for value in achieved]}."
            ),
        }

    def _inject_g1_carry_pose(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Keep the lifted arm pose and finger closure active while carrying."""
        if self._g1_carried_hand is None or self._last_g1_arm_ik is None:
            return decision
        if self._g1_carry_finger_positions is None:
            raise RuntimeError(
                "G1 carry control requires the finger command that established "
                "the active grasp."
            )
        upper_command = list(self._last_g1_arm_ik["upper_body_command"])
        resolved = dict(decision)
        controller_action = resolved.get("controller_action")
        if not isinstance(controller_action, dict):
            # Public actions arrive without a compiled controller payload.
            # Route them so locomotion keeps its planar command and stationary
            # actions receive the router's zero-speed hold. The upper-body
            # target is attached below so action_to_control cannot retract the
            # hand before attachment is verified.
            from tarsphere.harness.actions.router import ActionRouter

            action = str(resolved.get("action") or "")
            params = {
                key: value
                for key, value in resolved.items()
                if key
                not in {
                    "action",
                    "thinking",
                    "controller_action",
                    "controller_backend",
                    "controller_metadata",
                    "envelope",
                }
            }
            robot_body = getattr(self, "_robot_body", None)
            router = robot_body.router if robot_body is not None else ActionRouter()
            try:
                routed, _ = router.route(action, params=params)
            except ValueError:
                routed = {"move_by_speed": [0.0, 0.0, 0.0]}
            controller_action = dict(routed)
            resolved["controller_action"] = controller_action
        if isinstance(controller_action, dict) and "move_by_speed" in controller_action:
            move = list(controller_action.get("move_by_speed") or [])
            while len(move) < 3:
                move.append(0.0)
            # Preserve the ordinary locomotion command exactly. Attachment
            # changes only the upper-body target; it must not create a second
            # motion policy or silently cap rotate/translate commands.
            height = float(move[3]) if len(move) > 3 else self._g1_body_height_command()
            controller_action = dict(controller_action)
            controller_action["move_by_speed"] = [
                float(move[0]),
                float(move[1]),
                float(move[2]),
                height,
                upper_command,
            ]
            resolved["controller_action"] = controller_action
            resolved["controller_backend"] = "move_by_speed"
        # A friction grasp depends on continuous position control at the
        # fingers.  Re-issuing the same closed target beside every body command
        # preserves that normal force; it does not constrain or move the object.
        from tarsphere.embodiment.dex3 import dex3_controller_name

        controller_action = resolved.get("controller_action")
        if isinstance(controller_action, dict):
            controller_action = dict(controller_action)
            controller_action[dex3_controller_name(self._g1_carried_hand)] = [
                list(self._g1_carry_finger_positions)
            ]
            resolved["controller_action"] = controller_action
        # Path/move-to-point actions otherwise remain unchanged. Their nested
        # body call resolves the upper target from the persistent cache while
        # the sibling Dex3 command maintains closure.
        metadata = resolved.get("controller_metadata")
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        metadata["g1_carry_hold"] = {
            "hand": self._g1_carried_hand,
            "upper_body_command": upper_command,
        }
        resolved["controller_metadata"] = metadata
        return resolved

    def _include_dex3_controller_action(
        self, decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge an explicit Dex3 transition beside arm and body control.

        Arm IK is compiled before symbolic actions are routed, so an explicit
        hand target must be merged into the compiled arm command.  Placement is
        deliberately two-stage: keep the fingers closed while the held object
        moves onto its support, then execute the opening trajectory only after
        the live support-geometry gate passes.
        """
        action = str(decision.get("action") or "")
        explicit_hand_action = action in {
            "grasp_object",
            "place_object",
            "set_g1_gripper_preset",
            "move_end_effector",
        }
        if not explicit_hand_action:
            return decision

        params = {
            key: value
            for key, value in decision.items()
            if key
            not in {
                "action",
                "thinking",
                "controller_action",
                "controller_backend",
                "controller_metadata",
                "envelope",
            }
        }
        routed_action, _ = self._robot_body.router.route(
            action,
            params=params,
        )
        hand_controllers = {
            name: value
            for name, value in routed_action.items()
            if name.endswith("_hand_joint_controller")
        }
        if len(hand_controllers) != 1:
            raise RuntimeError(
                "A Dex3 action must resolve to exactly one hand "
                f"controller; got {hand_controllers!r}."
            )

        metadata = decision.get("controller_metadata")
        placement_transition = bool(
            action == "place_object"
            and isinstance(metadata, dict)
            and isinstance(metadata.get("g1_place_ik_result"), dict)
        )
        retained_release = bool(
            action == "place_object"
            and getattr(self, "_g1_carried_hand", None) in {"left", "right"}
        )
        if placement_transition or retained_release:
            from tarsphere.embodiment.dex3 import (
                dex3_controller_name,
                dex3_joint_positions,
            )

            arm_result = (
                metadata.get("g1_arm_ik_result") if isinstance(metadata, dict) else None
            )
            place_result = (
                metadata.get("g1_place_ik_result")
                if isinstance(metadata, dict)
                else None
            )
            hand = str(
                decision.get("hand")
                or (arm_result.get("hand") if isinstance(arm_result, dict) else None)
                or (
                    place_result.get("hand") if isinstance(place_result, dict) else None
                )
                or ""
            )
            if hand not in {"left", "right"}:
                raise RuntimeError(
                    "G1 release requires an explicit left or right hand."
                )
            if retained_release and hand != self._g1_carried_hand:
                raise RuntimeError(
                    "G1 release hand does not match the hand carrying the object: "
                    f"requested={hand!r}, carrying={self._g1_carried_hand!r}."
                )
            hand_controllers = {
                dex3_controller_name(hand): [list(dex3_joint_positions(hand, "grasp"))],
            }

        controller_action = decision.get("controller_action")
        if isinstance(controller_action, dict):
            merged_action = dict(controller_action)
        else:
            params = {
                key: value
                for key, value in decision.items()
                if key
                not in {
                    "action",
                    "thinking",
                    "controller_backend",
                    "controller_metadata",
                    "envelope",
                }
            }
            merged_action, _ = self._robot_body.router.route(
                action,
                params=params,
            )
        merged_action.update(hand_controllers)
        resolved = dict(decision)
        resolved["controller_action"] = merged_action
        return resolved

    def _execute_g1_hand_opening(
        self,
        hand: str,
        *,
        decision_step: int,
    ) -> Dict[str, Any]:
        """Open one Dex3 hand while preserving the reached placement pose.

        The object retention has already been removed at the verified support
        pose.  This trajectory keeps base, pelvis height, waist yaw, and both
        arms at their measured values while the selected fingers open over
        their physical joint-travel duration.
        """

        from tarsphere.embodiment.dex3 import (
            DEX3_OPENING_PHYSICS_STEPS,
            DEX3_OPEN_JOINT_POSITIONS,
            DEX3_RELEASE_OPENING_TOLERANCE_RAD,
        )

        normalized_hand = str(hand).strip().lower()
        if normalized_hand not in {"left", "right"}:
            raise ValueError("Dex3 release hand must be left or right.")
        selected_arm = self._arm_joint_positions(normalized_hand)
        other_hand = "left" if normalized_hand == "right" else "right"
        other_arm = self._arm_joint_positions(other_hand)
        if selected_arm is None or other_arm is None:
            raise RuntimeError(
                "Dex3 release requires the measured positions of both G1 arms."
            )
        joint_state = self._g1_curobo_joint_positions()
        waypoint = (
            self._g1_body_height_command(),
            float(joint_state["waist_yaw_joint"]),
            float(joint_state["waist_pitch_joint"]),
            *(float(value) for value in selected_arm),
        )
        finger_start = tuple(self._dex3_joint_positions(normalized_hand))
        self._execute_g1_control_samples(
            phase="support_contact_hand_opening",
            samples=(waypoint,) * DEX3_OPENING_PHYSICS_STEPS,
            hand=normalized_hand,
            other_arm_start=other_arm,
            other_arm_end=other_arm,
            finger_start=finger_start,
            finger_end=DEX3_OPEN_JOINT_POSITIONS[normalized_hand],
            prepare_final_frame=True,
        )
        tracking = self._dex3_tracking_report(
            normalized_hand,
            preset="open",
        )
        maximum_error = max(
            abs(float(value)) for value in tracking["position_errors_rad"]
        )
        opened = maximum_error <= DEX3_RELEASE_OPENING_TOLERANCE_RAD
        if not opened:
            raise RuntimeError(
                "Dex3 did not reach the physical open-hand release pose: "
                f"maximum_joint_error_rad={maximum_error:.4f}, "
                "tolerance_rad="
                f"{DEX3_RELEASE_OPENING_TOLERANCE_RAD:.4f}."
            )
        return {
            "decision_step": int(decision_step),
            "physics_step": int(self._physics_steps),
            "hand": normalized_hand,
            "measured_joint_positions_rad": list(tracking["measured_positions_rad"]),
            "open_joint_positions_rad": list(tracking["commanded_positions_rad"]),
            "joint_tolerance_rad": DEX3_RELEASE_OPENING_TOLERANCE_RAD,
        }

    def _remember_g1_body_posture(self, decision: Dict[str, Any]) -> None:
        """Keep crouch active until the Agent explicitly requests standing."""
        action = str(decision.get("action") or "")
        if action == "crouch":
            pelvis_height_m = float(decision.get("pelvis_height_m"))
            if not (
                G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M
                <= pelvis_height_m
                <= G1_MAXIMUM_CROUCH_HEIGHT_COMMAND_M
            ):
                raise ValueError(
                    "crouch pelvis_height_m lies outside the certified "
                    f"[{G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M}, "
                    f"{G1_MAXIMUM_CROUCH_HEIGHT_COMMAND_M}] m range"
                )
            self._g1_body_height_command_m = pelvis_height_m
        elif action == "stand_up":
            self._g1_body_height_command_m = G1_STANDING_HEIGHT_COMMAND_M
        self._publish_g1_body_height_command(
            required=action in G1_BODY_POSTURE_ACTIONS,
        )

    def _publish_g1_body_height_command(self, *, required: bool = True) -> None:
        """Share the persistent pelvis-height command with nested locomotion.

        The controller reads the pelvis height from this cache, while
        ``_g1_commanded_body_posture`` reports what was requested. A lost write
        therefore leaves the robot crouched while the posture reads standing,
        and the standing-only gait check would clear it to walk from a crouch.
        Whenever the command actually changed, losing the write is terminal.
        """

        # No environment yet means no controller to disagree with; the command
        # is still recorded and reaches the controller on the first live step.
        if getattr(self, "_env", None) is None:
            return
        try:
            robot = self._runtime_g1_robot()
        except RuntimeError as exc:
            if required:
                raise RuntimeError(
                    "Failed to publish the G1 pelvis-height command "
                    f"{self._g1_body_height_command():.3f} m to the live "
                    f"robot: {exc}"
                ) from exc
            return
        robot._tarsphere_body_height = self._g1_body_height_command()

    def _g1_commanded_body_posture(self) -> str:
        """Return standing/crouched from the persistent height command."""

        return g1_body_posture_from_height_command(self._g1_body_height_command())

    def _reject_crouched_g1_locomotion(self, action: str) -> None:
        """Fail loud if a standing-only gait action is issued from a crouch."""

        require_standing_g1_locomotion(
            action,
            body_posture=self._g1_commanded_body_posture(),
        )

    def _settle_dynamic_grasp_target(self, *, object_id: str) -> Dict[str, Any]:
        """Let one released pickable reach support equilibrium before the arm."""

        if self._g1_attachment is None:
            raise RuntimeError(
                "G1 support settle requires an active attachment executor."
            )
        steps = int(G1_GRASP_SUPPORT_SETTLE_PHYSICS_STEPS)
        remaining = self._remaining_physics_steps()
        if remaining is not None and remaining < steps:
            raise RuntimeError(
                f"G1 support settle needs {steps} physics ticks but only "
                f"{remaining} remain in the episode."
            )
        zero_speed = {"move_by_speed": [0.0, 0.0, 0.0]}
        before = self._g1_attachment.grasp_target_displacement_report(
            object_id=object_id,
        )
        # Half a simulated second of physics belongs to the whole scene, not
        # only the pickable: stepping the env directly would freeze the Humans,
        # ignore a scene-server abort and miss a fall for the entire settle.
        result = self._robot_body.step_simulation(
            env=self._env,
            action_name="support_settle",
            env_action=zero_speed,
            physics_step_budget=steps,
            control_for_tick=lambda _tick: zero_speed,
            step_human_bodies=self._step_human_bodies,
            pump_kit_updates=self._pump_kit_updates,
            pump_motion_window=self._pump_human_motion_window,
            global_tick=self._physics_steps,
            sample_body_dynamics=self._g1_body_dynamics,
            diagnose_physics_tick=self._on_physics_tick,
            abort_requested=self._abort_requested,
        )
        if result.observation:
            self._obs = result.observation
        self._physics_steps += result.physics_steps
        steps = int(result.physics_steps)
        after = self._g1_attachment.grasp_target_displacement_report(
            object_id=object_id,
        )
        return {
            **after,
            "physics_steps": steps,
            "displacement_before_settle_m": float(before["displacement_m"]),
        }

    def _restore_failed_g1_grasp_approach(
        self,
        *,
        hand: str,
        object_id: str,
        other_arm: Sequence[float],
        restore_body_height_m: float,
    ) -> None:
        """Open the hand, freeze the target, and retract after a failed grasp.

        A later ``grasp_object`` retry cannot plan from a reaching arm, and
        ``activate_grasp_target`` refuses a body that is still dynamic or
        listed in ``physical_holds``. Restore both after a contact miss or a
        lost physical hold. The retract ends at the pre-grasp body-height
        command, not the crouched height left by the failed reach.
        """

        from tarsphere.embodiment.dex3 import (
            DEX3_CLOSURE_PHYSICS_STEPS,
            DEX3_OPEN_JOINT_POSITIONS,
        )
        from tarsphere.infra.manipulation.g1_grasp import (
            G1_WAIST_JOINT_NAMES,
            g1_locomotion_zero_manipulation_waypoint,
        )

        if self._g1_attachment is not None:
            self._g1_attachment.restore_grasp_target(object_id=object_id)
        current_arm = self._arm_joint_positions(hand)
        if current_arm is None:
            return
        remaining = self._remaining_physics_steps()
        retract_ticks = int(DEX3_CLOSURE_PHYSICS_STEPS)
        if remaining is not None and remaining < retract_ticks:
            return
        joints = self._g1_curobo_joint_positions()
        start = (
            self._g1_body_height_command(),
            *(float(joints[name]) for name in G1_WAIST_JOINT_NAMES),
            *tuple(float(value) for value in current_arm),
        )
        end = g1_locomotion_zero_manipulation_waypoint(
            hand,
            pelvis_height_command_m=float(restore_body_height_m),
        )
        waypoint_len = len(start)
        samples = tuple(
            tuple(
                start[index] + (end[index] - start[index]) * fraction
                for index in range(waypoint_len)
            )
            for fraction in (
                (tick + 1) / retract_ticks for tick in range(retract_ticks)
            )
        )
        open_fingers = DEX3_OPEN_JOINT_POSITIONS[hand]
        self._execute_g1_control_samples(
            phase="failed_grasp_retract",
            samples=samples,
            hand=hand,
            other_arm_start=other_arm,
            other_arm_end=other_arm,
            finger_start=open_fingers,
            finger_end=open_fingers,
        )
        print(
            "[G1-GRASP] retracted the reaching arm after a contact miss so "
            "the next pose can plan; excluded_closing_centres="
            f"{len(getattr(self, '_g1_excluded_grasp_centres', ()))}.",
            flush=True,
        )

    def _g1_body_height_command(self) -> float:
        """Return the current persistent G1 policy height command."""
        return float(
            getattr(
                self,
                "_g1_body_height_command_m",
                G1_STANDING_HEIGHT_COMMAND_M,
            )
        )

    def _g1_measured_body_height_m(
        self, pelvis_position_world_m: Sequence[float]
    ) -> float:
        """Return live pelvis height above the episode walking floor.

        The locomotion height command is this same quantity. Mapping it
        through ``g1_curobo_pelvis_model_height_from_body_height_m`` places
        cuRobo's synthetic root at the measured stance.
        """

        position = np.asarray(
            pelvis_position_world_m,
            dtype=np.float64,
        ).reshape(-1)
        if position.size != 3 or not np.all(np.isfinite(position)):
            raise RuntimeError(
                "Cannot measure G1 body height without a finite live pelvis "
                f"position, got {pelvis_position_world_m!r}."
            )
        floor_z = float(self._episode_floor_z())
        if not math.isfinite(floor_z):
            raise RuntimeError(f"Episode walking floor is non-finite: {floor_z!r}.")
        measured = float(position[2]) - floor_z
        if not math.isfinite(measured):
            raise RuntimeError(
                "Live G1 body height is non-finite: "
                f"pelvis_z={float(position[2])}, floor_z={floor_z}."
            )
        return measured

    def _g1_planning_root_from_world(self) -> np.ndarray:
        """Return the live cuRobo root frame beneath the current pelvis.

        cuRobo models G1 body height with a synthetic vertical joint above a
        root fixed to the current stance. The USD robot root can remain at its
        spawn pose while locomotion moves the articulated pelvis, so the live
        pelvis—not that stale prim—is the authoritative frame.
        """

        pelvis_position, pelvis_orientation = self._live_g1_pelvis_world_pose()
        transform = np.asarray(
            frame_from_world_pose(
                frame_position_world_m=np.asarray(
                    pelvis_position,
                    dtype=np.float64,
                ).reshape(-1),
                frame_orientation_world_wxyz=np.asarray(
                    pelvis_orientation,
                    dtype=np.float64,
                ).reshape(-1),
            ),
            dtype=np.float64,
        )
        transform[:3, 3] += np.asarray(
            [
                0.0,
                0.0,
                g1_curobo_pelvis_model_height_from_body_height_m(
                    self._g1_measured_body_height_m(pelvis_position),
                ),
            ]
        )
        return transform

    def _g1_curobo_joint_positions(self) -> Dict[str, float]:
        """Read the complete live joint state required by G1 cuRobo."""

        measured_joint_names = tuple(
            name for name in G1_CUROBO_JOINT_NAMES if name != "base_j_z"
        )
        robot = self._live_robots()[0]
        articulation = getattr(robot, "articulation", None)
        if articulation is None:
            raise RuntimeError("Cannot plan a G1 grasp without a live articulation.")
        positions = articulation.get_joint_positions()
        names = getattr(articulation, "dof_names", None)
        if positions is None or not names:
            raise RuntimeError("Cannot read the live G1 joint state for cuRobo.")
        name_to_index = {str(name): index for index, name in enumerate(names)}
        missing = [name for name in measured_joint_names if name not in name_to_index]
        if missing:
            raise RuntimeError(
                f"Live G1 articulation is missing cuRobo joints: {missing!r}."
            )
        values = np.asarray(
            positions.cpu() if hasattr(positions, "cpu") else positions,
            dtype=np.float64,
        ).reshape(-1)
        state = {
            name: float(values[name_to_index[name]]) for name in measured_joint_names
        }
        pelvis_position_world_m, _ = self._live_g1_pelvis_world_pose()
        state["base_j_z"] = g1_curobo_pelvis_model_height_from_body_height_m(
            self._g1_measured_body_height_m(pelvis_position_world_m),
        )
        return state

    def _apply_camera_gaze_tilt_decision(self, decision: Dict[str, Any]) -> None:
        """Aim the ego camera before the decision's physics window opens.

        ``tilt_camera`` changes only where the camera looks, so it is applied
        here rather than routed to a controller.  Doing it before physics is what
        makes the action useful: the frame the agent receives at the end of this
        step is already taken from the new angle.
        """
        if str(decision.get("action") or "") != "tilt_camera":
            return
        pitch_deg = decision.get("pitch_deg")
        if pitch_deg is None:
            raise ValueError(f"tilt_camera decision missing pitch_deg: {decision!r}")
        self.set_camera_gaze_tilt_deg(pitch_deg)

    def _world_point_in_robot_frame(
        self, point_world: np.ndarray, *, frame_name: str
    ) -> np.ndarray:
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        root_path = self._robot_root_prim_path()
        if stage is None or not root_path:
            raise RuntimeError("G1 arm IK cannot resolve the live USD robot root.")
        frame_path = f"{root_path}/{frame_name}"
        prim = stage.GetPrimAtPath(frame_path)
        if prim is None or not prim.IsValid():
            raise RuntimeError(f"G1 arm IK frame is missing: {frame_path}")
        world_from_frame = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        frame_from_world = world_from_frame.GetInverse()
        transformed = frame_from_world.Transform(
            Gf.Vec3d(*(float(value) for value in point_world))
        )
        return np.asarray(
            [float(transformed[0]), float(transformed[1]), float(transformed[2])],
            dtype=np.float64,
        )

    def _robot_frame_point_in_world(
        self, point_frame: np.ndarray, *, frame_name: str
    ) -> np.ndarray:
        """Transform one metric point from a live robot frame into world."""

        point = np.asarray(point_frame, dtype=np.float64).reshape(-1)
        if point.size != 3 or not np.all(np.isfinite(point)):
            raise ValueError(f"{frame_name} point must contain three finite values.")
        affine = self._world_to_robot_frame_affine(frame_name)
        linear = np.asarray(affine["linear"], dtype=np.float64)
        offset = np.asarray(affine["offset"], dtype=np.float64)
        if linear.shape != (3, 3) or offset.shape != (3,):
            raise RuntimeError(
                f"Live robot frame {frame_name!r} has a malformed affine."
            )
        try:
            point_world = np.linalg.solve(linear, point - offset)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"Live robot frame {frame_name!r} is not invertible."
            ) from exc
        if not np.all(np.isfinite(point_world)):
            raise RuntimeError(
                f"Live robot frame {frame_name!r} produced a non-finite point."
            )
        return point_world

    def _robot_frame_pose_in_world(self, frame_name: str) -> np.ndarray:
        """Return one live robot-frame pose as a rigid world transform."""

        affine = self._world_to_robot_frame_affine(frame_name)
        world_to_frame = np.eye(4, dtype=np.float64)
        world_to_frame[:3, :3] = np.asarray(
            affine["linear"],
            dtype=np.float64,
        )
        world_to_frame[:3, 3] = np.asarray(
            affine["offset"],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(world_to_frame)):
            raise RuntimeError(
                f"Live robot frame {frame_name!r} has a non-finite pose."
            )
        try:
            frame_to_world = np.linalg.inv(world_to_frame)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"Live robot frame {frame_name!r} is not invertible."
            ) from exc
        if not np.allclose(
            frame_to_world[:3, :3].T @ frame_to_world[:3, :3],
            np.eye(3),
            atol=1e-5,
        ):
            raise RuntimeError(f"Live robot frame {frame_name!r} is not rigid.")
        return frame_to_world

    def _world_direction_in_robot_frame(
        self, direction_world: Sequence[float], *, frame_name: str
    ) -> np.ndarray:
        """Rotate a world unit direction into a robot frame, dropping translation.

        Taken as the difference of two transformed points rather than by pulling
        the rotation out of the matrix, so it uses the one transform path
        :meth:`_world_point_in_robot_frame` already owns and cannot disagree
        with the target position it accompanies.
        """
        direction = np.asarray(direction_world, dtype=np.float64).reshape(-1)
        if direction.size != 3 or not np.all(np.isfinite(direction)):
            raise ValueError(
                "a grasp axis must contain three finite values, got "
                f"{direction_world!r}."
            )
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            raise ValueError("a grasp axis cannot have zero length.")
        origin_frame = self._world_point_in_robot_frame(
            np.zeros(3, dtype=np.float64),
            frame_name=frame_name,
        )
        tip_frame = self._world_point_in_robot_frame(
            direction / norm,
            frame_name=frame_name,
        )
        rotated = tip_frame - origin_frame
        rotated_norm = float(np.linalg.norm(rotated))
        if rotated_norm <= 1e-9:
            raise RuntimeError(
                f"The {frame_name} transform collapsed a grasp axis to zero "
                "length; the live robot frame is not a rigid transform."
            )
        return rotated / rotated_norm

    def _arm_joint_positions(self, hand: str) -> Optional[list[float]]:
        from tarsphere.infra.manipulation.g1_arm_ik import (
            G1_LEFT_ARM_JOINT_NAMES,
            G1_RIGHT_ARM_JOINT_NAMES,
        )

        joint_names = (
            G1_LEFT_ARM_JOINT_NAMES
            if str(hand).lower() == "left"
            else G1_RIGHT_ARM_JOINT_NAMES
        )

        robot = self._live_robots()[0]
        articulation = getattr(robot, "articulation", None)
        if articulation is None:
            return None
        positions = articulation.get_joint_positions()
        if positions is None:
            return None
        names = getattr(articulation, "dof_names", None)
        if not names:
            inner = getattr(articulation, "_articulation", None)
            names = getattr(inner, "dof_names", None)
        if not names:
            return None
        name_to_index = {str(name): index for index, name in enumerate(names)}
        if any(name not in name_to_index for name in joint_names):
            return None
        values = np.asarray(
            positions.cpu() if hasattr(positions, "cpu") else positions
        ).reshape(-1)
        return [float(values[name_to_index[name]]) for name in joint_names]

    def _dex3_joint_positions(self, hand: str) -> list[float]:
        """Read one live Dex3 hand in its seven-joint order."""
        from tarsphere.embodiment.dex3 import (
            DEX3_JOINT_NAMES,
        )

        normalized_hand = str(hand).strip().lower()
        if normalized_hand not in DEX3_JOINT_NAMES:
            raise ValueError(f"Dex3 hand must be left or right, got {hand!r}.")
        joint_names = DEX3_JOINT_NAMES[normalized_hand]
        robot = self._live_robots()[0]
        articulation = getattr(robot, "articulation", None)
        if articulation is None:
            raise RuntimeError(
                "Cannot read Dex3 joints without a live G1 articulation."
            )
        positions = articulation.get_joint_positions()
        names = getattr(articulation, "dof_names", None)
        if positions is None or not names:
            raise RuntimeError("Cannot read Dex3 joints from the live G1 articulation.")
        name_to_index = {str(name): index for index, name in enumerate(names)}
        missing = [name for name in joint_names if name not in name_to_index]
        if missing:
            raise RuntimeError(
                f"Live G1 articulation is missing Dex3 joints: {missing}."
            )
        values = np.asarray(
            positions.cpu() if hasattr(positions, "cpu") else positions
        ).reshape(-1)
        return [float(values[name_to_index[name]]) for name in joint_names]

    def _dex3_tracking_report(
        self,
        hand: str,
        *,
        preset: str,
    ) -> Dict[str, Any]:
        """Report commanded and measured Dex3 coordinates in joint order."""
        from tarsphere.embodiment.dex3 import (
            DEX3_JOINT_NAMES,
            dex3_joint_positions,
        )

        normalized_hand = str(hand).strip().lower()
        joint_names = DEX3_JOINT_NAMES[normalized_hand]
        robot = self._live_robots()[0]
        articulation = getattr(robot, "articulation", None)
        if articulation is None:
            raise RuntimeError(
                "Cannot measure Dex3 tracking without a live articulation."
            )
        names = getattr(articulation, "dof_names", None)
        native_articulation = articulation.unwrap()
        limits = native_articulation._articulation_view.get_dof_limits()[0]
        if not names or limits is None:
            raise RuntimeError(
                "Cannot measure Dex3 tracking without joint names and limits."
            )
        name_to_index = {str(name): index for index, name in enumerate(names)}
        indices = [name_to_index[name] for name in joint_names]
        limit_values = np.asarray(
            limits.cpu() if hasattr(limits, "cpu") else limits,
            dtype=np.float64,
        )
        measured = self._dex3_joint_positions(normalized_hand)
        commanded = dex3_joint_positions(normalized_hand, preset)
        applied_action = native_articulation.get_applied_action()
        applied_positions = getattr(applied_action, "joint_positions", None)
        if applied_positions is None:
            raise RuntimeError(
                "Cannot measure Dex3 tracking without the live applied "
                "joint-position action."
            )
        applied_values = np.asarray(
            (
                applied_positions.cpu()
                if hasattr(applied_positions, "cpu")
                else applied_positions
            ),
            dtype=np.float64,
        ).reshape(-1)
        return {
            "joint_names": list(joint_names),
            "articulation_indices": indices,
            "commanded_positions_rad": list(commanded),
            "applied_positions_rad": [
                float(applied_values[index]) for index in indices
            ],
            "measured_positions_rad": measured,
            "position_errors_rad": [
                float(actual - target) for actual, target in zip(measured, commanded)
            ],
            "limits_rad": [
                [
                    float(limit_values[index, 0]),
                    float(limit_values[index, 1]),
                ]
                for index in indices
            ],
        }

    def _start_room_observer_episode(self) -> None:
        if self._room_observer is None or self.image_dir is None:
            return
        self._room_observer.start_episode(
            self.image_dir,
            route_points_xy=episode_route_points_xy(self.episode),
            policy_camera_path=(
                self._agent_camera_path()
                if "presentation_ego" in self.room_observer_view_names
                else None
            ),
        )

    def configure_continuous_ego_video(
        self,
        *,
        enabled: bool,
        output_dir: Path,
        slug: str,
        capture_stride: int = 8,
        include_bev: bool = True,
    ) -> None:
        """Configure the next reset's synchronized ego/trajectory recording."""

        if self._continuous_ego_video.active:
            raise RuntimeError(
                "Cannot reconfigure continuous ego video during an episode."
            )
        self._continuous_ego_video = ContinuousEgoVideoRecorder(
            capture_stride=capture_stride,
            physics_hz=float(SIMULATION_PHYSICS_HZ),
            resolution=ROBOT_EGO_CAMERA_RESOLUTION,
            include_bev=include_bev,
        )
        self._continuous_ego_video_enabled = bool(enabled)
        self._continuous_ego_video_output_dir = Path(output_dir)
        self._continuous_ego_video_slug = str(slug).strip()
        if self._continuous_ego_video_enabled and not self._continuous_ego_video_slug:
            raise ValueError("Continuous ego video slug must be non-empty.")

    def _start_continuous_ego_video(self) -> None:
        if not self._continuous_ego_video_enabled:
            return
        if self._continuous_ego_video_output_dir is None:
            raise RuntimeError("Continuous ego video output directory is unset.")
        if not self._continuous_ego_video_slug:
            raise RuntimeError("Continuous ego video slug is unset.")
        world_knowledge = getattr(self.episode, "world_knowledge", None)
        if not isinstance(world_knowledge, Mapping):
            raise RuntimeError(
                "Continuous ego trajectory BEV requires episode world_knowledge."
            )
        scene_map = world_knowledge.get("scene_map")
        if not isinstance(scene_map, Mapping):
            raise RuntimeError(
                "Continuous ego trajectory BEV requires world_knowledge.scene_map."
            )
        if not self._agent_camera_path():
            raise RuntimeError("Continuous ego video requires the robot ego camera.")
        snapshot = getattr(self, "_ego_snapshot", None)
        if snapshot is None or int(snapshot.physics_step) != int(self._physics_steps):
            raise RuntimeError(
                "Continuous ego video requires the reset observation snapshot."
            )
        self._continuous_ego_video.start_episode(
            output_dir=self._continuous_ego_video_output_dir,
            slug=self._continuous_ego_video_slug,
            scene_map=scene_map,
            route_points_xy=episode_route_points_xy(self.episode),
        )
        self._record_continuous_ego_snapshot(snapshot)

    def finish_continuous_ego_video(
        self,
    ) -> ContinuousEgoVideoArtifacts | None:
        """Preserve the synchronized terminal view, then close the recording."""

        if not self._continuous_ego_video.active:
            return None
        try:
            self._capture_ego_snapshot(
                capture_observer=False,
                write_image=False,
            )
        except Exception as capture_error:
            try:
                self._continuous_ego_video.finish_episode()
            except Exception as close_error:
                raise RuntimeError(
                    f"Terminal ego capture failed: {capture_error}; "
                    f"video finalization also failed: {close_error}"
                ) from close_error
            raise
        return self._continuous_ego_video.finish_episode()

    def _live_room_observer_pose(self) -> tuple[list[float], float]:
        """Read the live robot pose while a multi-tick action is in flight.

        ``room_observer_focus_human`` redirects the observer onto a Human
        instead.  Off by default and never set on an evaluation path: it exists
        for the Human-animation verification runner
        (for example, an interactive Human-action diagnostic), where the robot is
        irrelevant and its framing leaves the character off-screen.
        """
        focus_human = getattr(self, "room_observer_focus_human", None)
        if focus_human:
            position = self._human_positions.get(str(focus_human))
            if position is None:
                raise RuntimeError(
                    "Room observer cannot read the position of focus human "
                    f"{focus_human!r}; known humans="
                    f"{sorted(self._human_positions)}"
                )
            yaw_rad = float(self._human_yaws.get(str(focus_human), 0.0))
            return [float(value) for value in position], yaw_rad
        if self._env is not None:
            tasks = tuple(self._env.runner.current_tasks.values())
            robots = tuple(tasks[0].robots.values()) if len(tasks) == 1 else ()
            if len(robots) == 1:
                get_pose = getattr(robots[0], "get_pose", None)
                if callable(get_pose):
                    position, orientation = get_pose()
                    position_values = np.asarray(
                        position,
                        dtype=np.float64,
                    ).reshape(-1)
                    quaternion = np.asarray(
                        orientation,
                        dtype=np.float64,
                    ).reshape(-1)
                    if (
                        position_values.size == 3
                        and quaternion.size == 4
                        and np.all(np.isfinite(position_values))
                        and np.all(np.isfinite(quaternion))
                    ):
                        w, x, y, z = quaternion.tolist()
                        yaw_rad = math.atan2(
                            2.0 * (w * z + x * y),
                            1.0 - 2.0 * (y * y + z * z),
                        )
                        return position_values.tolist(), float(yaw_rad)
        position = self._robot_position()
        if position is None:
            raise RuntimeError("Room observer cannot read the live robot position.")
        return position, self._robot_yaw()

    def _begin_decision_render_window(self, step_budget: int) -> None:
        """Apply the configured render mode inside one decision window.

        Agent RGB-D is never taken from these native renders. ``observe()``
        serves one stamped ego snapshot, so extra demo-mode Kit frames cannot
        change ``capture_id`` or any measured field.

        ``evaluation`` withholds cadence renders unless continuous video was
        explicitly enabled. Video reads native frames without advancing physics
        or publishing intermediate policy observations. ``demo`` keeps TARSphere's
        native cadence, clamped to ``room_observer.capture_stride`` when the
        recorder is active, so the video is smooth.

        ``omni.anim.people`` advances BehaviorScript/AnimGraph on Kit updates.
        Evaluation mode therefore does not drive mid-window Human locomotion;
        that conflict is reported to the parent rather than special-cased here.
        """
        if self._env is None or not self.enable_ego_sensors:
            return
        if self._decision_render_state is not None:
            raise RuntimeError("Nested decision render windows are not supported.")
        runner = self._env.runner
        original_interval = int(runner.render_interval)
        original_trigger = int(runner.render_trigger)
        self._decision_render_state = (
            original_interval,
            original_trigger,
            False,
        )
        continuous_ego_video = getattr(self, "_continuous_ego_video", None)
        room_observer = getattr(self, "_room_observer", None)
        room_observer_recording = (
            room_observer is not None
            and self.image_dir is not None
            and room_observer.accepts_captures(self.image_dir)
        )
        recording = (
            continuous_ego_video is not None and continuous_ego_video.active
        ) or room_observer_recording
        if (
            getattr(self, "render_mode", RenderMode.EVALUATION) is RenderMode.EVALUATION
            and not recording
        ):
            window = max(1, int(step_budget))
            runner.render_interval = window + original_trigger + 1
            return
        capture_strides: list[int] = []
        if room_observer_recording:
            capture_strides.append(int(room_observer.capture_stride))
        if continuous_ego_video is not None and continuous_ego_video.active:
            capture_strides.append(int(continuous_ego_video.capture_stride))
        if capture_strides:
            # The recorder reads native Isaac renders and never advances the
            # timeline itself. Match the fastest requested physics-time stride;
            # e.g. 240 Hz / 8 ticks = a true 30 fps video.
            video_interval = min(capture_strides) - 1
            runner.render_interval = (
                video_interval
                if self.render_mode is RenderMode.EVALUATION
                else min(original_interval, video_interval)
            )

    def _capture_private_mapping_tick(
        self,
        action_name: str,
        physics_step: int,
    ) -> None:
        """Sample ego RGB-D on a fixed simulation-time mapping clock.

        These frames update only the private online occupancy grid. They do
        not become decision snapshots, visual history, FrontierNet inputs, or
        policy calls.
        """

        if str(action_name) not in _PRIVATE_MAPPING_LOCOMOTION_ACTIONS:
            return
        if not set(self._private_mapping_action_space).intersection(
            {"navigate_to_frontier", "navigate_to_pointgoal", "navigate_to_entity"}
        ):
            return
        observer = self._private_mapping_observer
        if observer is None:
            raise RuntimeError(
                "Online navigation is active but no private mapping observer "
                "was bound for this episode."
            )
        mapping_hz = float(self.agent_harness_config.controller.online_mapping_hz)
        if mapping_hz <= 0.0:
            return
        interval_ticks = max(1, int(round(float(SIMULATION_PHYSICS_HZ) / mapping_hz)))
        current_step = int(physics_step)
        if current_step - int(self._private_mapping_last_physics_step) < interval_ticks:
            return

        render_started = time.monotonic()
        snapshot = self._capture_ego_snapshot(
            physics_step=current_step,
            capture_observer=False,
            record_continuous=False,
            write_image=False,
        )
        render_seconds = time.monotonic() - render_started
        observation = self._observation_from_ego_snapshot(snapshot)
        lidar = self._capture_lidar()
        if lidar is None:
            raise RuntimeError(
                "Private online mapping could not capture enabled LiDAR "
                f"at physics step {current_step}."
            )
        observation["lidar"] = lidar

        frame_index = int(self._private_mapping_frame_index)
        callback_started = time.monotonic()
        report = observer(
            observation=observation,
            action_space=list(self._private_mapping_action_space),
            physics_step=current_step,
            frame_index=frame_index,
        )
        callback_seconds = time.monotonic() - callback_started
        if not isinstance(report, Mapping):
            raise TypeError(
                "Private mapping observer must return one diagnostics mapping."
            )
        self._private_mapping_last_physics_step = current_step
        self._private_mapping_frame_index = frame_index + 1
        stats = self._private_mapping_stats
        stats["captures"] = int(stats["captures"]) + 1
        stats["integrated_frames"] = int(stats["integrated_frames"]) + int(
            bool(report.get("integrated", False))
        )
        stats["render_seconds"] = float(stats["render_seconds"]) + render_seconds
        fusion_seconds = float(report.get("fusion_seconds", 0.0))
        artifact_seconds = float(report.get("artifact_seconds", 0.0))
        stats["fusion_seconds"] = float(stats["fusion_seconds"]) + fusion_seconds
        stats["artifact_seconds"] = float(stats["artifact_seconds"]) + artifact_seconds
        stats["mapping_callback_seconds"] = (
            float(stats["mapping_callback_seconds"]) + callback_seconds
        )
        area_gain = max(
            0.0,
            float(report.get("observed_area_after_m2", 0.0))
            - float(report.get("observed_area_before_m2", 0.0)),
        )
        stats["observed_area_gain_m2"] = (
            float(stats["observed_area_gain_m2"]) + area_gain
        )
        print(
            "[ONLINE-MAP] motion sample "
            f"frame={frame_index} physics_step={current_step} "
            f"action={action_name} area_gain_m2={area_gain:.3f} "
            f"render_ms={render_seconds * 1000.0:.1f} "
            f"fusion_ms={fusion_seconds * 1000.0:.1f} "
            f"artifact_ms={artifact_seconds * 1000.0:.1f}",
            flush=True,
        )

    def _on_physics_tick(self, timing: str, step_index: int) -> None:
        """Drive room-observer capture around one native physics tick.

        This is not diagnostics: the recorder's prepare/finish/cancel calls
        happen here, and skipping them loses video frames. The opt-in memory
        trace at the end is the only diagnostic part.
        """
        if getattr(self, "human_motion_backend", "kinematic") == "animation":
            if timing in {"after", "error", "interrupted"}:
                import omni.kit.loop._loop as omni_loop

                # Native env.step may render via app.update(). The explicit
                # 60 Hz human pump already accounts for elapsed physics time;
                # a render must not add another animation/timeline step.
                # Restore before readback, including the failed-step path.
                omni_loop.acquire_loop_interface().set_manual_step_size(
                    float(HUMAN_ANIMATION_KIT_STEP_SIZE_S)
                )
        if timing == "interrupted":
            # Normal windows commit after step_simulation returns. A failed
            # window never returns its result, but its successful native
            # steps still belong to this episode and its video clock.
            if type(step_index) is not int or step_index < 0:
                raise ValueError(
                    "Interrupted motion needs a nonnegative completed-tick count."
                )
            self._physics_steps += step_index
            if getattr(self, "human_motion_backend", "kinematic") == "animation":
                self._human_body.set_physics_step(self._physics_steps)
            if self._room_observer is not None:
                self._room_observer.cancel_capture()
            return
        if getattr(self, "human_motion_backend", "kinematic") == "animation":
            tick = self._physics_steps + int(step_index)
            self._human_body.set_physics_step(tick + (timing in {"before", "after"}))
            if timing == "after":
                # Poll native command phases after the actual Kit/physics tick;
                # no render or simulation advance occurs in this state read.
                self._human_body.step(0.0)
                native_trace = getattr(
                    self._human_body, "poll_native_motion_trace", None
                )
                if callable(native_trace):
                    native_trace(observation_boundary="after_physics_tick")
                if self._env is not None and bool(
                    getattr(self._env.runner, "_render", False)
                ):
                    self._poll_human_animation_graph_after_render()
        path = getattr(self, "_execution_path", None)
        if timing == "after" and path is not None:
            position, _ = self._live_g1_pelvis_world_pose()
            path.record_tick(position)
        continuous_ego_video = getattr(self, "_continuous_ego_video", None)
        synchronized_video_tick = (
            timing == "after"
            and self._env is not None
            and bool(self._env.runner._render)
            and continuous_ego_video is not None
            and continuous_ego_video.active
        )
        if synchronized_video_tick:
            # Refresh all attached RenderProducts before reading either camera.
            # Native render alone may leave annotators on an older transform.
            self._force_one_ego_render(capture_observer=False)
        if (
            self._room_observer is not None
            and self.image_dir is not None
            and self._room_observer.accepts_captures(self.image_dir)
        ):
            if self._env is None:
                raise RuntimeError(
                    "Room observer cannot record without the live Isaac runner."
                )
            runner = self._env.runner
            if timing == "before":
                native_render_due = int(runner.render_trigger) + 1 > int(
                    runner.render_interval
                )
                if native_render_due:
                    position, yaw_rad = self._live_room_observer_pose()
                    self._room_observer.prepare_capture(
                        image_dir=self.image_dir,
                        physics_step=(self._physics_steps + int(step_index) + 1),
                        position=position,
                        yaw_rad=yaw_rad,
                    )
            elif timing == "after" and bool(runner._render):
                self._last_room_observer_paths = self._room_observer.finish_capture()
            elif timing == "error":
                self._room_observer.cancel_capture()
        if synchronized_video_tick:
            camera = self._robot_camera_sensor()
            if camera is None:
                raise RuntimeError(
                    "Continuous ego recording lost the robot camera sensor."
                )
            camera_data = camera.get_data()
            if not isinstance(camera_data, Mapping) or camera_data.get("rgba") is None:
                raise RuntimeError(
                    "Continuous ego recording received no native RGBA frame."
                )
            position, yaw_rad = self._live_room_observer_pose()
            continuous_ego_video.capture(
                physics_step=(self._physics_steps + int(step_index) + 1),
                rgba=camera_data["rgba"],
                position=position,
                yaw_rad=yaw_rad,
            )
        # Log the first three tick boundaries and then sample sparsely. Full
        # 240 Hz PTY output perturbs the native allocation stall being traced.
        if int(step_index) <= 2 or int(step_index) % 32 == 0:
            _rollout_memory_diagnostic(
                f"physics.{timing}.{int(step_index)}",
                self._physics_steps + int(step_index),
            )
        if (
            timing == "before"
            and getattr(self, "human_motion_backend", "kinematic") == "animation"
        ):
            import omni.kit.loop._loop as omni_loop

            # Arm only after camera preparation succeeds: an observer error
            # before env.step must not leave the animation clock frozen.
            omni_loop.acquire_loop_interface().set_manual_step_size(0.0)

    def _prepare_final_decision_frame(self) -> None:
        """Force a native Kit render on the last physics tick of a demo window."""
        if self._env is None or self._decision_render_state is None:
            return
        if getattr(self, "render_mode", RenderMode.EVALUATION) is RenderMode.EVALUATION:
            return
        original_interval, original_trigger, _ = self._decision_render_state
        del original_trigger
        runner = self._env.runner
        runner.render_interval = original_interval
        runner.render_trigger = original_interval
        self._decision_render_state = (original_interval, 0, True)

    def _end_decision_render_window(self) -> None:
        """Restore TARSphere rendering even when a physics step raises."""
        if self._env is None or self._decision_render_state is None:
            self._decision_render_state = None
            return
        original_interval, original_trigger, final_prepared = (
            self._decision_render_state
        )
        runner = self._env.runner
        runner.render_interval = original_interval
        if not final_prepared:
            runner.render_trigger = original_trigger
        self._decision_render_state = None

    def _runtime_g1_robot(self) -> Any:
        """Return the sole live G1 instance that owns the proprioceptive state."""

        if self._env is None:
            raise RuntimeError("Cannot sample G1 before the Isaac environment exists.")
        tasks = tuple(self._env.runner.current_tasks.values())
        if len(tasks) != 1:
            raise RuntimeError(
                f"Expected one live task while sampling G1, found {len(tasks)}."
            )
        robots = tuple(tasks[0].robots.values())
        if len(robots) != 1:
            raise RuntimeError(
                f"Expected one live G1 while sampling balance, found {len(robots)}."
            )
        robot = robots[0]
        if not callable(getattr(robot, "get_robot_base", None)):
            raise RuntimeError("Live G1 does not expose its pelvis rigid body.")
        if not callable(getattr(robot, "get_rigid_bodies", None)):
            raise RuntimeError("Live G1 does not expose its rigid-body inventory.")
        return robot

    def _g1_body_dynamics(self) -> G1BodyDynamics:
        """Measure IMU tilt/rate and pelvis height without scene information."""

        robot = self._runtime_g1_robot()
        imu_bodies = tuple(
            body
            for body in robot.get_rigid_bodies()
            if str(getattr(body, "name", "")).endswith("/imu_link")
        )
        if len(imu_bodies) != 1:
            raise RuntimeError(
                f"Expected exactly one G1 imu_link rigid body, found {len(imu_bodies)}."
            )
        _, imu_orientation = imu_bodies[0].get_pose()
        imu_angular_velocity = imu_bodies[0].get_angular_velocity()
        pelvis_position, _ = robot.get_robot_base().get_pose()
        pelvis_position = np.asarray(pelvis_position, dtype=np.float64).reshape(-1)
        if pelvis_position.size != 3 or not np.all(np.isfinite(pelvis_position)):
            raise RuntimeError(
                f"G1 pelvis returned an invalid live position: {pelvis_position!r}"
            )
        return measure_g1_body_dynamics(
            imu_orientation_wxyz=imu_orientation,
            imu_angular_velocity_world_rad_s=imu_angular_velocity,
            pelvis_height_m=float(pelvis_position[2]),
        )

    def _assert_robot_initial_pose(self) -> None:
        """Reject invalid scene bring-up before policy or task scoring begins."""
        robots = self._live_robots()
        if len(robots) != 1:
            raise RuntimeError("Episode initialization requires one live robot.")
        position, _ = robots[0].get_robot_base().get_pose()
        if hasattr(position, "cpu"):
            position = position.cpu().numpy()
        actual = np.asarray(position, dtype=float).reshape(-1)
        expected = np.asarray(self.episode.robot.start_position, dtype=float)
        if (
            actual.shape != (3,)
            or not np.all(np.isfinite(actual))
            or np.linalg.norm(actual - expected) > G1_INITIAL_POSE_TOLERANCE_M
        ):
            raise RuntimeError(
                "Invalid episode initialization: robot moved away from its "
                f"authored spawn during warmup; expected={expected.tolist()}, "
                f"actual={actual.tolist()}. No policy decision is valid."
            )

    def _robot_stability(self) -> Dict[str, float]:
        position = self._robot_position()
        orientation = self._obs.get("orientation")
        if position is None or orientation is None:
            raise RuntimeError(
                "Cannot validate G1 stability without pose and orientation."
            )
        if hasattr(orientation, "cpu"):
            orientation = orientation.cpu().numpy()
        quat = np.asarray(orientation, dtype=float).reshape(-1)
        if quat.shape[0] != 4 or not np.all(np.isfinite(quat)):
            raise RuntimeError(f"Invalid G1 orientation for stability check: {quat!r}")
        norm = float(np.linalg.norm(quat))
        if norm <= 1e-8:
            raise RuntimeError("G1 orientation quaternion has zero norm.")
        w, x, y, z = (quat / norm).tolist()
        del w, z
        up_alignment = float(1.0 - 2.0 * (x * x + y * y))
        tilt_deg = float(math.degrees(math.acos(np.clip(up_alignment, -1.0, 1.0))))
        return {
            "pelvis_z_m": float(position[2]),
            "up_alignment": up_alignment,
            "tilt_deg": tilt_deg,
        }

    def _assert_robot_upright(
        self, *, action_name: str, env_action: Dict[str, Any]
    ) -> None:
        stability = self._robot_stability()
        minimum_valid_pelvis_height_m = (
            G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M - G1_PELVIS_HEIGHT_TRACKING_TOLERANCE_M
        )
        if (
            stability["pelvis_z_m"] >= minimum_valid_pelvis_height_m
            and stability["up_alignment"] >= 0.45
        ):
            return
        monitor = self._robot_contact_monitor
        contacts = monitor.drain_contacts() if monitor is not None else ()
        human_violations = (
            monitor.drain_human_safety_violations() if monitor is not None else ()
        )
        raise RobotFallError(
            action_name=action_name,
            env_action=env_action,
            physics_steps=self._physics_steps,
            stability=stability,
            observation=self.observe(),
            physical_contacts=[contact.as_dict() for contact in contacts],
            physical_contact_details=(
                monitor.public_details(
                    contacts,
                    robot_position=self._robot_position(),
                    robot_yaw_rad=self._robot_yaw(),
                )
                if monitor is not None
                else []
            ),
            human_safety_violations=[v.as_dict() for v in human_violations],
            contact_monitor_snapshot=(
                monitor.snapshot() if monitor is not None else None
            ),
        )

    def set_human_speech(self, message: str, *, speaker: str):
        """Record who said what, as the scene's most recent public utterance.

        ``speaker`` is required: with several Humans in an episode an
        unattributed line is unusable — the agent cannot answer "who asked me
        this?" and neither can another Human. The stored record is the single
        authority; the observation text, the Human ego observations, and the
        world snapshot all render from it.
        """
        speaker = str(speaker).strip()
        if not speaker:
            raise ValueError(
                "set_human_speech requires a speaker; an unattributed "
                "utterance cannot be resolved in a multi-Human episode."
            )
        self._last_utterance = {
            "speaker": speaker,
            "text": str(message),
            "physics_steps": self._physics_steps,
        }

    def apply_human_action(
        self, human_name: str, action: Dict[str, Any]
    ) -> HumanBodyResult:
        """Apply a high-level Human command and hand the result back.

        The result used to be consumed for its ``feedback`` and dropped.  A
        caller that wanted to know whether the command was accepted therefore
        read attributes off ``None`` and silently saw ``executed=False`` even
        for a command the executor had accepted -- which is how a working
        ``sit`` was first reported as not applied.
        """
        command = str(action.get("command") or "")
        if self.human_motion_backend == "animation":
            self._human_body.set_physics_step(self._physics_steps)
        self._human_action_log.append(
            {
                "physics_steps": self._physics_steps,
                "human": human_name,
                "action": dict(action),
            }
        )

        if command == "perform_task":
            from tarsphere.infra.control.human_task_execution import execute_human_task

            result = execute_human_task(
                self.episode, human_name, action, self._human_body.get_state
            )
        else:
            result = self._human_body.apply_action(human_name, action)
        native_trace = getattr(self._human_body, "poll_native_motion_trace", None)
        if callable(native_trace):
            native_trace(
                observation_boundary="after_human_action_admission", force=True
            )
        self._human_action_log[-1].update(
            {
                "executed": result.executed,
                "feedback": result.feedback,
                "position": list(result.state.position)
                if result.state is not None
                else None,
            }
        )
        if result.executed and command in {"speak", "ask", "report_done"}:
            self.set_human_speech(
                str(action.get("message", "")),
                speaker=human_name,
            )
        if result.feedback:
            self._interaction_feedback = result.feedback
        return result

    def get_human_body_replay_log(self) -> list[dict]:
        """Native animation phase changes on the same clock as captured frames."""
        if self.human_motion_backend == "animation":
            return self._human_body.get_replay_log()
        return []

    def evaluate_success(self, check: DiagnosticCheck) -> bool:
        if check.type == "robot_near_human":
            human_id = str(check.params.get("human_id") or "").strip()
            if not human_id:
                raise ValueError("robot_near_human requires params.human_id")
            robot_position = self._robot_position()
            human_position = self._human_positions.get(human_id)
            if robot_position is None or human_position is None:
                return False
            distance = float(
                np.linalg.norm(
                    np.asarray(robot_position[:2]) - np.asarray(human_position[:2])
                )
            )
            return distance <= float(check.params.get("distance", 2.0))
        if check.type == "robot_near_position":
            position = self._robot_position()
            target = check.params.get("position")
            if position is None or target is None:
                return False
            distance = float(
                np.linalg.norm(np.asarray(position[:2]) - np.asarray(target[:2]))
            )
            return distance <= float(check.params.get("distance", 1.0))
        if check.type == "robot_holding_object":
            object_id = str(check.params.get("object_id") or "").strip()
            if not object_id:
                raise ValueError("robot_holding_object requires params.object_id")
            if self._g1_attachment is None:
                return False
            objects = self._g1_attachment.snapshot().get("objects", {})
            state = objects.get(object_id)
            physical_hold = (
                state.get("physical_hold") if isinstance(state, dict) else None
            )
            return bool(
                isinstance(state, dict)
                and isinstance(physical_hold, dict)
                and physical_hold.get("held") is True
                and physical_hold.get("transport_ready") is True
                and physical_hold.get("hand") in {"left", "right"}
            )
        if check.type == "object_supported_by":
            object_id = str(check.params.get("object_id") or "").strip()
            destination_id = str(
                check.params.get("destination_object_id") or ""
            ).strip()
            if not object_id or not destination_id:
                raise ValueError(
                    "object_supported_by requires params.object_id and "
                    "params.destination_object_id"
                )
            tolerance_m = float(check.params.get("distance", 0.10))
            if not math.isfinite(tolerance_m) or tolerance_m <= 0.0:
                raise ValueError(
                    "object_supported_by params.distance must be positive and finite"
                )
            if self._g1_attachment is None:
                return False
            objects = self._g1_attachment.snapshot().get("objects", {})
            target = objects.get(object_id)
            destination = objects.get(destination_id)
            if not isinstance(target, dict) or not isinstance(destination, dict):
                return False
            if (
                target.get("attached")
                or target.get("attachment_joint_present")
                or target.get("attachment_joint_valid")
                or target.get("physically_held")
                or target.get("held_by_hand") in {"left", "right"}
                or target.get("kinematic_enabled") is not False
            ):
                return False
            target_min = np.asarray(target.get("bbox_min"), dtype=float)
            target_max = np.asarray(target.get("bbox_max"), dtype=float)
            destination_min = np.asarray(destination.get("bbox_min"), dtype=float)
            destination_max = np.asarray(destination.get("bbox_max"), dtype=float)
            if any(
                value.shape != (3,) or not np.all(np.isfinite(value))
                for value in (
                    target_min,
                    target_max,
                    destination_min,
                    destination_max,
                )
            ):
                return False
            overlap_xy = np.minimum(target_max[:2], destination_max[:2]) - np.maximum(
                target_min[:2], destination_min[:2]
            )
            return bool(
                np.all(overlap_xy > 0.0)
                and abs(float(target_min[2] - destination_max[2])) <= tolerance_m
            )
        if check.type == "handover_accepted":
            human_id = str(check.params.get("human_id") or "").strip()
            object_id = str(check.params.get("object_id") or "").strip()
            if not human_id:
                raise ValueError("handover_accepted requires params.human_id")
            if not object_id:
                raise ValueError("handover_accepted requires params.object_id")
            return self.handover_was_accepted(
                human_id=human_id,
                object_id=object_id,
            )
        return False

    def live_scene_object_bounds(
        self,
        prim_path: str,
    ) -> tuple[list[float], list[float]]:
        """Return one scene-asset prim's current USD bounds in metres."""
        if self._scene_asset_unit_in_meters is None:
            raise RuntimeError(
                "Live scene-object bounds require a loaded scene unit contract"
            )
        import omni.usd

        from tarsphere.infra.simulation.usd_geometry import world_bounds

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Live scene-object bounds require an attached USD stage")
        return world_bounds(stage, prim_path)

    @property
    def scene_asset_unit_in_meters(self) -> float:
        """Return the loaded scene asset's frozen authored-unit contract."""
        if self._scene_asset_unit_in_meters is None:
            raise RuntimeError(
                "Scene asset units are unavailable before scene creation"
            )
        return self._scene_asset_unit_in_meters

    def close(self, *, skip_kit_cleanup: bool = False):
        """Tear the environment down.

        ``skip_kit_cleanup`` exits Kit without its own cleanup pass, which
        dodges an upstream teardown SIGSEGV in ``libomni.syntheticdata`` that
        can turn a finished rollout into exit code 139. It leaves the process
        unable to build another SimulationApp, so only the caller that owns
        process shutdown may pass it — not any caller that might reach
        ``IsaacAutonomousEnv(...)`` again.
        """
        self._close_robot_contact_monitor()
        self._close_g1_attachment()
        self._restore_episode_asset_adaptation()
        self._restore_physical_scene_preparation()
        self._cleanup_human_ego_cameras()
        if self._room_observer is not None:
            self._room_observer.close()
        self._human_collision_proxy_paths.clear()
        self._human_collision_errors.clear()
        if self._env is not None:
            if skip_kit_cleanup:
                # TARSphere's Env.close() only forwards to this call.
                self._env.simulation_app.close(skip_cleanup=True)
            else:
                self._env.close()
            self._env = None
        self._navmesh = None
        self._navmesh_baked = False

    def _apply_and_settle_episode_assets(
        self,
        zero_speed: Dict[str, Any],
    ) -> None:
        adaptation = adaptation_from_episode(self.episode)
        if adaptation is None:
            return
        if self._episode_asset_adapter is not None:
            raise RuntimeError("episode asset adapter lifecycle is already active")
        if self._scene_asset_unit_in_meters is None:
            raise RuntimeError("episode assets require the loaded scene unit contract")
        incompatible = [
            (item.object_id, item.asset_meters_per_unit)
            for item in adaptation.instances
            if not math.isclose(
                item.asset_meters_per_unit,
                self._scene_asset_unit_in_meters,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ]
        if incompatible:
            raise RuntimeError(
                "G1 attachment currently requires adapted assets to share the "
                "scene object-local unit; received "
                f"{incompatible}, scene={self._scene_asset_unit_in_meters}"
            )
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("cannot apply episode assets: USD stage unavailable")
        adapter = EpisodeAssetAdapter(adaptation)
        self._episode_asset_adapter = adapter
        try:
            adapter.apply(stage)
            for _ in range(adaptation.settle_physics_steps):
                self._obs, _, _, _, _ = self._env.step(action=zero_speed)
                self._physics_steps += 1
                adapter.capture_settle_sample()
            diagnostics = adapter.verify_settled()
        except Exception:
            adapter.restore()
            self._episode_asset_adapter = None
            raise
        print(
            "[INFO] Episode asset adaptation settled: "
            f"adaptation={adaptation.adaptation_id}, "
            f"instances={len(adaptation.instances)}, "
            f"diagnostics={diagnostics}",
            flush=True,
        )

    def _restore_episode_asset_adaptation(self) -> None:
        adapter = self._episode_asset_adapter
        if adapter is None:
            return
        adapter.restore()
        self._episode_asset_adapter = None

    def _apply_physical_scene_preparation(self) -> None:
        preparation = preparation_from_episode(self.episode)
        if preparation is None:
            return
        if self._physical_scene_preparer is not None:
            raise RuntimeError("physical scene preparation lifecycle is already active")
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("physical scene preparation requires a USD stage")
        preparer = PhysicalScenePreparer(preparation)
        self._physical_scene_preparer = preparer
        try:
            diagnostics = preparer.apply(stage)
        except Exception:
            self._restore_physical_scene_preparation()
            raise
        print(
            f"[INFO] Applied physical scene preparation: {diagnostics}",
            flush=True,
        )

    def _restore_physical_scene_preparation(self) -> None:
        preparer = self._physical_scene_preparer
        if preparer is None:
            return
        preparer.restore()
        self._physical_scene_preparer = None

    def _configure_g1_attachment(self) -> None:
        from tarsphere.infra.manipulation.g1_attachment import (
            G1AttachmentExecutor,
            manipulation_objects_from_episode,
        )

        self._close_g1_attachment()
        objects = manipulation_objects_from_episode(self.episode)
        if not objects:
            return
        root_path = self._robot_root_prim_path()
        if not root_path:
            raise RuntimeError("Cannot resolve live G1 robot root for attachment.")
        if self._scene_asset_unit_in_meters is None:
            raise RuntimeError(
                "Cannot configure G1 attachment before the scene asset unit is known."
            )
        executor = G1AttachmentExecutor(
            objects,
            robot_root_path=root_path,
            object_local_unit_in_meters=self._scene_asset_unit_in_meters,
            on_world_authored=self._invalidate_ego_snapshot,
        )
        executor.start()
        self._g1_attachment = executor

    def _close_g1_attachment(self) -> None:
        if self._g1_attachment is None:
            self._g1_carried_hand = None
            self._g1_carry_finger_positions = None
            return
        self._g1_attachment.close()
        self._g1_attachment = None
        self._g1_carried_hand = None
        self._g1_carry_finger_positions = None

    def _configure_robot_contact_monitor(self) -> None:
        from tarsphere.infra.control.robot_contact_monitor import RobotContactMonitor

        self._close_robot_contact_monitor()
        robot_root_path = self._robot_root_prim_path()
        if not robot_root_path:
            raise RuntimeError(
                "Cannot configure PhysX contact evidence without a robot root."
            )
        monitor = RobotContactMonitor(
            robot_root_path=robot_root_path,
            human_proxy_paths=self._human_collision_proxy_paths,
            is_retention_contact=self._is_grasp_retention_contact,
        )
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError(
                "Cannot configure PhysX contact evidence without a USD stage."
            )
        monitor.start(stage)
        self._robot_contact_monitor = monitor
        print(
            "[INFO] Robot native PhysX contact evidence enabled: "
            f"{monitor.snapshot()}.",
            flush=True,
        )

    def _is_grasp_retention_contact(self, robot_path: str, object_path: str) -> bool:
        attachment = self._g1_attachment
        return attachment is not None and attachment.is_retention_contact(
            robot_path,
            object_path,
        )

    def _close_robot_contact_monitor(self) -> None:
        if self._robot_contact_monitor is None:
            return
        self._robot_contact_monitor.close()
        self._robot_contact_monitor = None

    @staticmethod
    def _bake_material_deduped_scene(scene_usd: Path, deduped: Path) -> bool:
        """Delegate to the shared bake used by BEV and every Isaac entrypoint."""
        from tarsphere.infra.simulation.scene_materials import (
            _bake_material_deduped_scene,
        )

        return _bake_material_deduped_scene(scene_usd, deduped)

    @staticmethod
    def _maybe_use_material_deduped_scene(scene_usd: Path) -> Path:
        """Delegate to the shared material-dedup overlay (nav/raw/interaction)."""
        from tarsphere.infra.simulation.scene_materials import (
            ensure_material_deduped_usd,
        )

        return ensure_material_deduped_usd(scene_usd)

    def _create_env(self):
        t_create_start = time.monotonic()
        t0 = time.monotonic()

        asset_root = configure_simulator_assets()
        t_asset = time.monotonic() - t0

        t0 = time.monotonic()
        from tarsphere.simulator.core.config import Config, SimConfig
        from tarsphere.simulator.core.gym_env import Env
        from tarsphere.simulator.extensions import import_extensions
        from tarsphere.embodiment.robot_profiles import (
            get_robot_profile,
            load_simulator_robot_components,
            patch_simulator_robot_observation_layout,
        )

        robot_type = self.episode.robot.robot_type or "G1Robot"
        robot_profile = get_robot_profile(robot_type)
        robot_usd = asset_root / robot_profile.usd_rel_path
        if not robot_usd.is_file():
            raise FileNotFoundError(
                f"Robot USD is missing for {robot_type}: {robot_usd}. "
                "Install the RobotProfile-declared robot USD "
                f"({robot_profile.usd_rel_path})."
            )
        if robot_type == "G1Robot":
            from tarsphere.embodiment.g1_asset import (
                G1_DEX3_DONOR_REL_PATH,
                create_g1_with_gravity_neutral_dex3,
            )

            dex3_donor_usd = asset_root / G1_DEX3_DONOR_REL_PATH
            robot_physics_usd = create_g1_with_gravity_neutral_dex3(
                robot_usd,
                dex3_donor_usd,
            )
        else:
            robot_physics_usd = robot_usd
        RobotCfgCls, robot_camera_cfg, robot_controller_cfgs = (
            load_simulator_robot_components(robot_type)
        )
        from tarsphere.simulator.extensions.configs.tasks import SingleInferenceTaskCfg

        import_extensions()
        patch_simulator_robot_observation_layout(robot_type)
        self._patch_simulator_rep_camera_for_isaac51()
        t_import = time.monotonic() - t0

        scene_path = self._resolve_asset(self.episode.scene.asset_path, asset_root)
        if not scene_path.is_file() and not scene_path.is_dir():
            raise FileNotFoundError(
                f"Real scene USD is missing: {scene_path}\n"
                "Extract the corresponding GRScenes archive so that "
                "GRScenes-100/<scene_group>/scenes exists. "
                "The runner will not load empty.usd instead."
            )

        # Determine the actual USD file to pass.
        if scene_path.is_dir():
            scene_usd = scene_path / "start_result_navigation.usd"
            if not scene_usd.is_file():
                raise FileNotFoundError(
                    f"Scene directory does not contain start_result_navigation.usd: {scene_path}"
                )
        else:
            scene_usd = scene_path

        from tarsphere.infra.simulation.scene_materials import resolve_runtime_scene_usd

        scene_usd = resolve_runtime_scene_usd(scene_usd)
        self._loaded_scene_usd_path = scene_usd.resolve()

        # Record the spawn baked into the config so reset() knows when an episode
        # loaded via load_episode() needs a teleport to a different start.
        self._configured_robot_start = tuple(
            float(v) for v in self.episode.robot.start_position
        )
        self._configured_robot_orientation = self.episode.robot.start_orientation
        camera = None
        sensors: list[Any] = []
        if self.enable_ego_sensors:
            camera = robot_camera_cfg.update(
                name="camera",
                resolution=ROBOT_EGO_CAMERA_RESOLUTION,
                rgba=True,
                depth=True,
                # The evaluated agent observes RGB-D. A full-scene semantic
                # bounding-box annotator is neither agent input nor evaluator
                # evidence and can consume unbounded host memory while first
                # rendering a GRScene. Keep the camera contract minimal and
                # independent of the unrelated NavMesh lifecycle.
                landmarks=False,
                pointcloud=False,
            )
            sensors.append(camera)
        robot = RobotCfgCls(
            name=self.episode.robot.entity_name,
            usd_path=str(robot_physics_usd),
            position=self.episode.robot.start_position,
            orientation=self.episode.robot.start_orientation,
            controllers=robot_controller_cfgs,
            sensors=sensors,
        )

        config = Config(
            simulator=SimConfig(
                physics_dt=1.0 / SIMULATION_PHYSICS_HZ,
                rendering_dt=1.0 / SIMULATION_PHYSICS_HZ,
                # Demo-mode cadence stored on the TARSphere runner.
                # Evaluation windows stretch this so no native render fires
                # between decision boundaries. The base runtime used 100;
                # 10 is the smooth-video demo default.
                rendering_interval=10,
                use_fabric=False,
                headless=self.headless,
                native=False,
                webrtc=self.enable_streaming,
            ),
            task_configs=[
                SingleInferenceTaskCfg(
                    scene_asset_path=str(scene_usd),
                    scene_scale=self.episode.scene.scale,
                    robots=[robot],
                    objects=[],
                    # SingleInferenceTask never terminates on this legacy field.
                    # Keep the optional value aligned with TARSPHERE's sole simulated-
                    # time budget instead of deriving it from decision count.
                    max_steps=self.episode.runtime.max_physics_steps,
                )
            ],
        )
        from tarsphere.infra.simulation.scene_materials import prepare_grscene_materials

        prepare_grscene_materials(scene_path)

        from tarsphere.infra.simulation.isaac_sensors_physics_stub import (
            ensure_sensors_physics_stub,
        )

        ensure_sensors_physics_stub()

        # MDL resolution accesses many small files. Report remote runtime roots
        # before opening the scene; the Kit configuration loads materials asynchronously.
        from tarsphere.infra.simulation.local_stage import warn_on_networked_roots

        warn_on_networked_roots()

        t0 = time.monotonic()
        env = Env(config)
        self._install_native_render_accounting(env)
        from tarsphere.infra.simulation.scene_materials import (
            finalize_loaded_scene_materials,
        )

        finalize_loaded_scene_materials(
            scene_path,
            update_fn=self._pump_kit_updates,
            label="runtime",
        )
        t_env = time.monotonic() - t0
        # USD Python modules become available only after Env has started the
        # Isaac SimulationApp. Read the referenced asset's own unit metadata
        # here, before any attachment executor is configured.
        self._scene_asset_unit_in_meters = _read_scene_asset_unit_in_meters(scene_usd)
        _require_metric_scene_scale(
            self.episode.scene.scale,
            asset_unit_in_meters=self._scene_asset_unit_in_meters,
        )
        print(
            "[INFO] Scene asset unit contract: "
            f"1 authored unit = {self._scene_asset_unit_in_meters:g} m "
            f"and scene.scale={self.episode.scene.scale!r} "
            f"({scene_usd}).",
            flush=True,
        )

        t_total = time.monotonic() - t_create_start
        print(
            f"[PERF] _create_env: asset={t_asset:.1f}s  import={t_import:.1f}s  "
            f"Env(config)={t_env:.1f}s  total={t_total:.1f}s",
            flush=True,
        )
        return env

    def _apply_episode_lighting(self) -> None:
        """Restore scene lighting, then apply this episode's light condition."""

        import omni.usd

        stage = omni.usd.get_context().get_stage()
        rendering = apply_evaluation_scene_rendering(
            stage,
            self.episode.scene.asset_path,
            perceptual_condition=condition_for_episode(self.episode),
        )
        paths = rendering.spatial_lighting_paths
        repair = scene_lighting_repair_for_asset(self.episode.scene.asset_path)
        source_light_adjustment = (
            repair.authored_intensity_scale != 1.0
            or repair.authored_intensity_cap is not None
        )
        if paths or source_light_adjustment:
            print(
                "[INFO] Scene lighting repair applied: "
                f"scene={self.episode.scene.asset_path}, "
                f"source_scale={repair.authored_intensity_scale}, "
                f"source_cap={repair.authored_intensity_cap}, "
                f"fixtures={list(paths)}.",
                flush=True,
            )
        if condition_for_episode(self.episode) == "low_light":
            print(
                "[INFO] Perceptual low-light condition applied: "
                f"film_iso={rendering.render_settings['/rtx/post/tonemap/filmIso']}.",
                flush=True,
            )

    @staticmethod
    def _patch_simulator_rep_camera_for_isaac51() -> None:
        """Patch TARSphere RepCamera to use stable Isaac 5.1 RGB annotators.

        The ``IsaacsimCamera`` sensor implementation still attaches ``LdrColor``.
        In Isaac Sim 5.1 that path can fail during the first robot camera
        creation with ``Unable to write from unknown dtype, kind=f, size=0`` in
        ``omni.syntheticdata``; Kit then often segfaults while shutting down.
        The official Replicator RGB annotator gives the same semantic output
        needed by TARSphere (an RGB frame) and avoids the broken LdrColor activation.
        """
        global _SIMULATOR_REP_CAMERA_PATCHED
        if _SIMULATOR_REP_CAMERA_PATCHED:
            return

        try:
            from tarsphere.simulator.extensions.sensors.rep_camera import RepCamera
        except Exception as exc:
            raise RuntimeError(f"Cannot patch TARSphere RepCamera: {exc}") from exc

        original_restore = RepCamera.restore_sensor_info

        def _patch_isaacsim_camera_class() -> None:
            try:
                from tarsphere.simulator.core.sensor.isaacsim.camera import (
                    IsaacsimCamera,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot import TARSphere IsaacsimCamera for camera patch: {exc}"
                ) from exc

            if getattr(IsaacsimCamera, "_tarsphere_rgb_annotator_patch", False):
                return

            def init_rp_annotators(self):  # type: ignore[no-untyped-def]
                import omni.replicator.core as rep

                if self.rgba:
                    errors: list[str] = []
                    for annotator_name in ("rgb", "rgba"):
                        try:
                            annotator = rep.AnnotatorRegistry.get_annotator(
                                annotator_name
                            )
                            annotator.attach(self.rp)
                            self.rp_annotators["rgba"] = annotator
                            self._tarsphere_rgb_annotator_name = annotator_name
                            break
                        except Exception as exc:
                            errors.append(f"{annotator_name}: {exc}")
                    else:
                        raise RuntimeError(
                            "Failed to attach an Isaac Replicator RGB annotator "
                            f"for robot ego camera. Tried rgb/rgba. Errors: {errors}"
                        )

                if self.distance_to_image_plane:
                    annotator = rep.AnnotatorRegistry.get_annotator(
                        "distance_to_image_plane"
                    )
                    annotator.attach(self.rp)
                    self.rp_annotators["distance_to_image_plane"] = annotator
                if self.bounding_box_2d_tight:
                    annotator = rep.AnnotatorRegistry.get_annotator(
                        "bounding_box_2d_tight"
                    )
                    annotator.attach(self.rp)
                    self.rp_annotators["bounding_box_2d_tight"] = annotator
                if self.camera_params:
                    annotator = rep.AnnotatorRegistry.get_annotator("camera_params")
                    annotator.attach(self.rp)
                    self.rp_annotators["camera_params"] = annotator

            IsaacsimCamera.init_rp_annotators = init_rp_annotators
            IsaacsimCamera._tarsphere_rgb_annotator_patch = True

        def patched_restore(self):  # type: ignore[no-untyped-def]
            _patch_isaacsim_camera_class()
            return original_restore(self)

        RepCamera.restore_sensor_info = patched_restore
        RepCamera._tarsphere_restore_sensor_info_patch = True
        _SIMULATOR_REP_CAMERA_PATCHED = True

    def _enable_anim_people_extensions(self):
        """Enable official Isaac animated-people extensions and settings."""
        try:
            import carb
            import omni.kit.app
        except Exception as exc:
            raise RuntimeError(
                f"Cannot configure omni.anim.people before Isaac Sim is initialized: {exc}"
            ) from exc

        app = omni.kit.app.get_app()
        manager = app.get_extension_manager()
        required_extensions = (
            "omni.kit.scripting",
            "omni.anim.timeline",
            "omni.anim.graph.core",
            "omni.anim.retarget.core",
            "omni.anim.navigation.bundle",
            "omni.anim.navigation.core",
            "omni.anim.people",
            "isaacsim.replicator.agent.core",
            "omni.kit.mesh.raycast",
        )
        if self.human_motion_backend == "kinematic":
            required_extensions = (
                "isaacsim.replicator.agent.core",
                "omni.kit.mesh.raycast",
            )
        for extension in required_extensions:
            try:
                result = manager.set_extension_enabled_immediate(extension, True)
            except Exception as exc:
                raise RuntimeError(
                    f'Failed to enable required Isaac extension "{extension}": {exc}'
                ) from exc
            if result is False:
                raise RuntimeError(
                    f'Isaac extension "{extension}" could not be enabled.'
                )
            if TARSPHERE_ISAAC_DEBUG:
                print(f"[DEBUG] Extension enabled: {extension}", flush=True)

        settings = carb.settings.get_settings()
        asset_root = tarsphere_asset_path("characters", "IsaacPeople")
        if not asset_root.is_dir():
            raise FileNotFoundError(
                f"Isaac People character assets are missing: {asset_root}"
            )
        if self.human_motion_backend == "kinematic":
            settings.set(
                "/exts/isaacsim.replicator.agent/characters_parent_prim_path",
                "/World/Characters",
            )
            self._human_body.set_character_assets_path(str(asset_root))
            return
        from tarsphere.humans.runtime_capabilities import (
            resolve_isaac_people_runtime_assets,
        )

        runtime_assets = resolve_isaac_people_runtime_assets(asset_root)
        biped_path = runtime_assets.biped_setup_usd

        settings.set("/exts/omni.anim.people/navigation_settings/navmesh_enabled", True)
        settings.set(
            "/exts/omni.anim.people/navigation_settings/dynamic_avoidance_enabled", True
        )
        # Set the official arrival tolerance explicitly so Isaac People, the
        # body pose poller, actor scheduler, and oracle all consume the same
        # contract instead of relying on independent defaults.
        settings.set(
            "/exts/omni.anim.people/final_target_distance",
            HUMAN_GOTO_ARRIVAL_TOLERANCE_M,
        )
        settings.set(
            "/persistent/exts/omni.anim.people/character_prim_path", "/World/Characters"
        )
        settings.set(
            "/persistent/exts/omni.anim.people/asset_settings/character_assets_path",
            str(asset_root),
        )
        settings.set(
            "/exts/isaacsim.replicator.agent/asset_settings/default_biped_assets_path",
            str(biped_path),
        )
        settings.set(
            "/exts/isaacsim.replicator.agent/asset_settings/fallback_biped_assets_path",
            str(biped_path),
        )
        settings.set(
            "/exts/isaacsim.replicator.agent/asset_settings/default_character_asset_path",
            str(asset_root),
        )
        settings.set(
            "/exts/isaacsim.replicator.agent/asset_settings/fallback_character_asset_path",
            str(asset_root),
        )
        settings.set(
            "/exts/isaacsim.replicator.agent/characters_parent_prim_path",
            "/World/Characters",
        )
        if callable(getattr(self._human_body, "set_character_assets_path", None)):
            self._human_body.set_character_assets_path(str(asset_root))

    def _configure_anim_people_navigation(self):
        """Build the NavMesh required by omni.anim.people GoTo commands.

        Creates a NavMeshVolume, configures scene meshes for official
        NavMesh baking, then refreshes the SDG pipeline after PhysX re-attach.

        The NavMesh bake is the slowest phase (~500 s for a full GRScene).
        Once baked successfully within an Isaac session the result persists
        on the USD stage, so subsequent ``reset()`` calls can skip the bake
        entirely and only refresh the runtime handles.
        """
        if self._navmesh_baked:
            print(
                "[INFO] NavMesh already baked this session — skipping full bake.",
                flush=True,
            )
            # In the persistent worker the runner just advanced to the next
            # episode config, which clears + reloads /World/env_0 (new robot,
            # re-referenced scene) and re-attaches PhysX — invalidating the SDG
            # pipeline.  Refresh robot handles and camera sensors so the new
            # robot's RGBD works on the rebuilt scene.  (The baked NavMesh is
            # untouched: it lives in the navigation core and the NavMeshVolume is
            # at /World/, outside /World/env_0.)
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            self._refresh_simulator_runtime_after_navmesh_attach(stage)
            return
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.anim.navigation.bundle")
        enable_extension("omni.anim.navigation.core")
        for _ in range(30):
            self._pump_kit_updates(count=1)
        print(
            "[INFO] NavMesh extensions force-enabled via enable_extension().",
            flush=True,
        )

        import carb
        import NavSchema
        import omni.anim.navigation.core as nav
        import omni.usd
        from pxr import Sdf, Usd, UsdGeom

        usd_context = omni.usd.get_context()
        stage = usd_context.get_stage()
        if stage is None:
            raise RuntimeError(
                "USD stage is unavailable; cannot configure anim_people navigation."
            )

        # Ensure edits go to the root layer (not session layer) so the C++
        # NavMesh baker can discover the NavMeshVolume prim.
        root_layer = stage.GetRootLayer()
        with Usd.EditContext(stage, root_layer):
            center, scale = self._anim_people_navmesh_volume_transform(stage)
            settings = carb.settings.get_settings()
            world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
            if not stage.HasDefaultPrim() and world and world.IsValid():
                stage.SetDefaultPrim(world)

            preferred_volume = stage.GetPrimAtPath(ANIM_PEOPLE_NAVMESH_VOLUME_PATH)
            if (
                preferred_volume
                and preferred_volume.IsValid()
                and preferred_volume.GetTypeName() == "NavMeshVolume"
            ):
                volume = preferred_volume
                created_volume = False
            else:
                volume = NavSchema.NavMeshVolume.Define(
                    stage,
                    ANIM_PEOPLE_NAVMESH_VOLUME_PATH,
                ).GetPrim()
                created_volume = True
            if (
                not volume
                or not volume.IsValid()
                or volume.GetTypeName() != "NavMeshVolume"
            ):
                raise RuntimeError(
                    f"Could not create {ANIM_PEOPLE_NAVMESH_VOLUME_PATH} for "
                    "omni.anim.people navigation."
                )
            volume_schema = NavSchema.NavMeshVolume(volume)
            volume_type_attr = volume_schema.GetNavVolumeTypeAttr()
            if volume_type_attr:
                volume_type_attr.Set("Include")
            NavSchema.NavMeshAreaAPI.Apply(volume)
            area_attr = volume.GetAttribute("nav:area")
            if not area_attr:
                area_attr = volume.CreateAttribute(
                    "nav:area",
                    Sdf.ValueTypeNames.String,
                    custom=False,
                )
            area_attr.Set("Walkable")
            self._set_navmesh_volume_transform(volume, center, scale)
            print(
                f"[INFO] {'Created' if created_volume else 'Configured'} "
                "Isaac NavMeshVolume for omni.anim.people: "
                f"path={volume.GetPath()}, center={center}, scale={scale}",
                flush=True,
            )
            vol_min = [center[i] - scale[i] / 2 for i in range(3)]
            vol_max = [center[i] + scale[i] / 2 for i in range(3)]
            print(
                f"[INFO] NavMeshVolume world bounds: "
                f"X=[{vol_min[0]:.3f}, {vol_max[0]:.3f}] "
                f"Y=[{vol_min[1]:.3f}, {vol_max[1]:.3f}] "
                f"Z=[{vol_min[2]:.3f}, {vol_max[2]:.3f}]",
                flush=True,
            )
            settings.set(
                "/exts/omni.anim.people/navigation_settings/navmesh_enabled", True
            )
            self._configure_navmesh_bake_settings(settings, nav.NavMeshSettings, stage)
            self._configure_navmesh_bake_meshes(stage, center, scale)

            # Pump updates so the C++ baker can discover the newly created
            # NavMeshVolume. The working standalone test pumps 30 updates after
            # volume creation before baking.
            self._pump_kit_updates(count=30)
            print(
                "[INFO] Pumped 30 kit updates after NavMeshVolume setup.",
                flush=True,
            )

            # Stage stability check: count UsdGeom.Mesh prims before bake.
            mesh_count = sum(1 for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
            print(
                f"[INFO] Pre-bake stage: {mesh_count} UsdGeom.Mesh prims.",
                flush=True,
            )

        # The baker needs the stage-attach event to discover its volume and meshes.
        self._attach_current_stage_for_navmesh(stage)

        debug_stage_path = os.environ.get("TARSPHERE_DEBUG_NAVMESH_STAGE", "").strip()
        if debug_stage_path:
            try:
                stage.GetRootLayer().Export(debug_stage_path)
                print(
                    f"[INFO] Exported NavMesh debug stage to {debug_stage_path}.",
                    flush=True,
                )
            except Exception as exc:
                print(f"[WARN] Could not export NavMesh debug stage: {exc}", flush=True)

        # The C++ NavMesh baker requires the Kit event loop to properly
        # register NavMeshVolume prims created on the USD stage.  The
        # official omni.anim.navigation.core tests and
        # isaacsim.replicator.agent.core/simulation.py both use async
        # bake.  We replicate that pattern here.
        import asyncio

        async def _async_bake():
            navigation = nav.acquire_interface()
            # Clear any stale cache that might interfere with the bake.
            try:
                navigation.clear_cache_dir()
                print("[INFO] Cleared NavMesh cache before bake.", flush=True)
            except Exception as exc:
                print(f"[WARN] clear_cache_dir failed (non-fatal): {exc}", flush=True)
            print("[INFO] Starting NavMesh bake for omni.anim.people.", flush=True)
            try:
                result = self._bake_navmesh_with_updates(navigation)
            except Exception as exc:
                raise RuntimeError(
                    f"NavMesh bake raised an exception for omni.anim.people: {exc}"
                ) from exc
            # The C++ baker discovers NavMesh volumes through stage change
            # notifications.  On first bake attempt it may report "lack of
            # NavMesh volumes" because the notification hasn't propagated yet.
            # Retry after pumping additional updates.
            if result[1] is None:
                print(
                    "[WARN] First bake attempt returned no NavMesh. "
                    "Pumping 120 additional updates for volume discovery...",
                    flush=True,
                )
                self._pump_kit_updates(count=120)
                try:
                    result = self._bake_navmesh_with_updates(navigation)
                except Exception as exc:
                    raise RuntimeError(
                        f"NavMesh bake retry raised an exception: {exc}"
                    ) from exc
            return result

        future = asyncio.ensure_future(_async_bake())
        # Pump the Kit event loop until the async bake completes.
        max_wait = 300  # seconds
        deadline = time.monotonic() + max_wait
        while not future.done() and time.monotonic() < deadline:
            self._pump_kit_updates(count=1)
        if not future.done():
            future.cancel()
            raise RuntimeError(f"NavMesh bake did not complete within {max_wait}s.")
        bake_result, navmesh, triangle_count = future.result()
        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)

        # Detailed diagnostic logging regardless of outcome.
        if navmesh is None:
            print(
                "[ERROR] NavMesh bake FAILED — get_navmesh() returned None. "
                f"bake_result={bake_result}, meters_per_unit={meters_per_unit}, "
                f"volume_path={volume.GetPath()}, center={center}, scale={scale}.",
                flush=True,
            )
        else:
            print(
                f"[INFO] NavMesh object obtained: triangles={triangle_count}, "
                f"bake_result={bake_result}, meters_per_unit={meters_per_unit}.",
                flush=True,
            )

        if bake_result is False or navmesh is None or triangle_count <= 0:
            raise RuntimeError(
                "NavMesh bake completed but produced no usable NavMesh for "
                "omni.anim.people GoTo commands. "
                f"bake_result={bake_result}, triangles={triangle_count}, "
                f"volume_path={volume.GetPath()}, center={center}, scale={scale}, "
                f"meters_per_unit={meters_per_unit}."
            )
        print(
            "[INFO] NavMesh bake succeeded for omni.anim.people: "
            f"triangles={triangle_count}, meters_per_unit={meters_per_unit}",
            flush=True,
        )
        self._navmesh_baked = True
        self._navmesh = navmesh

        # Validate NavMesh usability with a query point at the volume center.
        try:
            target = carb.Float3(float(center[0]), float(center[1]), 0.0)
            closest, island = navmesh.query_closest_point(
                target=target,
                agent_radius=HUMAN_NAVMESH_AGENT_RADIUS_M,
            )
            if closest:
                print(
                    f"[INFO] NavMesh query validation: "
                    f"query=({center[0]:.2f}, {center[1]:.2f}, 0.0) "
                    f"closest=({closest.x:.3f}, {closest.y:.3f}, {closest.z:.3f}) "
                    f"island={island}",
                    flush=True,
                )
            else:
                print(
                    "[WARN] NavMesh query_closest_point returned None at volume "
                    f"center ({center[0]:.2f}, {center[1]:.2f}). NavMesh may be "
                    "incomplete.",
                    flush=True,
                )
        except Exception as exc:
            print(f"[WARN] NavMesh query validation failed: {exc}", flush=True)

        # The repair below exists solely to undo the damage the stage attach
        # does: the USD attach invalidates the PhysX scene, and the PhysX
        # re-attach then corrupts the Replicator SDG pipeline.  On the fast path
        # no attach happens, so there is nothing to repair -- and running it
        # anyway fails ("Failed to get rigid body velocities", verified
        # 2026-08-19) because it tears down handles that are still valid.
        if getattr(IsaacAutonomousEnv, "_navmesh_attach_performed", False):
            self._refresh_simulator_runtime_after_navmesh_attach(stage)
            IsaacAutonomousEnv._navmesh_attach_performed = False
        else:
            print(
                "[INFO] No stage attach was performed for the NavMesh bake; "
                "TARSphere PhysX handles and camera sensors are still "
                "valid, skipping the post-attach repair entirely.",
                flush=True,
            )

    def navmesh_projection(self, point: Any) -> Dict[str, Any]:
        """Project one authored point onto the live AnimPeople NavMesh."""
        if not self._navmesh_baked or self._navmesh is None:
            raise RuntimeError(
                "AnimPeople NavMesh is unavailable; bake the scene before "
                "querying Human route points."
            )
        if not self._is_xyz_point(point):
            raise ValueError(f"NavMesh query point must be XYZ-like: {point!r}")
        import carb

        query = [
            float(point[0]),
            float(point[1]),
            float(point[2] if len(point) > 2 else 0.0),
        ]
        closest, island = self._navmesh.query_closest_point(
            target=carb.Float3(*query),
            agent_radius=HUMAN_NAVMESH_AGENT_RADIUS_M,
        )
        if closest is None:
            raise RuntimeError(f"NavMesh has no closest point for {query!r}.")
        projected = [float(closest.x), float(closest.y), float(closest.z)]
        return {
            "query": query,
            "closest": projected,
            "projection_distance_xy_m": float(
                math.hypot(projected[0] - query[0], projected[1] - query[1])
            ),
            "island": int(island),
            "agent_radius_m": HUMAN_NAVMESH_AGENT_RADIUS_M,
        }

    def navmesh_route(self, start: Any, target: Any) -> Dict[str, Any]:
        """Query the exact live AnimPeople path contract for one Human GoTo."""
        if not self._navmesh_baked or self._navmesh is None:
            raise RuntimeError(
                "AnimPeople NavMesh is unavailable; bake the scene before "
                "querying Human routes."
            )
        if not self._is_xyz_point(start) or not self._is_xyz_point(target):
            raise ValueError(
                f"Human NavMesh route must contain XYZ-like endpoints: "
                f"{start!r} -> {target!r}"
            )
        import carb

        start_xyz = [float(start[index]) for index in range(3)]
        target_xyz = [float(target[index]) for index in range(3)]
        path = self._navmesh.query_shortest_path(
            carb.Float3(*start_xyz),
            carb.Float3(*target_xyz),
            agent_radius=HUMAN_NAVMESH_AGENT_RADIUS_M,
        )
        raw_points = list(path.get_points()) if path is not None else []

        def _xyz(point: Any) -> list[float]:
            try:
                return [float(point[index]) for index in range(3)]
            except (TypeError, IndexError):
                return [float(point.x), float(point.y), float(point.z)]

        points = [_xyz(point) for point in raw_points]
        endpoint = points[-1] if points else None
        endpoint_distance = (
            math.inf
            if endpoint is None
            else float(
                math.hypot(endpoint[0] - target_xyz[0], endpoint[1] - target_xyz[1])
            )
        )
        return {
            "start": start_xyz,
            "target": target_xyz,
            "agent_radius_m": HUMAN_NAVMESH_AGENT_RADIUS_M,
            "point_count": len(points),
            "endpoint": endpoint,
            "endpoint_distance_xy_m": endpoint_distance,
            "reachable": bool(
                points and endpoint_distance <= HUMAN_GOTO_ARRIVAL_TOLERANCE_M
            ),
        }

    def validate_human_navmesh_contract(
        self, episode: TARSphereEpisode
    ) -> list[Dict[str, Any]]:
        """Validate every authored Human GoTo against the baked NavMesh.

        Static occupancy is only a coarse authoring layer.  Release validity
        requires each Human start and target to project within the official
        arrival tolerance and to belong to the same live NavMesh island.
        """
        goto_actions: list[tuple[str, str, Any]] = []
        evaluation = episode.evaluation
        for milestone in evaluation.milestones if evaluation else ():
            for action in milestone.event.human_actions:
                if str(action.get("command") or "") != "goto":
                    continue
                goto_actions.append(
                    (
                        milestone.milestone_id,
                        str(action.get("human") or action.get("actor") or ""),
                        action.get("target"),
                    )
                )
        if not goto_actions:
            return []

        humans_by_alias: Dict[str, Any] = {}
        for human in getattr(episode, "humans", []) or []:
            humans_by_alias[str(human.id)] = human

        evidence: list[Dict[str, Any]] = []
        failures: list[Dict[str, Any]] = []
        current_positions = {
            str(human.id): list(human.start_position)
            for human in getattr(episode, "humans", []) or []
        }
        for milestone_id, human_alias, raw_target in goto_actions:
            human = humans_by_alias.get(human_alias)
            if human is None:
                raise ValueError(
                    f"Human GoTo references unknown actor {human_alias!r} at "
                    f"milestone {milestone_id!r}."
                )
            target = resolve_human_action_point(episode, raw_target)
            route_start = current_positions[str(human.id)]
            start_report = self.navmesh_projection(route_start)
            target_report = self.navmesh_projection(target)
            # Authored Human poses may sit just outside the walkable surface
            # while remaining inside the official arrival tolerance.  The
            # path query itself requires points on the NavMesh, so validate
            # the authored-to-mesh offsets below and query connectivity
            # between their canonical projections.
            route_report = self.navmesh_route(
                start_report["closest"],
                target_report["closest"],
            )
            same_island = start_report["island"] == target_report["island"]
            within_arrival_closure = (
                start_report["projection_distance_xy_m"]
                <= HUMAN_GOTO_ARRIVAL_TOLERANCE_M
                and target_report["projection_distance_xy_m"]
                <= HUMAN_GOTO_ARRIVAL_TOLERANCE_M
            )
            report = {
                "milestone_id": milestone_id,
                "human_id": str(human.id),
                "human_name": str(getattr(human, "name", "") or human.id),
                "target_ref": raw_target,
                "start": start_report,
                "target": target_report,
                "route": route_report,
                "same_island": same_island,
                "arrival_tolerance_m": HUMAN_GOTO_ARRIVAL_TOLERANCE_M,
                "passed": bool(
                    same_island and within_arrival_closure and route_report["reachable"]
                ),
            }
            evidence.append(report)
            if not report["passed"]:
                failures.append(report)
            current_positions[str(human.id)] = list(target)
        if failures:
            raise RuntimeError(
                "HUMAN_NAVMESH_CONTRACT_FAILED: authored Human GoTo cannot "
                "close under the live baked NavMesh and official arrival "
                f"tolerance; findings={failures!r}"
            )
        return evidence

    def _refresh_simulator_runtime_after_navmesh_attach(self, stage: Any) -> None:
        """Refresh TARSphere robot handles after NavMesh USD stage attach.

        The navigation core needs ``attach_stage_with_callback(stage_id)`` to
        discover runtime-created NavMeshVolume prims.  That USD attach
        invalidates the PhysX active scene, so PhysX must be re-attached.
        PhysX re-attach corrupts the Replicator SDG pipeline (Og graph
        nodes point to the destroyed C++ physics scene).  We fix this by
        destroying the corrupt SDG pipeline prim and re-creating camera
        sensors, which triggers Replicator to rebuild the pipeline on the
        new physics scene.
        """
        runner = getattr(self._env, "runner", None)
        world = getattr(runner, "_world", None)
        if world is None:
            raise RuntimeError(
                "TARSphere World is unavailable after NavMesh bake; cannot "
                "refresh robot handles for AnimPeople navigation."
            )

        try:
            physics_sim_view = getattr(world, "physics_sim_view", None)
            if physics_sim_view is None:
                from isaacsim.core.simulation_manager import SimulationManager

                SimulationManager._on_stop("tarsphere_navmesh_refresh")
                self._reattach_physx_stage_for_runtime(stage)
                SimulationManager._create_simulation_view("tarsphere_navmesh_refresh")
                physics_sim_view = getattr(world, "physics_sim_view", None)
            if physics_sim_view is None:
                raise RuntimeError("World.physics_sim_view is None after refresh.")
            self._refresh_simulator_robot_rigid_body_handles(runner, physics_sim_view)
            self._pump_kit_updates(count=3)

            # PhysX re-attach corrupts the Replicator SDG Og graph.
            # Destroy the C++ Og graph nodes, reset SyntheticData, and
            # remove USD prims so camera sensor recreation builds a
            # fresh graph on the new physics scene.
            self._destroy_stale_sdg_pipeline(stage)
            # Brief pump to let the Og runtime finalize graph destruction
            # before we create new render products.
            self._pump_kit_updates(count=10)

            # Re-create camera sensors with fresh render products and
            # annotators connected to the new physics scene.
            self._refresh_simulator_camera_sensors(runner)
            if self.enable_ego_sensors:
                self._configure_robot_ego_camera()
                self._validate_dualvln_ego_camera_contract()
                self._pump_kit_updates(count=10)

            # Finalize the TARSphere scene LAST, after all kit updates,
            # because timeline stop/play callbacks can clear World._scene.
            runner_scene = getattr(runner, "_scene", None)
            if runner_scene is not None and callable(
                getattr(runner_scene, "unwrap", None)
            ):
                scene = runner_scene.unwrap()
                world._scene = scene  # noqa: SLF001
                scene._finalize(physics_sim_view)  # noqa: SLF001
                world._task_scene_built = True  # noqa: SLF001
            elif hasattr(world, "_scene"):
                world.scene._finalize(physics_sim_view)  # noqa: SLF001
            else:
                from isaacsim.core.api.scenes.scene import Scene

                world._scene = Scene()  # noqa: SLF001
                world._scene._finalize(physics_sim_view)  # noqa: SLF001
                world._task_scene_built = True  # noqa: SLF001
        except Exception as exc:
            raise RuntimeError(
                "NavMesh bake succeeded, but refreshing TARSphere runtime handles "
                f"failed before robot warmup: {exc}"
            ) from exc

        print(
            "[INFO] Refreshed TARSphere runtime handles after NavMesh bake.",
            flush=True,
        )

    @staticmethod
    def _refresh_simulator_robot_rigid_body_handles(
        runner: Any,
        physics_sim_view: Any,
    ) -> None:
        """Refresh robot link wrappers without recreating camera sensors."""
        for task in getattr(runner, "current_tasks", {}).values():
            for robot in getattr(task, "robots", {}).values():
                clean_stale = getattr(robot, "clean_stale_rigid_bodies", None)
                if callable(clean_stale):
                    clean_stale()
                rigid_body_map = getattr(robot, "_rigid_body_map", None)
                if isinstance(rigid_body_map, dict):
                    rigid_body_map.clear()
                create_rigid_bodies = getattr(robot, "create_rigid_bodies", None)
                if callable(create_rigid_bodies):
                    create_rigid_bodies()
                for rigid_body in getattr(robot, "_rigid_body_map", {}).values():
                    unwrap = getattr(rigid_body, "unwrap", None)
                    rigid_prim = unwrap() if callable(unwrap) else None
                    initialize = getattr(rigid_prim, "initialize", None)
                    if callable(initialize):
                        initialize(physics_sim_view)

                prim_path = str(
                    getattr(getattr(robot, "config", None), "prim_path", "") or ""
                )
                rigid_body_map = getattr(robot, "_rigid_body_map", {})
                if prim_path:
                    base = rigid_body_map.get(f"{prim_path}/pelvis")
                    right_ankle = rigid_body_map.get(f"{prim_path}/right_ankle_link")
                    left_ankle = rigid_body_map.get(f"{prim_path}/left_ankle_link")
                    if base is not None:
                        robot._robot_base = base
                    if right_ankle is not None:
                        robot._robot_right_ankle = right_ankle
                    if left_ankle is not None:
                        robot._robot_left_ankle = left_ankle
                # Re-initialize the articulation view before set_gains().
                # PhysX re-attach invalidates the internal _physics_view.
                # robot.articulation is IsaacsimArticulation wrapper;
                # its ._articulation is the Isaac Sim SingleArticulation.
                articulation_wrapper = getattr(robot, "articulation", None)
                if articulation_wrapper is not None:
                    inner_art = getattr(articulation_wrapper, "_articulation", None)
                    if inner_art is not None:
                        art_init = getattr(inner_art, "initialize", None)
                        if callable(art_init):
                            art_init(physics_sim_view)
                set_gains = getattr(robot, "set_gains", None)
                if callable(set_gains):
                    set_gains()

    @staticmethod
    def _destroy_stale_sdg_pipeline(stage: Any) -> None:
        """Completely destroy the stale SDG pipeline after PhysX re-attach.

        When PhysX detaches and re-attaches the stage, the Replicator
        SDG pipeline's Og (OmniGraph) nodes reference the destroyed
        C++ physics scene.  Simply removing the USD prims is **not
        enough** — the C++ Og runtime keeps the graph registered in
        memory.  We must:

        1. Destroy every node in the C++ Og graph via
           ``graph.destroy_node()`` so the runtime unregisters them.
        2. Reset the ``SyntheticData`` Python caches.
        3. Remove the USD prims from both root and session layers.

        Only then can a new render product create a fresh SDG pipeline
        at ``/Render/PostProcess/SDGPipeline``.
        """
        from pxr import Usd

        # ── Step 1: Destroy C++ Og graph nodes ────────────────────────
        og_nodes_destroyed = 0
        try:
            import omni.graph.core as og
        except ImportError as exc:
            raise RuntimeError(
                f"omni.graph.core is required for SDG pipeline cleanup: {exc}"
            ) from exc

        graph_path = "/Render/PostProcess/SDGPipeline"
        try:
            graph = og.get_graph_by_path(graph_path)
        except Exception as exc:
            print(f"[WARN] Cannot look up Og graph: {exc}", flush=True)
            graph = None

        if graph is not None:
            try:
                is_valid = graph.is_valid()
            except Exception as exc:
                print(f"[WARN] Og graph.is_valid() failed: {exc}", flush=True)
                is_valid = False

            if is_valid:
                for node in list(graph.get_nodes()):
                    try:
                        if node.is_valid():
                            # omni.graph.core.Graph.destroy_node signature is
                            # destroy_node(node_path, update_usd) -- NOT modify_usd.
                            graph.destroy_node(
                                node.get_prim_path(),
                                update_usd=True,
                            )
                            og_nodes_destroyed += 1
                    except Exception as exc:
                        print(
                            f"[WARN] Failed to destroy Og node: {exc}",
                            flush=True,
                        )
                # NOTE: a prior block tried graph.get_graph() to destroy the
                # orchestration graph, but omni.graph.core.Graph has no
                # get_graph() (it lives on GraphContext/Node) -- the call always
                # raised.  Per-node destroy above + the USD prim removal in
                # Step 3 already fully clean the pipeline, so the block was
                # removed.

        # ── Step 2: Reset SyntheticData Python caches ─────────────────
        try:
            from omni.syntheticdata import SyntheticData

            SyntheticData.Get().reset(usd=False)
        except Exception as exc:
            print(
                f"[WARN] SyntheticData.reset() failed (non-fatal): {exc}",
                flush=True,
            )

        # ── Step 3: Remove USD prims from both layers ─────────────────
        removed = []
        for path_str in (
            "/Render/PostProcess/SDGPipeline",
            "/Render/PostProcess",
            "/Render/OrmIsolatedLighting",
        ):
            prim = stage.GetPrimAtPath(path_str)
            if prim and prim.IsValid():
                stage.RemovePrim(path_str)
                removed.append(path_str)
        # Also remove from session layer (where Og stores graph state).
        session_layer = stage.GetSessionLayer()
        with Usd.EditContext(stage, session_layer):
            for path_str in (
                "/Render/PostProcess/SDGPipeline",
                "/Render/PostProcess",
            ):
                prim = stage.GetPrimAtPath(path_str)
                if prim and prim.IsValid():
                    stage.RemovePrim(path_str)
                    removed.append(f"{path_str} (session)")
        print(
            f"[INFO] Destroyed stale SDG pipeline: "
            f"og_nodes_destroyed={og_nodes_destroyed}, "
            f"prims_removed={removed}",
            flush=True,
        )

    def _refresh_simulator_camera_sensors(self, runner: Any) -> None:
        """Re-create Replicator render products and annotators from scratch.

        After PhysX re-attach + SDG pipeline destruction, every camera
        sensor must be rebuilt with fresh render products.  The Og graph
        has already been torn down by ``_destroy_stale_sdg_pipeline()``,
        so we:

        1. Try ``cleanup()`` on the old camera (best-effort — may fail
           because the backing Og graph is gone).
        2. If cleanup fails, manually nullify internal state so
           ``restore_sensor_info()`` can proceed.
        3. Call ``restore_sensor_info()`` which creates a brand-new
           ``IsaacsimCamera`` with a fresh render product and annotators
           connected to the new physics scene.
        """
        refreshed = 0
        for task in getattr(runner, "current_tasks", {}).values():
            for robot in getattr(task, "robots", {}).values():
                for sensor in getattr(robot, "sensors", {}).values():
                    # Step 1: Best-effort cleanup of old camera.
                    internal_cam = getattr(sensor, "_camera", None)
                    if internal_cam is not None:
                        try:
                            cleanup_fn = getattr(internal_cam, "cleanup", None)
                            if callable(cleanup_fn):
                                cleanup_fn()
                        except Exception as exc:
                            # Cleanup failed (Og graph already destroyed).
                            # Manually reset internal state so restore can proceed.
                            print(
                                f"[WARN] Camera cleanup() failed "
                                f"({type(exc).__name__}: {exc}), "
                                "resetting state manually.",
                                flush=True,
                            )
                            try:
                                internal_cam.rp = None  # noqa: SLF001
                                internal_cam.rp_annotators = {}  # noqa: SLF001
                            except Exception as exc:
                                print(
                                    f"[WARN] Failed to reset camera state: {exc}",
                                    flush=True,
                                )

                    # Step 2: Create brand-new camera with fresh render product.
                    # Recreated cameras retain the same minimal RGB-D contract
                    # as cold-start cameras. Semantic bounding boxes are not
                    # part of the evaluated agent observation.
                    config = getattr(sensor, "config", None)
                    if config is not None and hasattr(config, "landmarks"):
                        config.landmarks = False
                    restore = getattr(sensor, "restore_sensor_info", None)
                    if callable(restore):
                        try:
                            restore()
                            refreshed += 1
                        except Exception as exc:
                            print(
                                f"[ERROR] restore_sensor_info() failed: {exc}",
                                flush=True,
                            )
                            raise RuntimeError(
                                "Failed to recreate camera sensor after "
                                f"NavMesh bake SDG pipeline reset: {exc}"
                            ) from exc
        print(
            f"[INFO] Recreated {refreshed} camera sensor(s) with fresh "
            "render products after SDG pipeline reset.",
            flush=True,
        )

    @staticmethod
    def _reattach_physx_stage_for_runtime(stage: Any) -> None:
        """Re-attach the current USD stage to PhysX after NavMesh-only USD attach."""
        try:
            import omni.physx
            import omni.timeline
            from pxr import UsdUtils
        except Exception as exc:
            raise RuntimeError(f"Cannot import PhysX stage attach APIs: {exc}") from exc

        stage_cache = UsdUtils.StageCache.Get()
        stage_id = stage_cache.GetId(stage).ToLongInt()
        if stage_id < 0:
            stage_id = stage_cache.Insert(stage).ToLongInt()

        timeline = omni.timeline.get_timeline_interface()
        was_playing = bool(timeline and timeline.is_playing())
        if was_playing:
            timeline.stop()
            IsaacAutonomousEnv._pump_kit_updates(count=3)

        physx_sim = omni.physx.get_physx_simulation_interface()
        try:
            physx_sim.detach_stage()
        except Exception as exc:
            print(
                f"[WARN] PhysX detach before runtime reattach failed: {exc}", flush=True
            )
        attach_result = physx_sim.attach_stage(stage_id)
        if not attach_result:
            raise RuntimeError(f"PhysX attach_stage({stage_id}) returned False.")

        if timeline is not None and not timeline.is_playing():
            timeline.play()
            IsaacAutonomousEnv._pump_kit_updates(count=3)
        print(
            "[INFO] Re-attached current USD stage to PhysX after NavMesh bake: "
            f"stage_id={stage_id}, was_playing={was_playing}.",
            flush=True,
        )

    def _anim_people_navmesh_volume_transform(
        self, stage: Any
    ) -> tuple[list[float], list[float]]:
        """Return a NavMeshVolume covering the loaded navigable scene mesh set."""
        return self._official_occupancy_navmesh_volume_transform()

    def _official_occupancy_navmesh_volume_transform(
        self,
    ) -> tuple[list[float], list[float]]:
        """Use the episode-declared official navigation domain.

        GRScenes may contain render/physics child meshes whose authored extent
        is invalid even though the official occupancy layer is complete.  The
        occupancy axes are the benchmark's canonical world-space navigation
        bounds, so they are a stronger source for the AnimPeople bake volume
        than taking the union of every render mesh.
        """
        occupancy_path = _compiled_occupancy_path(self.episode)
        occupancy = np.load(occupancy_path, mmap_mode="r")
        if occupancy.ndim != 2 or occupancy.shape[0] < 3 or occupancy.shape[1] < 3:
            raise ValueError(
                f"Official occupancy layer has invalid shape {occupancy.shape}: "
                f"{occupancy_path}"
            )
        x_axis = np.asarray(occupancy[0, 1:], dtype=float)
        y_axis = np.asarray(occupancy[1:, 0], dtype=float)
        if (
            not np.all(np.isfinite(x_axis))
            or not np.all(np.isfinite(y_axis))
            or not np.all(np.diff(x_axis) > 0)
            or not np.all(np.diff(y_axis) < 0)
        ):
            raise ValueError(f"Official occupancy axes are invalid: {occupancy_path}")

        scene_min = np.asarray(
            [
                float(x_axis[0]) - ANIM_PEOPLE_NAVMESH_MARGIN_METERS,
                float(y_axis[-1]) - ANIM_PEOPLE_NAVMESH_MARGIN_METERS,
                -ANIM_PEOPLE_NAVMESH_FLOOR_OFFSET_METERS,
            ]
        )
        scene_max = np.asarray(
            [
                float(x_axis[-1]) + ANIM_PEOPLE_NAVMESH_MARGIN_METERS,
                float(y_axis[0]) + ANIM_PEOPLE_NAVMESH_MARGIN_METERS,
                ANIM_PEOPLE_NAVMESH_HEIGHT_METERS,
            ]
        )
        size = np.maximum(
            scene_max - scene_min,
            np.asarray(
                [
                    ANIM_PEOPLE_NAVMESH_MIN_SIZE_METERS,
                    ANIM_PEOPLE_NAVMESH_MIN_SIZE_METERS,
                    ANIM_PEOPLE_NAVMESH_HEIGHT_METERS
                    + ANIM_PEOPLE_NAVMESH_FLOOR_OFFSET_METERS,
                ]
            ),
        )
        center = scene_min + size * 0.5
        print(
            "[INFO] NavMeshVolume bounds from official occupancy axes: "
            f"source={occupancy_path}, center={center.tolist()}, scale={size.tolist()}",
            flush=True,
        )
        return [float(v) for v in center], [float(v) for v in size]

    @staticmethod
    def compute_scene_navmesh_volume_bounds(
        stage: Any,
    ) -> tuple[list[float], list[float]]:
        """Compute a NavMeshVolume from real scene meshes in the loaded USD stage.

        The bake volume must represent the whole loaded scene, not the current
        episode's actor/target subset. Render-only background meshes such as
        HDR spheres are excluded so the volume tracks the indoor asset instead
        of the outdoor backdrop. Real geometry with no reliable world-space
        bound is a hard error; GRScene's empty Mesh containers are ignored only
        when their descendant meshes carry the actual geometry.
        """
        from pxr import UsdGeom

        try:
            meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
        except Exception:
            meters_per_unit = 1.0

        purposes = []
        for token_name in ("default_", "render", "proxy"):
            token = getattr(UsdGeom.Tokens, token_name, None)
            if token is not None:
                purposes.append(token)
        if not purposes:
            purposes = ["default"]
        try:
            bbox_cache = UsdGeom.BBoxCache(0.0, purposes, True)
        except Exception:
            bbox_cache = None

        scene_min: Optional[np.ndarray] = None
        scene_max: Optional[np.ndarray] = None
        scene_mesh_count = 0
        empty_container_count = 0
        empty_container_samples: list[str] = []
        failed_bounds: list[str] = []
        for prim in list(stage.TraverseAll()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            path = str(prim.GetPath())
            if not IsaacAutonomousEnv._is_scene_navmesh_candidate_path(path):
                continue
            scene_mesh_count += 1
            bounds = IsaacAutonomousEnv._mesh_world_bounds(
                prim,
                meters_per_unit,
                bbox_cache,
            )
            if bounds is None:
                if IsaacAutonomousEnv._mesh_is_empty_geometry_container(prim):
                    empty_container_count += 1
                    if len(empty_container_samples) < 5:
                        empty_container_samples.append(path)
                    continue
                failed_bounds.append(path)
                continue
            mesh_min, mesh_max = bounds
            scene_min = (
                mesh_min if scene_min is None else np.minimum(scene_min, mesh_min)
            )
            scene_max = (
                mesh_max if scene_max is None else np.maximum(scene_max, mesh_max)
            )

        if scene_mesh_count == 0:
            raise RuntimeError(
                "Cannot compute NavMeshVolume: no scene mesh prims found under "
                "/World/env_0/scene."
            )
        if failed_bounds:
            raise RuntimeError(
                "Cannot compute full-scene NavMeshVolume because some scene "
                f"meshes have no reliable world bounds: {failed_bounds[:10]} "
                f"(total={len(failed_bounds)})."
            )
        if scene_min is None or scene_max is None:
            raise RuntimeError("Cannot compute NavMeshVolume: scene bounds are empty.")

        scene_min = np.asarray(scene_min, dtype=float)
        scene_max = np.asarray(scene_max, dtype=float)
        scene_min[:2] -= ANIM_PEOPLE_NAVMESH_MARGIN_METERS
        scene_max[:2] += ANIM_PEOPLE_NAVMESH_MARGIN_METERS
        scene_min[2] = -ANIM_PEOPLE_NAVMESH_FLOOR_OFFSET_METERS
        scene_max[2] = ANIM_PEOPLE_NAVMESH_HEIGHT_METERS
        size = np.maximum(
            scene_max - scene_min,
            np.asarray(
                [
                    ANIM_PEOPLE_NAVMESH_MIN_SIZE_METERS,
                    ANIM_PEOPLE_NAVMESH_MIN_SIZE_METERS,
                    ANIM_PEOPLE_NAVMESH_HEIGHT_METERS
                    + ANIM_PEOPLE_NAVMESH_FLOOR_OFFSET_METERS,
                ],
                dtype=float,
            ),
        )
        center = scene_min + size * 0.5
        print(
            "[INFO] Full-scene NavMeshVolume bounds from scene meshes: "
            f"scene_mesh_count={scene_mesh_count}, center={center.tolist()}, "
            f"scale={size.tolist()}",
            flush=True,
        )
        if empty_container_count:
            print(
                "[INFO] Ignored empty scene Mesh containers while computing "
                "NavMeshVolume bounds: "
                f"count={empty_container_count}, samples={empty_container_samples}.",
                flush=True,
            )
        return [float(v) for v in center], [float(v) for v in size]

    @staticmethod
    def _is_xyz_point(value: Any) -> bool:
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return False
        try:
            [float(item) for item in value[:3]]
        except (TypeError, ValueError):
            return False
        return True

    @staticmethod
    def _set_navmesh_volume_transform(
        volume_prim: Any, center: list[float], scale: list[float]
    ):
        """Resize the official NavMeshVolume prim to cover the loaded scene.

        The C++ NavMesh baker reads the ``extent`` attribute directly to
        determine the volume's world-space bounding box — it does **not**
        apply xformOps (scale / translate).  We therefore write the actual
        world-space bounds into ``extent`` and leave the xformOps at identity.
        """
        from pxr import Gf, Sdf, UsdGeom, Vt

        # Compute world-space min/max from center + scale.
        ws_min = [float(center[i]) - float(scale[i]) * 0.5 for i in range(3)]
        ws_max = [float(center[i]) + float(scale[i]) * 0.5 for i in range(3)]

        extent_attr = volume_prim.GetAttribute("extent")
        if not extent_attr:
            extent_attr = volume_prim.CreateAttribute(
                "extent",
                Sdf.ValueTypeNames.Float3Array,
                custom=False,
            )
        extent_attr.Set(
            Vt.Vec3fArray(
                [
                    Gf.Vec3f(*ws_min),
                    Gf.Vec3f(*ws_max),
                ]
            )
        )
        print(
            f"[INFO] NavMeshVolume extent set to world-space bounds: "
            f"min={ws_min}, max={ws_max}",
            flush=True,
        )

        # Identity xformOps (baker doesn't use them, but keep stage valid).
        xformable = UsdGeom.Xformable(volume_prim)
        xformable.ClearXformOpOrder()
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))

    @staticmethod
    def _navmesh_agent_settings(meters_per_unit: float) -> Dict[str, float]:
        """Return navigation bake dimensions in the current stage units."""
        if not np.isfinite(meters_per_unit) or meters_per_unit <= 0:
            meters_per_unit = 1.0

        def stage_units(meters: float) -> float:
            return float(meters / meters_per_unit)

        return {
            "agent_min_height": stage_units(
                ANIM_PEOPLE_NAVMESH_AGENT_MIN_HEIGHT_METERS
            ),
            "agent_min_radius": stage_units(
                ANIM_PEOPLE_NAVMESH_AGENT_MIN_RADIUS_METERS
            ),
            "agent_max_radius": stage_units(
                ANIM_PEOPLE_NAVMESH_AGENT_MAX_RADIUS_METERS
            ),
            "agent_max_step_height": stage_units(
                ANIM_PEOPLE_NAVMESH_AGENT_MAX_STEP_HEIGHT_METERS
            ),
            "agent_max_floor_slope": ANIM_PEOPLE_NAVMESH_AGENT_MAX_FLOOR_SLOPE_DEGREES,
            "agent_min_island_radius": stage_units(
                ANIM_PEOPLE_NAVMESH_AGENT_MIN_ISLAND_RADIUS_METERS
            ),
        }

    @staticmethod
    def _configure_navmesh_bake_settings(
        settings: Any, navmesh_settings: Any, stage: Any
    ) -> None:
        """Write deterministic bake settings for meter and centimeter stages."""
        from pxr import UsdGeom

        use_gpu_value = os.environ.get("TARSPHERE_NAVMESH_USE_GPU", "1")
        if use_gpu_value not in {"0", "1"}:
            raise ValueError("TARSPHERE_NAVMESH_USE_GPU must be exactly 0 or 1.")
        use_gpu = use_gpu_value == "1"
        # Official native CPU/GPU baker selection; geometry and agent settings
        # below are identical. MeshTools' CUDA device is independent of Kit's
        # renderer GPU selection, so demo launches can select native CPU bake.
        gpu_path = navmesh_settings.USE_GPU_NAVMESH_SETTING_PATH
        settings.set(gpu_path, use_gpu)
        actual_use_gpu = settings.get(gpu_path)
        if actual_use_gpu is not use_gpu:
            raise RuntimeError(
                "Native NavMesh backend setting was not applied: "
                f"path={gpu_path}, requested={use_gpu!r}, read_back={actual_use_gpu!r}"
            )

        meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
        agent = IsaacAutonomousEnv._navmesh_agent_settings(meters_per_unit)
        IsaacAutonomousEnv._configure_navmesh_layer_metadata(stage, agent)
        settings.set(navmesh_settings.AUTO_REBAKE_SETTING_PATH, False)
        settings.set(navmesh_settings.CACHE_ENABLED_SETTING_PATH, False)
        settings.set(
            navmesh_settings.EXCLUDE_RIGID_BODIES_PATH,
            ANIM_PEOPLE_NAVMESH_EXCLUDE_RIGID_BODIES,
        )
        settings.set(
            navmesh_settings.AGENT_MIN_HEIGHT_SETTING_PATH, agent["agent_min_height"]
        )
        settings.set(
            navmesh_settings.AGENT_MIN_RADIUS_SETTING_PATH, agent["agent_min_radius"]
        )
        settings.set(
            navmesh_settings.AGENT_MAX_RADIUS_SETTING_PATH, agent["agent_max_radius"]
        )
        settings.set(
            navmesh_settings.AGENT_MAX_STEP_HEIGHT_SETTING_PATH,
            agent["agent_max_step_height"],
        )
        settings.set(
            navmesh_settings.AGENT_MAX_FLOOR_SLOPE_SETTING_PATH,
            agent["agent_max_floor_slope"],
        )
        settings.set(
            navmesh_settings.AGENT_MIN_ISLAND_RADIUS_SETTING_PATH,
            agent["agent_min_island_radius"],
        )
        print(
            "[INFO] NavMesh bake settings for omni.anim.people: "
            f"bake_backend={'gpu' if use_gpu else 'cpu'}, "
            f"use_gpu_read_back={actual_use_gpu}, "
            f"meters_per_unit={meters_per_unit}, "
            f"agent_min_height={agent['agent_min_height']:.3f}, "
            f"agent_max_radius={agent['agent_max_radius']:.3f}, "
            f"agent_min_island_radius={agent['agent_min_island_radius']:.3f}, "
            f"exclude_rigid_bodies={ANIM_PEOPLE_NAVMESH_EXCLUDE_RIGID_BODIES}",
            flush=True,
        )
        # Read back critical settings to confirm they were written.
        for name, path in (
            (
                "agent_min_island_radius",
                navmesh_settings.AGENT_MIN_ISLAND_RADIUS_SETTING_PATH,
            ),
            ("agent_min_height", navmesh_settings.AGENT_MIN_HEIGHT_SETTING_PATH),
        ):
            actual = settings.get(path)
            if TARSPHERE_ISAAC_DEBUG:
                print(
                    f"[DEBUG] NavMesh setting read-back: {name} "
                    f"path={path} written={agent.get(name)} read_back={actual}",
                    flush=True,
                )

    @staticmethod
    def _configure_navmesh_layer_metadata(stage: Any, agent: Dict[str, float]) -> None:
        """Mirror official test-scene navmesh settings into layer custom data."""
        from pxr import Gf

        layer = stage.GetRootLayer()
        custom_data = dict(getattr(layer, "customLayerData", {}) or {})
        custom_data["navmeshSettings"] = {
            "agentMaxFloorSlope": float(agent["agent_max_floor_slope"]),
            "agentMaxRadius": float(agent["agent_max_radius"]),
            "agentMaxStepHeight": float(agent["agent_max_step_height"]),
            "agentMinHeight": float(agent["agent_min_height"]),
            "excludeRigidBodies": ANIM_PEOPLE_NAVMESH_EXCLUDE_RIGID_BODIES,
            "areas": {
                "0": {
                    "areaName": "Walkable",
                    "color": Gf.Vec3f(0.2, 0.8, 1.0),
                    "defaultCost": 1.0,
                },
                "1": {
                    "areaName": "NotWalkable",
                    "color": Gf.Vec3f(1.0, 0.0, 0.0),
                    "defaultCost": -1.0,
                },
            },
        }
        layer.customLayerData = custom_data

    @staticmethod
    def _navmesh_triangle_count(navmesh: Any) -> int:
        if navmesh is None:
            return 0
        try:
            vertices = navmesh.get_draw_triangles(0)
        except Exception as exc:
            print(f"[WARN] navmesh.get_draw_triangles() failed: {exc}", flush=True)
            return 0
        return int(len(vertices) // 3)

    @staticmethod
    def _bake_navmesh_with_updates(
        navigation: Any,
        *,
        timeout_seconds: float = 180.0,
    ) -> tuple[Any, Any, int]:
        wait_baking = getattr(navigation, "start_navmesh_baking_and_wait", None)
        if callable(wait_baking):
            bake_result = wait_baking()
            # The standalone NavMesh test pumps 120 updates after bake
            # before reading get_navmesh().  Fewer pumps may return None
            # because the C++ baker hasn't finished writing results.
            IsaacAutonomousEnv._pump_kit_updates(count=120)
            navmesh = navigation.get_navmesh()
            return (
                bake_result,
                navmesh,
                IsaacAutonomousEnv._navmesh_triangle_count(navmesh),
            )

        start_baking = getattr(navigation, "start_navmesh_baking", None)
        if not callable(start_baking):
            raise RuntimeError(
                "Isaac navigation interface exposes neither start_navmesh_baking "
                "nor start_navmesh_baking_and_wait."
            )

        bake_result = start_baking()
        deadline = time.monotonic() + timeout_seconds
        navmesh = navigation.get_navmesh()
        triangle_count = IsaacAutonomousEnv._navmesh_triangle_count(navmesh)
        while triangle_count <= 0 and time.monotonic() < deadline:
            IsaacAutonomousEnv._pump_kit_updates(count=1)
            navmesh = navigation.get_navmesh()
            triangle_count = IsaacAutonomousEnv._navmesh_triangle_count(navmesh)
        return bake_result, navmesh, triangle_count

    @staticmethod
    def _pump_kit_updates(count: int = 1) -> None:
        from tarsphere.infra.simulation.kit_updates import pump_kit_updates

        pump_kit_updates(count=count)

    @staticmethod
    def _configure_human_animation_clock() -> dict:
        """Align the Kit loop step size with the Human animation pump cadence.

        The pump cadence alone is not enough.  Each ``app.update()`` reports
        ``manual_step_size`` seconds of elapsed time to
        ``character_behavior.on_update`` no matter how many physics ticks
        actually passed, so pumping every N ticks while the step size stays at
        1/240 under-reports elapsed time N-fold and the characters crawl at 1/N
        speed.  Measured: 0.14 m/s against a nominal 1.2 m/s at N=4 with the
        step size left alone, versus 1.03 m/s once it matches.

        ``simulation_context.py:460`` sets this from ``rendering_dt``, but only
        at construction time, and this environment builds the sim with
        ``rendering_dt = 1/240``.  Changing it afterwards therefore has to go
        through ``omni.kit.loop._loop`` directly.

        Physics is deliberately untouched: the Kit loop and
        ``_physics_context.set_physics_dt`` are separate clocks.  The physics dt
        is read back and returned so a caller can prove it stayed at 1/240 --
        it governs robot dynamics and must never be assumed.
        """
        report: dict = {
            "requested_step_size_s": float(HUMAN_ANIMATION_KIT_STEP_SIZE_S),
            "pump_every_ticks": int(HUMAN_ANIMATION_PUMP_EVERY_TICKS),
            "applied_step_size_s": None,
            "physics_dt": None,
            "physics_dt_source": None,
        }
        try:
            import omni.kit.loop._loop as omni_loop

            loop = omni_loop.acquire_loop_interface()
        except Exception as exc:
            raise RuntimeError(
                "Cannot reach the Kit loop interface to set the Human "
                f"animation step size: {exc}"
            ) from exc

        loop.set_manual_step_size(float(HUMAN_ANIMATION_KIT_STEP_SIZE_S))
        loop.set_manual_mode(True)
        try:
            report["applied_step_size_s"] = float(loop.get_manual_step_size())
        except Exception:
            pass
        return report

    def _assert_physics_dt_unchanged(self, report: dict) -> None:
        """Prove the animation clock change left robot physics at 1/240.

        Read through ``runner._world`` rather than
        ``SimulationContext.instance()``: the singleton returned None on this
        stack in two separate measurements, so a None from it would be silently
        read as "nothing to check" -- exactly the failure this guard exists to
        prevent.
        """
        expected = 1.0 / SIMULATION_PHYSICS_HZ
        runner = getattr(self._env, "runner", None)
        world = getattr(runner, "_world", None)
        physics_dt = None
        source = None
        for handle, attribute, label in (
            (world, "get_physics_dt", "runner._world.get_physics_dt"),
            (
                getattr(world, "_physics_context", None),
                "get_physics_dt",
                "runner._world._physics_context.get_physics_dt",
            ),
        ):
            getter = getattr(handle, attribute, None)
            if not callable(getter):
                continue
            try:
                physics_dt = float(getter())
                source = label
                break
            except Exception:
                continue

        report["physics_dt"] = physics_dt
        report["physics_dt_source"] = source
        if physics_dt is None:
            raise RuntimeError(
                "Could not read physics_dt back after setting the Human "
                "animation step size. The Kit loop step size and the physics "
                "dt are separate clocks, but that separation must be observed "
                "rather than assumed: physics dt governs robot dynamics."
            )
        if abs(physics_dt - expected) > 1e-9:
            raise RuntimeError(
                f"Setting the Human animation step size changed physics_dt to "
                f"{physics_dt} (expected {expected}). Robot dynamics run on "
                "this clock; refusing to continue."
            )

    def _pump_human_motion_window(self, *, physics_tick: int) -> int:
        """Gate for :meth:`RobotBodyExecutor.step_simulation`.

        Animated catalog actors receive one update on each matching tick,
        including idle LookAround. Native rendering contributes no extra time.
        """
        return self._human_body.pump_motion_window(
            physics_tick=int(physics_tick),
            pump_every=int(HUMAN_ANIMATION_PUMP_EVERY_TICKS),
        )

    @staticmethod
    def _configure_navmesh_bake_meshes(
        stage: Any,
        center: list[float],
        scale: list[float],
    ) -> None:
        """Configure bake mesh inclusion for the scene NavMesh.

        Real loaded scene meshes are included. Render-only environment meshes
        and runtime actors (robots, characters, human collision proxies, and ego
        cameras) are excluded so they do not become static NavMesh obstacles.
        """
        del center, scale
        import NavSchema
        from pxr import UsdGeom

        excluded = 0
        included = 0
        included_scene = 0
        failed: list[str] = []
        decisions: list[tuple[Any, str, bool, str]] = []
        reason_counts: Dict[str, int] = {}
        exclude_prefixes = (
            "/World/env_0/robots",
            "/World/Characters",
            "/World/HumanCollisionProxies",
            "/World/HumanEgoCameras",
        )

        for prim in list(stage.TraverseAll()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            path = str(prim.GetPath())
            if path.startswith(exclude_prefixes):
                decisions.append((prim, path, True, "runtime_actor"))
                continue

            scene_candidate = IsaacAutonomousEnv._is_scene_navmesh_candidate_path(path)
            if scene_candidate:
                decisions.append((prim, path, False, "scene_mesh"))
            else:
                decisions.append((prim, path, True, "non_scene_mesh"))

        included_samples: list[str] = []
        excluded_samples: list[str] = []
        for prim, path, should_exclude, reason in decisions:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            try:
                if should_exclude:
                    NavSchema.NavMeshExcludeAPI.Apply(prim)
                    excluded += 1
                    if len(excluded_samples) < 5:
                        excluded_samples.append(f"{path} ({reason})")
                else:
                    try:
                        prim.RemoveAPI(NavSchema.NavMeshExcludeAPI)
                    except Exception as exc:
                        failed.append(f"{path} remove NavMeshExcludeAPI: {exc}")
                    included += 1
                    if IsaacAutonomousEnv._is_scene_navmesh_candidate_path(path):
                        included_scene += 1
                    if len(included_samples) < 5:
                        included_samples.append(f"{path} ({reason})")
            except Exception as exc:
                failed.append(f"{path}: {exc}")
        print(
            "[INFO] Configured NavMesh bake mesh filters: "
            f"included_meshes={included}, excluded_meshes={excluded}, "
            f"included_scene_meshes={included_scene}, failed_meshes={len(failed)}, "
            f"reasons={reason_counts}.",
            flush=True,
        )
        if TARSPHERE_ISAAC_DEBUG and included_samples:
            print(
                "[DEBUG] Included NavMesh mesh samples: " + "; ".join(included_samples),
                flush=True,
            )
        if TARSPHERE_ISAAC_DEBUG and excluded_samples:
            print(
                "[DEBUG] Excluded NavMesh mesh samples: " + "; ".join(excluded_samples),
                flush=True,
            )
        if failed:
            raise RuntimeError(
                "Failed to configure required NavMesh bake meshes: "
                + "; ".join(failed[:10])
            )

    @staticmethod
    def _mesh_world_bounds(
        prim: Any,
        meters_per_unit: float,
        bbox_cache: Any = None,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """World-space (min, max) AABB of a mesh prim in meters.

        Prefers ``UsdGeom.BBoxCache.ComputeWorldBound``, but GRScene meshes
        often defeat it (empty range / exception at non-1.0 stage metrics).
        If the cache cannot produce a usable bound, uses the mesh's ``extent``
        attribute transformed by the prim's local-to-world matrix, which is
        reliable for authored extents.
        Returns ``None`` when no usable bound can be computed.
        """
        from pxr import UsdGeom

        mesh_min: Optional[np.ndarray] = None
        mesh_max: Optional[np.ndarray] = None

        # Primary: BBoxCache world bound (when the caller has cache support).
        if bbox_cache is not None:
            try:
                aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
                mesh_range = (
                    aligned.GetRange() if hasattr(aligned, "GetRange") else aligned
                )
                if not (hasattr(mesh_range, "IsEmpty") and mesh_range.IsEmpty()):
                    mesh_min = (
                        np.asarray(mesh_range.GetMin(), dtype=float) * meters_per_unit
                    )
                    mesh_max = (
                        np.asarray(mesh_range.GetMax(), dtype=float) * meters_per_unit
                    )
            except Exception:
                mesh_min = mesh_max = None

        # Alternate path: authored extent attribute + local-to-world transform.
        # GRScene meshes are placed via ancestor xform references, so the local
        # extent must be transformed into world space for a correct AABB.
        if mesh_min is None or mesh_max is None:
            try:
                from pxr import Gf

                mesh = UsdGeom.Mesh(prim)
                extent = mesh.GetExtentAttr().Get()
                if not extent or len(extent) < 2:
                    return None
                xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
                lo = extent[0]
                hi = extent[1]
                corners = [
                    (lo[0], lo[1], lo[2]),
                    (lo[0], lo[1], hi[2]),
                    (lo[0], hi[1], lo[2]),
                    (lo[0], hi[1], hi[2]),
                    (hi[0], lo[1], lo[2]),
                    (hi[0], lo[1], hi[2]),
                    (hi[0], hi[1], lo[2]),
                    (hi[0], hi[1], hi[2]),
                ]
                transformed = [
                    xf.Transform(Gf.Vec3d(cx, cy, cz)) for cx, cy, cz in corners
                ]
                arr = np.asarray(
                    [[float(c[0]), float(c[1]), float(c[2])] for c in transformed],
                    dtype=float,
                )
                mesh_min = arr.min(axis=0) * meters_per_unit
                mesh_max = arr.max(axis=0) * meters_per_unit
            except Exception:
                return None

        if (
            mesh_min is None
            or mesh_max is None
            or not np.all(np.isfinite(mesh_min))
            or not np.all(np.isfinite(mesh_max))
        ):
            return None
        return mesh_min, mesh_max

    @staticmethod
    def _mesh_is_empty_geometry_container(prim: Any) -> bool:
        """Return true for GRScene Mesh prims that only group child meshes.

        Some GRScene assets declare intermediate grouping prims as ``Mesh`` even
        though they have no local points, faces, or authored extent.  Their child
        meshes carry the actual geometry and are traversed separately.  Such
        containers are safe to ignore for volume bounds; a mesh with neither
        local geometry nor child geometry remains a hard error.
        """
        from pxr import UsdGeom

        if IsaacAutonomousEnv._mesh_has_local_geometry(prim):
            return False

        def has_descendant_geometry(parent: Any) -> bool:
            get_children = getattr(parent, "GetChildren", None)
            if not callable(get_children):
                return False
            for child in get_children():
                if child.IsA(
                    UsdGeom.Mesh
                ) and IsaacAutonomousEnv._mesh_has_local_geometry(child):
                    return True
                if has_descendant_geometry(child):
                    return True
            return False

        return has_descendant_geometry(prim)

    @staticmethod
    def _mesh_has_local_geometry(prim: Any) -> bool:
        from pxr import UsdGeom

        try:
            mesh = UsdGeom.Mesh(prim)
            extent = mesh.GetExtentAttr().Get()
            if extent and len(extent) >= 2:
                return True
            points = mesh.GetPointsAttr().Get()
            if points is not None and len(points) > 0:
                return True
            face_counts = mesh.GetFaceVertexCountsAttr().Get()
            if face_counts is not None and len(face_counts) > 0:
                return True
        except Exception:
            return False
        return False

    @staticmethod
    def _mesh_intersects_bounds(
        bbox_cache: Any,
        prim: Any,
        volume_min: np.ndarray,
        volume_max: np.ndarray,
        meters_per_unit: float = 1.0,
    ) -> Optional[bool]:
        bounds = IsaacAutonomousEnv._mesh_world_bounds(
            prim, meters_per_unit, bbox_cache
        )
        if bounds is None:
            return None
        mesh_min, mesh_max = bounds
        return bool(np.all(mesh_max >= volume_min) and np.all(mesh_min <= volume_max))

    @staticmethod
    def _is_scene_navmesh_candidate_path(path: str) -> bool:
        """Return true for loaded scene meshes that must be baked."""
        normalized = str(path)
        if not normalized.startswith("/World/env_0/scene"):
            return False
        return not any(
            part in normalized for part in ANIM_PEOPLE_NAVMESH_RENDER_ONLY_PATH_PARTS
        )

    @staticmethod
    def _attach_current_stage_for_navmesh(stage: Any) -> None:
        """Re-attach TARSphere's current stage so navigation core sees volumes."""
        IsaacAutonomousEnv._navmesh_attach_performed = True
        try:
            import omni.usd
            from pxr import UsdUtils
        except Exception as exc:
            raise RuntimeError(
                f"Cannot import stage attach APIs for NavMesh bake: {exc}"
            ) from exc

        stage_cache = UsdUtils.StageCache.Get()
        stage_id = stage_cache.GetId(stage).ToLongInt()
        if stage_id < 0:
            stage_id = stage_cache.Insert(stage).ToLongInt()
        usd_context = omni.usd.get_context()
        try:
            context_stage_id = usd_context.get_stage_id()
        except Exception as exc:
            print(f"[WARN] get_stage_id() failed: {exc}", flush=True)
            context_stage_id = None
        try:
            attach_result = usd_context.attach_stage_with_callback(stage_id)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to attach USD stage {stage_id} before NavMesh bake: {exc}"
            ) from exc

        IsaacAutonomousEnv._pump_kit_updates(count=120)
        print(
            "[INFO] Re-attached current USD stage for NavMesh bake: "
            f"stage_id={stage_id}, previous_context_stage_id={context_stage_id}, "
            f"usd_attach_result={attach_result}, physx_attach_result=skipped.",
            flush=True,
        )

    def _prepare_anim_people_character_runtime(self) -> None:
        """Verify omni.anim.graph.core saw the stage; no attach replay needed.

        ``omni.anim.graph.core`` registers an ``omni::kit::IStageUpdate`` node at
        extension startup and builds its CharacterManager from that node's
        stage-attach callback.  TARSphere creates the process's one and only
        stage inside ``World()`` (it references the scene in with ``create_prim``
        and never calls ``open_stage``), so there is exactly one attach event per
        run.  This code used to enable the extension afterwards, from
        ``reset()``, which registered the node after that event had passed: the
        callback never fired, ``CharacterManager::Initialize()`` never ran, and
        the log carried ``CharacterManager::Shutdown() called without a prior
        successful call to CharacterManager::Initialize()``.  The old workaround
        replayed the missed event with ``attach_stage_with_callback``, which
        re-populated all 16.9M triangles through Hydra at a measured ~105s per
        run, and then needed a full TARSphere/PhysX repair afterwards because
        an attach raises ``StageEventType.OPENED`` and invalidates ``World``.

        The extensions are now declared in ``tarsphere/infra/kit/
        tarsphere_isaac_python.kit``, so Kit starts them with the SimulationApp,
        before the stage exists.  Verified 2026-08-19 by
        the extension-load diagnostic: all eight report enabled at startup,
        their ``[ext: ...] startup`` lines precede ``World()``, and no
        Shutdown-without-Initialize warning is emitted.  So there is nothing to
        replay -- only the timeline to start, which the anim graph and the
        BehaviorScripts need in order to advance.
        """
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError(
                "No USD stage is attached; cannot initialize the "
                "omni.anim.graph.core CharacterManager."
            )
        self._start_human_animation_timeline()
        self._human_body.mark_character_runtime_prepared()
        print(
            "[INFO] omni.anim.graph.core CharacterManager is live from the "
            "experience-file extension load (--no-navmesh path); skipped the "
            "legacy stage-attach replay.",
            flush=True,
        )

    @staticmethod
    def _resolve_character_asset(asset_root: Path, character: str) -> Path:
        """Resolve the catalog-declared USD for one authorable Human asset."""
        from tarsphere.humans.character_catalog import resolve_character_usd

        return resolve_character_usd(
            character,
            character_assets_root=asset_root / "characters" / "IsaacPeople",
        )

    @staticmethod
    def _start_human_animation_timeline():
        """Best-effort timeline start so referenced USD motion clips advance."""
        try:
            import omni.timeline

            timeline = omni.timeline.get_timeline_interface()
            if timeline is not None and not timeline.is_playing():
                timeline.play()
        except Exception as exc:
            print(
                f"[WARN] Could not start Isaac timeline: {exc}. "
                "Human animations may not advance.",
                flush=True,
            )

    @staticmethod
    def _set_camera_fov(
        camera_prim_path: str,
        horizontal_fov_deg: float,
        *,
        aspect_ratio: float = EGO_CAMERA_ASPECT_RATIO,
    ) -> None:
        """Set the horizontal FOV on a USD Camera prim.

        Adjusts ``focalLength`` to achieve the desired ``horizontal_fov_deg``
        given the current ``horizontalAperture`` and keeps the vertical
        aperture consistent with the render product aspect ratio.  This works
        for both the robot camera and the human ego camera.

        Relationship: FOV = 2 * atan(horiz_aperture / (2 * focal_length))
        ⟹ focal_length = horiz_aperture / (2 * tan(FOV / 2))
        """
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError(f"Cannot set FOV: USD stage unavailable.")
        prim = stage.GetPrimAtPath(camera_prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(
                f"Cannot set FOV: camera prim not found at {camera_prim_path}"
            )
        camera = UsdGeom.Camera(prim)
        if not camera:
            raise RuntimeError(
                f"Cannot set FOV: prim at {camera_prim_path} is not a Camera"
            )
        horiz_aperture = camera.GetHorizontalApertureAttr().Get()
        if not horiz_aperture or horiz_aperture <= 0:
            # Default USD camera aperture in tenths of a mm → ~20.955mm
            horiz_aperture = 20.955
            camera.GetHorizontalApertureAttr().Set(horiz_aperture)
        if aspect_ratio and aspect_ratio > 0:
            camera.GetVerticalApertureAttr().Set(horiz_aperture / float(aspect_ratio))
        fov_rad = math.radians(horizontal_fov_deg)
        focal_length = horiz_aperture / (2.0 * math.tan(fov_rad / 2.0))
        camera.GetFocalLengthAttr().Set(focal_length)

    def _configure_robot_ego_camera(self) -> None:
        """Apply TARSphere robot camera intrinsics/extrinsics after sensor rebuild."""
        camera_path = self._agent_camera_path()
        if not camera_path:
            raise RuntimeError(
                "Cannot configure robot ego camera: TARSphere did not create "
                'a sensor named "camera" for the robot.'
            )
        self._validate_robot_camera_path(camera_path)
        self._capture_camera_rest_orientation(camera_path)
        self._persist_camera_gaze_on_stage()
        self._set_camera_fov(
            camera_path,
            ROBOT_CAMERA_HORIZONTAL_FOV_DEG,
            aspect_ratio=ROBOT_EGO_CAMERA_ASPECT_RATIO,
        )
        if not self._robot_camera_mount_log_emitted:
            self._robot_camera_mount_log_emitted = True
            print(
                f"[INFO] Robot ego camera configured: path={camera_path} "
                f"fov_deg={ROBOT_CAMERA_HORIZONTAL_FOV_DEG:.1f} "
                f"mount_pitch_deg={self._robot_camera_mount_pitch_deg():.1f} "
                f"gaze_tilt_deg={self.camera_gaze_tilt_deg():.1f}",
                flush=True,
            )

    def _validate_dualvln_ego_camera_contract(self) -> None:
        """Require DualVLN to consume the exact evaluated ego optics."""

        physical_actions = getattr(
            getattr(self.episode, "agent_action_space", None),
            "physical",
            (),
        )
        if "navigate" not in {str(action) for action in physical_actions}:
            return
        robot_type = getattr(self.episode.robot, "robot_type", None) or "G1Robot"
        if robot_type != "G1Robot":
            return
        camera_path = self._agent_camera_path()
        if not camera_path:
            raise RuntimeError(
                "navigate is on but the evaluated G1 ego camera is missing."
            )
        if tuple(ROBOT_EGO_CAMERA_RESOLUTION) != (
            DUALVLN_RGB_WIDTH,
            DUALVLN_RGB_HEIGHT,
        ):
            raise RuntimeError(
                "DualVLN consumes the evaluated ego render product directly; "
                "its required resolution is "
                f"{DUALVLN_RGB_WIDTH}x{DUALVLN_RGB_HEIGHT}, but the ego camera "
                f"is {ROBOT_EGO_CAMERA_RESOLUTION[0]}x"
                f"{ROBOT_EGO_CAMERA_RESOLUTION[1]}."
            )
        if (
            abs(
                float(ROBOT_CAMERA_HORIZONTAL_FOV_DEG)
                - float(DUALVLN_HORIZONTAL_FOV_DEG)
            )
            > 1e-6
        ):
            raise RuntimeError(
                "DualVLN and the evaluated ego camera disagree on horizontal "
                f"FOV: {DUALVLN_HORIZONTAL_FOV_DEG:.3f}° versus "
                f"{ROBOT_CAMERA_HORIZONTAL_FOV_DEG:.3f}°."
            )

    def _dualvln_camera_pitch_deg(self) -> float:
        """Shared ego pitch during the current DualVLN inner-loop action."""

        extra = float(self._dualvln_look_down_extra_deg)
        allowed = (0.0, DUALVLN_LOOK_DOWN_EXTRA_DEG)
        if extra not in allowed:
            raise RuntimeError(
                "DualVLN camera tilt extra must be 0 (harness) or calibrated "
                f"LOOKDOWN {DUALVLN_LOOK_DOWN_EXTRA_DEG}°; got {extra}°."
            )
        return self._robot_camera_prim_pitch_deg() + extra

    def _reset_dualvln_camera_tilt(self) -> None:
        """Restore the evaluated ego gaze before returning from navigate."""

        changed = self._dualvln_look_down_extra_deg != 0.0
        self._dualvln_look_down_extra_deg = 0.0
        if self._camera_rest_local_quat is None or not self._agent_camera_path():
            return
        self._persist_camera_gaze_on_stage()
        if changed:
            self._invalidate_ego_snapshot()

    def _apply_dualvln_look_down(self) -> None:
        """Apply one calibrated LOOKDOWN step to the shared ego camera.

        The action id remains Habitat-compatible. Its camera displacement is
        G1-specific and repeated LOOKDOWN is idempotent rather than
        accumulating an unbounded tilt.
        """

        camera_path = self._agent_camera_path()
        if not camera_path:
            raise RuntimeError(
                "DualVLN LOOKDOWN (Habitat action 5) cannot run: "
                "the evaluated ego camera is missing."
            )
        if self._camera_rest_local_quat is None:
            raise RuntimeError(
                "DualVLN LOOKDOWN (Habitat action 5) cannot run: "
                "G1 ego camera rest orientation was not captured. Refusing "
                "a hold/no-tilt stand-in."
            )
        self._dualvln_look_down_extra_deg = DUALVLN_LOOK_DOWN_EXTRA_DEG
        self._persist_camera_gaze_on_stage()
        expected = self._dualvln_camera_pitch_deg()
        target = self._robot_camera_prim_pitch_deg() + DUALVLN_LOOK_DOWN_EXTRA_DEG
        if abs(expected - target) > 1e-3:
            raise RuntimeError(
                "LOOKDOWN authored the wrong calibrated ego pitch. "
                f"Expected {target}°, got extra="
                f"{self._dualvln_look_down_extra_deg}° pitch={expected}°."
            )
        actual_q = self._camera_local_orientation_wxyz(
            self._require_robot_camera_prim(camera_path)
        )
        expected_q = camera_local_orientation_wxyz(
            self._camera_rest_local_quat,
            expected,
        )
        if max(abs(a - b) for a, b in zip(actual_q, expected_q)) > 1e-3:
            raise RuntimeError(
                "DualVLN LOOKDOWN failed to author the ego camera "
                f"pitch {expected:.3f}° at {camera_path}. "
                f"expected_quat={expected_q} actual_quat={actual_q}. "
                "Refusing a hold/no-tilt stand-in."
            )
        self._invalidate_ego_snapshot()
        print(
            "[INFO] DualVLN LOOKDOWN: shared ego extra "
            f"{DUALVLN_LOOK_DOWN_EXTRA_DEG:.1f}° "
            f"(camera_pitch={expected:.1f}°)",
            flush=True,
        )

    def _apply_dualvln_look_up(self) -> None:
        """Undo one calibrated LOOKDOWN step, returning to rest pitch."""

        camera_path = self._agent_camera_path()
        if not camera_path:
            raise RuntimeError(
                "DualVLN LOOKUP (Habitat action 4) cannot run: "
                "the evaluated ego camera is missing."
            )
        if self._camera_rest_local_quat is None:
            raise RuntimeError(
                "DualVLN LOOKUP (Habitat action 4) cannot run: "
                "G1 ego camera rest orientation was not captured. Refusing "
                "a hold/no-tilt stand-in."
            )
        extra = float(self._dualvln_look_down_extra_deg) + float(
            DUALVLN_LOOK_UP_EXTRA_DEG
        )
        self._dualvln_look_down_extra_deg = min(0.0, extra)
        self._persist_camera_gaze_on_stage()
        expected = self._dualvln_camera_pitch_deg()
        actual_q = self._camera_local_orientation_wxyz(
            self._require_robot_camera_prim(camera_path)
        )
        expected_q = camera_local_orientation_wxyz(
            self._camera_rest_local_quat,
            expected,
        )
        if max(abs(a - b) for a, b in zip(actual_q, expected_q)) > 1e-3:
            raise RuntimeError(
                "DualVLN LOOKUP failed to author the ego camera "
                f"pitch {expected:.3f}° at {camera_path}. "
                f"expected_quat={expected_q} actual_quat={actual_q}. "
                "Refusing a hold/no-tilt stand-in."
            )
        self._invalidate_ego_snapshot()
        print(
            "[INFO] DualVLN LOOKUP: extra now "
            f"{self._dualvln_look_down_extra_deg:.1f}° "
            f"(camera_pitch={expected:.1f}°)",
            flush=True,
        )

    def _validate_robot_camera_path(self, camera_path: str) -> None:
        """Require the sensor to remain attached to its declared robot link."""
        from tarsphere.embodiment.robot_profiles import get_robot_profile

        profile = get_robot_profile(self.episode.robot.robot_type)
        expected_suffix = "/" + profile.camera_prim_path.strip("/")
        if not str(camera_path).endswith(expected_suffix):
            raise RuntimeError(
                f"{profile.name} ego camera resolved to {camera_path!r}; expected "
                f"the declared robot link suffix {expected_suffix!r}. Refusing "
                "an automatically relocated camera because it changes the "
                "benchmark observation contract."
            )

    def _robot_camera_mount_pitch_deg(self) -> float:
        """Return the fixed camera mount calibration for this robot profile.

        This value uses the public negative-down convention.  It is the asset's
        mounted zero, not the Agent's gaze, and does not change during an
        episode.  The live angle is this plus the commanded gaze tilt;
        :meth:`_robot_camera_prim_pitch_deg` owns the asset-local mapping.
        """
        from tarsphere.embodiment.robot_profiles import get_robot_profile

        profile = get_robot_profile(self.episode.robot.robot_type)
        default = float(getattr(profile, "camera_mount_pitch_deg", 0.0) or 0.0)
        env_name = str(getattr(profile, "camera_mount_pitch_env", "") or "").strip()
        if env_name:
            return _float_env(env_name, default)
        return default

    def camera_gaze_tilt_deg(self) -> float:
        """Return the agent's commanded gaze tilt, in degrees, negative down.

        Absolute offset from the mount calibration, 0 meaning no additional
        offset beyond the profile's mounted default. This is the only piece of the camera's orientation the agent
        controls, and it survives across decision steps — including walking —
        until the agent aims again or the episode resets.
        """
        return float(self._camera_gaze_tilt_deg)

    def set_camera_gaze_tilt_deg(self, pitch_deg: float) -> None:
        """Aim the ego camera at ``pitch_deg`` and rewrite the camera prim.

        ``pitch_deg`` is absolute against the mount calibration and must already
        be inside +/-``CAMERA_GAZE_TILT_LIMIT_DEG``; the ``tilt_camera`` tool
        rejects an out-of-range request as a malformed argument at the contract
        boundary.  An out-of-range angle reaching here means a producer bypassed
        that boundary, so it fails rather than being quietly clipped to the limit
        and reported as the angle the agent asked for.
        """
        requested = float(pitch_deg)
        limit = float(CAMERA_GAZE_TILT_LIMIT_DEG)
        if not math.isfinite(requested) or abs(requested) > limit:
            raise ValueError(
                f"Camera gaze tilt {pitch_deg!r} is outside the contracted "
                f"+/-{limit:.1f} deg range."
            )
        self._camera_gaze_tilt_deg = requested
        camera_path = self._agent_camera_path()
        if not camera_path:
            raise RuntimeError(
                "Cannot aim the ego camera: TARSphere did not create a sensor "
                'named "camera" for the robot.'
            )
        self._persist_camera_gaze_on_stage()
        # Replicator renders from Fabric state, while XformOp.Set authors USD.
        # Publish the new camera transform before the decision render window;
        # otherwise the returned RGB-D frame can remain at the previous gaze
        # even though world_T_cam already reflects the authored transform.
        # A camera-only edit must not add an unaccounted Human animation tick.
        self._force_one_ego_render(capture_observer=False)
        # Aiming the camera changes the measured world without advancing
        # physics, so the stamped snapshot's physics-step check cannot see it.
        self._invalidate_ego_snapshot()

    def _robot_camera_prim_pitch_deg(self) -> float:
        """Map the public gaze convention onto the calibrated USD camera op.

        The public contract and this mounted camera's local RotateX operation
        both use negative pitch for a downward gaze.  Keeping the mapping in
        one method makes the asset-local convention explicit and testable.
        """
        return self._robot_camera_total_pitch_deg()

    def _robot_camera_total_pitch_deg(self) -> float:
        """Return optical-frame pitch for observations and back-projection.

        The public convention is negative-down, hence mount calibration plus
        gaze tilt.  The equivalent asset-local RotateX value is deliberately
        computed by :meth:`_robot_camera_prim_pitch_deg` instead of leaking
        USD coordinates into observations.
        """
        return self._robot_camera_mount_pitch_deg() + self.camera_gaze_tilt_deg()

    def _install_camera_gaze_persistence(self) -> None:
        """Re-author gaze on every TARSphere physics tick, including walks.

        Isaac's camera ``XFormPrim`` keeps only translate/orient/scale in
        Fabric.  A sidecar RotateX written by ``tilt_camera`` survives the
        idle observation and is discarded as soon as the parent link moves,
        so the next walking frame snaps back to the navigation view.  Wrapping
        ``env.step`` rewrites the Fabric-tracked orient from episode state
        before that frame is captured.
        """
        env = self._env
        if env is None or getattr(env, "_tarsphere_camera_gaze_step_wrapped", False):
            return
        original_step = env.step

        def step_with_persistent_gaze(*args: Any, **kwargs: Any) -> Any:
            self._persist_camera_gaze_on_stage()
            return original_step(*args, **kwargs)

        env.step = step_with_persistent_gaze  # type: ignore[method-assign]
        env._tarsphere_camera_gaze_step_wrapped = True

    def _persist_camera_gaze_on_stage(self) -> None:
        """Write the commanded mount+gaze pitch onto the live camera prim."""
        if (
            self._camera_gaze_persist_suspended
            or not self.enable_ego_sensors
            or self._env is None
        ):
            return
        camera_path = self._agent_camera_path()
        if not camera_path or self._camera_rest_local_quat is None:
            return
        pitch_deg = self._dualvln_camera_pitch_deg()
        orientation = camera_local_orientation_wxyz(
            self._camera_rest_local_quat,
            pitch_deg,
        )
        self._set_robot_camera_pitch(
            camera_path,
            pitch_deg,
            rest_orientation_wxyz=self._camera_rest_local_quat,
        )
        self._push_camera_orientation_to_xform_prim(orientation)

    def _capture_camera_rest_orientation(self, camera_prim_path: str) -> None:
        """Remember the camera's asset-local orient before TARSphere pitch is baked."""
        if self._camera_rest_local_quat is not None:
            return
        prim = self._require_robot_camera_prim(camera_prim_path)
        from pxr import UsdGeom

        self._strip_sidecar_camera_pitch_op(UsdGeom.Xformable(prim), prim)
        self._camera_rest_local_quat = self._camera_local_orientation_wxyz(prim)

    def _push_camera_orientation_to_xform_prim(
        self,
        orientation_wxyz: Sequence[float],
    ) -> None:
        """Keep Isaac's camera XFormPrim cache on the same local orient as USD."""
        sensor = self._robot_camera_sensor()
        camera = getattr(sensor, "_camera", None) if sensor is not None else None
        xform = getattr(camera, "prim", None)
        # The caller already established that ego sensors are enabled, so a
        # missing XFormPrim means the camera is broken. Returning would leave
        # USD and Isaac's cached local pose disagreeing about where it looks.
        if xform is None or not callable(getattr(xform, "set_local_pose", None)):
            raise RuntimeError(
                "Cannot synchronize the robot camera orientation: the camera "
                f"XFormPrim is unavailable (sensor={sensor!r})."
            )
        xform.set_local_pose(
            orientation=np.asarray(orientation_wxyz, dtype=np.float64),
        )

    @staticmethod
    def _require_robot_camera_prim(camera_prim_path: str) -> Any:
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Cannot set robot camera pitch: USD stage unavailable.")
        prim = stage.GetPrimAtPath(camera_prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(
                f"Cannot set robot camera pitch: camera prim not found at "
                f"{camera_prim_path}"
            )
        if not prim.IsA(UsdGeom.Camera):
            raise RuntimeError(
                f"Cannot set robot camera pitch: prim at {camera_prim_path} "
                "is not a Camera"
            )
        return prim

    @staticmethod
    def _strip_sidecar_camera_pitch_op(xform: Any, prim: Any) -> None:
        remaining = [
            op
            for op in xform.GetOrderedXformOps()
            if op.GetOpName() != _CAMERA_SIDECAR_PITCH_OP
        ]
        if prim.HasAttribute(_CAMERA_SIDECAR_PITCH_OP):
            prim.RemoveProperty(_CAMERA_SIDECAR_PITCH_OP)
        if remaining:
            xform.SetXformOpOrder(remaining)

    @staticmethod
    def _camera_local_orientation_wxyz(
        prim: Any,
    ) -> tuple[float, float, float, float]:
        attr = prim.GetAttribute("xformOp:orient")
        value = attr.Get() if attr is not None else None
        if value is None:
            return (1.0, 0.0, 0.0, 0.0)
        imag = value.GetImaginary()
        return (
            float(value.GetReal()),
            float(imag[0]),
            float(imag[1]),
            float(imag[2]),
        )

    @staticmethod
    def _set_robot_camera_pitch(
        camera_prim_path: str,
        pitch_deg: float,
        *,
        rest_orientation_wxyz: Sequence[float] | None = None,
    ) -> None:
        """Write mount+gaze pitch into the Fabric-tracked local orient.

        ``pitch_deg`` is already expressed in the camera prim's local RotateX
        coordinates.  Conversion from the public optical-frame gaze convention
        happens in :meth:`_robot_camera_prim_pitch_deg`.  The value is authored
        on ``xformOp:orient`` rather than a sidecar RotateX because Isaac's
        camera ``XFormPrim`` only keeps translate/orient/scale after the parent
        link starts moving.
        """
        from pxr import Gf, UsdGeom

        prim = IsaacAutonomousEnv._require_robot_camera_prim(camera_prim_path)
        xform = UsdGeom.Xformable(prim)
        IsaacAutonomousEnv._strip_sidecar_camera_pitch_op(xform, prim)
        rest = (
            tuple(float(value) for value in rest_orientation_wxyz)
            if rest_orientation_wxyz is not None
            else IsaacAutonomousEnv._camera_local_orientation_wxyz(prim)
        )
        if len(rest) != 4:
            raise ValueError(
                "Camera rest orientation must be a (w, x, y, z) quaternion, "
                f"got {rest_orientation_wxyz!r}."
            )
        orientation = camera_local_orientation_wxyz(rest, pitch_deg)
        if not prim.HasAttribute("xformOp:orient"):
            xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
        attr = prim.GetAttribute("xformOp:orient")
        if attr.GetTypeName() == "quatf":
            attr.Set(Gf.Quatf(*orientation))
        else:
            attr.Set(Gf.Quatd(*orientation))

    @staticmethod
    def _configure_mdl_system_path(scene_path: Path):
        """Recover missing modules and configure MDL paths before Isaac starts."""
        from tarsphere.infra.simulation.scene_materials import prepare_grscene_materials

        prepare_grscene_materials(scene_path)

    @staticmethod
    def _resolve_asset(asset_path: str, asset_root: Path) -> Path:
        path = resolve_tarsphere_asset(asset_path)
        if path.is_absolute():
            return path
        return asset_root / path

    def _capture_rgbd(self) -> Dict[str, Any]:
        camera = self._obs.get("sensors", {}).get("camera", {})
        rgba = camera.get("rgba")
        depth = camera.get("depth")
        if rgba is None:
            if TARSPHERE_ISAAC_DEBUG and self._physics_steps < 5:
                print(
                    f"[DEBUG-RGBD] step={self._physics_steps} camera_keys={list(camera.keys())} "
                    f"rgba=None depth={depth is not None}",
                    flush=True,
                )
            raise RuntimeError(
                f"Robot ego camera returned no RGB frame at physics step {self._physics_steps}; "
                f"camera_keys={sorted(str(key) for key in camera.keys())}"
            )
        rgba = np.asarray(rgba.cpu() if hasattr(rgba, "cpu") else rgba)
        if rgba.size == 0:
            raise RuntimeError(
                f"Robot ego camera returned empty RGB frame at physics step {self._physics_steps}."
            )
        rgb = rgba[0, :, :, :3] if rgba.ndim == 4 else rgba[:, :, :3]
        rgb_stats = rgb_frame_stats(rgb)
        depth, depth_info = prepare_depth_frame(depth, rgb.shape[:2])
        if not rgbd_frame_is_observable(
            rgb,
            depth_status=str(depth_info.get("depth_status") or ""),
        ):
            raise RuntimeError(
                "Robot ego camera produced no observable RGB-D signal at "
                f"physics step {self._physics_steps}; rgb_stats={rgb_stats}, "
                f"depth_status={depth_info.get('depth_status')!r}"
            )
        if TARSPHERE_ISAAC_DEBUG and self._physics_steps < 5:
            import numpy as _np

            d_info = "None"
            if depth is not None:
                d_arr = _np.asarray(depth.cpu() if hasattr(depth, "cpu") else depth)
                d_info = str(list(_np.squeeze(d_arr).shape))
            print(
                f"[DEBUG-RGBD2] step={self._physics_steps} rgb={rgb.shape} depth_none={depth is None} depth_shape={d_info}",
                flush=True,
            )
        result = {
            # Raw per-pixel arrays so spatial memory (BEV back-projection) can
            # run without re-parsing the colourised PNG.  rgb is HxWx3 uint8,
            # depth is HxW float32 metres (same shape; ``prepare_depth_frame``
            # already aligns it to rgb).
            "rgb": rgb,
            "depth": depth,
            **depth_info,
        }
        tight_boxes = camera.get("bounding_box_2d_tight")
        if isinstance(tight_boxes, dict):
            # Keep simulator-private semantic annotations available to the
            # Golden oracle.  Evaluated models still receive only RGB-D; the
            # emitted model contract contains ordinary 2-D boxes.
            result["bounding_box_2d_tight"] = tight_boxes
        return result

    def _setup_human_ego_cameras(self):
        """Create one real Isaac RGBD camera per human when the stage exists."""
        if self._env is None:
            return

        try:
            import omni.usd
            import omni.replicator.core as rep
            from isaacsim.sensors.camera import Camera
            from pxr import UsdGeom
        except Exception as exc:
            for human_id in self._human_positions:
                self._human_ego_camera_errors[human_id] = str(exc)
            return

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            for human_id in self._human_positions:
                self._human_ego_camera_errors[human_id] = "Isaac stage is unavailable."
            return

        UsdGeom.Xform.Define(stage, "/World/HumanEgoCameras")
        for human_id in self._human_positions:
            if human_id in self._human_ego_cameras:
                self._sync_human_ego_camera_pose(human_id)
                continue

            camera_prim_path = f"/World/HumanEgoCameras/{human_id}_EgoCamera"
            try:
                # Let Camera() own the prim creation — don't call
                # UsdGeom.Camera.Define separately, it conflicts.
                self._human_ego_camera_paths[human_id] = camera_prim_path
                sensor = Camera(
                    prim_path=camera_prim_path,
                    name=f"{human_id}_ego_camera",
                    resolution=HUMAN_EGO_CAMERA_RESOLUTION,
                )
                render_product = rep.create.render_product(
                    camera_prim_path,
                    HUMAN_EGO_CAMERA_RESOLUTION,
                )
                annotators = {
                    "rgba": rep.AnnotatorRegistry.get_annotator("rgb"),
                    "distance_to_image_plane": rep.AnnotatorRegistry.get_annotator(
                        "distance_to_image_plane"
                    ),
                }
                for annotator in annotators.values():
                    annotator.attach(render_product)
                self._human_ego_cameras[human_id] = {
                    "sensor": sensor,
                    "render_product": render_product,
                    "annotators": annotators,
                }
                # Match the human ego view to the robot ego view for comparable
                # observation geometry without indoor wide-angle distortion.
                self._set_camera_fov(
                    camera_prim_path,
                    HUMAN_EGO_CAMERA_HORIZONTAL_FOV_DEG,
                    aspect_ratio=EGO_CAMERA_ASPECT_RATIO,
                )
                self._sync_human_ego_camera_pose(human_id)
                self._human_ego_camera_errors.pop(human_id, None)
            except Exception as exc:
                self._human_ego_camera_errors[human_id] = str(exc)

    def _warmup_human_ego_cameras(self, max_steps: int = 5):
        if self._env is None or not self._human_ego_cameras:
            return

        zero_action = {"move_by_speed": [0.0, 0.0, 0.0]}
        for _ in range(max_steps):
            if all(
                self._human_ego_camera_has_frame(human_id)
                for human_id in self._human_ego_cameras
            ):
                return
            for human_id in self._human_ego_cameras:
                self._sync_human_ego_camera_pose(human_id)
            self._step_human_ego_render()
            if all(
                self._human_ego_camera_has_frame(human_id)
                for human_id in self._human_ego_cameras
            ):
                return
            self._obs, _, terminated, truncated, _ = self._env.step(action=zero_action)
            self._physics_steps += 1
            if terminated or truncated:
                return

    def _human_ego_camera_has_frame(self, human_id: str) -> bool:
        try:
            rgba, depth = self._read_human_ego_camera(human_id)
            return (
                rgba is not None
                and np.asarray(rgba).size > 0
                and depth is not None
                and np.asarray(depth).size > 0
            )
        except Exception as exc:
            self._human_ego_camera_errors[human_id] = str(exc)
            return False

    @staticmethod
    def _step_human_ego_render():
        try:
            import omni.replicator.core as rep

            rep.orchestrator.step(
                rt_subframes=4,
                delta_time=0.0,
                pause_timeline=False,
                wait_for_render=True,
            )
        except Exception as exc:
            print(f"[WARN] Human ego warmup render step failed: {exc}", flush=True)

    def _read_human_ego_camera(self, human_id: str) -> Tuple[Any, Any]:
        camera = self._human_ego_cameras.get(human_id)
        if camera is None:
            return None, None
        if isinstance(camera, dict):
            annotators = camera.get("annotators", {})
            rgba_annotator = annotators.get("rgba")
            depth_annotator = annotators.get("distance_to_image_plane")
            rgba = rgba_annotator.get_data() if rgba_annotator is not None else None
            depth = depth_annotator.get_data() if depth_annotator is not None else None
            return rgba, depth

        rgba = camera.get_rgba()
        depth_getter = getattr(camera, "get_depth", None)
        if callable(depth_getter):
            depth = depth_getter()
        else:
            depth = camera.get_distance_to_image_plane()
        return rgba, depth

    def _cleanup_human_ego_cameras(self):
        for camera in self._human_ego_cameras.values():
            if isinstance(camera, dict):
                render_product = camera.get("render_product")
                for annotator in camera.get("annotators", {}).values():
                    try:
                        annotator.detach(render_product)
                    except Exception as exc:
                        print(f"[WARN] Ego annotator detach failed: {exc}", flush=True)
                sensor = camera.get("sensor")
                destroy_render_product = getattr(render_product, "destroy", None)
                if callable(destroy_render_product):
                    try:
                        destroy_render_product()
                    except Exception as exc:
                        print(
                            f"[WARN] Ego render product destroy failed: {exc}",
                            flush=True,
                        )
                camera = sensor
            for method_name in ("destroy", "cleanup"):
                cleanup = getattr(camera, method_name, None)
                if not callable(cleanup):
                    continue
                try:
                    cleanup()
                except Exception as exc:
                    print(
                        f"[WARN] Ego sensor {method_name}() failed: {exc}", flush=True
                    )
                break
        self._human_ego_cameras.clear()
        # Remove stale ego-camera prims so a persistent worker does not accumulate
        # one dead camera prim per episode when human ids change.  Cleanup: best
        # effort, failures are warned and swallowed.
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
        except Exception as exc:
            stage = None
            if self._human_ego_camera_paths:
                print(
                    f"[WARN] Cannot access stage for ego-camera cleanup: {exc}",
                    flush=True,
                )
        if stage is not None:
            for camera_prim_path in self._human_ego_camera_paths.values():
                prim = stage.GetPrimAtPath(camera_prim_path)
                if prim and prim.IsValid():
                    try:
                        stage.RemovePrim(camera_prim_path)
                    except Exception as exc:
                        print(
                            f"[WARN] Failed to remove ego camera {camera_prim_path}: {exc}",
                            flush=True,
                        )
        self._human_ego_camera_paths.clear()
        self._human_ego_camera_errors.clear()
        self._human_ego_camera_poses.clear()

    def _capture_human_ego_view(self, human_id: str) -> Dict[str, Any]:
        """Capture human ego RGB image (no depth)."""
        camera = self._human_ego_cameras.get(human_id)
        if camera is None:
            error = self._human_ego_camera_errors.get(human_id)
            return {
                "status": "unavailable",
                "reason": error or "Human ego camera is not initialized yet.",
            }

        try:
            self._sync_human_ego_camera_pose(human_id)
            self._step_human_ego_render()
            rgba, _ = self._read_human_ego_camera(human_id)
            if rgba is None:
                self._step_human_ego_render()
                rgba, _ = self._read_human_ego_camera(human_id)
            if rgba is None:
                return {"status": "missing_rgb"}
            rgba = np.asarray(rgba.cpu() if hasattr(rgba, "cpu") else rgba)
            if rgba.size == 0:
                self._step_human_ego_render()
                rgba, _ = self._read_human_ego_camera(human_id)
                if rgba is None:
                    return {"status": "missing_rgb"}
                rgba = np.asarray(rgba.cpu() if hasattr(rgba, "cpu") else rgba)
            if rgba.size == 0:
                return {"status": "empty_rgb"}
            rgb = rgba[0, :, :, :3] if rgba.ndim == 4 else rgba[:, :, :3]

            result = {"status": "valid"}
            if self.image_dir is not None:
                # Save as pure RGB PNG (no depth composite).
                from PIL import Image

                img = Image.fromarray(rgb.astype(np.uint8))
                path = (
                    self.image_dir
                    / "human_ego"
                    / human_id
                    / f"step_{self._physics_steps:06d}_rgb.png"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                img.save(str(path))
                result["image_path"] = str(path)
            return result
        except Exception as exc:
            self._human_ego_camera_errors[human_id] = str(exc)
            return {
                "status": "error",
                "reason": str(exc),
            }

    def _step_human_bodies(self, dt: float):
        self._human_body.step(dt)

    def _sync_human_body_state(self, human_id: str, state: HumanBodyState):
        self._human_positions[human_id] = list(state.position)
        self._human_yaws[human_id] = float(state.yaw_rad)
        self._sync_human_collision_proxy_pose(human_id)
        self._sync_human_ego_camera_pose(human_id)

    def _assert_human_initial_poses(self) -> None:
        """Reject an animation runtime that lost authored Human world poses."""
        failures = []
        for human in self.episode.humans:
            state = self._human_body.get_state(str(human.id))
            if state is None:
                failures.append(
                    {
                        "human_id": str(human.id),
                        "reason": "runtime state is missing",
                    }
                )
                continue
            expected = [
                float(human.start_position[0]),
                float(human.start_position[1]),
            ]
            actual = [float(state.position[0]), float(state.position[1])]
            distance = math.dist(expected, actual)
            if state.pose_stale or distance > HUMAN_INITIAL_POSE_TOLERANCE_M:
                failures.append(
                    {
                        "human_id": str(human.id),
                        "expected_xy": expected,
                        "runtime_xy": actual,
                        "error_m": distance,
                        "pose_stale": bool(state.pose_stale),
                        "pose_source": state.metadata.get("pose_source"),
                        "pose_poll_error": state.metadata.get("pose_poll_error"),
                    }
                )
        if failures:
            raise RuntimeError(
                "HUMAN_INITIAL_POSE_CONTRACT_FAILED: the Isaac People runtime "
                "does not agree with the authored world pose; refusing to "
                f"create collision or evaluation evidence. findings={failures!r}"
            )

    def _sync_all_human_prims(self):
        for human_id in self._human_positions:
            self._sync_human_collision_proxy_pose(human_id)

    def _cleanup_human_collision_proxies(self):
        """Remove previous colliders before resetting the authoritative Human cast."""
        if not self._human_collision_proxy_paths:
            return
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError(
                "Cannot remove previous Human colliders without an Isaac stage"
            )
        for proxy_path in self._human_collision_proxy_paths.values():
            prim = stage.GetPrimAtPath(proxy_path)
            if prim and prim.IsValid() and not stage.RemovePrim(proxy_path):
                raise RuntimeError(
                    f"Failed to remove previous Human collider {proxy_path}"
                )

    def _setup_human_collision_proxies(self):
        """Create physical capsules that follow the authoritative Human pose."""
        if self._env is None or not self._human_positions:
            return
        try:
            import omni.usd
            from pxr import UsdGeom, UsdPhysics
        except Exception as exc:
            for human_id in self._human_positions:
                self._human_collision_errors[human_id] = str(exc)
                self._human_body.set_physical_status(
                    human_id,
                    "collision_proxy_unavailable",
                    error=str(exc),
                )
            raise RuntimeError("Human collision backend is unavailable") from exc

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            for human_id in self._human_positions:
                self._human_collision_errors[human_id] = "Isaac stage is unavailable."
                self._human_body.set_physical_status(
                    human_id,
                    "collision_proxy_unavailable",
                    error="Isaac stage is unavailable.",
                )
            raise RuntimeError("Human collision proxies require an active Isaac stage")

        UsdGeom.Xform.Define(stage, "/World/HumanCollisionProxies")
        for human_id in self._human_positions:
            proxy_path = f"/World/HumanCollisionProxies/{human_id}_BodyCapsule"
            try:
                capsule = UsdGeom.Capsule.Define(stage, proxy_path)
                self._human_collision_proxy_paths[human_id] = proxy_path
                capsule.GetRadiusAttr().Set(HUMAN_COLLISION_RADIUS_METERS)
                capsule.GetHeightAttr().Set(HUMAN_COLLISION_HEIGHT_METERS)
                axis_attr = capsule.GetAxisAttr()
                if axis_attr:
                    axis_attr.Set("Z")
                prim = capsule.GetPrim()
                UsdPhysics.CollisionAPI.Apply(prim)
                rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
                rigid_body.CreateKinematicEnabledAttr().Set(True)
                # Hide the capsule from rendering — it's physics-only.
                UsdGeom.Imageable(prim).GetVisibilityAttr().Set("invisible")
                self._human_collision_errors.pop(human_id, None)
                self._human_body.set_physical_status(
                    human_id,
                    f"{self.human_motion_backend}_collision_proxy",
                )
                self._sync_human_collision_proxy_pose(human_id)
            except Exception as exc:
                self._human_collision_errors[human_id] = str(exc)
                self._human_body.set_physical_status(
                    human_id,
                    "collision_proxy_error",
                    error=str(exc),
                )
                raise RuntimeError(
                    f"Failed to create Human collision proxy for {human_id}"
                ) from exc

    def _sync_human_collision_proxy_pose(self, human_id: str):
        proxy_path = self._human_collision_proxy_paths.get(human_id)
        position = self._human_positions.get(human_id)
        if self._env is None or proxy_path is None or position is None:
            return
        center = np.asarray(position, dtype=float) + np.asarray(
            [0.0, 0.0, HUMAN_COLLISION_HEIGHT_METERS * 0.5], dtype=float
        )
        try:
            from omni.isaac.core.prims.xform_prim import XFormPrim

            proxy = XFormPrim(proxy_path)
            proxy.set_world_pose(
                position=center,
                orientation=np.asarray(
                    self._yaw_to_quaternion(self._human_yaws.get(human_id, 0.0)),
                    dtype=float,
                ),
            )
        except Exception as exc:
            self._human_collision_errors[human_id] = (
                f"Failed to move human collision proxy {proxy_path}: {exc}"
            )
            self._human_body.set_physical_status(
                human_id,
                "collision_proxy_error",
                error=self._human_collision_errors[human_id],
            )
            raise RuntimeError(self._human_collision_errors[human_id]) from exc

    def _sync_human_ego_camera_pose(self, human_id: str):
        camera_path = self._human_ego_camera_paths.get(human_id)
        position = self._human_positions.get(human_id)
        if camera_path is None or position is None:
            return

        yaw = self._human_yaws.get(human_id, 0.0)
        camera_yaw = yaw + HUMAN_EGO_CAMERA_YAW_OFFSET_RAD
        forward = np.asarray(
            [np.cos(camera_yaw), np.sin(camera_yaw), 0.0],
            dtype=float,
        )
        eye = (
            np.asarray(position, dtype=float)
            + forward * HUMAN_EGO_FORWARD_OFFSET_METERS
            + np.asarray([0.0, 0.0, HUMAN_EGO_EYE_HEIGHT_METERS], dtype=float)
        )
        orientation = np.asarray(
            self._yaw_to_quaternion(camera_yaw),
            dtype=float,
        )
        target = eye + forward + np.asarray([0.0, 0.0, -0.05], dtype=float)
        pose_debug = {
            "camera_path": camera_path,
            "position": [float(v) for v in eye],
            "target": [float(v) for v in target],
            "yaw_rad": float(yaw),
            "camera_yaw_rad": float(camera_yaw),
            "yaw_offset_rad": float(HUMAN_EGO_CAMERA_YAW_OFFSET_RAD),
            "forward": [float(v) for v in forward],
            "orientation_wxyz": [float(v) for v in orientation],
            "fov_deg": float(HUMAN_EGO_CAMERA_HORIZONTAL_FOV_DEG),
            "eye_height_m": float(HUMAN_EGO_EYE_HEIGHT_METERS),
            "forward_offset_m": float(HUMAN_EGO_FORWARD_OFFSET_METERS),
        }
        self._human_ego_camera_poses[human_id] = dict(pose_debug)
        try:
            from isaacsim.core.utils.viewports import set_camera_view

            set_camera_view(
                eye=eye,
                target=target,
                camera_prim_path=camera_path,
            )
            pose_debug["pose_update_method"] = "set_camera_view"
            self._human_ego_camera_poses[human_id] = dict(pose_debug)
            self._human_ego_camera_errors.pop(human_id, None)
            return
        except Exception as exc:
            print(
                f"[WARN] set_camera_view failed for {camera_path}, "
                f"trying Camera.set_world_pose: {exc}",
                flush=True,
            )

        camera = self._human_ego_cameras.get(human_id)
        pose_target = camera.get("sensor") if isinstance(camera, dict) else camera
        if pose_target is not None and callable(
            getattr(pose_target, "set_world_pose", None)
        ):
            try:
                pose_target.set_world_pose(
                    position=eye,
                    orientation=orientation,
                    camera_axes="world",
                )
                pose_debug["pose_update_method"] = "camera_set_world_pose"
                self._human_ego_camera_poses[human_id] = dict(pose_debug)
                self._human_ego_camera_errors.pop(human_id, None)
                return
            except Exception as exc:
                self._human_ego_camera_errors[human_id] = (
                    f"Failed to update human ego camera pose {camera_path}: {exc}"
                )

        try:
            from omni.isaac.core.prims.xform_prim import XFormPrim

            prim = XFormPrim(camera_path)
            prim.set_world_pose(
                position=eye,
                orientation=orientation,
            )
            pose_debug["pose_update_method"] = "xform_prim_set_world_pose"
            self._human_ego_camera_poses[human_id] = dict(pose_debug)
            self._human_ego_camera_errors.pop(human_id, None)
        except Exception as exc:
            self._human_ego_camera_errors[human_id] = (
                f"Failed to update human ego camera pose {camera_path}: {exc}"
            )

    @staticmethod
    def _yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
        half = yaw * 0.5
        return (float(np.cos(half)), 0.0, 0.0, float(np.sin(half)))

    def _robot_position(self) -> list:
        position = self._obs.get("position")
        if position is None:
            raise RobotPoseUnavailableError(
                "Runtime observation is missing the robot base position."
            )
        try:
            if hasattr(position, "cpu"):
                position = position.cpu().numpy()
            vector = np.asarray(position, dtype=float).reshape(-1)
        except Exception as exc:
            raise RobotPoseUnavailableError(
                "Runtime robot base position could not be decoded."
            ) from exc
        if vector.shape[0] < 3 or not np.all(np.isfinite(vector[:3])):
            raise RobotPoseUnavailableError(
                f"Runtime robot base position is invalid: shape={vector.shape}."
            )
        return vector[:3].tolist()

    def _robot_yaw(self) -> float:
        """World yaw (rad) of the robot base from its root prim's local +X axis.

        Prefer the live TARSphere robot-base orientation.  Humanoid
        locomotion policies update the pelvis rigid body, while the USD root
        xform can remain at the spawn orientation; using the root alone makes
        rotate/turn validation report a stale yaw.
        """
        failure_reasons: list[str] = []
        orientation = self._obs.get("orientation")
        if orientation is not None:
            try:
                if hasattr(orientation, "cpu"):
                    orientation = orientation.cpu().numpy()
                quat = np.asarray(orientation, dtype=float).reshape(-1)
                if quat.shape[0] == 4 and np.all(np.isfinite(quat)):
                    w, x, y, z = quat.tolist()
                    yaw = float(
                        np.arctan2(
                            2.0 * (w * z + x * y),
                            1.0 - 2.0 * (y * y + z * z),
                        )
                    )
                    if np.isfinite(yaw):
                        return yaw
                failure_reasons.append(
                    "runtime orientation was not a finite quaternion"
                )
            except Exception as exc:
                failure_reasons.append(
                    f"runtime orientation decode failed: {type(exc).__name__}: {exc}"
                )
        else:
            failure_reasons.append("runtime orientation is missing")

        try:
            import omni.usd
            from pxr import Gf, UsdGeom

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                raise RobotPoseUnavailableError(
                    "USD stage unavailable while reading robot yaw."
                )
            root_path = self._robot_root_prim_path()
            if not root_path:
                raise RobotPoseUnavailableError(
                    "Cannot infer robot root prim path from active camera sensor."
                )
            prim = stage.GetPrimAtPath(root_path)
            if prim is None or not prim.IsValid():
                raise RobotPoseUnavailableError(
                    f"Robot root prim not found while reading yaw: {root_path}"
                )
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
            x_axis = matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
            yaw = float(np.arctan2(float(x_axis[1]), float(x_axis[0])))
            if np.isfinite(yaw):
                return yaw
        except Exception as exc:
            failure_reasons.append(
                f"USD yaw lookup failed: {type(exc).__name__}: {exc}"
            )
        raise RobotPoseUnavailableError(
            "Robot yaw is unavailable; " + "; ".join(failure_reasons)
        )

    def _robot_root_prim_path(self) -> Optional[str]:
        """Resolve the sole live robot root without assuming its asset name."""

        camera_path = self._agent_camera_path()
        if camera_path:
            parts = [part for part in str(camera_path).split("/") if part]
            try:
                robots_idx = parts.index("robots")
            except ValueError:
                pass
            else:
                if len(parts) > robots_idx + 1:
                    return "/" + "/".join(parts[: robots_idx + 2])

        try:
            import omni.usd
        except Exception:
            return None
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None
        robots_root = stage.GetPrimAtPath("/World/env_0/robots")
        if not robots_root or not robots_root.IsValid():
            return None
        robot_roots = [
            child for child in robots_root.GetChildren() if child and child.IsValid()
        ]
        if len(robot_roots) != 1:
            return None
        return str(robot_roots[0].GetPath())

    def _robot_pose(self) -> Dict[str, Any]:
        """Compact pose descriptor consumed by BEV / spatial memory modules.

        ``position`` is the robot base ``[x, y, z]``; ``yaw_rad`` its heading;
        ``camera_height_m`` / ``camera_pitch_deg`` describe the ego camera so a
        depth back-projection can map pixels into world coordinates.
        ``camera_pitch_deg`` is the camera's *live* total pitch — mount
        calibration plus the agent's commanded gaze tilt — because reporting the
        calibration alone would bias every back-projected coordinate by the tilt
        without raising anything.  ``camera_gaze_tilt_deg`` is only the agent's
        own share of it, carried separately because that is the number the agent
        commanded and can command back to 0; the mount calibration is not its
        doing and differs per robot. Missing live position or yaw is an
        infrastructure error: silently substituting an origin pose would
        corrupt spatial evidence and failure attribution.
        """
        position = self._robot_position()
        return {
            "position": position,
            "orientation": self._obs.get("orientation"),
            "yaw_rad": self._robot_yaw(),
            "camera_height_m": float(ROBOT_CAMERA_HEIGHT_M),
            "camera_pitch_deg": float(self._robot_camera_total_pitch_deg()),
            "camera_gaze_tilt_deg": float(self.camera_gaze_tilt_deg()),
            "body_posture": g1_body_posture_from_height_command(
                self._g1_body_height_command()
            ),
        }

    def _audit_lidar_collider_coverage(self) -> Dict[str, Any]:
        """Measure which scene surfaces can return a PhysX LiDAR beam."""

        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError(
                "Cannot audit LiDAR collider coverage because the USD stage "
                "is unavailable."
            )
        scene_root = stage.GetPrimAtPath("/World/env_0/scene")
        if not scene_root or not scene_root.IsValid():
            raise RuntimeError(
                "Cannot audit LiDAR collider coverage because "
                "/World/env_0/scene is missing."
            )

        renderable_prims = 0
        collidable_prims = 0
        for prim in Usd.PrimRange.AllPrims(scene_root):
            if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Gprim):
                continue
            renderable_prims += 1
            current = prim
            while current and current.IsValid():
                if current.HasAPI(UsdPhysics.CollisionAPI):
                    collidable_prims += 1
                    break
                if current == scene_root:
                    break
                current = current.GetParent()

        if renderable_prims == 0:
            raise RuntimeError(
                "LiDAR is enabled but the scene contains no renderable "
                "geometry to audit."
            )
        if collidable_prims == 0:
            raise RuntimeError(
                "LiDAR is enabled but no scene surface carries CollisionAPI; "
                "every ray would incorrectly look like open space."
            )
        coverage_ratio = collidable_prims / renderable_prims
        report = {
            "renderable_prims": renderable_prims,
            "collidable_prims": collidable_prims,
            "coverage_ratio": round(coverage_ratio, 4),
        }
        print(
            "[INFO] LiDAR collider coverage: "
            f"{collidable_prims}/{renderable_prims} renderable prims "
            f"({coverage_ratio * 100:.1f}%).",
            flush=True,
        )
        return report

    def _capture_lidar(self) -> Optional[Dict[str, Any]]:
        """Measure one body-frame LiDAR scan from current PhysX colliders."""

        position = self._robot_position()
        if position is None or len(position) < 2:
            return None
        try:
            import omni.physx

            query = omni.physx.get_physx_scene_query_interface()
        except Exception:
            return None
        if query is None:
            return None
        if self._lidar_collider_audit is None:
            self._lidar_collider_audit = self._audit_lidar_collider_coverage()

        floor_z = self._episode_floor_z()
        origin_z = floor_z + LIDAR_ORIGIN_HEIGHT_M
        yaw_rad = float(self._robot_yaw())
        robot_root = str(self._robot_root_prim_path() or "")
        elevations = beam_elevations_deg()
        ranges_m: list[Optional[float]] = []

        for bearing_deg in beam_azimuths_deg():
            world_angle = yaw_rad + math.radians(float(bearing_deg))
            surface_offset = g1_body_surface_offset_m(float(bearing_deg))
            nearest: Optional[float] = None
            for elevation_deg in elevations:
                distance = self._cast_lidar_beam(
                    query=query,
                    origin_xy=(float(position[0]), float(position[1])),
                    origin_z=origin_z,
                    world_angle_rad=world_angle,
                    elevation_deg=float(elevation_deg),
                    surface_offset_m=surface_offset,
                    floor_z=floor_z,
                    robot_root=robot_root,
                )
                if distance is not None and (nearest is None or distance < nearest):
                    nearest = distance
            ranges_m.append(nearest)

        try:
            lidar = build_lidar(
                ranges_m,
                max_range_m=LIDAR_MAX_RANGE_M,
                origin_height_m=LIDAR_ORIGIN_HEIGHT_M,
                elevations_deg=elevations,
            )
        except ValueError as exc:
            raise RuntimeError(f"LiDAR produced invalid measurements: {exc}") from exc
        payload = lidar.as_dict()
        payload["backend"] = "physx_raycast"
        payload["collider_coverage"] = dict(self._lidar_collider_audit)
        return payload

    def _cast_lidar_beam(
        self,
        *,
        query: Any,
        origin_xy: tuple[float, float],
        origin_z: float,
        world_angle_rad: float,
        elevation_deg: float,
        surface_offset_m: float,
        floor_z: float,
        robot_root: str,
    ) -> Optional[float]:
        """Return the nearest horizontal obstacle distance for one ray."""

        elevation_rad = math.radians(elevation_deg)
        cos_elevation = math.cos(elevation_rad)
        sin_elevation = math.sin(elevation_rad)
        if cos_elevation <= 1e-6:
            return None
        direction = (
            math.cos(world_angle_rad) * cos_elevation,
            math.sin(world_angle_rad) * cos_elevation,
            sin_elevation,
        )
        ray_limit = LIDAR_MAX_RANGE_M / cos_elevation
        if abs(sin_elevation) > 1e-6:
            vertical_limit = (
                floor_z + G1_STANDING_BODY_TOP_M
                if sin_elevation > 0.0
                else floor_z + LIDAR_FLOOR_TOLERANCE_M
            )
            surface_z = origin_z + sin_elevation * (surface_offset_m / cos_elevation)
            ray_limit = min(
                ray_limit,
                (vertical_limit - surface_z) / sin_elevation,
            )
        if ray_limit <= 1e-4:
            return None

        travelled = 0.0
        for _ in range(8):
            remaining = ray_limit - travelled
            if remaining <= 1e-4:
                return None
            start = surface_offset_m / cos_elevation + travelled
            origin = (
                origin_xy[0] + direction[0] * start,
                origin_xy[1] + direction[1] * start,
                origin_z + direction[2] * start,
            )
            try:
                hit = query.raycast_closest(origin, direction, float(remaining))
            except Exception:
                return None
            if not isinstance(hit, dict) or not bool(hit.get("hit", False)):
                return None
            measured = max(0.0, float(hit.get("distance", remaining)))
            slant_distance = travelled + measured
            hit_z = origin_z + direction[2] * (
                surface_offset_m / cos_elevation + slant_distance
            )
            hit_description = " ".join(
                str(hit.get(key, ""))
                for key in ("rigidBody", "collision", "body", "path")
            )
            ignored = (
                (bool(robot_root) and robot_root in hit_description)
                or f"{ROOM_OBSERVER_RECORDER_ROOT}/" in hit_description
                or hit_z <= floor_z + LIDAR_FLOOR_TOLERANCE_M
                or hit_z >= floor_z + G1_STANDING_BODY_TOP_M
            )
            if not ignored:
                return slant_distance * cos_elevation
            travelled += measured + 1e-3
        return None

    def _episode_floor_z(self) -> float:
        """Return the walking-floor elevation declared by the episode spawn."""

        spawn = self.episode.robot.start_position
        return float(spawn[2]) - G1_STANDING_HEIGHT_COMMAND_M

    def _robot_proprio(self) -> Dict[str, Any]:
        """Robot-owned proprioception for policy backends.

        This contains only the evaluated robot's own state.  It is safe for
        non-oracle policy inputs and intentionally excludes scene ground truth,
        target metadata, diagnostic checks, and human runtime internals.
        """
        state: Dict[str, Any] = {}
        joint_positions = self._obs.get("joint_positions")
        if joint_positions is not None:
            if hasattr(joint_positions, "cpu"):
                joint_positions = joint_positions.cpu()
            state["joint_positions"] = np.asarray(joint_positions).reshape(-1).tolist()
        controllers = self._obs.get("controllers")
        if isinstance(controllers, dict):
            state["controllers"] = self._proprio_jsonable(controllers)
        pose = self._robot_pose()
        state["base_pose"] = {
            "position": pose.get("position"),
            "yaw_rad": pose.get("yaw_rad"),
        }
        return state

    @staticmethod
    def _proprio_jsonable(value: Any) -> Any:
        if hasattr(value, "cpu"):
            value = value.cpu()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {
                str(key): IsaacAutonomousEnv._proprio_jsonable(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [IsaacAutonomousEnv._proprio_jsonable(item) for item in value]
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.bool_):
            return bool(value)
        return value

    def _apply_robot_start_pose(self) -> bool:
        """Teleport the robot to the current episode's start if needed.

        ``env.reset()`` restores the robot to the pose baked into the config at
        ``_create_env`` time, but a soft reset keeps the live scene and may leave
        the robot at the previous episode's final pose.  Compare against the
        current physical pose, not only against the configured spawn, so repeated
        same-start episodes still reset to their initial state.
        """
        if self._env is None or self._configured_robot_start is None:
            return False
        desired_pos = tuple(float(v) for v in self.episode.robot.start_position)
        desired_ori = self.episode.robot.start_orientation

        current_pos = self._robot_position()
        current_xy_error = float("inf")
        current_z_error = float("inf")
        if current_pos is not None and len(current_pos) >= 3:
            current_xy_error = math.hypot(
                float(current_pos[0]) - desired_pos[0],
                float(current_pos[1]) - desired_pos[1],
            )
            current_z_error = abs(float(current_pos[2]) - desired_pos[2])

        desired_matches_config = desired_pos == tuple(self._configured_robot_start)
        orientation_matches_config = (
            desired_ori is None and self._configured_robot_orientation is None
        ) or (
            desired_ori is not None
            and self._configured_robot_orientation is not None
            and tuple(float(v) for v in desired_ori)
            == tuple(float(v) for v in self._configured_robot_orientation)
        )
        if (
            current_xy_error <= 0.05
            and current_z_error <= 0.10
            and desired_matches_config
            and orientation_matches_config
        ):
            return False
        orientation = (
            [float(v) for v in desired_ori] if desired_ori is not None else None
        )
        teleported = self._teleport_robot(list(desired_pos), orientation)
        if teleported:
            print(
                "[INFO] Teleported robot to episode start "
                f"position={list(desired_pos)}, orientation={orientation} "
                f"(configured spawn was {list(self._configured_robot_start)}).",
                flush=True,
            )
        return teleported

    def _teleport_robot(
        self,
        position: list[float],
        orientation: Optional[list[float]] = None,
    ) -> bool:
        """Move the H1 articulation root to a world pose (immediate teleport).

        For a controlled H1 this moves the whole body rigidly; the balance
        controller keeps it standing on the next physics step.  Candidates are
        tried in order: the TARSphere ``robot.articulation`` wrapper (inherits
        ``set_world_pose`` via PoseMixin → ``unwrap().set_world_pose``), the inner
        isaacsim ``SingleArticulation``, then the pelvis rigid body.  Best-effort:
        logs a loud warning and returns False if none expose the API, so the
        episode still runs (from the configured spawn) rather than crashing.
        """
        runner = getattr(self._env, "runner", None)
        tasks = getattr(runner, "current_tasks", {}) or {}
        pos = np.asarray(position, dtype=float)
        ori = np.asarray(orientation, dtype=float) if orientation is not None else None
        teleported = False
        for task in tasks.values():
            for robot in getattr(task, "robots", {}).values():
                art = getattr(robot, "articulation", None)
                base = getattr(robot, "_robot_base", None)
                if base is None:
                    get_base = getattr(robot, "get_robot_base", None)
                    if callable(get_base):
                        try:
                            base = get_base()
                        except Exception:
                            base = None
                candidates = []
                if art is not None:
                    candidates.append(art)
                    inner = getattr(art, "_articulation", None)
                    if inner is not None and inner is not art:
                        candidates.append(inner)
                if base is not None:
                    candidates.append(base)
                for cand in candidates:
                    set_pose = getattr(cand, "set_world_pose", None)
                    if not callable(set_pose):
                        continue
                    try:
                        set_pose(pos, ori)
                        teleported = True
                        break
                    except Exception as exc:
                        print(
                            f"[WARN] set_world_pose on {type(cand).__name__} "
                            f"failed: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
        if not teleported:
            n_tasks = len(tasks)
            n_robots = sum(len(getattr(t, "robots", {}) or {}) for t in tasks.values())
            diag_parts = []
            for task in tasks.values():
                for robot in getattr(task, "robots", {}).values():
                    art = getattr(robot, "articulation", None)
                    diag_parts.append(
                        f"robot={type(robot).__name__} "
                        f"has_articulation={art is not None} "
                        f"articulation={type(art).__name__ if art is not None else None} "
                        f"has_set_world_pose={callable(getattr(art, 'set_world_pose', None)) if art is not None else 'n/a'}"
                    )
            print(
                "[WARN] No robot articulation exposed set_world_pose; robot stays "
                "at the configured spawn. Robot start override ignored. "
                f"tasks={n_tasks} robots={n_robots} "
                f"diag=[{'; '.join(diag_parts)}]",
                flush=True,
            )
        if teleported:
            self._invalidate_ego_snapshot()
        return teleported

    def set_viewer_mode(self, mode: str) -> None:
        """Switch the Kit viewport between first-person ego and third-person.

        Ego RGB-D sensors are created at init when ``enable_ego_sensors`` is
        True. This method only retargets the visible viewport; it does not
        create or destroy cameras.
        """
        if mode not in {"first_person", "third_person"}:
            raise ValueError(
                f"set_viewer_mode expects 'first_person' or 'third_person'; "
                f"got {mode!r}."
            )
        if self.headless:
            raise RuntimeError(
                "set_viewer_mode requires a visible Isaac viewport; the env "
                "was constructed headless."
            )
        if mode == "first_person":
            if not self.enable_ego_sensors:
                raise RuntimeError(
                    "set_viewer_mode('first_person') requires enable_ego_sensors=True "
                    "at IsaacAutonomousEnv construction so the robot camera exists."
                )
            if not self._agent_camera_path():
                raise RuntimeError(
                    "set_viewer_mode('first_person') cannot find the robot ego "
                    "camera prim. The sensors must be created at startup."
                )
        self.viewer_mode = mode
        self._configure_viewer()

    def _configure_viewer(self):
        """Set an initial useful camera position; the user can freely orbit afterwards."""
        if self.headless or self.viewer_mode == "free":
            return

        from isaacsim.core.utils.viewports import (
            set_active_viewport_camera,
            set_camera_view,
        )

        if self.viewer_mode == "first_person":
            camera_path = self._agent_camera_path()
            if camera_path:
                set_active_viewport_camera(camera_path)
            return

        # third_person: set an initial over-the-shoulder position once.
        # After this the user can freely orbit/pan/zoom the viewport with the mouse.
        position = self._robot_position()
        if position is None:
            position = list(self.episode.robot.start_position)
        position = np.asarray(position, dtype=float)

        # Tight over-the-shoulder camera for GRScenes rooms (2-3m wide).
        # 0.6m behind, 0.3m right, 0.35m above → looks 1.5m ahead at eye level.
        eye = position + np.asarray([-0.6, -0.3, 0.35])
        target = position + np.asarray([1.5, 0.0, 1.0])
        set_active_viewport_camera("/OmniverseKit_Persp")
        set_camera_view(
            eye=eye,
            target=target,
            camera_prim_path="/OmniverseKit_Persp",
        )

    def _agent_camera_path(self) -> Optional[str]:
        camera = self._robot_camera_sensor()
        return None if camera is None else camera.camera_prim_path

    def _robot_camera_sensor(self) -> Any:
        """Return the evaluated robot's TARSphere camera sensor, if any."""
        if self._env is None:
            return None
        for task in self._env.runner.current_tasks.values():
            for robot in task.robots.values():
                camera = robot.sensors.get("camera")
                if camera is not None:
                    return camera
        return None

    @staticmethod
    def _robot_camera_world_T_optical(camera_path: str) -> list[list[float]]:
        """Read the live USD camera pose in OpenCV optical coordinates."""
        import omni.usd
        from pxr import Gf, Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Cannot read robot ego camera: USD stage unavailable.")
        prim = stage.GetPrimAtPath(camera_path)
        if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
            raise RuntimeError(f"Robot ego Camera prim is invalid: {camera_path}")
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default(),
        )

        def unit(vector: Any) -> np.ndarray:
            value = np.asarray(vector, dtype=np.float64)
            norm = float(np.linalg.norm(value))
            if norm <= 0.0:
                raise RuntimeError(
                    f"Robot ego Camera transform has a zero axis: {camera_path}"
                )
            return value / norm

        # USD Camera: +X right, +Y up, -Z forward.
        # Evaluator optical frame: +X right, +Y down, +Z forward.
        right = unit(transform.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0)))
        down = unit(transform.TransformDir(Gf.Vec3d(0.0, -1.0, 0.0)))
        forward = unit(transform.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)))
        world_T_cam = np.eye(4, dtype=np.float64)
        world_T_cam[:3, :3] = np.stack([right, down, forward], axis=1)
        world_T_cam[:3, 3] = np.asarray(
            transform.ExtractTranslation(),
            dtype=np.float64,
        )
        return world_T_cam.tolist()
