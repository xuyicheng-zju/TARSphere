"""Psi0 SIMPLE tabletop action contract for G1.

The released tabletop policy returns 36 values. Hands and arms occupy the
first 28; waist roll, pitch, yaw and pelvis height occupy indices 28--31.
Those four values are absolute targets. The remaining locomotion channels
are held at zero by this adapter. The checkpoint and coordinate conventions
below follow the SIMPLE G1WholebodyTabletopGraspMP implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass

PSI0_ACTION_DIM = 36
PSI0_STATE_DIM = 32
PSI0_HAND_DIM = 14
PSI0_ARM_DIM = 14
# vx, vy, vyaw, target_yaw. Torso and height are applied, not frozen.
PSI0_FROZEN_LOCO_DIM = 4
PSI0_IMAGE_KEY = "rgb_head_stereo_left"
PSI0_DATASET_NAME = "simple"
# Psi0Agent.get_action fallback when eval does not pass task.instruction.
PSI0_DEFAULT_INSTRUCTION = "bend to pick up the object"
# SIMPLE G1WholebodyTabletopGraspMP language DR (official eval instruction).
PSI0_TABLETOP_EVAL_INSTRUCTION = "grasp the object on the tabletop."
PSI0_DEFAULT_HEIGHT_M = 0.75
# G1Wholebody.z_offset — official floating-base stand height, metres.
PSI0_OFFICIAL_Z_OFFSET_M = 0.785
# G1Wholebody.apply_action loco_command overwrites AMO command[5] (torso pitch).
# SIMPLE src/simple/robots/g1_wholebody.py:329 — ``self.command[5] = -0.15``.
PSI0_OFFICIAL_STAND_TORSO_PITCH_RAD = -0.15
# Official head_stereo is identity on the head. SIMPLE
# src/simple/tasks/g1_wholebody_tabletop_grasp_mp.py:104-116
# (D435i, eye_in_head, 640×360, fov=110°, pose position=[0,0,0]) and
# src/simple/robots/g1_wholebody.py:58 (head_camera_orientation=[1,0,0,0]).
# Extra rotateX on the USD Camera rest is 0. TARSphere G1Robot.camera_mount_pitch_deg
# is -15; that extra look-down is not official and must not be stacked with gaze.
PSI0_OFFICIAL_HEAD_CAMERA_EXTRA_PITCH_DEG = 0.0
# TARSphere profile extra mount. Recorded so the sidecar can fail-loud and undo it.
PSI0_TARSPHERE_D435_MOUNT_PITCH_DEG = -15.0
# Official tabletop XY: robot_region x∈[-1.1,-1.0], target x∈[-0.67,-0.62]
# (g1_wholebody_tabletop_grasp_mp.py:156-158). Mid-to-mid ≈ 0.405 m.
PSI0_OFFICIAL_TABLETOP_STANCE_DISTANCE_M = 0.40
# TARSphere stance distance that keeps the tabletop target inside the composed
# D435's 110° field of view after applying the mount correction.
PSI0_TARSPHERE_D435_VISIBLE_STANCE_DISTANCE_M = 0.75
# Official-adjacent pelvis height window for action[31] → move_by_speed[3].
# AMO receives d_height = action[31] - 0.75; TARSphere commands the absolute value.
PSI0_PELVIS_HEIGHT_MIN_M = 0.40
PSI0_PELVIS_HEIGHT_MAX_M = 1.10
PSI0_TABLETOP_TASK = "G1WholebodyTabletopGrasp-v0"
# SIMPLE task metadata.render_hz and serve_psi0_simple --action-exec-horizon.
PSI0_CONTROL_HZ = 30
PSI0_ACTION_EXEC_HORIZON = 24
# Official head_stereo StereoCameraCfg (D435i, eye_in_head).
PSI0_HEAD_CAMERA_WIDTH = 640
PSI0_HEAD_CAMERA_HEIGHT = 360
PSI0_HEAD_CAMERA_HFOV_DEG = 110.0
PSI0_TABLETOP_HF_REPO = "USC-PSI-Lab/psi-model"
PSI0_TABLETOP_REMOTE_DIR = (
    "psi0/simple-checkpoints/"
    "g1wholebodytabletopgrasp-v0.simple.flow1000.cosine.lr1.0e-04.b128.gpus8.2603181503"
)
PSI0_TABLETOP_CKPT_STEP = 40000
# This ckpt's serve transform, not the HE 240x320 default.
PSI0_TABLETOP_IMAGE_HW = (180, 320)
TARSPHERE_PSI0_ENDPOINT_ENV = "TARSPHERE_PSI0_ENDPOINT"
TARSPHERE_PSI0_TIMEOUT_S_ENV = "TARSPHERE_PSI0_TIMEOUT_S"
DEFAULT_PSI0_PORT = 22085
DEFAULT_PSI0_TIMEOUT_S = 60.0
SONIC_CKPT_MARKERS = ("sonic-wbcbox", "/sonic/", "serve_psi0_sonic")

# Official torso_rpy order: roll, pitch, yaw. Matches SIMPLE Psi0Agent.
PSI0_WAIST_JOINT_NAMES: tuple[str, str, str] = (
    "waist_roll_joint",
    "waist_pitch_joint",
    "waist_yaw_joint",
)
# Psi0 / SIMPLE left-hand action order is already TARSphere Dex3 order.
PSI0_LEFT_HAND_JOINT_NAMES: tuple[str, ...] = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
)
# Right hand is stored in SIMPLE robot qpos order (thumb, index, middle).
PSI0_RIGHT_HAND_JOINT_NAMES: tuple[str, ...] = (
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)
PSI0_LEFT_ARM_JOINT_NAMES: tuple[str, ...] = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
PSI0_RIGHT_ARM_JOINT_NAMES: tuple[str, ...] = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
PSI0_PROPRIO_JOINT_NAMES: tuple[str, ...] = (
    *PSI0_LEFT_HAND_JOINT_NAMES,
    *PSI0_RIGHT_HAND_JOINT_NAMES,
    *PSI0_LEFT_ARM_JOINT_NAMES,
    *PSI0_RIGHT_ARM_JOINT_NAMES,
)


class Psi0ContractError(RuntimeError):
    """Official Psi0 observation or action contract was violated."""


@dataclass(frozen=True)
class Psi0ActionChunks:
    """One denormalized 36-D Psi0 row, split on the official SIMPLE layout.

    Attributes:
        left_hand: Absolute left Dex3 joints in Psi0/TARSphere thumb-middle-index order.
        right_hand: Absolute right Dex3 joints in SIMPLE thumb-index-middle order.
        left_arm: Absolute left arm joints, shoulder to wrist-yaw.
        right_arm: Absolute right arm joints, shoulder to wrist-yaw.
        torso_rpy: Absolute torso roll/pitch/yaw. Applied to unlocked G1 waist.
        height_m: Absolute pelvis height. Commanded on ``move_by_speed[3]``.
        vx: Linear x command. Frozen; never sent to ONNX speed.
        vy: Linear y command. Frozen; never sent to ONNX speed.
        vyaw: Turning flag / yaw rate. Frozen; never sent to ONNX speed.
        target_yaw: Heading command. Frozen; never sent to ONNX speed.
    """

    left_hand: tuple[float, ...]
    right_hand: tuple[float, ...]
    left_arm: tuple[float, ...]
    right_arm: tuple[float, ...]
    torso_rpy: tuple[float, float, float]
    height_m: float
    vx: float
    vy: float
    vyaw: float
    target_yaw: float


def require_not_sonic_checkpoint(run_dir: str) -> None:
    """Fail if the serve path is a live-SONIC whole-body checkpoint."""

    text = str(run_dir).lower()
    hit = next((marker for marker in SONIC_CKPT_MARKERS if marker in text), None)
    if hit is None:
        return
    raise Psi0ContractError(
        f"Checkpoint {run_dir!r} requires live SONIC ({hit!r}). "
        "This smoke freezes System-0 loco and has no SONIC executor."
    )


def split_psi0_action(row: object) -> Psi0ActionChunks:
    """Split one official 36-D absolute action. Do not treat it as Δq."""

    values = _finite_row(row, PSI0_ACTION_DIM, "action")
    return Psi0ActionChunks(
        left_hand=tuple(values[0:7]),
        right_hand=tuple(values[7:14]),
        left_arm=tuple(values[14:21]),
        right_arm=tuple(values[21:28]),
        torso_rpy=(values[28], values[29], values[30]),
        height_m=values[31],
        vx=values[32],
        vy=values[33],
        vyaw=values[34],
        target_yaw=values[35],
    )


def require_psi0_state(row: object) -> tuple[float, ...]:
    """Return the official 32-D proprio row. Server pads it to 36."""

    return _finite_row(row, PSI0_STATE_DIM, "state")


def _finite_row(row: object, expected: int, label: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in row)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise Psi0ContractError(
            f"Psi0 {label} must be a finite {expected}-D vector."
        ) from exc
    if len(values) != expected:
        raise Psi0ContractError(
            f"Psi0 {label} must have {expected} dims, got {len(values)}."
        )
    if not all(math.isfinite(item) for item in values):
        raise Psi0ContractError(f"Psi0 {label} contains a non-finite value.")
    return values
