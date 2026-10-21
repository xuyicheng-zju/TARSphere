"""Apply official SIMPLE Psi0Agent joints onto G1 position controllers.

The official 36-D row is absolute joint position, not Δq. Hands, arms, and
``torso_rpy`` (action 28–30) are commanded. ``height`` (action 31) is commanded as the absolute pelvis-height on
``move_by_speed[3]``. Official AMO instead receives
``d_height = action[31] - 0.75``; that tracker is the remaining residual.
Loco dims (vx, vy, vyaw, target_yaw) stay a zero-speed stance hold and
are never written into ``move_by_speed[0:3]``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tarsphere.embodiment.dex3 import (
    DEX3_JOINT_NAMES,
    arm_controller_name,
    dex3_controller_name,
    waist_pitch_controller_name,
    waist_roll_controller_name,
    waist_yaw_controller_name,
)
from tarsphere.embodiment.g1_asset import (
    G1_STABLE_ROOT_PRIM,
    require_unlocked_waist_roll,
)
from tarsphere.policies.psi0.contract import (
    PSI0_FROZEN_LOCO_DIM,
    PSI0_PELVIS_HEIGHT_MAX_M,
    PSI0_PELVIS_HEIGHT_MIN_M,
    PSI0_WAIST_JOINT_NAMES,
    Psi0ActionChunks,
    Psi0ContractError,
    split_psi0_action,
)
from tarsphere.infra.manipulation.g1_arm_ik import G1_UPPER_COMMAND_SCALE


def remap_right_hand_to_dex3(right_hand: tuple[float, ...]) -> tuple[float, ...]:
    """Map SIMPLE right-hand (thumb, index, middle) onto TARSphere Dex3 order."""

    if len(right_hand) != 7:
        raise Psi0ContractError(
            f"Psi0 right hand must have 7 joints, got {len(right_hand)}."
        )
    # Explicit named remap: thumb 0-2, middle 0-1, index 0-1.
    return (
        right_hand[0],
        right_hand[1],
        right_hand[2],
        right_hand[5],
        right_hand[6],
        right_hand[3],
        right_hand[4],
    )


def dex3_right_from_psi0(right_hand: tuple[float, ...]) -> tuple[float, ...]:
    """Public name for the official SIMPLE→Dex3 right-hand remap."""

    return remap_right_hand_to_dex3(right_hand)


def g1_fixed_base_controller_action(
    chunks: Psi0ActionChunks | object,
    *,
    current_joint_positions: Mapping[str, float],
    stage: object | None = None,
    robot_root: str = G1_STABLE_ROOT_PRIM,
) -> dict[str, Any]:
    """Return one TARSphere controller payload for a standing G1.

    When ``stage`` is supplied, fail unless waist roll is a revolute DOF.
    Offline mapper tests omit ``stage``.
    """

    if not isinstance(chunks, Psi0ActionChunks):
        chunks = split_psi0_action(chunks)
    if stage is not None:
        require_unlocked_waist_roll(stage, robot_root)
    if len(DEX3_JOINT_NAMES["left"]) != 7 or len(DEX3_JOINT_NAMES["right"]) != 7:
        raise Psi0ContractError("TARSphere Dex3 controller order is not 7-DoF.")
    left_hand = chunks.left_hand
    if len(left_hand) != 7:
        raise Psi0ContractError(
            f"Psi0 left hand must have 7 joints, got {len(left_hand)}."
        )
    right_hand = remap_right_hand_to_dex3(chunks.right_hand)
    left_arm = chunks.left_arm
    right_arm = chunks.right_arm
    if len(left_arm) != 7 or len(right_arm) != 7:
        raise Psi0ContractError("Psi0 arms must each have 7 joints.")
    if not current_joint_positions:
        raise Psi0ContractError(
            "Cannot apply Psi0 joints without live G1 proprioception."
        )
    missing_waist = [
        name for name in PSI0_WAIST_JOINT_NAMES if name not in current_joint_positions
    ]
    if missing_waist:
        raise Psi0ContractError(
            "Cannot apply official torso_rpy; live proprioception is missing "
            f"{missing_waist}. Load create_g1_with_unlocked_waist_roll "
            "so waist_roll_joint is a measured revolute DOF."
        )
    height_m = float(chunks.height_m)
    if not (PSI0_PELVIS_HEIGHT_MIN_M <= height_m <= PSI0_PELVIS_HEIGHT_MAX_M):
        raise Psi0ContractError(
            "Psi0 action[31] is not a pelvis height for move_by_speed[3]: "
            f"{height_m!r} m (required "
            f"{PSI0_PELVIS_HEIGHT_MIN_M}–{PSI0_PELVIS_HEIGHT_MAX_M} m)."
        )
    waist_roll, waist_pitch, waist_yaw = (
        float(chunks.torso_rpy[0]),
        float(chunks.torso_rpy[1]),
        float(chunks.torso_rpy[2]),
    )
    # Official 15-D ONNX upper: waist_yaw position command * 0.25 + default.
    # Dedicated yaw JointController carries the Psi0 position target.
    upper_body_command = [
        waist_yaw / G1_UPPER_COMMAND_SCALE,
        *(value / G1_UPPER_COMMAND_SCALE for value in left_arm),
        *(value / G1_UPPER_COMMAND_SCALE for value in right_arm),
    ]
    if len(upper_body_command) != 15:
        raise Psi0ContractError(
            "Psi0 ONNX upper-body command must be 15-D "
            f"(waist_yaw + 14 arm), got {len(upper_body_command)}."
        )
    return {
        "move_by_speed": [
            0.0,
            0.0,
            0.0,
            height_m,
            upper_body_command,
        ],
        waist_yaw_controller_name(): [[waist_yaw]],
        waist_roll_controller_name(): [[waist_roll]],
        waist_pitch_controller_name(): [[waist_pitch]],
        arm_controller_name("left"): [list(left_arm)],
        arm_controller_name("right"): [list(right_arm)],
        dex3_controller_name("left"): [list(left_hand)],
        dex3_controller_name("right"): [list(right_hand)],
        "psi0_frozen_loco_dim": PSI0_FROZEN_LOCO_DIM,
    }
