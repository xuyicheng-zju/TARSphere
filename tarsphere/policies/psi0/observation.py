"""Build the official SIMPLE Psi0Agent observation from a live G1."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from tarsphere.policies.psi0.contract import (
    PSI0_DEFAULT_HEIGHT_M,
    PSI0_IMAGE_KEY,
    PSI0_LEFT_ARM_JOINT_NAMES,
    PSI0_LEFT_HAND_JOINT_NAMES,
    PSI0_PROPRIO_JOINT_NAMES,
    PSI0_RIGHT_ARM_JOINT_NAMES,
    PSI0_RIGHT_HAND_JOINT_NAMES,
    PSI0_STATE_DIM,
    PSI0_WAIST_JOINT_NAMES,
    Psi0ContractError,
    require_psi0_state,
)


def require_head_rgb(frame: object) -> np.ndarray:
    """Return one official head RGB ``(H, W, 3)`` uint8 array."""

    image = np.asarray(frame)
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise Psi0ContractError(
                f"Psi0 head RGB batch must have size 1, got {image.shape}."
            )
        image = image[0]
    if image.ndim != 3 or image.shape[2] < 3:
        raise Psi0ContractError(f"Psi0 head RGB must be HxWx3, got {image.shape}.")
    if image.shape[0] < 8 or image.shape[1] < 8:
        raise Psi0ContractError(f"Psi0 head RGB is too small: {image.shape}.")
    rgb = image[:, :, :3]
    if np.issubdtype(rgb.dtype, np.floating):
        if rgb.max() <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0.0, 255.0)
    return np.ascontiguousarray(rgb.astype(np.uint8))


def torso_rpyh_from_named_joints(
    joint_positions: Mapping[str, float],
    *,
    pelvis_height_m: float,
) -> tuple[float, float, float, float]:
    """Pack official state[28:32] from live waist joints and pelvis height."""

    missing = [name for name in PSI0_WAIST_JOINT_NAMES if name not in joint_positions]
    if missing:
        raise Psi0ContractError(
            f"Live G1 is missing Psi0 waist joints {missing}. "
            "Load create_g1_with_unlocked_waist_roll so "
            "waist_roll_joint is a measured revolute DOF."
        )
    height = float(pelvis_height_m)
    if height <= 0.0:
        raise Psi0ContractError(
            f"Psi0 observation pelvis height must be positive, got {height!r}."
        )
    return (
        float(joint_positions["waist_roll_joint"]),
        float(joint_positions["waist_pitch_joint"]),
        float(joint_positions["waist_yaw_joint"]),
        height,
    )


def proprio_from_named_joints(
    joint_positions: Mapping[str, float],
    *,
    torso_rpyh: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Pack official 32-D ``states`` by joint name, not by TARSphere dof index."""

    missing = [name for name in PSI0_PROPRIO_JOINT_NAMES if name not in joint_positions]
    if missing:
        raise Psi0ContractError(f"Live G1 is missing Psi0 proprio joints {missing}.")
    if torso_rpyh is None:
        waist_present = all(name in joint_positions for name in PSI0_WAIST_JOINT_NAMES)
        if waist_present:
            rpyh = torso_rpyh_from_named_joints(
                joint_positions,
                pelvis_height_m=PSI0_DEFAULT_HEIGHT_M,
            )
        else:
            rpyh = (0.0, 0.0, 0.0, PSI0_DEFAULT_HEIGHT_M)
    else:
        rpyh = torso_rpyh
    if len(rpyh) != 4:
        raise Psi0ContractError("Psi0 torso_rpyh must be (roll, pitch, yaw, height).")
    values = [float(joint_positions[name]) for name in PSI0_PROPRIO_JOINT_NAMES]
    values.extend(float(item) for item in rpyh)
    packed = require_psi0_state(values)
    if len(packed) != PSI0_STATE_DIM:
        raise Psi0ContractError(
            f"Packed Psi0 state must be {PSI0_STATE_DIM}-D, got {len(packed)}."
        )
    return np.asarray(packed, dtype=np.float32)[None]


def observation_from_g1(
    *,
    head_rgb: object,
    joint_positions: Mapping[str, float],
    torso_rpyh: tuple[float, float, float, float] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return official ``image`` and ``state`` dicts for POST /act."""

    image = {PSI0_IMAGE_KEY: require_head_rgb(head_rgb)}
    state = {
        "states": proprio_from_named_joints(
            joint_positions,
            torso_rpyh=torso_rpyh,
        ),
    }
    return image, state


def named_joint_positions(robot: object) -> dict[str, float]:
    """Read every Psi0-actuated joint from a live TARSphere articulation."""

    articulation = getattr(robot, "articulation", None)
    if articulation is None:
        raise Psi0ContractError("Psi0 observation needs a robot articulation.")
    positions = articulation.get_joint_positions()
    names = getattr(articulation, "dof_names", None)
    if positions is None or not names:
        raise Psi0ContractError("Live G1 articulation has no named joint positions.")
    values = np.asarray(
        positions.cpu() if hasattr(positions, "cpu") else positions,
        dtype=np.float64,
    ).reshape(-1)
    name_to_index = {str(name): index for index, name in enumerate(names)}
    required = (*PSI0_PROPRIO_JOINT_NAMES, *PSI0_WAIST_JOINT_NAMES)
    missing = [name for name in required if name not in name_to_index]
    if missing:
        hint = ""
        if "waist_roll_joint" in missing:
            hint = (
                " Load create_g1_with_unlocked_waist_roll so "
                "waist_roll_joint is a measured revolute DOF."
            )
        raise Psi0ContractError(
            f"Live G1 dof list is missing Psi0 joints {missing}.{hint}"
        )
    return {name: float(values[name_to_index[name]]) for name in required}


# Names kept here so tests can assert the official SIMPLE remap.
_ = (
    PSI0_LEFT_HAND_JOINT_NAMES,
    PSI0_RIGHT_HAND_JOINT_NAMES,
    PSI0_LEFT_ARM_JOINT_NAMES,
    PSI0_RIGHT_ARM_JOINT_NAMES,
)
