"""Isaac adapter for applying Psi0 actions to an unlocked-waist G1."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tarsphere.embodiment.dex3 import (
    arm_controller_name,
    dex3_controller_name,
    waist_pitch_controller_name,
    waist_roll_controller_name,
    waist_yaw_controller_name,
)
from tarsphere.embodiment.g1_asset import (
    G1_STABLE_ROOT_PRIM,
    create_g1_with_unlocked_waist_roll,
    require_unlocked_waist_roll,
)
from tarsphere.policies.psi0.apply import g1_fixed_base_controller_action
from tarsphere.policies.psi0.contract import (
    PSI0_HEAD_CAMERA_HFOV_DEG,
    PSI0_OFFICIAL_HEAD_CAMERA_EXTRA_PITCH_DEG,
    PSI0_OFFICIAL_TABLETOP_STANCE_DISTANCE_M,
    Psi0ContractError,
    split_psi0_action,
)
from tarsphere.policies.psi0.observation import named_joint_positions
from tarsphere.infra.control.physical_contracts import G1_STANDING_HEIGHT_COMMAND_M

LIVE_ROBOTS_PARENT = "/World/env_0/robots"

PSI0_POSITION_CONTROLLER_NAMES: tuple[str, ...] = (
    waist_yaw_controller_name(),
    waist_roll_controller_name(),
    waist_pitch_controller_name(),
    arm_controller_name("left"),
    arm_controller_name("right"),
    dex3_controller_name("left"),
    dex3_controller_name("right"),
)


def psi0_unlocked_g1_usd(
    *,
    stable_g1_usd: Path | str,
    dex3_donor_usd: Path | str,
) -> Path:
    """Compose the unlocked-waist Dex3 G1 required by the Psi0 adapter."""

    return create_g1_with_unlocked_waist_roll(stable_g1_usd, dex3_donor_usd)


def psi0_g1_controller_cfgs() -> list[Any]:
    """Return the G1 controllers required for the complete Psi0 action layout."""

    from tarsphere.embodiment.robot_profiles import (
        load_simulator_robot_components,
    )
    from tarsphere.embodiment.robots.g1 import (
        left_arm_joint_controller_cfg,
        waist_roll_joint_controller_cfg,
        waist_yaw_joint_controller_cfg,
    )

    _, _, certified = load_simulator_robot_components("G1Robot")
    extras = (
        waist_yaw_joint_controller_cfg,
        waist_roll_joint_controller_cfg,
        left_arm_joint_controller_cfg,
    )
    return [*certified, *extras]


def require_psi0_position_controllers(robot: object) -> None:
    """Fail unless this robot instance registered the Psi0 position set."""

    controllers = getattr(robot, "controllers", None)
    if not isinstance(controllers, Mapping):
        raise Psi0ContractError(
            "Psi0 Isaac entry needs a robot with a controllers mapping."
        )
    present = set(controllers)
    missing = [name for name in PSI0_POSITION_CONTROLLER_NAMES if name not in present]
    if missing:
        raise Psi0ContractError(
            "Psi0 Isaac entry is missing position controllers "
            f"{missing}. Register them with psi0_g1_controller_cfgs; "
            "do not dump VLA joints into move_by_speed speed slots."
        )
    if "move_by_speed" not in present:
        raise Psi0ContractError("Psi0 stance hold requires move_by_speed on the robot.")


def live_isaac_stage() -> object:
    """Return the loaded USD stage. Fail if Isaac has no stage."""

    try:
        from omni.usd import get_context
    except ImportError as exc:
        raise Psi0ContractError(
            "Psi0 Isaac entry needs omni.usd to check unlocked waist roll."
        ) from exc
    stage = get_context().get_stage()
    if stage is None:
        raise Psi0ContractError(
            "Psi0 Isaac entry has no live USD stage. Spawn the unlocked "
            "overlay before apply."
        )
    return stage


def live_g1_root_prim(stage: object) -> str:
    """Return the remapped live G1 root. Overlay defaultPrim is not it."""

    parent = stage.GetPrimAtPath(LIVE_ROBOTS_PARENT)
    if not parent or not parent.IsValid():
        raise Psi0ContractError(f"No remapped robot parent at {LIVE_ROBOTS_PARENT}.")
    found: list[str] = []
    for child in parent.GetChildren():
        root = str(child.GetPath())
        imu = stage.GetPrimAtPath(f"{root}/imu_link")
        if imu is not None and imu.IsValid():
            found.append(root)
    if len(found) != 1:
        raise Psi0ContractError(
            "Expected exactly one remapped G1 with imu_link under "
            f"{LIVE_ROBOTS_PARENT}, found {found!r}. Overlay defaultPrim "
            f"{G1_STABLE_ROOT_PRIM} is not the live instance path."
        )
    return found[0]


def require_psi0_unlocked_waist_roll(
    stage: object | None = None,
    robot_root: str | None = None,
) -> tuple[object, str]:
    """Fail unless the live stage has revolute waist roll."""

    live_stage = stage if stage is not None else live_isaac_stage()
    root = robot_root if robot_root is not None else live_g1_root_prim(live_stage)
    try:
        require_unlocked_waist_roll(live_stage, root)
    except RuntimeError as exc:
        raise Psi0ContractError(str(exc)) from exc
    return live_stage, root


def first_runtime_robot(env: object) -> object:
    runner = getattr(env, "runner", None)
    if runner is None:
        raise Psi0ContractError("TARSphere runner is unavailable.")
    tasks = getattr(runner, "current_tasks", None) or {}
    for task in tasks.values():
        robots = getattr(task, "robots", None) or {}
        for robot in robots.values():
            return robot
    raise Psi0ContractError("No runtime robot was created by TARSphere.")


def official_tabletop_stance_xy(
    cup_xy: tuple[float, float],
    robot_yaw_rad: float,
    *,
    distance_m: float = PSI0_OFFICIAL_TABLETOP_STANCE_DISTANCE_M,
) -> tuple[float, float]:
    """Place the pelvis ``distance_m`` behind the cup along facing yaw.

    Official SpatialDR puts the G1 about 0.40 m from the tabletop object
    (``g1_wholebody_tabletop_grasp_mp.py:156-158``). Yaw 0 faces +X, so
    the stance is cup − distance · (cos yaw, sin yaw).
    """

    import math

    yaw = float(robot_yaw_rad)
    gap = float(distance_m)
    if not math.isfinite(gap) or gap <= 0.0:
        raise Psi0ContractError(
            f"Official tabletop stance distance must be positive, got {gap!r}."
        )
    return (
        float(cup_xy[0]) - gap * math.cos(yaw),
        float(cup_xy[1]) - gap * math.sin(yaw),
    )


def official_d435_extra_pitch_deg(tarsphere_mount_pitch_deg: float) -> float:
    """Return the mount correction for the official identity camera pose.

    SIMPLE defines an identity ``head_stereo`` orientation and applies its
    look-down through the standing torso pitch. TARSphere exposes the composed
    D435 mount pitch separately, so the adapter applies that mount exactly once.
    """

    import math

    mount = float(tarsphere_mount_pitch_deg)
    if not math.isfinite(mount):
        raise Psi0ContractError(
            f"TARSphere D435 mount pitch must be finite, got {tarsphere_mount_pitch_deg!r}."
        )
    if abs(mount - PSI0_OFFICIAL_HEAD_CAMERA_EXTRA_PITCH_DEG) <= 1e-6:
        return float(PSI0_OFFICIAL_HEAD_CAMERA_EXTRA_PITCH_DEG)
    return mount


def stance_hold_action(
    pelvis_height_m: float = G1_STANDING_HEIGHT_COMMAND_M,
    *,
    waist_pitch_rad: float | None = None,
) -> dict[str, Any]:
    """Zero-speed ONNX hold used before a Psi0 step. Not VLA output.

    ``waist_pitch_rad`` applies official SIMPLE stand torso pitch
    (``G1Wholebody.apply_action`` forces AMO ``command[5] = -0.15``).
    """

    payload: dict[str, Any] = {
        "move_by_speed": [0.0, 0.0, 0.0, float(pelvis_height_m)],
    }
    if waist_pitch_rad is not None:
        payload[waist_pitch_controller_name()] = [[float(waist_pitch_rad)]]
    return payload


def project_target_to_ego_pixel(
    *,
    camera_from_world: object,
    target_world_xyz: tuple[float, float, float],
    image_wh: tuple[int, int],
    horizontal_fov_deg: float = PSI0_HEAD_CAMERA_HFOV_DEG,
) -> tuple[float, float, float]:
    """Project a world point into the official D435 ego pixel frame."""

    import math

    import numpy as np

    width, height = int(image_wh[0]), int(image_wh[1])
    camera_t_world = np.asarray(camera_from_world, dtype=np.float64)
    if camera_t_world.shape != (4, 4):
        raise Psi0ContractError(
            f"Ego camera transform must be 4x4, got {camera_t_world.shape}."
        )
    target_camera = camera_t_world @ np.asarray(
        [*target_world_xyz, 1.0],
        dtype=np.float64,
    )
    depth_m = float(target_camera[2])
    focal_px = width / (2.0 * math.tan(math.radians(float(horizontal_fov_deg)) / 2.0))
    pixel_u = width / 2.0 + focal_px * float(target_camera[0]) / max(depth_m, 1e-6)
    pixel_v = height / 2.0 + focal_px * float(target_camera[1]) / max(depth_m, 1e-6)
    return float(pixel_u), float(pixel_v), depth_m


def require_cup_in_ego_rgb(
    *,
    camera_from_world: object,
    target_world_xyz: tuple[float, float, float],
    rgb: object,
    horizontal_fov_deg: float = PSI0_HEAD_CAMERA_HFOV_DEG,
) -> dict[str, float]:
    """Fail before /act when the known cup is outside D435 RGB."""

    import numpy as np

    frame = np.asarray(rgb)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise Psi0ContractError(f"Psi0 ego RGB must be HxWx3, got {frame.shape}.")
    height, width = int(frame.shape[0]), int(frame.shape[1])
    pixel_u, pixel_v, depth_m = project_target_to_ego_pixel(
        camera_from_world=camera_from_world,
        target_world_xyz=target_world_xyz,
        image_wh=(width, height),
        horizontal_fov_deg=horizontal_fov_deg,
    )
    if depth_m <= 0.0:
        raise Psi0ContractError(
            "Psi0 D435 points away from the cup: "
            f"depth_m={depth_m:.3f} pixel=({pixel_u:.1f},{pixel_v:.1f})."
        )
    # Keep the target away from the image boundary where only a partial object
    # or table edge may be visible.
    margin_x = width * 0.08
    margin_y = height * 0.08
    if not (
        margin_x <= pixel_u < width - margin_x
        and margin_y <= pixel_v < height - margin_y
    ):
        raise Psi0ContractError(
            "Psi0 D435 ego does not contain the cup. "
            f"pixel=({pixel_u:.1f},{pixel_v:.1f}) frame={width}x{height} "
            f"depth_m={depth_m:.3f}. Do not call /act on empty table/sky."
        )
    rgb_std = float(np.asarray(frame[:, :, :3], dtype=np.float32).std())
    if rgb_std < 2.0:
        raise Psi0ContractError(
            f"Psi0 ego RGB is nearly uniform before /act: std={rgb_std:.3f}."
        )
    return {
        "target_pixel_u": round(pixel_u, 2),
        "target_pixel_v": round(pixel_v, 2),
        "target_depth_m": round(depth_m, 4),
        "rgb_std": round(rgb_std, 4),
    }


def configure_psi0_table_camera(
    *,
    runtime_robot: object,
    camera_path: str,
    tarsphere_mount_pitch_deg: float,
    image_wh: tuple[int, int],
    horizontal_fov_deg: float = PSI0_HEAD_CAMERA_HFOV_DEG,
) -> dict[str, Any]:
    """Configure the TARSphere D435 for Psi0's identity head camera.

    The SIMPLE camera is mounted at the head origin with 110° horizontal FOV.
    This adapter applies the TARSphere mount correction without adding a
    separate harness gaze command.
    """

    import numpy as np

    from tarsphere.infra.simulation.isaac_env import (
        IsaacAutonomousEnv,
        camera_local_orientation_wxyz,
    )

    extra_pitch_deg = official_d435_extra_pitch_deg(tarsphere_mount_pitch_deg)
    prim = IsaacAutonomousEnv._require_robot_camera_prim(camera_path)
    rest_orientation = IsaacAutonomousEnv._camera_local_orientation_wxyz(prim)
    orientation = camera_local_orientation_wxyz(
        rest_orientation,
        extra_pitch_deg,
    )
    IsaacAutonomousEnv._set_robot_camera_pitch(
        camera_path,
        extra_pitch_deg,
        rest_orientation_wxyz=rest_orientation,
    )
    width, height = int(image_wh[0]), int(image_wh[1])
    IsaacAutonomousEnv._set_camera_fov(
        camera_path,
        float(horizontal_fov_deg),
        aspect_ratio=width / max(height, 1),
    )
    sensors = getattr(runtime_robot, "sensors", None)
    sensor = sensors.get("camera") if isinstance(sensors, dict) else None
    camera = getattr(sensor, "_camera", None) if sensor is not None else None
    xform_prim = getattr(camera, "prim", None)
    if xform_prim is None or not callable(getattr(xform_prim, "set_local_pose", None)):
        raise Psi0ContractError(
            "Cannot synchronize the Psi0 D435 with Isaac Fabric: "
            f"sensor={sensor!r} camera_path={camera_path!r}."
        )
    xform_prim.set_local_pose(
        orientation=np.asarray(orientation, dtype=np.float64),
    )
    return {
        "camera_path": camera_path,
        "tarsphere_mount_pitch_deg": float(tarsphere_mount_pitch_deg),
        "tarsphere_d435_mount_not_official": (
            abs(extra_pitch_deg - PSI0_OFFICIAL_HEAD_CAMERA_EXTRA_PITCH_DEG) > 1e-6
        ),
        "official_extra_pitch_deg": extra_pitch_deg,
        "gaze_tilt_deg": 0.0,
        "total_pitch_deg": extra_pitch_deg,
        "horizontal_fov_deg": float(horizontal_fov_deg),
        "image_wh": [width, height],
        "rest_orientation_wxyz": list(rest_orientation),
        "orientation_wxyz": list(orientation),
        "_xform_prim": xform_prim,
    }


def persist_psi0_table_camera(calibration: Mapping[str, Any]) -> None:
    """Keep the official-adjacent D435 pose in USD and Fabric."""

    import numpy as np

    from tarsphere.infra.simulation.isaac_env import IsaacAutonomousEnv

    IsaacAutonomousEnv._set_robot_camera_pitch(
        calibration["camera_path"],
        calibration["total_pitch_deg"],
        rest_orientation_wxyz=calibration["rest_orientation_wxyz"],
    )
    calibration["_xform_prim"].set_local_pose(
        orientation=np.asarray(
            calibration["orientation_wxyz"],
            dtype=np.float64,
        ),
    )


def live_camera_from_world(camera_path: str) -> object:
    """Return the live 4x4 optical camera-from-world matrix."""

    import numpy as np

    from tarsphere.infra.simulation.isaac_env import IsaacAutonomousEnv

    world_from_camera = np.asarray(
        IsaacAutonomousEnv._robot_camera_world_T_optical(camera_path),
        dtype=np.float64,
    )
    try:
        return np.linalg.inv(world_from_camera)
    except np.linalg.LinAlgError as exc:
        raise Psi0ContractError(
            "Psi0 D435 world-from-camera pose is not invertible."
        ) from exc


def apply_fixed_base_psi0_action_step(
    robot: object,
    action_row: object,
    *,
    stage: object | None = None,
    robot_root: str | None = None,
) -> dict[str, Any]:
    """Compile one official 36-D row onto the standing G1 controllers."""

    require_psi0_position_controllers(robot)
    live_stage, root = require_psi0_unlocked_waist_roll(stage, robot_root)
    current = named_joint_positions(robot)
    return g1_fixed_base_controller_action(
        split_psi0_action(action_row),
        current_joint_positions=current,
        stage=live_stage,
        robot_root=root,
    )
