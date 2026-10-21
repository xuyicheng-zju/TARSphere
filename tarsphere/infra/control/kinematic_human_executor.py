"""Visible Human motion without AnimPeople command queues or NavMesh."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

from tarsphere.infra.control.human_body import HumanBodyResult, HumanBodyState
from tarsphere.infra.control.kinematic_human_motion import KinematicHumanMotion


class KinematicHumanExecutor:
    """Execute validated Human routes; all poses use world heading radians.

    ``route_resolver`` must return the complete collision-checked XYZ route
    including the requested destination, using the Human's own clearance.
    ``characters`` owns visible assets; the environment's ``pose_sync`` owns
    collision and sensing synchronization. Neither may silently ignore errors.
    """

    def __init__(
        self,
        *,
        characters: Any,
        route_resolver: Callable,
        pose_sync: Callable,
        default_speed_mps: float = 1.0,
    ):
        if not math.isfinite(default_speed_mps) or default_speed_mps <= 0:
            raise ValueError("Human default speed must be finite and positive")
        self._characters = characters
        self._route_resolver = route_resolver
        self._pose_sync = pose_sync
        self._speed = float(default_speed_mps)
        self._states: dict[str, HumanBodyState] = {}
        self._motions: dict[str, tuple[KinematicHumanMotion, float]] = {}

    def set_character_assets_path(self, path: str) -> None:
        self._characters.set_assets_path(path)

    def reset(self, humans: Iterable[Any]) -> None:
        humans = tuple(humans)
        ids = [str(human.id) for human in humans]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate Human identities in reset")
        self._characters.clear()
        self._states.clear()
        self._motions.clear()
        for human in humans:
            scene_bound = bool(
                str(getattr(human, "scene_person_prim_path", "") or "").strip()
            )
            state = HumanBodyState(
                human_id=str(human.id),
                position=[float(value) for value in human.start_position],
                yaw_rad=float(human.start_yaw),
                body_status=(
                    "scene_bound_static_person"
                    if scene_bound
                    else "kinematic_character"
                ),
                physical_status="collision_proxy_pending",
                metadata={
                    "pose_source": (
                        "authored_scene_person_prim"
                        if scene_bound
                        else "kinematic_physics_clock"
                    )
                },
            )
            self._characters.spawn(human)
            self._states[state.human_id] = state
            self._sync(state.human_id)

    def get_state(self, human_id: str) -> HumanBodyState | None:
        return self._states.get(human_id)

    def all_states(self) -> dict[str, HumanBodyState]:
        return dict(self._states)

    def position(self, human_id: str) -> list[float] | None:
        state = self.get_state(human_id)
        return None if state is None else list(state.position)

    def yaw(self, human_id: str) -> float:
        return self._states[human_id].yaw_rad

    def set_physical_status(
        self, human_id: str, status: str, *, error: str = ""
    ) -> None:
        state = self._states[human_id]
        state.physical_status = status
        state.last_error = error

    def apply_action(self, human_id: str, action: dict[str, Any]) -> HumanBodyResult:
        state = self._states[human_id]
        command = str(action.get("command", ""))
        if command in {"speak", "ask", "report_done"}:
            return HumanBodyResult(human_id, command, True, state=state)
        if command == "idle":
            self._motions.pop(human_id, None)
            state.active_action = ""
            state.target_position = None
            state.action_status = "idle"
            return HumanBodyResult(human_id, command, True, state=state)
        if command != "goto":
            return HumanBodyResult(
                human_id,
                command,
                False,
                f"{command!r} is not supported by kinematic Human motion",
                state,
            )
        if self._characters.is_scene_bound(human_id):
            return HumanBodyResult(
                human_id,
                command,
                False,
                "A scene-bound static Human cannot execute goto.",
                state,
            )
        target = action["target"]
        route = self._route_resolver(human_id, state.position, target, action)
        trajectory = KinematicHumanMotion(
            start=state.position,
            waypoints=route,
            speed_mps=float(action.get("speed_mps", self._speed)),
            initial_yaw_rad=state.yaw_rad,
            final_yaw_rad=action.get("yaw_rad"),
        )
        if math.dist(trajectory.points[-1], target) > 1e-6:
            raise ValueError("Human route must end at the requested destination")
        self._motions[human_id] = (trajectory, 0.0)
        state.active_action = "goto"
        state.target_position = list(trajectory.points[-1])
        state.action_status = "ongoing"
        state.metadata["route_world_xyz"] = [list(point) for point in trajectory.points]
        state.metadata["speed_mps"] = trajectory.speed_mps
        return HumanBodyResult(human_id, command, True, state=state)

    def step(self, dt: float) -> None:
        dt = float(dt)
        if not math.isfinite(dt) or dt < 0:
            raise ValueError("Human physics dt must be finite and nonnegative")
        for human_id, (trajectory, elapsed) in tuple(self._motions.items()):
            elapsed += dt
            pose = trajectory.sample(elapsed)
            state = self._states[human_id]
            state.position = list(pose.position)
            state.yaw_rad = pose.yaw_rad
            state.metadata["motion_elapsed_s"] = elapsed
            state.metadata["motion_distance_m"] = pose.distance_m
            endpoint_error_m = math.dist(
                state.position[:2],
                trajectory.points[-1][:2],
            )
            state.metadata["endpoint_error_m"] = endpoint_error_m
            if pose.completed:
                state.action_status = "succeed"
                state.active_action = ""
                self._motions.pop(human_id)
            else:
                self._motions[human_id] = (trajectory, elapsed)
            self._sync(human_id)

    def _sync(self, human_id: str) -> None:
        state = self._states[human_id]
        self._characters.set_pose(human_id, state.position, state.yaw_rad)
        self._pose_sync(human_id, state)

    def has_active_motion(self) -> bool:
        return bool(self._motions)

    def active_motion_ids(self) -> list[str]:
        return list(self._motions)

    def pump_motion_window(self, *, physics_tick: int, pump_every: int) -> int:
        # No Kit animation clock is needed. Native rendering remains external.
        return 0


class UsdKinematicCharacters:
    """Load visible Humans for deterministic root-pose motion.

    Scene-native people keep their authored pose.  Catalog characters ship in
    a skeletal bind pose, so a small static animation lowers both arms before
    the first rendered observation.  This does not attach AnimPeople, advance
    an animation clock, or change the kinematic route executor.
    """

    _IDLE_ARM_DROP_DEG = 80.0

    def __init__(self, *, default_character: str):
        self._default_character = default_character
        self._assets_path = ""
        self._paths: dict[str, str] = {}
        self._owned_paths: set[str] = set()
        self._scene_bound_ids: set[str] = set()

    def set_assets_path(self, path: str) -> None:
        self._assets_path = path

    def clear(self) -> None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Kinematic Human reset requires a live USD stage")
        failures: list[str] = []
        for path in self._owned_paths:
            if not stage.GetPrimAtPath(path).IsValid():
                continue
            try:
                removed = stage.RemovePrim(path)
            except Exception as exc:
                failures.append(f"{path}: {exc}")
                continue
            if not removed:
                failures.append(path)
        if failures:
            raise RuntimeError(
                "Could not remove previous Human asset(s): "
                + "; ".join(sorted(failures))
            )
        self._paths.clear()
        self._owned_paths.clear()
        self._scene_bound_ids.clear()

    def spawn(self, human: Any) -> None:
        from isaacsim.core.utils.semantics import add_update_semantics
        from isaacsim.replicator.agent.core.stage_util import CharacterUtil

        from tarsphere.humans.character_catalog import resolve_character_usd

        source_path = str(getattr(human, "scene_person_prim_path", "") or "").strip()
        if source_path:
            self._bind_scene_person(human, source_path, add_update_semantics)
            return
        if not self._assets_path:
            raise RuntimeError("Kinematic Human character asset root is not configured")
        asset = resolve_character_usd(
            human.character_name or self._default_character,
            character_assets_root=self._assets_path,
        )
        prim = CharacterUtil.load_character_usd_to_stage(
            str(asset),
            list(human.start_position),
            0,
            str(human.id),
        )
        if prim is None or not prim.IsValid():
            raise RuntimeError(f"Could not spawn kinematic Human {human.id}")
        self._paths[str(human.id)] = str(prim.GetPath())
        self._owned_paths.add(str(prim.GetPath()))
        add_update_semantics(prim, str(human.id), "class")
        self._apply_static_idle_pose(str(human.id), prim)

    def _apply_static_idle_pose(self, human_id: str, root_prim: Any) -> None:
        """Author a relaxed, time-independent pose for one catalog avatar.

        Isaac People catalog USDs are authored in a T-shaped bind pose.  The
        animation backend replaces it through an AnimationGraph; the default
        kinematic backend deliberately has no graph or NavMesh.  Bind a local
        ``UsdSkelAnimation`` that copies the asset's rest transforms and only
        lowers the upper arms.  Exact joint suffixes are required so an asset
        with an incompatible rig fails visibly instead of reverting to a
        misleading T-pose.
        """
        import omni.usd
        from pxr import Gf, Usd, UsdSkel

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Kinematic Human idle pose requires a live USD stage")

        skeleton_prims = [
            prim for prim in Usd.PrimRange(root_prim) if prim.IsA(UsdSkel.Skeleton)
        ]
        if len(skeleton_prims) != 1:
            raise RuntimeError(
                f"Kinematic Human {human_id!r} must contain exactly one skeleton; "
                f"found {len(skeleton_prims)} below {root_prim.GetPath()}."
            )

        skeleton_prim = skeleton_prims[0]
        skeleton = UsdSkel.Skeleton(skeleton_prim)
        joints = list(skeleton.GetJointsAttr().Get() or [])
        rest = skeleton.GetRestTransformsAttr().Get()
        if not joints or rest is None or len(rest) != len(joints):
            raise RuntimeError(
                f"Kinematic Human {human_id!r} has an incomplete skeleton rest pose."
            )
        joint_indices = {
            str(joint).rsplit("/", 1)[-1]: index for index, joint in enumerate(joints)
        }
        required = {"L_Upperarm", "R_Upperarm"}
        missing = sorted(required - joint_indices.keys())
        if missing:
            raise RuntimeError(
                f"Kinematic Human {human_id!r} cannot author the catalog idle pose; "
                f"missing joints: {missing}."
            )

        translations, rotations, scales = UsdSkel.DecomposeTransforms(rest)
        rotations = list(rotations)
        for joint_name, angle_deg in (
            ("L_Upperarm", -self._IDLE_ARM_DROP_DEG),
            ("R_Upperarm", self._IDLE_ARM_DROP_DEG),
        ):
            adjustment = Gf.Quatf(
                Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), angle_deg).GetQuat()
            )
            index = joint_indices[joint_name]
            rotations[index] = rotations[index] * adjustment

        animation_path = root_prim.GetPath().AppendChild("HCAKinematicIdlePose")
        animation = UsdSkel.Animation.Define(stage, animation_path)
        animation.CreateJointsAttr(joints)
        animation.CreateTranslationsAttr(translations)
        animation.CreateRotationsAttr(rotations)
        animation.CreateScalesAttr(scales)
        relationship = UsdSkel.BindingAPI.Apply(
            skeleton_prim
        ).CreateAnimationSourceRel()
        if not relationship.SetTargets([animation_path]):
            raise RuntimeError(
                f"Could not bind the static idle pose for kinematic Human {human_id!r}."
            )

    def _bind_scene_person(
        self, human: Any, source_path: str, add_semantics: Any
    ) -> None:
        import omni.usd
        import numpy as np
        from tarsphere.infra.control.scene_person_geometry import (
            tight_visible_mesh_bounds,
        )

        runtime_path = source_path.replace("/Root", "/World/env_0/scene", 1)
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Scene-bound Human requires a live USD stage.")
        prim = stage.GetPrimAtPath(runtime_path)
        if prim is None or not prim.IsValid():
            raise RuntimeError(
                f"Scene-bound Human {human.id!r} prim is missing: {runtime_path}"
            )
        actual_min, actual_max = tight_visible_mesh_bounds(prim)
        expected_min = np.asarray(human.scene_person_bounds_min_m, dtype=float)
        expected_max = np.asarray(human.scene_person_bounds_max_m, dtype=float)
        error_m = float(
            max(
                np.max(np.abs(actual_min - expected_min)),
                np.max(np.abs(actual_max - expected_max)),
            )
        )
        if error_m > 0.03:
            raise RuntimeError(
                f"Scene-bound Human {human.id!r} geometry is stale: "
                f"bounds error {error_m:.3f} m at {runtime_path}; "
                f"expected_min={expected_min.tolist()} "
                f"expected_max={expected_max.tolist()} "
                f"actual_min={actual_min.tolist()} actual_max={actual_max.tolist()}"
            )
        add_semantics(prim, str(human.id), "class")
        human_id = str(human.id)
        self._paths[human_id] = runtime_path
        self._scene_bound_ids.add(human_id)

    def is_scene_bound(self, human_id: str) -> bool:
        return str(human_id) in self._scene_bound_ids

    def set_pose(self, human_id: str, position: list[float], yaw_rad: float) -> None:
        if self.is_scene_bound(human_id):
            return
        import numpy as np
        from omni.isaac.core.prims.xform_prim import XFormPrim

        from tarsphere.infra.control.physical_contracts import (
            HUMAN_CHARACTER_YAW_OFFSET_RAD,
        )

        angle = yaw_rad + HUMAN_CHARACTER_YAW_OFFSET_RAD
        XFormPrim(self._paths[human_id]).set_world_pose(
            position=np.asarray(position, dtype=float),
            orientation=np.asarray([math.cos(angle / 2), 0, 0, math.sin(angle / 2)]),
        )
