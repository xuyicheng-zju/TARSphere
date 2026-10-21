"""Apply, verify, and restore compiled flat-support scene preparation."""

from __future__ import annotations

import math
from typing import Any, Optional

from tarsphere.infra.simulation.scene_preparation_contract import (
    PhysicalScenePreparation,
)
from tarsphere.infra.manipulation.g1_attachment import (
    G1ObjectLocalTransform,
    read_object_local_transform,
    restore_authored_xform_ops,
)
from tarsphere.infra.simulation.usd_geometry import world_bounds
from tarsphere.infra.manipulation.pickable_physics import (
    reset_rigid_body_to_authored_pose,
)


_MAX_ALIGNMENT_ERROR_M = 0.002
_MAX_TARGET_FACT_ERROR_M = 0.02


class PhysicalScenePreparationError(RuntimeError):
    """Raised when a compiled scene edit cannot become live geometry."""


def preparation_from_episode(episode: Any) -> Optional[PhysicalScenePreparation]:
    """Read the one compiled preparation contract without guessing defaults."""
    world = getattr(episode, "world_knowledge", None) or {}
    payload = world.get("physical_scene_preparation")
    if payload is None:
        return None
    return PhysicalScenePreparation.model_validate(payload)


def validate_live_manipulation_target_pose(
    stage: Any,
    episode: Any,
    *,
    tolerance_m: float = _MAX_TARGET_FACT_ERROR_M,
) -> dict[str, Any] | None:
    """Require the live pickable pose to agree with the episode's world fact.

    A GRScene may store an interactive object away from the task-authored
    support surface.  ``physical_scene_preparation`` is the explicit contract
    that relocates such an object.  If that contract is accidentally omitted,
    perception sees an empty support while evaluation still names the authored
    target pose.  Fail before the Agent is called instead of running an
    impossible-but-apparently-valid episode.
    """

    if stage is None:
        raise PhysicalScenePreparationError("USD stage is unavailable")
    if not math.isfinite(float(tolerance_m)) or float(tolerance_m) <= 0.0:
        raise ValueError("tolerance_m must be finite and positive")

    world = getattr(episode, "world_knowledge", None) or {}
    task = world.get("physical_task")
    if not isinstance(task, dict):
        return None
    target_id = str(task.get("target_object_id") or "").strip()
    if not target_id:
        return None
    facts = {
        str(item.get("object_id") or "").strip(): item
        for item in (world.get("objects") or ())
        if isinstance(item, dict) and str(item.get("object_id") or "").strip()
    }
    fact = facts.get(target_id)
    if fact is None:
        raise PhysicalScenePreparationError(
            f"Manipulation target {target_id!r} has no world object fact."
        )
    runtime_prim = str(fact.get("runtime_prim") or "").strip()
    expected_position = fact.get("position_xyz")
    if (
        not runtime_prim
        or not isinstance(expected_position, (list, tuple))
        or (len(expected_position) != 3)
    ):
        raise PhysicalScenePreparationError(
            f"Manipulation target {target_id!r} requires runtime_prim and "
            "a three-value position_xyz world fact."
        )
    expected = [float(value) for value in expected_position]
    if not all(math.isfinite(value) for value in expected):
        raise PhysicalScenePreparationError(
            f"Manipulation target {target_id!r} position_xyz must be finite."
        )
    lower, upper = world_bounds(stage, runtime_prim)
    measured = [0.5 * (float(lower[index]) + float(upper[index])) for index in range(3)]
    error_m = math.dist(expected, measured)
    if error_m > float(tolerance_m):
        raise PhysicalScenePreparationError(
            f"Live manipulation target {target_id!r} disagrees with its "
            f"episode world fact by {error_m:.3f} m: expected_center="
            f"{expected!r}, measured_center={measured!r}. Add or correct "
            "world_knowledge.physical_scene_preparation before running the "
            "Agent."
        )
    return {
        "object_id": target_id,
        "runtime_prim": runtime_prim,
        "expected_center_xyz": expected,
        "measured_center_xyz": measured,
        "error_m": error_m,
        "tolerance_m": float(tolerance_m),
    }


class PhysicalScenePreparer:
    """Own one episode's reversible hide-and-relocate lifecycle."""

    def __init__(self, preparation: PhysicalScenePreparation) -> None:
        self.preparation = preparation
        self._stage: Any = None
        self._hidden_active_states: dict[str, bool] = {}
        self._saved_local_transforms: dict[str, G1ObjectLocalTransform] = {}

    @property
    def applied(self) -> bool:
        return self._stage is not None

    def apply(self, stage: Any) -> dict[str, Any]:
        """Hide unstable prims and align pickables to flat support surfaces."""
        if self.applied:
            raise PhysicalScenePreparationError(
                "physical scene preparation is already active"
            )
        if stage is None:
            raise PhysicalScenePreparationError("USD stage is unavailable")

        hidden: dict[str, bool] = {}
        for path in self.preparation.hidden_runtime_prims:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                raise PhysicalScenePreparationError(
                    f"physical scene preparation cannot find prim: {path}"
                )
            hidden[path] = bool(prim.IsActive())

        saved: dict[str, G1ObjectLocalTransform] = {}
        for item in self.preparation.relocated_pickables:
            prim = stage.GetPrimAtPath(item.runtime_prim)
            if not prim or not prim.IsValid():
                raise PhysicalScenePreparationError(
                    "physical scene preparation cannot find pickable prim: "
                    f"{item.runtime_prim}"
                )
            saved[item.runtime_prim] = read_object_local_transform(
                stage,
                item.runtime_prim,
            )

        self._stage = stage
        self._hidden_active_states = hidden
        self._saved_local_transforms = saved
        diagnostics: dict[str, Any] = {
            "hidden_runtime_prims": list(self.preparation.hidden_runtime_prims),
            "relocated_pickables": {},
        }
        try:
            for path in self.preparation.hidden_runtime_prims:
                stage.GetPrimAtPath(path).SetActive(False)
            for item in self.preparation.relocated_pickables:
                lower, upper = world_bounds(stage, item.runtime_prim)
                center_x = 0.5 * (float(lower[0]) + float(upper[0]))
                center_y = 0.5 * (float(lower[1]) + float(upper[1]))
                delta_x = 0.0
                delta_y = 0.0
                if item.support_xy is not None:
                    delta_x = float(item.support_xy[0]) - center_x
                    delta_y = float(item.support_xy[1]) - center_y
                delta_z = float(item.support_surface_z_m) - float(lower[2])
                if any(abs(value) > 1e-9 for value in (delta_x, delta_y, delta_z)):
                    _shift_prim_world(
                        stage,
                        item.runtime_prim,
                        delta_x,
                        delta_y,
                        delta_z,
                    )
                aligned_lower, aligned_upper = world_bounds(
                    stage,
                    item.runtime_prim,
                )
                aligned_center_x = 0.5 * (
                    float(aligned_lower[0]) + float(aligned_upper[0])
                )
                aligned_center_y = 0.5 * (
                    float(aligned_lower[1]) + float(aligned_upper[1])
                )
                alignment_error = abs(
                    float(aligned_lower[2]) - float(item.support_surface_z_m)
                )
                if item.support_xy is not None:
                    alignment_error = max(
                        alignment_error,
                        abs(aligned_center_x - float(item.support_xy[0])),
                        abs(aligned_center_y - float(item.support_xy[1])),
                    )
                if alignment_error > _MAX_ALIGNMENT_ERROR_M:
                    raise PhysicalScenePreparationError(
                        f"{item.object_id} did not align to flat support: "
                        f"expected_z={item.support_surface_z_m:.6f}, "
                        f"measured_z={aligned_lower[2]:.6f}, "
                        f"error_m={alignment_error:.6f}"
                    )
                diagnostics["relocated_pickables"][item.object_id] = {
                    "runtime_prim": item.runtime_prim,
                    "support_surface_z_m": item.support_surface_z_m,
                    "support_xy": (
                        None
                        if item.support_xy is None
                        else [float(item.support_xy[0]), float(item.support_xy[1])]
                    ),
                    "aligned_bounds_min_xyz": list(aligned_lower),
                    "aligned_bounds_max_xyz": list(aligned_upper),
                    "alignment_error_m": alignment_error,
                }
                reset_rigid_body_to_authored_pose(stage, item.runtime_prim)
        except Exception:
            self.restore()
            raise
        return diagnostics

    def restore(self) -> None:
        """Restore hidden prim activity and pickable authored transforms."""
        stage = self._stage
        if stage is None:
            return

        for path, local_transform in self._saved_local_transforms.items():
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                raise PhysicalScenePreparationError(
                    f"physical scene preparation cannot restore missing prim: {path}"
                )
            restore_authored_xform_ops(prim, local_transform)
        for path, active in self._hidden_active_states.items():
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                raise PhysicalScenePreparationError(
                    f"physical scene preparation cannot restore hidden prim: {path}"
                )
            prim.SetActive(active)
        self._stage = None
        self._hidden_active_states.clear()
        self._saved_local_transforms.clear()


def _shift_prim_world(
    stage: Any,
    prim_path: str,
    delta_x: float,
    delta_y: float,
    delta_z: float,
) -> None:
    """Translate one prim in world metres."""
    if all(abs(value) <= 1e-12 for value in (delta_x, delta_y, delta_z)):
        return
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise PhysicalScenePreparationError(f"cannot shift missing prim: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    parent_to_world = Gf.Matrix4d(
        xformable.ComputeParentToWorldTransform(Usd.TimeCode.Default())
    )
    if abs(parent_to_world.GetDeterminant()) <= 1e-12:
        raise PhysicalScenePreparationError(
            f"cannot shift prim {prim_path} under a singular parent transform"
        )
    delta_parent = parent_to_world.GetInverse().TransformDir(
        Gf.Vec3d(float(delta_x), float(delta_y), float(delta_z))
    )
    for operation in xformable.GetOrderedXformOps():
        if operation.GetOpType() != UsdGeom.XformOp.TypeTranslate:
            continue
        attribute_name = str(operation.GetAttr().GetName())
        if attribute_name.split(":")[-1] == "pivot":
            continue
        current = operation.Get()
        operation.Set(
            Gf.Vec3d(
                float(current[0]) + float(delta_parent[0]),
                float(current[1]) + float(delta_parent[1]),
                float(current[2]) + float(delta_parent[2]),
            )
        )
        return
    raise PhysicalScenePreparationError(
        f"cannot shift prim without a translate op: {prim_path}"
    )


__all__ = [
    "PhysicalScenePreparer",
    "PhysicalScenePreparationError",
    "preparation_from_episode",
    "validate_live_manipulation_target_pose",
]
