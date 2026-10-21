"""Apply, verify, and restore the compiled episode asset adaptation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Optional

from tarsphere.assets import resolve_tarsphere_asset
from tarsphere.bench.asset_adaptation import EpisodeAssetAdaptation


_SETTLE_SAMPLE_COUNT = 16
_MAX_SETTLED_STEP_M = 0.002
_MAX_PLANAR_DRIFT_M = 0.03
_MAX_VERTICAL_DRIFT_M = 0.04
_MAX_MEASURED_SIZE_ERROR_M = 0.01


class EpisodeAssetRuntimeError(RuntimeError):
    """Raised when a compiled episode edit cannot become live evidence."""


def adaptation_from_episode(episode: Any) -> Optional[EpisodeAssetAdaptation]:
    """Read the one compiled adaptation contract without guessing defaults."""
    world = getattr(episode, "world_knowledge", None) or {}
    payload = world.get("asset_adaptation")
    if payload is None:
        return None
    adaptation = EpisodeAssetAdaptation.model_validate(payload)
    scene_id = str(world.get("scene_id") or "").strip()
    if scene_id != adaptation.scene_id:
        raise EpisodeAssetRuntimeError(
            "compiled asset adaptation and world knowledge use different "
            f"scenes: {adaptation.scene_id!r} != {scene_id!r}"
        )
    return adaptation


class EpisodeAssetAdapter:
    """Own one adaptation's complete reversible USD/PhysX lifecycle."""

    def __init__(
        self,
        adaptation: EpisodeAssetAdaptation,
        *,
        asset_resolver: Callable[[str], Path] = resolve_tarsphere_asset,
    ) -> None:
        self.adaptation = adaptation
        self._asset_resolver = asset_resolver
        self._stage: Any = None
        self._hidden_active_states: dict[str, bool] = {}
        self._settle_samples: list[dict[str, tuple[float, float, float]]] = []

    @property
    def applied(self) -> bool:
        return self._stage is not None

    def apply(self, stage: Any) -> None:
        """Introduce exact USD references and activate their physical bodies."""
        if self.applied:
            raise EpisodeAssetRuntimeError("episode asset adaptation is active")
        if stage is None:
            raise EpisodeAssetRuntimeError("USD stage is unavailable")
        payloads = {
            instance.object_id: self._asset_resolver(instance.asset_path)
            for instance in self.adaptation.instances
        }
        missing = [
            f"{object_id}: {path}"
            for object_id, path in payloads.items()
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "episode adaptation asset payloads are missing; install the "
                "tracked GRScenes object library: " + "; ".join(missing)
            )
        root = stage.GetPrimAtPath(self.adaptation.runtime_root)
        if root and root.IsValid():
            raise EpisodeAssetRuntimeError(
                "episode adaptation runtime root already exists: "
                f"{self.adaptation.runtime_root}"
            )
        hidden = {}
        for path in self.adaptation.hidden_runtime_prims:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                raise EpisodeAssetRuntimeError(
                    f"episode adaptation cannot find routed anchor prim: {path}"
                )
            hidden[path] = bool(prim.IsActive())

        self._stage = stage
        self._hidden_active_states = hidden
        try:
            from pxr import Gf, UsdGeom

            UsdGeom.Xform.Define(stage, self.adaptation.runtime_root)
            for path in self.adaptation.hidden_runtime_prims:
                stage.GetPrimAtPath(path).SetActive(False)
            for instance in self.adaptation.instances:
                # Do not author a local Xform type name here: that stronger
                # opinion would replace a referenced Cube/Mesh default prim's
                # schema and make its geometry disappear. An override receives
                # the referenced payload's own type while remaining Xformable.
                prim = stage.OverridePrim(instance.runtime_prim)
                if not prim.GetReferences().AddReference(
                    str(payloads[instance.object_id])
                ):
                    raise EpisodeAssetRuntimeError(
                        f"failed to reference asset for {instance.object_id}"
                    )
                from tarsphere.infra.simulation.scene_materials import (
                    localize_referenced_asset_materials,
                )

                localize_referenced_asset_materials(prim, payloads[instance.object_id])
                transform = Gf.Transform()
                transform.SetScale(
                    Gf.Vec3d(
                        instance.asset_meters_per_unit,
                        instance.asset_meters_per_unit,
                        instance.asset_meters_per_unit,
                    )
                )
                x, y, z, w = instance.orientation_xyzw
                transform.SetRotation(
                    Gf.Rotation(
                        Gf.Quatd(
                            w,
                            Gf.Vec3d(x, y, z),
                        )
                    )
                )
                transform.SetTranslation(Gf.Vec3d(*instance.position_xyz_m))
                xformable = UsdGeom.Xformable(prim)
                xformable.ClearXformOpOrder()
                transform_op = xformable.AddTransformOp()
                transform_op.Set(transform.GetMatrix())

                # The compiled position denotes the geometric centre, not
                # the USD pivot. Several certified native assets retain a
                # displaced child mesh; placing their pivot at the requested
                # centre can move the object off its support before physics.
                from tarsphere.infra.simulation.usd_geometry import world_bounds
                from pxr import Usd

                lower, upper = world_bounds(stage, instance.runtime_prim)
                world_delta = Gf.Vec3d(
                    *(
                        instance.position_xyz_m[axis] - (lower[axis] + upper[axis]) / 2
                        for axis in range(3)
                    )
                )
                parent_world = UsdGeom.XformCache(
                    Usd.TimeCode.Default()
                ).GetLocalToWorldTransform(prim.GetParent())
                local_delta = parent_world.GetInverse().TransformDir(world_delta)
                transform.SetTranslation(transform.GetTranslation() + local_delta)
                transform_op.Set(transform.GetMatrix())

                from tarsphere.infra.manipulation.pickable_physics import (
                    configure_dynamic_pickable,
                )

                configure_dynamic_pickable(
                    prim,
                    mass_kg=self.adaptation.spawn_mass_kg,
                    kinematic=False,
                )
            self.capture_settle_sample()
        except Exception:
            self.restore()
            raise

    def capture_settle_sample(self) -> None:
        """Record measured world centres for the bounded settle gate."""
        if not self.applied:
            raise EpisodeAssetRuntimeError(
                "cannot sample an inactive episode adaptation"
            )
        from tarsphere.infra.simulation.usd_geometry import world_bounds

        sample = {}
        for instance in self.adaptation.instances:
            lower, upper = world_bounds(self._stage, instance.runtime_prim)
            sample[instance.object_id] = tuple(
                (float(lower[axis]) + float(upper[axis])) / 2.0 for axis in range(3)
            )
        self._settle_samples.append(sample)
        if len(self._settle_samples) > _SETTLE_SAMPLE_COUNT:
            self._settle_samples.pop(0)

    def verify_settled(self) -> dict[str, Any]:
        """Require stable, correctly sized, non-overlapping live instances."""
        if len(self._settle_samples) < _SETTLE_SAMPLE_COUNT:
            raise EpisodeAssetRuntimeError(
                "spawn settle verification requires "
                f"{_SETTLE_SAMPLE_COUNT} measured samples; got "
                f"{len(self._settle_samples)}"
            )
        from tarsphere.infra.simulation.usd_geometry import (
            unrotated_world_size,
            world_bounds,
        )

        diagnostics: dict[str, Any] = {}
        live_bounds = {}
        for instance in self.adaptation.instances:
            positions = [sample[instance.object_id] for sample in self._settle_samples]
            max_step = max(
                _distance(first, second)
                for first, second in zip(positions, positions[1:])
            )
            final = positions[-1]
            planned = instance.position_xyz_m
            planar_drift = math.hypot(
                final[0] - planned[0],
                final[1] - planned[1],
            )
            vertical_drift = abs(final[2] - planned[2])
            lower, upper = world_bounds(self._stage, instance.runtime_prim)
            world_size = tuple(
                float(upper[axis]) - float(lower[axis]) for axis in range(3)
            )
            size = unrotated_world_size(self._stage, instance.runtime_prim)
            size_error = max(
                abs(size[axis] - instance.size_xyz_m[axis]) for axis in range(3)
            )
            if max_step > _MAX_SETTLED_STEP_M:
                raise EpisodeAssetRuntimeError(
                    f"{instance.object_id} did not settle: "
                    f"max_sample_step_m={max_step:.6f}"
                )
            if (
                planar_drift > _MAX_PLANAR_DRIFT_M
                or vertical_drift > _MAX_VERTICAL_DRIFT_M
            ):
                raise EpisodeAssetRuntimeError(
                    f"{instance.object_id} drifted from its grounded pose: "
                    f"planar={planar_drift:.6f} m, "
                    f"vertical={vertical_drift:.6f} m"
                )
            if size_error > _MAX_MEASURED_SIZE_ERROR_M:
                from pxr import Gf, Usd, UsdGeom

                prim = self._stage.GetPrimAtPath(instance.runtime_prim)
                transform = Gf.Transform(
                    UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default()
                    )
                )
                rotation = transform.GetRotation().GetQuat()
                orientation = [*rotation.GetImaginary(), rotation.GetReal()]
                raise EpisodeAssetRuntimeError(
                    f"{instance.object_id} live size contradicts its catalog "
                    f"measurement: max_error_m={size_error:.6f}; "
                    f"expected_size_xyz_m={list(instance.size_xyz_m)}; "
                    f"measured_asset_axis_size_xyz_m={list(size)}; "
                    f"measured_world_aabb_size_xyz_m={list(world_size)}; "
                    f"measured_orientation_xyzw={orientation}; "
                    f"measured_center_xyz_m={list(final)}"
                )
            live_bounds[instance.object_id] = (
                tuple(float(value) for value in lower),
                tuple(float(value) for value in upper),
            )
            diagnostics[instance.object_id] = {
                "final_center_xyz_m": list(final),
                "max_sample_step_m": max_step,
                "planar_drift_m": planar_drift,
                "vertical_drift_m": vertical_drift,
                "max_size_error_m": size_error,
                "measured_asset_axis_size_xyz_m": list(size),
                "measured_world_aabb_size_xyz_m": list(world_size),
            }
        for first_index, first in enumerate(self.adaptation.instances):
            for second in self.adaptation.instances[first_index + 1 :]:
                if _aabb_overlap(
                    live_bounds[first.object_id],
                    live_bounds[second.object_id],
                ):
                    raise EpisodeAssetRuntimeError(
                        "adapted live assets overlap after settling: "
                        f"{first.object_id}, {second.object_id}"
                    )
        return diagnostics

    def restore(self) -> None:
        """Remove introduced prims and restore exact native active states."""
        stage = self._stage
        if stage is None:
            return
        root = stage.GetPrimAtPath(self.adaptation.runtime_root)
        if root and root.IsValid():
            stage.RemovePrim(self.adaptation.runtime_root)
        for path, active in self._hidden_active_states.items():
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                prim.SetActive(active)
        self._stage = None
        self._hidden_active_states.clear()
        self._settle_samples.clear()


def _distance(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    return math.sqrt(sum((first[axis] - second[axis]) ** 2 for axis in range(3)))


def _aabb_overlap(
    first: tuple[tuple[float, float, float], tuple[float, float, float]],
    second: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> bool:
    return all(
        first[0][axis] < second[1][axis] and second[0][axis] < first[1][axis]
        for axis in range(3)
    )


__all__ = [
    "EpisodeAssetAdapter",
    "EpisodeAssetRuntimeError",
    "adaptation_from_episode",
]
