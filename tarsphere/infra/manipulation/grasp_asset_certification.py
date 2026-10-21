"""Load the selected real asset into the shared grasp-certification table."""

from __future__ import annotations

from typing import Any

from tarsphere.bench.certification.grasp_asset_certification import (
    GraspAssetCertificationTrial,
)
from tarsphere.assets import resolve_tarsphere_asset
from tarsphere.infra.simulation.usd_geometry import world_bounds


_LIVE_BOUNDS_TOLERANCE_M = 0.002


class GraspAssetGeometryMismatch(ValueError):
    """The referenced asset disagrees with its frozen geometry screen."""


def _replace_target(
    stage: Any, trial: GraspAssetCertificationTrial, asset_path: Any
) -> Any:
    """Suppress weaker-layer children before inserting exactly one new asset."""
    from pxr import Gf, UsdGeom

    stage.RemovePrim(trial.runtime_prim)
    target = UsdGeom.Xform.Define(stage, trial.runtime_prim)
    # RemovePrim edits only the current layer. Scene references can still supply
    # Bottle/Asset; explicitly deactivate those composed children.
    for child in list(target.GetPrim().GetChildren()):
        child.SetActive(False)
    asset = stage.OverridePrim(trial.runtime_prim + "/CertifiedAsset")
    asset.SetActive(True)
    if not asset.GetReferences().AddReference(str(asset_path)):
        raise RuntimeError(f"Cannot reference certification asset: {asset_path}")
    transform = Gf.Transform()
    unit = trial.candidate.asset_meters_per_unit
    transform.SetScale(Gf.Vec3d(unit, unit, unit))
    transform.SetTranslation(Gf.Vec3d(*trial.target_translation_xyz_m))
    target.ClearXformOpOrder()
    target.AddTransformOp().Set(transform.GetMatrix())
    return asset


def validate_grasp_asset_geometry(trial: GraspAssetCertificationTrial) -> None:
    """Check the actual reference composition before starting an Isaac worker."""
    from pxr import Usd, UsdGeom

    asset_path = resolve_tarsphere_asset(trial.candidate.asset_path)
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)
    stage = Usd.Stage.CreateInMemory()
    scene = UsdGeom.Xform.Define(stage, "/World/env_0/scene")
    scene.GetPrim().GetReferences().AddReference(
        str(resolve_tarsphere_asset("scenes/g1_grasp_table.usda"))
    )
    _replace_target(stage, trial, asset_path)
    _validate_loaded_bounds(stage, trial)


def _validate_loaded_bounds(stage: Any, trial: GraspAssetCertificationTrial) -> None:
    measured_min, measured_max = world_bounds(stage, trial.runtime_prim)
    expected_min = trial.target_bounds_min_xyz_m
    expected_max = trial.target_bounds_max_xyz_m
    maximum_error = max(
        abs(float(measured) - float(expected))
        for measured, expected in zip(
            (*measured_min, *measured_max),
            (*expected_min, *expected_max),
        )
    )
    if maximum_error > _LIVE_BOUNDS_TOLERANCE_M:
        raise GraspAssetGeometryMismatch(
            f"Asset geometry mismatch: asset={trial.candidate.asset_key!r}, "
            f"measured_bounds={(measured_min, measured_max)!r}, "
            f"expected_bounds={(expected_min, expected_max)!r}, "
            f"maximum_error_m={maximum_error:.6f}"
        )


def synchronize_grasp_asset(episode: Any) -> bool:
    """Replace the table target when the episode declares an asset certificate."""

    world = getattr(episode, "world_knowledge", None) or {}
    payload = world.get("grasp_asset_certification")
    if payload is None:
        return False
    trial = GraspAssetCertificationTrial.model_validate(payload)
    asset_path = resolve_tarsphere_asset(trial.candidate.asset_path)
    if not asset_path.is_file():
        raise FileNotFoundError(f"grasp certification asset is missing: {asset_path}")

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("grasp asset synchronization requires a USD stage")
    target_prim = _replace_target(stage, trial, asset_path)
    from tarsphere.infra.simulation.scene_materials import (
        localize_referenced_asset_materials,
    )

    localize_referenced_asset_materials(target_prim, asset_path)
    _validate_loaded_bounds(stage, trial)
    return True


__all__ = ["synchronize_grasp_asset"]
