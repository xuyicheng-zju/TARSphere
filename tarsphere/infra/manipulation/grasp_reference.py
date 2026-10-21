"""Create or remove the known-positive grasp reference in a live GRScene."""

from __future__ import annotations

from typing import Any

from tarsphere.bench.certification.grasp_system import (
    GRASP_REFERENCE_PRIM_PATH,
    GraspSystemTrial,
)


def synchronize_grasp_reference(episode: Any) -> bool:
    """Match the live reference prim to the current calibration episode."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Cannot synchronize grasp reference without a USD stage")
    existing = stage.GetPrimAtPath(GRASP_REFERENCE_PRIM_PATH)
    if existing and existing.IsValid():
        stage.RemovePrim(GRASP_REFERENCE_PRIM_PATH)

    world = getattr(episode, "world_knowledge", None) or {}
    payload = world.get("grasp_system_trial")
    if payload is None:
        return False
    trial = GraspSystemTrial.model_validate(payload)
    reference = trial.reference_object
    if reference.prim_path != GRASP_REFERENCE_PRIM_PATH:
        raise ValueError("grasp-system trial uses a non-canonical reference prim")

    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, reference.prim_path)
    cube.CreateSizeAttr(reference.edge_length_m / reference.local_unit_in_meters)
    cube.CreateDisplayColorAttr().Set(
        [
            Gf.Vec3f(0.95, 0.12, 0.04),
        ]
    )
    transform = Gf.Transform()
    transform.SetScale(
        Gf.Vec3d(
            reference.local_unit_in_meters,
            reference.local_unit_in_meters,
            reference.local_unit_in_meters,
        )
    )
    transform.SetTranslation(Gf.Vec3d(*reference.center_xyz))
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(transform.GetMatrix())

    prim = cube.GetPrim()
    collision = UsdPhysics.CollisionAPI.Apply(prim)
    collision.CreateCollisionEnabledAttr().Set(True)
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_body.CreateRigidBodyEnabledAttr().Set(True)
    rigid_body.CreateKinematicEnabledAttr().Set(True)
    prim.CreateAttribute(
        "tarsphere:graspSystemReference",
        Sdf.ValueTypeNames.Bool,
        custom=True,
    ).Set(True)
    return True


__all__ = ["synchronize_grasp_reference"]
