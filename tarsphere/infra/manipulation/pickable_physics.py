"""One runtime path for promoting a referenced asset to a movable object."""

from __future__ import annotations

import math
from typing import Any, Optional


# Render-only MDL overrides for pickable constant-base-colour shaders.
# GRScenes' MDL scales that branch by (1 - metallic) * opacity; headless RTX
# then renders the authored colour near-black. Forcing metallic off and
# opacity to 1 makes surviving albedo equal BaseColor_Color, except when
# authored Opacity is at or below 0.05 (legitimate glass): metallic is still
# cleared, opacity is left exactly as authored. Physics attributes are not
# touched. Textured shaders (IsBaseColorTex >= 0.5) are left as authored.
PICKABLE_CONSTANT_BASE_COLOUR_METALLIC = 0.0
PICKABLE_CONSTANT_BASE_COLOUR_OPACITY = 1.0
PICKABLE_PRESERVE_AUTHORED_OPACITY_AT_OR_BELOW = 0.05


def reset_rigid_body_to_authored_pose(stage: Any, prim_path: str) -> None:
    """Teleport a registered body for episode initialization, not locomotion.

    USD edits to a live kinematic body are motion targets, not a physical
    reset. Write the same pose to PhysX before any tick;
    otherwise relocating a pickable can sweep it through the robot.
    Prims without RigidBodyAPI have no physical state to reset.
    """
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Cannot reset missing pickable: {prim_path}")
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        return
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    transform = Gf.Transform(matrix)
    position = transform.GetTranslation()
    quaternion = transform.GetRotation().GetQuat()
    imaginary = quaternion.GetImaginary()
    pose = np.asarray([[*position, *imaginary, quaternion.GetReal()]], dtype=np.float32)

    import omni.physics.tensors as tensors
    from omni.physx import get_physx_simulation_interface

    get_physx_simulation_interface().flush_changes()
    simulation = tensors.create_simulation_view("numpy")
    simulation.set_subspace_roots("/")
    body = simulation.create_rigid_body_view(prim_path)
    if body.count != 1:
        raise RuntimeError(
            f"Expected one registered PhysX body for {prim_path}; got {body.count}"
        )
    shape_count = int(body.max_shapes)
    if shape_count <= 0:
        raise RuntimeError(
            f"Registered PhysX pickable has no collision shapes: {prim_path}"
        )
    print(
        f"[PICKABLE-PHYSICS] reset body={prim_path} collision_shapes={shape_count}",
        flush=True,
    )
    indices = np.asarray([0], dtype=np.uint32)
    body.set_transforms(pose, indices)
    rigid_body = UsdPhysics.RigidBodyAPI(prim)
    if rigid_body.GetKinematicEnabledAttr().Get():
        # PhysX derives kinematic velocity from the target; a matching target
        # removes the relocation sweep. Explicit velocity setters are illegal
        # for kinematic bodies.
        body.set_kinematic_targets(pose, indices)
    else:
        body.set_velocities(np.zeros((1, 6), dtype=np.float32), indices)
    measured = body.get_transforms()
    if not np.allclose(measured[:, :3], pose[:, :3], rtol=0.0, atol=1e-5):
        raise RuntimeError(
            f"PhysX pickable reset disagrees with USD: {prim_path}, "
            f"expected={pose.tolist()}, measured={measured.tolist()}"
        )


def configure_dynamic_pickable(
    prim: Any,
    *,
    mass_kg: Optional[float] = None,
    kinematic: bool = False,
) -> None:
    """Configure one semantic root as a legal compound dynamic rigid body.

    GRScenes object payloads may carry nested rigid bodies and static triangle
    mesh collision.  Both are illegal once the semantic root becomes movable,
    so every caller uses this same normalization before PhysX registration.
    """
    from pxr import Gf, PhysxSchema, UsdPhysics
    from tarsphere.simulator.core.util.physics import set_collider

    if not prim or not prim.IsValid():
        raise RuntimeError("Dynamic pickable requires a valid USD prim.")
    if mass_kg is not None and (
        not math.isfinite(float(mass_kg)) or float(mass_kg) <= 0.0
    ):
        raise ValueError("Dynamic pickable mass must be finite and positive.")

    _remove_nested_rigid_bodies(prim)
    _configure_dynamic_collision_approximation(prim)
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
        if not rigid_body:
            raise RuntimeError(
                f"Cannot enable rigid body for pickable {prim.GetPath()}."
            )
        set_collider(prim)
    else:
        rigid_body = UsdPhysics.RigidBodyAPI(prim)
    rigid_body.CreateRigidBodyEnabledAttr(True).Set(True)
    rigid_body.CreateKinematicEnabledAttr(bool(kinematic)).Set(bool(kinematic))
    rigid_body.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    rigid_body.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_body.CreateDisableGravityAttr(False).Set(False)
    if mass_kg is not None:
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        if not mass_api:
            raise RuntimeError(f"Cannot apply mass to pickable {prim.GetPath()}.")
        mass_api.CreateMassAttr().Set(float(mass_kg))
    repair_pickable_constant_base_colour(prim)


def repair_pickable_constant_base_colour(prim: Any) -> int:
    """Force constant-branch MDL colour to survive on one pickable prim.

    Only shaders bound under this prim whose ``IsBaseColorTex`` selects the
    constant base-colour branch are rewritten. Metallic is always cleared.
    Opacity is forced to 1 unless the authored value is at or below
    ``PICKABLE_PRESERVE_AUTHORED_OPACITY_AT_OR_BELOW``. Shared materials that
    live outside the pickable subtree are refused rather than restyled in
    place.
    """
    from pxr import Usd, UsdGeom, UsdShade

    if not prim or not prim.IsValid():
        raise RuntimeError(
            "Pickable constant-base-colour repair requires a valid USD prim."
        )
    pickable_path = prim.GetPath()
    try:
        descendants = Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
    except TypeError:
        descendants = Usd.PrimRange(prim)
    repaired = 0
    for descendant in descendants:
        if not descendant.IsA(UsdGeom.Mesh):
            continue
        material, _ = UsdShade.MaterialBindingAPI(descendant).ComputeBoundMaterial()
        if not material:
            continue
        material_prim = material.GetPrim()
        material_path = material_prim.GetPath()
        if not (
            material_path == pickable_path or material_path.HasPrefix(pickable_path)
        ):
            raise RuntimeError(
                "Pickable constant-base-colour repair cannot mutate material "
                f"{material_path} bound by {descendant.GetPath()} because it "
                f"is outside pickable {pickable_path} and may be shared with "
                "other scene objects."
            )
        if descendant.IsInstanceProxy() or material_prim.IsInstanceProxy():
            raise RuntimeError(
                "Pickable constant-base-colour repair cannot author overrides "
                f"on instance-proxy material {material_path} for "
                f"{descendant.GetPath()}."
            )
        source = material.ComputeSurfaceSource("mdl") or material.ComputeSurfaceSource()
        shader = UsdShade.Shader(source[0]) if source and source[0] else None
        if shader is None or not shader:
            continue
        textured = _shader_uses_base_colour_texture(shader)
        if textured is None or textured:
            continue
        metallic = shader.GetInput("Metallic_Color")
        if metallic:
            if not metallic.Set(_constant_branch_metallic_override(metallic.Get())):
                raise RuntimeError(
                    "Failed to clear pickable constant-branch metallic: "
                    f"{shader.GetPath()}"
                )
        opacity = shader.GetInput("Opacity")
        if opacity and not _preserve_authored_opacity(opacity.Get()):
            if not opacity.Set(PICKABLE_CONSTANT_BASE_COLOUR_OPACITY):
                raise RuntimeError(
                    "Failed to set pickable constant-branch opacity: "
                    f"{shader.GetPath()}"
                )
        repaired += 1
    return repaired


def _preserve_authored_opacity(current: Any) -> bool:
    """Whether constant-branch Opacity is legitimate glass and must be kept."""
    if current is None:
        return False
    if isinstance(current, (int, float)):
        value = float(current)
    else:
        try:
            value = float(current[0])
        except (TypeError, IndexError, ValueError):
            return False
    return math.isfinite(value) and (
        value < PICKABLE_PRESERVE_AUTHORED_OPACITY_AT_OR_BELOW
        or math.isclose(
            value,
            PICKABLE_PRESERVE_AUTHORED_OPACITY_AT_OR_BELOW,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    )


def _shader_uses_base_colour_texture(shader: Any) -> Optional[bool]:
    port = shader.GetInput("IsBaseColorTex")
    if port is None:
        return None
    value = port.Get()
    if value is None:
        return None
    if isinstance(value, (int, float)):
        flag = float(value)
    else:
        try:
            flag = float(value[0])
        except (TypeError, IndexError, ValueError):
            return None
    if not math.isfinite(flag):
        return None
    return flag >= 0.5


def _constant_branch_metallic_override(current: Any) -> Any:
    from pxr import Gf

    if current is None:
        return Gf.Vec4f(
            PICKABLE_CONSTANT_BASE_COLOUR_METALLIC,
            PICKABLE_CONSTANT_BASE_COLOUR_METALLIC,
            PICKABLE_CONSTANT_BASE_COLOUR_METALLIC,
            1.0,
        )
    if isinstance(current, (int, float)):
        return PICKABLE_CONSTANT_BASE_COLOUR_METALLIC
    try:
        if len(current) >= 4:
            return type(current)(
                PICKABLE_CONSTANT_BASE_COLOUR_METALLIC,
                PICKABLE_CONSTANT_BASE_COLOUR_METALLIC,
                PICKABLE_CONSTANT_BASE_COLOUR_METALLIC,
                float(current[3]),
            )
        if len(current) == 3:
            return type(current)(
                PICKABLE_CONSTANT_BASE_COLOUR_METALLIC,
                PICKABLE_CONSTANT_BASE_COLOUR_METALLIC,
                PICKABLE_CONSTANT_BASE_COLOUR_METALLIC,
            )
    except TypeError:
        pass
    return Gf.Vec4f(
        PICKABLE_CONSTANT_BASE_COLOUR_METALLIC,
        PICKABLE_CONSTANT_BASE_COLOUR_METALLIC,
        PICKABLE_CONSTANT_BASE_COLOUR_METALLIC,
        1.0,
    )


def _remove_nested_rigid_bodies(prim: Any) -> int:
    """Collapse modern and legacy descendant bodies into one rigid root.

    GRScenes payloads predate the schema-applied USD Physics convention.  A
    mesh may therefore author ``physics:rigidBodyEnabled`` directly while
    reporting no :class:`RigidBodyAPI`.  Looking only at ``HasAPI`` leaves that
    hidden body active below the promoted semantic root and gives PhysX two
    conflicting mass/inertia authorities for one pickable object.
    """
    from pxr import Usd, UsdPhysics

    try:
        from pxr import PhysxSchema
    except ImportError:  # OpenUSD-only validation environments omit PhysX.
        PhysxSchema = None

    try:
        descendants = Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
    except TypeError:
        descendants = Usd.PrimRange(prim)
    removed = 0
    for descendant in descendants:
        if descendant == prim:
            continue
        has_rigid_body_api = descendant.HasAPI(UsdPhysics.RigidBodyAPI)
        legacy_enabled_attr = descendant.GetAttribute("physics:rigidBodyEnabled")
        legacy_enabled = bool(
            legacy_enabled_attr.Get()
            if legacy_enabled_attr and legacy_enabled_attr.IsValid()
            else False
        )
        if not has_rigid_body_api and not legacy_enabled:
            continue
        if descendant.IsInstanceProxy():
            raise RuntimeError(
                "Pickable contains an instance-proxy rigid body that cannot "
                f"be normalized: {descendant.GetPath()}"
            )
        if PhysxSchema is not None and descendant.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            descendant.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
        if has_rigid_body_api:
            descendant.RemoveAPI(UsdPhysics.RigidBodyAPI)
        # A referenced legacy property remains authored even after a modern
        # API schema is removed.  Override it explicitly in the live stage so
        # PhysX registers this descendant only as compound collision geometry.
        if legacy_enabled_attr and legacy_enabled_attr.IsValid():
            if not legacy_enabled_attr.Set(False):
                raise RuntimeError(
                    "Failed to disable nested legacy pickable rigid body: "
                    f"{descendant.GetPath()}"
                )
        if descendant.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(
                f"Failed to remove nested pickable rigid body: {descendant.GetPath()}"
            )
        remaining_enabled_attr = descendant.GetAttribute("physics:rigidBodyEnabled")
        if (
            remaining_enabled_attr
            and remaining_enabled_attr.IsValid()
            and bool(remaining_enabled_attr.Get())
        ):
            raise RuntimeError(
                "Failed to disable nested pickable rigid-body property: "
                f"{descendant.GetPath()}"
            )
        removed += 1
    return removed


def _configure_dynamic_collision_approximation(prim: Any) -> int:
    """Replace static triangle meshes with movable convex decomposition."""
    from pxr import Usd, UsdPhysics

    try:
        descendants = Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
    except TypeError:
        descendants = Usd.PrimRange(prim)
    configured = 0
    for descendant in descendants:
        if descendant.IsInstanceProxy():
            continue
        approximation = descendant.GetAttribute("physics:approximation")
        if approximation and approximation.IsValid():
            approximation.Set(UsdPhysics.Tokens.convexDecomposition)
            configured += 1
            continue
        collision_enabled = descendant.GetAttribute("physics:collisionEnabled")
        if (
            descendant.GetTypeName() == "Mesh"
            and collision_enabled
            and collision_enabled.IsValid()
        ):
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(descendant)
            mesh_collision.CreateApproximationAttr().Set(
                UsdPhysics.Tokens.convexDecomposition
            )
            configured += 1
    return configured


__all__ = [
    "PICKABLE_CONSTANT_BASE_COLOUR_METALLIC",
    "PICKABLE_CONSTANT_BASE_COLOUR_OPACITY",
    "PICKABLE_PRESERVE_AUTHORED_OPACITY_AT_OR_BELOW",
    "configure_dynamic_pickable",
    "repair_pickable_constant_base_colour",
]
