"""Articulated joint metadata and offline kinematic helpers.

Live **oracle success** must not use joint DriveAPI / set_joint as the open
path. Prefer ``tarsphere.infra.manipulation.artic_pull_oracle`` which records real
``navigate → IK grasp → pull trajectory`` operation steps.

This module still exposes joint enumeration, open targets, and optional
DriveAPI authoring for diagnostics / future PhysX assist — never as the
certificate success path.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from pxr import Gf, Usd, UsdGeom, UsdPhysics


JointKind = Literal["revolute", "prismatic", "other"]
SemanticAction = Literal[
    "open_close_door",
    "pull_push_drawer",
    "open_close_lid",
    "open_mixed",
    "none",
]


@dataclass(frozen=True)
class ArticJointSpec:
    """One authored furniture joint ready for oracle open."""

    joint_path: str
    kind: JointKind
    axis: str
    lower: float
    upper: float
    body0_path: str
    body1_path: str
    travel: float
    meters_per_unit: float

    @property
    def travel_m(self) -> float:
        """Prismatic travel in metres; revolute returns NaN."""
        if self.kind != "prismatic":
            return float("nan")
        return abs(self.travel) * float(self.meters_per_unit)

    @property
    def travel_rad(self) -> float:
        """Revolute travel in radians; prismatic returns NaN."""
        if self.kind != "revolute":
            return float("nan")
        return math.radians(abs(self.travel))


@dataclass(frozen=True)
class ArticOpenPlan:
    """Oracle open command for one articulated object."""

    object_id: str
    category: str
    object_prim_path: str
    primary: ArticJointSpec
    open_fraction: float
    target_value: float
    semantic_action: SemanticAction
    all_joints: tuple[ArticJointSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["primary"] = asdict(self.primary)
        payload["all_joints"] = [asdict(j) for j in self.all_joints]
        return payload


def interaction_usd_path(assets_root: Path, scene_id: str) -> Path:
    scene_dir = assets_root / "scenes/GRScenes-100/home_scenes/scenes" / scene_id
    matches = sorted(scene_dir.glob("start_result_interaction.usd"))
    if not matches:
        raise FileNotFoundError(f"interaction USD missing for {scene_id}: {scene_dir}")
    return matches[0]


def find_object_prim(stage: Usd.Stage, object_id: str) -> Usd.Prim:
    """Locate ``category/model_*`` under Meshes/{BaseAnimation,Animation}."""
    name = object_id.split("/", 1)[1]
    hits: list[Usd.Prim] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.endswith("/" + name):
            hits.append(prim)
    if not hits:
        raise LookupError(f"object prim not found for {object_id}")
    # Prefer BaseAnimation / Animation furniture roots.
    preferred = [
        p
        for p in hits
        if "/Meshes/BaseAnimation/" in str(p.GetPath())
        or "/Meshes/Animation/" in str(p.GetPath())
    ]
    return preferred[0] if preferred else hits[0]


def _joint_kind(type_name: str) -> JointKind:
    if "Revolute" in type_name:
        return "revolute"
    if "Prismatic" in type_name:
        return "prismatic"
    return "other"


def _rel_target(prim: Usd.Prim, name: str) -> str:
    rel = prim.GetRelationship(name)
    targets = rel.GetTargets() if rel else []
    if not targets:
        return ""
    return str(targets[0])


def enumerate_joints(
    stage: Usd.Stage,
    object_prim: Usd.Prim,
) -> list[ArticJointSpec]:
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage) or 0.01)
    out: list[ArticJointSpec] = []
    for prim in Usd.PrimRange(object_prim):
        type_name = prim.GetTypeName()
        if "Joint" not in type_name:
            continue
        kind = _joint_kind(type_name)
        if kind == "other":
            continue
        axis_attr = prim.GetAttribute("physics:axis")
        lower_attr = prim.GetAttribute("physics:lowerLimit")
        upper_attr = prim.GetAttribute("physics:upperLimit")
        axis = str(axis_attr.Get()) if axis_attr and axis_attr.Get() else "X"
        lower = (
            float(lower_attr.Get())
            if lower_attr and lower_attr.Get() is not None
            else 0.0
        )
        upper = (
            float(upper_attr.Get())
            if upper_attr and upper_attr.Get() is not None
            else 0.0
        )
        travel = upper - lower
        if not math.isfinite(travel) or abs(travel) < 1e-6:
            continue
        out.append(
            ArticJointSpec(
                joint_path=str(prim.GetPath()),
                kind=kind,
                axis=axis,
                lower=lower,
                upper=upper,
                body0_path=_rel_target(prim, "physics:body0"),
                body1_path=_rel_target(prim, "physics:body1"),
                travel=travel,
                meters_per_unit=mpu,
            )
        )
    return out


def semantic_action_for(
    category: str,
    joints: list[ArticJointSpec],
) -> SemanticAction:
    if not joints:
        return "none"
    n_rev = sum(1 for j in joints if j.kind == "revolute")
    n_pri = sum(1 for j in joints if j.kind == "prismatic")
    if category in {"trashcan", "toilet"}:
        return "open_close_lid"
    if category in {"desk", "table", "chestofdrawers", "nightstand"}:
        return "pull_push_drawer"
    if n_rev and n_pri:
        return "open_mixed"
    if n_pri and not n_rev:
        return "pull_push_drawer"
    if category in {
        "cabinet",
        "shoecabinet",
        "sideboardcabinet",
        "tvstand",
        "dishwasher",
        "refrigerator",
        "washingmachine",
        "microwave",
        "oven",
    }:
        return "open_close_door" if n_rev else "pull_push_drawer"
    if n_rev:
        return "open_close_door"
    return "pull_push_drawer"


def select_primary_joint(
    category: str,
    joints: list[ArticJointSpec],
) -> ArticJointSpec:
    if not joints:
        raise ValueError("no joints to select")
    action = semantic_action_for(category, joints)
    preferred_kind: Optional[JointKind]
    if action in {"pull_push_drawer"}:
        preferred_kind = "prismatic"
    elif action in {"open_close_door", "open_close_lid"}:
        preferred_kind = "revolute"
    else:
        preferred_kind = None
    ranked = sorted(
        joints,
        key=lambda j: (
            0 if preferred_kind and j.kind == preferred_kind else 1,
            -abs(j.travel),
            j.joint_path,
        ),
    )
    return ranked[0]


def open_target_value(joint: ArticJointSpec, open_fraction: float) -> float:
    """Joint coordinate at the commanded open fraction of [lower, upper]."""
    frac = min(1.0, max(0.0, float(open_fraction)))
    # Open away from the closed/lower stop toward upper.
    return float(joint.lower + frac * (joint.upper - joint.lower))


def plan_artic_open(
    *,
    stage: Usd.Stage,
    object_id: str,
    category: str,
    open_fraction: float = 0.80,
) -> ArticOpenPlan:
    object_prim = find_object_prim(stage, object_id)
    joints = enumerate_joints(stage, object_prim)
    if not joints:
        raise ValueError(f"{object_id} has no usable joints")
    primary = select_primary_joint(category, joints)
    target = open_target_value(primary, open_fraction)
    return ArticOpenPlan(
        object_id=object_id,
        category=category,
        object_prim_path=str(object_prim.GetPath()),
        primary=primary,
        open_fraction=float(open_fraction),
        target_value=target,
        semantic_action=semantic_action_for(category, joints),
        all_joints=tuple(joints),
    )


def apply_drive_api(
    stage: Usd.Stage,
    plan: ArticOpenPlan,
    *,
    stiffness: float = 1.0e5,
    damping: float = 1.0e4,
    max_force: float = 1.0e5,
) -> str:
    """Author a Physics Drive on the primary joint for live Isaac execution."""
    joint = stage.GetPrimAtPath(plan.primary.joint_path)
    if not joint:
        raise LookupError(plan.primary.joint_path)
    drive_type = "angular" if plan.primary.kind == "revolute" else "linear"
    drive = UsdPhysics.DriveAPI.Apply(joint, drive_type)
    drive.CreateTypeAttr().Set(drive_type)
    drive.CreateStiffnessAttr().Set(float(stiffness))
    drive.CreateDampingAttr().Set(float(damping))
    drive.CreateMaxForceAttr().Set(float(max_force))
    drive.CreateTargetPositionAttr().Set(float(plan.target_value))
    return drive_type


def _axis_vector(axis: str) -> Gf.Vec3d:
    key = str(axis).upper()
    if key == "Y":
        return Gf.Vec3d(0.0, 1.0, 0.0)
    if key == "Z":
        return Gf.Vec3d(0.0, 0.0, 1.0)
    return Gf.Vec3d(1.0, 0.0, 0.0)


def _gf_quat_from_attr(value: Any) -> Gf.Quatd:
    # USD may store Quatf as (w, x, y, z) or Gf.Quat*
    if isinstance(value, Gf.Quatd):
        return Gf.Quatd(value)
    if isinstance(value, Gf.Quatf):
        return Gf.Quatd(value)
    w, x, y, z = [float(v) for v in value]
    return Gf.Quatd(w, Gf.Vec3d(x, y, z))


def _xformable_world(prim: Usd.Prim) -> Gf.Matrix4d:
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    return Gf.Matrix4d(cache.GetLocalToWorldTransform(prim))


def _set_world_transform(prim: Usd.Prim, world: Gf.Matrix4d) -> None:
    """Overwrite the prim's local transform so its world pose equals ``world``."""
    parent = prim.GetParent()
    if parent and parent.IsValid() and parent != prim.GetStage().GetPseudoRoot():
        parent_world = _xformable_world(parent)
        local = world * parent_world.GetInverse()
    else:
        local = Gf.Matrix4d(world)
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    op = xformable.AddTransformOp()
    op.Set(local)


def kinematic_open(
    stage: Usd.Stage,
    plan: ArticOpenPlan,
) -> dict[str, Any]:
    """Kinematically open the moving link about the primary joint.

    This is the offline oracle stand-in: prove the moving body can be placed at
    the commanded open configuration using authored joint frames/limits.
    """
    joint_prim = stage.GetPrimAtPath(plan.primary.joint_path)
    body0 = stage.GetPrimAtPath(plan.primary.body0_path)
    body1 = stage.GetPrimAtPath(plan.primary.body1_path)
    if not joint_prim or not body0 or not body1:
        raise LookupError("joint bodies missing")

    local_pos0 = joint_prim.GetAttribute("physics:localPos0").Get()
    local_rot0 = joint_prim.GetAttribute("physics:localRot0").Get()
    local_pos1 = joint_prim.GetAttribute("physics:localPos1").Get()
    local_rot1 = joint_prim.GetAttribute("physics:localRot1").Get()

    body0_world = _xformable_world(body0)
    body1_before = _xformable_world(body1)

    # Joint frame in body0.
    joint_in_body0 = Gf.Matrix4d()
    joint_in_body0.SetTranslate(Gf.Vec3d(*[float(v) for v in local_pos0]))
    joint_in_body0.SetRotate(_gf_quat_from_attr(local_rot0))
    joint_world = joint_in_body0 * body0_world

    # Body1 frame expressed in the joint frame at the closed pose.
    joint_in_body1 = Gf.Matrix4d()
    joint_in_body1.SetTranslate(Gf.Vec3d(*[float(v) for v in local_pos1]))
    joint_in_body1.SetRotate(_gf_quat_from_attr(local_rot1))
    body1_in_joint_closed = joint_in_body1.GetInverse()

    axis = _axis_vector(plan.primary.axis)
    delta = float(plan.target_value - plan.primary.lower)
    if plan.primary.kind == "revolute":
        motion = Gf.Matrix4d(1.0)
        motion.SetRotate(Gf.Rotation(axis, float(delta)))
        body1_in_joint_open = body1_in_joint_closed * motion
        body1_world_open = body1_in_joint_open * joint_world
    else:
        # Prismatic: shift body1 in world along the joint axis direction,
        # using authored limit units directly (ignore parent non-uniform /
        # metric scale on the axis length so 40cm limits move 40 stage units).
        rot = joint_world.ExtractRotation()
        axis_world = Gf.Vec3d(rot.TransformDir(axis)).GetNormalized()
        body1_world_closed = body1_in_joint_closed * joint_world
        body1_world_open = Gf.Matrix4d(body1_world_closed)
        body1_world_open.SetTranslateOnly(
            body1_world_closed.ExtractTranslation() + axis_world * float(delta)
        )
    _set_world_transform(body1, body1_world_open)
    body1_after = _xformable_world(body1)

    # Evidence: translation of body1 origin and rotation delta.
    t0 = body1_before.ExtractTranslation()
    t1 = body1_after.ExtractTranslation()
    translation_m = (t1 - t0) * plan.primary.meters_per_unit
    rot0 = body1_before.ExtractRotationQuat()
    rot1 = body1_after.ExtractRotationQuat()
    rel = rot1 * rot0.GetInverse()
    angle_deg = abs(Gf.Rotation(rel).GetAngle())

    return {
        "body1_path": plan.primary.body1_path,
        "translation_delta_m": [
            round(float(translation_m[0]), 5),
            round(float(translation_m[1]), 5),
            round(float(translation_m[2]), 5),
        ],
        "translation_norm_m": round(float(translation_m.GetLength()), 5),
        "rotation_delta_deg": round(float(angle_deg), 4),
        "target_value": plan.target_value,
        "kind": plan.primary.kind,
    }


def open_gate_passes(
    *,
    plan: ArticOpenPlan,
    measured_value: float,
    tolerance_fraction: float = 0.10,
) -> bool:
    """Pass when measured joint coordinate reached the open band."""
    target = plan.target_value
    span = abs(plan.primary.upper - plan.primary.lower)
    tol = max(1e-4, float(tolerance_fraction) * span)
    # Opening toward upper from lower.
    if plan.primary.upper >= plan.primary.lower:
        return float(measured_value) + 1e-9 >= float(target) - tol
    return float(measured_value) - 1e-9 <= float(target) + tol


def kinematic_gate_passes(
    plan: ArticOpenPlan,
    evidence: dict[str, Any],
    *,
    min_fraction: float = 0.40,
) -> bool:
    """Offline gate using body motion magnitude vs authored travel."""
    if plan.primary.kind == "revolute":
        expected_deg = abs(plan.target_value - plan.primary.lower) * min_fraction
        return float(evidence["rotation_delta_deg"]) + 1e-6 >= expected_deg
    expected_m = (
        abs(plan.target_value - plan.primary.lower) * plan.primary.meters_per_unit
    )
    moved = float(evidence["translation_norm_m"])
    # Prefer fractional travel; also accept a clear absolute drawer pull.
    return moved + 1e-6 >= expected_m * min_fraction or moved >= 0.05
