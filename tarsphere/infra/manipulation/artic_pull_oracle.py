"""Articulated-open oracle as an ideal but realistic action upper bound.

The oracle must record **real operation steps** an agent could issue:

``navigate → face → pregrasp → grasp → pull waypoints → settle``

Joint ``DriveAPI`` / ``set_joint_positions`` is **forbidden** as the success
path. Joint metadata only tells us *which* DoF to follow and how far the
end-effector must travel while contacting the moving link.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional, Sequence

from pxr import Gf, Usd, UsdGeom

from tarsphere.infra.manipulation.artic_open import (
    ArticJointSpec,
    ArticOpenPlan,
    _axis_vector,
    _gf_quat_from_attr,
    _xformable_world,
    plan_artic_open,
)


StepKind = Literal[
    "navigate",
    "face",
    "reach_pregrasp",
    "reach_grasp",
    "close_gripper",
    "pull",
    "settle",
    "open_gripper",
]


@dataclass(frozen=True)
class ArticOracleStep:
    """One countable oracle operation (ideal upper-bound atom)."""

    kind: StepKind
    index: int
    # End-effector target in metres when this step moves the arm.
    ee_xyz_m: Optional[tuple[float, float, float]] = None
    # Base yaw when this step reorients the robot.
    yaw_rad: Optional[float] = None
    # Navigation polyline in metres when kind == navigate.
    path_xyz_m: Optional[tuple[tuple[float, float, float], ...]] = None
    # Joint coordinate commanded indirectly by following the moving link.
    joint_value: Optional[float] = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArticPullOraclePlan:
    """Full artic-open oracle plan with countable real operation steps."""

    open_plan: ArticOpenPlan
    # World handle grasp point at the closed configuration (metres).
    handle_xyz_m: tuple[float, float, float]
    # Approach just outside the handle along the pull direction (metres).
    pregrasp_xyz_m: tuple[float, float, float]
    # EE targets that track the handle while the joint opens (metres).
    pull_waypoints_xyz_m: tuple[tuple[float, float, float], ...]
    # Joint coordinates corresponding 1:1 with pull waypoints.
    pull_joint_values: tuple[float, ...]
    # Unit pull/tangent direction in world metres at the closed pose.
    pull_axis_world: tuple[float, float, float]
    # Robot base approach from the manip-nav route (metres XY).
    approach_xy_m: tuple[float, float]
    # Optional nav polyline already snapped for G1/Franka (metres XYZ).
    nav_path_xyz_m: tuple[tuple[float, float, float], ...]
    # Face yaw looking from approach toward the handle.
    face_yaw_rad: float
    steps: tuple[ArticOracleStep, ...]
    # Convenience: number of countable operations in ``steps``.
    n_operation_steps: int
    # Explicit contract flag for evaluators / certificates.
    forbids_joint_drive: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "open_plan": self.open_plan.to_dict(),
            "handle_xyz_m": list(self.handle_xyz_m),
            "pregrasp_xyz_m": list(self.pregrasp_xyz_m),
            "pull_waypoints_xyz_m": [list(p) for p in self.pull_waypoints_xyz_m],
            "pull_joint_values": list(self.pull_joint_values),
            "pull_axis_world": list(self.pull_axis_world),
            "approach_xy_m": list(self.approach_xy_m),
            "nav_path_xyz_m": [list(p) for p in self.nav_path_xyz_m],
            "face_yaw_rad": self.face_yaw_rad,
            "steps": [step.to_dict() for step in self.steps],
            "n_operation_steps": self.n_operation_steps,
            "forbids_joint_drive": self.forbids_joint_drive,
            "task_sequence": [
                "navigate",
                "face",
                "grasp",
                "pull",
                "settle",
            ],
        }


def _as_vec3(values: Sequence[float]) -> Gf.Vec3d:
    return Gf.Vec3d(float(values[0]), float(values[1]), float(values[2]))


def _tuple3(vec: Gf.Vec3d) -> tuple[float, float, float]:
    return (round(float(vec[0]), 5), round(float(vec[1]), 5), round(float(vec[2]), 5))


def _yaw_toward(from_xy: Sequence[float], to_xy: Sequence[float]) -> float:
    return math.atan2(
        float(to_xy[1]) - float(from_xy[1]),
        float(to_xy[0]) - float(from_xy[0]),
    )


def _joint_world_frame(
    stage: Usd.Stage,
    joint: ArticJointSpec,
) -> tuple[Gf.Matrix4d, Gf.Vec3d]:
    """Return (joint_world_matrix, axis_dir_world_unit)."""
    joint_prim = stage.GetPrimAtPath(joint.joint_path)
    body0 = stage.GetPrimAtPath(joint.body0_path)
    if not joint_prim or not body0:
        raise LookupError(f"missing joint bodies for {joint.joint_path}")
    local_pos0 = joint_prim.GetAttribute("physics:localPos0").Get()
    local_rot0 = joint_prim.GetAttribute("physics:localRot0").Get()
    body0_world = _xformable_world(body0)
    joint_in_body0 = Gf.Matrix4d()
    joint_in_body0.SetTranslate(_as_vec3(local_pos0))
    joint_in_body0.SetRotate(_gf_quat_from_attr(local_rot0))
    joint_world = joint_in_body0 * body0_world
    axis_local = _axis_vector(joint.axis)
    axis_world = Gf.Vec3d(
        joint_world.ExtractRotation().TransformDir(axis_local)
    ).GetNormalized()
    return joint_world, axis_world


def _body_bounds_m(
    stage: Usd.Stage,
    prim_path: str,
    meters_per_unit: float,
) -> tuple[Gf.Vec3d, Gf.Vec3d, Gf.Vec3d]:
    """Return (center_m, half_extent_m, size_m) for one link."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise LookupError(prim_path)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    mn = box.GetMin()
    mx = box.GetMax()
    lower = Gf.Vec3d(mn) * float(meters_per_unit)
    upper = Gf.Vec3d(mx) * float(meters_per_unit)
    center = (lower + upper) * 0.5
    size = upper - lower
    half = size * 0.5
    return center, half, size


def estimate_handle_xyz_m(
    stage: Usd.Stage,
    plan: ArticOpenPlan,
    *,
    pull_axis_world: Gf.Vec3d,
    approach_xy_m: Sequence[float],
) -> tuple[float, float, float]:
    """Pick a grasp point on the moving link facing the robot / pull dir."""
    center, half, _ = _body_bounds_m(
        stage,
        plan.primary.body1_path,
        plan.primary.meters_per_unit,
    )
    # Prefer the AABB face whose outward normal best matches the direction
    # from the object toward the robot approach (i.e. the face you pull).
    to_robot = Gf.Vec3d(
        float(approach_xy_m[0]) - float(center[0]),
        float(approach_xy_m[1]) - float(center[1]),
        0.0,
    )
    if to_robot.GetLength() < 1e-6:
        to_robot = -Gf.Vec3d(pull_axis_world)
    to_robot.Normalize()

    # Candidate face offsets along ±X/±Y (horizontal drawers/doors).
    candidates = [
        Gf.Vec3d(half[0], 0.0, 0.0),
        Gf.Vec3d(-half[0], 0.0, 0.0),
        Gf.Vec3d(0.0, half[1], 0.0),
        Gf.Vec3d(0.0, -half[1], 0.0),
    ]
    best = candidates[0]
    best_score = -1e9
    for offset in candidates:
        if offset.GetLength() < 1e-6:
            continue
        normal = Gf.Vec3d(offset).GetNormalized()
        # Face toward robot and aligned with opening motion.
        score = float(Gf.Dot(normal, to_robot)) + 0.25 * float(
            Gf.Dot(normal, -Gf.Vec3d(pull_axis_world))
        )
        if score > best_score:
            best_score = score
            best = offset
    # Slightly inset so the EE targets the panel, not empty space outside.
    handle = center + best * 0.90
    # Keep grasp near mid-height of the moving panel.
    handle = Gf.Vec3d(handle[0], handle[1], float(center[2]))
    return _tuple3(handle)


def _prismatic_pull_waypoints(
    *,
    handle0: Gf.Vec3d,
    axis_world: Gf.Vec3d,
    plan: ArticOpenPlan,
    n_pull: int,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[float, ...]]:
    travel_m = (
        abs(plan.target_value - plan.primary.lower) * plan.primary.meters_per_unit
    )
    # Opening toward upper: move the handle along +axis (authored open dir).
    direction = Gf.Vec3d(axis_world).GetNormalized()
    waypoints: list[tuple[float, float, float]] = []
    values: list[float] = []
    for i in range(1, n_pull + 1):
        frac = float(i) / float(n_pull)
        joint_value = float(
            plan.primary.lower + frac * (plan.target_value - plan.primary.lower)
        )
        point = handle0 + direction * (travel_m * frac)
        waypoints.append(_tuple3(point))
        values.append(joint_value)
    return tuple(waypoints), tuple(values)


def _revolute_pull_waypoints(
    *,
    stage: Usd.Stage,
    handle0: Gf.Vec3d,
    plan: ArticOpenPlan,
    n_pull: int,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[float, ...],
    Gf.Vec3d,
]:
    joint_world, axis_world = _joint_world_frame(stage, plan.primary)
    hinge = Gf.Vec3d(joint_world.ExtractTranslation()) * plan.primary.meters_per_unit
    radius_vec = handle0 - hinge
    # Initial tangent = axis × radius (pull direction at closed pose).
    tangent = Gf.Cross(axis_world, radius_vec)
    if tangent.GetLength() < 1e-8:
        tangent = Gf.Cross(radius_vec, axis_world)
    tangent.Normalize()

    waypoints: list[tuple[float, float, float]] = []
    values: list[float] = []
    delta_deg = float(plan.target_value - plan.primary.lower)
    for i in range(1, n_pull + 1):
        frac = float(i) / float(n_pull)
        angle = delta_deg * frac
        joint_value = float(plan.primary.lower + angle)
        rot = Gf.Rotation(axis_world, angle)
        moved = hinge + Gf.Vec3d(rot.TransformDir(radius_vec))
        waypoints.append(_tuple3(moved))
        values.append(joint_value)
    return tuple(waypoints), tuple(values), tangent


def build_oracle_steps(
    *,
    nav_path_xyz_m: Sequence[Sequence[float]],
    face_yaw_rad: float,
    pregrasp_xyz_m: Sequence[float],
    handle_xyz_m: Sequence[float],
    pull_waypoints_xyz_m: Sequence[Sequence[float]],
    pull_joint_values: Sequence[float],
) -> tuple[ArticOracleStep, ...]:
    """Materialize the countable ideal operation sequence."""
    steps: list[ArticOracleStep] = []
    idx = 0

    def add(kind: StepKind, **kwargs: Any) -> None:
        nonlocal idx
        idx += 1
        steps.append(ArticOracleStep(kind=kind, index=idx, **kwargs))

    path = tuple(
        (float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0)
        for p in nav_path_xyz_m
    )
    add(
        "navigate",
        path_xyz_m=path,
        note="follow manip-nav route to approach",
    )
    add(
        "face",
        yaw_rad=float(face_yaw_rad),
        note="orient base toward handle",
    )
    add(
        "reach_pregrasp",
        ee_xyz_m=_tuple3(_as_vec3(pregrasp_xyz_m)),
        note="IK to pregrasp outside the handle",
    )
    add(
        "reach_grasp",
        ee_xyz_m=_tuple3(_as_vec3(handle_xyz_m)),
        note="IK onto handle / moving panel",
    )
    add("close_gripper", note="close gripper on handle")
    for wp, jv in zip(pull_waypoints_xyz_m, pull_joint_values):
        add(
            "pull",
            ee_xyz_m=_tuple3(_as_vec3(wp)),
            joint_value=float(jv),
            note="EE tracks handle along joint-allowed motion",
        )
    add("settle", note="hold open configuration")
    add("open_gripper", note="release handle after open gate")
    return tuple(steps)


def plan_artic_pull_oracle(
    *,
    stage: Usd.Stage,
    object_id: str,
    category: str,
    approach_xy_m: Sequence[float],
    nav_path_xy_m: Sequence[Sequence[float]] | None = None,
    open_fraction: float = 0.80,
    n_pull_waypoints: int = 8,
    pregrasp_clearance_m: float = 0.12,
) -> ArticPullOraclePlan:
    """Build an IK→pull oracle plan with countable real operation steps."""
    if n_pull_waypoints < 2:
        raise ValueError("n_pull_waypoints must be >= 2")
    open_plan = plan_artic_open(
        stage=stage,
        object_id=object_id,
        category=category,
        open_fraction=open_fraction,
    )
    _, axis_world = _joint_world_frame(stage, open_plan.primary)

    if open_plan.primary.kind == "prismatic":
        # For drawers, robot pulls opposite to the authored open axis if the
        # approach lies on the -axis side; otherwise pull along +axis.
        center, _, _ = _body_bounds_m(
            stage,
            open_plan.primary.body1_path,
            open_plan.primary.meters_per_unit,
        )
        to_approach = Gf.Vec3d(
            float(approach_xy_m[0]) - float(center[0]),
            float(approach_xy_m[1]) - float(center[1]),
            0.0,
        )
        open_dir = Gf.Vec3d(axis_world)
        if to_approach.GetLength() > 1e-6 and Gf.Dot(to_approach, open_dir) < 0.0:
            # Approach is on the negative side: opening still follows +axis
            # authored limits; EE pulls by moving with the panel along +axis.
            pass
        handle = estimate_handle_xyz_m(
            stage,
            open_plan,
            pull_axis_world=open_dir,
            approach_xy_m=approach_xy_m,
        )
        handle_v = _as_vec3(handle)
        waypoints, joint_values = _prismatic_pull_waypoints(
            handle0=handle_v,
            axis_world=open_dir,
            plan=open_plan,
            n_pull=n_pull_waypoints,
        )
        pull_axis = open_dir
        # Pregrasp: step back from handle toward the robot approach.
        toward_robot = Gf.Vec3d(
            float(approach_xy_m[0]) - handle_v[0],
            float(approach_xy_m[1]) - handle_v[1],
            0.0,
        )
        if toward_robot.GetLength() < 1e-6:
            toward_robot = -pull_axis
        toward_robot.Normalize()
        pregrasp = handle_v + toward_robot * float(pregrasp_clearance_m)
    else:
        handle = estimate_handle_xyz_m(
            stage,
            open_plan,
            pull_axis_world=axis_world,
            approach_xy_m=approach_xy_m,
        )
        handle_v = _as_vec3(handle)
        waypoints, joint_values, tangent = _revolute_pull_waypoints(
            stage=stage,
            handle0=handle_v,
            plan=open_plan,
            n_pull=n_pull_waypoints,
        )
        pull_axis = tangent
        toward_robot = Gf.Vec3d(
            float(approach_xy_m[0]) - handle_v[0],
            float(approach_xy_m[1]) - handle_v[1],
            0.0,
        )
        if toward_robot.GetLength() < 1e-6:
            toward_robot = pull_axis
        toward_robot.Normalize()
        pregrasp = handle_v + toward_robot * float(pregrasp_clearance_m)

    if nav_path_xy_m:
        nav_path = tuple((float(p[0]), float(p[1]), 0.0) for p in nav_path_xy_m)
    else:
        nav_path = ((float(approach_xy_m[0]), float(approach_xy_m[1]), 0.0),)

    face_yaw = _yaw_toward(approach_xy_m, handle)
    steps = build_oracle_steps(
        nav_path_xyz_m=nav_path,
        face_yaw_rad=face_yaw,
        pregrasp_xyz_m=pregrasp,
        handle_xyz_m=handle,
        pull_waypoints_xyz_m=waypoints,
        pull_joint_values=joint_values,
    )
    return ArticPullOraclePlan(
        open_plan=open_plan,
        handle_xyz_m=handle,
        pregrasp_xyz_m=_tuple3(pregrasp),
        pull_waypoints_xyz_m=waypoints,
        pull_joint_values=joint_values,
        pull_axis_world=_tuple3(pull_axis),
        approach_xy_m=(float(approach_xy_m[0]), float(approach_xy_m[1])),
        nav_path_xyz_m=nav_path,
        face_yaw_rad=float(face_yaw),
        steps=steps,
        n_operation_steps=len(steps),
        forbids_joint_drive=True,
    )


def summarize_step_counts(plan: ArticPullOraclePlan) -> dict[str, int]:
    """Count oracle steps by kind for certificates / upper-bound metrics."""
    counts: dict[str, int] = {}
    for step in plan.steps:
        counts[step.kind] = counts.get(step.kind, 0) + 1
    counts["total"] = plan.n_operation_steps
    counts["pull"] = counts.get("pull", 0)
    return counts
