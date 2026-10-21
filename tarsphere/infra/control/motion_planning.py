"""Collision geometry for arm motion planning.

Scene obstacles use live meshes. Furniture AABBs would fill open space between
legs or above thin surfaces and reject valid paths. A compact grasp target may
use ``TargetObjectObstacle``; a support uses ``SupportSurfaceObstacle``, the
actual horizontal mesh union extruded downward by 4 cm. Support holes and tier
boundaries are preserved."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np


SCENE_AABB_PLANNER_OBSTACLES_FORBIDDEN = (
    "Motion planners must not use scene AABB cuboids. Isaac/PhysX collision "
    "is triangle-mesh geometry; an AABB fills empty volume (table legs, air "
    "above a thin tabletop) and rejects motions that never hit live colliders. "
    "Do not reconstruct world_knowledge objects, supports, or furniture as "
    "filled cuboid scene obstacles. If a future planner needs collision "
    "against the room, it must consume the live mesh (or an equivalent "
    "non-AABB approximation of that mesh), not the object world AABB. Two "
    "mesh-tracking primitives are permitted: the grasp target as a "
    "TargetObjectObstacle through require_target_object_obstacle, because a "
    "graspable body is compact enough that its bounding primitive tracks the "
    "mesh; and the support contact mesh as a SupportSurfaceObstacle through "
    "require_support_surface_obstacle, because only that thin plate tracks "
    "the tabletop mesh. Those forms never admit a full furniture box, a "
    "support volume that includes legs or the air beneath the top, or the "
    "room."
)

TARGET_OBJECT_OBSTACLE_MUST_NAME_GRASP_TARGET = (
    "A planner obstacle must be the TargetObjectObstacle for the object being "
    "grasped. Passing any other object, a support, or a furniture bounding box "
    "is forbidden: see SCENE_AABB_PLANNER_OBSTACLES_FORBIDDEN."
)

SUPPORT_SURFACE_OBSTACLE_MUST_NAME_SUPPORT = (
    "A planner obstacle must be the SupportSurfaceObstacle for the authored "
    "source support. Passing any other object, a grasp target, or a furniture "
    "bounding box is forbidden: see SCENE_AABB_PLANNER_OBSTACLES_FORBIDDEN."
)

# Thickness of the closed mesh that represents the support contact layer. Only
# this plate is modelled because that is the region whose mesh the slab
# tracks; the volume below (legs, air) stays out per the AABB ban. The
# thinness guard on require_support_surface_obstacle is what keeps a full
# furniture box from entering under the SupportSurfaceObstacle type.
SUPPORT_SURFACE_SLAB_THICKNESS_M = 0.04


@dataclass(frozen=True)
class TargetObjectObstacle:
    """The bounding primitive of the object a planner is reaching for.

    This is one of the two world colliders a motion planner in this repository
    is allowed to carry. It is owned by the runtime that measured the live
    target, and it is expressed in the planner's own root frame so the planner
    needs no further transform.
    """

    # Scene object id of the grasp target. The planning entry point checks this
    # against the object it was asked to grasp, which is what stops a support
    # or a furniture box from entering the planner under this type.
    object_id: str
    # Centre of the target's live bounding primitive, in the planner root frame,
    # in metres.
    center_robot_m: tuple[float, float, float]
    # Full edge lengths of that primitive along its own axes, in metres. Full
    # extents, not half extents.
    extents_m: tuple[float, float, float]
    # Rotation from the planner root frame to the primitive's own axes, as a
    # wxyz quaternion. The runtime measures the target's box in world axes while
    # the planner works in the robot root frame, so the box is carried with its
    # orientation rather than re-fitted: re-fitting an axis-aligned box in a
    # rotated frame inflates it, which is the failure mode
    # SCENE_AABB_PLANNER_OBSTACLES_FORBIDDEN exists to prevent.
    orientation_robot_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not str(self.object_id).strip():
            raise ValueError(
                "A target-object planner obstacle must name its scene object."
            )
        for field_name, values, size in (
            ("center_robot_m", self.center_robot_m, 3),
            ("extents_m", self.extents_m, 3),
            ("orientation_robot_wxyz", self.orientation_robot_wxyz, 4),
        ):
            if len(values) != size or not all(
                math.isfinite(float(value)) for value in values
            ):
                raise ValueError(
                    f"Target-object obstacle {field_name} must contain {size} "
                    "finite values."
                )
        if any(float(value) <= 0.0 for value in self.extents_m):
            raise ValueError(
                "Target-object obstacle extents_m must all be positive; a "
                "degenerate box would let the planner reach through the "
                f"target: {self.extents_m!r}."
            )
        norm = math.sqrt(
            sum(float(value) ** 2 for value in self.orientation_robot_wxyz)
        )
        if not math.isclose(norm, 1.0, abs_tol=1e-6):
            raise ValueError(
                "Target-object obstacle orientation_robot_wxyz must be a unit "
                f"quaternion; got norm {norm:.6f}."
            )

    @property
    def horizontal_span_m(self) -> float:
        """Return the smaller horizontal edge, the span a gripper must admit."""

        return min(float(self.extents_m[0]), float(self.extents_m[1]))


@dataclass(frozen=True)
class SupportSurfaceObstacle:
    """Closed contact-layer mesh of the authored support in the planner root.

    Both planning and post-plan all-link clearance consume this same mesh.
    Its topology preserves empty regions; its thickness excludes legs/air.
    """

    # Scene object id of the authored source support. The planning entry point
    # checks this against the support it was asked to avoid, which is what
    # stops a grasp target or a furniture box from entering the planner under
    # this type.
    object_id: str
    # Live contact-layer vertices after the world-to-planner rigid transform.
    vertices_robot_m: tuple[tuple[float, float, float], ...]
    # Outward-wound triangular faces indexing vertices_robot_m; closed manifold.
    faces: tuple[tuple[int, int, int], ...]
    # Unit world-up direction expressed in the planner root; owns the measured
    # extrusion thickness even when the Robot root is tilted.
    normal_robot: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not str(self.object_id).strip():
            raise ValueError(
                "A support-surface planner obstacle must name its scene object."
            )
        vertices = np.asarray(self.vertices_robot_m, dtype=float)
        faces = np.asarray(self.faces)
        normal = np.asarray(self.normal_robot, dtype=float)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or len(vertices) < 6
            or not np.all(np.isfinite(vertices))
        ):
            raise ValueError("Support mesh requires finite N x 3 vertices")
        if (
            faces.ndim != 2
            or faces.shape[1] != 3
            or len(faces) < 8
            or not np.issubdtype(faces.dtype, np.integer)
            or np.min(faces) < 0
            or np.max(faces) >= len(vertices)
        ):
            raise ValueError("Support mesh requires valid integer triangle indices")
        if (
            normal.shape != (3,)
            or not np.all(np.isfinite(normal))
            or not math.isclose(float(np.linalg.norm(normal)), 1.0, abs_tol=1e-6)
        ):
            raise ValueError("Support mesh normal_robot must be a finite unit vector")
        heights = vertices @ normal
        thickness = float(np.ptp(heights))
        if not 0 < thickness <= SUPPORT_SURFACE_SLAB_THICKNESS_M + 1e-6:
            raise ValueError("Support contact mesh exceeds the 4 cm thickness contract")
        edges: dict[tuple[int, int], list[int]] = {}
        for triangle in faces:
            for a, b in zip(triangle, np.roll(triangle, -1)):
                edges.setdefault((min(a, b), max(a, b)), []).append(1 if a < b else -1)
        if any(len(signs) != 2 or sum(signs) for signs in edges.values()):
            raise ValueError(
                "Support contact mesh must be closed and consistently wound"
            )
        a, b, c = np.moveaxis(vertices[faces], 1, 0)
        edge_b, edge_c = b - a, c - a
        areas = np.linalg.norm(np.cross(edge_b, edge_c), axis=1)
        scale = np.linalg.norm(edge_b, axis=1) * np.linalg.norm(edge_c, axis=1)
        # A small, non-collinear source face is not a degenerate triangle.
        # Judge numerical degeneracy relative to its edges, not an absolute
        # square-metre cutoff that rejects representable sub-micron seams.
        if np.any(areas <= 64 * np.finfo(float).eps * scale):
            raise ValueError("Support contact mesh contains degenerate triangles")
        volume = float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)
        if volume <= 0:
            raise ValueError(
                "Support contact mesh must enclose positive outward volume"
            )


def forbid_scene_aabb_planner_obstacles(obstacles: Sequence[Any] | None) -> None:
    """Reject any scene AABB cuboid list on a motion planner.

    This covers the room: world_knowledge objects, supports and furniture. The
    two permitted primitives do not travel this path; they reach a planner as
    ``TargetObjectObstacle`` and ``SupportSurfaceObstacle`` through their
    require helpers.
    """

    if obstacles:
        raise RuntimeError(SCENE_AABB_PLANNER_OBSTACLES_FORBIDDEN)


def require_target_object_obstacle(
    obstacle: Any,
    *,
    grasp_target_object_id: str,
) -> TargetObjectObstacle:
    """Return the one obstacle a planner may carry, or raise.

    Accepts exactly one ``TargetObjectObstacle`` naming ``grasp_target_object_id``.
    Every other shape — a sequence of cuboids, a differently named object, a
    raw AABB pair — is a scene obstacle and is refused.
    """

    if isinstance(obstacle, TargetObjectObstacle):
        if str(obstacle.object_id) != str(grasp_target_object_id):
            raise RuntimeError(
                f"{TARGET_OBJECT_OBSTACLE_MUST_NAME_GRASP_TARGET} "
                f"Obstacle names {obstacle.object_id!r} but the grasp target "
                f"is {grasp_target_object_id!r}."
            )
        return obstacle
    if obstacle is None:
        raise RuntimeError(
            "The grasp target obstacle is required: planning the approach "
            "without it drives the palm into the pinned target. "
            f"{TARGET_OBJECT_OBSTACLE_MUST_NAME_GRASP_TARGET}"
        )
    raise RuntimeError(
        f"{TARGET_OBJECT_OBSTACLE_MUST_NAME_GRASP_TARGET} "
        f"Received {type(obstacle).__name__}."
    )


def require_support_surface_obstacle(
    obstacle: Any,
    *,
    support_object_id: str,
) -> SupportSurfaceObstacle:
    """Accept only the validated contact mesh naming the authored support."""

    if isinstance(obstacle, SupportSurfaceObstacle):
        if str(obstacle.object_id) != str(support_object_id):
            raise RuntimeError(
                f"{SUPPORT_SURFACE_OBSTACLE_MUST_NAME_SUPPORT} "
                f"Obstacle names {obstacle.object_id!r} but the source support "
                f"is {support_object_id!r}."
            )
        # This boundary receives planner-root coordinates. Validate the
        # representation consumed by Warp, not a world-frame float32 cast
        # whose precision may be coarser than the actual planning frame.
        with np.errstate(over="ignore", invalid="ignore"):
            vertices = np.asarray(obstacle.vertices_robot_m, dtype=np.float32)
            faces = np.asarray(obstacle.faces, dtype=np.int64)
            a, b, c = np.moveaxis(vertices[faces], 1, 0)
            cross = np.cross(b - a, c - a)
        if not np.all(np.isfinite(cross)):
            raise ValueError(
                "Support contact mesh collapses in planner float32 coordinates"
            )
        collapsed = np.all(cross == 0, axis=1)
        if not np.any(collapsed):
            return obstacle

        # Warp consumes these exact float32 vertices. Rebuild the closed mesh
        # at that unavoidable precision so sub-ULP sliver faces cannot enter
        # its BVH. Manifold removes only faces that have collapsed after the
        # cast; reject any topology or meaningful volume change.
        import manifold3d as mf

        source = mf.Manifold(
            mf.Mesh64(
                np.asarray(obstacle.vertices_robot_m, dtype=np.float64),
                faces.astype(np.uint64),
            )
        )
        planner = mf.Manifold(
            mf.Mesh64(
                vertices.astype(np.float64),
                faces.astype(np.uint64),
            )
        )
        if (
            source.status() != mf.Error.NoError
            or planner.status() != mf.Error.NoError
            or source.genus() != planner.genus()
            or not math.isclose(
                source.volume(),
                planner.volume(),
                rel_tol=1e-4,
                abs_tol=1e-10,
            )
        ):
            raise ValueError(
                "Support contact mesh collapses in planner float32 coordinates"
            )
        mesh = planner.to_mesh64()
        repaired_vertices = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
        repaired_faces = np.asarray(mesh.tri_verts, dtype=np.int64)
        quantized_vertices = {tuple(map(float, row)) for row in vertices}
        if any(
            tuple(map(float, row)) not in quantized_vertices
            for row in repaired_vertices
        ):
            raise ValueError("Support contact mesh repair introduced new vertices")
        repaired = SupportSurfaceObstacle(
            object_id=obstacle.object_id,
            vertices_robot_m=tuple(map(tuple, repaired_vertices)),
            faces=tuple(map(tuple, repaired_faces)),
            normal_robot=obstacle.normal_robot,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            a, b, c = np.moveaxis(
                repaired_vertices.astype(np.float32)[repaired_faces], 1, 0
            )
            repaired_cross = np.cross(b - a, c - a)
        if not np.all(np.isfinite(repaired_cross)) or np.any(
            np.all(repaired_cross == 0, axis=1)
        ):
            raise ValueError(
                "Support contact mesh remains degenerate in planner float32"
            )
        print(
            "[G1-SUPPORT] Rebuilt the contact mesh at planner float32 "
            f"precision: {len(faces)} -> {len(repaired_faces)} faces, "
            f"genus={planner.genus()}.",
            flush=True,
        )
        return repaired
    if obstacle is None:
        raise RuntimeError(
            "The support-surface mesh is required: planning the approach "
            "without it lets a backhand reach pass through the desk. "
            f"{SUPPORT_SURFACE_OBSTACLE_MUST_NAME_SUPPORT}"
        )
    raise RuntimeError(
        f"{SUPPORT_SURFACE_OBSTACLE_MUST_NAME_SUPPORT} "
        f"Received {type(obstacle).__name__}."
    )


def resample_control_trajectory(
    waypoints: Sequence[Sequence[float]],
    *,
    waypoint_interval_s: float,
    control_frequency_hz: float,
) -> tuple[tuple[float, ...], ...]:
    """Interpolate planner waypoints at a controller's fixed update rate.

    The first returned command is the state one controller tick after the
    current state, and the last is exactly the planner endpoint. This avoids
    spending a physics tick re-commanding the already measured start state.
    """

    positions = np.asarray(waypoints, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[0] < 1 or positions.shape[1] < 1:
        raise ValueError("A control trajectory needs a non-empty 2D waypoint array.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("A control trajectory must contain only finite values.")
    if not math.isfinite(waypoint_interval_s) or waypoint_interval_s <= 0.0:
        raise ValueError("waypoint_interval_s must be finite and positive.")
    if not math.isfinite(control_frequency_hz) or control_frequency_hz <= 0.0:
        raise ValueError("control_frequency_hz must be finite and positive.")
    if positions.shape[0] == 1:
        return (tuple(float(value) for value in positions[0]),)

    duration_s = (positions.shape[0] - 1) * float(waypoint_interval_s)
    tick_count = max(1, int(math.ceil(duration_s * control_frequency_hz)))
    command_times = np.minimum(
        np.arange(1, tick_count + 1, dtype=np.float64) / float(control_frequency_hz),
        duration_s,
    )
    source_times = np.arange(positions.shape[0], dtype=np.float64) * float(
        waypoint_interval_s
    )
    sampled = np.column_stack(
        [
            np.interp(command_times, source_times, positions[:, coordinate])
            for coordinate in range(positions.shape[1])
        ]
    )
    sampled[-1] = positions[-1]
    return tuple(tuple(float(value) for value in waypoint) for waypoint in sampled)


__all__ = [
    "SCENE_AABB_PLANNER_OBSTACLES_FORBIDDEN",
    "SUPPORT_SURFACE_SLAB_THICKNESS_M",
    "SUPPORT_SURFACE_OBSTACLE_MUST_NAME_SUPPORT",
    "TARGET_OBJECT_OBSTACLE_MUST_NAME_GRASP_TARGET",
    "SupportSurfaceObstacle",
    "TargetObjectObstacle",
    "forbid_scene_aabb_planner_obstacles",
    "require_support_surface_obstacle",
    "require_target_object_obstacle",
    "resample_control_trajectory",
]
