"""Read-only rigid G1 visual surface for presentation-camera visibility.

USD supplies immutable link-local triangles; public rigid-body world poses supply
the current articulation. A cached triangle BVH finds the *first robot surface*
on each camera ray, so a hidden back surface cannot invent an environment block.
This is finite visibility sampling, not an all-pixels or contact certificate.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import posixpath
import time

import numpy as np


def _array(value, shape, label):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise RuntimeError(f"Observer geometry invalid {label}: {result.shape}")
    return result


def _rotation(quaternion):
    q = _array(quaternion, (4,), "world quaternion wxyz")
    norm = np.linalg.norm(q)
    if abs(norm - 1) > 1e-3:
        raise RuntimeError(
            "Observer geometry requires a normalized public world quaternion."
        )
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _box_points(lower, upper):
    return np.array(list(itertools.product(*zip(lower, upper))))


class _TriangleTree:
    """Median BVH; batched slab/triangle operations, no Python triangle loop."""

    def __init__(self, triangles):
        self.triangles = triangles
        self.nodes = []
        lo, hi = triangles.min(axis=1), triangles.max(axis=1)
        centers = (lo + hi) / 2

        def build(indices):
            number = len(self.nodes)
            lower, upper = lo[indices].min(axis=0), hi[indices].max(axis=0)
            self.nodes.append(None)
            if len(indices) <= 64:
                self.nodes[number] = (lower, upper, None, None, indices)
            else:
                axis = np.ptp(centers[indices], axis=0).argmax()
                middle = len(indices) // 2
                order = np.argpartition(centers[indices, axis], middle)
                left, right = (
                    build(indices[order[:middle]]),
                    build(indices[order[middle:]]),
                )
                self.nodes[number] = (lower, upper, left, right, None)
            return number

        build(np.arange(len(triangles)))
        self.bounds = _box_points(lo.min(axis=0), hi.max(axis=0))

    def intersect(self, origin, directions, nearest):
        """Update closest distances for a bundle of unit rays, in link metres."""
        stack = [(0, np.arange(len(directions)))]
        while stack:
            index, rows = stack.pop()
            lower, upper, left, right, triangles = self.nodes[index]
            d = directions[rows]
            parallel = abs(d) < 1e-14
            inv = np.divide(1.0, d, out=np.zeros_like(d), where=~parallel)
            a, b = (lower - origin) * inv, (upper - origin) * inv
            low, high = np.minimum(a, b), np.maximum(a, b)
            low[parallel], high[parallel] = -np.inf, np.inf
            outside = np.any(parallel & ((origin < lower) | (origin > upper)), axis=1)
            near, far = low.max(axis=1), high.min(axis=1)
            keep = ~outside & (far >= np.maximum(near, 0)) & (near <= nearest[rows])
            rows = rows[keep]
            if not len(rows):
                continue
            if triangles is None:
                stack.extend(((right, rows), (left, rows)))
                continue
            t = self.triangles[triangles]
            e1, e2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
            p = np.cross(directions[rows, None, :], e2[None, :, :])
            det = np.einsum("rti,ti->rt", p, e1)
            valid = abs(det) > 1e-12
            reciprocal = np.divide(1.0, det, out=np.zeros_like(det), where=valid)
            offset = origin - t[:, 0]
            u = np.einsum("rti,ti->rt", p, offset) * reciprocal
            q = np.cross(offset, e1)
            v = (directions[rows] @ q.T) * reciprocal
            distance = np.einsum("ti,ti->t", e2, q)[None, :] * reciprocal
            valid &= (
                (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (distance > 1e-7)
            )
            distance[~valid] = np.inf
            nearest[rows] = np.minimum(nearest[rows], distance.min(axis=1))


class G1ObserverGeometry:
    """Cache a complete live robot's rigid, visible, triangular USD geometry.

    The installed public wrapper inventory includes every prim with a
    ``physics:rigidBodyEnabled`` attribute, including disabled bodies and IMU.
    ``get_pose`` is environment-relative; ``get_world_pose`` is mandatory here.
    Current Isaac stages use metres. Other runtime unit conventions are rejected.
    """

    def __init__(self, stage, robot):
        from pxr import Usd, UsdGeom

        begin = time.perf_counter()
        unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        if unit != 1.0:
            raise RuntimeError(
                "Observer public world-pose contract requires stage metresPerUnit=1."
            )
        self.robot = robot
        bodies = self._bodies()
        self.paths = tuple(sorted(bodies))
        root = posixpath.commonpath(self.paths)
        if root in bodies:
            root = posixpath.dirname(root)
        if root in ("", "/", "/World"):
            raise RuntimeError(
                "Observer rigid bodies do not identify one robot subtree."
            )
        root_prim = stage.GetPrimAtPath(root)
        if not root_prim:
            raise RuntimeError("Observer robot subtree is missing from USD.")
        prims = list(Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()))
        expected = {
            str(p.GetPath())
            for p in prims
            if p.GetAttribute("physics:rigidBodyEnabled")
        }
        if expected != set(self.paths):
            raise RuntimeError(
                f"Observer rigid-body inventory mismatch: missing={sorted(expected - set(self.paths))}, extra={sorted(set(self.paths) - expected)}"
            )
        transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
        grouped, inventory = {}, []
        digest = hashlib.sha256()
        for prim in prims:
            if not prim.IsActive() or not prim.IsA(UsdGeom.Mesh):
                continue
            image = UsdGeom.Imageable(prim)
            if (
                image.ComputeVisibility() == "invisible"
                or image.ComputePurpose() not in ("default", "render")
            ):
                continue
            mesh = UsdGeom.Mesh(prim)
            if any(
                a.ValueMightBeTimeVarying()
                for a in (
                    mesh.GetPointsAttr(),
                    mesh.GetFaceVertexIndicesAttr(),
                    mesh.GetFaceVertexCountsAttr(),
                )
            ):
                raise RuntimeError(
                    f"Observer requires rigid static mesh topology: {prim.GetPath()}"
                )
            ancestor = prim
            while ancestor and str(ancestor.GetPath()) not in bodies:
                ancestor = ancestor.GetParent()
            if not ancestor:
                raise RuntimeError(
                    f"Observer visible mesh has no registered rigid link: {prim.GetPath()}"
                )
            path = str(ancestor.GetPath())
            # Descendant xforms must be static; current link motion is supplied
            # only by the native public pose, never a possibly stale USD pose.
            child = prim
            while child != ancestor:
                if (
                    UsdGeom.Xformable(child)
                    and UsdGeom.Xformable(child).TransformMightBeTimeVarying()
                ):
                    raise RuntimeError(
                        f"Observer mesh has animated local transform: {child.GetPath()}"
                    )
                child = child.GetParent()
            vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
            if (
                vertices.ndim != 2
                or vertices.shape[1] != 3
                or not np.isfinite(vertices).all()
                or not len(counts)
                or np.any(counts != 3)
                or len(indices) != len(counts) * 3
                or indices.min() < 0
                or indices.max() >= len(vertices)
            ):
                raise RuntimeError(
                    f"Observer needs finite triangular visual mesh: {prim.GetPath()}"
                )
            body_world = np.array(transforms.GetLocalToWorldTransform(ancestor))
            # Public body poses have no scale: rigid link world transforms must
            # be proper rigid transforms; mesh-local scale is baked below.
            axes = body_world[:3, :3]
            if (
                not np.allclose(axes @ axes.T, np.eye(3), atol=1e-5)
                or np.linalg.det(axes) < 0
            ):
                raise RuntimeError(f"Observer unsupported scaled rigid body: {path}")
            local = np.array(transforms.GetLocalToWorldTransform(prim)) @ np.linalg.inv(
                body_world
            )
            vertices = vertices @ local[:3, :3] + local[3, :3]
            triangles = vertices[indices.reshape(-1, 3)]
            grouped.setdefault(path, []).append(triangles)
            record = {
                "mesh": str(prim.GetPath()),
                "body": path,
                "vertices": len(vertices),
                "triangles": len(triangles),
            }
            inventory.append(record)
            digest.update(json.dumps(record, sort_keys=True).encode())
            digest.update(triangles.tobytes())
        if not grouped:
            raise RuntimeError("Observer robot has no visible rigid mesh surfaces.")
        self.trees = {
            path: _TriangleTree(np.concatenate(parts))
            for path, parts in grouped.items()
        }
        self.authority = {
            "source": "cached_USD_link_local_visual_triangles_and_public_get_world_pose",
            "geometry_sha256": digest.hexdigest(),
            "body_paths": list(self.paths),
            "mesh_inventory": inventory,
            "stage_meters_per_unit": unit,
            "cache_build_seconds": time.perf_counter() - begin,
            "visibility_limit": "finite camera-ray samples; not a pixel-complete or contact certificate",
        }

    def _bodies(self):
        from pxr import Sdf

        result = {}
        for body in self.robot.get_rigid_bodies():
            name = str(body.name)
            path = Sdf.Path(name)
            if (
                not path.IsAbsolutePath()
                or not path.IsPrimPath()
                or name in result
                or not callable(getattr(body, "get_world_pose", None))
            ):
                raise RuntimeError(
                    f"Observer invalid rigid-body public inventory: {name}"
                )
            result[name] = body
        if not result:
            raise RuntimeError("Observer empty rigid-body public inventory.")
        return result

    def snapshot(self, physics_step):
        begin = time.perf_counter()
        bodies = self._bodies()
        if set(bodies) != set(self.paths):
            raise RuntimeError(
                "Observer rigid-body inventory changed; rebind after reset."
            )
        poses, records = {}, {}
        for path in self.paths:
            position, quaternion = bodies[path].get_world_pose()
            position = _array(position, (3,), "public world position metres")
            rotation = _rotation(quaternion)
            poses[path] = (position, rotation)
            records[path] = {
                "position_world_m": position.tolist(),
                "orientation_wxyz": np.asarray(quaternion, dtype=float).tolist(),
            }
        encoded = json.dumps(records, sort_keys=True).encode()
        evidence = {
            "physics_step": int(physics_step),
            "pose_sha256": hashlib.sha256(encoded).hexdigest(),
            "body_poses": records,
            "snapshot_seconds": time.perf_counter() - begin,
            "geometry_sha256": self.authority["geometry_sha256"],
        }
        return RobotSurfaceSnapshot(self.trees, poses, evidence)


class RobotSurfaceSnapshot:
    def __init__(self, trees, poses, evidence):
        self.trees, self.poses, self.evidence = trees, poses, evidence
        self.bounds = {
            path: tree.bounds @ poses[path][1].T + poses[path][0]
            for path, tree in trees.items()
        }
        self.framing_points = np.concatenate(list(self.bounds.values()))

    def visible_samples(self, eye, target):
        """7x11 projected lattice plus each link's projected centre.

        Lattice/centres are only ray directions. Returned positions are exact
        nearest intersections with the actual visual triangles, never box corners.
        Both triangle sides count, matching the double-sided scene visibility gate.
        """
        begin = time.perf_counter()
        eye, target = np.asarray(eye), np.asarray(target)
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, (0.0, 0.0, 1.0))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        projections = []
        for points in self.bounds.values():
            rel = points - eye
            z = rel @ forward
            if np.any(z <= 0):
                raise RuntimeError(
                    "Observer visible surface sampling requires robot in front of camera."
                )
            projections.append(np.column_stack((rel @ right / z, rel @ up / z)))
        all_xy = np.concatenate(projections)
        lower, upper = all_xy.min(axis=0), all_xy.max(axis=0)
        # Bin centres avoid tangent rays at extrema; per-link centres cover
        # narrow hands/feet missed by the coarse whole-body lattice.
        xs = lower[0] + (np.arange(7) + 0.5) / 7 * (upper[0] - lower[0])
        ys = lower[1] + (np.arange(11) + 0.5) / 11 * (upper[1] - lower[1])
        xy = np.vstack(
            (
                np.array(list(itertools.product(xs, ys))),
                np.array([(p.min(axis=0) + p.max(axis=0)) / 2 for p in projections]),
            )
        )
        xy = np.unique(xy, axis=0)
        directions = forward + xy[:, :1] * right + xy[:, 1:] * up
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        nearest = np.full(len(directions), np.inf)
        for path, tree in self.trees.items():
            position, rotation = self.poses[path]
            tree.intersect((eye - position) @ rotation, directions @ rotation, nearest)
        selected = np.isfinite(nearest)
        points = eye + directions[selected] * nearest[selected, None]
        if not len(points):
            raise RuntimeError("Observer geometry rays found no actual robot surface.")
        return points, {
            "candidate_ray_count": len(directions),
            "surface_sample_count": len(points),
            "surface_sampling_seconds": time.perf_counter() - begin,
            "surface_points_world_m": points.tolist(),
        }
