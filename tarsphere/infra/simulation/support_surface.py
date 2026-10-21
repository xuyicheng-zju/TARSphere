"""Extract the actual contact layer of a live authored support.

The planner receives its polygon union, including holes and disconnected
pieces, extruded downward, or a real closed thin tabletop with its curvature
preserved. Furniture bounding boxes never define this mesh.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from tarsphere.infra.control.motion_planning import SUPPORT_SURFACE_SLAB_THICKNESS_M
from tarsphere.infra.simulation.usd_geometry import collision_enabled


def _world_meshes(stage: Any, prim_path: str):
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"Support mesh requires an existing live prim: {prim_path}")
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not child.IsA(UsdGeom.Gprim) or not collision_enabled(child):
            continue
        if child.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(child)
            points = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            holes = set(mesh.GetHoleIndicesAttr().Get() or ())
            if all(
                value is None or len(value) == 0 for value in (points, counts, indices)
            ):
                descendants = Usd.PrimRange(child, Usd.TraverseInstanceProxies())
                if any(p != child and p.IsA(UsdGeom.Gprim) for p in descendants):
                    # GRScenes uses empty Mesh prims as geometry containers.
                    # Descendants are still traversed and validated individually.
                    continue
                raise ValueError(f"Empty live support mesh: {child.GetPath()}")
            if (
                points is None
                or counts is None
                or indices is None
                or sum(counts) != len(indices)
            ):
                raise ValueError(f"Malformed live support topology: {child.GetPath()}")
            faces, offset = [], 0
            for index, count in enumerate(counts):
                if index not in holes:
                    faces.append(list(indices[offset : offset + count]))
                offset += count
        elif child.IsA(UsdGeom.Cube):
            half = float(UsdGeom.Cube(child).GetSizeAttr().Get()) / 2
            points = [
                (x * half, y * half, z * half)
                for x in (-1, 1)
                for y in (-1, 1)
                for z in (-1, 1)
            ]
            faces = [
                (0, 2, 6, 4),
                (1, 5, 7, 3),
                (0, 4, 5, 1),
                (2, 3, 7, 6),
                (0, 1, 3, 2),
                (4, 6, 7, 5),
            ]
        else:
            raise ValueError(
                f"Unsupported live support collider: {child.GetPath()} ({child.GetTypeName()})"
            )
        transform = transforms.GetLocalToWorldTransform(child)
        # Runtime scene placement already converts authored units into metres.
        vertices = np.array(
            [transform.Transform(Gf.Vec3d(*map(float, p))) for p in points]
        )
        if not np.all(np.isfinite(vertices)):
            raise ValueError(f"Non-finite live support geometry: {child.GetPath()}")
        for face in faces:
            if len(face) < 3 or min(face) < 0 or max(face) >= len(vertices):
                raise ValueError(f"Invalid live support face: {child.GetPath()}")
        yield vertices, faces


def _world_faces(stage: Any, prim_path: str):
    for vertices, faces in _world_meshes(stage, prim_path):
        for face in faces:
            yield vertices[list(face)]


def live_support_solid_meshes(stage: Any, prim_path: str) -> tuple[np.ndarray, ...]:
    """Closed collision components only; never close holes or hull the owner.

    PhysX triangle-mesh overlaps test surfaces, not their enclosed interiors.
    Preserve signed shell orientation within each authored mesh: reversed inner
    shells subtract cavities, while distinct furniture meshes remain separate.
    Open/non-manifold components remain surface colliders, not invented solids.
    """
    from collections import defaultdict

    solids = []
    for vertices, raw_faces in _world_meshes(stage, prim_path):
        mesh_triangles = []
        vertices, remap = np.unique(vertices, axis=0, return_inverse=True)
        faces = [tuple(int(remap[i]) for i in face) for face in raw_faces]
        edges = defaultdict(list)
        for i, face in enumerate(faces):
            for a, b in zip(face, (*face[1:], face[0])):
                if a != b:
                    edges[tuple(sorted((a, b)))].append((i, 1 if a < b else -1))
        adjacency = defaultdict(list)
        boundary = set()
        for occurrences in edges.values():
            if len(occurrences) != 2:
                boundary.update(i for i, _ in occurrences)
            for a, sa in occurrences:
                for b, sb in occurrences:
                    if a != b:
                        adjacency[a].append((b, -sa * sb))
        unseen = set(range(len(faces)))
        while unseen:
            first = unseen.pop()
            orientation, pending = {first: 1}, [first]
            consistent = True
            while pending:
                current = pending.pop()
                for adjacent, factor in adjacency[current]:
                    sign = orientation[current] * factor
                    if adjacent in orientation:
                        consistent &= orientation[adjacent] == sign
                    else:
                        orientation[adjacent] = sign
                        unseen.discard(adjacent)
                        pending.append(adjacent)
            if boundary.intersection(orientation):
                continue
            if not consistent or any(sign != 1 for sign in orientation.values()):
                raise ValueError(f"Inconsistent closed support winding: {prim_path}")
            for i, sign in orientation.items():
                face = faces[i]
                mesh_triangles.extend(
                    (face[0], face[j], face[j + 1]) for j in range(1, len(face) - 1)
                )
        if mesh_triangles:
            solids.append(vertices[np.asarray(mesh_triangles, dtype=int)])
    return tuple(solids)


def inside_support_solid(point_xyz, meshes: Sequence[np.ndarray]) -> bool:
    """Test signed shell sums per authored mesh, preserving enclosed cavities."""
    for triangles in meshes:
        point = np.asarray(point_xyz, dtype=float)
        if np.any(point < triangles.min(axis=(0, 1))) or np.any(
            point > triangles.max(axis=(0, 1))
        ):
            continue
        vectors = triangles - point
        lengths = np.linalg.norm(vectors, axis=2)
        if np.any(lengths <= 1e-9):
            return True
        a, b, c = np.moveaxis(vectors, 1, 0)
        la, lb, lc = lengths.T
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        denominator = (
            la * lb * lc
            + np.einsum("ij,ij->i", a, b) * lc
            + np.einsum("ij,ij->i", b, c) * la
            + np.einsum("ij,ij->i", c, a) * lb
        )
        if abs(float((2 * np.arctan2(numerator, denominator)).sum())) > 2 * np.pi:
            return True
    return False


def _upper_contact_layer(stage, prim_path, footprint, bottom_z):
    """Extract the real piecewise-affine top even when it shares a mesh with legs.

    Overlay source triangle boundaries and their plane intersections, then keep
    the highest real face in each region. Extrusion is downward only; neither
    a hull nor a furniture bounding rectangle can fill a hole or bridge a gap.
    """
    from itertools import combinations
    from shapely import constrained_delaunay_triangles, union_all, STRtree
    from shapely.geometry import LineString, Polygon
    from shapely.geometry.polygon import orient
    from shapely.ops import polygonize, split

    projections, planes = [], []
    for face in _world_faces(stage, prim_path):
        for i in range(1, len(face) - 1):
            triangle = face[[0, i, i + 1]]
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            if normal[2] <= 1e-10 or np.any(np.abs(triangle[:, 2] - bottom_z) > 0.02):
                continue
            projections.append(Polygon(triangle[:, :2]))
            ab = -normal[:2] / normal[2]
            planes.append(np.r_[ab, triangle[0, 2] - ab @ triangle[0, :2]])
    if not projections:
        return None
    planes = np.asarray(planes)
    tree = STRtree(projections)
    pieces = []
    for region in polygonize(union_all([p.boundary for p in projections])):
        point = region.representative_point()
        indices = [int(i) for i in tree.query(point) if projections[i].covers(point)]
        if not indices:
            continue
        fragments = [region]
        # A genuine crossing of two sloped surfaces changes the upper envelope
        # inside their XY overlap, not necessarily on either triangle boundary.
        for a, b in combinations(indices, 2):
            plane = planes[a] - planes[b]
            coords = np.asarray(region.exterior.coords)
            differences = coords @ plane[:2] + plane[2]
            if differences.min() >= -1e-10 or differences.max() <= 1e-10:
                continue
            center = np.asarray(region.centroid.coords[0])
            center -= (
                plane[:2] * (center @ plane[:2] + plane[2]) / (plane[:2] @ plane[:2])
            )
            direction = np.array([-plane[1], plane[0]])
            direction /= np.linalg.norm(direction)
            span = 2 * np.max(np.linalg.norm(coords - center, axis=1)) + 1.0
            line = LineString([center - span * direction, center + span * direction])
            fragments = [
                part for fragment in fragments for part in split(fragment, line).geoms
            ]
        for fragment in fragments:
            xy = np.asarray(fragment.representative_point().coords[0])
            index = max(indices, key=lambda i: float(planes[i, :2] @ xy + planes[i, 2]))
            pieces.append((fragment, planes[index]))
    if not pieces:
        return None
    projection = union_all([piece for piece, _ in pieces])
    # Coverage alone tolerates sub-source-resolution cracks. Collision vertices
    # below retain the actual faces; this buffer never enters the planner mesh.
    scale = max(abs(float(v)) for p in projections for v in p.bounds)
    source_epsilon_m = max(1e-9, 4 * float(np.spacing(np.float32(scale))))
    if not projection.buffer(source_epsilon_m).covers(footprint):
        return None
    # cuRobo needs the real support contact layer in the hand's local closing
    # envelope, not decorative tiers on the far side of the same furniture.
    # Clip the already-resolved upper envelope to 10 cm beyond the target
    # footprint: this contains the Dex3 hand sweep while retaining every real
    # slope, seam, hole, and disconnected patch inside that envelope. It is a
    # local mesh crop, never a hull, bounding box, or invented plane.
    local_region = footprint.buffer(0.10, join_style="mitre")
    local_pieces = []
    for piece, plane in pieces:
        clipped = piece.intersection(local_region)
        polygons = (
            [clipped]
            if clipped.geom_type == "Polygon"
            else getattr(clipped, "geoms", ())
        )
        local_pieces.extend(
            (polygon, plane)
            for polygon in polygons
            if polygon.geom_type == "Polygon" and polygon.area > 1e-12
        )
    pieces = local_pieces
    if not pieces:
        return None
    target_heights, all_heights = [], []
    for piece, plane in pieces:
        all_heights.extend(
            (np.asarray(piece.exterior.coords) @ plane[:2] + plane[2]).tolist()
        )
        overlap = piece.intersection(footprint)
        for polygon in (
            [overlap]
            if overlap.geom_type == "Polygon"
            else getattr(overlap, "geoms", ())
        ):
            if polygon.geom_type == "Polygon" and polygon.area > 0:
                target_heights.extend(
                    (
                        np.asarray(polygon.exterior.coords) @ plane[:2] + plane[2]
                    ).tolist()
                )
    if not target_heights:
        return None
    lower = max(all_heights) - SUPPORT_SURFACE_SLAB_THICKNESS_M
    if min(all_heights) <= lower:
        return None
    # Boolean union removes internal prism walls. Merely checking each prism
    # as closed is insufficient: coincident opposite walls corrupt Warp SDF.
    import manifold3d as mf

    prism_faces = np.asarray(
        [
            (0, 1, 2),
            (5, 4, 3),
            (0, 3, 4),
            (0, 4, 1),
            (1, 4, 5),
            (1, 5, 2),
            (2, 5, 3),
            (2, 3, 0),
        ],
        dtype=np.uint64,
    )
    prisms = []
    for piece, plane in pieces:
        for triangle in constrained_delaunay_triangles(piece).geoms:
            if triangle.area <= 1e-12:
                continue
            xy = np.asarray(orient(triangle, sign=1.0).exterior.coords)[:3]
            vertices = np.r_[
                np.c_[xy, xy @ plane[:2] + plane[2]], np.c_[xy, np.full(3, lower)]
            ]
            prism = mf.Manifold(mf.Mesh64(vertices, prism_faces))
            if prism.status() != mf.Error.NoError:
                raise ValueError("Cannot form the real support contact prism")
            prisms.append(prism)
    # 0.1 micrometre only reconciles float32-scale tessellation; CPU regressions
    # retain 0.44 micrometre cracks and real holes, and the real tabletop varies
    # by millimetres. No geometric buffer enters the collision mesh.
    tolerance_m = 1e-7
    solid = (
        mf.Manifold.batch_boolean(prisms, mf.OpType.Add)
        .as_original()
        .simplify(tolerance_m)
    )
    mesh = solid.to_mesh64()
    # Keep candidate construction in world float64. This upper envelope may
    # lose to a higher real planar or thin contact layer below, so a losing
    # candidate must not fail an unrelated robot-pose/float32 check. The one
    # selected mesh is transformed exactly once by
    # ``support_surface_obstacle_in_planning_root`` and is then fail-loud
    # validated at the Warp boundary by ``require_support_surface_obstacle``.
    return (
        max(target_heights),
        np.asarray(mesh.vert_properties)[:, :3],
        np.asarray(mesh.tri_verts, dtype=np.int64),
    )


def live_support_surface_mesh(
    stage: Any,
    prim_path: str,
    *,
    target_bounds_xyz: Sequence[Sequence[float]],
    target_support_separation_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return closed world-metre vertices/faces at the target's actual layer.

    Match an actual surface under the target footprint within 2 cm of its
    measured bottom. The upper envelope wins over an underside; penetration
    beyond 1 mm is rejected. A live target--support PhysX separation, when
    available, measures actual penetration rather than comparing unrelated
    extrema of a tilted target AABB. Distinct lower tiers are not projected up.
    """
    from shapely import constrained_delaunay_triangles, set_precision, union_all
    from shapely.geometry import Polygon, box
    from shapely.geometry.polygon import orient

    bounds = np.asarray(target_bounds_xyz, dtype=float)
    if (
        bounds.shape != (2, 3)
        or not np.all(np.isfinite(bounds))
        or np.any(bounds[1] <= bounds[0])
    ):
        raise ValueError("Support contact requires finite ordered target bounds xyz")
    bottom_z = bounds[0, 2]
    if target_support_separation_m is not None:
        if (
            not np.isfinite(target_support_separation_m)
            or target_support_separation_m > 0.02
        ):
            raise ValueError("Support requires finite near-contact native separation")
        if target_support_separation_m < -0.001:
            raise ValueError(
                f"Target penetrates the actual upper support surface on {prim_path}"
            )

    def check_penetration(height):
        # An axis-aligned target bbox can extend below the contact point when
        # a book, carton, or bottle settles with a small tilt. It is not signed
        # target--support distance. Only the native pair query above may prove
        # penetration; when that measurement is unavailable, retain the real
        # support mesh without inventing a collision from unrelated extrema.
        del height

    footprint = box(*bounds[0, :2], *bounds[1, :2])
    layers: list[tuple[float, list]] = []
    for face in _world_faces(stage, prim_path):
        if np.ptp(face[:, 2]) > 1e-4:
            continue
        polygon = Polygon(face[:, :2])
        if polygon.area <= 1e-10:
            continue
        if not polygon.is_valid:
            raise ValueError(f"Invalid horizontal support polygon under {prim_path}")
        height = float(face[:, 2].mean())
        if abs(height - bottom_z) > 0.02:
            continue
        layer = next((p for z, p in layers if abs(z - height) <= 1e-4), None)
        if layer is None:
            layers.append((height, [polygon]))
        else:
            layer.append(polygon)
    # Normalize intersections BEFORE triangulation at the source float32
    # geometric resolution. A finer fixed grid preserves sub-source-precision
    # seams that become degenerate triangles or non-manifold edges in CDT.
    # This is topology normalization, not a collision margin or polygon buffer.
    coordinates = [
        np.asarray(p.exterior.coords) for _, polygons in layers for p in polygons
    ]
    extent = (
        float(np.ptp(np.concatenate(coordinates), axis=0).max()) if coordinates else 0.0
    )
    topology_grid_m = max(1e-9, float(np.spacing(np.float32(extent))))
    candidates = [
        (
            z,
            union_all(
                [set_precision(p, topology_grid_m) for p in polygons],
                grid_size=topology_grid_m,
            ),
        )
        for z, polygons in layers
    ]
    candidates = [(z, shape) for z, shape in candidates if shape.covers(footprint)]
    # Compare true thin components with planar layers: a curved upper face
    # must take precedence over its horizontal underside, including penetration.
    from tarsphere.infra.control.motion_planning import SupportSurfaceObstacle

    thin_surfaces = []
    for vertices, raw_faces in _world_meshes(stage, prim_path):
        if not 0 < np.ptp(vertices[:, 2]) <= SUPPORT_SURFACE_SLAB_THICKNESS_M:
            continue
        vertices, remap = np.unique(vertices, axis=0, return_inverse=True)
        triangles = np.asarray(
            [
                (remap[face[0]], remap[face[i]], remap[face[i + 1]])
                for face in raw_faces
                for i in range(1, len(face) - 1)
            ],
            dtype=np.int64,
        )
        projected, heights = [], []
        for triangle in vertices[triangles]:
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            if normal[2] <= 0:
                continue
            polygon = Polygon(triangle[:, :2])
            overlap = polygon.intersection(footprint)
            if overlap.is_empty or overlap.area <= 0:
                continue
            xy = np.asarray(overlap.exterior.coords)
            z = triangle[0, 2] - ((xy - triangle[0, :2]) @ normal[:2]) / normal[2]
            projected.append(polygon)
            heights.extend(z.tolist())
        if not projected or not union_all(projected).covers(footprint):
            continue
        height = max(heights)
        if abs(height - bottom_z) > 0.02:
            continue
        check_penetration(height)
        # A source mesh may be open or contain duplicate/internal faces. It
        # cannot enter the planner directly, but its real upper faces may
        # still define a valid closed contact layer through the union below.
        try:
            SupportSurfaceObstacle(
                prim_path,
                tuple(map(tuple, vertices)),
                tuple(map(tuple, triangles)),
                (0.0, 0.0, 1.0),
            )
        except ValueError:
            continue
        thin_surfaces.append((height, vertices, triangles))
    if thin_surfaces:
        height, vertices, triangles = max(thin_surfaces, key=lambda item: item[0])
        if not candidates or height > max(z for z, _ in candidates) + 1e-6:
            return vertices, triangles
    upper = _upper_contact_layer(stage, prim_path, footprint, bottom_z)
    if upper is not None and (
        not candidates or upper[0] > max(z for z, _ in candidates) + 1e-6
    ):
        height, vertices, triangles = upper
        check_penetration(height)
        SupportSurfaceObstacle(
            prim_path,
            tuple(map(tuple, vertices)),
            tuple(map(tuple, triangles)),
            (0.0, 0.0, 1.0),
        )
        return vertices, triangles
    if not candidates:
        raise ValueError(
            f"No actual horizontal contact layer under target on {prim_path}"
        )
    height, shape = max(candidates, key=lambda pair: pair[0])
    check_penetration(height)
    polygons = [shape] if shape.geom_type == "Polygon" else list(shape.geoms)
    vertices, faces, indices = [], [], {}

    def vertex(xy, z):
        key = tuple(float(v) for v in (*xy, z))
        if key not in indices:
            indices[key] = len(vertices)
            vertices.append(key)
        return indices[key]

    lower = height - SUPPORT_SURFACE_SLAB_THICKNESS_M
    for polygon in polygons:
        # Point-touching polygon components are separate closed solids. Welding
        # their coincident vertical edges would create a four-face edge.
        indices = {}
        polygon = orient(polygon, sign=1.0)
        triangles = constrained_delaunay_triangles(polygon)
        if abs(sum(t.area for t in triangles.geoms) - polygon.area) > 1e-8:
            raise ValueError(
                "Support triangulation does not preserve the contact surface"
            )
        for triangle in triangles.geoms:
            xy = list(orient(triangle, sign=1.0).exterior.coords)[:-1]
            faces.append(tuple(vertex(p, height) for p in xy))
            faces.append(tuple(vertex(p, lower) for p in reversed(xy)))
        # Outer ring is CCW, holes CW: their side walls face the real exterior.
        for ring in [polygon.exterior, *polygon.interiors]:
            for a, b in zip(ring.coords, list(ring.coords)[1:]):
                at, bt, ab, bb = (
                    vertex(a, height),
                    vertex(b, height),
                    vertex(a, lower),
                    vertex(b, lower),
                )
                faces.extend([(at, ab, bb), (at, bb, bt)])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)
