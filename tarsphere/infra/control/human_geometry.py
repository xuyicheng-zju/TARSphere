"""Measure posed Human surfaces from USD skin weights and native joint poses.

This diagnostic reads the rendered character's skinning inputs. It does not
change collision geometry, navigation clearances, or the character pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


def read_native_animation_graph_state(character, *, node_paths) -> dict:
    """Read node activity from the shipped People Biped graph without advancing it.

    Callers supply paths from the actual bound graph. The installed 107.3
    runtime requires absolute USD node paths; relative paths silently return
    False even for an active clip. This is
    graph telemetry, not a claim that the retargeted body has reached a pose:
    preserve all active nodes and compare them with the measured joints.
    An unavailable API or failed query remains unknown, never false/idle.
    """
    paths = tuple(node_paths)
    if len(paths) > 192 or any(not isinstance(path, str) for path in paths):
        raise ValueError("Graph telemetry requires at most 192 string node paths.")
    query = getattr(character, "is_node_active", None)
    nodes = {}
    errors = {}
    for path in paths:
        try:
            if not callable(query):
                raise RuntimeError(
                    "Native character.is_node_active API is unavailable."
                )
            active = query(path)
            if not isinstance(active, bool):
                raise ValueError(f"Native node activity must be bool, got {active!r}.")
            nodes[path] = active
        except Exception as exc:
            # Optional diagnostics must not stop native movement or recording.
            nodes[path] = None
            errors[path] = f"{type(exc).__name__}: {exc}"
    known_count = sum(value is not None for value in nodes.values())
    return {
        "status": "available"
        if not errors
        else "partial"
        if known_count
        else "unavailable",
        "api": "character.is_node_active",
        "graph_contract": "live AnimationGraph node paths",
        "nodes": nodes,
        "errors": errors,
    }


def read_native_biped_graph_state(character, graph_path: str) -> dict:
    """Read eight native nodes using a cached binding, without traversing USD.

    State prims are not reliable activity nodes in the installed compiled
    graph. Real clip/MotionMatching nodes identify a family; Blend can keep
    both inputs active. An active LookAround input does not prove a nonzero
    blend weight, and an active Sit clip does not prove seat contact.
    """
    if not graph_path or not graph_path.startswith("/") or graph_path == "/":
        return {
            "state": "unknown",
            "status": "unavailable",
            "graph_path": graph_path,
            "active_clip_families": [],
            "active_clip_paths": [],
            "nodes": {},
            "errors": {
                "binding": "Absolute live AnimationGraph binding is unavailable."
            },
        }
    clips = {
        "idle": ("Idle/LookAround_skelanim", "Idle/stand_idle_loop_skelanim"),
        "walk": ("Walk/MotionMatching", "Walk/stand_idle_loop_skelanim"),
        "sit": ("Sit/Sit_skelanim", "Sit/stand_idle_loop_skelanim"),
        "stand": ("Stand/Sit_skelanim",),
    }
    paths = {
        family: tuple(f"{graph_path}/StateMachine/{clip}" for clip in names)
        for family, names in clips.items()
    }
    raw = read_native_animation_graph_state(
        character,
        node_paths=(graph_path, *(path for group in paths.values() for path in group)),
    )
    active_paths = [
        path for group in paths.values() for path in group if raw["nodes"][path] is True
    ]
    families = [
        family
        for family, group in paths.items()
        if any(raw["nodes"][path] is True for path in group)
    ]
    status = raw["status"]
    errors = dict(raw["errors"])
    if raw["nodes"][graph_path] is not True:
        status = "unavailable"
        errors["binding"] = (
            "Native bound AnimationGraph root is not active; node reads cannot establish a pose family."
        )
    state = "unknown"
    if status == "available" and families:
        state = families[0] if len(families) == 1 else "mixed"
    return {
        "state": state,
        "status": status,
        "graph_path": graph_path,
        "active_clip_families": families,
        "active_clip_paths": active_paths,
        "nodes": raw["nodes"],
        "errors": errors,
    }


def describe_native_animation_graph(stage, skeleton_path: str, character) -> dict:
    """Inventory the live bound graph and probe bounded path conventions.

    Some native versions return False for unknown paths without an error.
    Consequently all-False reads of hard-coded asset paths cannot establish
    which state is active. Keep actual USD graph bindings and candidate path
    results so a subsequent physics-tick read can establish a working path.
    This diagnostic never changes the USD stage, variables, or animation time.
    """
    from pxr import Usd

    result = {
        "status": "unavailable",
        "skeleton_path": str(skeleton_path),
        "bindings": [],
        "graph_nodes": [],
        "candidate_paths": [],
        "inventory_truncated": False,
        "errors": [],
    }
    try:
        prim = stage.GetPrimAtPath(skeleton_path)
        graph_paths = []
        while prim and prim.IsValid() and not prim.IsPseudoRoot():
            relationship = prim.GetRelationship("animationGraph")
            if relationship:
                targets = [str(path) for path in relationship.GetForwardedTargets()]
                result["bindings"].append(
                    {"owner_prim": str(prim.GetPath()), "targets": targets}
                )
                graph_paths.extend(targets)
            prim = prim.GetParent()
        seen_candidates = set()

        def candidate(path):
            if path not in seen_candidates and len(result["candidate_paths"]) < 192:
                seen_candidates.add(path)
                result["candidate_paths"].append(path)

        for graph_path in dict.fromkeys(graph_paths):
            graph = stage.GetPrimAtPath(graph_path)
            if not graph or not graph.IsValid():
                result["errors"].append(
                    f"Bound graph prim is unavailable: {graph_path}"
                )
                continue
            for node in Usd.PrimRange(graph):
                if len(result["graph_nodes"]) >= 1000:
                    result["inventory_truncated"] = True
                    break
                absolute = str(node.GetPath())
                relative = absolute[len(graph_path) :]
                type_name = node.GetTypeName()
                result["graph_nodes"].append(
                    {
                        "path": absolute,
                        "relative_path": relative,
                        "type_name": type_name,
                    }
                )
                if type_name in {
                    "AnimationGraph",
                    "StateMachine",
                    "State",
                    "AnimationClip",
                    "MotionMatching",
                }:
                    for path in (
                        relative or "/",
                        relative.lstrip("/") or ".",
                        absolute,
                        "/" + graph.GetName() + relative,
                        graph.GetName() + relative,
                    ):
                        candidate(path)
        if not result["candidate_paths"]:
            result["errors"].append(
                "No live AnimationGraph binding/node candidates found."
            )
            return result
        result["query"] = read_native_animation_graph_state(
            character, node_paths=result["candidate_paths"]
        )
        result["status"] = result["query"]["status"]
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    return result


def read_native_joint_world_matrices(
    stage, skeleton_path: str, character
) -> dict[str, np.ndarray]:
    """Read every actual AnimGraph joint into USD row matrices in world metres.

    AnimGraph returns world translation in stage units and XYZW rotation, but
    no scale. The live Skeleton's inherited uniform scale, converted to
    metres, supplies that missing component. Nonuniform or mirrored skeletons
    cannot be reconstructed from this API and are rejected. Native joint
    names are queried by leaf; returned keys preserve full USD joint paths.
    """
    import carb
    from pxr import Gf, UsdGeom, UsdSkel

    prim = stage.GetPrimAtPath(skeleton_path)
    skeleton = UsdSkel.Skeleton(prim)
    if not skeleton:
        raise ValueError(f"Not a USD Skeleton: {skeleton_path}")
    names = tuple(str(name) for name in skeleton.GetJointsAttr().Get() or ())
    leaves = tuple(name.rsplit("/", 1)[-1] for name in names)
    if not names or len(set(leaves)) != len(leaves):
        raise ValueError("Native Human joint leaf names must be nonempty and unique.")
    unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    basis = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))[:3, :3]
    scales = np.linalg.svd(basis, compute_uv=False)
    if (
        not np.isfinite(scales).all()
        or min(scales) <= 0
        or not np.allclose(scales, scales[0], rtol=1e-5)
        or np.linalg.det(basis) <= 0
    ):
        raise ValueError(
            "Native Human geometry requires a positive uniform Skeleton world scale."
        )
    scale_m = float(scales[0]) * unit
    if not np.isfinite(scale_m) or scale_m <= 0:
        raise ValueError("Native Human geometry requires finite positive stage units.")
    # A position/quaternion API also cannot expose independent joint scaling.
    # The measured Elena asset has rigid bind transforms; reject another
    # contract rather than quietly producing the wrong skin surface.
    binds = np.asarray(skeleton.GetBindTransformsAttr().Get(), dtype=float)
    if binds.shape != (len(names), 4, 4) or not np.isfinite(binds).all():
        raise ValueError(
            "Native Human geometry requires one finite bind transform per joint."
        )
    if not np.allclose(
        np.linalg.svd(binds[:, :3, :3], compute_uv=False), 1.0, atol=1e-5
    ):
        raise ValueError(
            "Native Human geometry cannot recover independently scaled joints."
        )
    matrices = {}
    for name, leaf in zip(names, leaves):
        position = carb.Float3(float("nan"), float("nan"), float("nan"))
        rotation = carb.Float4(float("nan"), float("nan"), float("nan"), float("nan"))
        result = character.get_joint_transform(leaf, position, rotation)
        translation = (
            np.asarray([position.x, position.y, position.z], dtype=float) * unit
        )
        xyzw = np.asarray([rotation.x, rotation.y, rotation.z, rotation.w], dtype=float)
        norm = np.linalg.norm(xyzw)
        if (
            result is False
            or not np.isfinite(translation).all()
            # The native API returns an orientation quaternion, without a
            # unit-length guarantee. Valid retargeted finger poses measured
            # norms above 1.001 during Sit. Quaternion magnitude does not
            # change orientation: normalize finite nonzero values rather than
            # imposing an arbitrary near-one threshold.
            or not np.isfinite(xyzw).all()
            or not np.isfinite(norm)
            or norm <= 0.0
        ):
            raise ValueError(
                f"Native Human joint read failed or returned an invalid pose: {name}; "
                f"result={result!r}, position_m={translation.tolist()}, "
                f"rotation_xyzw={xyzw.tolist()}, quaternion_norm={float(norm)!r}"
            )
        xyzw /= norm
        matrix = np.asarray(
            Gf.Matrix4d().SetRotate(
                Gf.Quatd(float(xyzw[3]), Gf.Vec3d(*xyzw[:3])),
            )
        )
        matrix[:3, :3] *= scale_m
        matrix[3, :3] = translation
        matrices[name] = matrix
    return matrices


@dataclass(frozen=True)
class HumanSkin:
    """One linear-skinned mesh in skeleton bind coordinates."""

    # Live USD mesh path, retained as the key of each sampled surface.
    prim_path: str
    # Vertex XYZ after geomBindTransform, in the Skeleton's authored units.
    points: np.ndarray
    # Per-vertex influence indices into HumanGeometrySampler.joint_names.
    joint_indices: np.ndarray
    # Matching nonnegative influences; every vertex row sums to one.
    joint_weights: np.ndarray


class HumanGeometrySampler:
    """Apply the asset's skin weights to measured native joint transforms.

    Input transforms use USD row-vector convention (translation in row 3).
    They map skeleton joint-local coordinates into world metres, including
    scale. Keys are the exact joint paths returned by ``joint_names``; a caller
    may query the native character by leaf name but must retain this ordering.
    Missing joints and unsupported skinning methods fail explicitly.
    """

    def __init__(self, joint_names, bind_transforms, skins):
        self.joint_names = tuple(str(name) for name in joint_names)
        binds = np.asarray(bind_transforms, dtype=np.float64)
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("Human skeleton joint names must be nonempty and unique.")
        if binds.shape != (len(self.joint_names), 4, 4) or not np.isfinite(binds).all():
            raise ValueError(
                "Human bind transforms must be one finite 4x4 matrix per joint."
            )
        self._bind_inverses = np.linalg.inv(binds)
        self.skins = tuple(skins)
        if not self.skins:
            raise ValueError("Human skeleton has no bound skin meshes.")
        for skin in self.skins:
            points = np.asarray(skin.points)
            indices = np.asarray(skin.joint_indices)
            weights = np.asarray(skin.joint_weights)
            if (
                points.ndim != 2
                or points.shape[1] != 3
                or not np.isfinite(points).all()
            ):
                raise ValueError(f"{skin.prim_path}: points must be finite XYZ rows.")
            if (
                indices.ndim != 2
                or indices.shape != weights.shape
                or len(indices) != len(points)
            ):
                raise ValueError(
                    f"{skin.prim_path}: one influence row is required per vertex."
                )
            if (
                not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0)
                or np.any(indices >= len(self.joint_names))
            ):
                raise ValueError(
                    f"{skin.prim_path}: skin refers to an unknown joint index."
                )
            if (
                not np.isfinite(weights).all()
                or np.any(weights < 0)
                or not np.allclose(weights.sum(1), 1.0, atol=1e-4)
            ):
                raise ValueError(
                    f"{skin.prim_path}: skin weights must be finite, nonnegative and sum to one."
                )

    @classmethod
    def from_stage(cls, stage, skeleton_path: str):
        """Load the concrete Skeleton and every mesh bound to it from a stage."""
        from pxr import Gf, Usd, UsdGeom, UsdSkel

        skeleton = UsdSkel.Skeleton(stage.GetPrimAtPath(skeleton_path))
        if not skeleton:
            raise ValueError(f"Not a USD Skeleton: {skeleton_path}")
        joint_names = tuple(skeleton.GetJointsAttr().Get() or ())
        skins = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            binding = UsdSkel.BindingAPI(prim)
            bound_skeleton = binding.GetInheritedSkeleton()
            if not bound_skeleton or bound_skeleton.GetPath() != skeleton.GetPath():
                continue
            if binding.GetSkinningMethodAttr().Get() != UsdSkel.Tokens.classicLinear:
                raise ValueError(
                    f"{prim.GetPath()}: only classicLinear skinning can be measured here."
                )
            mesh_joint_names = tuple(binding.GetJointsAttr().Get() or joint_names)
            mesh_to_skeleton = np.asarray(
                [joint_names.index(name) for name in mesh_joint_names]
            )
            points = np.asarray(
                UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64
            )
            # USD defines omitted geomBindTransform as identity.
            geom_bind = binding.GetGeomBindTransformAttr().Get()
            geom_bind = np.asarray(Gf.Matrix4d(1.0) if geom_bind is None else geom_bind)
            bound_points = np.c_[points, np.ones(len(points))] @ geom_bind
            indices_pv = binding.GetJointIndicesPrimvar()
            weights_pv = binding.GetJointWeightsPrimvar()
            influences = weights_pv.GetElementSize()
            if influences < 1 or indices_pv.GetElementSize() != influences:
                raise ValueError(
                    f"{prim.GetPath()}: invalid skin influence element size."
                )
            indices = np.asarray(
                indices_pv.ComputeFlattened(Usd.TimeCode.Default()), dtype=np.int64
            ).reshape(-1, influences)
            weights = np.asarray(
                weights_pv.ComputeFlattened(Usd.TimeCode.Default()), dtype=np.float64
            ).reshape(-1, influences)
            if indices_pv.GetInterpolation() == UsdGeom.Tokens.constant:
                indices = np.broadcast_to(indices, (len(points), influences))
            if weights_pv.GetInterpolation() == UsdGeom.Tokens.constant:
                weights = np.broadcast_to(weights, (len(points), influences))
            skins.append(
                HumanSkin(
                    str(prim.GetPath()),
                    bound_points[:, :3],
                    mesh_to_skeleton[indices],
                    weights,
                )
            )
        return cls(joint_names, skeleton.GetBindTransformsAttr().Get(), skins)

    def sample(
        self, joint_world_matrices_m: Mapping[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Return each posed mesh's world XYZ vertices, in metres."""
        missing = set(self.joint_names) - set(joint_world_matrices_m)
        if missing:
            raise ValueError(f"Native Human pose is missing joints: {sorted(missing)}")
        native = np.asarray(
            [joint_world_matrices_m[name] for name in self.joint_names],
            dtype=np.float64,
        )
        if native.shape != self._bind_inverses.shape or not np.isfinite(native).all():
            raise ValueError(
                "Native Human joint transforms must be finite 4x4 matrices."
            )
        skin_transforms = self._bind_inverses @ native
        posed = {}
        for skin in self.skins:
            homogeneous = np.c_[skin.points, np.ones(len(skin.points))]
            influence_positions = np.einsum(
                "ni,nkij->nkj",
                homogeneous,
                skin_transforms[skin.joint_indices],
                optimize=True,
            )
            posed[skin.prim_path] = np.einsum(
                "nk,nkj->nj", skin.joint_weights, influence_positions
            )[:, :3]
        return posed

    @staticmethod
    def height_sections(meshes: Mapping[str, np.ndarray], height_edges_m) -> list[dict]:
        """Measured world bounds in requested absolute-height intervals.

        These point bounds are evidence for later surface/sweep checks. They
        are not a collision-free certificate and do not enlarge free space.
        """
        edges = np.asarray(height_edges_m, dtype=float)
        if (
            edges.ndim != 1
            or len(edges) < 2
            or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0)
        ):
            raise ValueError(
                "Height interval edges must be finite and strictly increasing."
            )
        points = np.concatenate(tuple(meshes.values()), axis=0)
        sections = []
        for low, high in zip(edges[:-1], edges[1:]):
            selected = points[(points[:, 2] >= low) & (points[:, 2] < high)]
            sections.append(
                {
                    "height_interval_m": [float(low), float(high)],
                    "vertex_count": len(selected),
                    "min_world_m": selected.min(0).tolist() if len(selected) else None,
                    "max_world_m": selected.max(0).tolist() if len(selected) else None,
                }
            )
        return sections
