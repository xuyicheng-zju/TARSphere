"""Generic scene loader. Consumes env + robot YAMLs, returns a SceneBundle
that the demo script and the trajectory exporter can both work from.

World frame == robot base frame. cuRobo plans here; viser visualizes here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh
import trimesh.transformations as tra
import yaml

import scene_synthesizer as synth

from registry import (
    CollisionObstacle,
    make_asset,
    make_collision,
)


@dataclass
class SceneObject:
    """One manipulation-target object in the scene.

    Multi-object scenes (e.g. ``clutter_pick_and_drop``) populate
    ``SceneBundle.objects`` with several of these; single-object scenes
    just have one.
    """

    asset_id: str  # unique id within the scene: "object", "object_0", ...
    mesh: trimesh.Trimesh  # mesh in its own local frame
    mesh_path: str  # absolute path to the on-disk mesh
    world_T: np.ndarray  # 4x4 world transform
    label: str | None = None  # human-readable: "box", "banana", "bowl"


@dataclass
class SceneBundle:
    synth_scene: Any  # scene_synthesizer.Scene
    object_world_T: np.ndarray  # 4x4 of object in world frame (legacy: objects[0])
    object_mesh: (
        trimesh.Trimesh
    )  # object in *its own* mesh-local frame (legacy: objects[0])
    object_mesh_path: str  # absolute path to the mesh file (legacy: objects[0])
    collision_world: List[CollisionObstacle]  # cuRobo obstacles
    vis_meshes: Dict[
        str, Tuple[trimesh.Trimesh, np.ndarray]
    ]  # name -> (mesh_local, T_world)
    robot_urdf_path: str
    robot_asset_root: str
    robot_base_T: np.ndarray
    env_cfg: Dict[str, Any]
    robot_cfg: Dict[str, Any]
    # New: full list of manipulation objects. For single-object scenes
    # this is ``[SceneObject(...)]``; for clutter scenes it contains all
    # objects placed on the support. The legacy ``object_*`` fields
    # above always mirror ``objects[0]`` so existing single-object code
    # paths (PickAndLiftTask, PickAndDropInBinTask, the original
    # simulate_and_export path) keep working unchanged.
    objects: List[SceneObject] = field(default_factory=list)


def _xyzw_to_matrix(translation: List[float], q_xyzw: List[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = translation
    if q_xyzw is None:
        return T
    x, y, z, w = q_xyzw
    if abs(x) < 1e-9 and abs(y) < 1e-9 and abs(z) < 1e-9 and abs(w - 1.0) < 1e-9:
        return T
    # tra.quaternion_matrix uses wxyz
    T[:3, :3] = tra.quaternion_matrix([w, x, y, z])[:3, :3]
    return T


def load_yaml(path: str | Path) -> Dict[str, Any]:
    from paths import expand as _expand_tokens

    p = Path(path)
    with p.open() as f:
        # Expand ${CUROBO_ASSETS}/${GRIPPERS}/${E2E}/${REPO} path tokens so
        # committed configs stay free of machine-specific absolute paths.
        return _expand_tokens(yaml.safe_load(f))


def build_scene(
    env_cfg: Dict[str, Any],
    robot_cfg: Dict[str, Any],
    mesh_file: str,
    seed: int | None = None,
) -> SceneBundle:
    """Construct a scene_synthesizer scene + cuRobo collision world.

    Args:
        env_cfg: parsed env YAML (envs/*.yaml)
        robot_cfg: parsed robot/gripper combo YAML (robots/*.yaml)
        mesh_file: path to the manipulation-target mesh (.obj/.stl/...)
        seed: seed for object pose randomization. If the env YAML
            declares ``object_slot.randomize.extents_xy``, the
            translation_offset is shifted by a uniform sample within
            ±extents/2 in the table-local frame, drawn from a numpy RNG
            seeded by this value. Pass the run's `--seed` so the
            randomization is reproducible per trial.
    """
    scene = synth.Scene()

    # Track world-frame transforms of every asset we add (id -> 4x4)
    asset_world_T: Dict[str, np.ndarray] = {}
    # Track scene_synthesizer Asset instances so collision factories can probe them
    asset_objects: Dict[str, Any] = {}

    # 1. Add env assets in order.
    for asset_cfg in env_cfg.get("assets", []):
        asset = make_asset(asset_cfg)
        asset_id = asset_cfg["id"]
        T_world = _xyzw_to_matrix(
            asset_cfg["pose"]["translation"],
            asset_cfg["pose"].get("quaternion_xyzw", [0, 0, 0, 1]),
        )

        scene.add_object(
            asset=asset,
            obj_id=asset_id,
            transform=T_world,
        )

        # Optionally label a support surface on the top of the asset.
        if "support_label" in asset_cfg:
            try:
                scene.label_support(asset_cfg["support_label"], min_area=0.05)
            except Exception:
                # Labeling is best-effort; the demo will still place via offset.
                pass

        asset_world_T[asset_id] = T_world
        asset_objects[asset_id] = asset

    # 2. Resolve the object slot.
    slot = env_cfg["object_slot"]

    # Load the manipulation object mesh.
    mesh_file_abs = str(Path(mesh_file).expanduser().resolve())
    object_mesh = trimesh.load(mesh_file_abs, force="mesh")
    if not isinstance(object_mesh, trimesh.Trimesh):
        object_mesh = trimesh.util.concatenate(list(object_mesh.geometry.values()))
    # Optional mesh_scale + pre_rotation, mirroring build_clutter_scene.
    # Needed for HOPE assets (mm units; canonical pose on side).
    scale = float(slot.get("mesh_scale", 1.0))
    if abs(scale - 1.0) > 1e-9:
        object_mesh = object_mesh.copy()
        object_mesh.apply_scale(scale)
    pre_rot = slot.get("pre_rotation")
    if pre_rot is not None:
        ax = [float(pre_rot[0]), float(pre_rot[1]), float(pre_rot[2])]
        ang_deg = float(pre_rot[3])
        R = tra.rotation_matrix(np.deg2rad(ang_deg), ax)
        object_mesh = object_mesh.copy()
        object_mesh.apply_transform(R)
    obj_bounds = object_mesh.bounds  # (2, 3)
    obj_local_center = (obj_bounds[0] + obj_bounds[1]) / 2.0

    object_world_T = np.eye(4)
    if "world_position" in slot:
        # Explicit world-frame position for the *base* of the object (its mesh
        # min-z corner). Useful for shelves / bins where the object sits on an
        # inner surface that isn't the parent's AABB top.
        wp = slot["world_position"]
        object_world_T[:3, 3] = [
            float(wp[0]) - obj_local_center[0],
            float(wp[1]) - obj_local_center[1],
            float(wp[2]) - float(obj_bounds[0, 2]),
        ]
    else:
        parent_id = slot.get("parent")
        if parent_id is None or parent_id not in asset_world_T:
            raise KeyError(
                f"object_slot needs 'world_position' or a valid 'parent' (got {parent_id!r})"
            )
        parent_T = asset_world_T[parent_id]
        parent_asset = asset_objects[parent_id]
        parent_bounds = parent_asset.mesh().bounds
        # `surface_mode` lets a bin specify "drop the object on the inner
        # bottom" instead of "drop on the AABB top".
        if slot.get("surface_mode") == "inside_bottom":
            wall_thickness = float(slot.get("inside_thickness", 0.005))
            parent_surface_local_z = float(parent_bounds[0, 2]) + wall_thickness
        else:
            parent_surface_local_z = float(parent_bounds[1, 2])
        surface_z_world = parent_T[2, 3] + parent_surface_local_z

        offset = np.array(slot.get("translation_offset", [0.0, 0.0, 0.0]), dtype=float)
        # Optional pose randomization. Sampled in the parent (table)
        # frame, so the table can be at any world pose without
        # affecting the randomization region semantics. Two knobs:
        #
        #   * extents_xy:    [ex, ey] — uniform xy offset in ±extents/2.
        #   * yaw_range_deg: [lo, hi] — uniform yaw (rotation about z)
        #     in degrees. Use [-180, 180] for full planar rotation.
        rcfg = slot.get("randomize")
        rand_yaw_rad = 0.0
        if rcfg is not None and (rcfg.get("extents_xy") or rcfg.get("yaw_range_deg")):
            r_seed = rcfg.get("seed")
            if r_seed is None:
                r_seed = seed
            rng = np.random.default_rng(int(r_seed) if r_seed is not None else None)
            log_bits = []
            if rcfg.get("extents_xy"):
                ex, ey = (float(v) for v in rcfg["extents_xy"])
                dx = rng.uniform(-ex / 2, ex / 2)
                dy = rng.uniform(-ey / 2, ey / 2)
                offset = offset + np.array([dx, dy, 0.0])
                log_bits.append(f"xy=({dx:+.3f}, {dy:+.3f})")
            if rcfg.get("yaw_range_deg"):
                ylo, yhi = (float(v) for v in rcfg["yaw_range_deg"])
                yaw_deg = rng.uniform(ylo, yhi)
                rand_yaw_rad = float(np.deg2rad(yaw_deg))
                log_bits.append(f"yaw={yaw_deg:+.1f}deg")
            print(
                f"[scene_builder] randomized object pose "
                f"(seed={r_seed}): {' '.join(log_bits)}"
            )
        object_world_T[:3, 3] = [
            parent_T[0, 3] + offset[0] - obj_local_center[0],
            parent_T[1, 3] + offset[1] - obj_local_center[1],
            surface_z_world + offset[2] - float(obj_bounds[0, 2]),
        ]
        # Planar yaw randomization (rotation about world +Z applied to
        # the object in-place about its xy centroid). Lets the box land
        # at any orientation on the table while still resting on the
        # same face. We compose: T_world = R_z(yaw) about centroid;
        # equivalent to translating centroid to origin, rotating, then
        # translating back.
        if abs(rand_yaw_rad) > 1e-9:
            cx = object_world_T[0, 3] + obj_local_center[0]
            cy = object_world_T[1, 3] + obj_local_center[1]
            cz = object_world_T[2, 3]
            R = tra.rotation_matrix(rand_yaw_rad, [0, 0, 1])
            # Move centroid to origin, rotate, move back.
            T_in = np.eye(4)
            T_in[:3, 3] = [-cx, -cy, 0.0]
            T_out = np.eye(4)
            T_out[:3, 3] = [cx, cy, 0.0]
            object_world_T = T_out @ R @ T_in @ object_world_T

    # Add object to scene.
    object_asset = synth.MeshAsset(mesh_file_abs)
    scene.add_object(
        asset=object_asset,
        obj_id="object",
        transform=object_world_T,
    )
    asset_world_T["object"] = object_world_T
    asset_objects["object"] = object_asset

    # 3. Add the robot (visual only — cuRobo plans separately).
    # The env YAML can override the robot's default base pose (e.g. shelf
    # scene mounts the robot off-axis next to the shelf instead of directly
    # in front of it). Env override > robot YAML default.
    robot_pose_cfg = env_cfg.get("robot_base_pose") or robot_cfg["robot_base_pose"]
    robot_base_T = _xyzw_to_matrix(
        robot_pose_cfg["translation"],
        robot_pose_cfg.get("quaternion_xyzw", [0, 0, 0, 1]),
    )
    try:
        # URDFAsset wants a configuration aligned with ALL actuated joints in
        # the URDF (including the gripper). cuRobo plans only the arm joints
        # (gripper finger_joint is locked at 0). Pass arm joints + zero-pad to
        # satisfy URDFAsset.
        arm_q = list(robot_cfg["curobo"]["default_joint_position"])
        try:
            import yourdfpy as _ydf

            _u = _ydf.URDF.load(
                robot_cfg["urdf_path"],
                build_collision_scene_graph=False,
                load_meshes=False,
            )
            n_actuated = len(_u.actuated_joint_names)
            arm_q = arm_q + [0.0] * max(0, n_actuated - len(arm_q))
        except Exception:
            pass
        robot_asset = synth.URDFAsset(
            robot_cfg["urdf_path"],
            configuration=arm_q,
        )
        scene.add_object(asset=robot_asset, obj_id="robot", transform=robot_base_T)
        asset_world_T["robot"] = robot_base_T
        asset_objects["robot"] = robot_asset
    except Exception as e:
        # URDFAsset can be picky about meshes — fall back to skipping robot mesh
        # in the scene_synthesizer scene; the demo's viser path renders the robot
        # via cuRobo's FK + URDF anyway.
        print(
            f"[scene_builder] URDFAsset load skipped ({e}); robot will be rendered via cuRobo FK only"
        )
        robot_asset = None

    # 4. Build cuRobo collision world (object is intentionally NOT added — the
    # gripper has to reach it).
    collision_world: List[CollisionObstacle] = []
    for asset_cfg in env_cfg.get("assets", []):
        obstacles = make_collision(asset_cfg, asset_objects[asset_cfg["id"]], scene)
        collision_world.extend(obstacles)
    # Extra clearance regions (cuboids/spheres declared in env YAML directly).
    for extra in env_cfg.get("extra_collision", []):
        if extra.get("type") == "cuboid":
            collision_world.append(
                CollisionObstacle(
                    name=extra["name"],
                    type="cuboid",
                    dims=extra["dims"],
                    pose=extra["pose"],
                )
            )

    # 5. Collect visual meshes for viser/MP4 rendering.
    vis_meshes: Dict[str, Tuple[trimesh.Trimesh, np.ndarray]] = {}
    for asset_id, asset in asset_objects.items():
        if asset_id == "robot":
            # Robot is rendered link-by-link via cuRobo FK; skip combined mesh.
            continue
        try:
            m = asset.mesh()
            if not isinstance(m, trimesh.Trimesh):
                m = (
                    trimesh.util.concatenate(list(m.geometry.values()))
                    if hasattr(m, "geometry")
                    else m
                )
            vis_meshes[asset_id] = (m, asset_world_T[asset_id])
        except Exception:
            pass

    return SceneBundle(
        synth_scene=scene,
        object_world_T=object_world_T,
        object_mesh=object_mesh,
        object_mesh_path=mesh_file_abs,
        collision_world=collision_world,
        vis_meshes=vis_meshes,
        robot_urdf_path=robot_cfg["urdf_path"],
        robot_asset_root=robot_cfg.get(
            "asset_root_path", str(Path(robot_cfg["urdf_path"]).parent)
        ),
        robot_base_T=robot_base_T,
        env_cfg=env_cfg,
        robot_cfg=robot_cfg,
        objects=[
            SceneObject(
                asset_id="object",
                mesh=object_mesh,
                mesh_path=mesh_file_abs,
                world_T=object_world_T.copy(),
                label=Path(mesh_file_abs).stem,
            )
        ],
    )


# ---------------------------------------------------------------------------
# Multi-object scene builder (for tasks like clutter_pick_and_drop)
# ---------------------------------------------------------------------------


def build_clutter_scene(
    env_cfg: Dict[str, Any], robot_cfg: Dict[str, Any], seed: int | None = None
) -> SceneBundle:
    """Build a multi-object scene from an env YAML that declares ``object_slots: [...]``.

    Each entry in ``object_slots`` is shaped like::

        - id: object_0
          label: box
          mesh: /abs/path/to/box.stl
          mesh_scale: 1.0       # optional
          parent: table
          surface: table_top    # optional, sits on parent.AABB top
          translation_offset: [dx, dy, 0.0]
          upright: true
          randomize:
            yaw_range_deg: [-180, 180]
            seed: null          # null = use the run's --seed

    For ``clutter_pick_and_drop``, we expect 4 entries arranged in a 1×4 row
    on the table top (the env YAML carries the row positions; we apply random
    yaw per entry from the env's randomize block).

    Returns a :class:`SceneBundle` with:
      * ``objects`` populated with all N objects' meshes + world transforms
      * legacy ``object_*`` fields mirroring ``objects[0]`` for backward compat
        (so kinematic export / single-object renderer fallback still work)
    """
    scene = synth.Scene()
    asset_world_T: Dict[str, np.ndarray] = {}
    asset_objects: Dict[str, Any] = {}

    # 1. Env assets (table, bin, ...) — same as build_scene.
    for asset_cfg in env_cfg.get("assets", []):
        asset = make_asset(asset_cfg)
        asset_id = asset_cfg["id"]
        T_world = _xyzw_to_matrix(
            asset_cfg["pose"]["translation"],
            asset_cfg["pose"].get("quaternion_xyzw", [0, 0, 0, 1]),
        )
        scene.add_object(asset=asset, obj_id=asset_id, transform=T_world)
        if "support_label" in asset_cfg:
            try:
                scene.label_support(asset_cfg["support_label"], min_area=0.05)
            except Exception:
                pass
        asset_world_T[asset_id] = T_world
        asset_objects[asset_id] = asset

    # 2. Object slots — one SceneObject per entry.
    slots = env_cfg.get("object_slots")
    if not slots:
        raise KeyError("build_clutter_scene needs env_cfg['object_slots'] (list)")

    rng = np.random.default_rng(int(seed) if seed is not None else None)
    objects: List[SceneObject] = []

    # Find a labeled support surface to place onto. We prefer the
    # support_id named by the first slot's ``support`` field; if absent,
    # the most recently labeled support in the scene is used.
    default_support = None
    for asset_cfg in env_cfg.get("assets", []):
        if asset_cfg.get("support_label"):
            default_support = asset_cfg["support_label"]
            break

    # Optional placement region in TABLE-LOCAL frame (m). Constrains the
    # collision-aware sampler to a rectangle that's inside the franka's
    # comfortable reach zone, well clear of the bin. Read from env YAML:
    #
    #   placement_region:
    #     x: [-0.20, 0.10]
    #     y: [-0.15, 0.40]
    region_cfg = env_cfg.get("placement_region")
    if region_cfg is not None:
        rx_lo, rx_hi = (float(v) for v in region_cfg["x"])
        ry_lo, ry_hi = (float(v) for v in region_cfg["y"])
        # Convert to world-frame bounds using the table's world translation.
        # We assume the support's parent's pose is the table's pose.
        # Find the table's translation (the first asset with support_label):
        table_world_xy = None
        for asset_cfg in env_cfg.get("assets", []):
            if asset_cfg.get("support_label"):
                t = asset_cfg["pose"]["translation"]
                table_world_xy = (float(t[0]), float(t[1]))
                break
        if table_world_xy is None:
            rx_lo = rx_hi = ry_lo = ry_hi = None
        else:
            tx, ty = table_world_xy
            world_rx_lo, world_rx_hi = tx + rx_lo, tx + rx_hi
            world_ry_lo, world_ry_hi = ty + ry_lo, ty + ry_hi
    else:
        rx_lo = rx_hi = ry_lo = ry_hi = None

    # Build a trimesh CollisionManager seeded with the env's static
    # meshes (the bin). Each successfully-placed object is added to the
    # manager so subsequent placements know to avoid it.
    placement_manager = trimesh.collision.CollisionManager()
    for asset_id, asset in asset_objects.items():
        if asset_id == "table":
            # Skip the table from the placement-collision set: objects
            # are supposed to REST on the table, not avoid it.
            continue
        try:
            m = asset.mesh()
            if not isinstance(m, trimesh.Trimesh):
                m = (
                    trimesh.util.concatenate(list(m.geometry.values()))
                    if hasattr(m, "geometry")
                    else m
                )
            T = asset_world_T[asset_id]
            placement_manager.add_object(asset_id, m, transform=T)
        except Exception:
            pass

    for slot in slots:
        asset_id = slot["id"]
        mesh_path = str(Path(slot["mesh"]).expanduser().resolve())
        mesh = trimesh.load(mesh_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        # Optional scale.
        scale = float(slot.get("mesh_scale", 1.0))
        if abs(scale - 1.0) > 1e-9:
            mesh = mesh.copy()
            mesh.apply_scale(scale)
        # Optional pre-placement rotation (axis-angle). Lets the YAML
        # reorient the mesh BEFORE it's placed on the support — useful
        # for HOPE assets whose canonical frame has the bottle's long
        # axis along local +Y (so it lies on its side when placed
        # without rotation); ``pre_rotation: [1, 0, 0, 90]`` stands it
        # upright on the table.
        pre_rot = slot.get("pre_rotation")
        if pre_rot is not None:
            ax = [float(pre_rot[0]), float(pre_rot[1]), float(pre_rot[2])]
            ang_deg = float(pre_rot[3])
            R = tra.rotation_matrix(np.deg2rad(ang_deg), ax)
            if not isinstance(mesh, trimesh.Trimesh) or mesh is mesh:  # noqa
                pass
            mesh = mesh.copy()
            mesh.apply_transform(R)
        obj_bounds = mesh.bounds
        obj_local_center = (obj_bounds[0] + obj_bounds[1]) / 2.0

        # Collision-aware placement: sample random (x, y, yaw) within
        # the env's placement_region until trimesh.CollisionManager
        # reports no overlap with the bin or with previously-placed
        # objects. Falls back to the YAML's deterministic row offset
        # if no non-colliding pose found in MAX_PLACEMENT_TRIES.
        obj_asset = synth.MeshAsset(mesh_path)
        if abs(scale - 1.0) > 1e-9:
            obj_asset = synth.MeshAsset(mesh_path, scale=scale)
        placed = False
        if rx_lo is not None and "world_position" not in slot:
            # Resting z = table_top_world + small offset above surface.
            # Use the same surface_z_world computation as the fallback
            # path so the object sits flush on the table.
            parent_id = slot["parent"]
            parent_T = asset_world_T[parent_id]
            parent_asset = asset_objects[parent_id]
            parent_bounds = parent_asset.mesh().bounds
            parent_surface_local_z = float(parent_bounds[1, 2])
            surface_z_world = parent_T[2, 3] + parent_surface_local_z
            obj_resting_z = surface_z_world - float(obj_bounds[0, 2])

            slot_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            MAX_PLACEMENT_TRIES = 200
            yaw_range = slot.get("randomize", {}).get("yaw_range_deg", [-180.0, 180.0])
            ylo, yhi = float(yaw_range[0]), float(yaw_range[1])
            for attempt in range(MAX_PLACEMENT_TRIES):
                x = slot_rng.uniform(world_rx_lo, world_rx_hi)
                y = slot_rng.uniform(world_ry_lo, world_ry_hi)
                yaw = slot_rng.uniform(ylo, yhi)
                R = tra.rotation_matrix(np.deg2rad(yaw), [0, 0, 1])
                # Compose: rotate mesh about its centroid in xy, then
                # translate to (x, y, obj_resting_z).
                T_centroid_to_origin = np.eye(4)
                T_centroid_to_origin[:3, 3] = [
                    -obj_local_center[0],
                    -obj_local_center[1],
                    0.0,
                ]
                T_translate = np.eye(4)
                T_translate[:3, 3] = [x, y, obj_resting_z]
                T_centroid_back = np.eye(4)
                T_centroid_back[:3, 3] = [obj_local_center[0], obj_local_center[1], 0.0]
                T_try = T_translate @ T_centroid_back @ R @ T_centroid_to_origin
                # Apply transform to a temp copy for collision check.
                m_try = mesh.copy()
                m_try.apply_transform(T_try)
                # CollisionManager wants the mesh + an id; transform=eye since
                # we baked it into the mesh.
                placement_manager.add_object("__try__", m_try)
                hit = placement_manager.in_collision_internal()
                placement_manager.remove_object("__try__")
                if not hit:
                    obj_world_T = T_try
                    # Commit this object to the placement manager so
                    # later objects see it as an obstacle.
                    placement_manager.add_object(asset_id, m_try)
                    placed = True
                    print(
                        f"[scene_builder] clutter object {asset_id}: "
                        f"placed at xy=({x:+.3f}, {y:+.3f}) yaw={yaw:+.1f}deg "
                        f"after {attempt + 1} tries"
                    )
                    scene.add_object(
                        asset=obj_asset, obj_id=asset_id, transform=obj_world_T
                    )
                    break
            if not placed:
                print(
                    f"[scene_builder] clutter object {asset_id}: "
                    f"collision-aware sampler exhausted {MAX_PLACEMENT_TRIES} tries; "
                    f"falling back to YAML row offset"
                )
        placed_via_synth = placed

        if not placed_via_synth:
            # Fallback path: compute world_T from the YAML's
            # translation_offset + apply yaw_range_deg manually. Same
            # logic as the single-object slot.
            if "world_position" in slot:
                wp_xyz = slot["world_position"]
                obj_world_T = np.eye(4)
                obj_world_T[:3, 3] = [
                    float(wp_xyz[0]) - obj_local_center[0],
                    float(wp_xyz[1]) - obj_local_center[1],
                    float(wp_xyz[2]) - float(obj_bounds[0, 2]),
                ]
            else:
                parent_id = slot["parent"]
                parent_T = asset_world_T[parent_id]
                parent_asset = asset_objects[parent_id]
                parent_bounds = parent_asset.mesh().bounds
                if slot.get("surface_mode") == "inside_bottom":
                    wall_thickness = float(slot.get("inside_thickness", 0.005))
                    parent_surface_local_z = float(parent_bounds[0, 2]) + wall_thickness
                else:
                    parent_surface_local_z = float(parent_bounds[1, 2])
                surface_z_world = parent_T[2, 3] + parent_surface_local_z
                offset = np.array(
                    slot.get("translation_offset", [0.0, 0.0, 0.0]), dtype=float
                )

                rcfg = slot.get("randomize") or {}
                r_seed = rcfg.get("seed")
                slot_rng = rng if r_seed is None else np.random.default_rng(int(r_seed))
                rand_yaw_rad = 0.0
                if rcfg.get("extents_xy"):
                    ex, ey = (float(v) for v in rcfg["extents_xy"])
                    offset = offset + np.array(
                        [
                            slot_rng.uniform(-ex / 2, ex / 2),
                            slot_rng.uniform(-ey / 2, ey / 2),
                            0.0,
                        ]
                    )
                if rcfg.get("yaw_range_deg"):
                    ylo, yhi = (float(v) for v in rcfg["yaw_range_deg"])
                    rand_yaw_rad = float(np.deg2rad(slot_rng.uniform(ylo, yhi)))

                obj_world_T = np.eye(4)
                obj_world_T[:3, 3] = [
                    parent_T[0, 3] + offset[0] - obj_local_center[0],
                    parent_T[1, 3] + offset[1] - obj_local_center[1],
                    surface_z_world + offset[2] - float(obj_bounds[0, 2]),
                ]
                if abs(rand_yaw_rad) > 1e-9:
                    cx = obj_world_T[0, 3] + obj_local_center[0]
                    cy = obj_world_T[1, 3] + obj_local_center[1]
                    R = tra.rotation_matrix(rand_yaw_rad, [0, 0, 1])
                    T_in = np.eye(4)
                    T_in[:3, 3] = [-cx, -cy, 0.0]
                    T_out = np.eye(4)
                    T_out[:3, 3] = [cx, cy, 0.0]
                    obj_world_T = T_out @ R @ T_in @ obj_world_T

            scene.add_object(asset=obj_asset, obj_id=asset_id, transform=obj_world_T)

        asset_world_T[asset_id] = obj_world_T
        asset_objects[asset_id] = obj_asset

        objects.append(
            SceneObject(
                asset_id=asset_id,
                mesh=mesh,
                mesh_path=mesh_path,
                world_T=obj_world_T,
                label=slot.get("label") or Path(mesh_path).stem,
            )
        )

    # 3. Robot.
    robot_pose_cfg = env_cfg.get("robot_base_pose") or robot_cfg["robot_base_pose"]
    robot_base_T = _xyzw_to_matrix(
        robot_pose_cfg["translation"],
        robot_pose_cfg.get("quaternion_xyzw", [0, 0, 0, 1]),
    )
    try:
        arm_q = list(robot_cfg["curobo"]["default_joint_position"])
        try:
            import yourdfpy as _ydf

            _u = _ydf.URDF.load(
                robot_cfg["urdf_path"],
                build_collision_scene_graph=False,
                load_meshes=False,
            )
            n_actuated = len(_u.actuated_joint_names)
            arm_q = arm_q + [0.0] * max(0, n_actuated - len(arm_q))
        except Exception:
            pass
        robot_asset = synth.URDFAsset(robot_cfg["urdf_path"], configuration=arm_q)
        scene.add_object(asset=robot_asset, obj_id="robot", transform=robot_base_T)
        asset_world_T["robot"] = robot_base_T
        asset_objects["robot"] = robot_asset
    except Exception as e:
        print(f"[scene_builder] URDFAsset load skipped ({e})")

    # 4. cuRobo collision world (env assets only — objects intentionally excluded;
    #    the task layer adds per-object cuboid obstacles as needed when planning).
    collision_world: List[CollisionObstacle] = []
    for asset_cfg in env_cfg.get("assets", []):
        obstacles = make_collision(asset_cfg, asset_objects[asset_cfg["id"]], scene)
        collision_world.extend(obstacles)
    for extra in env_cfg.get("extra_collision", []):
        if extra.get("type") == "cuboid":
            collision_world.append(
                CollisionObstacle(
                    name=extra["name"],
                    type="cuboid",
                    dims=extra["dims"],
                    pose=extra["pose"],
                )
            )

    # 5. Visual meshes for the renderer (env assets only — objects are
    #    written separately as a list).
    vis_meshes: Dict[str, Tuple[trimesh.Trimesh, np.ndarray]] = {}
    for asset_id, asset in asset_objects.items():
        if asset_id == "robot":
            continue
        if any(obj.asset_id == asset_id for obj in objects):
            # Objects are handled via SceneBundle.objects + per-frame poses
            continue
        try:
            m = asset.mesh()
            if not isinstance(m, trimesh.Trimesh):
                m = (
                    trimesh.util.concatenate(list(m.geometry.values()))
                    if hasattr(m, "geometry")
                    else m
                )
            vis_meshes[asset_id] = (m, asset_world_T[asset_id])
        except Exception:
            pass

    return SceneBundle(
        synth_scene=scene,
        # Legacy single-object accessors mirror objects[0] (used by the
        # renderer fallback path and any single-object code that still
        # touches these fields directly).
        object_world_T=objects[0].world_T,
        object_mesh=objects[0].mesh,
        object_mesh_path=objects[0].mesh_path,
        collision_world=collision_world,
        vis_meshes=vis_meshes,
        robot_urdf_path=robot_cfg["urdf_path"],
        robot_asset_root=robot_cfg.get(
            "asset_root_path", str(Path(robot_cfg["urdf_path"]).parent)
        ),
        robot_base_T=robot_base_T,
        env_cfg=env_cfg,
        robot_cfg=robot_cfg,
        objects=objects,
    )
