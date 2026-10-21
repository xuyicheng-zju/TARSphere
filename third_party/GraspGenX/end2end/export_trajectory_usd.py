#!/usr/bin/env python3
"""Export an end-to-end ``trajectory.json`` to a USD animation.

The trajectory JSON produced by :mod:`dynamic_playback` (and consumed by
:mod:`render_trajectory_mp4`) fully describes a pick-and-drop animation:
robot link meshes + per-frame transforms, static table/bin, per-frame
object poses, the textured object mesh, and the render camera. This script
bakes that into a single ``.usda`` stage so the trajectory can be scrubbed
in IsaacSim / Omniverse.

Design mirrors ``render_trajectory_mp4.py``: it is fully decoupled from the
sim — it only reads the JSON + the referenced meshes.

Scene graph::

    /World
      /Robot/<link_name>     UsdGeom.Mesh, xform time-sampled per frame
      /Static/table          UsdGeom.Mesh, single (default) xform
      /Static/bin            UsdGeom.Mesh, single (default) xform
      /Objects/<id>          UsdGeom.Mesh, xform time-sampled, textured
      /Camera                UsdGeom.Camera posed from the JSON camera block
      /Looks/<id>_mat        UsdShade.Material (object texture)

Matrix convention: our transforms are column-vector 4x4 (``world = T @ p``);
USD ``Gf.Matrix4d`` is row-major / row-vector (``v' = v . M``), so every 4x4
is transposed before being written.

Example::

    PYOPENGL_PLATFORM=egl uv run python end2end/export_trajectory_usd.py \\
        --trajectory end2end/runs/usd_franka_chocpudding/trajectory.json \\
        --output     end2end/runs/usd_franka_chocpudding/chocpudding.usda
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import trimesh
import trimesh.visual as tv
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

import json

logging.basicConfig(format="%(asctime)s [USD_EXPORT] %(message)s", level=logging.INFO)
log = logging.getLogger("usd_export")

# Display colors (linear RGB) for the non-textured prims, matching the MP4
# renderer's palette so the USD looks familiar.
ROBOT_COLOR = (0.51, 0.57, 0.65)
TABLE_COLOR = (0.78, 0.71, 0.55)
BIN_COLOR = (0.35, 0.51, 0.84)
OBJECT_COLOR = (0.35, 0.65, 0.35)


# ---------------------------------------------------------------------------
# Mesh resolution / loading (same rules as render_trajectory_mp4._resolve)
# ---------------------------------------------------------------------------
def _resolve(base_dir: Path, mesh_rel: str) -> Path | None:
    p = Path(mesh_rel)
    if p.is_absolute() and p.is_file():
        return p
    candidate = (base_dir / mesh_rel).resolve()
    if candidate.is_file():
        return candidate
    return None


def _load_mesh(
    cache: dict, base_dir: Path, mesh_rel: str, keep_texture: bool
) -> trimesh.Trimesh | None:
    """Load a mesh, optionally preserving its UV texture.

    ``keep_texture=False`` concatenates scene geometry into a single mesh
    (robot links are multi-part ``.dae`` files); colored prims don't need
    UVs. ``keep_texture=True`` loads the object ``.obj`` as-is so the
    ``TextureVisuals`` (UVs + material image) survive.
    """
    key = (mesh_rel, keep_texture)
    if key in cache:
        return cache[key]
    p = _resolve(base_dir, mesh_rel)
    if p is None:
        log.warning("Mesh not found: %s (base=%s)", mesh_rel, base_dir)
        cache[key] = None
        return None
    try:
        if keep_texture:
            m = trimesh.load(p, process=False)
            if isinstance(m, trimesh.Scene):
                geoms = [
                    g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)
                ]
                m = geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(geoms)
        else:
            m = trimesh.load(p, force="mesh")
            if not isinstance(m, trimesh.Trimesh):
                pieces = [
                    g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)
                ]
                m = trimesh.util.concatenate(pieces) if pieces else None
    except Exception as e:  # noqa: BLE001
        log.warning("Failed to load mesh %s: %s", p, e)
        m = None
    cache[key] = m
    return m


def _has_texture(mesh: trimesh.Trimesh) -> bool:
    return (
        isinstance(mesh.visual, tv.TextureVisuals)
        and getattr(mesh.visual, "uv", None) is not None
        and getattr(getattr(mesh.visual, "material", None), "image", None) is not None
    )


def _geom_rgb(visual) -> tuple:
    """Diffuse RGB (0-1) for a trimesh geometry's visual; ROBOT_COLOR fallback.

    Robot ``.dae``/``.obj`` links carry no image texture, only per-surface
    material colors (white shells, dark joints, ...). Pull that color from the
    PBR baseColorFactor, the simple-material diffuse, or the flat main_color.
    """
    mat = getattr(visual, "material", None)
    for attr in ("baseColorFactor", "diffuse"):
        c = getattr(mat, attr, None) if mat is not None else None
        if c is not None:
            c = np.asarray(c, dtype=float)
            if c.size >= 3:
                if c.max() > 1.0:
                    c = c / 255.0
                return (float(c[0]), float(c[1]), float(c[2]))
    mc = getattr(visual, "main_color", None)
    if mc is not None:
        c = np.asarray(mc, dtype=float)
        if c.size >= 3:
            if c.max() > 1.0:
                c = c / 255.0
            return (float(c[0]), float(c[1]), float(c[2]))
    return ROBOT_COLOR


def _load_robot_parts(cache: dict, base_dir: Path, mesh_rel: str):
    """Load a robot link mesh, preserving its per-surface material colors.

    Returns ``(mesh, face_rgb)`` where ``mesh`` is the combined Trimesh and
    ``face_rgb`` is an (n_faces, 3) float array of per-face diffuse colors, or
    ``None`` if the mesh can't be loaded. Unlike ``_load_mesh(keep_texture=
    False)`` (which flattens to one color) this keeps each submesh's color so
    the link renders with its real multi-material appearance.
    """
    key = (mesh_rel, "robotparts")
    if key in cache:
        return cache[key]
    p = _resolve(base_dir, mesh_rel)
    if p is None:
        log.warning("Mesh not found: %s (base=%s)", mesh_rel, base_dir)
        cache[key] = None
        return None
    try:
        loaded = trimesh.load(p, process=False)
        if isinstance(loaded, trimesh.Trimesh):
            geoms = [loaded]
        elif isinstance(loaded, trimesh.Scene):
            geoms = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh)]
        else:
            geoms = []
        geoms = [g for g in geoms if len(g.faces)]
        if not geoms:
            cache[key] = None
            return None
        verts_l, faces_l, rgb_l, off = [], [], [], 0
        for g in geoms:
            v = np.asarray(g.vertices, dtype=np.float64)
            f = np.asarray(g.faces, dtype=np.int64)
            verts_l.append(v)
            faces_l.append(f + off)
            rgb_l.append(np.tile(_geom_rgb(g.visual), (len(f), 1)))
            off += len(v)
        mesh = trimesh.Trimesh(
            vertices=np.concatenate(verts_l),
            faces=np.concatenate(faces_l),
            process=False,
        )
        face_rgb = np.concatenate(rgb_l).astype(np.float32)
        res = (mesh, face_rgb)
    except Exception as e:  # noqa: BLE001
        log.warning("Failed to load robot mesh %s: %s", p, e)
        res = None
    cache[key] = res
    return res


# ---------------------------------------------------------------------------
# USD helpers
# ---------------------------------------------------------------------------
def _gf_matrix(T) -> Gf.Matrix4d:
    """Convert a column-vector 4x4 numpy matrix to a USD (row-vector) matrix."""
    T = np.asarray(T, dtype=np.float64)
    return Gf.Matrix4d(*(T.T.flatten().tolist()))


def _define_mesh(stage: Usd.Stage, path: str, mesh: trimesh.Trimesh) -> UsdGeom.Mesh:
    """Bake a trimesh into a UsdGeom.Mesh prim (geometry only)."""
    prim = UsdGeom.Mesh.Define(stage, path)
    verts = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts))
    prim.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32))
    )
    prim.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
    # Per-vertex normals improve shading; harmless if absent.
    try:
        normals = np.ascontiguousarray(mesh.vertex_normals, dtype=np.float32)
        if normals.shape == verts.shape:
            prim.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
            prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    except Exception:  # noqa: BLE001
        pass
    prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    return prim


def _apply_matte_color(
    stage: Usd.Stage, prim: UsdGeom.Mesh, looks_scope: str, name: str, rgb
) -> None:
    """Bind a matte UsdPreviewSurface (high roughness, non-metallic).

    Bare ``displayColor`` makes Omniverse fall back to a default glossy
    material (the "shiny metal table/bin" look); an explicit
    UsdPreviewSurface with roughness≈0.9 / metallic=0 renders matte.
    The displayColor is kept as a fallback for viewers that ignore
    materials (e.g. quick usdview hydra).
    """
    prim.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    material = _matte_material(stage, looks_scope, name, rgb)
    UsdShade.MaterialBindingAPI(prim.GetPrim()).Bind(material)


def _matte_material(
    stage: Usd.Stage, looks_scope: str, name: str, rgb
) -> UsdShade.Material:
    """Define (without binding) a matte UsdPreviewSurface of color ``rgb``."""
    mat_path = f"{looks_scope}/{name}_mat"
    material = UsdShade.Material.Define(stage, mat_path)
    surface = UsdShade.Shader.Define(stage, f"{mat_path}/surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    return material


def _apply_robot_materials(
    stage: Usd.Stage, prim: UsdGeom.Mesh, looks_scope: str, name: str, face_rgb
) -> None:
    """Bind the link's per-face diffuse colors via GeomSubsets.

    Recreates the multi-material look (e.g. white shells + dark joints) rather
    than one flat ROBOT_COLOR. Near-identical colors are merged into a single
    material partition. A uniform per-face ``displayColor`` is also written as
    a fallback for material-ignoring viewers.
    """
    mesh = UsdGeom.Mesh(prim.GetPrim())
    face_rgb = np.ascontiguousarray(face_rgb, dtype=np.float32)
    pv = UsdGeom.PrimvarsAPI(prim.GetPrim()).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.uniform
    )
    pv.Set(Vt.Vec3fArray.FromNumpy(face_rgb))

    colors, inverse = np.unique(np.round(face_rgb, 4), axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    if len(colors) == 1:
        material = _matte_material(
            stage, looks_scope, name, tuple(float(c) for c in colors[0])
        )
        UsdShade.MaterialBindingAPI(prim.GetPrim()).Bind(material)
        return

    UsdGeom.Subset.SetFamilyType(mesh, "materialBind", UsdGeom.Tokens.partition)
    for i, c in enumerate(colors):
        idx = np.nonzero(inverse == i)[0].astype(np.int32)
        sub = UsdGeom.Subset.CreateGeomSubset(
            mesh,
            f"{name}_mat{i}",
            UsdGeom.Tokens.face,
            Vt.IntArray.FromNumpy(idx),
            "materialBind",
            UsdGeom.Tokens.partition,
        )
        material = _matte_material(
            stage, looks_scope, f"{name}_mat{i}", tuple(float(x) for x in c)
        )
        UsdShade.MaterialBindingAPI(sub.GetPrim()).Bind(material)


def _apply_metallic_color(
    stage: Usd.Stage,
    prim: UsdGeom.Mesh,
    looks_scope: str,
    name: str,
    rgb,
    roughness: float = 0.18,
) -> None:
    """Bind a shiny metallic UsdPreviewSurface (metallic=1, low roughness).

    Reads as polished metal in Isaac Sim / Omniverse RTX: the high
    metallicFactor + low roughness drive specular reflections, and the
    baseColor (``diffuseColor``) tints the metal. ``displayColor`` is kept
    as a fallback for material-ignoring viewers.
    """
    prim.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    mat_path = f"{looks_scope}/{name}_mat"
    material = UsdShade.Material.Define(stage, mat_path)
    surface = UsdShade.Shader.Define(stage, f"{mat_path}/surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(prim.GetPrim()).Bind(material)


def _add_ground_plane(
    stage: Usd.Stage,
    looks_scope: str,
    z: float = 0.0,
    half: float = 15.0,
    rgb=(0.45, 0.45, 0.45),
) -> None:
    """A 30x30 m matte grey ground quad at height ``z`` under the robot."""
    prim = UsdGeom.Mesh.Define(stage, "/World/Static/ground")
    pts = np.array(
        [[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]],
        dtype=np.float32,
    )
    prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
    prim.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.array([4], np.int32)))
    prim.CreateFaceVertexIndicesAttr(
        Vt.IntArray.FromNumpy(np.array([0, 1, 2, 3], np.int32))
    )
    prim.CreateNormalsAttr(
        Vt.Vec3fArray.FromNumpy(np.tile(np.array([0, 0, 1], np.float32), (4, 1)))
    )
    prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    _apply_matte_color(stage, prim, looks_scope, "ground", rgb)


def _bind_texture(
    stage: Usd.Stage,
    mesh_prim: UsdGeom.Mesh,
    mesh: trimesh.Trimesh,
    looks_scope: str,
    name: str,
    tex_rel: str,
) -> None:
    """Write UVs + a UsdPreviewSurface material reading ``tex_rel``."""
    # primvars:st (per-vertex UVs; trimesh uv is one row per vertex).
    uv = np.ascontiguousarray(mesh.visual.uv, dtype=np.float32)
    pv_api = UsdGeom.PrimvarsAPI(mesh_prim.GetPrim())
    st = pv_api.CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set(Vt.Vec2fArray.FromNumpy(uv))

    mat_path = f"{looks_scope}/{name}_mat"
    material = UsdShade.Material.Define(stage, mat_path)

    st_reader = UsdShade.Shader.Define(stage, f"{mat_path}/stReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    tex = UsdShade.Shader.Define(stage, f"{mat_path}/diffuseTexture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_rel)
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st_reader.ConnectableAPI(), "result"
    )
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tex_out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    surface = UsdShade.Shader.Define(stage, f"{mat_path}/surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex_out
    )
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")

    UsdShade.MaterialBindingAPI(mesh_prim.GetPrim()).Bind(material)


# Sphere area lights (radius 2 m, world-space positions). No distant "sun":
# the dome fill + these spheres light the scene uniformly from above.
SPHERE_LIGHTS = (
    ("RobotSphereLight", (0.0, 0.0, 3.0)),
    ("SphereLight_A", (0.0, 0.0, 3.0)),
    ("SphereLight_B", (0.0, 1.0, 3.0)),
    ("SphereLight_C", (0.0, -1.0, 3.0)),
)


def _add_lights(stage: Usd.Stage, robot_center, scope: str = "/World/Lights") -> None:
    """Add a dome (ambient) + several sphere lights so the scene is lit in
    Omniverse / IsaacSim, which otherwise render an unlit black stage.

    ``robot_center`` is accepted for signature stability but no longer used:
    the sphere lights sit at fixed world-space positions (``SPHERE_LIGHTS``).
    """
    UsdGeom.Scope.Define(stage, scope)

    # Ambient environment fill so nothing is pure black.
    dome = UsdLux.DomeLight.Define(stage, f"{scope}/DomeLight")
    dome.CreateIntensityAttr(500.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    # USD dome default maps the texture/colour around the sky; no texture =
    # uniform colour, which is exactly the soft fill we want.

    # Sphere area lights (radius 2 m) for soft local fill from above.
    for nm, pos in SPHERE_LIGHTS:
        sphere = UsdLux.SphereLight.Define(stage, f"{scope}/{nm}")
        sphere.CreateRadiusAttr(2.0)
        sphere.CreateIntensityAttr(800.0)
        sphere.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
        UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2]))
        )


def _add_camera(stage: Usd.Stage, path: str, cam: dict) -> None:
    """Add a UsdGeom.Camera looking from ``eye`` toward ``target``."""
    eye = np.asarray(cam.get("eye", [1.2, 0.0, 1.0]), dtype=np.float64)
    target = np.asarray(cam.get("target", [0.5, 0.0, 0.55]), dtype=np.float64)
    up = np.asarray(cam.get("up", [0.0, 0.0, 1.0]), dtype=np.float64)

    # USD cameras look down their local -Z with +Y up. Build camera-to-world.
    fwd = target - eye
    fwd /= np.linalg.norm(fwd) + 1e-12
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right) + 1e-12
    cam_up = np.cross(right, fwd)
    cam_T = np.eye(4)
    cam_T[:3, 0] = right
    cam_T[:3, 1] = cam_up
    cam_T[:3, 2] = -fwd  # -Z points toward the scene
    cam_T[:3, 3] = eye

    cam_prim = UsdGeom.Camera.Define(stage, path)
    cam_prim.AddTransformOp().Set(_gf_matrix(cam_T))
    # Match the MP4 renderer: pyrender PerspectiveCamera(yfov=pi/4) with a
    # 4:3 aspect (the 480x360 textureless render). USD has no FOV attr, so
    # derive focalLength + apertures from a fixed vertical aperture.
    yfov = np.pi / 4.0  # 45 deg vertical, == render_trajectory_mp4
    aspect = 4.0 / 3.0
    v_aperture = 15.2908  # USD default vertical aperture (mm)
    focal = v_aperture / (2.0 * np.tan(yfov / 2.0))
    cam_prim.CreateVerticalApertureAttr(v_aperture)
    cam_prim.CreateHorizontalApertureAttr(v_aperture * aspect)
    cam_prim.CreateFocalLengthAttr(float(focal))
    cam_prim.CreateFocusDistanceAttr(float(np.linalg.norm(target - eye)))
    cam_prim.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------
def export(
    traj_path: Path,
    output: Path,
    textured: bool = True,
    metallic: bool = False,
    metallic_color=None,
) -> Path:
    traj = json.loads(Path(traj_path).read_text())
    base_dir = Path(traj.get("base_dir") or traj_path.parent).resolve()
    fps = float(traj.get("fps", 30))
    frames = traj["frames"]
    n_frames = len(frames)
    log.info(
        "Loaded trajectory: %d frames @ %g fps (base_dir=%s)", n_frames, fps, base_dir
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tex_dir = output.parent / "textures"

    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(fps)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(max(0, n_frames - 1))

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Looks")

    mesh_cache: dict = {}

    # ---- Static geometry: table + bin (single xform) ----------------------
    for name, color in (("table", TABLE_COLOR), ("bin", BIN_COLOR)):
        meta = (traj.get("static") or {}).get(name)
        if not meta:
            continue
        m = _load_mesh(mesh_cache, base_dir, meta["mesh_rel"], keep_texture=False)
        if m is None:
            continue
        prim = _define_mesh(stage, f"/World/Static/{name}", m)
        _apply_matte_color(stage, prim, "/World/Looks", name, color)
        UsdGeom.Xformable(prim).AddTransformOp().Set(
            _gf_matrix(np.array(meta["transform"]))
        )
        log.info("  static[%s]: %d verts", name, len(m.vertices))

    # ---- Objects: textured, xform time-sampled ----------------------------
    objects = traj.get("objects")
    if objects is None:
        # Single-object legacy schema: static.object moves per-frame in parts.
        objects = []
    obj_xform_ops = {}
    for item in objects:
        oid = item["id"]
        keep_tex = textured and not metallic
        m = _load_mesh(mesh_cache, base_dir, item["mesh_rel"], keep_texture=keep_tex)
        if m is None:
            continue
        prim = _define_mesh(stage, f"/World/Objects/{oid}", m)
        if metallic:
            mc = metallic_color if metallic_color is not None else OBJECT_COLOR
            _apply_metallic_color(stage, prim, "/World/Looks", oid, mc)
            log.info("  object[%s]: %d verts (shiny metallic)", oid, len(m.vertices))
        elif textured and _has_texture(m):
            tex_dir.mkdir(parents=True, exist_ok=True)
            img = m.visual.material.image
            tex_file = tex_dir / f"{oid}.png"
            img.save(tex_file)
            _bind_texture(
                stage, prim, m, "/World/Looks", oid, f"./textures/{tex_file.name}"
            )
            log.info(
                "  object[%s]: %d verts, texture -> %s",
                oid,
                len(m.vertices),
                tex_file.name,
            )
        else:
            _apply_matte_color(stage, prim, "/World/Looks", oid, OBJECT_COLOR)
            log.info("  object[%s]: %d verts (no texture)", oid, len(m.vertices))
        obj_xform_ops[oid] = UsdGeom.Xformable(prim).AddTransformOp()

    # ---- Robot links: one prim per link, xform time-sampled ---------------
    # The grasped object is often tracked twice: once as a scene object
    # (textured, animated via object_poses) and again as a robot "part" rigidly
    # attached to the gripper. The part copy duplicates the geometry and, worse,
    # binds the SAME /World/Looks/<id>_mat path — overwriting the object's
    # texture with a flat color. Skip any part whose name is an object id.
    object_ids = {item["id"] for item in objects}
    link_xform_ops: dict = {}
    first_parts = frames[0].get("parts", [])
    for part in first_parts:
        name = part["name"]
        if name in object_ids:
            log.info("  robot: skipping part '%s' (already a textured object)", name)
            continue
        loaded = _load_robot_parts(mesh_cache, base_dir, part["mesh_rel"])
        if loaded is None:
            continue
        rmesh, face_rgb = loaded
        prim = _define_mesh(stage, f"/World/Robot/{name}", rmesh)
        _apply_robot_materials(stage, prim, "/World/Looks", name, face_rgb)
        link_xform_ops[name] = UsdGeom.Xformable(prim).AddTransformOp()
    log.info("  robot: %d link prims (per-material colors)", len(link_xform_ops))

    # ---- Time samples -----------------------------------------------------
    for t, frame in enumerate(frames):
        tc = Usd.TimeCode(t)
        for part in frame.get("parts", []):
            op = link_xform_ops.get(part["name"])
            if op is not None:
                op.Set(_gf_matrix(np.array(part["transform"])), tc)
        for oid, T in (frame.get("object_poses") or {}).items():
            op = obj_xform_ops.get(oid)
            if op is not None:
                op.Set(_gf_matrix(np.array(T)), tc)

    # ---- Ground plane -----------------------------------------------------
    # Floor = lowest static geometry (table legs reach the true floor; the
    # robot base may sit on a riser above it). Robot center for the sphere
    # light comes from panda_link0 at frame 0.
    link0 = next((p for p in first_parts if p["name"].endswith("link0")), None)
    base_xyz = np.array(link0["transform"])[:3, 3] if link0 else np.zeros(3)
    floor_z = 0.0
    for name in ("table", "bin"):
        meta = (traj.get("static") or {}).get(name)
        m = _load_mesh(mesh_cache, base_dir, meta["mesh_rel"], False) if meta else None
        if m is not None:
            v = np.c_[np.asarray(m.vertices), np.ones(len(m.vertices))]
            floor_z = min(
                floor_z, float((np.array(meta["transform"]) @ v.T).T[:, 2].min())
            )
    _add_ground_plane(stage, "/World/Looks", z=floor_z)
    log.info("  ground plane added (30x30 m @ z=%.3f)", floor_z)

    # ---- Lights -----------------------------------------------------------
    robot_center = (float(base_xyz[0]), float(base_xyz[1]), float(base_xyz[2]) + 0.6)
    _add_lights(stage, robot_center)
    log.info("  lights added (dome + %d sphere lights)", len(SPHERE_LIGHTS))

    # ---- Camera -----------------------------------------------------------
    cam = traj.get("camera")
    if cam:
        _add_camera(stage, "/World/Camera", cam)
        log.info("  camera added (eye=%s)", cam.get("eye"))

    stage.GetRootLayer().Save()
    log.info("USD saved: %s", output)
    return output


# ---------------------------------------------------------------------------
# Validation: compare each prim's world AABB at t=0 to the JSON-driven
# trimesh AABB. Proves the transpose + baking is correct without a GPU.
# ---------------------------------------------------------------------------
def validate(traj_path: Path, output: Path, tol: float = 1e-3) -> bool:
    traj = json.loads(Path(traj_path).read_text())
    base_dir = Path(traj.get("base_dir") or traj_path.parent).resolve()
    stage = Usd.Stage.Open(str(output))
    cache: dict = {}
    ok = True

    def _ref_aabb(mesh_rel, T, keep_tex):
        # Match USD ComputeWorldBound semantics: transform the 8 corners of
        # the local AABB, then take the axis-aligned range (NOT the tight
        # AABB of all transformed verts — those differ for rotated prims).
        m = _load_mesh(cache, base_dir, mesh_rel, keep_tex)
        if m is None:
            return None
        lo, hi = np.asarray(m.bounds)
        corners = np.array(
            [
                [x, y, z]
                for x in (lo[0], hi[0])
                for y in (lo[1], hi[1])
                for z in (lo[2], hi[2])
            ]
        )
        ch = np.c_[corners, np.ones(8)]
        w = (np.asarray(T) @ ch.T).T[:, :3]
        return w.min(0), w.max(0)

    def _usd_aabb(prim_path, time):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None
        bbox = (
            UsdGeom.Imageable(prim)
            .ComputeWorldBound(time, UsdGeom.Tokens.default_)
            .ComputeAlignedRange()
        )
        return np.array(bbox.GetMin()), np.array(bbox.GetMax())

    checks = []
    for name in ("table", "bin"):
        meta = (traj.get("static") or {}).get(name)
        if meta:
            checks.append(
                (
                    f"/World/Static/{name}",
                    meta["mesh_rel"],
                    meta["transform"],
                    False,
                    Usd.TimeCode.Default(),
                )
            )
    frame0 = traj["frames"][0]
    for item in (traj.get("objects") or [])[:1]:
        T0 = frame0.get("object_poses", {}).get(item["id"])
        if T0:
            checks.append(
                (
                    f"/World/Objects/{item['id']}",
                    item["mesh_rel"],
                    T0,
                    True,
                    Usd.TimeCode(0),
                )
            )
    for part in frame0.get("parts", [])[:1]:
        checks.append(
            (
                f"/World/Robot/{part['name']}",
                part["mesh_rel"],
                part["transform"],
                False,
                Usd.TimeCode(0),
            )
        )

    for prim_path, mesh_rel, T, keep_tex, time in checks:
        ref = _ref_aabb(mesh_rel, T, keep_tex)
        got = _usd_aabb(prim_path, time)
        if ref is None or got is None:
            log.warning("  [validate] skip %s (mesh/prim missing)", prim_path)
            continue
        err = max(np.abs(ref[0] - got[0]).max(), np.abs(ref[1] - got[1]).max())
        status = "OK" if err < tol else "MISMATCH"
        if err >= tol:
            ok = False
        log.info(
            "  [validate] %-28s AABB err=%.2e  %s",
            prim_path.split("/")[-1],
            err,
            status,
        )
    log.info("Validation %s", "PASSED" if ok else "FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajectory", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument(
        "--textured",
        dest="textured",
        action="store_true",
        default=True,
        help="bake the object texture (default)",
    )
    ap.add_argument("--no-textured", dest="textured", action="store_false")
    ap.add_argument(
        "--metallic",
        action="store_true",
        help="Bind a shiny metallic material to all objects "
        "(overrides texture), rendered as polished metal in "
        "Isaac Sim.",
    )
    ap.add_argument(
        "--metallic-color",
        type=str,
        default=None,
        help="Base color for --metallic objects as 'r,g,b' in "
        "[0,1] or 0-255 (default brushed silver).",
    )
    ap.add_argument(
        "--no-validate", dest="validate", action="store_false", default=True
    )
    args = ap.parse_args()

    mcolor = None
    if args.metallic_color:
        vals = [float(v) for v in args.metallic_color.split(",")]
        if max(vals) > 1.0:
            vals = [v / 255.0 for v in vals]
        mcolor = tuple(vals)

    out = export(
        args.trajectory,
        args.output,
        textured=args.textured,
        metallic=args.metallic,
        metallic_color=mcolor,
    )
    if args.validate:
        validate(args.trajectory, out)


if __name__ == "__main__":
    main()
