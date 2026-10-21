#!/usr/bin/env python3
"""Render a trajectory JSON exported by e2e_grasp_demo to an MP4 video.

Pyrender + EGL → numbered PNGs → ffmpeg encode. Adapted from
NewtonDataGen/scripts/render_trajectory_mp4.py to handle a full scene
(static env + object + per-frame robot link poses) and an optional grasp
overlay.

Usage:
  python -m end2end.render_trajectory_mp4 \
      --trajectory runs/<ts>/trajectory.json \
      --output     runs/<ts>/trajectory.mp4
"""

from __future__ import annotations

# Force EGL + headless pyglet *before* importing pyrender. These take effect
# on import; nothing earlier in this module imports OpenGL machinery.
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("PYGLET_HEADLESS", "true")

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyrender
import trimesh
from PIL import Image

logging.basicConfig(format="%(asctime)s [RENDER] %(message)s", level=logging.INFO)
log = logging.getLogger("render_mp4")


# ---------------------------------------------------------------------------
# Camera and scene helpers
# ---------------------------------------------------------------------------


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n


def camera_pose_from_lookat(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    z = _normalize(eye - target)
    x = _normalize(np.cross(up, z))
    if np.linalg.norm(x) < 1e-8:
        alt_up = (
            np.array([0, 1, 0])
            if abs(up[2]) > 0.9
            else np.array([0, 0, 1], dtype=float)
        )
        x = _normalize(np.cross(alt_up, z))
    y = np.cross(z, x)

    pose = np.eye(4)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = eye
    return pose


def _resolve(base_dir: Path, mesh_rel: str) -> Optional[Path]:
    """Resolve mesh_rel against base_dir, falling back to absolute path."""
    p = Path(mesh_rel)
    if p.is_absolute() and p.is_file():
        return p
    candidate = (base_dir / mesh_rel).resolve()
    if candidate.is_file():
        return candidate
    return None


def _load_mesh(
    cache: Dict[str, Optional[trimesh.Trimesh]], base_dir: Path, mesh_rel: str
) -> Optional[trimesh.Trimesh]:
    if mesh_rel in cache:
        return cache[mesh_rel]
    p = _resolve(base_dir, mesh_rel)
    if p is None:
        log.warning("Mesh not found: %s (base=%s)", mesh_rel, base_dir)
        cache[mesh_rel] = None
        return None
    try:
        m = trimesh.load(p, force="mesh")
        if not isinstance(m, trimesh.Trimesh):
            # Loaded a Scene; concatenate sub-meshes.
            pieces = [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)]
            m = trimesh.util.concatenate(pieces) if pieces else None
    except Exception as e:
        log.warning("Failed to load mesh %s: %s", p, e)
        m = None
    cache[mesh_rel] = m
    return m


# ---------------------------------------------------------------------------
# Scene building
# ---------------------------------------------------------------------------

ROBOT_COLOR = np.array([130, 145, 165, 255], dtype=np.uint8) / 255.0
TABLE_COLOR = np.array([200, 180, 140, 255], dtype=np.uint8) / 255.0
OBJECT_COLOR = np.array([90, 165, 90, 255], dtype=np.uint8) / 255.0
GRASP_COLOR = np.array([220, 60, 60, 255], dtype=np.uint8) / 255.0
# Distinct color for the drop bin so it visually stands apart from the
# tabletop. Used by ``pick_and_drop_in_bin``-style scenes.
BIN_COLOR = np.array([90, 130, 215, 255], dtype=np.uint8) / 255.0
# Per-object colors cycled through for multi-object scenes (clutter task).
# Index by object slot in `traj.objects[*]`.
OBJECT_COLORS = [
    np.array([90, 165, 90, 255], dtype=np.uint8) / 255.0,  # green
    np.array([215, 165, 60, 255], dtype=np.uint8) / 255.0,  # gold
    np.array([180, 90, 180, 255], dtype=np.uint8) / 255.0,  # purple
    np.array([220, 90, 90, 255], dtype=np.uint8) / 255.0,  # red
    np.array([90, 200, 200, 255], dtype=np.uint8) / 255.0,  # cyan
]


# Module-level flag set by --no-texture / --textures from main(). When
# False, the renderer ALWAYS uses the uniform-color path, even for
# meshes that carry texture coordinates. Disabling textures roughly
# 4–6x speeds up rendering for textured HOPE scenes (the pyrender
# texture upload + sampling is the bottleneck on EGL).
RENDER_USE_TEXTURES = True

# Optional uniform object styling (set by --object-color / --object-metallic).
# When OBJECT_OVERRIDE_COLOR is not None, every multi-object entry is painted
# with that single RGBA color instead of cycling OBJECT_COLORS. When
# OBJECT_METALLIC is True, object meshes use a high-metallic / low-roughness
# PBR material so they read as shiny metal.
OBJECT_OVERRIDE_COLOR: Optional[np.ndarray] = None
OBJECT_METALLIC = False


def _add_mesh(
    scene: pyrender.Scene,
    mesh: trimesh.Trimesh,
    T: np.ndarray,
    color: np.ndarray,
    metallic: bool = False,
):
    if mesh is None or len(mesh.faces) == 0:
        return
    # Two render paths:
    #  * If the trimesh carries a real UV-mapped texture (HOPE-dataset
    #    assets and similar) AND --textures is on, keep it — render
    #    with the embedded material so the texture image actually
    #    appears in the MP4.
    #  * Otherwise, strip whatever inherited visuals trimesh attached
    #    (which may be the source OBJ's per-group materials — e.g. the
    #    bin's floor faces inheriting the table color) and overlay
    #    uniform vertex_colors so the whole mesh paints with the
    #    requested color. (face_colors trips pyrender's smooth-mesh
    #    check; vertex_colors works for both smooth and faceted meshes.)
    import trimesh.visual as _tv

    keep_texture = (
        RENDER_USE_TEXTURES
        and isinstance(mesh.visual, _tv.TextureVisuals)
        and getattr(mesh.visual, "uv", None) is not None
        and getattr(getattr(mesh.visual, "material", None), "image", None) is not None
    )
    if keep_texture:
        pm = pyrender.Mesh.from_trimesh(mesh)
        scene.add(pm, pose=T)
        return

    rgba_255 = (np.asarray(color) * 255.0).clip(0, 255).astype(np.uint8)
    try:
        mesh_clean = mesh.copy()
        mesh_clean.visual = trimesh.visual.ColorVisuals(
            mesh_clean,
            vertex_colors=np.tile(rgba_255[None, :], (len(mesh_clean.vertices), 1)),
        )
    except Exception:
        mesh_clean = mesh
    pm = pyrender.Mesh.from_trimesh(
        mesh_clean,
        material=pyrender.MetallicRoughnessMaterial(
            baseColorFactor=color,
            metallicFactor=0.95 if metallic else 0.2,
            roughnessFactor=0.18 if metallic else 0.6,
        ),
    )
    scene.add(pm, pose=T)


def _build_scene(
    frame_idx: int,
    frame: dict,
    traj: dict,
    base_dir: Path,
    mesh_cache: Dict[str, Optional[trimesh.Trimesh]],
    width: int,
    height: int,
    show_grasps: bool,
    skip_link_names: Optional[List[str]] = None,
) -> Optional[pyrender.Scene]:
    # Per-frame camera takes priority over the trajectory's global camera —
    # used by wrist/in-hand camera renders where the camera moves with a link.
    cam = frame.get("camera") or traj.get("camera", {})
    eye = cam.get("eye", [1.0, -0.8, 1.0])
    target = cam.get("target", [0.5, 0.0, 0.7])
    up = cam.get("up", [0.0, 0.0, 1.0])
    skip_link_names = set(skip_link_names or [])

    scene = pyrender.Scene(
        ambient_light=np.array([0.35, 0.35, 0.35, 1.0]),
        bg_color=np.array(traj.get("background_color", [0.95, 0.95, 0.95, 1.0])),
    )

    # Static env meshes (table, bin, ...). Multi-object scenes carry their
    # objects in ``traj.objects`` (a list) plus per-frame ``object_poses``
    # so they don't appear here.
    for name, item in (traj.get("static") or {}).items():
        m = _load_mesh(mesh_cache, base_dir, item["mesh_rel"])
        if m is None:
            continue
        T = np.asarray(item["transform"], dtype=float)
        obj_metallic = False
        if name == "object":
            color = (
                OBJECT_OVERRIDE_COLOR
                if OBJECT_OVERRIDE_COLOR is not None
                else OBJECT_COLOR
            )
            obj_metallic = OBJECT_METALLIC
        elif name == "bin":
            color = BIN_COLOR
        else:
            color = TABLE_COLOR
        _add_mesh(scene, m, T, color, metallic=obj_metallic)

    # Multi-object: paint each entry of ``traj.objects`` at its per-frame
    # world transform from ``frame.object_poses[asset_id]``. Each object
    # gets a distinct color cycled from OBJECT_COLORS.
    objects_meta = traj.get("objects") or []
    object_poses = frame.get("object_poses") or {}
    rendered_object_ids: set[str] = set()
    for i, obj_meta in enumerate(objects_meta):
        obj_id = obj_meta["id"]
        T_list = object_poses.get(obj_id)
        if T_list is None:
            continue
        m = _load_mesh(mesh_cache, base_dir, obj_meta["mesh_rel"])
        if m is None:
            continue
        T = np.asarray(T_list, dtype=float)
        color = (
            OBJECT_OVERRIDE_COLOR
            if OBJECT_OVERRIDE_COLOR is not None
            else OBJECT_COLORS[i % len(OBJECT_COLORS)]
        )
        _add_mesh(scene, m, T, color, metallic=OBJECT_METALLIC)
        rendered_object_ids.add(obj_id)

    # Per-frame robot links (plus a per-frame "object" entry in dynamic
    # playback mode, where the manipulation target moves under physics).
    # Skip the "object" entry if we already rendered it as a multi-object
    # in the block above — otherwise we'd draw it twice (once green, once
    # whichever multi-object color we cycled to).
    for part in frame.get("parts", []):
        pname = part.get("name")
        if pname in skip_link_names:
            continue
        if pname == "object" and rendered_object_ids:
            continue
        m = _load_mesh(mesh_cache, base_dir, part["mesh_rel"])
        if m is None:
            continue
        T = np.asarray(part["transform"], dtype=float)
        color = OBJECT_COLOR if pname == "object" else ROBOT_COLOR
        _add_mesh(scene, m, T, color)

    # Optional grasp overlay
    if show_grasps:
        grasps = (traj.get("annotations") or {}).get("all_grasps") or []
        target_grasp = (traj.get("annotations") or {}).get("target_grasp_transform")
        # Render grasps as small coordinate-frame triads using thin cylinders.
        for g in grasps:
            _draw_frame(scene, np.asarray(g, dtype=float), length=0.04, radius=0.002)
        if target_grasp is not None:
            _draw_frame(
                scene,
                np.asarray(target_grasp, dtype=float),
                length=0.08,
                radius=0.0035,
                color=GRASP_COLOR,
            )

    # Camera + lighting
    aspect = float(width) / float(height)
    pyr_cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=aspect)
    cam_pose = camera_pose_from_lookat(eye, target, up=up)
    scene.add(pyr_cam, pose=cam_pose)

    key_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.5)
    scene.add(key_light, pose=cam_pose)
    fill_eye = (np.asarray(eye) * np.array([-1.0, -1.0, 1.0])).tolist()
    fill_pose = camera_pose_from_lookat(fill_eye, target, up=up)
    fill_light = pyrender.DirectionalLight(color=[0.85, 0.85, 0.95], intensity=2.0)
    scene.add(fill_light, pose=fill_pose)

    return scene


def _draw_frame(
    scene: pyrender.Scene,
    T: np.ndarray,
    length: float = 0.05,
    radius: float = 0.002,
    color: Optional[np.ndarray] = None,
):
    """Draw an XYZ triad (RGB) at T, or a single color frame if color provided."""
    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=8)
    # Move so cylinder origin is at base, extending +Z.
    cyl.apply_translation([0, 0, length / 2.0])
    axis_colors = ([1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1])
    axis_rotations = (
        trimesh.transformations.rotation_matrix(np.pi / 2.0, [0, 1, 0]),  # X
        trimesh.transformations.rotation_matrix(-np.pi / 2.0, [1, 0, 0]),  # Y
        np.eye(4),  # Z
    )
    for ac, R in zip(axis_colors, axis_rotations):
        m = cyl.copy()
        m.apply_transform(R)
        c = color if color is not None else np.asarray(ac)
        _add_mesh(scene, m, T, c)


# ---------------------------------------------------------------------------
# Main render loop
# ---------------------------------------------------------------------------


def render(
    traj_path: Path,
    output: Path,
    base_dir: Path,
    resolution: Tuple[int, int],
    frame_skip: int,
    fps_override: int,
    show_grasps: bool,
    skip_link_names: Optional[List[str]] = None,
):
    traj = json.loads(traj_path.read_text())
    frames = traj.get("frames", [])
    if not frames:
        log.warning("No frames in trajectory; nothing to render.")
        return

    width, height = resolution
    traj_fps = int(traj.get("fps", 25))
    fps = fps_override if fps_override > 0 else max(1, traj_fps // max(1, frame_skip))

    log.info(
        "Trajectory: %d frames @ %d fps; rendering @ %dx%d, skip=%d, fps=%d",
        len(frames),
        traj_fps,
        width,
        height,
        frame_skip,
        fps,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="render_e2e_"))

    renderer = pyrender.OffscreenRenderer(width, height)
    mesh_cache: Dict[str, Optional[trimesh.Trimesh]] = {}
    rendered = 0

    try:
        for i, frame in enumerate(frames):
            if i % max(1, frame_skip) != 0:
                continue
            scene = _build_scene(
                i,
                frame,
                traj,
                base_dir,
                mesh_cache,
                width,
                height,
                show_grasps,
                skip_link_names=skip_link_names,
            )
            if scene is None:
                continue
            color, _ = renderer.render(scene)
            out_png = tmp_dir / f"frame_{rendered:06d}.png"
            Image.fromarray(color).save(out_png)
            rendered += 1
            if rendered % 50 == 0:
                log.info("  rendered %d frames", rendered)
    finally:
        renderer.delete()

    if rendered == 0:
        log.warning("0 frames rendered.")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    log.info("rendered %d frames; encoding MP4", rendered)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(tmp_dir / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        "-preset",
        "medium",
        str(output),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if res.returncode != 0:
        log.error("ffmpeg failed:\n%s", res.stderr)
        sys.exit(1)
    size_mb = output.stat().st_size / 1e6
    log.info("MP4 saved: %s (%.1f MB)", output, size_mb)


def main():
    ap = argparse.ArgumentParser(description="Render trajectory JSON to MP4")
    ap.add_argument("--trajectory", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory for resolving relative mesh paths "
        "(defaults to trajectory['base_dir'] or trajectory file's parent)",
    )
    ap.add_argument("--resolution", type=str, default="960x720")
    ap.add_argument("--frame-skip", type=int, default=1)
    ap.add_argument("--fps", type=int, default=0)
    ap.add_argument(
        "--show-grasps",
        action="store_true",
        help="Overlay predicted grasps + target grasp on every frame",
    )
    ap.add_argument(
        "--skip-link",
        action="append",
        default=[],
        help="Skip rendering robot link by name (repeatable). "
        "Useful for wrist-camera renders where you don't want to "
        "see the robot arm in front of the camera.",
    )
    ap.add_argument(
        "--no-texture",
        action="store_true",
        help="Disable texture rendering even for HOPE meshes "
        "that carry UV+image. ~4–6x faster on EGL since "
        "pyrender's texture upload is the bottleneck.",
    )
    ap.add_argument(
        "--object-color",
        type=str,
        default=None,
        help="Paint every object one uniform color instead of "
        "cycling per-object colors. Accepts 'r,g,b' floats in "
        "[0,1] or 0-255 (auto-detected), e.g. '0.78,0.80,0.82'.",
    )
    ap.add_argument(
        "--object-metallic",
        action="store_true",
        help="Render objects with a shiny metallic PBR material "
        "(high metallicFactor, low roughness).",
    )
    args = ap.parse_args()
    global RENDER_USE_TEXTURES, OBJECT_OVERRIDE_COLOR, OBJECT_METALLIC
    RENDER_USE_TEXTURES = not args.no_texture
    if args.no_texture:
        log.info("RENDER_USE_TEXTURES = False (textureless, fast path)")
    if args.object_color:
        vals = [float(v) for v in args.object_color.split(",")]
        if len(vals) != 3:
            ap.error("--object-color expects 'r,g,b'")
        if max(vals) > 1.0:  # given in 0-255
            vals = [v / 255.0 for v in vals]
        OBJECT_OVERRIDE_COLOR = np.array(vals + [1.0], dtype=float)
        log.info("OBJECT_OVERRIDE_COLOR = %s", OBJECT_OVERRIDE_COLOR[:3])
    OBJECT_METALLIC = args.object_metallic
    if OBJECT_METALLIC:
        log.info("OBJECT_METALLIC = True (shiny metal)")

    w, h = (int(x) for x in args.resolution.split("x"))
    traj_path: Path = args.trajectory
    base_dir = args.base_dir
    if base_dir is None:
        # Try the JSON's hint, else default to trajectory file's parent.
        try:
            data = json.loads(traj_path.read_text())
            base_dir = Path(data.get("base_dir") or traj_path.parent)
        except Exception:
            base_dir = traj_path.parent
    base_dir = Path(base_dir).resolve()

    render(
        traj_path=traj_path,
        output=args.output,
        base_dir=base_dir,
        resolution=(w, h),
        frame_skip=args.frame_skip,
        fps_override=args.fps,
        show_grasps=args.show_grasps,
        skip_link_names=args.skip_link,
    )


if __name__ == "__main__":
    main()
