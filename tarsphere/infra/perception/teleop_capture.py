"""Observation-only ego stills, published as one complete diagnostic bundle.

Capture runs on the Isaac main thread. It never commands an actor, changes the
viewport or policy camera, pauses the timeline, or exposes segmentation to the
Robot Agent. A single temporary camera/RenderProduct is released before PNG
encoding. Unchanged live articulation and native Human poses are required.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import threading
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from tarsphere.infra.control.human_native_motion_trace import (
    json_value,
    read_character_transform,
)


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    result = np.asarray(value, dtype=float)
    if not result.size or not np.isfinite(result).all():
        raise ValueError("Still capture requires finite, nonempty native measurements.")
    return result


def _same_state(before: Any, after: Any) -> bool:
    if isinstance(before, dict):
        return (
            isinstance(after, dict)
            and before.keys() == after.keys()
            and all(_same_state(value, after[key]) for key, value in before.items())
        )
    if isinstance(before, list):
        return (
            isinstance(after, list)
            and len(before) == len(after)
            and all(_same_state(a, b) for a, b in zip(before, after))
        )
    if isinstance(before, float):
        return isinstance(after, (int, float)) and math.isclose(
            before, after, rel_tol=0.0, abs_tol=1e-6
        )
    return type(before) is type(after) and before == after


def _actor_masks(
    ids: np.ndarray, labels: Mapping, roots: Mapping[str, str]
) -> dict[str, np.ndarray]:
    """Combine native leaf-instance IDs by exact USD ancestry, never by name."""
    ids = np.asarray(ids)
    if ids.ndim != 2 or ids.dtype.kind not in "iu":
        raise ValueError("Native instance IDs must be a two-dimensional integer image.")
    paths = {}
    visible = set(map(int, np.unique(ids)))
    for key, label in labels.items():
        if isinstance(label, Mapping):
            label = label.get("primPath") or label.get("prim_path")
        if isinstance(label, str) and label.startswith("/"):
            paths[int(key)] = label
        elif int(key) in visible - {0, 1}:
            raise ValueError(f"Visible instance {key} has no native USD path.")
    if visible - set(paths) - {0, 1}:
        raise ValueError("Native instance labels do not cover the captured image.")
    return {
        name: np.isin(
            ids,
            [
                key
                for key, path in paths.items()
                if path == root or path.startswith(root.rstrip("/") + "/")
            ],
        )
        for name, root in roots.items()
    }


class TeleopCapture:
    """Capture the current ego view without advancing physical or actor state.

    ``output_dir`` must resolve outside the source tree. ``resolution`` is a
    native pixel size whose aspect ratio must match the unchanged ego optics.
    ``actor_layers`` adds native visible-pixel RGBA layers for the robot, each
    declared Human, and a declared physical-task target. Empty layers mean zero
    visible pixels; they are not synthesized or painted through occluders.

    ``capture(label)`` returns metadata with an absolute ``directory`` and
    bundle-relative ``files``. Repeated labels create separate bundles. A bundle
    is visible only after rendering, state verification, resource release and
    PNG/JSON encoding all succeed. No failed capture is published.
    """

    def __init__(
        self,
        env: Any,
        output_dir: Path | str,
        resolution: tuple[int, int] = (2560, 1920),
        *,
        actor_layers: bool = False,
    ) -> None:
        if len(resolution) != 2 or any(
            isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in resolution
        ):
            raise ValueError("Still capture resolution requires two positive integers.")
        self.output_dir = Path(output_dir).expanduser().resolve()
        if self.output_dir.is_relative_to(Path(__file__).resolve().parents[3]):
            raise ValueError("Still captures must be written outside the source tree.")
        self.env = env
        self.resolution = tuple(resolution)
        self.actor_layers = bool(actor_layers)
        self._busy = False
        self._closed = False

    def capture(self, label: str = "frame") -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Still capture is closed.")
        if self._busy:
            raise RuntimeError("A still capture is already in progress.")
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Still capture must run on the Isaac main thread.")
        if not isinstance(label, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,64}", label
        ):
            raise ValueError(
                "Capture label must be 1–64 ASCII letters, digits, _ or -."
            )
        self._busy = True
        runtime = None
        try:
            runtime = _NativeCapture(self.env)
            before = runtime.snapshot()
            try:
                camera = runtime.prepare(self.resolution, self.actor_layers)
                rgb, segmentation = runtime.render()
            finally:
                # Release RTX resources even when rendering/annotation fails.
                runtime.close()
            after = runtime.snapshot()
            if not _same_state(before, after):
                raise RuntimeError(
                    "Still capture changed physics, actor poses or a source camera; "
                    "no files were published. Actor state is never teleported back."
                )
            height, width = rgb.shape[:2]
            if rgb.dtype != np.uint8 or rgb.shape != (*self.resolution[::-1], 3):
                raise RuntimeError(
                    "Native RGB does not match the requested resolution."
                )
            images = {"rgb.png": Image.fromarray(rgb)}
            layers = {}
            if self.actor_layers:
                ids, labels = segmentation
                if np.asarray(ids).shape != (height, width):
                    raise ValueError(
                        "Native RGB and instance IDs must share one canvas."
                    )
                masks = _actor_masks(ids, labels, before["actor_roots"])
                for index, (actor, mask) in enumerate(masks.items()):
                    path = f"actors/{index:03d}.png"
                    rgba = np.zeros((height, width, 4), dtype=np.uint8)
                    rgba[mask, :3] = rgb[mask]
                    rgba[mask, 3] = 255
                    images[path] = Image.fromarray(rgba)
                    layers[actor] = {"path": path, "visible_pixels": int(mask.sum())}
            capture_id = f"{label}_{uuid4().hex}"
            directory = self.output_dir / capture_id
            metadata = {
                "schema": "tarsphere.teleop_capture.v1",
                "capture_id": capture_id,
                "label": label,
                "directory": str(directory),
                "resolution": list(self.resolution),
                "physics_step": before["physics_step"],
                "simulation_time_s": before["simulation_time_s"],
                "camera": camera,
                "camera_signature": hashlib.sha256(
                    json.dumps(camera, sort_keys=True, allow_nan=False).encode()
                ).hexdigest(),
                "state": before,
                "freeze_verified": True,
                "files": {"rgb": "rgb.png", "actor_layers": layers},
            }
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=".capture_", dir=self.output_dir
            ) as tmp:
                staging = Path(tmp) / "bundle"
                staging.mkdir()
                for path, image in images.items():
                    target = staging / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    image.save(target, format="PNG", optimize=False, compress_level=6)
                (staging / "metadata.json").write_text(
                    json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)
                    + "\n",
                    encoding="utf-8",
                )
                if directory.exists():
                    raise FileExistsError(f"Still capture already exists: {directory}")
                staging.rename(directory)
            return metadata
        finally:
            self._busy = False

    def close(self) -> None:
        if self._busy:
            raise RuntimeError("Cannot close a still capture while it is in progress.")
        self._closed = True


class _NativeCapture:
    """One temporary native ego camera/product; no reset or actor-control APIs."""

    def __init__(self, env: Any) -> None:
        import omni.usd

        self.env = env
        self.stage = omni.usd.get_context().get_stage()
        if self.stage is None:
            raise RuntimeError("Still capture requires a live Isaac stage.")
        self.source_path = env._agent_camera_path()
        if not self.source_path:
            raise RuntimeError("Still capture requires a live robot ego camera.")
        self.camera_path = self.source_path + "/TARSphereStill_" + uuid4().hex
        self.product = None
        self.annotators = {}

    def camera_description(self, path: str) -> dict[str, Any]:
        from pxr import Usd, UsdGeom

        camera = UsdGeom.Camera(self.stage.GetPrimAtPath(path))
        if not camera:
            raise RuntimeError(f"Capture source is not a USD Camera: {path}")
        optics = {}
        for name in UsdGeom.Camera.GetSchemaAttributeNames(includeInherited=False):
            value = camera.GetPrim().GetAttribute(name).Get()
            if value is not None:
                optics[str(name)] = json_value(
                    value if isinstance(value, (str, bool, int, float)) else list(value)
                )
        return {
            "path": path,
            "authored_world_transform": _array(
                camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            ).tolist(),
            "optics": optics,
        }

    def snapshot(self) -> dict[str, Any]:
        import omni.timeline
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Usd, UsdGeom, UsdSkel

        timeline = omni.timeline.get_timeline_interface()
        robot = self.env._live_robots()
        if len(robot) != 1:
            raise RuntimeError("Still capture requires one live episode robot.")
        robot = robot[0]
        position, orientation = robot.get_robot_base().get_pose()
        root = self.env._robot_root_prim_path()
        if not root or not self.stage.GetPrimAtPath(root):
            raise RuntimeError("Still capture cannot resolve the live robot root.")
        actor_roots = {f"robot:{self.env.episode.robot.entity_name}": root}
        humans = {}
        for human in self.env.episode.humans:
            human_id = str(human.id)
            body = self.env._human_body
            path = body._character_prim_paths[human_id]
            actor_roots[f"human:{human_id}"] = path
            if human_id in body._scene_bound_ids:
                humans[human_id] = {
                    "pose_source": "scene_bound_static_person",
                    "world_transform": _array(
                        UsdGeom.Xformable(
                            self.stage.GetPrimAtPath(path)
                        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    ).tolist(),
                }
                continue
            import omni.anim.graph.core as ag

            skelroot = body._character_skelroot_paths[human_id]
            character = ag.get_character(skelroot)
            if character is None:
                raise RuntimeError(f"No native animation character for {human_id!r}.")
            joints = {}
            for prim in Usd.PrimRange(self.stage.GetPrimAtPath(skelroot)):
                if prim.IsA(UsdSkel.Skeleton):
                    names = list(UsdSkel.Skeleton(prim).GetJointsAttr().Get() or ())
                    leaves = [str(name).rsplit("/", 1)[-1] for name in names]
                    if len(set(leaves)) != len(leaves):
                        raise RuntimeError(
                            "Native Human skeleton has ambiguous joint names."
                        )
                    joints[str(prim.GetPath())] = {
                        str(name): read_character_transform(character, leaf)
                        for name, leaf in zip(names, leaves)
                    }
            if not joints or not any(joints.values()):
                raise RuntimeError(f"No readable native skeleton for {human_id!r}.")
            humans[human_id] = {
                "world_pose": read_character_transform(character),
                "joints": joints,
            }
        objects = {}
        attachment = self.env._g1_attachment
        if attachment is not None:
            for item in attachment.objects:
                if item.pickable:
                    objects[item.object_id] = _array(
                        UsdGeom.Xformable(
                            self.stage.GetPrimAtPath(item.prim_path)
                        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    ).tolist()
            target_id = (
                self.env.episode.world_knowledge.get("physical_task") or {}
            ).get("target_object_id")
            if target_id:
                targets = [
                    item for item in attachment.objects if item.object_id == target_id
                ]
                if len(targets) != 1:
                    raise RuntimeError(
                        "Declared capture target has no unique live object."
                    )
                actor_roots[f"object:{target_id}"] = targets[0].prim_path
        viewport = None
        if not self.env.headless:
            from omni.kit.viewport.utility import get_active_viewport

            active = get_active_viewport()
            if active is not None:
                viewport = self.camera_description(str(active.camera_path))
        return {
            "physics_step": int(self.env.physics_steps),
            "engine_physics_step": int(SimulationManager.get_num_physics_steps()),
            "simulation_time_s": float(SimulationManager.get_simulation_time()),
            "timeline_time_s": float(timeline.get_current_time()),
            "timeline_playing": bool(timeline.is_playing()),
            "actor_roots": actor_roots,
            "robot": {
                "position": _array(position).tolist(),
                "orientation_wxyz": _array(orientation).tolist(),
                "joint_positions": _array(
                    robot.articulation.get_joint_positions()
                ).tolist(),
            },
            "humans": humans,
            "objects_authored_world_transforms": objects,
            "ego_camera": self.camera_description(self.source_path),
            "viewport_camera": viewport,
        }

    def prepare(
        self, resolution: tuple[int, int], actor_layers: bool
    ) -> dict[str, Any]:
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom

        source = UsdGeom.Camera(self.stage.GetPrimAtPath(self.source_path))
        native_resolution = tuple(self.env._robot_camera_sensor().resolution)
        if len(native_resolution) != 2 or any(v <= 0 for v in native_resolution):
            raise ValueError(
                "The live ego render product has no valid native resolution."
            )
        native_aspect = float(native_resolution[0]) / float(native_resolution[1])
        horizontal = float(source.GetHorizontalApertureAttr().Get())
        if not math.isfinite(horizontal) or horizontal <= 0:
            raise ValueError(
                "The live ego camera has no finite positive horizontal aperture."
            )
        if not math.isclose(native_aspect, resolution[0] / resolution[1], rel_tol=1e-5):
            raise ValueError(
                "Capture resolution must preserve the native ego aspect ratio."
            )
        # Identity child follows live articulation/Fabric and gaze exactly, as
        # RoomObserver's presentation ego does. No policy-camera attribute is set.
        camera = UsdGeom.Camera.Define(self.stage, self.camera_path)
        camera.MakeMatrixXform().Set(Gf.Matrix4d(1.0))
        for name in UsdGeom.Camera.GetSchemaAttributeNames(includeInherited=False):
            value = source.GetPrim().GetAttribute(name).Get()
            if value is not None:
                camera.GetPrim().GetAttribute(name).Set(value)
        # RepCamera derives square-pixel vertical intrinsics from horizontal
        # aperture and RP resolution. An authored USD vertical aperture can be
        # stale; normalize only this independent copy to the actual native RP.
        camera.GetVerticalApertureAttr().Set(horizontal / native_aspect)
        self.product = rep.create.render_product(self.camera_path, resolution)
        self.product.hydra_texture.set_updates_enabled(False)
        for role, name in [("rgb", "rgb")] + (
            [("ids", "instance_id_segmentation")] if actor_layers else []
        ):
            annotator = rep.AnnotatorRegistry.get_annotator(
                name, init_params={"colorize": False} if role == "ids" else {}
            )
            self.annotators[role] = annotator
            annotator.attach(self.product)
        return {
            "source_path": self.source_path,
            "source_resolution": list(native_resolution),
            "pose_source": "identity_child_of_live_ego_camera",
            "source": self.camera_description(self.source_path),
            "optics": self.camera_description(self.camera_path)["optics"],
        }

    def render(self) -> tuple[np.ndarray, tuple | None]:
        import omni.replicator.core as rep

        with _render_only_updates():
            self.product.hydra_texture.set_updates_enabled(True)
            try:
                # Bounded native warmup, at one physical instant. Never upscale
                # or substitute cached RGB when a new annotator is not ready.
                for _ in range(8):
                    rep.orchestrator.step(
                        rt_subframes=1,
                        delta_time=0.0,
                        pause_timeline=False,
                        wait_for_render=True,
                    )
                    self.env._record_render()
                    rgba = np.asarray(self.annotators["rgb"].get_data())
                    segmentation = (
                        self.annotators["ids"].get_data()
                        if "ids" in self.annotators
                        else None
                    )
                    if not rgba.size or (
                        segmentation is not None
                        and not np.asarray(segmentation["data"]).size
                    ):
                        continue
                    if rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
                        raise RuntimeError(
                            "Native still annotator returned invalid RGB shape."
                        )
                    return rgba[..., :3].copy(), (
                        (
                            np.asarray(segmentation["data"]).copy(),
                            dict(segmentation["info"]["idToLabels"]),
                        )
                        if segmentation is not None
                        else None
                    )
                raise RuntimeError(
                    "Native still annotators were not ready after 8 zero-time renders."
                )
            finally:
                self.product.hydra_texture.set_updates_enabled(False)

    def close(self) -> None:
        errors = []
        for annotator in self.annotators.values():
            try:
                annotator.detach(self.product)
            except Exception as exc:
                errors.append(exc)
        self.annotators.clear()
        if self.product is not None:
            try:
                self.product.destroy()
            except Exception as exc:
                errors.append(exc)
            self.product = None
        try:
            if self.stage.GetPrimAtPath(self.camera_path):
                self.stage.RemovePrim(self.camera_path)
        except Exception as exc:
            errors.append(exc)
        if errors:
            raise RuntimeError("Cannot release still capture resources: " + str(errors))


@contextmanager
def _render_only_updates():
    """Use kit_updates' no-physics contract, restoring an unset setting as unset."""
    import carb.settings

    settings = carb.settings.get_settings()
    path = "/app/player/playSimulations"
    previous = settings.get(path)
    settings.set_bool(path, False)
    try:
        yield
    finally:
        if previous is None:
            settings.destroy_item(path)
        else:
            settings.set(path, previous)
