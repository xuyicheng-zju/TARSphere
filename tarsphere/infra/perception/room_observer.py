"""Third-person recording synchronized to Isaac's native render cadence.

Each configured view owns one camera, RenderProduct, and RGB annotator. Every
camera is positioned before a native Isaac render tick and every annotator is
read after that same tick. Recording never advances Replicator or simulation
time, and its frames are never exposed to the Robot Agent.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from tarsphere.infra.control.physical_contracts import (
    G1_STANDING_BODY_TOP_M,
    G1_STANDING_HEIGHT_COMMAND_M,
    SIMULATION_PHYSICS_HZ,
)


# Scene server publishes the first N names. Default N=1 keeps only the
# translation-following overview; rear_quarter / side / overhead remain
# available when callers raise room_observer_views or pass view_names.
ROOM_OBSERVER_VIEW_NAMES = (
    "follow",
    "rear_quarter",
    "side",
    "overhead",
    "route_bev",
    "presentation_ego",
)
ROOM_OBSERVER_NATIVE_RENDER_WARMUP_LIMIT = 8
ROOM_OBSERVER_RECORDER_ROOT = "/World/TARSPHERERecorders"
ROUTE_BEV_CLIP_HEIGHT_M = 2.05
ROOM_OBSERVER_FAR_CLIP_M = 1_000_000.0
OBSERVER_FOCAL_LENGTH_MM = 10.5
OBSERVER_HORIZONTAL_APERTURE_MM = 20.955
OBSERVER_EYE_CLEARANCE_M = 0.12
OBSERVER_NEAR_CLIP_M = 0.05
OBSERVER_FOLLOW_RECOVERY_SPEED_M_S = 1.2
OBSERVER_FOLLOW_MIN_HORIZONTAL_DISTANCE_M = 1.65


def set_observer_camera_pose(stage, camera_path: str, eye, target) -> None:
    """Author the USD camera directly; headless capture has no viewport dependency."""
    from pxr import Gf, Usd, UsdGeom

    camera = stage.GetPrimAtPath(camera_path)
    if not camera.IsA(UsdGeom.Camera):
        raise RuntimeError(f"Observer camera is unavailable: {camera_path}")
    unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    position = Gf.Vec3d(*(float(v) / unit for v in eye))
    focus = Gf.Vec3d(*(float(v) / unit for v in target))
    direction = focus - position
    if direction.GetLength() < 1e-6:
        raise ValueError("Observer camera position and target must differ.")
    up = (
        Gf.Vec3d(0, 1, 0)
        if abs(direction.GetNormalized()[2]) > 0.999
        else Gf.Vec3d(0, 0, 1)
    )
    world = Gf.Matrix4d().SetLookAt(position, focus, up).GetInverse()
    parent = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
        camera.GetParent()
    )
    UsdGeom.Xformable(camera).MakeMatrixXform().Set(world * parent.GetInverse())


def room_observer_resolution(
    view_name: str, resolution: tuple[int, int]
) -> tuple[int, int]:
    """Each raw camera uses the requested width; ego preserves its 4:3 optics."""
    if view_name == "presentation_ego":
        width = int(resolution[0])
        if width % 8:
            raise ValueError(
                "Presentation ego width must be divisible by 8 for an even 4:3 image."
            )
        return width, width * 3 // 4
    return resolution


def episode_route_points_xy(episode: object) -> tuple[tuple[float, float], ...]:
    """Extract authored route points for generic observer framing.

    Only ``world_knowledge.episode_route`` is traversed.  Scene objects and
    private ground truth therefore cannot accidentally expand the view to the
    full asset, and the observer remains policy-invisible diagnostic output.
    """

    world_knowledge = getattr(episode, "world_knowledge", None)
    if not isinstance(world_knowledge, Mapping):
        return ()
    route = world_knowledge.get("episode_route")
    if not isinstance(route, Mapping):
        return ()
    points: list[tuple[float, float]] = []

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            position = value.get("position_xy")
            if (
                isinstance(position, Sequence)
                and not isinstance(position, (str, bytes))
                and len(position) == 2
            ):
                xy = (float(position[0]), float(position[1]))
                if all(math.isfinite(component) for component in xy):
                    points.append(xy)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for nested in value:
                visit(nested)

    visit(route)
    return tuple(points)


class RoomObserverRecorder:
    """Capture policy-invisible observer views, including an optional ego camera."""

    def __init__(
        self,
        *,
        view_names: Sequence[str] = ROOM_OBSERVER_VIEW_NAMES[:-1],
        resolution: tuple[int, int] = (640, 360),
        capture_stride: int = 1,
        max_frames_per_view: int | None = None,
        gate_presentation_updates: bool = False,
    ) -> None:
        selected_views = tuple(str(name) for name in view_names)
        unknown_views = sorted(set(selected_views) - set(ROOM_OBSERVER_VIEW_NAMES))
        if not selected_views or unknown_views:
            raise ValueError(
                "room observer view_names must contain one or more of "
                f"{ROOM_OBSERVER_VIEW_NAMES!r}; got {selected_views!r}"
            )
        if len(set(selected_views)) != len(selected_views):
            raise ValueError(
                "room observer view_names must not contain duplicates: "
                f"{selected_views!r}"
            )
        if resolution[0] <= 0 or resolution[1] <= 0:
            raise ValueError(f"Invalid room observer resolution: {resolution!r}")
        if capture_stride < 1:
            raise ValueError(
                f"room observer capture_stride must be >= 1, got {capture_stride}"
            )
        if max_frames_per_view is not None and max_frames_per_view < 1:
            raise ValueError(
                "room observer max_frames_per_view must be >= 1 or None, "
                f"got {max_frames_per_view}"
            )
        self.view_names = selected_views
        if gate_presentation_updates and "presentation_ego" not in selected_views:
            raise ValueError("Presentation update gating requires presentation_ego.")
        self.gate_presentation_updates = bool(gate_presentation_updates)
        self._presentation_updates_enabled: bool | None = None
        self.resolution = (int(resolution[0]), int(resolution[1]))
        for name in selected_views:
            room_observer_resolution(name, self.resolution)
        # Only an optical/pose source. Its sensor and render product are never changed.
        self._policy_camera_path: str | None = None
        self.capture_stride = int(capture_stride)
        self.max_frames_per_view = (
            None if max_frames_per_view is None else int(max_frames_per_view)
        )
        self._camera_root_path = f"{ROOM_OBSERVER_RECORDER_ROOT}/RoomObserver"
        self._camera_paths: dict[str, str] = {}
        self._render_products: dict[str, object] = {}
        self._rgb_annotators: dict[str, object] = {}
        self._last_capture_physics_step: int | None = None
        self._scheduled_captures = 0
        self._active_image_dir: Path | None = None
        self._last_paths: dict[str, str] = {}
        self._pending_capture: tuple[Path, int] | None = None
        self._pending_schedule: tuple[int | None, int, dict[str, str]] | None = None
        self._render_resources_ready = False
        self._native_render_warmup_attempts = 0
        self._route_bev_pose: (
            tuple[tuple[float, float, float], tuple[float, float, float]] | None
        ) = None
        self._fixed_view_poses: dict[
            str, tuple[tuple[float, ...], tuple[float, ...]]
        ] = {}
        # Last accepted eye and its tracking anchor, both world metres. Holding
        # the eye on recovery must not advance the anchor and accumulate drift.
        self._follow_reference_pose: tuple[np.ndarray, np.ndarray] | None = None
        self._follow_last_physics_step: int | None = None
        self._pending_camera_selection: dict[str, dict] = {}
        self._pending_camera_targets: dict[str, tuple[float, ...]] = {}
        # Framing reference inferred at the initial standing pose, not a floor measurement.
        self._framing_floor_z: float | None = None
        self._video_streams: dict[str, object] = {}

    def start_video(self, output_dir: Path, slug: str) -> None:
        """Stream batch recordings directly to MP4 instead of accumulating PNGs.

        Interactive scene-server and component calibration still use snapshots.
        Video streams own their lifecycle and must finish before the next episode.
        """
        from tarsphere.infra.perception.observer_video import ObserverVideoStream

        if self._active_image_dir is None or self._video_streams:
            raise RuntimeError(
                "Start the observer episode, and finish prior video streams first."
            )
        try:
            for view in self.view_names:
                self._video_streams[view] = ObserverVideoStream(
                    output_path=Path(output_dir) / f"{slug}_{view}.mp4",
                    resolution=room_observer_resolution(view, self.resolution),
                    stride=self.capture_stride,
                )
        except Exception:
            self.finish_video()
            raise

    def finish_video(self) -> dict:
        streams, self._video_streams = self._video_streams, {}
        results, errors = {}, []
        for view, stream in streams.items():
            try:
                results[view] = {**stream.close(), "view": view}
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(
                f"Observer video encoding failed: {errors!r}"
            ) from errors[0]
        return results

    @property
    def video_capture_steps(self) -> dict[str, int | None]:
        """Last source physics tick actually captured by each open video stream."""
        return {
            name: None if stream.last is None else int(stream.last[0])
            for name, stream in self._video_streams.items()
            if not stream.closed
        }

    def prepare_initial_video_capture(
        self,
        *,
        image_dir: Path,
        physics_step: int,
        position: list[float],
        yaw_rad: float,
    ) -> bool:
        """Start the video sample epoch even if reset saved a PNG at this tick."""
        steps = self.video_capture_steps
        if set(steps) != set(self.view_names) or any(
            tick is not None for tick in steps.values()
        ):
            raise RuntimeError(
                "Initial observer capture requires every requested video stream to be open and empty."
            )
        if self._pending_schedule is not None or self._pending_capture is not None:
            raise RuntimeError(
                "Cannot start initial video capture while a render is pending."
            )
        if not self.accepts_captures(image_dir):
            raise RuntimeError(
                "Initial video capture requires the active episode directory."
            )
        self._last_capture_physics_step = None
        self._scheduled_captures = 0
        return self.prepare_capture(
            image_dir=image_dir,
            physics_step=physics_step,
            position=position,
            yaw_rad=yaw_rad,
        )

    def set_fixed_view_pose(
        self,
        view_name: str,
        eye: Sequence[float],
        target: Sequence[float],
    ) -> None:
        """Frame a component demo from fixed world coordinates, in meters.

        Presentation ego always follows the policy camera and cannot be fixed.
        This changes only an observer camera, never an actor or physics state.
        """
        if view_name not in self.view_names or view_name == "presentation_ego":
            raise ValueError(f"Not a configured third-person view: {view_name}")
        if len(eye) != 3 or len(target) != 3:
            raise ValueError("Fixed observer eye and target require three coordinates.")
        position = tuple(float(v) for v in eye)
        look_at = tuple(float(v) for v in target)
        if (
            not all(math.isfinite(v) for v in (*position, *look_at))
            or math.dist(position, look_at) < 1e-6
        ):
            raise ValueError(
                "Fixed observer requires distinct finite eye and target positions."
            )
        self._fixed_view_poses[view_name] = (position, look_at)

    def start_episode(
        self,
        image_dir: Path,
        *,
        route_points_xy: Sequence[Sequence[float]] = (),
        policy_camera_path: str | None = None,
    ) -> None:
        """Create the render channel before motion and reset its schedule.

        The persistent scene server reuses the same episode image directory
        after ``rmtree``.  An early return on path equality left the previous
        episode's physics-step cursor in place, so a later rollout whose
        ``physics_step`` restarted at 0 never sampled again and produced no
        third-person frames.
        """
        resolved = Path(image_dir)
        if self._video_streams:
            raise RuntimeError(
                "Finish observer video streams before starting another episode."
            )
        if self._pending_capture is not None:
            self.cancel_capture()
        if "presentation_ego" in self.view_names:
            if not policy_camera_path:
                raise ValueError(
                    "Presentation ego requires the live policy camera path."
                )
            if self._policy_camera_path != policy_camera_path:
                self._release_render_resources()
            self._policy_camera_path = policy_camera_path
        self._ensure_initialized()
        self._active_image_dir = resolved
        self._last_capture_physics_step = None
        self._scheduled_captures = 0
        self._last_paths = {}
        self._pending_capture = None
        self._pending_schedule = None
        self._render_resources_ready = False
        self._native_render_warmup_attempts = 0
        self._follow_reference_pose = None
        self._follow_last_physics_step = None
        self._pending_camera_selection = {}
        self._pending_camera_targets = {}
        self._framing_floor_z = None
        self._route_bev_pose = self._route_bev_pose_from_points(
            route_points_xy,
            resolution=self.resolution,
        )

    def accepts_captures(self, image_dir: Path) -> bool:
        """Whether ``start_episode`` has already claimed ``image_dir``.

        Scene baking renders before any episode owns an image directory, so a
        caller that renders opportunistically must ask before preparing.
        """
        return self._active_image_dir == Path(image_dir)

    def prepare_capture(
        self,
        *,
        image_dir: Path,
        physics_step: int,
        position: list[float],
        yaw_rad: float,
    ) -> bool:
        """Position the observer before a native render when sampling is due.

        ``capture_stride`` is expressed in physics ticks.  Calling this method
        does not render, step physics, or read an annotator.  The matching
        :meth:`finish_capture` call must occur only after Isaac reports that
        the native world step rendered a frame.
        """
        if len(position) < 3:
            raise ValueError(f"Room observer requires robot XYZ, got {position!r}")
        if self._pending_schedule is not None:
            raise RuntimeError(
                "Room observer cannot prepare a new frame before the previous "
                "native-render capture is finished."
            )

        resolved_image_dir = Path(image_dir)
        if self._active_image_dir != resolved_image_dir:
            raise RuntimeError(
                "Room observer must start the episode before preparing frames."
            )
        schedule = (
            self._last_capture_physics_step,
            self._scheduled_captures,
            dict(self._last_paths),
        )
        if not self._capture_due(resolved_image_dir, int(physics_step)):
            return False
        self._pending_schedule = schedule
        try:
            self._ensure_initialized()
            poses = {
                name: pose
                for name, pose in self._view_poses(position, yaw_rad).items()
                if name in self.view_names
            }
            if self._route_bev_pose is not None and "route_bev" in poses:
                poses["route_bev"] = self._route_bev_pose
            poses.update(self._fixed_view_poses)
            if self._framing_floor_z is None:
                self._framing_floor_z = (
                    float(position[2]) - G1_STANDING_HEIGHT_COMMAND_M
                )
            self._pending_camera_selection = {}
            collision_safe = self._collision_safe_poses(
                {
                    name: pose
                    for name, pose in poses.items()
                    if name != "route_bev" and name not in self._fixed_view_poses
                },
                physics_step=int(physics_step),
            )
            # Fixed calibrated eyes and route overview retain their authored pose.
            poses.update(collision_safe)
            self._position_cameras(poses)
            self._pending_camera_targets = {
                name: tuple(pose[1]) for name, pose in poses.items()
            }
            # Only a scheduled, positioned capture may arm this separate RP.
            # Keep it armed across native render and the existing zero-dt refresh.
            self._set_presentation_updates(True)
        except Exception:
            self.cancel_capture()
            raise
        self._pending_capture = (resolved_image_dir, int(physics_step))
        return True

    def finish_capture(self) -> dict[str, str]:
        """Read the frame produced by the prepared native render tick."""
        if self._pending_capture is None:
            return dict(self._last_paths)
        from PIL import Image

        image_dir, physics_step = self._pending_capture
        rgb_frames: dict[str, np.ndarray] = {}
        try:
            for view_name in self.view_names:
                rgba = self._rgb_annotators[view_name].get_data()
                rgba = np.asarray(rgba.cpu() if hasattr(rgba, "cpu") else rgba)
                if rgba.size == 0:
                    return self._handle_unready_native_render(view_name)
                if rgba.ndim == 4:
                    rgba = rgba[0]
                if rgba.ndim != 3 or rgba.shape[2] < 3:
                    self.cancel_capture()
                    raise RuntimeError(
                        f"Room observer {view_name} returned invalid RGB shape "
                        f"{rgba.shape!r}."
                    )
                rgb_frames[view_name] = rgba[:, :, :3].astype(np.uint8)
                width, height = room_observer_resolution(view_name, self.resolution)
                if rgb_frames[view_name].shape != (height, width, 3):
                    self.cancel_capture()
                    raise RuntimeError(
                        f"{view_name}: native render resolution differs from {(width, height)}."
                    )
        except BaseException:
            if self.gate_presentation_updates:
                self.cancel_capture()
            raise
        finally:
            # Native reads/copies are complete; do not pay for presentation
            # rendering during camera evidence, CPU encoding or other app updates.
            self._set_presentation_updates(False)

        camera_evidence = self._read_camera_evidence() if self._video_streams else {}
        self._pending_capture = None
        self._pending_schedule = None
        self._render_resources_ready = True
        self._native_render_warmup_attempts = 0
        paths: dict[str, str] = {}
        for view_name, rgb in rgb_frames.items():
            if self._video_streams:
                self._video_streams[view_name].capture(
                    rgb,
                    physics_step,
                    camera=camera_evidence.get(view_name),
                )
                continue
            image = Image.fromarray(rgb, mode="RGB")
            path = (
                image_dir / "room_observer" / view_name / f"step_{physics_step:06d}.png"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            paths[view_name] = str(path)
        self._last_paths = dict(paths)
        self._pending_camera_targets = {}
        if (
            self.max_frames_per_view is not None
            and self._scheduled_captures >= self.max_frames_per_view
        ):
            # The diagnostic contract is complete for this episode. Keeping
            # the unused RenderProduct attached makes RTX retain its render
            # buffers throughout long locomotion rollouts, even though no
            # further observer frame can be emitted.
            self._release_render_resources()
        return paths

    def _handle_unready_native_render(self, view_name: str) -> dict[str, str]:
        """Warm a new RenderProduct without consuming physics-time samples.

        Replicator annotators can be empty on their first native render after
        attachment.  That is resource initialization, not an episode frame.
        Once any complete frame has been observed, an empty annotator is a
        runtime failure and must remain fail-loud.
        """

        self.cancel_capture()
        if self._render_resources_ready:
            raise RuntimeError(f"Room observer {view_name} lost its native RGB stream.")
        self._native_render_warmup_attempts += 1
        if (
            self._native_render_warmup_attempts
            > ROOM_OBSERVER_NATIVE_RENDER_WARMUP_LIMIT
        ):
            raise RuntimeError(
                f"Room observer {view_name} did not produce RGB within "
                f"{ROOM_OBSERVER_NATIVE_RENDER_WARMUP_LIMIT} native renders."
            )
        return dict(self._last_paths)

    def cancel_capture(self) -> None:
        """Roll back sampling when the paired native physics step failed."""
        try:
            self._set_presentation_updates(False)
        finally:
            if self._pending_schedule is not None:
                (
                    self._last_capture_physics_step,
                    self._scheduled_captures,
                    self._last_paths,
                ) = self._pending_schedule
            self._pending_capture = None
            self._pending_schedule = None
            self._pending_camera_targets = {}

    def _set_presentation_updates(self, enabled: bool) -> None:
        """Opt-in per-product switch; never touch the policy camera or a clock."""
        if not self.gate_presentation_updates:
            return
        product = self._render_products.get("presentation_ego")
        if product is None or self._presentation_updates_enabled is enabled:
            return
        texture = getattr(product, "hydra_texture", None)
        if texture is None:
            raise RuntimeError(
                "Presentation RenderProduct has no HydraTexture update interface."
            )
        setter = getattr(texture, "set_updates_enabled", None)
        # A native setter may change state before raising. Invalidate the cache
        # first so cancellation still attempts a real disable after that failure.
        self._presentation_updates_enabled = None
        if callable(setter):
            setter(enabled)
        elif hasattr(texture, "updates_enabled"):
            texture.updates_enabled = enabled
        else:
            raise RuntimeError(
                "Presentation HydraTexture cannot gate updates on this SDK."
            )
        self._presentation_updates_enabled = enabled

    def set_presentation_update_gating(self, enabled: bool) -> None:
        """Explicit diagnostic mode switch between complete capture transactions.

        Disabling gating restores the original always-updating presentation RP.
        No product is rebuilt and no render, physics or animation tick is issued.
        """
        if self._pending_capture is not None or self._pending_schedule is not None:
            raise RuntimeError(
                "Cannot switch presentation update gating during a pending capture."
            )
        enabled = bool(enabled)
        if enabled == self.gate_presentation_updates:
            return
        if enabled:
            if "presentation_ego" not in self.view_names:
                raise ValueError(
                    "Presentation update gating requires presentation_ego."
                )
            self.gate_presentation_updates = True
            self._set_presentation_updates(False)
        else:
            self._set_presentation_updates(True)
            self.gate_presentation_updates = False

    def _capture_due(self, image_dir: Path, physics_step: int) -> bool:
        """Advance the deterministic physics-time sampling schedule."""
        if self._active_image_dir != Path(image_dir):
            raise RuntimeError(
                "Room observer sampling requires the active episode directory."
            )
        if (
            self.max_frames_per_view is not None
            and self._scheduled_captures >= self.max_frames_per_view
        ):
            return False
        due = (
            self._last_capture_physics_step is None
            or int(physics_step) - self._last_capture_physics_step
            >= self.capture_stride
        )
        if due:
            self._last_capture_physics_step = int(physics_step)
            self._scheduled_captures += 1
        return due

    def _ensure_initialized(self) -> None:
        if set(self._render_products) == set(self.view_names) and set(
            self._rgb_annotators
        ) == set(self.view_names):
            return
        if self._camera_paths or self._render_products or self._rgb_annotators:
            self._release_render_resources()
        import omni.replicator.core as rep
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Isaac stage unavailable for room observer.")
        UsdGeom.Xform.Define(stage, ROOM_OBSERVER_RECORDER_ROOT)
        UsdGeom.Xform.Define(stage, self._camera_root_path)
        try:
            for view_name in self.view_names:
                # A child camera inherits the live articulation and gaze transform
                # exactly, including motion during the upcoming physics tick.
                camera_path = (
                    f"{self._policy_camera_path}/TARSpherePresentationCamera"
                    if view_name == "presentation_ego"
                    else f"{self._camera_root_path}/{view_name}"
                )
                camera = UsdGeom.Camera.Define(stage, camera_path)
                # A wide lens keeps the full humanoid visible in narrow rooms
                # where collision-safe camera distance is necessarily short.
                if view_name == "presentation_ego":
                    self._copy_policy_camera_optics(stage, camera)
                else:
                    camera.GetFocalLengthAttr().Set(
                        18.0 if view_name == "route_bev" else OBSERVER_FOCAL_LENGTH_MM
                    )
                    if view_name != "route_bev":
                        camera.GetHorizontalApertureAttr().Set(
                            OBSERVER_HORIZONTAL_APERTURE_MM
                        )
                        camera.GetVerticalApertureAttr().Set(
                            OBSERVER_HORIZONTAL_APERTURE_MM
                            * self.resolution[1]
                            / self.resolution[0]
                        )
                        # USD's default near plane is 1 stage unit: a whole
                        # metre in these scenes, cutting door/wall surfaces
                        # despite a clear eye. Keep it inside the queried eye
                        # volume; author distance attributes in stage units.
                        unit = float(UsdGeom.GetStageMetersPerUnit(stage))
                        camera.GetClippingRangeAttr().Set(
                            (
                                OBSERVER_NEAR_CLIP_M / unit,
                                ROOM_OBSERVER_FAR_CLIP_M / unit,
                            )
                        )
                render_product = rep.create.render_product(
                    camera_path, room_observer_resolution(view_name, self.resolution)
                )
                self._camera_paths[view_name] = camera_path
                self._render_products[view_name] = render_product
                if view_name == "presentation_ego":
                    self._set_presentation_updates(False)
                annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                annotator.attach(render_product)
                self._rgb_annotators[view_name] = annotator
            self._render_resources_ready = False
            self._native_render_warmup_attempts = 0
        except Exception:
            self._release_render_resources()
            raise

    def _copy_policy_camera_optics(self, stage, destination) -> None:
        """Read policy optics; write only the presentation camera, never its parent."""
        from pxr import UsdGeom

        source = UsdGeom.Camera(stage.GetPrimAtPath(self._policy_camera_path))
        if not source:
            raise RuntimeError(
                "The policy camera is unavailable for presentation capture."
            )
        horizontal = float(source.GetHorizontalApertureAttr().Get())
        vertical = float(source.GetVerticalApertureAttr().Get())
        if vertical <= 0 or abs(horizontal / vertical - 4 / 3) > 1e-5:
            raise ValueError(
                "Presentation ego requires the unchanged 4:3 policy-camera optics."
            )
        # Camera schema attributes exclude inherited Xform operations. The child
        # uses its identity local transform and follows the parent's world pose.
        for name in UsdGeom.Camera.GetSchemaAttributeNames(includeInherited=False):
            value = source.GetPrim().GetAttribute(name).Get()
            if value is not None:
                destination.GetPrim().GetAttribute(name).Set(value)

    def _position_cameras(
        self,
        poses: dict[
            str,
            tuple[tuple[float, float, float], tuple[float, float, float]],
        ],
    ) -> None:
        """Apply all requested views before the simulator's native render."""

        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Isaac stage unavailable for room observer.")

        for view_name in self.view_names:
            if view_name == "presentation_ego":
                from pxr import UsdGeom

                self._copy_policy_camera_optics(
                    stage,
                    UsdGeom.Camera(stage.GetPrimAtPath(self._camera_paths[view_name])),
                )
                continue
            eye, target = self._fixed_view_poses.get(view_name, poses[view_name])
            if view_name == "route_bev":
                camera = stage.GetPrimAtPath(self._camera_paths[view_name])
                if not camera.IsValid():
                    raise RuntimeError(
                        "Route BEV camera disappeared before frame capture."
                    )
                from pxr import UsdGeom

                UsdGeom.Camera(camera).GetClippingRangeAttr().Set(
                    (
                        self._route_bev_near_clip_m(eye, target)
                        / UsdGeom.GetStageMetersPerUnit(stage),
                        ROOM_OBSERVER_FAR_CLIP_M / UsdGeom.GetStageMetersPerUnit(stage),
                    )
                )
            set_observer_camera_pose(stage, self._camera_paths[view_name], eye, target)

    def _read_camera_evidence(self) -> dict[str, dict]:
        """Read authored observer transforms after the paired native render.

        Manifest ``camera`` entries describe this source frame, never the
        later video sample tick. Positions and clipping distances are world
        metres; direction is a unit vector. The presentation ego inherits a
        live articulation/Fabric transform, so this USD-only measurement does
        not claim to measure its pose. Offline PNG encoding also has no pose.
        Dynamic-view selection records source-clock recovery and each tested
        candidate's first failed safety check. A safety cut is not a claim of
        continuous full-body visibility between the two rendered frames.
        """
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Observer camera evidence requires a live USD stage.")
        unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        cache = UsdGeom.XformCache()
        evidence = {}
        for name in self.view_names:
            if name == "presentation_ego":
                continue
            camera = UsdGeom.Camera(stage.GetPrimAtPath(self._camera_paths[name]))
            if not camera:
                raise RuntimeError(
                    f"Observer camera disappeared before pose capture: {name}"
                )
            world = cache.GetLocalToWorldTransform(camera.GetPrim())
            eye = np.asarray(world.ExtractTranslation(), dtype=float) * unit
            forward = np.asarray(world.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized())
            target = np.asarray(self._pending_camera_targets[name], dtype=float)
            expected = target - eye
            expected /= np.linalg.norm(expected)
            if not np.all(np.isfinite(eye)) or not np.allclose(
                forward, expected, atol=1e-6
            ):
                raise RuntimeError(
                    f"Observer {name} authored pose differs from its look-at target."
                )
            evidence[name] = {
                "pose_source": "authored_usd_camera_after_render",
                "eye_world_m": eye.tolist(),
                "target_world_m": target.tolist(),
                "forward_world_unit": forward.tolist(),
                "clipping_range_m": [
                    float(v) * unit for v in camera.GetClippingRangeAttr().Get()
                ],
            }
            if name in self._pending_camera_selection:
                evidence[name]["selection"] = self._pending_camera_selection[name]
        return evidence

    def close(self) -> None:
        try:
            self.finish_video()
        finally:
            self._release_render_resources()
        self._active_image_dir = None
        self._last_capture_physics_step = None
        self._scheduled_captures = 0
        self._last_paths = {}
        self._pending_capture = None
        self._pending_schedule = None
        self._render_resources_ready = False
        self._native_render_warmup_attempts = 0
        self._route_bev_pose = None
        self._follow_reference_pose = None
        self._follow_last_physics_step = None
        self._pending_camera_selection = {}

    def _release_render_resources(self) -> None:
        """Detach and destroy RTX resources without resetting frame history."""
        try:
            self._set_presentation_updates(False)
        finally:
            # Unsupported or failing update switches must not leak the new RP.
            for view_name, annotator in tuple(self._rgb_annotators.items()):
                render_product = self._render_products.get(view_name)
                if render_product is not None:
                    annotator.detach(render_product)
            for render_product in tuple(self._render_products.values()):
                destroy = getattr(render_product, "destroy", None)
                if callable(destroy):
                    destroy()
            self._rgb_annotators.clear()
            self._render_products.clear()
            self._camera_paths.clear()
            self._presentation_updates_enabled = None

    @staticmethod
    def _view_poses(
        position: list[float], yaw_rad: float
    ) -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
        x, y, z = (float(position[0]), float(position[1]), float(position[2]))
        forward = (math.cos(float(yaw_rad)), math.sin(float(yaw_rad)))
        left = (-forward[1], forward[0])
        # ``position`` is the G1 pelvis, not the floor. Ordinary eyes stay at
        # indoor standing height and aim at mid-body, including the feet.
        target = (x, y, z - 0.05)
        return {
            "rear_quarter": (
                (
                    x - 1.45 * forward[0] - 0.55 * left[0],
                    y - 1.45 * forward[1] - 0.55 * left[1],
                    z + 0.85,
                ),
                target,
            ),
            "side": (
                (
                    x + 2.60 * left[0],
                    y + 2.60 * left[1],
                    z + 0.95,
                ),
                target,
            ),
            "overhead": (
                (x - 1.20 * forward[0], y - 1.20 * forward[1], z + 1.20),
                target,
            ),
            # Initial world bearing; the accepted bearing persists across yaw
            # changes and is replaced only when the view becomes obstructed.
            "follow": (
                (x + 1.576, y - 2.068, z + 0.95),
                target,
            ),
            # Generic fallback for episodes without an authored route.  A
            # route-bearing episode replaces this with one fixed overview.
            "route_bev": (
                (x + 0.20, y - 0.20, z + 4.20),
                (x, y, z + 0.20),
            ),
        }

    @staticmethod
    def _route_bev_pose_from_points(
        route_points_xy: Sequence[Sequence[float]],
        *,
        resolution: tuple[int, int],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        """Frame any authored episode route without scene-specific poses."""

        if not route_points_xy:
            return None
        points = np.asarray(route_points_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError(
                f"route_bev points must be a finite Nx2 sequence; got {points.shape!r}."
            )
        lower = points.min(axis=0)
        upper = points.max(axis=0)
        center = 0.5 * (lower + upper)
        margin_m = 1.0
        width_m = max(2.0, float(upper[0] - lower[0]) + 2.0 * margin_m)
        height_m = max(2.0, float(upper[1] - lower[1]) + 2.0 * margin_m)
        aspect = float(resolution[0]) / float(resolution[1])
        framed_height_m = max(height_m, width_m / max(aspect, 1e-6))
        # With the recorder's 18 mm lens, the vertical half-span is about
        # 0.42 m per metre of eye height.  This keeps a one-metre context band
        # around arbitrary routes while remaining more zoomed than a room BEV.
        eye_height_m = max(4.0, framed_height_m / (2.0 * 0.42))
        eye = (
            float(center[0] + 0.04 * eye_height_m),
            float(center[1] - 0.04 * eye_height_m),
            float(0.35 + eye_height_m),
        )
        target = (float(center[0]), float(center[1]), 0.35)
        return eye, target

    @staticmethod
    def _route_bev_near_clip_m(
        eye: Sequence[float],
        target: Sequence[float],
    ) -> float:
        """Clip high room shells while retaining people and the robot.

        A route overview must sit above an indoor scene to frame the complete
        trajectory.  GRScene ceilings and wall caps otherwise fill that
        camera even though they are useful and remain fully visible to the
        robot.  The observer-only near plane intersects the optical axis at a
        fixed human-safe elevation; it changes neither the USD scene nor any
        policy observation.
        """

        eye_arr = np.asarray(eye, dtype=np.float64)
        target_arr = np.asarray(target, dtype=np.float64)
        if eye_arr.shape != (3,) or target_arr.shape != (3,):
            raise ValueError("route_bev eye and target must each contain XYZ.")
        view = target_arr - eye_arr
        distance_m = float(np.linalg.norm(view))
        vertical_drop_m = float(eye_arr[2] - target_arr[2])
        if distance_m <= 1e-6 or vertical_drop_m <= 1e-6:
            raise ValueError("route_bev camera must look downward.")
        clip_drop_m = float(eye_arr[2] - ROUTE_BEV_CLIP_HEIGHT_M)
        if clip_drop_m <= 0.0:
            raise ValueError("route_bev camera must be above its ceiling clip height.")
        return max(0.01, distance_m * clip_drop_m / vertical_drop_m)

    @staticmethod
    def _scene_query():
        """PhysX queries use stage units; camera framing uses metres."""
        try:
            import omni.physx
            import omni.usd
            from pxr import UsdGeom

            query = omni.physx.get_physx_scene_query_interface()
            stage = omni.usd.get_context().get_stage()
            unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        except Exception as exc:
            raise RuntimeError(
                "Observer visibility requires live PhysX scene queries."
            ) from exc
        if (
            not math.isfinite(unit)
            or unit <= 0
            or not all(
                callable(getattr(query, name, None))
                for name in ("raycast_all", "overlap_sphere_any", "sweep_sphere_any")
            )
        ):
            raise RuntimeError(
                "Observer visibility scene query is unavailable or invalid."
            )
        return query, unit

    @staticmethod
    def _scene_ray_distance(
        query, origin, end, unit: float, *, evidence: dict | None = None
    ) -> float:
        """Nearest collider in metres, excluding only the robot and recorder.

        All-hits avoids stepping through a thin wall while skipping a robot
        collider. NPCs remain occluders. Both triangle sides block the camera.
        """
        ray = np.asarray(end) - np.asarray(origin)
        length = float(np.linalg.norm(ray))
        nearest = length
        errors = []

        def report(hit):
            nonlocal nearest
            try:
                distance = float(hit.distance) * unit
                paths = f"{hit.rigid_body} {hit.collision}"
                if not math.isfinite(distance) or distance < 0 or not paths.strip():
                    raise ValueError("missing collider path or invalid distance")
                if (
                    "/robots/" not in paths
                    and f"{ROOM_OBSERVER_RECORDER_ROOT}/" not in paths
                ):
                    if distance < nearest:
                        nearest = distance
                        if evidence is not None:
                            evidence.update(
                                distance_m=distance,
                                collision=str(hit.collision),
                                rigid_body=str(hit.rigid_body),
                            )
            except Exception as exc:
                errors.append(exc)
            return True

        try:
            result = query.raycast_all(
                tuple(float(v) / unit for v in origin),
                tuple(float(v) for v in ray / length),
                length / unit,
                report,
                True,
            )
        except Exception as exc:
            raise RuntimeError("Observer visibility ray query failed.") from exc
        if errors or not isinstance(result, (bool, np.bool_)):
            raise RuntimeError(
                f"Observer visibility ray query returned invalid evidence: {errors!r}"
            )
        return nearest

    def _framing_points(self, target: np.ndarray) -> np.ndarray:
        """Sample the standing G1 envelope, retaining floor height on crouch.

        These are presentation framing points, not measured joint positions or
        a certification that every rendered body pixel is visible.
        """
        floor = self._framing_floor_z
        if floor is None:
            raise RuntimeError(
                "Observer framing requires the episode standing reference."
            )
        points = []
        for height in (floor + 0.05, float(target[2]), floor + G1_STANDING_BODY_TOP_M):
            # Include diagonals: cross-shaped samples left envelope corners
            # unchecked near doorway edges.
            for dx, dy in (
                (dx, dy) for dx in (-0.25, 0.0, 0.25) for dy in (-0.25, 0.0, 0.25)
            ):
                points.append((float(target[0]) + dx, float(target[1]) + dy, height))
        return np.asarray(points)

    def _eye_clearance_radius_m(self) -> float:
        # The overlap sphere contains the entire near-plane rectangle, not
        # just the pinhole. This does not require the distant background to
        # be empty; the 27 body-envelope rays below check subject visibility.
        horizontal_tan = OBSERVER_HORIZONTAL_APERTURE_MM / (
            2 * OBSERVER_FOCAL_LENGTH_MM
        )
        vertical_tan = horizontal_tan * self.resolution[1] / self.resolution[0]
        return max(
            OBSERVER_EYE_CLEARANCE_M,
            OBSERVER_NEAR_CLIP_M * math.sqrt(1 + horizontal_tan**2 + vertical_tan**2),
        )

    def _camera_has_clear_view(self, query, eye, target, points, unit: float) -> bool:
        return self._camera_view_rejection(query, eye, target, points, unit) is None

    def _camera_view_rejection(
        self, query, eye, target, points, unit: float
    ) -> dict | None:
        """First failed predicate, using exactly the visibility queries already made."""
        try:
            overlaps = query.overlap_sphere_any(
                self._eye_clearance_radius_m() / unit,
                tuple(float(v) / unit for v in eye),
            )
        except Exception as exc:
            raise RuntimeError("Observer camera clearance query failed.") from exc
        if not isinstance(overlaps, (bool, np.bool_)):
            raise RuntimeError(
                "Observer camera clearance query returned invalid evidence."
            )
        if overlaps:
            return {"reason": "eye_volume_overlap"}
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, (0.0, 0.0, 1.0))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        relative = points - eye
        depth = relative @ forward
        # Leave a visible border instead of accepting points at the pixel edge.
        horizontal_tan = (
            0.9 * OBSERVER_HORIZONTAL_APERTURE_MM / (2 * OBSERVER_FOCAL_LENGTH_MM)
        )
        vertical_tan = horizontal_tan * self.resolution[1] / self.resolution[0]
        outside = (
            (depth <= 0)
            | (abs(relative @ right) > depth * horizontal_tan)
            | (abs(relative @ up) > depth * vertical_tan)
        )
        if np.any(outside):
            return {
                "reason": "body_frustum",
                "point_indices": np.flatnonzero(outside).tolist(),
            }
        for index, point in enumerate(points):
            hit = {}
            distance = self._scene_ray_distance(query, point, eye, unit, evidence=hit)
            if distance < math.dist(point, eye) - 1e-5:
                return {
                    "reason": "body_occlusion",
                    "point_index": index,
                    "point_world_m": point.tolist(),
                    "hit": hit,
                }
        return None

    def _camera_move_is_clear(self, query, previous_eye, eye, unit: float) -> bool:
        """Validate a continuous eye-volume move; never smooth through a wall."""
        displacement = np.asarray(eye) - previous_eye
        distance = float(np.linalg.norm(displacement))
        if distance < 1e-8:
            return True
        try:
            blocked = query.sweep_sphere_any(
                self._eye_clearance_radius_m() / unit,
                tuple(float(v) / unit for v in previous_eye),
                tuple(float(v) for v in displacement / distance),
                distance / unit,
                True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Observer camera movement clearance query failed."
            ) from exc
        if not isinstance(blocked, (bool, np.bool_)):
            raise RuntimeError(
                "Observer camera movement query returned invalid evidence."
            )
        return not blocked

    @staticmethod
    def _radius_preserving_intermediate(
        previous_eye, endpoint, target, linear, budget_m
    ):
        """Intersect the minimum-radius circle with the physical-step budget.

        Keep the linear candidate's Z step. On the XY circle, take the largest
        feasible short-angle step toward the safe endpoint without passing its
        bearing. This supplies a candidate, never a scene-clearance assertion.
        """
        radius = OBSERVER_FOLLOW_MIN_HORIZONTAL_DISTANCE_M
        dz = float(linear[2] - previous_eye[2])
        geometry = {"radius_m": radius, "vertical_step_m": dz, "budget_m": budget_m}
        if budget_m <= 0 or abs(dz) > budget_m:
            return None, dict(
                geometry, status="rejected", reason="no_horizontal_motion_budget"
            )
        xy_budget = math.sqrt(max(0.0, budget_m**2 - dz**2))
        origin = np.asarray(previous_eye[:2]) - target[:2]
        origin_radius = float(np.linalg.norm(origin))
        geometry.update(xy_budget_m=xy_budget, previous_radius_m=origin_radius)
        if abs(radius - origin_radius) > xy_budget + 1e-10:
            return None, dict(
                geometry, status="rejected", reason="circle_outside_motion_budget"
            )
        end_vector = np.asarray(endpoint[:2]) - target[:2]
        end_angle = math.atan2(end_vector[1], end_vector[0])
        if origin_radius < 1e-10:
            # All bearings are equally distant when the old eye is at center.
            angle = end_angle
        else:
            origin_angle = math.atan2(origin[1], origin[0])
            cosine = (radius**2 + origin_radius**2 - xy_budget**2) / (
                2 * radius * origin_radius
            )
            angular_budget = math.acos(max(-1.0, min(1.0, cosine)))
            angle_to_endpoint = math.atan2(
                math.sin(end_angle - origin_angle), math.cos(end_angle - origin_angle)
            )
            angular_step = math.copysign(
                min(abs(angle_to_endpoint), angular_budget), angle_to_endpoint
            )
            angle = origin_angle + angular_step
            geometry.update(
                endpoint_angle_delta_rad=angle_to_endpoint,
                candidate_angle_delta_rad=angular_step,
            )
        candidate = np.array(
            (
                target[0] + radius * math.cos(angle),
                target[1] + radius * math.sin(angle),
                linear[2],
            )
        )
        step = float(np.linalg.norm(candidate - previous_eye))
        progress = float(
            np.linalg.norm(endpoint - previous_eye)
            - np.linalg.norm(endpoint - candidate)
        )
        geometry.update(candidate_step_m=step, endpoint_progress_m=progress)
        if step > budget_m + 1e-8:
            return None, dict(
                geometry, status="rejected", reason="candidate_exceeds_motion_budget"
            )
        if progress <= 1e-8:
            return None, dict(
                geometry, status="rejected", reason="no_endpoint_progress"
            )
        return candidate, dict(geometry, status="candidate_requires_scene_checks")

    def _collision_safe_poses(self, poses, *, physics_step: int):
        """Recover at a physical-time speed limit without weakening safety.

        Every selected intermediate eye gets the same checks as an endpoint.
        If it is unsafe, retain a still-safe old eye. A necessary safe cut is
        explicit; inability to find any verified view fails the capture.
        Selection evidence reports the first failed predicate for each tested
        candidate, not results for checks that were never executed.
        """
        if not poses:
            return {}
        if physics_step < 0:
            raise ValueError("Observer camera physics step must be nonnegative.")
        query, unit = self._scene_query()
        resolved = {}
        next_reference = self._follow_reference_pose
        next_follow_step = self._follow_last_physics_step
        for name, (eye, target) in poses.items():
            target_arr = np.asarray(target, dtype=float)
            nominal = np.asarray(eye, dtype=float) - target_arr
            points = self._framing_points(target_arr)
            reference = self._follow_reference_pose if name == "follow" else None
            previous_eye = reference[0] if reference is not None else None
            trace = {
                "physics_step": int(physics_step),
                "time_basis": "source_physics_ticks",
                "coordinates": "world_metres",
                "rejection_semantics": "first_failed_predicate_per_attempt",
                "attempts": [],
            }
            self._pending_camera_selection[name] = trace
            max_step = None
            if reference is not None:
                if self._follow_last_physics_step is None:
                    raise RuntimeError(
                        "Observer follow reference has no physical timestamp."
                    )
                delta = physics_step - self._follow_last_physics_step
                if delta < 0:
                    raise RuntimeError("Observer follow physics clock moved backward.")
                max_step = (
                    OBSERVER_FOLLOW_RECOVERY_SPEED_M_S * delta / SIMULATION_PHYSICS_HZ
                )
                trace.update(
                    previous_eye_world_m=previous_eye.tolist(),
                    anchor_target_world_m=reference[1].tolist(),
                    target_anchor_drift_m=float(
                        np.linalg.norm(target_arr - reference[1])
                    ),
                    physics_tick_delta=int(delta),
                    max_recovery_step_m=max_step,
                )

            def verified(candidate, kind):
                attempt = {
                    "kind": kind,
                    "eye_world_m": candidate.tolist(),
                    "accepted": False,
                }
                trace["attempts"].append(attempt)
                try:
                    if (
                        name == "follow"
                        and np.linalg.norm((candidate - target_arr)[:2])
                        < OBSERVER_FOLLOW_MIN_HORIZONTAL_DISTANCE_M - 1e-6
                    ):
                        rejection = {"reason": "minimum_camera_distance"}
                    elif previous_eye is not None and not self._camera_move_is_clear(
                        query, previous_eye, candidate, unit
                    ):
                        rejection = {"reason": "eye_sweep_blocked"}
                    else:
                        rejection = self._camera_view_rejection(
                            query, candidate, target_arr, points, unit
                        )
                except Exception as exc:
                    attempt["rejection"] = {"reason": "query_error", "error": str(exc)}
                    raise
                attempt["rejection"] = rejection
                attempt["accepted"] = rejection is None
                return rejection is None

            selected = None
            if name == "follow":
                preferred = (
                    nominal if reference is None else previous_eye - reference[1]
                )
                preferred_eye = target_arr + preferred
                preferred_safe = verified(preferred_eye, "preferred")
                # The old eye remains eligible regardless of accumulated drift.
                # Ordinary safe tracking does not need a second 27-ray query;
                # validate this fallback whenever recovery is actually needed.
                needs_recovery = previous_eye is not None and (
                    not preferred_safe
                    or np.linalg.norm(preferred_eye - previous_eye) > max_step + 1e-8
                )
                held_safe = needs_recovery and verified(previous_eye, "held")
                endpoint = preferred_eye if preferred_safe else None
                if endpoint is None:
                    bearing = math.atan2(preferred[1], preferred[0])
                    radius_now = float(np.linalg.norm(preferred[:2]))
                    radii = sorted(
                        {
                            2.6,
                            2.15,
                            OBSERVER_FOLLOW_MIN_HORIZONTAL_DISTANCE_M,
                            *(
                                float(radius)
                                for radius in (radius_now - 0.1, radius_now + 0.1)
                                if OBSERVER_FOLLOW_MIN_HORIZONTAL_DISTANCE_M
                                <= radius
                                <= 2.6
                            ),
                        }
                    )
                    candidates = []
                    for degrees in (
                        0,
                        10,
                        -10,
                        20,
                        -20,
                        30,
                        -30,
                        60,
                        -60,
                        90,
                        -90,
                        120,
                        -120,
                        150,
                        -150,
                        180,
                    ):
                        angle = bearing + math.radians(degrees)
                        for radius in radii:
                            candidates.append(
                                target_arr
                                + (
                                    radius * math.cos(angle),
                                    radius * math.sin(angle),
                                    nominal[2],
                                )
                            )
                    origin = previous_eye if previous_eye is not None else preferred_eye
                    candidates.sort(
                        key=lambda value: float(np.linalg.norm(value - origin))
                    )
                    for candidate in candidates:
                        if np.allclose(candidate, preferred_eye, atol=1e-8, rtol=0) or (
                            previous_eye is not None
                            and np.allclose(candidate, previous_eye, atol=1e-8, rtol=0)
                        ):
                            continue
                        if verified(candidate, "recovery_endpoint"):
                            endpoint = candidate
                            break
                if endpoint is not None:
                    trace["recovery_target_world_m"] = endpoint.tolist()
                    distance = (
                        0.0
                        if previous_eye is None
                        else float(np.linalg.norm(endpoint - previous_eye))
                    )
                    if previous_eye is None or distance <= max_step + 1e-8:
                        selected = endpoint
                        trace["branch"] = (
                            "initial_safe_view"
                            if previous_eye is None
                            else "tracked_safe_endpoint"
                        )
                    else:
                        intermediate = previous_eye + (endpoint - previous_eye) * (
                            max_step / distance
                        )
                        if verified(intermediate, "intermediate"):
                            selected = intermediate
                            trace["branch"] = "rate_limited_recovery"
                        else:
                            if (
                                trace["attempts"][-1]["rejection"]["reason"]
                                == "minimum_camera_distance"
                            ):
                                radial, geometry = self._radius_preserving_intermediate(
                                    previous_eye,
                                    endpoint,
                                    target_arr,
                                    intermediate,
                                    max_step,
                                )
                                trace["radius_recovery_geometry"] = geometry
                                if radial is not None and verified(
                                    radial, "radius_preserving_intermediate"
                                ):
                                    selected = radial
                                    trace["branch"] = "radius_preserving_recovery"
                            if selected is None:
                                if held_safe:
                                    selected = previous_eye
                                    trace["branch"] = "held_safe_eye"
                                else:
                                    selected = endpoint
                                    trace["branch"] = "safety_recovery_cut"
                                    trace["cut_reason"] = (
                                        "held_and_rate_limited_eye_unsafe"
                                    )
                elif held_safe:
                    selected = previous_eye
                    trace["branch"] = "held_safe_eye"
                if selected is not None:
                    if previous_eye is None or not np.allclose(
                        selected, previous_eye, atol=1e-8, rtol=0
                    ):
                        next_reference = (selected.copy(), target_arr.copy())
                    next_follow_step = int(physics_step)
            else:
                for scale in (1.0, 0.85, 0.7):
                    offset = nominal * scale
                    if np.linalg.norm(offset) >= 1.1 and verified(
                        target_arr + offset, "endpoint"
                    ):
                        selected = target_arr + offset
                        trace["branch"] = "safe_endpoint"
                        break
            if selected is None:
                trace["branch"] = "no_safe_candidate"
                raise RuntimeError(
                    f"Observer {name}: no clear full-body camera pose; "
                    f"inspect scene collider coverage. Selection evidence: {trace!r}"
                )
            trace["selected_eye_world_m"] = selected.tolist()
            resolved[name] = (
                tuple(float(v) for v in selected),
                tuple(float(v) for v in target_arr),
            )
        # A failure in another requested view must not advance the follow state.
        self._follow_reference_pose = next_reference
        self._follow_last_physics_step = next_follow_step
        return resolved


__all__ = [
    "ROOM_OBSERVER_RECORDER_ROOT",
    "ROOM_OBSERVER_VIEW_NAMES",
    "RoomObserverRecorder",
    "episode_route_points_xy",
]
