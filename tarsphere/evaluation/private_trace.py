"""Collect evaluator-private physical evidence from runner snapshots.

``PrivateTraceCollector`` stores post-action robot/NPC poses, manipulation,
placement and safety telemetry together with the pre-action perception used
for the decision. Camera calibration is recorded once and the episode is
finalized after execution. The runner supplies all data; the collector does
not query the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from tarsphere import geometry as G


@dataclass
class PrivateTraceStep:
    """Evaluator-private physical evidence for one decision step."""

    step: int
    # Execution is evaluated on the state after the chosen action has run.
    robot: dict[str, Any] = field(default_factory=dict)
    humans: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Perception is evaluated against the exact observation on which the
    # decision was made, before that action changes the camera or actors.
    perception_robot: dict[str, Any] = field(default_factory=dict)
    perception_humans: dict[str, dict[str, Any]] = field(default_factory=dict)
    camera: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    manipulation: dict[str, Any] = field(default_factory=dict)
    placement_release: dict[str, Any] = field(default_factory=dict)
    execution_safety: dict[str, Any] = field(default_factory=dict)
    rgb_path: str | None = None
    depth_path: str | None = None
    # Raw depth is retained only for a pre-action frame on which the model
    # actually emitted the perception contract. Post-action depth is not an
    # public Agent input. Arrays are not serialized by as_dict().
    depth: Any = None
    perception_rgb_path: str | None = None
    perception_depth_path: str | None = None
    perception_depth: Any = None
    handover: dict[str, Any] = field(default_factory=dict)
    # Physics-sampled XY path length during this complete decision, including
    # internal navigation prefixes and braking. None means unmeasured, not zero.
    travelled_distance_m: float | None = None

    def as_dict(self) -> dict[str, Any]:
        out = {
            "step": self.step,
            "robot": dict(self.robot),
            "humans": {k: dict(v) for k, v in self.humans.items()},
            "decision": dict(self.decision),
        }
        if self.rgb_path:
            out["rgb_path"] = self.rgb_path
        if self.depth_path:
            out["depth_path"] = self.depth_path
        if self.camera.get("capture_id") is not None:
            out["capture_id"] = int(self.camera["capture_id"])
        if self.camera.get("physics_step") is not None:
            out["physics_step"] = int(self.camera["physics_step"])
        if self.manipulation:
            out["manipulation"] = {
                **self.manipulation,
                "objects": {
                    str(key): dict(value)
                    for key, value in (self.manipulation.get("objects") or {}).items()
                },
            }
        if self.placement_release:
            out["placement_release"] = dict(self.placement_release)
        if self.execution_safety:
            out["execution_safety"] = dict(self.execution_safety)
        if self.handover:
            out["handover"] = dict(self.handover)
        if self.travelled_distance_m is not None:
            out["travelled_distance_m"] = self.travelled_distance_m
        return out

    def as_perception_dict(self) -> dict[str, Any]:
        """Return the pre-action actor state paired with this decision."""
        out = {
            "step": self.step,
            "robot": dict(self.perception_robot),
            "humans": {
                key: dict(value) for key, value in self.perception_humans.items()
            },
            "decision": dict(self.decision),
        }
        if self.perception_rgb_path:
            out["rgb_path"] = self.perception_rgb_path
        if self.perception_depth_path:
            out["depth_path"] = self.perception_depth_path
        if self.camera.get("capture_id") is not None:
            out["capture_id"] = int(self.camera["capture_id"])
        if self.camera.get("physics_step") is not None:
            out["physics_step"] = int(self.camera["physics_step"])
        return out


class PrivateTraceCollector:
    """Collect physical evidence without exposing it to the evaluated Agent."""

    def __init__(self) -> None:
        self.steps: list[PrivateTraceStep] = []
        self._camera_params: dict[str, Any] = {}
        self._object_volumes: dict[str, G.BBox3D] = {}
        self._scenegraph_path: str | None = None

    def record_camera_params(
        self,
        *,
        fov_deg: float,
        width: int,
        height: int,
    ) -> None:
        """Record the robot camera intrinsics source (K is derived on demand).

        Called once per episode (the camera intrinsics are constant). The
        per-step optical transform is recorded from the live USD camera and
        paired with the corresponding decision-time RGB-D frame.
        """
        self._camera_params = {
            "fov_deg": float(fov_deg),
            "width": int(width),
            "height": int(height),
        }

    def record_object_volumes(self, volumes: dict[str, G.BBox3D]) -> None:
        """Record GT object 3D volumes from the static scene graph."""
        self._object_volumes = dict(volumes)

    def record_scenegraph_path(self, path: str) -> None:
        self._scenegraph_path = path

    def record_step(
        self,
        *,
        step: int,
        observation: dict[str, Any] | None,
        perception_observation: dict[str, Any] | None,
        decision: dict[str, Any] | None,
        world_snapshot: dict[str, Any] | None,
        perception_world_snapshot: dict[str, Any] | None,
        placement_release: dict[str, Any] | None = None,
        handover: dict[str, Any] | None = None,
        travelled_distance_m: float | None = None,
    ) -> PrivateTraceStep:
        """Record one decision step and return its private evidence.

        ``observation`` is the env.observe() dict (has 'pose', 'depth', 'rgb',
        optionally 'rgbd_png'). ``decision`` is the agent's decision dict
        (has 'action' and now optionally 'envelope'). ``world_snapshot`` is
        env.get_world_snapshot() (has 'humans').
        """
        observation = observation or {}
        perception_observation = perception_observation or {}
        decision = decision or {}
        snapshot = world_snapshot or {}
        perception_snapshot = perception_world_snapshot or {}
        if travelled_distance_m is not None:
            travelled_distance_m = float(travelled_distance_m)
            if not math.isfinite(travelled_distance_m) or travelled_distance_m < 0:
                raise ValueError(
                    "Measured travelled_distance_m must be finite and nonnegative."
                )

        robot = self._robot_state(observation)
        humans = self._human_states(snapshot)
        perception_robot = self._robot_state(perception_observation)
        perception_humans = self._human_states(perception_snapshot)

        # The decision envelope (tool call + perception/plan/...).
        dec_record: dict[str, Any] = {
            "action": decision.get("action"),
            "call_name": decision.get("call_name"),
            "action_step": decision.get("action_step", step),
            "policy_step": decision.get("policy_step"),
            "policy_inference": decision.get("policy_inference"),
            "action_index": decision.get("action_index"),
            "tool_call_index": decision.get("tool_call_index"),
            "action_sequence_length": decision.get("action_sequence_length"),
            "replan_reason": decision.get("replan_reason"),
            "diagnostic_only": bool(decision.get("diagnostic_only", False)),
        }
        if isinstance(decision.get("agent_harness_config"), dict):
            dec_record["agent_harness_config"] = decision["agent_harness_config"]
        if "envelope" in decision:
            dec_record["envelope"] = decision["envelope"]
        if decision.get("message"):
            dec_record["message"] = decision["message"]

        rgb_path = observation.get("image_path")
        depth_path = observation.get("depth_path")
        # Capture the raw depth array when the env provides one (Isaac's
        # observe() puts it under 'depth'). Headless mock envs may omit it;
        # the evaluator's depth_loader then returns None and perception_3d
        # records a missing-depth error rather than crashing.
        depth_array = None
        perception_rgb_path = perception_observation.get("image_path")
        perception_depth_path = perception_observation.get("depth_path")
        envelope = decision.get("envelope")
        perception_emitted = isinstance(envelope, dict) and isinstance(
            envelope.get("perception"), dict
        )
        perception_depth_array = (
            perception_observation.get("depth") if perception_emitted else None
        )
        perception_camera = perception_observation.get("camera_params") or {}
        camera = dict(self._camera_params)
        camera["world_T_cam"] = perception_camera.get("world_T_cam")
        if perception_camera.get("capture_id") is not None:
            camera["capture_id"] = int(perception_camera["capture_id"])
        if perception_camera.get("physics_step") is not None:
            camera["physics_step"] = int(perception_camera["physics_step"])

        collected = PrivateTraceStep(
            step=step,
            robot=robot,
            humans=humans,
            perception_robot=perception_robot,
            perception_humans=perception_humans,
            camera=camera,
            decision=dec_record,
            manipulation=dict(snapshot.get("manipulation") or {}),
            placement_release=dict(placement_release or {}),
            execution_safety=dict(snapshot.get("execution_safety") or {}),
            rgb_path=rgb_path,
            depth_path=depth_path,
            depth=depth_array,
            perception_rgb_path=perception_rgb_path,
            perception_depth_path=perception_depth_path,
            perception_depth=perception_depth_array,
            handover=dict(handover or {}),
            travelled_distance_m=travelled_distance_m,
        )
        self.steps.append(collected)
        return collected

    @staticmethod
    def _robot_state(observation: dict[str, Any]) -> dict[str, Any]:
        pose = observation.get("pose") or {}
        return {
            "position": pose.get("position"),
            "yaw_rad": pose.get("yaw_rad"),
            "camera_height_m": pose.get("camera_height_m", 1.0),
            "camera_pitch_deg": pose.get("camera_pitch_deg", 0.0),
        }

    @staticmethod
    def _human_states(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
        humans: dict[str, dict[str, Any]] = {}
        for hid, hstate in (snapshot.get("humans") or {}).items():
            if not isinstance(hstate, dict):
                continue
            body_metadata = hstate.get("body_metadata") or {}
            humans[str(hid)] = {
                "position": hstate.get("position"),
                "yaw_rad": hstate.get("yaw_rad"),
                "animation_state": hstate.get("animation_state"),
                "action_status": hstate.get("action_status"),
                "active_action": hstate.get("active_action"),
                "target_position": hstate.get("target_position"),
                "robot_distance_m": hstate.get("robot_distance_m"),
                "pose_stale": hstate.get("pose_stale"),
                "motion_elapsed_s": body_metadata.get("motion_elapsed_s"),
                "motion_distance_m": body_metadata.get("motion_distance_m"),
                "endpoint_error_m": body_metadata.get("endpoint_error_m"),
            }
        return humans

    def select_perception_evidence(
        self,
        *,
        decision_steps: list[int],
        anchor_step: int,
        depth: np.ndarray | None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Select pre-action RGB-D and actor evidence for offline analysis."""
        selected_steps = set(decision_steps)
        steps_subset = [step for step in self.steps if step.step in selected_steps]
        anchor = next(
            (step for step in steps_subset if step.step == anchor_step),
            None,
        )

        camera: dict[str, Any] = {}
        if anchor is not None:
            params = anchor.camera or self._camera_params
            transform = params.get("world_T_cam")
            if params and transform is not None:
                camera = {
                    "K": G.fov_to_intrinsics(
                        params["fov_deg"],
                        params["width"],
                        params["height"],
                    ),
                    "world_T_cam": np.asarray(transform, dtype=np.float64),
                    "image_size": (params["height"], params["width"]),
                    "depth": depth,
                }

        backend: dict[str, Any] = {
            "steps": [step.as_perception_dict() for step in steps_subset],
            "camera": camera,
            "object_volumes": dict(self._object_volumes),
            "scenegraph_path": self._scenegraph_path,
        }
        if extra:
            backend.update(extra)
        return backend

    def select_physical_evidence(
        self,
        *,
        decision_steps: list[int] | None = None,
        anchor_step: int | None = None,
        depth: np.ndarray | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Select private physical evidence for certification or diagnosis.

        - ``decision_steps``: decision indices to include. Defaults to the
          complete uninterrupted rollout.
        - ``anchor_step``: the step whose camera frame should be used for
          geometric analysis. Defaults to the first selected
          step. K and world_T_cam are computed from that step's robot pose.
        - ``depth``: the depth map at the anchor step (the caller loads it
          from disk / the env). Required for bbox_to_3d_volume.
        - ``extra``: arbitrary additional backend keys (judge callable,
          certification evidence, controller verdicts, ...).
        """
        if not self.steps:
            steps_subset: list[PrivateTraceStep] = []
        elif decision_steps is not None:
            selected_steps = set(decision_steps)
            steps_subset = [s for s in self.steps if s.step in selected_steps]
        else:
            steps_subset = list(self.steps)

        anchor = (
            next((s for s in steps_subset if s.step == anchor_step), None)
            if anchor_step is not None
            else (steps_subset[0] if steps_subset else None)
        )

        # Execution/interaction/adaptation evaluation does not require a
        # camera. Keep it optional so a physical-only trace is a valid
        # backend. When present, use the same calibrated optical transform as
        # perception rather than reconstructing it from the robot root.
        camera: dict[str, Any] = {}
        if anchor is not None:
            cam_params = anchor.camera or self._camera_params
            required = {"fov_deg", "width", "height", "world_T_cam"}
            if required.issubset(cam_params) and cam_params["world_T_cam"] is not None:
                camera = {
                    "K": G.fov_to_intrinsics(
                        cam_params["fov_deg"],
                        cam_params["width"],
                        cam_params["height"],
                    ),
                    "world_T_cam": np.asarray(
                        cam_params["world_T_cam"],
                        dtype=np.float64,
                    ),
                    "image_size": (cam_params["height"], cam_params["width"]),
                    "depth": depth,
                }

        backend: dict[str, Any] = {
            "steps": [s.as_dict() for s in steps_subset],
            "manipulation_steps": [s.as_dict() for s in self.steps if s.manipulation],
            "camera": camera,
            "object_volumes": dict(self._object_volumes),
            "scenegraph_path": self._scenegraph_path,
        }
        path_length = self._path_length(steps_subset)
        if path_length is not None:
            backend["path_length_actual"] = path_length
        safety = self._selected_safety(steps_subset)
        backend.update(safety)
        if extra:
            backend.update(extra)
        return backend

    @staticmethod
    def _path_length(steps: list[PrivateTraceStep]) -> float | None:
        """Sum complete physical measurements, never endpoint chord estimates."""
        if not steps or any(step.travelled_distance_m is None for step in steps):
            return None
        return float(sum(step.travelled_distance_m for step in steps))

    def _selected_safety(
        self,
        steps: list[PrivateTraceStep],
    ) -> dict[str, Any]:
        if not steps or not all(step.execution_safety for step in steps):
            return {
                "safety_evidence_status": "unavailable",
                "safety_missing_evidence": ["selected_step_execution_safety"],
            }
        # Selected rows need not be contiguous. Subtract each row's real
        # predecessor, never bridge an omitted action and charge its contacts.
        counter_names = (
            "collision_count",
            "human_contact_count",
            "human_safety_violation_count",
        )
        totals = dict.fromkeys(counter_names, 0)
        indices = {id(step): index for index, step in enumerate(self.steps)}
        for step in steps:
            index = indices[id(step)]
            baseline = (
                self.steps[index - 1].execution_safety
                if index
                else dict.fromkeys(counter_names, 0)
            )
            if any(
                name not in sample
                for sample in (baseline, step.execution_safety)
                for name in counter_names
            ):
                return {
                    "safety_evidence_status": "unavailable",
                    "safety_missing_evidence": [
                        "selected_step_or_predecessor_counters"
                    ],
                }
            for name in counter_names:
                current, previous = step.execution_safety[name], baseline[name]
                if any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in (current, previous)
                ):
                    raise ValueError(f"{name} must be a nonnegative integer.")
                if current < previous:
                    raise ValueError(f"{name} decreased within the physical trace.")
                totals[name] += current - previous
        latest = steps[-1].execution_safety
        return {
            "safety_evidence_status": "available",
            "safety_missing_evidence": [],
            "collision_count": totals["collision_count"],
            "human_contact": totals["human_contact_count"] > 0,
            "human_safety_violation": totals["human_safety_violation_count"] > 0,
            "collision_signal_source": latest.get("source"),
        }

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def reset(self) -> None:
        self.steps.clear()
        self._camera_params = {}
        self._object_volumes = {}
        self._scenegraph_path = None


__all__ = ["PrivateTraceCollector", "PrivateTraceStep"]
