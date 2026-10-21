"""Deterministic certification policies for G1 physical execution.

These policies are not evaluated Robot Agents and never call a language model.
The grasp-only gate projects the episode-frozen target volume into the
synchronized ego frame, then exercises GraspGenX, cuRobo, Dex3 closure,
retention, lift, and return to standing. The full physical-task gate then
measures a bounded backward–turn–forward loaded-body sequence.
"""

from __future__ import annotations

from enum import Enum
import math
from typing import Any

import numpy as np
from pydantic import ValidationError

from tarsphere.bench.compilation.schema import EpisodeObject
from tarsphere.bench.spatial.physical_task.schema import (
    GraspAction,
    NavigationRoute,
)
from tarsphere.harness.agent import agent_observation_from_runtime
from tarsphere.harness.core.types import ToolCall
from tarsphere.geometry import fov_to_intrinsics, transform_to_world
from tarsphere.harness.harness import AgentHarnessConfig
from tarsphere.infra.manipulation.graspgenx import grasp_pose_payload
from tarsphere.embodiment.dex3 import G1_MANIPULATION_HAND
from tarsphere.infra.control.physical_contracts import (
    LOCOMOTION_MAX_BODY_FORWARD_M,
    LOCOMOTION_MIN_STEP_DISTANCE_M,
    LOCOMOTION_MAX_TURN_DEG,
)


GRASP_CERTIFICATION_CAMERA_TILT_DEG = -30.0
PHYSICAL_TASK_NAVIGATION_TOLERANCE_M = 0.12
PHYSICAL_TASK_HEADING_TOLERANCE_DEG = 5.0
_RETENTION_LOSS_SIGNAL = "lost live hand contact"
_CARRY_MOTION_SEQUENCE = (
    # Use the largest legal reverse request. Reverse gait starts with a weight
    # transfer and therefore proves direction here; the following turn and
    # sustained forward leg certify loaded-body mobility.
    ("backward", (-0.50, 0.0, 0.0)),
    # The released G1 gait turns more reliably while stepping than from a
    # static loaded stance.  A reverse arc also continues to clear the source
    # support instead of steering the carried object back toward it.
    ("turn", (-0.30, 0.0, -30.0)),
    ("forward", (0.75, 0.0, 0.0)),
)
_CARRY_MINIMUM_TRANSLATION_M = 0.10
_CARRY_MINIMUM_TURN_DEG = 15.0
_CARRY_MAXIMUM_ATTEMPTS_PER_MANOEUVRE = 3


def target_rgb_material_evidence(
    rgb: Any,
    target_mask: np.ndarray,
) -> dict[str, Any]:
    """Measure the rendered target pixels selected by synchronized geometry."""

    frame = np.asarray(rgb)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise RuntimeError(
            f"Asset material evidence requires HxWx3 RGB; got {frame.shape!r}."
        )
    if frame.shape[:2] != target_mask.shape:
        raise RuntimeError(
            "Asset RGB and synchronized target mask disagree: "
            f"rgb={frame.shape[:2]} mask={target_mask.shape}."
        )
    pixels = np.asarray(frame[..., :3], dtype=np.float32)[target_mask]
    if pixels.size == 0 or not np.all(np.isfinite(pixels)):
        raise RuntimeError("Asset material evidence has no finite target pixels.")
    # Normalized floating-point RGB is accepted, but evidence is always stored
    # in the renderer's canonical 8-bit display range.
    if float(np.max(pixels)) <= 1.5:
        pixels = pixels * 255.0
    pixels = np.clip(pixels, 0.0, 255.0)
    luminance = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
    percentiles = np.quantile(luminance, [0.1, 0.5, 0.9])
    magenta_fallback = (
        (pixels[:, 0] >= 180.0) & (pixels[:, 1] <= 100.0) & (pixels[:, 2] >= 180.0)
    )
    return {
        "target_pixel_count": int(pixels.shape[0]),
        "luminance_mean": round(float(np.mean(luminance)), 3),
        "luminance_p10": round(float(percentiles[0]), 3),
        "luminance_p50": round(float(percentiles[1]), 3),
        "luminance_p90": round(float(percentiles[2]), 3),
        "near_black_ratio": round(float(np.mean(luminance <= 5.0)), 4),
        "near_white_ratio": round(float(np.mean(luminance >= 250.0)), 4),
        "any_channel_clipped_ratio": round(
            float(np.mean(np.any(pixels >= 255.0, axis=1))), 4
        ),
        "magenta_fallback_ratio": round(float(np.mean(magenta_fallback)), 4),
        "rgb_channel_mean": [
            round(float(value), 3) for value in np.mean(pixels, axis=0)
        ],
        "measurement": "synchronized_depth_world_volume_target_mask",
    }


class PhysicalCertificationFailure(RuntimeError):
    """A measured physical execution failed, not the certification service."""


class PhysicalTaskStage(str, Enum):
    """Observable stage of deterministic physical-task certification."""

    NAVIGATE_TO_TARGET = "navigate_to_target"
    GRASP = "grasp"
    TRANSPORT = "transport"
    REPORT = "report"
    COMPLETE = "complete"


class GraspSkillCertificationAgent:
    """Run the known-positive tilt–grasp–report certification sequence."""

    name = "grasp_skill_certification"
    certification_only = True

    def __init__(
        self,
        *,
        agent_harness_config: AgentHarnessConfig,
        tools: Any,
        grasp_backend: Any,
    ) -> None:
        self.agent_harness_config = agent_harness_config
        self.tools = tools
        self.grasp_backend = grasp_backend
        self.hand = G1_MANIPULATION_HAND
        self._target_name = ""
        self._target_center_world: tuple[float, float, float] | None = None
        self._target_bounds_world: (
            tuple[tuple[float, float, float], tuple[float, float, float]] | None
        ) = None
        self._action_step = 0
        self._next_action = "tilt_camera"
        self._last_action = ""
        self._latest_feedback: dict[str, Any] = {}

    @property
    def execution_stats(self) -> dict[str, Any]:
        """Expose that the certificate used actions but zero policy calls."""

        return {
            "action_steps": self._action_step,
            "policy_steps": 0,
            "pending_actions": 0,
            "discarded_actions": 0,
            "replan_counts": {},
        }

    def configure_episode(self, episode: Any) -> None:
        """Bind the single episode-frozen pickable and its authored geometry."""

        action_spec = getattr(episode, "agent_action_space", None)
        declared_actions = {
            str(action)
            for family in (
                "cognitive",
                "communicative",
                "physical",
            )
            for action in (
                getattr(action_spec, family, ()) if action_spec is not None else ()
            )
        }
        required_actions = {"tilt_camera", "grasp_object", "report_done"}
        missing_from_episode = sorted(required_actions - declared_actions)
        if missing_from_episode:
            raise RuntimeError(
                "Grasp-skill certification episode is missing actions: "
                f"{missing_from_episode}."
            )
        missing_tools = sorted(
            {"tilt_camera", "report_done"} - set(self.tools.spec_names())
        )
        if missing_tools:
            raise RuntimeError(
                f"Grasp-skill certification tools are not wired: {missing_tools}."
            )

        world = getattr(episode, "world_knowledge", None) or {}
        objects = world.get("objects") if isinstance(world, dict) else None
        pickables = [
            item
            for item in (objects or [])
            if isinstance(item, dict)
            and str(item.get("interaction") or "") == "pickable"
        ]
        if len(pickables) != 1:
            raise RuntimeError(
                "Grasp-skill certification requires exactly one authored "
                f"pickable; found {len(pickables)}."
            )
        self._target_name = str(pickables[0].get("name") or "").strip()
        if not self._target_name:
            raise RuntimeError(
                "Grasp-skill certification pickable requires a visual name."
            )
        try:
            self._target_center_world = tuple(
                float(value) for value in pickables[0]["position_xyz"]
            )
            self._target_bounds_world = (
                tuple(float(value) for value in pickables[0]["bounds_min_xyz"]),
                tuple(float(value) for value in pickables[0]["bounds_max_xyz"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Grasp-skill certification pickable requires its frozen world "
                "centre and ordered 3-D bounds."
            ) from exc
        if (
            len(self._target_center_world) != 3
            or any(len(bound) != 3 for bound in self._target_bounds_world)
            or any(
                self._target_bounds_world[0][axis] > self._target_bounds_world[1][axis]
                for axis in range(3)
            )
        ):
            raise RuntimeError(
                "Grasp-skill certification target geometry must be ordered 3-D "
                "world coordinates."
            )

    def configure_episode_humans(self, humans: list[Any]) -> None:
        """Reject Humans in the Human-free grasp certification protocol."""

        if humans:
            raise ValueError("Grasp-skill certification does not admit Humans.")

    def reset(self) -> None:
        self._action_step = 0
        self._next_action = "tilt_camera"
        self._last_action = ""
        self._latest_feedback = {}

    def act(
        self,
        observation: dict[str, Any],
        instruction: str,
        action_space: list[str] | None = None,
        *,
        decision_requirements: Any = None,
    ) -> dict[str, Any]:
        """Issue one deterministic public tool call from live sensor input."""

        del instruction, decision_requirements
        if self._next_action == "report_done":
            grasp = self._latest_feedback.get("grasp_skill") or {}
            if not bool(grasp.get("transport_ready", False)):
                raise RuntimeError(
                    "Grasp certification cannot report completion: the shared "
                    "skill did not return transport_ready=true. "
                    f"feedback={self._latest_feedback!r}"
                )

        arguments = {
            "tilt_camera": {
                "pitch_deg": GRASP_CERTIFICATION_CAMERA_TILT_DEG,
            },
            "grasp_object": {
                "target": self._target_name,
            },
            "report_done": {},
        }[self._next_action]
        if action_space is not None and self._next_action not in action_space:
            raise RuntimeError(
                f"Certification action {self._next_action!r} is unavailable."
            )
        agent_observation = agent_observation_from_runtime(
            observation,
            camera_views=self.agent_harness_config.protocol.camera_views,
        )
        call = ToolCall(
            tool=self._next_action,
            args=arguments,
            call_id=f"grasp-certification-{self._action_step}",
        )
        result = (
            self._coordinate_grounded_grasp(call, agent_observation)
            if call.tool == "grasp_object"
            else self.tools.execute(call, {"observation": agent_observation})
        )
        if result.is_error:
            raise RuntimeError(f"Deterministic {call.tool} failed: {result.content}")

        action = dict(result.action)
        action.update(
            {
                "action_step": self._action_step,
                "policy_step": None,
                "policy_inference": False,
                "replan_reason": "deterministic_physical_certification",
                "discarded_actions": 0,
                "action_index": 0,
                "action_sequence_length": 1,
                "call_name": call.tool,
                "reasoning": (
                    "Execute the frozen physical certification sequence; no "
                    "language-model policy is active."
                ),
                "tool_results": [
                    {
                        "name": result.name,
                        "content": result.content,
                        "is_error": False,
                        "metadata": dict(result.metadata),
                    }
                ],
                "agent_harness_config": self.agent_harness_config.as_dict(),
                "diagnostic_only": True,
            }
        )
        self._last_action = call.tool
        self._next_action = {
            "tilt_camera": "grasp_object",
            "grasp_object": "report_done",
            "report_done": "report_done",
        }[call.tool]
        self._action_step += 1
        return action

    def update_info(self, info: dict[str, Any]) -> None:
        """Retain public execution evidence for the next certification gate."""

        self._latest_feedback = dict(info)
        if self._last_action == "grasp_object" and not bool(
            self._latest_feedback.get("executed", False)
        ):
            raise PhysicalCertificationFailure(
                "Deterministic grasp execution failed: "
                f"{self._latest_feedback.get('feedback') or 'no feedback'}"
            )

    def _coordinate_grounded_grasp(
        self,
        call: ToolCall,
        observation: Any,
    ) -> Any:
        """Run GraspGenX on depth points inside the frozen target volume."""

        from tarsphere.harness.core.types import ToolResult

        if self._target_center_world is None or self._target_bounds_world is None:
            raise RuntimeError("Certification target geometry is not configured.")
        ego = observation.views.get("ego")
        if ego is None or ego.depth is None:
            raise RuntimeError(
                "Coordinate-grounded grasp certification requires synchronized "
                "ego depth."
            )
        depth = np.squeeze(np.asarray(ego.depth, dtype=np.float32))
        if depth.ndim != 2:
            raise RuntimeError(f"Certification depth must be HxW; got {depth.shape!r}.")
        camera = dict(ego.camera_params or {})
        try:
            intrinsics = fov_to_intrinsics(
                float(camera["fov_deg"]),
                int(camera.get("width") or depth.shape[1]),
                int(camera.get("height") or depth.shape[0]),
            )
            world_T_camera = np.asarray(
                camera["world_T_cam"],
                dtype=np.float64,
            )
            # The runtime pairs the planning root with the capture that
            # produced these extrinsics, so both must come from one frame.
            capture_id = int(camera["capture_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Certification camera calibration is incomplete: {exc}"
            ) from exc
        if world_T_camera.shape != (4, 4) or not np.all(np.isfinite(world_T_camera)):
            raise RuntimeError(
                "Certification world_T_cam must be a finite 4x4 transform."
            )

        lower = np.asarray(self._target_bounds_world[0], dtype=np.float64)
        upper = np.asarray(self._target_bounds_world[1], dtype=np.float64)
        corners_world = np.asarray(
            [
                [x, y, z]
                for x in (lower[0], upper[0])
                for y in (lower[1], upper[1])
                for z in (lower[2], upper[2])
            ],
            dtype=np.float64,
        )
        camera_T_world = np.linalg.inv(world_T_camera)
        corners_camera = (
            np.concatenate(
                [corners_world, np.ones((8, 1), dtype=np.float64)],
                axis=1,
            )
            @ camera_T_world.T
        )[:, :3]
        if np.any(corners_camera[:, 2] <= 0.0):
            raise RuntimeError(
                "The frozen target volume is not fully in front of the ego "
                "camera; coordinate-grounded certification cannot form a "
                "single target projection."
            )
        projected = corners_camera @ intrinsics.T
        pixels = projected[:, :2] / projected[:, 2:3]
        height, width = depth.shape
        x1 = max(0, int(np.floor(np.min(pixels[:, 0]))))
        y1 = max(0, int(np.floor(np.min(pixels[:, 1]))))
        x2 = min(width, int(np.ceil(np.max(pixels[:, 0]))) + 1)
        y2 = min(height, int(np.ceil(np.max(pixels[:, 1]))) + 1)
        if x2 <= x1 or y2 <= y1:
            raise RuntimeError(
                "The frozen target volume projects outside the current ego "
                f"frame: bbox={[x1, y1, x2, y2]} frame={width}x{height}."
            )
        region_depth = depth[y1:y2, x1:x2]
        region_valid = np.isfinite(region_depth) & (region_depth > 0.0)
        region_rows, region_columns = np.nonzero(region_valid)
        region_ranges = region_depth[region_valid].astype(
            np.float64,
            copy=False,
        )
        image_rows = region_rows + y1
        image_columns = region_columns + x1
        image_pixels = np.stack(
            [
                image_columns,
                image_rows,
                np.ones_like(image_columns),
            ],
            axis=1,
        ).astype(np.float64, copy=False)
        points_camera = (image_pixels @ np.linalg.inv(intrinsics).T) * region_ranges[
            :, None
        ]
        points_world = transform_to_world(points_camera, world_T_camera)
        volume_supported = np.all(
            (points_world >= lower) & (points_world <= upper),
            axis=1,
        )
        target_mask = np.zeros(depth.shape, dtype=bool)
        target_mask[image_rows[volume_supported], image_columns[volume_supported]] = (
            True
        )
        support_count = int(np.count_nonzero(target_mask))
        if support_count < 100:
            raise RuntimeError(
                "The frozen target volume has insufficient synchronized "
                "depth support for GraspGenX: "
                f"{support_count} pixels in bbox={[x1, y1, x2, y2]}."
            )
        material_evidence = target_rgb_material_evidence(ego.rgb, target_mask)
        candidates = tuple(
            self.grasp_backend.propose(
                depth_m=depth,
                intrinsics=intrinsics,
                target_mask=target_mask,
            )
        )
        if not candidates:
            raise RuntimeError(
                "GraspGenX returned no pose for the coordinate-grounded target."
            )
        target_center_camera = np.median(
            points_camera[volume_supported],
            axis=0,
        )
        action = {
            "action": "grasp_object",
            "hand": self.hand,
            "preset": "grasp",
            "target": self._target_name,
            "controller_metadata": {
                "agent_grasp": {
                    "target": self._target_name,
                    "target_center_world_m": list(self._target_center_world),
                    "target_center_camera_m": [
                        float(value) for value in target_center_camera
                    ],
                    "bbox_2d": [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ],
                    "semantic_score": 1.0,
                    "world_T_camera": world_T_camera.tolist(),
                    "capture_id": capture_id,
                    "grasp_pose_candidates": [
                        grasp_pose_payload(candidate) for candidate in candidates
                    ],
                },
            },
        }
        return ToolResult(
            call_id=call.call_id,
            name=call.tool,
            action=action,
            content=(
                f"Coordinate-grounded GraspGenX produced {len(candidates)} "
                f"poses from {support_count} synchronized target pixels."
            ),
            metadata={
                "target": self._target_name,
                "hand": self.hand,
                "target_center_world_m": list(self._target_center_world),
                "target_depth_pixels": support_count,
                "projected_target_bbox_2d": [x1, y1, x2, y2],
                "graspgenx_pose_count": len(candidates),
                "grounding_mode": "certification_world_volume",
                "target_rgb_material_evidence": material_evidence,
            },
        )


class PhysicalTaskCertificationAgent(GraspSkillCertificationAgent):
    """Certify navigation, grasp, and a short loaded body manoeuvre.

    This benchmark-private policy receives an already grounded physical task
    and its certified approach route. It never plans from simulator occupancy
    and never bypasses a public action. After grasping, one backward, turning,
    and forward command test loaded-body stability without invoking the general
    navigation harness. The grasp call retains the exact
    synchronized coordinate chain implemented by ``GraspSkillCertificationAgent``.
    """

    name = "physical_task_certification"

    def __init__(
        self,
        *,
        agent_harness_config: AgentHarnessConfig,
        tools: Any,
        grasp_backend: Any,
    ) -> None:
        super().__init__(
            agent_harness_config=agent_harness_config,
            tools=tools,
            grasp_backend=grasp_backend,
        )
        self.stage = PhysicalTaskStage.NAVIGATE_TO_TARGET
        self._route: tuple[tuple[float, float], ...] = ()
        self._route_index = 0
        self._target_yaw_rad = 0.0
        self._carry_motion_index = 0
        self._target_object_id = ""
        self._grasp_tilted = False
        self._holding_hand: str | None = None
        self._carry_progress = 0.0
        self._carry_motion_attempts = 0
        self._configured = False

    def certification_status(self) -> dict[str, Any]:
        """Expose the exact deterministic state reached by this trial."""

        return {
            "stage": self.stage.value,
            "route_index": self._route_index,
            "carry_motion_index": self._carry_motion_index,
            "carry_motion_attempts": self._carry_motion_attempts,
            "holding_hand": self._holding_hand,
        }

    def configure_episode(self, episode: Any) -> None:
        """Bind one certified approach and loaded-grasp stability trial."""

        self._configured = False
        declared_actions = self._declared_actions(episode)
        required_actions = {
            "move_body",
            "tilt_camera",
            "grasp_object",
            "report_done",
        }
        missing_actions = sorted(required_actions - declared_actions)
        if missing_actions:
            raise RuntimeError(
                "Physical-task certification episode is missing actions: "
                f"{missing_actions}."
            )
        required_tools = required_actions - {"grasp_object"}
        missing_tools = sorted(required_tools - set(self.tools.spec_names()))
        if missing_tools:
            raise RuntimeError(
                f"Physical-task certification tools are not wired: {missing_tools}."
            )

        world = getattr(episode, "world_knowledge", None)
        if not isinstance(world, dict):
            raise RuntimeError("Physical-task certification requires world_knowledge.")
        try:
            physical_task = world.get("physical_task")
            if not isinstance(physical_task, dict):
                raise TypeError("physical_task must be a mapping")
            target_object_id = str(physical_task.get("target_object_id") or "").strip()
            if not target_object_id:
                raise ValueError("physical_task.target_object_id is required")
            grasp = GraspAction.model_validate(physical_task.get("grasp"))
            navigate_to_target = NavigationRoute.model_validate(
                world.get("navigate_to_target")
            )
            objects = tuple(
                EpisodeObject.model_validate(item)
                for item in world.get("objects") or ()
            )
        except ValidationError as exc:
            raise RuntimeError(
                "Physical-task certification requires a valid compiled task, "
                f"episode route, and object set: {exc}"
            ) from exc
        self._target_object_id = target_object_id
        target = next(
            (item for item in objects if item.object_id == self._target_object_id),
            None,
        )
        if target is None:
            raise RuntimeError(
                "Physical task target is absent from world_knowledge.objects: "
                f"{self._target_object_id!r}."
            )
        self._configure_target_geometry(target)

        humans = tuple(getattr(episode, "humans", ()) or ())
        if humans:
            raise RuntimeError(
                "Nav–grasp–nav certification is Human-free; "
                f"found {len(humans)} Human definitions."
            )

        self._route = self._without_consecutive_duplicates(
            (
                *(point.position_xy for point in navigate_to_target.points),
                grasp.robot_stance_xy,
            )
        )
        self._target_yaw_rad = grasp.robot_yaw_rad
        self._configured = True
        self.reset()

    def reset(self) -> None:
        """Return the deterministic program to its first route leg."""

        self._action_step = 0
        self._last_action = ""
        self._latest_feedback = {}
        self.stage = PhysicalTaskStage.NAVIGATE_TO_TARGET
        self._route_index = 0
        self._carry_motion_index = 0
        self._carry_progress = 0.0
        self._carry_motion_attempts = 0
        self._grasp_tilted = False
        self._holding_hand = None

    def act(
        self,
        observation: dict[str, Any],
        instruction: str,
        action_space: list[str] | None = None,
        *,
        decision_requirements: Any = None,
    ) -> dict[str, Any]:
        """Issue the next public action from the certified task program."""

        del instruction, decision_requirements
        if not self._configured:
            raise RuntimeError("Physical-task certification episode is not configured.")
        if self.stage is PhysicalTaskStage.COMPLETE:
            raise RuntimeError(
                "Physical-task certification already reported completion."
            )
        agent_observation = agent_observation_from_runtime(
            observation,
            camera_views=self.agent_harness_config.protocol.camera_views,
        )
        call, controller_metadata = self._next_call(observation)
        if action_space is not None and call.tool not in action_space:
            raise RuntimeError(f"Certification action {call.tool!r} is unavailable.")
        result = (
            self._coordinate_grounded_grasp(call, agent_observation)
            if call.tool == "grasp_object"
            else self.tools.execute(call, {"observation": agent_observation})
        )
        if result.is_error:
            raise RuntimeError(f"Deterministic {call.tool} failed: {result.content}")
        action = dict(result.action)
        if controller_metadata:
            existing_metadata = action.get("controller_metadata")
            if existing_metadata is not None and not isinstance(
                existing_metadata, dict
            ):
                raise TypeError("Action controller_metadata must be a mapping.")
            action["controller_metadata"] = {
                **dict(existing_metadata or {}),
                **controller_metadata,
            }
        action.update(
            {
                "action_step": self._action_step,
                "policy_step": None,
                "policy_inference": False,
                "replan_reason": "deterministic_physical_task_certification",
                "discarded_actions": 0,
                "action_index": 0,
                "action_sequence_length": 1,
                "call_name": call.tool,
                "diagnostic": (
                    "Execute the certified physical-task program through the "
                    "public action boundary; no language-model policy is active."
                ),
                "tool_results": [
                    {
                        "name": result.name,
                        "content": result.content,
                        "is_error": False,
                        "metadata": dict(result.metadata),
                    }
                ],
                "agent_harness_config": self.agent_harness_config.as_dict(),
                "diagnostic_only": True,
            }
        )
        self._last_action = call.tool
        self._action_step += 1
        if call.tool == "report_done":
            self.stage = PhysicalTaskStage.COMPLETE
        return action

    def update_info(self, info: dict[str, Any]) -> None:
        """Advance only after the preceding physical transition succeeds."""

        self._latest_feedback = dict(info)
        if not bool(info.get("executed", False)):
            raise PhysicalCertificationFailure(
                f"Deterministic {self._last_action} execution failed: "
                f"{info.get('feedback') or 'no feedback'}"
            )
        if self.stage in {
            PhysicalTaskStage.TRANSPORT,
            PhysicalTaskStage.REPORT,
        }:
            self._require_retained_object(info)
        if self._last_action == "tilt_camera":
            self._grasp_tilted = True
        elif self._last_action == "grasp_object":
            grasp = info.get("grasp_skill")
            if not isinstance(grasp, dict) or not bool(
                grasp.get("transport_ready", False)
            ):
                raise RuntimeError(
                    "Physical-task grasp did not establish "
                    f"transport_ready=true: {info!r}"
                )
            holding_hand = str(grasp.get("hand") or "").strip().lower()
            if holding_hand != G1_MANIPULATION_HAND:
                raise RuntimeError(
                    "Physical-task grasp did not report the certified right "
                    f"hand: actual={holding_hand!r}."
                )
            self._holding_hand = holding_hand
            self.stage = PhysicalTaskStage.TRANSPORT
            self._carry_motion_index = 0
            self._carry_motion_attempts = 0
        elif (
            self.stage is PhysicalTaskStage.TRANSPORT
            and self._last_action == "move_body"
        ):
            progress, required = self._measured_carry_progress(info)
            self._carry_motion_attempts += 1
            self._carry_progress += progress
            if self._carry_progress >= required:
                self._carry_motion_index += 1
                self._carry_progress = 0.0
                self._carry_motion_attempts = 0
            elif self._carry_motion_attempts >= _CARRY_MAXIMUM_ATTEMPTS_PER_MANOEUVRE:
                label = _CARRY_MOTION_SEQUENCE[self._carry_motion_index][0]
                raise RuntimeError(
                    f"Carry {label} made only {self._carry_progress:.3f} "
                    f"of the required {required:.3f} progress after "
                    f"{self._carry_motion_attempts} attempts."
                )

    def _next_call(
        self,
        observation: dict[str, Any],
    ) -> tuple[ToolCall, dict[str, Any] | None]:
        while True:
            if self.stage is PhysicalTaskStage.NAVIGATE_TO_TARGET:
                motion = self._route_call(
                    observation,
                    route=self._route,
                    terminal_yaw_rad=self._target_yaw_rad,
                )
                if motion is not None:
                    return motion, None
                self.stage = PhysicalTaskStage.GRASP
                self._route_index = 0
                continue
            if self.stage is PhysicalTaskStage.GRASP:
                call_name = "grasp_object" if self._grasp_tilted else "tilt_camera"
                arguments = (
                    {"target": self._target_name}
                    if call_name == "grasp_object"
                    else {"pitch_deg": GRASP_CERTIFICATION_CAMERA_TILT_DEG}
                )
                return self._call(call_name, arguments), None
            if self.stage is PhysicalTaskStage.TRANSPORT:
                self._require_retained_object(self._latest_feedback)
                if self._carry_motion_index < len(_CARRY_MOTION_SEQUENCE):
                    _, delta = _CARRY_MOTION_SEQUENCE[self._carry_motion_index]
                    return self._call("move_body", {"delta": list(delta)}), None
                self.stage = PhysicalTaskStage.REPORT
                self._route_index = 0
                continue
            if self.stage is PhysicalTaskStage.REPORT:
                return self._call("report_done", {}), None
            raise RuntimeError(f"Unsupported physical-task stage: {self.stage.value}.")

    def _route_call(
        self,
        observation: dict[str, Any],
        *,
        route: tuple[tuple[float, float], ...],
        terminal_yaw_rad: float,
    ) -> ToolCall | None:
        position, yaw_rad = self._robot_pose(observation)
        while self._route_index < len(route):
            goal = route[self._route_index]
            dx = goal[0] - position[0]
            dy = goal[1] - position[1]
            distance = math.hypot(dx, dy)
            if distance <= PHYSICAL_TASK_NAVIGATION_TOLERANCE_M:
                self._route_index += 1
                continue
            desired_yaw = math.atan2(dy, dx)
            heading_error = math.atan2(
                math.sin(desired_yaw - yaw_rad),
                math.cos(desired_yaw - yaw_rad),
            )
            heading_error_deg = math.degrees(heading_error)
            if abs(heading_error_deg) > PHYSICAL_TASK_HEADING_TOLERANCE_DEG:
                turn_deg = max(
                    -LOCOMOTION_MAX_TURN_DEG,
                    min(LOCOMOTION_MAX_TURN_DEG, heading_error_deg),
                )
                return self._call("move_body", {"delta": [0.0, 0.0, turn_deg]})
            forward_m = min(distance, LOCOMOTION_MAX_BODY_FORWARD_M)
            if forward_m < LOCOMOTION_MIN_STEP_DISTANCE_M:
                raise RuntimeError(
                    "Certified route residual lies outside its completion "
                    "tolerance but below the public locomotion minimum: "
                    f"distance_m={distance:.4f}."
                )
            return self._call("move_body", {"delta": [forward_m, 0.0, 0.0]})

        yaw_error = math.atan2(
            math.sin(terminal_yaw_rad - yaw_rad),
            math.cos(terminal_yaw_rad - yaw_rad),
        )
        yaw_error_deg = math.degrees(yaw_error)
        if abs(yaw_error_deg) <= PHYSICAL_TASK_HEADING_TOLERANCE_DEG:
            return None
        turn_deg = max(
            -LOCOMOTION_MAX_TURN_DEG,
            min(LOCOMOTION_MAX_TURN_DEG, yaw_error_deg),
        )
        return self._call("move_body", {"delta": [0.0, 0.0, turn_deg]})

    def _call(self, name: str, arguments: dict[str, Any]) -> ToolCall:
        return ToolCall(
            tool=name,
            args=arguments,
            call_id=f"physical-task-certification-{self._action_step}",
        )

    def _configure_target_geometry(self, target: EpisodeObject) -> None:
        self._target_name = str(target.name or "").strip()
        if not self._target_name:
            raise RuntimeError("Physical-task target requires a public visual name.")
        if target.interaction != "pickable":
            raise RuntimeError("Physical-task target must be declared pickable.")
        self._target_center_world = target.position_xyz
        self._target_bounds_world = (
            target.bounds_min_xyz,
            target.bounds_max_xyz,
        )
        if any(
            self._target_bounds_world[0][axis] > self._target_bounds_world[1][axis]
            for axis in range(3)
        ):
            raise RuntimeError("Physical-task target bounds must be ordered.")

    @staticmethod
    def _declared_actions(episode: Any) -> set[str]:
        action_spec = getattr(episode, "agent_action_space", None)
        return {
            str(action)
            for family in (
                "cognitive",
                "communicative",
                "physical",
            )
            for action in (
                getattr(action_spec, family, ()) if action_spec is not None else ()
            )
        }

    @staticmethod
    def _without_consecutive_duplicates(
        points: tuple[tuple[float, float], ...],
    ) -> tuple[tuple[float, float], ...]:
        unique: list[tuple[float, float]] = []
        for point in points:
            if not unique or math.dist(unique[-1], point) > 1e-9:
                unique.append(point)
        return tuple(unique)

    @staticmethod
    def _robot_pose(
        observation: dict[str, Any],
    ) -> tuple[tuple[float, float], float]:
        pose = observation.get("pose")
        if not isinstance(pose, dict):
            raise RuntimeError(
                "Physical-task certification requires the observed robot pose."
            )
        position = pose.get("position")
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            raise RuntimeError(
                "Physical-task certification robot position requires x and y."
            )
        values = (float(position[0]), float(position[1]))
        yaw_rad = float(pose.get("yaw_rad"))
        if not all(math.isfinite(value) for value in (*values, yaw_rad)):
            raise RuntimeError("Physical-task certification robot pose must be finite.")
        return values, yaw_rad

    @staticmethod
    def _require_retained_object(info: dict[str, Any]) -> None:
        feedback = str(info.get("feedback") or "")
        if _RETENTION_LOSS_SIGNAL in feedback:
            raise RuntimeError(
                f"Physical-task transport lost object retention: {feedback}"
            )

    def _measured_carry_progress(
        self,
        info: dict[str, Any],
    ) -> tuple[float, float]:
        """Measure signed progress for the active loaded-body manoeuvre.

        One public velocity window is not guaranteed to complete a humanoid
        gait cycle.  The certificate therefore repeats the same legal action
        and accumulates measured progress; it never substitutes commanded
        distance for physical displacement.
        """

        label, requested = _CARRY_MOTION_SEQUENCE[self._carry_motion_index]
        outcome = info.get("motion_outcome")
        if not isinstance(outcome, dict) or not bool(outcome.get("available")):
            raise RuntimeError(
                f"Carry {label} requires a measured body-motion outcome."
            )
        achieved = outcome.get("achieved_body_delta")
        if not isinstance(achieved, dict):
            raise RuntimeError(f"Carry {label} outcome lacks achieved_body_delta.")
        if label == "turn":
            progress = math.copysign(1.0, requested[2]) * float(
                achieved.get("turn_deg", 0.0)
            )
            required = _CARRY_MINIMUM_TURN_DEG
            unit = "deg"
        else:
            request_xy = np.asarray(requested[:2], dtype=np.float64)
            achieved_xy = np.asarray(
                [
                    float(achieved.get("forward_m", 0.0)),
                    float(achieved.get("left_m", 0.0)),
                ]
            )
            progress = float(
                np.dot(request_xy, achieved_xy) / np.linalg.norm(request_xy)
            )
            required = _CARRY_MINIMUM_TRANSLATION_M
            unit = "m"
        if not math.isfinite(progress):
            raise RuntimeError(
                f"Carry {label} produced non-finite measured progress in {unit}."
            )
        return progress, required


class GraspAssetCertificationAgent(PhysicalTaskCertificationAgent):
    """Certify one isolated asset through grasp, carry, and RGB material gates."""

    name = "grasp_asset_certification"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._target_material_evidence: dict[str, Any] = {}

    @property
    def execution_stats(self) -> dict[str, Any]:
        stats = dict(super().execution_stats)
        stats["asset_certification"] = self.certification_status()
        return stats

    def certification_status(self) -> dict[str, Any]:
        status = dict(super().certification_status())
        status.update(
            {
                "protocol": "isolated_grasp_backward_turn_forward_v1",
                "carry_motion_count": len(_CARRY_MOTION_SEQUENCE),
                "carry_complete": (
                    self._carry_motion_index == len(_CARRY_MOTION_SEQUENCE)
                    and self.stage is PhysicalTaskStage.COMPLETE
                ),
                "target_rgb_material_evidence": dict(self._target_material_evidence),
            }
        )
        return status

    def configure_episode(self, episode: Any) -> None:
        """Bind the isolated table target without adding navigation as a gate."""

        self._configured = False
        GraspSkillCertificationAgent.configure_episode(self, episode)
        if "move_body" not in self._declared_actions(episode):
            raise RuntimeError("Asset certification episode is missing move_body.")
        if "move_body" not in set(self.tools.spec_names()):
            raise RuntimeError("Asset certification move_body tool is not wired.")
        self._configured = True
        self.reset()

    def reset(self) -> None:
        """Start at the episode-frozen grasp stance in the isolated scene."""

        super().reset()
        self.stage = PhysicalTaskStage.GRASP
        self._target_material_evidence = {}

    def _coordinate_grounded_grasp(self, call: ToolCall, observation: Any) -> Any:
        result = super()._coordinate_grounded_grasp(call, observation)
        evidence = result.metadata.get("target_rgb_material_evidence")
        if not isinstance(evidence, dict) or not evidence:
            raise RuntimeError(
                "Asset certification did not produce target material evidence."
            )
        self._target_material_evidence = dict(evidence)
        return result


__all__ = [
    "GRASP_CERTIFICATION_CAMERA_TILT_DEG",
    "GraspAssetCertificationAgent",
    "GraspSkillCertificationAgent",
    "PhysicalTaskCertificationAgent",
    "PhysicalTaskStage",
    "target_rgb_material_evidence",
]
