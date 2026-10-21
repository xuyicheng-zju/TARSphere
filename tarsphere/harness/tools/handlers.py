"""Tool handlers: execute tool calls and return results.

Each handler implements the ToolHandler protocol (execute(call, context) -> ToolResult).
These handlers are the atomic-tool layer of the envelope protocol (D13):
small direct APIs for communication (speak/ask/handover_object), perception
(locate_object), control
(move/turn/wait), manipulation (end-effector/grasp), and completion
(report_done). Each is one call the model can reach for on its own; nothing here
runs a workflow on its behalf.

"One call" is not "one motion". ``grasp_object`` grounds its target with
SAM3 and returns an executable grasp action in a single call; the controller
behind that action may execute several arm motions. What a
handler must never do is decide something the benchmark is scoring — no
handler here routes the robot anywhere or picks its target for it.

Design rules:
- No hidden substitute path. If a tool fails, it raises or returns ToolResult(is_error=True).
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from tarsphere.harness.core.failures import (
    FailureReason,
    tool_failure,
    unexpected_tool_failure,
)
from tarsphere.harness.actions.router import requires_environment_execution
from tarsphere.harness.core.types import (
    CompositeDeferredToolEffect,
    ToolCall,
    ToolResult,
    tool_turn_action,
)
from tarsphere.harness.perception.annotated_frame import (
    GROUNDING_CANDIDATE_LABELS,
    annotate_grounder_candidates,
    annotate_instance_mask,
    rgb_frame_to_jpeg_data_url,
)
from tarsphere.harness.memory.semantic import (
    ReplaceSemanticEntityEffect,
    SemanticEntityRecord,
)
from tarsphere.harness.memory.episodic import ReplaceGroundingEffect
from tarsphere.harness.perception.object_grounding import (
    EgoFrameUnavailableError,
    ObjectGeometryUnavailableError,
    ObjectNotVisibleError,
    ground_visible_candidates,
    measure_selected_instance,
)
from tarsphere.infra.manipulation.graspgenx import (
    NoGraspPoseError,
    grasp_pose_payload,
)
from tarsphere.infra.manipulation.agent_grasp import (
    GRASP_FALLBACK_TARGET_NOT_IN_VIEW,
    GRASP_FALLBACK_VIEW_UNUSABLE,
    GRASP_GROUNDING_SAM3,
    grasp_target_observation,
)
from tarsphere.infra.control.physical_contracts import (
    CAMERA_GAZE_TILT_LIMIT_DEG,
    G1_GRASP_ABSOLUTE_MAX_HORIZONTAL_REACH_M,
    G1_END_EFFECTOR_DELTA_MAX_ROTATION_RAD,
    G1_END_EFFECTOR_DELTA_MAX_TRANSLATION_M,
    G1_MAXIMUM_CROUCH_HEIGHT_COMMAND_M,
    G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M,
    G1_NAVIGATION_FOOTPRINT_RADIUS_M,
    G1_RELATIVE_POSE_LATERAL_TRANSLATION_TOLERANCE_M,
    LOCOMOTION_MAX_BACKWARD_DISTANCE_M,
    LOCOMOTION_MAX_BODY_FORWARD_M,
    LOCOMOTION_MAX_BODY_LEFT_M,
    LOCOMOTION_MAX_LATERAL_DISTANCE_M,
    LOCOMOTION_MAX_STEP_DISTANCE_M,
    LOCOMOTION_MAX_TURN_DEG,
    LOCOMOTION_MAX_YIELD_SECONDS,
    LOCOMOTION_MIN_STEP_DISTANCE_M,
    LOCOMOTION_MIN_BODY_FORWARD_M,
    LOCOMOTION_MIN_TURN_DEG,
    LOCOMOTION_MIN_YIELD_SECONDS,
    NAVIGATION_PATH_ENTRY_YAW_TOLERANCE_RAD,
    NAVIGATION_SEGMENT_MAX_TURN_RAD,
    ROBOT_HUMAN_CENTER_CLEARANCE_M,
)


# The PointGoal handler owns this execution-only margin. A command that ends
# exactly on an A* waypoint can cross the inflated free-space boundary under
# the measured G1 gait's centimetre-scale endpoint drift. Long commands
# already stop before their waypoint.
POINTGOAL_WAYPOINT_ENDPOINT_MARGIN_M = 0.03
from tarsphere.geometry import (
    backproject_mask,
    fov_to_intrinsics,
    transform_to_world,
)
from tarsphere.embodiment.dex3 import G1_MANIPULATION_HAND


_XNAVDP_MAX_OFFICIAL_POINTGOAL_NORM_M = 25.0
# ---------------------------------------------------------------------------
# Locomotion handlers
# ---------------------------------------------------------------------------


def _bounded_argument(
    call: ToolCall,
    key: str,
    *,
    minimum: float,
    maximum: float,
    unit: str,
    meaning: str,
    remedy: str,
) -> tuple[float | None, ToolResult | None]:
    """Read a required numeric argument in range, or the failure to hand back.

    Range enforcement belongs here rather than in the action router: an
    out-of-range argument is the agent's mistake to see and correct, so it must
    come back as a tool failure it can read, not as an exception that ends the
    episode.

    ``meaning`` is the short phrase naming what the number is ("how far to
    walk"), and ``remedy`` is the sentence telling the agent what to do about an
    out-of-range value.  Both are the caller's because the bound means something
    different for a walking leg than for a camera angle, and a message that
    explains the wrong one is worse than none.
    """
    if key not in call.args:
        return None, tool_failure(
            call,
            FailureReason.MISSING_ARGUMENT,
            f"{call.tool} requires `{key}` — {meaning}, between "
            f"{minimum} and {maximum} {unit}.",
        )
    raw = call.args.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None, tool_failure(
            call,
            FailureReason.MALFORMED_ARGUMENT,
            f"{call.tool} `{key}` must be a number in {unit}; got {raw!r}.",
        )
    value = float(raw)
    if not minimum <= value <= maximum:
        return None, tool_failure(
            call,
            FailureReason.MALFORMED_ARGUMENT,
            f"{call.tool} `{key}` must be between {minimum} and {maximum} "
            f"{unit}; got {value}. {remedy}",
        )
    return value, None


# The remediation every motion magnitude shares: a leg or a hold that is too
# long is split, not clipped, so the agent looks again between the parts.
_SPLIT_THE_MOVE = (
    "Split a longer move into several calls so you look again between them."
)


class MoveBodyHandler:
    """Validate one continuous body-frame delta without choosing it."""

    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        del context
        raw_delta = call.args.get("delta")
        if not isinstance(raw_delta, (list, tuple)) or len(raw_delta) != 3:
            return tool_failure(
                call,
                FailureReason.MALFORMED_ARGUMENT,
                "move_body requires `delta` as exactly three numbers "
                "`[x, y, theta]` in body-frame metres, metres, and degrees.",
            )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in raw_delta
        ):
            return tool_failure(
                call,
                FailureReason.MALFORMED_ARGUMENT,
                f"move_body `delta` must contain only numbers; got {raw_delta!r}.",
            )
        delta = [float(value) for value in raw_delta]
        bounds = (
            (LOCOMOTION_MIN_BODY_FORWARD_M, LOCOMOTION_MAX_BODY_FORWARD_M),
            (-LOCOMOTION_MAX_BODY_LEFT_M, LOCOMOTION_MAX_BODY_LEFT_M),
            (-LOCOMOTION_MAX_TURN_DEG, LOCOMOTION_MAX_TURN_DEG),
        )
        if any(
            not minimum <= value <= maximum
            for value, (minimum, maximum) in zip(delta, bounds)
        ):
            return tool_failure(
                call,
                FailureReason.MALFORMED_ARGUMENT,
                "move_body `delta` is outside the allowed body-frame bounds: "
                f"x=[{LOCOMOTION_MIN_BODY_FORWARD_M}, "
                f"{LOCOMOTION_MAX_BODY_FORWARD_M}] m, "
                f"y=[-{LOCOMOTION_MAX_BODY_LEFT_M}, "
                f"{LOCOMOTION_MAX_BODY_LEFT_M}] m, "
                f"theta=[-{LOCOMOTION_MAX_TURN_DEG}, "
                f"{LOCOMOTION_MAX_TURN_DEG}] degrees. {_SPLIT_THE_MOVE}",
            )

        active_translation = max(abs(delta[0]), abs(delta[1]))
        active_turn = abs(delta[2])
        if active_translation < LOCOMOTION_MIN_STEP_DISTANCE_M and (
            active_turn < LOCOMOTION_MIN_TURN_DEG
        ):
            return tool_failure(
                call,
                FailureReason.MALFORMED_ARGUMENT,
                "move_body requires at least 0.1 m of translation or 5 degrees "
                "of turning; use wait when the body should remain still.",
            )
        return ToolResult(
            call_id=call.call_id,
            name="move_body",
            action={"action": "move_body", "delta": delta},
            content=(
                "Applying body-frame delta: "
                f"x {delta[0]:+.2f} m, y {delta[1]:+.2f} m, "
                f"theta {delta[2]:+.1f}°."
            ),
        )


class MoveEndEffectorHandler:
    """Validate one local right-palm delta without selecting a target pose."""

    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        del context
        raw_delta = call.args.get("delta")
        if not isinstance(raw_delta, (list, tuple)) or len(raw_delta) != 7:
            return tool_failure(
                call,
                FailureReason.MALFORMED_ARGUMENT,
                "move_end_effector requires `delta` as exactly seven numbers: "
                "`[dx, dy, dz, drx, dry, drz, gripper]`.",
            )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in raw_delta
        ):
            return tool_failure(
                call,
                FailureReason.MALFORMED_ARGUMENT,
                "move_end_effector `delta` must contain only numbers.",
            )
        delta = [float(value) for value in raw_delta]
        if not all(np.isfinite(value) for value in delta):
            return tool_failure(
                call,
                FailureReason.MALFORMED_ARGUMENT,
                "move_end_effector `delta` must contain only finite values.",
            )
        if any(
            abs(value) > G1_END_EFFECTOR_DELTA_MAX_TRANSLATION_M for value in delta[:3]
        ):
            return tool_failure(
                call,
                FailureReason.MALFORMED_ARGUMENT,
                "move_end_effector translation exceeds the per-axis limit of "
                f"{G1_END_EFFECTOR_DELTA_MAX_TRANSLATION_M:.2f} m; split the "
                "motion so a new ego observation is available between calls.",
            )
        rotation_norm = float(np.linalg.norm(delta[3:6]))
        if rotation_norm > G1_END_EFFECTOR_DELTA_MAX_ROTATION_RAD + 1e-12:
            return tool_failure(
                call,
                FailureReason.MALFORMED_ARGUMENT,
                "move_end_effector rotation-vector norm exceeds "
                f"{G1_END_EFFECTOR_DELTA_MAX_ROTATION_RAD:.4f} rad; split the "
                "rotation across observed calls.",
            )
        if not 0.0 <= delta[6] <= 1.0:
            return tool_failure(
                call,
                FailureReason.MALFORMED_ARGUMENT,
                "move_end_effector gripper must lie in [0, 1], where 0 is "
                "fully open and 1 is fully closed.",
            )
        return ToolResult(
            call_id=call.call_id,
            name="move_end_effector",
            action={"action": "move_end_effector", "delta": delta},
            content=(
                "Applying a local right-palm 6D delta with gripper closure "
                f"{delta[6]:.2f}."
            ),
        )


class MoveForwardHandler:
    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        del context
        distance_m, failure = _bounded_argument(
            call,
            "distance_m",
            minimum=LOCOMOTION_MIN_STEP_DISTANCE_M,
            maximum=LOCOMOTION_MAX_STEP_DISTANCE_M,
            unit="metres",
            meaning="how far to walk",
            remedy=_SPLIT_THE_MOVE,
        )
        if failure is not None:
            return failure
        return ToolResult(
            call_id=call.call_id,
            name="move_forward",
            action={"action": "move_forward", "distance_m": distance_m},
            content=f"Walking forward about {distance_m:.2f} m.",
        )


class MoveBackwardHandler:
    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        del context
        distance_m, failure = _bounded_argument(
            call,
            "distance_m",
            minimum=LOCOMOTION_MIN_STEP_DISTANCE_M,
            maximum=LOCOMOTION_MAX_BACKWARD_DISTANCE_M,
            unit="metres",
            meaning="how far to back up",
            remedy=_SPLIT_THE_MOVE,
        )
        if failure is not None:
            return failure
        return ToolResult(
            call_id=call.call_id,
            name="move_backward",
            action={"action": "move_backward", "distance_m": distance_m},
            content=f"Backing up about {distance_m:.2f} m.",
        )


class MoveLeftHandler:
    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        del context
        distance_m, failure = _bounded_argument(
            call,
            "distance_m",
            minimum=LOCOMOTION_MIN_STEP_DISTANCE_M,
            maximum=LOCOMOTION_MAX_LATERAL_DISTANCE_M,
            unit="metres",
            meaning="how far to strafe left",
            remedy=_SPLIT_THE_MOVE,
        )
        if failure is not None:
            return failure
        return ToolResult(
            call_id=call.call_id,
            name="move_left",
            action={"action": "move_left", "distance_m": distance_m},
            content=f"Strafing left about {distance_m:.2f} m.",
        )


class MoveRightHandler:
    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        del context
        distance_m, failure = _bounded_argument(
            call,
            "distance_m",
            minimum=LOCOMOTION_MIN_STEP_DISTANCE_M,
            maximum=LOCOMOTION_MAX_LATERAL_DISTANCE_M,
            unit="metres",
            meaning="how far to strafe right",
            remedy=_SPLIT_THE_MOVE,
        )
        if failure is not None:
            return failure
        return ToolResult(
            call_id=call.call_id,
            name="move_right",
            action={"action": "move_right", "distance_m": distance_m},
            content=f"Strafing right about {distance_m:.2f} m.",
        )


class TurnLeftHandler:
    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        del context
        degrees, failure = _bounded_argument(
            call,
            "degrees",
            minimum=LOCOMOTION_MIN_TURN_DEG,
            maximum=LOCOMOTION_MAX_TURN_DEG,
            unit="degrees",
            meaning="how far to rotate",
            remedy=_SPLIT_THE_MOVE,
        )
        if failure is not None:
            return failure
        return ToolResult(
            call_id=call.call_id,
            name="turn_left",
            action={"action": "turn_left", "degrees": degrees},
            content=f"Turning left about {degrees:.0f}°.",
        )


class TurnRightHandler:
    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        del context
        degrees, failure = _bounded_argument(
            call,
            "degrees",
            minimum=LOCOMOTION_MIN_TURN_DEG,
            maximum=LOCOMOTION_MAX_TURN_DEG,
            unit="degrees",
            meaning="how far to rotate",
            remedy=_SPLIT_THE_MOVE,
        )
        if failure is not None:
            return failure
        return ToolResult(
            call_id=call.call_id,
            name="turn_right",
            action={"action": "turn_right", "degrees": degrees},
            content=f"Turning right about {degrees:.0f}°.",
        )


class WaitHandler:
    """Hold the pose for a chosen number of seconds while the world moves."""

    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        del context
        seconds, failure = _bounded_argument(
            call,
            "seconds",
            minimum=LOCOMOTION_MIN_YIELD_SECONDS,
            maximum=LOCOMOTION_MAX_YIELD_SECONDS,
            unit="seconds",
            meaning="how long to hold",
            remedy=_SPLIT_THE_MOVE,
        )
        if failure is not None:
            return failure
        return ToolResult(
            call_id=call.call_id,
            name="wait",
            action={"action": "wait", "seconds": seconds},
            content=f"Holding still for {seconds:.1f} s.",
        )


class TiltCameraHandler:
    """Aim the ego camera up or down without moving the body."""

    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        del context
        pitch_deg, failure = _bounded_argument(
            call,
            "pitch_deg",
            minimum=-CAMERA_GAZE_TILT_LIMIT_DEG,
            maximum=CAMERA_GAZE_TILT_LIMIT_DEG,
            unit="degrees",
            meaning="where to aim, relative to the default navigation view",
            remedy=(
                "Past that angle the horizon leaves the frame, and the horizon "
                "is what ties this image to the map. Change the camera angle, "
                "body pose, or viewpoint using current evidence rather than "
                "requesting an out-of-range tilt."
            ),
        )
        if failure is not None:
            return failure
        return ToolResult(
            call_id=call.call_id,
            name="tilt_camera",
            action={"action": "tilt_camera", "pitch_deg": pitch_deg},
            content=(
                f"Aiming the camera {pitch_deg:+.0f}° from its default view."
                if pitch_deg
                else "Returning the camera to its default navigation view."
            ),
        )


class CrouchHandler:
    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        del context
        pelvis_height_m, failure = _bounded_argument(
            call,
            "pelvis_height_m",
            minimum=G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M,
            maximum=G1_MAXIMUM_CROUCH_HEIGHT_COMMAND_M,
            unit="metres",
            meaning="the desired pelvis height above the floor",
            remedy=(
                "Use a lower value for a deeper reach and a higher value for "
                "a shallow reach; stand_up restores the walking height."
            ),
        )
        if failure is not None:
            return failure
        return ToolResult(
            call_id=call.call_id,
            name="crouch",
            action={
                "action": "crouch",
                "pelvis_height_m": pelvis_height_m,
            },
            content=f"Lowering pelvis to {pelvis_height_m:.2f} m.",
        )


class StandUpHandler:
    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        del context
        return ToolResult(
            call_id=call.call_id,
            name="stand_up",
            action={"action": "stand_up"},
            content="Returning to the certified standing height.",
        )


# ---------------------------------------------------------------------------
# Communication handlers
# ---------------------------------------------------------------------------


class SpeakHandler:
    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        recipient = str(call.args.get("recipient", "")).strip()
        if not recipient:
            return tool_failure(
                call,
                FailureReason.MISSING_ARGUMENT,
                "speak requires `recipient` — the public name of the person "
                "being addressed.",
            )
        message = str(call.args.get("message", ""))
        if not message.strip():
            return tool_failure(
                call,
                FailureReason.MISSING_ARGUMENT,
                "speak requires `message` — what to say out loud.",
            )
        return ToolResult(
            call_id=call.call_id,
            name="speak",
            action={
                "action": "speak",
                "recipient": recipient,
                "message": message,
            },
            content=f'Said to {recipient}: "{message}"',
        )


class AskHandler:
    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        recipient = str(call.args.get("recipient", "")).strip()
        if not recipient:
            return tool_failure(
                call,
                FailureReason.MISSING_ARGUMENT,
                "ask requires `recipient` — the public name of the person being asked.",
            )
        intent_id = str(call.args.get("intent_id", "")).strip()
        message = str(call.args.get("message", ""))
        if not message.strip():
            return tool_failure(
                call,
                FailureReason.MISSING_ARGUMENT,
                "ask requires `message` — the question to put to the human.",
            )
        return ToolResult(
            call_id=call.call_id,
            name="ask",
            action={
                "action": "ask",
                "recipient": recipient,
                **({"intent_id": intent_id} if intent_id else {}),
                "message": message,
            },
            content=f'Asked {recipient} with intent {intent_id}: "{message}"',
        )


class ReportDoneHandler:
    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            call_id=call.call_id,
            name="report_done",
            action={"action": "report_done"},
            content="Task reported as done.",
        )


class HandoverObjectHandler:
    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        del context
        recipient = call.args.get("recipient")
        if not isinstance(recipient, str) or not recipient.strip():
            return tool_failure(
                call,
                FailureReason.MISSING_ARGUMENT,
                "handover_object requires `recipient` — the public name or "
                "visible description of the person.",
            )
        return ToolResult(
            call_id=call.call_id,
            name="handover_object",
            action={
                "action": "handover_object",
                "recipient": recipient.strip(),
            },
            content="Offered the held object.",
        )


# ---------------------------------------------------------------------------
# Perception handler: ground a named object in RGB and lift it privately
# ---------------------------------------------------------------------------


def _action_step(context: Dict[str, Any]) -> int:
    """The primitive action step this call belongs to.

    The step states when the RGB grounding was acquired, so the Agent can tell
    current evidence from a grounding captured before the object moved.
    """
    try:
        return int(context.get("action_step", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _target_observation(
    target_center_camera_m: Any,
    world_T_camera: Any,
) -> dict[str, float]:
    """Return the compact observation derived directly from ego depth."""

    return grasp_target_observation(
        target_center_camera_m=target_center_camera_m,
        world_T_camera=world_T_camera,
    )


def _nearest_target_horizontal_distance_from_robot_m(
    *,
    grounding: Any,
    observation: Any,
) -> float | None:
    """Measure robot-base distance to the nearest observed target surface.

    ``None`` means the observation has no public base pose, so this cheap gate
    cannot prove an impossibility and must leave reachability to GraspGenX and
    cuRobo. A finite measurement uses every synchronized mask-depth point,
    rather than the object centre, so a large object is not rejected while its
    near surface remains reachable.
    """

    pose = getattr(observation, "pose", None)
    if not isinstance(pose, dict) or pose.get("position") is None:
        return None
    robot_position = np.asarray(pose["position"], dtype=np.float64).reshape(-1)
    if robot_position.size < 2 or not np.all(np.isfinite(robot_position[:2])):
        raise ValueError("grasp_object robot pose must contain a finite XY position")
    points_camera = backproject_mask(
        np.asarray(grounding.target_mask, dtype=bool),
        np.asarray(grounding.depth_m),
        np.asarray(grounding.camera_intrinsics, dtype=np.float64),
    )
    if not len(points_camera):
        raise ValueError("grounded grasp target has no valid synchronized depth point")
    points_world = transform_to_world(
        points_camera,
        np.asarray(grounding.world_T_camera, dtype=np.float64),
    )
    distances = np.linalg.norm(
        points_world[:, :2] - robot_position[None, :2],
        axis=1,
    )
    return float(np.min(distances))


def _grasp_grounding_evidence(
    *,
    target: str,
    observation: Any,
    grounding: Any,
    candidate_label: str | None = None,
) -> tuple[tuple[str, ...], str]:
    """Return the SAM3 mask overlay and prose framing for grasp results."""

    ego = observation.views["ego"]
    detection = grounding.detection
    annotated = annotate_instance_mask(
        ego.rgb,
        grounding.target_mask,
        label=(
            f"{candidate_label}: {detection.label}"
            if candidate_label
            else str(detection.label)
        ),
        score=float(detection.score),
    )
    if not annotated:
        raise RuntimeError("mandatory SAM3 grasp overlay encoding returned empty")
    images = (annotated,)
    framing = (
        f"The returned image highlights the region SAM3 segmented for "
        f"{target!r}. If the highlight is on the wrong object, retry "
        "grasp_object with a more specific phrase before acting on this result."
    )
    return images, framing


class LocateObjectHandler:
    """Ground a named object in RGB and retain its measured location.

    SAM3 ranks current-frame instances, selects the highest-scoring usable
    candidate, and lifts it with synchronized depth. The Agent receives a
    marked overlay and sensor-relative measurements, never world coordinates.

    The lifting reuses the evaluator's camera geometry (``tarsphere.geometry``),
    but this is the agent's own sensing: it runs on the ego frame, only when
    asked, and reports what the grounder actually saw. ``grasp_object``
    independently regrounds its current frame.

    Args:
        semantic_backend: a loaded SAM3PerceptionBackend (resident for the
            episode).
        memory: EpisodicMemory, which retains the measured location.
    """

    def __init__(
        self,
        *,
        semantic_backend: Any,
        memory: Any = None,
    ) -> None:
        self._backend = semantic_backend
        self._memory = memory
        # The selected candidate is measured against the same capture.
        self._pending_candidates: dict[str, Any] | None = None

    def reset_pending(self) -> None:
        """Drop episode-scoped candidate marks (called on episode reset)."""
        self._pending_candidates = None

    @property
    def backend(self) -> Any:
        """The wired semantic backend, for entrypoint warmup."""
        return self._backend

    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        """Locate the best-matching candidate in one call.

        SAM3 grounds the target on the current frame, the highest-scoring
        usable candidate is measured, optionally retained by the configured
        memory protocol, and returned with its annotated overlay. The tool
        identity is the turn
        outcome; the runner observes and does not step physics.
        """
        target = str(call.args.get("target") or "").strip()
        if not target:
            return tool_failure(
                call,
                FailureReason.MISSING_ARGUMENT,
                "locate_object requires `target` — the phrase to look for in "
                'the current ego view (e.g. "red cup").',
            )
        observation = context.get("observation")
        try:
            candidates = ground_visible_candidates(
                semantic_backend=self._backend,
                observation=observation,
                target=target,
            )
        except EgoFrameUnavailableError as exc:
            return tool_failure(
                call,
                FailureReason.BACKEND_UNAVAILABLE,
                f"locate_object cannot observe the current ego frame: {exc}.",
                target=target,
            )
        except ObjectNotVisibleError:
            return tool_failure(
                call,
                FailureReason.NOT_IN_VIEW,
                f'locate_object did not ground "{target}" in the current ego '
                "frame. The grounder may have missed it (small, backlit, or "
                "unusual phrasing), so this is not proof of absence. Try a "
                "more specific or plainer phrase, or turn or move for a "
                "closer view before concluding it is not here.",
                target=target,
                detected=False,
            )
        except ObjectGeometryUnavailableError as exc:
            return tool_failure(
                call,
                FailureReason.DEPTH_UNAVAILABLE,
                f"locate_object cannot obtain current metric geometry for "
                f'"{target}": {exc}.',
                target=target,
                detected=True,
                world_xy=None,
            )
        except Exception as exc:  # noqa: BLE001 — surface, don't swallow
            return unexpected_tool_failure(call, exc)

        ego = observation.views["ego"]
        capture_id = int(ego.camera_params["capture_id"])
        marked = candidates.top(len(GROUNDING_CANDIDATE_LABELS))
        self._pending_candidates = {
            "target": target,
            "capture_id": capture_id,
            "candidates": candidates,
            "marked": marked,
        }
        top_result = self._execute_selection(
            call,
            target=target,
            selection=GROUNDING_CANDIDATE_LABELS[0],
            context=context,
        )
        if top_result.is_error:
            return top_result
        annotated = annotate_grounder_candidates(ego.rgb, marked)
        labels_used = GROUNDING_CANDIDATE_LABELS[: len(marked)]
        return ToolResult(
            call_id=call.call_id,
            name=call.tool,
            action=tool_turn_action(call.tool),
            content=top_result.content,
            metadata={
                **top_result.metadata,
                "grounding_candidate_labels": list(labels_used),
                "candidate_count": len(marked),
                "candidate_scores": [float(detection.score) for detection in marked],
            },
            images=tuple(top_result.images) + ((annotated,) if annotated else ()),
        )

    def _execute_selection(
        self,
        call: ToolCall,
        *,
        target: str,
        selection: str,
        context: Dict[str, Any],
    ) -> ToolResult:
        """Measure the selected candidate for this capture, then save it."""
        selection = selection.upper()
        pending = self._pending_candidates
        observation = context.get("observation")
        ego = (
            observation.views.get("ego")
            if observation is not None and observation.views
            else None
        )
        capture_id = None
        if ego is not None:
            try:
                capture_id = int(ego.camera_params["capture_id"])
            except (AttributeError, KeyError, TypeError, ValueError):
                capture_id = None
        if pending is None:
            return tool_failure(
                call,
                FailureReason.BACKEND_UNAVAILABLE,
                "locate_object could not ground any candidate for "
                f"{target!r}; nothing was measured or saved.",
                target=target,
            )
        marked = pending["marked"]
        labels_used = GROUNDING_CANDIDATE_LABELS[: len(marked)]
        if selection not in labels_used:
            return tool_failure(
                call,
                FailureReason.UNKNOWN_VALUE,
                "locate_object could not ground any usable candidate for "
                f"{target!r}; nothing was measured or saved.",
                target=target,
            )
        chosen = marked[labels_used.index(selection)]
        candidates = pending["candidates"]
        try:
            grounding = measure_selected_instance(candidates, chosen)
        except ObjectGeometryUnavailableError as exc:
            return tool_failure(
                call,
                FailureReason.DEPTH_UNAVAILABLE,
                f"locate_object cannot measure selected candidate "
                f'{selection} for "{target}": {exc}.',
                target=target,
                detected=True,
                world_xy=None,
            )
        except Exception as exc:  # noqa: BLE001 — surface, don't swallow
            return unexpected_tool_failure(call, exc)
        self._pending_candidates = None

        bbox = grounding.detection.box_xyxy  # corrected in oracle diagnostics
        world_xy = list(grounding.target_center_world_m[:2])
        target_observation = _target_observation(
            grounding.target_center_camera_m,
            grounding.world_T_camera,
        )

        retain_spatial_memory = bool(self._memory is not None and True)
        if retain_spatial_memory:
            self._memory.remember_grounding(
                label=target,
                world_xy=world_xy,
                world_z_m=float(grounding.target_center_world_m[2]),
                grounding_depth_m=float(grounding.target_center_camera_m[2]),
                bbox_2d=list(bbox),
                score=float(chosen.score),
                action_step=_action_step(context),
            )
            remember_current = getattr(
                self._memory,
                "remember_current_view_grounding",
                None,
            )
            if callable(remember_current):
                remember_current(
                    label=target,
                    capture_id=int(capture_id),
                    grounding=grounding,
                )

        return ToolResult(
            call_id=call.call_id,
            name=call.tool,
            action=tool_turn_action(call.tool),
            content=(
                f"locate_object {'saved' if retain_spatial_memory else 'measured'} "
                f'"{target}" as candidate {selection} '
                f"(score {chosen.score:.2f}, pixel box {list(bbox)}). "
                "Same-capture depth places its center relative to the ego "
                "camera at horizontal distance="
                f"{target_observation['horizontal_distance_m']:.2f} m, "
                "height="
                f"{target_observation['height_from_camera_m']:+.2f} m, "
                f"bearing={target_observation['bearing_deg']:+.1f}° "
                "(positive height is above the camera; positive bearing is "
                "left). These are current "
                "sensor-relative measurements, not a base goal or world "
                "coordinate. "
                + (
                    "The landmark now enters spatial memory and the exploration BEV."
                    if retain_spatial_memory
                    else "No persistent spatial landmark is retained in this run."
                )
            ),
            metadata={
                "target": target,
                "detected": True,
                "selected_label": selection,
                "label": chosen.label,
                "score": float(chosen.score),
                "bbox_2d": list(bbox),
                "world_xy": world_xy,
                # Evaluator-only provenance. RobotAgent keeps this coordinate
                # out of model-visible tool results.
                "target_center_world_m": list(grounding.target_center_world_m),
                "target_observation": target_observation,
            },
        )


_SEMANTIC_ENTITY_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _semantic_ego_frame(
    call: ToolCall,
    context: Dict[str, Any],
) -> tuple[Any | None, ToolResult | None]:
    observation = context.get("observation")
    ego = (
        observation.views.get("ego")
        if observation is not None and observation.views
        else None
    )
    if ego is None or ego.rgb is None:
        return None, tool_failure(
            call,
            FailureReason.BACKEND_UNAVAILABLE,
            f"{call.tool} requires the current ego RGB frame.",
        )
    camera = dict(ego.camera_params or {})
    try:
        capture_id = int(camera["capture_id"])
        depth_capture_id = int(camera["depth_capture_id"])
    except (KeyError, TypeError, ValueError):
        return None, tool_failure(
            call,
            FailureReason.DEPTH_UNAVAILABLE,
            f"{call.tool} requires stamped, synchronized current RGB-D.",
        )
    if capture_id != depth_capture_id:
        return None, tool_failure(
            call,
            FailureReason.DEPTH_UNAVAILABLE,
            f"{call.tool} refuses mismatched RGB-D capture ids.",
        )
    return ego, None


def _place_world_point(*, ego: Any, image_point: Any) -> tuple[float, float, float]:
    if not isinstance(image_point, list) or len(image_point) != 2:
        raise ValueError(
            "`image_point` is required for a place and must be normalized [x,y]."
        )
    normalized = np.asarray(image_point, dtype=np.float64)
    if (
        not np.all(np.isfinite(normalized))
        or np.any(normalized < 0.0)
        or np.any(normalized > 1.0)
    ):
        raise ValueError("`image_point` coordinates must each lie in [0,1].")
    depth = np.asarray(ego.depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("Current ego metric depth is not an HxW raster.")
    height, width = depth.shape
    pixel_x = int(round(float(normalized[0]) * (width - 1)))
    pixel_y = int(round(float(normalized[1]) * (height - 1)))
    radius = 3
    x1, x2 = max(0, pixel_x - radius), min(width, pixel_x + radius + 1)
    y1, y2 = max(0, pixel_y - radius), min(height, pixel_y + radius + 1)
    patch = depth[y1:y2, x1:x2]
    valid = np.isfinite(patch) & (patch > 0.0)
    if not valid.any():
        raise ValueError(
            "No valid metric depth supports the selected current-image point."
        )
    ys, xs = np.nonzero(valid)
    patch_depths = patch[ys, xs]
    median_index = int(np.argsort(patch_depths)[len(patch_depths) // 2])
    u = float(x1 + xs[median_index])
    v = float(y1 + ys[median_index])
    z = float(patch_depths[median_index])
    camera = dict(ego.camera_params or {})
    intrinsics = fov_to_intrinsics(
        float(camera["fov_deg"]),
        int(camera.get("width") or width),
        int(camera.get("height") or height),
    )
    point_camera = np.asarray(
        [
            [
                (u - intrinsics[0, 2]) * z / intrinsics[0, 0],
                (v - intrinsics[1, 2]) * z / intrinsics[1, 1],
                z,
            ]
        ],
        dtype=np.float64,
    )
    world_T_camera = np.asarray(camera["world_T_cam"], dtype=np.float64)
    if world_T_camera.shape != (4, 4) or not np.all(np.isfinite(world_T_camera)):
        raise ValueError("Current ego world_T_cam must be a finite 4x4 transform.")
    world = transform_to_world(point_camera, world_T_camera)[0]
    return tuple(float(value) for value in world)


class RememberEntityHandler:
    """Stage one current-view semantic landmark without advancing physics.

    An `object` landmark is grounded with SAM3 and the best candidate is
    measured and staged in the same call; the annotated overlay is returned.
    Humans are grounded through the public visibility table and places
    through picked depth.
    """

    def __init__(self, *, memory: Any, semantic_backend: Any | None) -> None:
        self._memory = memory
        self._semantic_backend = semantic_backend
        # One live candidate set for the object branch, like locate_object's.
        self._pending_candidates: dict[str, Any] | None = None

    def reset_pending(self) -> None:
        """Drop episode-scoped candidate marks (called on episode reset)."""
        self._pending_candidates = None

    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        entity_id = str(call.args.get("entity_id") or "").strip()
        entity_type = str(call.args.get("entity_type") or "").strip()
        label = str(call.args.get("label") or "").strip()
        description = str(call.args.get("description") or "").strip()
        if not _SEMANTIC_ENTITY_ID.fullmatch(entity_id):
            return tool_failure(
                call,
                FailureReason.MALFORMED_ARGUMENT,
                "remember_entity `entity_id` must be lower snake_case and at "
                "most 64 characters.",
            )
        if "select" in call.args:
            return tool_failure(
                call,
                FailureReason.UNKNOWN_VALUE,
                "remember_entity has no `select` argument; object landmarks "
                "use the best current-frame candidate in one call.",
            )
        ego, failure = _semantic_ego_frame(call, context)
        if failure is not None:
            return failure
        observation = context["observation"]
        if entity_type == "object":
            marking = self._execute_object_marking(
                call,
                entity_id=entity_id,
                label=label,
                ego=ego,
                observation=observation,
            )
            if marking.is_error or not marking.metadata.get(
                "grounding_candidate_labels",
            ):
                return marking
            return self._execute_object_selection(
                call,
                entity_id=entity_id,
                entity_type=entity_type,
                label=label,
                description=description,
                selection=marking.metadata["grounding_candidate_labels"][0],
                ego=ego,
                context=context,
            )
        return self._execute_immediate(
            call,
            entity_id=entity_id,
            entity_type=entity_type,
            label=label,
            description=description,
            selection="",
            ego=ego,
            observation=observation,
            context=context,
        )

    def _execute_object_marking(
        self,
        call: ToolCall,
        *,
        entity_id: str,
        label: str,
        ego: Any,
        observation: Any,
    ) -> ToolResult:
        """Rank SAM3 instances for the label and return letter marks."""
        if self._semantic_backend is None:
            return tool_failure(
                call,
                FailureReason.BACKEND_UNAVAILABLE,
                "remember_entity requires the semantic grounder to mark an "
                "object landmark.",
            )
        try:
            candidates = ground_visible_candidates(
                semantic_backend=self._semantic_backend,
                observation=observation,
                target=label,
            )
        except ObjectNotVisibleError:
            return tool_failure(
                call,
                FailureReason.NOT_IN_VIEW,
                f"remember_entity did not ground object {label!r} in the "
                "current ego frame; the grounder may have missed it, which is "
                "not proof of absence. Try a plainer phrase or a closer view.",
            )
        except ObjectGeometryUnavailableError as exc:
            return tool_failure(
                call,
                FailureReason.DEPTH_UNAVAILABLE,
                f"remember_entity cannot ground {label!r}: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 — surface, don't swallow
            return unexpected_tool_failure(call, exc)
        capture_id = int(ego.camera_params["capture_id"])
        marked = candidates.top(len(GROUNDING_CANDIDATE_LABELS))
        self._pending_candidates = {
            "entity_id": entity_id,
            "label": label,
            "capture_id": capture_id,
            "candidates": candidates,
            "marked": marked,
        }
        annotated = annotate_grounder_candidates(ego.rgb, marked)
        labels_used = GROUNDING_CANDIDATE_LABELS[: len(marked)]
        ranking = "; ".join(
            f"{mark} = {detection.label!r} ({float(detection.score):.2f})"
            for mark, detection in zip(labels_used, marked)
        )
        return ToolResult(
            call_id=call.call_id,
            name=call.tool,
            action=tool_turn_action(call.tool),
            content=(
                f"remember_entity grounded {len(marked)} candidate "
                f"instance(s) for {label!r}: {ranking}."
            ),
            metadata={
                "grounding_candidate_labels": list(labels_used),
                "entity_id": entity_id,
                "target": label,
                "capture_id": capture_id,
                "candidate_scores": [float(detection.score) for detection in marked],
            },
            images=(annotated,) if annotated else (),
        )

    def _execute_object_selection(
        self,
        call: ToolCall,
        *,
        entity_id: str,
        entity_type: str,
        label: str,
        description: str,
        selection: str,
        ego: Any,
        context: Dict[str, Any],
    ) -> ToolResult:
        """Measure the chosen candidate and stage the landmark record."""
        capture_id = int(ego.camera_params["capture_id"])
        pending = self._pending_candidates
        marked = pending["marked"] if pending else []
        labels_used = GROUNDING_CANDIDATE_LABELS[: len(marked)]
        if selection not in labels_used:
            return tool_failure(
                call,
                FailureReason.UNKNOWN_VALUE,
                f"remember_entity could not ground any candidate for "
                f"{label!r}; nothing was saved.",
                entity_id=entity_id,
            )
        chosen = marked[labels_used.index(selection)]
        try:
            grounding = measure_selected_instance(
                (pending or {}).get("candidates"),
                chosen,
            )
        except ObjectGeometryUnavailableError as exc:
            return tool_failure(
                call,
                FailureReason.DEPTH_UNAVAILABLE,
                f"remember_entity cannot measure selected candidate "
                f"{selection} for {label!r}: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 — surface, don't swallow
            return unexpected_tool_failure(call, exc)
        self._pending_candidates = None

        record = SemanticEntityRecord(
            entity_id=entity_id,
            entity_type="object",
            label=label,
            description=description,
            world_xyz=tuple(float(value) for value in grounding.target_center_world_m),
            observed_action_step=_action_step(context),
            capture_id=capture_id,
            keyframe_rgb=np.asarray(ego.rgb, dtype=np.uint8).copy(),
            bbox_2d=tuple(float(value) for value in grounding.detection.box_xyxy),
        )
        semantic_effect = ReplaceSemanticEntityEffect(
            self._memory.semantic,
            record,
        )
        deferred_effect = CompositeDeferredToolEffect(
            semantic_effect,
            ReplaceGroundingEffect(
                memory=self._memory,
                label=label,
                grounding=grounding,
                bbox=list(record.bbox_2d or ()),
                action_step=record.observed_action_step,
                capture_id=capture_id,
            ),
        )
        annotated = annotate_grounder_candidates(ego.rgb, marked)
        return ToolResult(
            call_id=call.call_id,
            name=call.tool,
            action=tool_turn_action(call.tool),
            content=(
                f"remember_entity saved {entity_id!r} from candidate "
                f"{selection} (score {float(chosen.score):.2f}) of the "
                "current capture as an object landmark."
            ),
            metadata={
                "entity_id": entity_id,
                "entity_type": "object",
                "selected_label": selection,
                "capture_id": capture_id,
                "observed_action_step": record.observed_action_step,
            },
            images=(annotated,) if annotated else (),
            deferred_effect=deferred_effect,
        )

    def _execute_immediate(
        self,
        call: ToolCall,
        *,
        entity_id: str,
        entity_type: str,
        label: str,
        description: str,
        selection: str,
        ego: Any,
        observation: Any,
        context: Dict[str, Any],
    ) -> ToolResult:
        bbox: tuple[float, float, float, float] | None = None
        grounding = None
        try:
            if entity_type == "human":
                if selection:
                    return tool_failure(
                        call,
                        FailureReason.MALFORMED_ARGUMENT,
                        "remember_entity `select` applies only to object "
                        "landmarks; humans are grounded by their catalog "
                        "name.",
                    )
                visible = dict(getattr(observation, "visible_humans", None) or {})
                matched_name = next(
                    (
                        name
                        for name in visible
                        if str(name).casefold() == label.casefold()
                    ),
                    None,
                )
                if matched_name is None:
                    return tool_failure(
                        call,
                        FailureReason.NOT_IN_VIEW,
                        f"remember_entity: Human {label!r} is not grounded in "
                        "the current ego RGB-D frame.",
                    )
                xy = visible[matched_name]
                world_xyz = (float(xy[0]), float(xy[1]), 0.0)
            elif entity_type == "place":
                if selection:
                    return tool_failure(
                        call,
                        FailureReason.MALFORMED_ARGUMENT,
                        "remember_entity `select` applies only to object "
                        "landmarks; places are grounded by `image_point`.",
                    )
                world_xyz = _place_world_point(
                    ego=ego,
                    image_point=call.args.get("image_point"),
                )
            else:
                return tool_failure(
                    call,
                    FailureReason.UNKNOWN_VALUE,
                    f"remember_entity does not support type {entity_type!r}.",
                )
        except ObjectGeometryUnavailableError as exc:
            return tool_failure(
                call,
                FailureReason.DEPTH_UNAVAILABLE,
                f"remember_entity cannot ground {label!r}: {exc}",
            )
        except (KeyError, TypeError, ValueError) as exc:
            return tool_failure(
                call,
                FailureReason.MALFORMED_ARGUMENT,
                f"remember_entity cannot store {label!r}: {exc}",
            )

        record = SemanticEntityRecord(
            entity_id=entity_id,
            entity_type=entity_type,
            label=label,
            description=description,
            world_xyz=tuple(float(value) for value in world_xyz),
            observed_action_step=_action_step(context),
            capture_id=int(ego.camera_params["capture_id"]),
            keyframe_rgb=np.asarray(ego.rgb, dtype=np.uint8).copy(),
            bbox_2d=bbox,
        )
        semantic_effect = ReplaceSemanticEntityEffect(
            self._memory.semantic,
            record,
        )
        deferred_effect = semantic_effect
        if grounding is not None:
            deferred_effect = CompositeDeferredToolEffect(
                semantic_effect,
                ReplaceGroundingEffect(
                    memory=self._memory,
                    label=label,
                    grounding=grounding,
                    bbox=list(bbox or ()),
                    action_step=_action_step(context),
                    capture_id=record.capture_id,
                ),
            )
        return ToolResult(
            call_id=call.call_id,
            name=call.tool,
            action=tool_turn_action(call.tool),
            content=(
                f"remember_entity saved {entity_id!r} from the current "
                f"capture as a {entity_type} landmark."
            ),
            metadata={
                "entity_id": entity_id,
                "entity_type": entity_type,
                "capture_id": record.capture_id,
                "observed_action_step": record.observed_action_step,
            },
            deferred_effect=deferred_effect,
        )


class InspectEntityHandler:
    """Return a stored semantic keyframe without changing world state."""

    def __init__(self, *, memory: Any) -> None:
        self._memory = memory

    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        del context
        entity_id = str(call.args.get("entity_id") or "").strip()
        record = self._memory.semantic.get(entity_id)
        if record is None:
            return tool_failure(
                call,
                FailureReason.UNKNOWN_VALUE,
                f"inspect_entity has no saved entity {entity_id!r}.",
            )
        image = rgb_frame_to_jpeg_data_url(record.keyframe_rgb)
        if image is None:
            return tool_failure(
                call,
                FailureReason.BACKEND_UNAVAILABLE,
                f"inspect_entity could not encode keyframe {entity_id!r}.",
            )
        return ToolResult(
            call_id=call.call_id,
            name=call.tool,
            action=tool_turn_action(call.tool),
            content=(
                f"inspect_entity: historical keyframe for {entity_id!r}, "
                f"saved at action step {record.observed_action_step}. It is "
                "not a live observation."
            ),
            metadata={
                "entity_id": entity_id,
                "capture_id": record.capture_id,
                "observed_action_step": record.observed_action_step,
            },
            images=(image,),
        )


# ---------------------------------------------------------------------------
# Manipulation handlers
# ---------------------------------------------------------------------------


def _grasp_grounding_failure(
    call: ToolCall,
    *,
    message: str,
    error_type: str,
    requested_target: str,
    images: tuple[str, ...] = (),
) -> ToolResult:
    return tool_failure(
        call,
        error_type,
        message,
        images=images,
        requested_target=requested_target,
    )


def _build_grasp_action(
    *,
    target: str,
    hand: str,
    grounding_mode: str,
    target_center_world_m: Sequence[float],
    target_center_camera_m: Sequence[float],
    bbox_2d: Sequence[float],
    semantic_score: float,
    world_T_camera: np.ndarray,
    capture_id: int,
    candidates: Sequence[Any],
    remembered_action_step: int | None = None,
) -> dict[str, Any]:
    agent_grasp: dict[str, Any] = {
        "target": target,
        "target_center_world_m": list(target_center_world_m),
        "target_center_camera_m": list(target_center_camera_m),
        "bbox_2d": list(bbox_2d),
        "semantic_score": float(semantic_score),
        "world_T_camera": np.asarray(world_T_camera, dtype=np.float64).tolist(),
        "capture_id": int(capture_id),
        "grounding_mode": grounding_mode,
        "grasp_pose_candidates": [
            grasp_pose_payload(candidate) for candidate in candidates
        ],
    }
    if remembered_action_step is not None:
        agent_grasp["remembered_action_step"] = int(remembered_action_step)
    return {
        "action": "grasp_object",
        "hand": hand,
        "preset": "grasp",
        "target": target,
        "controller_metadata": {"agent_grasp": agent_grasp},
    }


class GraspObjectHandler:
    """Ground a physical pick-up from RGB with tool-private metric depth.

    The public call is one manipulation skill, not an arm-control dialogue.
    SAM3 selects the named pixels, same-capture private depth supplies metric
    geometry, and GraspGenX proposes native three-finger poses. The action
    carries only those measured proposals into Isaac; the runtime then owns
    one closing-pose reach, physical contact, and retention. Only after that
    gate does a second plan lift the object and restore the walking posture.

    Every attempt runs SAM3 exactly once on the current RGB frame, selects
    the highest-scoring usable candidate, and returns its labeled overlay
    alongside the executed grasp. Grounding always uses the live frame;
    earlier localization never substitutes for it.
    """

    def __init__(
        self,
        *,
        semantic_backend: Any,
        grasp_backend: Any,
        memory: Any = None,
    ) -> None:
        self._semantic_backend = semantic_backend
        self._grasp_backend = grasp_backend
        self._memory = memory

    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        target = str(call.args.get("target") or "").strip()
        if not target:
            return tool_failure(
                call,
                FailureReason.MISSING_ARGUMENT,
                "grasp_object requires `target` — the object to pick up in the "
                'current ego view (e.g. "steel bottle").',
            )
        if "hand" in call.args:
            return tool_failure(
                call,
                FailureReason.UNKNOWN_VALUE,
                "grasp_object has no `hand` argument; TARSphere manipulation "
                "uses the physically certified right hand.",
            )
        if "confirm_preview_id" in call.args:
            return tool_failure(
                call,
                FailureReason.UNKNOWN_VALUE,
                "grasp_object has no `confirm_preview_id` argument; each call "
                "grounds and executes from the current frame.",
            )
        hand = G1_MANIPULATION_HAND
        observation = context.get("observation")
        ego = (
            observation.views.get("ego")
            if observation is not None and observation.views
            else None
        )
        try:
            capture_id = int(ego.camera_params["capture_id"])
        except (AttributeError, KeyError, TypeError, ValueError):
            return tool_failure(
                call,
                FailureReason.BACKEND_UNAVAILABLE,
                "grasp_object requires a stamped current ego RGB-D capture.",
                requested_target=target,
            )
        try:
            candidates = ground_visible_candidates(
                semantic_backend=self._semantic_backend,
                observation=observation,
                target=target,
            )
        except (ObjectNotVisibleError, ObjectGeometryUnavailableError) as exc:
            return self._grounding_refusal(call, target=target, exc=exc)
        except EgoFrameUnavailableError as exc:
            return tool_failure(
                call,
                FailureReason.BACKEND_UNAVAILABLE,
                f"grasp_object cannot observe the current ego frame: {exc}.",
                requested_target=target,
            )
        except Exception as exc:  # noqa: BLE001 — backend defects stay visible
            return unexpected_tool_failure(call, exc)

        # Measure every detected candidate once; pick the highest-scoring one
        # with usable same-capture mask/depth. The labeled overlay of that
        # candidate is returned with the executed action for audit.
        measured: list[tuple[float, Any, Any]] = []
        unavailable: list[dict[str, Any]] = []
        for index, detection in enumerate(candidates.instances):
            try:
                grounding = measure_selected_instance(candidates, detection)
            except ObjectGeometryUnavailableError as exc:
                unavailable.append(
                    {"candidate_index": index, "reason": str(exc)},
                )
                continue
            measured.append((float(detection.score), grounding, detection))
        if not measured:
            return self._grounding_refusal(
                call,
                target=target,
                exc=ObjectGeometryUnavailableError(
                    "No detected candidate has usable same-capture mask/depth: "
                    f"{unavailable}",
                ),
            )
        measured.sort(key=lambda item: item[0], reverse=True)
        semantic_score, grounding, detection = measured[0]
        try:
            grounding_images, grounding_framing = _grasp_grounding_evidence(
                target=target,
                observation=observation,
                grounding=grounding,
                candidate_label="selected candidate",
            )
            if not grounding_images:
                raise RuntimeError("empty candidate overlay")
        except Exception as exc:  # noqa: BLE001
            return tool_failure(
                call,
                FailureReason.BACKEND_UNAVAILABLE,
                "grasp_object could not render the current-frame SAM3 "
                f"overlay: {type(exc).__name__}: {exc}",
                requested_target=target,
            )
        grounding_images = tuple(grounding_images)
        grounding_framing = str(grounding_framing)
        action_step = _action_step(context)
        bbox_2d = list(grounding.detection.box_xyxy)
        del detection

        nearest_target_distance_m = _nearest_target_horizontal_distance_from_robot_m(
            grounding=grounding,
            observation=observation,
        )
        if (
            nearest_target_distance_m is not None
            and nearest_target_distance_m > G1_GRASP_ABSOLUTE_MAX_HORIZONTAL_REACH_M
        ):
            return tool_failure(
                call,
                FailureReason.OUT_OF_REACH,
                f"grasp_object cannot reach {target!r} from this stance: its "
                f"nearest observed surface is {nearest_target_distance_m:.2f} "
                "m horizontally from the robot base, beyond the G1 right "
                f"arm absolute limit of "
                f"{G1_GRASP_ABSOLUTE_MAX_HORIZONTAL_REACH_M:.2f} m. Move "
                "closer while keeping the target visible, then retry.",
                requested_target=target,
                nearest_target_horizontal_distance_m=(nearest_target_distance_m),
                absolute_max_horizontal_reach_m=(
                    G1_GRASP_ABSOLUTE_MAX_HORIZONTAL_REACH_M
                ),
                graspgenx_skipped=True,
                images=grounding_images,
            )
        try:
            grasp_candidates = self._grasp_backend.propose(
                depth_m=grounding.depth_m,
                intrinsics=grounding.camera_intrinsics,
                target_mask=grounding.target_mask,
            )
        except NoGraspPoseError as exc:
            return tool_failure(
                call,
                FailureReason.NO_GRASP_CANDIDATE,
                (
                    f'GraspGenX found no usable native-hand pose for "{target}" '
                    f"in the current view: {exc} Change stance, distance, camera "
                    "view, or target visibility before retrying.\n"
                    f"{grounding_framing}"
                ),
                images=grounding_images,
                requested_target=target,
            )
        except Exception as exc:  # noqa: BLE001 — no geometric fallback
            return tool_failure(
                call,
                FailureReason.BACKEND_UNAVAILABLE,
                (
                    f"grasp_object could not obtain a GraspGenX pose: "
                    f"{type(exc).__name__}: {exc}\n{grounding_framing}"
                ),
                images=grounding_images,
                requested_target=target,
                exception_type=type(exc).__name__,
            )
        if not grasp_candidates:
            return tool_failure(
                call,
                FailureReason.BACKEND_UNAVAILABLE,
                (
                    "grasp_object received no GraspGenX poses; no substitute "
                    f"pose will be invented.\n{grounding_framing}"
                ),
                images=grounding_images,
                requested_target=target,
            )

        action = _build_grasp_action(
            target=target,
            hand=hand,
            grounding_mode=GRASP_GROUNDING_SAM3,
            target_center_world_m=grounding.target_center_world_m,
            target_center_camera_m=grounding.target_center_camera_m,
            bbox_2d=bbox_2d,
            semantic_score=semantic_score,
            world_T_camera=grounding.world_T_camera,
            capture_id=capture_id,
            candidates=grasp_candidates,
        )
        return ToolResult(
            call_id=call.call_id,
            name=call.tool,
            action=action,
            content=(
                f"Selected the highest-confidence SAM3 candidate for "
                f"{target!r} (score {semantic_score:.2f}; "
                f"{len(measured) - 1} other candidate(s) skipped). "
                f"GraspGenX found {len(grasp_candidates)} current-view pose(s); "
                "executing one closing-pose reach and measured closure. A "
                "retained grasp will then start the independent lift and "
                f"walking-posture plan within this decision.\n"
                f"{grounding_framing}"
            ),
            metadata={
                "target": target,
                "hand": hand,
                "target_center_world_m": list(grounding.target_center_world_m),
                "semantic_score": semantic_score,
                "graspgenx_pose_count": len(grasp_candidates),
                "grounding_mode": "fresh_current_rgb_with_private_depth",
            },
            images=grounding_images,
            deferred_effect=(
                ReplaceGroundingEffect(
                    memory=self._memory,
                    label=target,
                    grounding=grounding,
                    bbox=bbox_2d,
                    action_step=action_step,
                    capture_id=capture_id,
                )
                if self._memory is not None
                else None
            ),
        )

    def _grounding_refusal(
        self,
        call: ToolCall,
        *,
        target: str,
        exc: Exception,
    ) -> ToolResult:
        if isinstance(exc, ObjectNotVisibleError):
            return _grasp_grounding_failure(
                call,
                message=GRASP_FALLBACK_TARGET_NOT_IN_VIEW,
                error_type=FailureReason.NOT_IN_VIEW,
                requested_target=target,
            )
        return _grasp_grounding_failure(
            call,
            message=f"{GRASP_FALLBACK_VIEW_UNUSABLE} Detail: {exc}",
            error_type=FailureReason.DEPTH_UNAVAILABLE,
            requested_target=target,
        )


class _PointGoalValidator:
    """Validate one finite, non-zero, current-body-relative PointGoal."""

    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        del context

        args = call.args or {}
        try:
            forward_m = float(args["forward_m"])
            left_m = float(args["left_m"])
        except (KeyError, TypeError, ValueError) as exc:
            return tool_failure(
                call,
                FailureReason.MALFORMED_ARGUMENT,
                f"navigate_to_pointgoal requires numeric forward_m and left_m: {exc}.",
            )
        if not math.isfinite(forward_m) or not math.isfinite(left_m):
            return tool_failure(
                call,
                FailureReason.MALFORMED_ARGUMENT,
                "navigate_to_pointgoal coordinates must be finite metres.",
            )
        distance_m = math.hypot(forward_m, left_m)
        if distance_m == 0.0:
            return tool_failure(
                call,
                FailureReason.MALFORMED_ARGUMENT,
                "navigate_to_pointgoal requires a non-zero displacement from "
                "the current base.",
            )
        target_description = str(args.get("target_description", "")).strip()
        if not target_description:
            return tool_failure(
                call,
                FailureReason.MISSING_ARGUMENT,
                "navigate_to_pointgoal requires a concise target_description "
                "for the diagnostic trace.",
            )
        return ToolResult(
            call_id=call.call_id,
            name=call.tool,
            action=tool_turn_action(call.tool),
            content=(
                "Validated body-relative PointGoal at "
                f"{forward_m:+.2f} m forward and {left_m:+.2f} m left."
            ),
            metadata={
                "pointgoal_body_xy": [forward_m, left_m],
                "pointgoal_distance_m": distance_m,
                "target_description": target_description,
            },
        )


def _navigation_human_exclusions(
    context: Mapping[str, Any],
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[dict[str, Any], ...],
]:
    """Validate private live-Human context and bind physical center clearance."""

    raw = context.get("navigation_humans_world_xy")
    if raw is None:
        return (), ()
    if not isinstance(raw, Mapping):
        raise ValueError(
            "navigation_humans_world_xy must map Human IDs to live world XY."
        )
    zones: list[tuple[float, float, float]] = []
    evidence: list[dict[str, Any]] = []
    for raw_human_id in sorted(raw, key=str):
        human_id = str(raw_human_id).strip()
        position = np.asarray(raw[raw_human_id], dtype=np.float64).reshape(-1)
        if not human_id or position.size != 2 or not np.all(np.isfinite(position)):
            raise ValueError(
                "Every navigation Human requires a non-empty ID and exactly "
                "two finite live world coordinates."
            )
        zone = (
            float(position[0]),
            float(position[1]),
            ROBOT_HUMAN_CENTER_CLEARANCE_M,
        )
        zones.append(zone)
        evidence.append(
            {
                "human_id": human_id,
                "position_world_xy": [zone[0], zone[1]],
                "center_clearance_m": zone[2],
            }
        )
    return tuple(zones), tuple(evidence)


class NavigateToPointGoalHandler(_PointGoalValidator):
    """Prefer A*, then use X-NavDP when the partial map cannot route the goal."""

    def __init__(
        self,
        *,
        record_dir: str | None = None,
        x_navdp_backend: Any | None = None,
    ) -> None:
        self._record_dir = record_dir
        self._attempt_index = 0
        self._x_navdp_backend = x_navdp_backend

        # Run-owned arrival region. Ordinary navigation keeps the public
        # 0.50 m region; a GT reference may bind a stricter measured stance.
        self._goal_search_radius_m = 0.50
        self._exact_endpoint = False
        self._preserve_heading = False
        self._pending_turn_goal_xy: np.ndarray | None = None
        self._pending_turn_target_yaw_rad: float | None = None
        self._pending_turn_checkpoint_xy: np.ndarray | None = None

    def configure_goal_region(
        self,
        *,
        search_radius_m: float,
        exact_endpoint: bool = False,
        preserve_heading: bool = False,
    ) -> None:
        """Bind arrival size and optional resolved endpoint; never alter clearance."""
        radius = float(search_radius_m)
        if not np.isfinite(radius) or not 0.0 < radius <= 0.50:
            raise ValueError("Navigation goal radius must be in (0, 0.50] metres.")
        self._goal_search_radius_m = radius
        self._exact_endpoint = bool(exact_endpoint)
        self._preserve_heading = bool(preserve_heading)

    @property
    def x_navdp_backend(self) -> Any:
        """Same episode-scoped policy used by the runtime closed-loop executor."""
        return self._x_navdp_backend

    def reset(self) -> None:
        self._attempt_index = 0
        self._exact_endpoint = False
        self._preserve_heading = False
        self._goal_search_radius_m = 0.50
        self._pending_turn_goal_xy = None
        self._pending_turn_target_yaw_rad = None
        self._pending_turn_checkpoint_xy = None
        reset = getattr(self._x_navdp_backend, "reset", None)
        if callable(reset):
            reset()

    def configure_episode(self, episode: Any) -> None:
        configure = getattr(self._x_navdp_backend, "configure_episode", None)
        if callable(configure):
            configure(episode)

    def _record_attempt(
        self,
        *,
        grid: Any,
        record_dir: str | None,
        action_step: int,
        status: str,
        failure_reason: str | None,
        target_description: str,
        pointgoal_body_xy: Sequence[float],
        robot_xy: np.ndarray,
        requested_world_xy: np.ndarray,
        goal_world_xy: np.ndarray | None,
        path: list[np.ndarray] | None,
        human_exclusions: tuple[dict[str, Any], ...],
    ) -> tuple[str | None, str | None]:
        if not record_dir:
            return None, None
        import json
        import os

        stem = (
            f"pointgoal_map_step_{int(action_step):04d}_"
            f"attempt_{self._attempt_index:02d}"
        )
        self._attempt_index += 1
        map_path = os.path.join(str(record_dir), f"{stem}.png")
        evidence_path = os.path.join(str(record_dir), f"{stem}.json")
        state_path = os.path.join(str(record_dir), f"{stem}.npz")
        grid.dump_png(
            map_path,
            robot_xy=robot_xy,
            path_waypoints=path,
            selected_xy=requested_world_xy,
            goal_xy=goal_world_xy,
            include_planning_buffer=True,
        )
        grid.dump_npz(state_path)
        with open(evidence_path, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "action_step": int(action_step),
                    "status": status,
                    "failure_reason": failure_reason,
                    "target_description": target_description,
                    "pointgoal_body_xy": [
                        float(pointgoal_body_xy[0]),
                        float(pointgoal_body_xy[1]),
                    ],
                    "robot_world_xy": robot_xy.tolist(),
                    "requested_world_xy": requested_world_xy.tolist(),
                    "safe_goal_world_xy": (
                        goal_world_xy.tolist() if goal_world_xy is not None else None
                    ),
                    "astar_path_world_xy": (
                        [np.asarray(point).tolist() for point in path]
                        if path is not None
                        else None
                    ),
                    "navigation_map_source": str(grid.map_source),
                    "dynamic_human_exclusions": list(human_exclusions),
                    "goal_resolution": "exact_endpoint"
                    if self._exact_endpoint
                    else "goal_region",
                    "online_map_state_path": state_path,
                    "map_coverage": grid.coverage_stats(),
                },
                stream,
                indent=2,
                sort_keys=True,
            )
        return map_path, evidence_path

    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        return self.execute_navigation(
            call,
            context,
            lateral_tolerance_m=0.25,
            allow_navdp=True,
        )

    def execute_navigation(
        self,
        call: ToolCall,
        context: Dict[str, Any],
        *,
        lateral_tolerance_m: float | None,
        allow_navdp: bool,
    ) -> ToolResult:
        """Shared physical prefix; geometry/fallback policy belongs to the tool."""
        import math

        from tarsphere.infra.control.relative_pose import (
            world_xy_from_body_displacement,
        )
        from tarsphere.infra.perception.online_mapping import (
            OnlineOccupancyGrid,
        )

        validated = super().execute(call, context)
        if validated.is_error:
            return validated
        metadata = validated.metadata
        pointgoal = np.asarray(metadata["pointgoal_body_xy"], dtype=np.float64)
        target_description = str(metadata["target_description"])
        observation = context.get("observation")
        pose = getattr(observation, "pose", None)
        if not isinstance(pose, dict) or pose.get("position") is None:
            return tool_failure(
                call,
                FailureReason.BACKEND_UNAVAILABLE,
                "navigate_to_pointgoal: robot pose is unavailable.",
            )
        position = np.asarray(pose["position"], dtype=np.float64).reshape(-1)
        if position.size < 2 or not np.all(np.isfinite(position[:2])):
            return tool_failure(
                call,
                FailureReason.BACKEND_UNAVAILABLE,
                "navigate_to_pointgoal: robot XY is non-finite.",
            )
        yaw = float(pose.get("yaw_rad", 0.0))
        if not math.isfinite(yaw):
            return tool_failure(
                call,
                FailureReason.BACKEND_UNAVAILABLE,
                "navigate_to_pointgoal: robot yaw is non-finite.",
            )
        robot_xy = position[:2].copy()
        requested_world_xy = np.asarray(
            world_xy_from_body_displacement(
                displacement_forward_left_m=pointgoal,
                body_world_xy=robot_xy,
                body_yaw_rad=yaw,
            )
        )
        if (
            self._pending_turn_goal_xy is not None
            and np.linalg.norm(requested_world_xy - self._pending_turn_goal_xy)
            > LOCOMOTION_MIN_STEP_DISTANCE_M
        ):
            # A different PointGoal invalidates an unfinished corner turn.
            self._pending_turn_goal_xy = None
            self._pending_turn_target_yaw_rad = None
            self._pending_turn_checkpoint_xy = None
        mapping_error = context.get("online_map_error")
        if mapping_error:
            return tool_failure(
                call,
                FailureReason.BACKEND_UNAVAILABLE,
                "navigate_to_pointgoal: the latest synchronized observation "
                f"could not update the online map ({mapping_error}). Refusing "
                "to plan against stale geometry.",
            )
        grid = context.get("online_occupancy_grid")
        if not isinstance(grid, OnlineOccupancyGrid):
            return tool_failure(
                call,
                FailureReason.BACKEND_UNAVAILABLE,
                "navigate_to_pointgoal: the selected occupancy backend is "
                "not initialized.",
            )
        if not math.isclose(
            grid.robot_radius_m,
            G1_NAVIGATION_FOOTPRINT_RADIUS_M,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            return tool_failure(
                call,
                FailureReason.BACKEND_UNAVAILABLE,
                "navigate_to_pointgoal: occupancy clearance does not match "
                "the G1 footprint.",
            )
        human_exclusion_zones, human_exclusions = _navigation_human_exclusions(context)
        request_planner = (
            grid._planner(world_exclusion_zones=human_exclusion_zones)
            if human_exclusion_zones
            else grid._planner()
        )

        if self._exact_endpoint:
            # A reference prefix has already been resolved by the same A*.
            # Re-selecting a region here compounds snapping and may report
            # arrival before the caller's measured stance is valid.
            if human_exclusion_zones:
                path, astar_failure = grid.astar_with_reason(
                    robot_xy,
                    requested_world_xy,
                    world_exclusion_zones=human_exclusion_zones,
                )
            else:
                path, astar_failure = grid.astar_with_reason(
                    robot_xy,
                    requested_world_xy,
                )
            goal_world_xy = requested_world_xy.copy() if path is not None else None
        else:
            astar_kwargs = {
                "lateral_tolerance_m": lateral_tolerance_m,
                "search_radius_m": self._goal_search_radius_m,
            }
            if human_exclusion_zones:
                astar_kwargs["world_exclusion_zones"] = human_exclusion_zones
            path, goal_world_xy, astar_failure = grid.astar_to_goal_region(
                robot_xy,
                requested_world_xy,
                **astar_kwargs,
            )
        # The exact endpoint is an already selected short execution prefix.
        # A second A* can reintroduce a tiny soft-cost corner and force a
        # table-side strafe. Preserve the direct chord whenever the unchanged
        # hard embodiment mask clears it; obstacle intersections stay rejected.
        if (
            self._exact_endpoint
            and path is not None
            and len(path) > 2
            and np.linalg.norm(requested_world_xy - robot_xy)
            <= min(
                LOCOMOTION_MAX_BACKWARD_DISTANCE_M, LOCOMOTION_MAX_LATERAL_DISTANCE_M
            )
            and request_planner.is_line_collision_free(
                robot_xy,
                requested_world_xy,
            )
        ):
            path = [robot_xy.copy(), requested_world_xy.copy()]
        # A* may snap an obstructed start to a free cell for geometric search.
        # That does not validate a physical departure from the measured pose.
        # Apply this before turns, translation, and the already-reached case.
        buffer_exit = None
        buffer_exit_destination = None
        if not grid.is_navigable(robot_xy):
            # Prefer a legal exit that advances the unchanged A* request. A
            # fixed angular scan can otherwise choose the opposite side of a
            # narrow aisle on every closed-loop replan and undo real route
            # progress. The grid still proves raw-cell and footprint clearance
            # before returning any endpoint.
            buffer_exit_destination = grid.clearance_buffer_exit(
                robot_xy,
                preferred_goal_xy=requested_world_xy,
            )
            buffer_exit = buffer_exit_destination
            if buffer_exit is None:
                path, goal_world_xy, astar_failure = (
                    None,
                    None,
                    "measured_start_obstructed",
                )
            else:
                # A* cannot start inside its inflated buffer. Execute only one
                # measured relative-pose tolerance at a time toward the proven
                # exit, then re-observe. A single long lateral/backward command
                # from contact can amplify centimetre-scale gait drift into a
                # fall even though the centre-line geometry is valid.
                segment = buffer_exit - robot_xy
                length = float(np.linalg.norm(segment))
                executable_exit_step_m = max(
                    G1_RELATIVE_POSE_LATERAL_TRANSLATION_TOLERANCE_M,
                    LOCOMOTION_MIN_STEP_DISTANCE_M,
                )
                if length > executable_exit_step_m:
                    buffer_exit = robot_xy + segment * (executable_exit_step_m / length)
                path, goal_world_xy, astar_failure = (
                    [robot_xy, buffer_exit],
                    buffer_exit,
                    None,
                )
        elif human_exclusion_zones and not request_planner.is_line_collision_free(
            robot_xy, robot_xy
        ):
            path, goal_world_xy, astar_failure = (
                None,
                None,
                "measured_start_inside_human_clearance",
            )
        elif (
            path is not None
            and len(path) >= 2
            and not request_planner.is_line_collision_free(
                robot_xy, np.asarray(path[1], dtype=np.float64)
            )
        ):
            path, goal_world_xy, astar_failure = (
                None,
                None,
                "measured_departure_obstructed",
            )
        pending_checkpoint = None
        pending_turn_error = None
        if buffer_exit is None and self._pending_turn_target_yaw_rad is not None:
            if grid.can_turn_in_place(robot_xy):
                pending_turn_error = (
                    self._pending_turn_target_yaw_rad - yaw + math.pi
                ) % (2 * math.pi) - math.pi
                if abs(pending_turn_error) <= NAVIGATION_PATH_ENTRY_YAW_TOLERANCE_RAD:
                    self._pending_turn_goal_xy = None
                    self._pending_turn_target_yaw_rad = None
                    self._pending_turn_checkpoint_xy = None
                    pending_turn_error = None
                elif path is None:
                    # This turn was already justified by the previous
                    # collision-checked checkpoint. A transient full-route
                    # miss must not erase the executable pending action.
                    path = [robot_xy.copy(), robot_xy.copy()]
                    goal_world_xy, astar_failure = requested_world_xy.copy(), None
            elif self._pending_turn_checkpoint_xy is not None:
                checkpoint = self._pending_turn_checkpoint_xy
                segment = checkpoint - robot_xy
                checkpoint_distance_m = float(np.linalg.norm(segment))
                if 1e-9 < checkpoint_distance_m < LOCOMOTION_MIN_STEP_DISTANCE_M:
                    # A checkpoint is an internal safety waypoint, not an
                    # endpoint contract.  G1 cannot reliably execute a final
                    # centimetre-scale translation, so do not keep emitting a
                    # sub-deadband command while the measured body still lacks
                    # turn clearance.  Extend the same proven escape ray to
                    # the first collision-checked pose where a turn is legal.
                    direction = segment / checkpoint_distance_m
                    max_extension_m = min(
                        LOCOMOTION_MAX_BACKWARD_DISTANCE_M,
                        LOCOMOTION_MAX_LATERAL_DISTANCE_M,
                    )
                    extended_checkpoint = None
                    for distance_m in np.arange(
                        LOCOMOTION_MIN_STEP_DISTANCE_M,
                        max_extension_m + grid.resolution * 0.5,
                        grid.resolution,
                    ):
                        candidate = robot_xy + direction * distance_m
                        if not (
                            request_planner.is_line_collision_free(
                                robot_xy,
                                candidate,
                            )
                            and grid.translation_sweep_clear(
                                robot_xy,
                                candidate,
                                yaw,
                            )
                        ):
                            break
                        if grid.can_turn_in_place(candidate):
                            extended_checkpoint = candidate
                            break
                    if extended_checkpoint is not None:
                        checkpoint = extended_checkpoint
                        self._pending_turn_checkpoint_xy = checkpoint.copy()
                        segment = checkpoint - robot_xy
                    else:
                        # The old checkpoint cannot produce an executable
                        # continuation from the measured pose.  Drop only this
                        # internal hint and let the unchanged A* path choose a
                        # fresh collision-checked departure below.
                        self._pending_turn_goal_xy = None
                        self._pending_turn_target_yaw_rad = None
                        self._pending_turn_checkpoint_xy = None
                        segment = np.zeros(2, dtype=np.float64)
                if (
                    np.linalg.norm(segment) > 1e-9
                    and request_planner.is_line_collision_free(robot_xy, checkpoint)
                    and grid.translation_sweep_clear(robot_xy, checkpoint, yaw)
                ):
                    pending_checkpoint = checkpoint
                    path = [robot_xy.copy(), checkpoint.copy()]
                    goal_world_xy, astar_failure = checkpoint.copy(), None
        action_step = int(context.get("action_step", 0) or 0)
        record_dir = self._record_dir or context.get("record_dir")
        if path is None:
            map_path, evidence_path = self._record_attempt(
                grid=grid,
                record_dir=record_dir,
                action_step=action_step,
                status="no_route",
                failure_reason=astar_failure,
                target_description=target_description,
                pointgoal_body_xy=pointgoal,
                robot_xy=robot_xy,
                requested_world_xy=requested_world_xy,
                goal_world_xy=None,
                path=None,
                human_exclusions=human_exclusions,
            )
            failure_metadata = {
                **metadata,
                "goal_resolution": "exact_endpoint"
                if self._exact_endpoint
                else "goal_region",
                "requested_world_xy": requested_world_xy.tolist(),
                "goal_world_xy": None,
                "navigation_map_source": str(grid.map_source),
                "dynamic_human_exclusions": list(human_exclusions),
                "online_map_path": map_path,
                "online_map_evidence_path": evidence_path,
            }
            if (
                allow_navdp
                and not self._exact_endpoint
                and not human_exclusion_zones
                and astar_failure
                not in {"measured_start_obstructed", "measured_departure_obstructed"}
                and self._x_navdp_backend is not None
            ):
                return _prepare_x_navdp_navigation(
                    call=call,
                    context=context,
                    pointgoal_body_xy=pointgoal,
                    target_description=target_description,
                    backend_name="x_navdp_fallback",
                    fallback_reason=f"astar_{astar_failure}",
                    extra_command={
                        "astar_requested_world_xy": requested_world_xy.tolist(),
                        "astar_goal_world_xy": None,
                        "astar_map_source": str(grid.map_source),
                    },
                    extra_metadata=failure_metadata,
                )
            return ToolResult(
                call_id=call.call_id,
                name=call.tool,
                action=tool_turn_action(call.tool),
                content=(
                    f"No A* route to the allowed ground region within {self._goal_search_radius_m:.2f} m "
                    f"of {target_description!r} ({astar_failure}). "
                    "No movement executed. Choose an intermediate goal using "
                    "current observations; the target location was not changed."
                ),
                metadata={
                    **failure_metadata,
                    "planning_status": "no_route",
                    "planning_reason": astar_failure,
                },
            )

        map_path, evidence_path = self._record_attempt(
            grid=grid,
            record_dir=record_dir,
            action_step=action_step,
            status="planned",
            failure_reason=None,
            target_description=target_description,
            pointgoal_body_xy=pointgoal,
            robot_xy=robot_xy,
            requested_world_xy=requested_world_xy,
            goal_world_xy=goal_world_xy,
            path=path,
            human_exclusions=human_exclusions,
        )
        remaining_m = float(
            sum(
                np.linalg.norm(np.asarray(end) - np.asarray(start))
                for start, end in zip(path, path[1:])
            )
        )
        # A strict stance region must also narrow the controller deadband.
        # Otherwise a 6 cm snap plus the ordinary 10 cm stop can leave a
        # reference outside its 12 cm measured arrival criterion forever.
        # This is a collision-checked A* segment, not a public move_body call.
        stop_distance_m = min(
            LOCOMOTION_MIN_STEP_DISTANCE_M, self._goal_search_radius_m
        )
        if pending_checkpoint is not None:
            segment = pending_checkpoint - robot_xy
            distance_m = min(
                float(np.linalg.norm(segment)),
                min(
                    LOCOMOTION_MAX_BACKWARD_DISTANCE_M,
                    LOCOMOTION_MAX_LATERAL_DISTANCE_M,
                ),
            )
            bounded_segment = segment * (distance_m / np.linalg.norm(segment))
            phase = "pending_turn_checkpoint"
            endpoint_margin_m = 0.0
            action = {
                "action": "move_body",
                "delta": [
                    math.cos(yaw) * bounded_segment[0]
                    + math.sin(yaw) * bounded_segment[1],
                    -math.sin(yaw) * bounded_segment[0]
                    + math.cos(yaw) * bounded_segment[1],
                    0.0,
                ],
            }
        elif pending_turn_error is not None:
            # Complete the turn for which the prior decision deliberately
            # translated to a body-clear checkpoint. Replanning A* first can
            # pull the robot straight back into the narrow corner and repeat
            # the same translation forever.
            phase = "pending_turn_alignment"
            endpoint_margin_m = 0.0
            action = {
                "action": "move_body",
                "delta": [
                    0.0,
                    0.0,
                    float(
                        np.clip(
                            math.degrees(pending_turn_error),
                            -min(
                                LOCOMOTION_MAX_TURN_DEG,
                                math.degrees(NAVIGATION_SEGMENT_MAX_TURN_RAD),
                            ),
                            min(
                                LOCOMOTION_MAX_TURN_DEG,
                                math.degrees(NAVIGATION_SEGMENT_MAX_TURN_RAD),
                            ),
                        )
                    ),
                ],
            }
        elif buffer_exit is not None:
            segment = buffer_exit - robot_xy
            phase = "clearance_buffer_exit"
            endpoint_margin_m = 0.0
            action = {
                "action": "move_body",
                "delta": [
                    math.cos(yaw) * segment[0] + math.sin(yaw) * segment[1],
                    -math.sin(yaw) * segment[0] + math.cos(yaw) * segment[1],
                    0.0,
                ],
            }
        elif len(path) < 2 or remaining_m < stop_distance_m:
            action = tool_turn_action(call.tool)
            phase = "already_reached"
            endpoint_margin_m = 0.0
        else:
            next_waypoint = np.asarray(path[1], dtype=np.float64)
            segment = next_waypoint - robot_xy
            segment_distance_m = float(np.linalg.norm(segment))
            target_yaw = math.atan2(float(segment[1]), float(segment[0]))
            heading_error = (target_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
            # Near obstacles, even a long first straight segment should start
            # with a bounded sideways/backward prefix, not an in-place turn.
            # Reassess measured clearance after it; turn normally in open space.
            # Never skip a corner or use the chord to the ultimate destination.
            translation_limit_m = min(
                LOCOMOTION_MAX_BACKWARD_DISTANCE_M,
                LOCOMOTION_MAX_LATERAL_DISTANCE_M,
            )
            turn_clear = grid.can_turn_in_place(robot_xy)
            bounded_path_translation = (
                abs(heading_error) > NAVIGATION_PATH_ENTRY_YAW_TOLERANCE_RAD
                # The stop radius applies to the whole route, not an
                # intermediate corner. A short first segment must not force
                # an otherwise unsafe in-place turn.
                and segment_distance_m > 1e-9
                # An ordinary route must align once it reaches clear space,
                # including after moving out to a turn checkpoint. Otherwise
                # short A* prefixes can pull it back into the same corner.
                and (not turn_clear or self._preserve_heading)
            )
            if (
                abs(heading_error) > NAVIGATION_PATH_ENTRY_YAW_TOLERANCE_RAD
                and not bounded_path_translation
            ):
                phase = "heading_alignment"
                endpoint_margin_m = 0.0
                action = {
                    "action": "move_body",
                    "delta": [
                        0.0,
                        0.0,
                        float(
                            np.clip(
                                math.degrees(heading_error),
                                -min(
                                    LOCOMOTION_MAX_TURN_DEG,
                                    math.degrees(NAVIGATION_SEGMENT_MAX_TURN_RAD),
                                ),
                                min(
                                    LOCOMOTION_MAX_TURN_DEG,
                                    math.degrees(NAVIGATION_SEGMENT_MAX_TURN_RAD),
                                ),
                            )
                        ),
                    ],
                }
            else:
                phase = "forward_progress"
                if bounded_path_translation:
                    phase = (
                        "terminal_translation" if len(path) == 2 else "path_translation"
                    )
                # Preserve the collision-checked world segment. A tolerated
                # heading error is not permission to replace it with the
                # current body-forward ray (which may cut a corner). G1's
                # shared executor accepts a continuous XY body delta.
                distance_m = min(
                    segment_distance_m,
                    translation_limit_m
                    if bounded_path_translation
                    else LOCOMOTION_MAX_BODY_FORWARD_M,
                )
                endpoint_margin_m = 0.0
                if (
                    not (self._exact_endpoint and len(path) == 2)
                    and distance_m == segment_distance_m
                    and segment_distance_m
                    >= LOCOMOTION_MIN_STEP_DISTANCE_M
                    + POINTGOAL_WAYPOINT_ENDPOINT_MARGIN_M
                ):
                    distance_m = max(
                        stop_distance_m,
                        distance_m - POINTGOAL_WAYPOINT_ENDPOINT_MARGIN_M,
                    )
                    endpoint_margin_m = POINTGOAL_WAYPOINT_ENDPOINT_MARGIN_M
                action = {
                    "action": "move_body",
                    "delta": [
                        distance_m * math.cos(heading_error),
                        distance_m * math.sin(heading_error),
                        0.0,
                    ],
                }
        body_clearance_preference = "not_requested"
        if (
            buffer_exit is None
            and phase != "pending_turn_checkpoint"
            and not self._preserve_heading
            and action["action"] == "move_body"
            and action["delta"][2] == 0.0
        ):
            forward, left = action["delta"][:2]
            end = robot_xy + np.array(
                [
                    math.cos(yaw) * forward - math.sin(yaw) * left,
                    math.sin(yaw) * forward + math.cos(yaw) * left,
                ]
            )
            body_clearance_preference = "sweep_clear"
            if not grid.translation_sweep_clear(robot_xy, end, yaw):
                error = (
                    math.atan2(end[1] - robot_xy[1], end[0] - robot_xy[0])
                    - yaw
                    + math.pi
                ) % (2 * math.pi) - math.pi
                checkpoint = None
                turn_clear = grid.can_turn_in_place(robot_xy)
                if turn_clear and abs(error) > NAVIGATION_PATH_ENTRY_YAW_TOLERANCE_RAD:
                    phase = "heading_alignment"
                    action = {
                        "action": "move_body",
                        "delta": [
                            0.0,
                            0.0,
                            float(
                                np.clip(
                                    math.degrees(error),
                                    -math.degrees(NAVIGATION_SEGMENT_MAX_TURN_RAD),
                                    math.degrees(NAVIGATION_SEGMENT_MAX_TURN_RAD),
                                )
                            ),
                        ],
                    }
                    body_clearance_preference = "align_before_body_sweep"
                elif not turn_clear:
                    checkpoint = grid.translation_turn_checkpoint(
                        robot_xy, yaw, goal_world_xy
                    )
                if checkpoint is not None and request_planner.is_line_collision_free(
                    robot_xy,
                    checkpoint,
                ):
                    shift = checkpoint - robot_xy
                    action = {
                        "action": "move_body",
                        "delta": [
                            math.cos(yaw) * shift[0] + math.sin(yaw) * shift[1],
                            -math.sin(yaw) * shift[0] + math.cos(yaw) * shift[1],
                            0.0,
                        ],
                    }
                    phase = "turn_clearance_translation"
                    body_clearance_preference = "physical_turn_checkpoint"
                    self._pending_turn_goal_xy = requested_world_xy.copy()
                    self._pending_turn_target_yaw_rad = math.atan2(
                        float(end[1] - robot_xy[1]),
                        float(end[0] - robot_xy[0]),
                    )
                    self._pending_turn_checkpoint_xy = checkpoint.copy()
                elif body_clearance_preference != "align_before_body_sweep":
                    body_clearance_preference = (
                        "no_clearer_alternative_original_footprint_action"
                    )
                if body_clearance_preference in {
                    "align_before_body_sweep",
                    "physical_turn_checkpoint",
                }:
                    endpoint_margin_m = 0.0
        action["navigation_command"] = {
            "backend": "astar_pointgoal",
            "target_description": target_description,
            "pointgoal_body_xy": pointgoal.tolist(),
            "pointgoal_distance_m": float(np.linalg.norm(pointgoal)),
            "command_body_delta": list(action.get("delta") or [0.0, 0.0, 0.0]),
            "phase": phase,
            "planned_remaining_m": remaining_m,
            "map_source": str(grid.map_source),
            "dynamic_human_exclusion_count": len(human_exclusions),
        }
        return ToolResult(
            call_id=call.call_id,
            name=call.tool,
            action=action,
            content=(
                f"A* resolved the relative PointGoal for "
                f"{target_description!r}; phase={phase}, "
                f"path remaining={remaining_m:.2f} m. Reassess from the fresh "
                "observation after this physical step."
            ),
            metadata={
                **metadata,
                "navigation_backend": "astar_pointgoal",
                "goal_search_radius_m": self._goal_search_radius_m,
                "goal_resolution": (
                    "clearance_buffer_exit"
                    if buffer_exit is not None
                    else "exact_endpoint"
                    if self._exact_endpoint
                    else "goal_region"
                ),
                "requested_world_xy": requested_world_xy.tolist(),
                "goal_world_xy": goal_world_xy.tolist(),
                "goal_was_snapped": bool(
                    not np.allclose(requested_world_xy, goal_world_xy)
                    and buffer_exit is None
                ),
                "remaining_m": remaining_m,
                "navigation_motion_phase": phase,
                "waypoint_endpoint_margin_m": endpoint_margin_m,
                "body_clearance_preference": body_clearance_preference,
                "clearance_buffer_destination_world_xy": (
                    buffer_exit_destination.tolist()
                    if buffer_exit_destination is not None
                    else None
                ),
                "preserve_heading": self._preserve_heading,
                "navigation_map_source": str(grid.map_source),
                "dynamic_human_exclusions": list(human_exclusions),
                "online_map_path": map_path,
                "online_map_evidence_path": evidence_path,
            },
        )


def _delegated_navigation_payload(
    call: ToolCall,
    delegated: ToolResult,
) -> tuple[dict[str, Any] | None, tuple[dict[str, Any], ...]]:
    """Keep a controller command; rewrite in-handler results to this tool."""

    if delegated.actions:
        return None, delegated.actions
    if delegated.action is None:
        raise RuntimeError(f"{call.tool} delegated a result without a runtime payload")
    payload = dict(delegated.action)
    if requires_environment_execution(payload):
        return payload, ()
    identity = tool_turn_action(call.tool)
    command = payload.get("navigation_command")
    if isinstance(command, dict):
        identity["navigation_command"] = command
    if delegated.is_error:
        identity["is_error"] = True
    return identity, ()


class NavigateToEntityHandler:
    """Resolve a saved world landmark into the current body frame and use A*."""

    def __init__(
        self,
        *,
        memory: Any,
        pointgoal_handler: NavigateToPointGoalHandler,
    ) -> None:
        self._memory = memory
        self._pointgoal_handler = pointgoal_handler

    def reset(self) -> None:
        self._pointgoal_handler.reset()

    def configure_episode(self, episode: Any) -> None:
        self._pointgoal_handler.configure_episode(episode)

    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        entity_id = str(call.args.get("entity_id") or "").strip()
        record = self._memory.semantic.get(entity_id)
        if record is None:
            return tool_failure(
                call,
                FailureReason.UNKNOWN_VALUE,
                f"navigate_to_entity has no saved entity {entity_id!r}.",
            )
        observation = context.get("observation")
        pose = getattr(observation, "pose", None)
        if not isinstance(pose, dict) or pose.get("position") is None:
            return tool_failure(
                call,
                FailureReason.BACKEND_UNAVAILABLE,
                "navigate_to_entity requires the current robot pose.",
            )
        position = np.asarray(pose["position"], dtype=np.float64).reshape(-1)
        if position.size < 2 or not np.all(np.isfinite(position[:2])):
            return tool_failure(
                call,
                FailureReason.BACKEND_UNAVAILABLE,
                "navigate_to_entity received no finite robot XY.",
            )
        yaw = float(pose.get("yaw_rad", 0.0))
        if not np.isfinite(yaw):
            return tool_failure(
                call,
                FailureReason.BACKEND_UNAVAILABLE,
                "navigate_to_entity received no finite robot yaw.",
            )
        delta_world = np.asarray(record.world_xyz[:2]) - position[:2]
        cosine = float(np.cos(yaw))
        sine = float(np.sin(yaw))
        forward_m = cosine * float(delta_world[0]) + sine * float(delta_world[1])
        left_m = -sine * float(delta_world[0]) + cosine * float(delta_world[1])
        if float(np.linalg.norm(delta_world)) < 1e-6:
            return ToolResult(
                call_id=call.call_id,
                name=call.tool,
                action=tool_turn_action(call.tool),
                content=(
                    f"navigate_to_entity: already at saved landmark {entity_id!r}; "
                    "use fresh visual evidence to verify the current target."
                ),
                metadata={"entity_id": entity_id, "already_reached": True},
            )
        delegated_call = ToolCall(
            call_id=call.call_id,
            tool="navigate_to_pointgoal",
            args={
                "forward_m": forward_m,
                "left_m": left_m,
                "target_description": (
                    f"saved {record.entity_type} landmark {entity_id}: {record.label}"
                ),
            },
        )
        delegated = self._pointgoal_handler.execute_navigation(
            delegated_call,
            context,
            lateral_tolerance_m=None,
            allow_navdp=False,
        )
        action, actions = _delegated_navigation_payload(call, delegated)
        return ToolResult(
            call_id=call.call_id,
            name=call.tool,
            content=delegated.content,
            action=action,
            actions=actions,
            is_error=delegated.is_error,
            metadata={
                **dict(delegated.metadata or {}),
                "entity_id": entity_id,
                "entity_type": record.entity_type,
                "landmark_observed_action_step": record.observed_action_step,
            },
            images=delegated.images,
            deferred_effect=delegated.deferred_effect,
        )


class NavigateToFrontierHandler:
    """Resolve one current visual frontier through the shared PointGoal path."""

    def __init__(
        self,
        backend: Any,
        *,
        pointgoal_handler: NavigateToPointGoalHandler,
    ) -> None:
        self._backend = backend
        self._pointgoal_handler = pointgoal_handler

    @property
    def backend(self) -> Any:
        return self._backend

    def reset(self) -> None:
        self._backend.reset()

    def configure_episode(self, episode: Any) -> None:
        configure = getattr(self._backend, "configure_episode", None)
        if callable(configure):
            configure(episode)

    def execute(self, call: ToolCall, context: Dict[str, Any]) -> ToolResult:
        label = str(call.args.get("label", "")).strip().upper()
        description = str(call.args.get("target_description", "")).strip()
        if label not in {"A", "B", "C"} or not description:
            return tool_failure(
                call,
                FailureReason.MALFORMED_ARGUMENT,
                "navigate_to_frontier requires a current A/B/C label and a "
                "non-empty target_description.",
            )
        observation = context.get("observation")
        pose = getattr(observation, "pose", None)
        if not isinstance(pose, dict) or pose.get("position") is None:
            return tool_failure(
                call,
                FailureReason.BACKEND_UNAVAILABLE,
                "navigate_to_frontier requires the current robot pose.",
            )
        position = np.asarray(pose["position"], dtype=np.float64).reshape(-1)
        if position.size < 2 or not np.all(np.isfinite(position[:2])):
            return tool_failure(
                call,
                FailureReason.BACKEND_UNAVAILABLE,
                "navigate_to_frontier requires finite current robot XY.",
            )
        robot_xy = position[:2]
        yaw = float(pose.get("yaw_rad", 0.0))
        if not np.isfinite(yaw):
            return tool_failure(
                call,
                FailureReason.BACKEND_UNAVAILABLE,
                "navigate_to_frontier requires finite current robot yaw.",
            )
        action_step = int(context.get("action_step", 0) or 0)
        try:
            frontier = self._backend.latest_frontier(
                label=label,
                action_step=action_step,
            )
        except (KeyError, ValueError) as exc:
            return tool_failure(
                call,
                FailureReason.MALFORMED_ARGUMENT,
                f"navigate_to_frontier rejected label {label!r}: {exc}",
            )
        target_world_xy = np.asarray(frontier.world_xyz[:2], dtype=np.float64)
        # Preserve the visual anchor. The shared PointGoal handler owns the
        # corridor-constrained goal-set search and records both endpoints.
        delta_world = target_world_xy - robot_xy
        cosine = float(np.cos(yaw))
        sine = float(np.sin(yaw))
        pointgoal_body_xy = [
            cosine * float(delta_world[0]) + sine * float(delta_world[1]),
            -sine * float(delta_world[0]) + cosine * float(delta_world[1]),
        ]
        delegated = self._pointgoal_handler.execute(
            ToolCall(
                call_id=call.call_id,
                tool="navigate_to_pointgoal",
                args={
                    "forward_m": pointgoal_body_xy[0],
                    "left_m": pointgoal_body_xy[1],
                    "target_description": description,
                },
            ),
            context,
        )
        action, actions = _delegated_navigation_payload(call, delegated)
        if action is not None:
            navigation_command = action.get("navigation_command")
            if isinstance(navigation_command, dict):
                navigation_command = dict(navigation_command)
                navigation_command.update(
                    {
                        "frontier_label": label,
                        "frontier_id": frontier.frontier_id,
                        "frontier_information_gain_m3": float(
                            frontier.information_gain_m3
                        ),
                    }
                )
                action["navigation_command"] = navigation_command
        return ToolResult(
            call_id=call.call_id,
            name=call.tool,
            action=action,
            actions=actions,
            is_error=delegated.is_error,
            content=(
                f"OpenFrontier selected current candidate {label}; {delegated.content}"
            ),
            metadata={
                **dict(delegated.metadata or {}),
                "frontier_label": label,
                "frontier_id": frontier.frontier_id,
                "frontier_information_gain_m3": float(frontier.information_gain_m3),
                "target_description": description,
            },
            images=delegated.images,
            deferred_effect=delegated.deferred_effect,
        )


def _prepare_x_navdp_navigation(
    *,
    call: ToolCall,
    context: Dict[str, Any],
    pointgoal_body_xy: Any,
    target_description: str,
    backend_name: str,
    fallback_reason: str | None = None,
    extra_command: Dict[str, Any] | None = None,
    extra_metadata: Dict[str, Any] | None = None,
) -> ToolResult:
    """Prepare one grounded PointGoal; inference belongs to physical execution."""

    pointgoal = np.asarray(pointgoal_body_xy, dtype=np.float64).reshape(-1)
    if pointgoal.size != 2 or not np.all(np.isfinite(pointgoal)):
        return tool_failure(
            call,
            FailureReason.MALFORMED_ARGUMENT,
            f"{call.tool}: X-NavDP requires two finite PointGoal metres.",
        )
    pointgoal_norm_m = float(np.linalg.norm(pointgoal))
    if pointgoal_norm_m > _XNAVDP_MAX_OFFICIAL_POINTGOAL_NORM_M:
        return tool_failure(
            call,
            FailureReason.MALFORMED_ARGUMENT,
            f"{call.tool}: the official X-NavDP backend accepts PointGoals up "
            "to 25 m; choose a grounded intermediate destination.",
        )
    observation = context.get("observation")
    if observation is None:
        return tool_failure(
            call,
            FailureReason.BACKEND_UNAVAILABLE,
            f"{call.tool}: current synchronized observation is unavailable to X-NavDP.",
        )
    navigation_command = {
        "backend": backend_name,
        "target_description": str(target_description),
        "pointgoal_body_xy": pointgoal.tolist(),
        "pointgoal_distance_m": pointgoal_norm_m,
        "phase": "closed_loop_navigation",
    }
    if fallback_reason is not None:
        navigation_command["fallback_reason"] = str(fallback_reason)
    navigation_command.update(extra_command or {})
    metadata = {
        "pointgoal_body_xy": pointgoal.tolist(),
        "pointgoal_distance_m": pointgoal_norm_m,
        "target_description": str(target_description),
        "navigation_backend": backend_name,
    }
    if fallback_reason is not None:
        metadata["fallback_reason"] = str(fallback_reason)
    metadata.update(extra_metadata or {})
    prefix = (
        "A* found no connected route; X-NavDP will track"
        if fallback_reason is not None
        else "X-NavDP will track"
    )
    return ToolResult(
        call_id=call.call_id,
        name=call.tool,
        action={
            "action": "navigate_to_pointgoal",
            "navigation_command": navigation_command,
        },
        content=(
            f"{prefix} the requested destination {target_description!r}. "
            "The runtime tracks this fixed destination with fresh RGB-D until "
            "arrival, sustained lack of progress, or navigation timeout."
        ),
        metadata=metadata,
    )


__all__ = [
    "MoveBodyHandler",
    "MoveEndEffectorHandler",
    "MoveForwardHandler",
    "MoveBackwardHandler",
    "MoveLeftHandler",
    "MoveRightHandler",
    "TurnLeftHandler",
    "TurnRightHandler",
    "WaitHandler",
    "CrouchHandler",
    "StandUpHandler",
    "SpeakHandler",
    "AskHandler",
    "ReportDoneHandler",
    "HandoverObjectHandler",
    "LocateObjectHandler",
    "GraspObjectHandler",
    "NavigateToPointGoalHandler",
]
