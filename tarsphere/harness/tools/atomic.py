"""Atomic tools — first-class calls at the Agent decision boundary.

Per the envelope architecture (EVALUATION.md): atomic tools are the model's call
option. Each maps to a controller primitive (move/turn/wait), one manipulation
capability (end-effector/grasp), or one act of sensing
(locate_object), with NO internal "small-brain" optimisation.
Travel is atomic here so the agent owns every leg of its own trajectory.

Atomic means one call boundary, not one controller motion. ``grasp_object``
grounds the current frame and starts the planned reach, closure, and carry
sequence in one call. Its measured acquisition and retention outcome remains
part of task progress and failure attribution.

This module owns the atomic-tool registry population: it pairs each atomic
ToolSpec with its handler and registers them into a ToolRegistry. The handlers
themselves live in ``handlers.py``.

The signature of a registered atomic tool is: takes a ToolCall, returns a
ToolResult whose ``action`` is a symbolic action dict consumed by the
runtime/controller adapter — or ``None`` for a call that reports a fact instead
of moving the robot (``locate_object``, ``inspect_entity``).
"""

from __future__ import annotations

from typing import Any

from tarsphere.harness.tools.handlers import (
    AskHandler,
    CrouchHandler,
    GraspObjectHandler,
    HandoverObjectHandler,
    InspectEntityHandler,
    LocateObjectHandler,
    MoveBodyHandler,
    MoveEndEffectorHandler,
    MoveBackwardHandler,
    MoveForwardHandler,
    MoveLeftHandler,
    MoveRightHandler,
    NavigateToPointGoalHandler,
    NavigateToEntityHandler,
    NavigateToFrontierHandler,
    RememberEntityHandler,
    ReportDoneHandler,
    SpeakHandler,
    StandUpHandler,
    TiltCameraHandler,
    TurnLeftHandler,
    TurnRightHandler,
    WaitHandler,
)
from tarsphere.harness.tools.registry import ToolRegistry
from tarsphere.harness.tools.specs import (
    ASK,
    CROUCH,
    GRASP_OBJECT,
    HANDOVER_OBJECT,
    INSPECT_ENTITY,
    LOCATE_OBJECT,
    MOVE_BODY,
    MOVE_END_EFFECTOR,
    MOVE_BACKWARD,
    MOVE_FORWARD,
    MOVE_LEFT,
    MOVE_RIGHT,
    NAVIGATE_TO_POINTGOAL,
    NAVIGATE_TO_ENTITY,
    NAVIGATE_TO_FRONTIER,
    REMEMBER_ENTITY,
    REPORT_DONE,
    SPEAK,
    STAND_UP,
    TILT_CAMERA,
    TURN_LEFT,
    TURN_RIGHT,
    WAIT,
)


# ---------------------------------------------------------------------------
# Registry population
# ---------------------------------------------------------------------------


def register_atomic_tools(
    registry: ToolRegistry,
    *,
    semantic_backend: Any | None = None,
    grasp_backend: Any | None = None,
    memory: Any | None = None,
    navigation_backend: Any | None = None,
    navigation_fallback_backend: Any | None = None,
) -> None:
    """Register every first-class atomic tool into ``registry``.

    Every optional dependency gates its own tool. Without a
    ``semantic_backend`` (SAM3) there is no semantic perception; without both
    synchronized SAM3 and GraspGenX services there is no ``grasp_object``. An
    unwired tool stays unregistered rather than being advertised and then
    failing when the model calls it.

    ``memory`` is optional spatial memory for ``locate_object`` and for
    retaining the identity/provenance of successful current-view grounding.
    """
    # Locomotion atomics.
    registry.register(MOVE_BODY, MoveBodyHandler())
    registry.register(MOVE_END_EFFECTOR, MoveEndEffectorHandler())
    registry.register(MOVE_FORWARD, MoveForwardHandler())
    registry.register(MOVE_BACKWARD, MoveBackwardHandler())
    registry.register(MOVE_LEFT, MoveLeftHandler())
    registry.register(MOVE_RIGHT, MoveRightHandler())
    registry.register(TURN_LEFT, TurnLeftHandler())
    registry.register(TURN_RIGHT, TurnRightHandler())
    registry.register(WAIT, WaitHandler())
    registry.register(CROUCH, CrouchHandler())
    registry.register(STAND_UP, StandUpHandler())
    # Gaze: aims the camera, not the body, but shares the body's action path.
    registry.register(TILT_CAMERA, TiltCameraHandler())
    # Communication atomics.
    registry.register(SPEAK, SpeakHandler())
    registry.register(ASK, AskHandler())
    registry.register(REPORT_DONE, ReportDoneHandler())
    registry.register(HANDOVER_OBJECT, HandoverObjectHandler())
    if semantic_backend is not None:
        locate_spec = LOCATE_OBJECT
        registry.register(
            locate_spec,
            LocateObjectHandler(semantic_backend=semantic_backend, memory=memory),
        )
    if memory is not None:
        registry.register(
            REMEMBER_ENTITY,
            RememberEntityHandler(
                memory=memory,
                semantic_backend=semantic_backend,
            ),
        )
        registry.register(INSPECT_ENTITY, InspectEntityHandler(memory=memory))
    # Manipulation atomics. A grasp is advertised only when its complete
    # current-frame perception chain exists; no canonical-pose fallback.
    if semantic_backend is not None and grasp_backend is not None:
        registry.register(
            GRASP_OBJECT,
            GraspObjectHandler(
                semantic_backend=semantic_backend,
                grasp_backend=grasp_backend,
                memory=memory,
            ),
        )
    # OpenFrontier adds visual
    # A/B/C targets without replacing the general relative-PointGoal tool.
    # A/B/C selection is map-independent. After the Agent chooses a label,
    # both public navigation tools share the same A*-first, X-NavDP recovery
    # path against the partial online map.
    pointgoal_handler: Any
    pointgoal_spec = NAVIGATE_TO_POINTGOAL
    pointgoal_handler = NavigateToPointGoalHandler(
        x_navdp_backend=navigation_fallback_backend,
    )
    if navigation_backend is not None:
        registry.register(
            NAVIGATE_TO_FRONTIER,
            NavigateToFrontierHandler(
                navigation_backend,
                pointgoal_handler=pointgoal_handler,
            ),
        )
    registry.register(pointgoal_spec, pointgoal_handler)
    if memory is not None:
        registry.register(
            NAVIGATE_TO_ENTITY,
            NavigateToEntityHandler(
                memory=memory,
                pointgoal_handler=pointgoal_handler,
            ),
        )


__all__ = [
    "register_atomic_tools",
]
