"""Canonical event-driven episode runner with a strict Agent/evaluator boundary."""

from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, List, Optional

from tarsphere.api_usage import aggregate_model_calls
from tarsphere.harness.core.failures import BACKEND_FAILURES
from tarsphere.configs.secrets import EvaluationSupportServiceError
from tarsphere.contracts import DecisionRequirements
from tarsphere.contracts.task_profile import (
    ComponentOutageProfile,
    SpatialBlockingProfile,
)
from tarsphere.bench.schema import TARSphereEpisode
from tarsphere.infra.action_errors import ActionRefusedError
from tarsphere.infra.episode.event_bus import EventBus
from tarsphere.infra.episode.errors import EpisodeInitializationError, RobotFallError
from tarsphere.infra.episode.human_actor_runtime import QuestionOutcome
from tarsphere.infra.episode.disruption_runtime import DisruptionRuntime
from tarsphere.harness.actions.router import (
    decision_has_compiled_controller,
    requires_environment_execution,
)
from tarsphere.harness.harness import AgentHarnessConfig
from tarsphere.infra.episode.rollout_control import (
    EpisodeAbortError,
    EpisodeStallError,
    ProgressWatchdog,
    write_heartbeat,
)
from tarsphere.infra.manipulation.agent_grasp import GRASP_FAILURE_CODES
from tarsphere.evaluation.engine import (
    causal_graph_completed,
    completed_milestone_ids,
    milestone_dag_reaches,
    is_internal_task_phase,
)
from tarsphere.evaluation.schema import (
    EpisodeTrajectory,
    MilestoneReference,
    PRIMITIVE_REACTION_SAMPLING_PHASE,
    ReactionSamplingPhase,
    DECLARATIVE_MILESTONE_EVALUATORS,
    TrajectoryStep,
)


FORBIDDEN_AGENT_KEYS = {
    "navigation_trace",
    "navigation_motion_samples",
    "expected_action",
    "ground_truth",
    "hidden_context",
    "hidden_state",
    "human_positions",
    "humans",
    "accepted_handovers",
    "handover",
    "latent_need",
    "measured_distance_m",
    "phase",
    "phases",
    "render_count",
    "render_mode",
    "start_position",
    "success",
    "diagnostic_checks",
    "target_position",
    "world_knowledge",
}

COMMUNICATIVE_ACTIONS = {
    "ask",
    "speak",
    "report_done",
    "handover_object",
}
OBSERVATION_LOG_KEYS = (
    "text",
    "image_path",
    "depth_status",
    "depth_valid_ratio",
)

SPATIAL_CLEARANCE_RETRY_MESSAGE = (
    "I still cannot get around you safely. Please move a few steps back "
    "toward the way you came, and stay nearby so I can let you through."
)


def _tool_result_error_type(result: Mapping[str, Any]) -> str:
    metadata = result.get("metadata")
    return str(
        result.get("error_type")
        or (metadata.get("error_type") if isinstance(metadata, Mapping) else "")
        or ""
    ).strip()


def agent_has_pending_actions(agent: Any) -> bool:
    """Whether the current policy's serial block still has executable work."""

    stats = getattr(agent, "execution_stats", None)
    if not isinstance(stats, dict):
        return False
    pending = stats.get("pending_actions", 0)
    if type(pending) is not int or pending < 0:
        raise ValueError("Agent pending_actions must be a non-negative integer")
    return pending > 0


def agent_declared_plan_complete_and_idle(agent: Any) -> bool:
    """Return the Agent's own terminal plan belief after its block drains.

    This reads only Agent-authored bookkeeping. It does not inspect task
    milestones, private world state, or Judge output, and therefore cannot
    turn a mistaken completion claim into benchmark success.
    """

    memory = getattr(agent, "memory", None)
    planner = getattr(memory, "strategy_planner", None)
    plan = getattr(planner, "active_plan", None)
    steps = getattr(plan, "steps", None)
    completed_steps = getattr(plan, "completed_steps", None)
    if not isinstance(steps, list) or not steps:
        return False
    if not isinstance(completed_steps, int) or completed_steps < len(steps):
        return False
    return not agent_has_pending_actions(agent)


def _evaluation_agent_state(decision: Mapping[str, Any]) -> dict[str, Any]:
    """First accepted task outputs, without rewriting the raw model envelope."""

    state = dict(decision.get("envelope") or {})
    for batch in reversed(decision.get("accepted_event_output_batches", [])):
        state.update(batch)
    return state


def _nested_value(mapping: Any, *path: str) -> Any:
    """Read one evaluator-private nested value without inventing defaults."""

    current = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _optional_counter(environment: Any, name: str) -> Optional[int]:
    """Read a published environment counter, or None when it publishes none."""

    value = getattr(environment, name, None)
    return None if value is None else int(value)


def _merge_reaction_samples(
    *phase_samples: Mapping[str, bool],
) -> dict[str, bool]:
    """Merge disjoint evaluator phases without hiding duplicate sampling."""

    merged: dict[str, bool] = {}
    for sample in phase_samples:
        duplicate = set(merged).intersection(sample)
        if duplicate:
            raise ValueError(
                "primitive reactions were sampled in more than one phase: "
                f"{sorted(duplicate)}"
            )
        merged.update({str(key): bool(value) for key, value in sample.items()})
    return merged


def _execute_environment_action(
    environment: Any,
    decision: dict[str, Any],
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    """Execute one action while preserving measured falls as task evidence."""

    start_physics_steps = getattr(environment, "_physics_steps", None)
    try:
        return environment.step(decision)
    except RobotFallError as fall:
        return (
            fall.observation,
            True,
            {
                "executed": False,
                "feedback": str(fall),
                "physics_steps": int(fall.details["physics_steps"]),
                "action_physics_steps": (
                    int(fall.details["physics_steps"]) - int(start_physics_steps)
                    if start_physics_steps is not None
                    else None
                ),
                "robot_stability": dict(fall.details["stability"]),
                "physical_contact": bool(fall.details["physical_contacts"]),
                "physical_contact_details": fall.details["physical_contact_details"],
                "native_contacts": list(fall.details["physical_contacts"]),
                "safety_violation": {
                    "type": "robot_fall",
                    "events": [dict(fall.details)],
                },
            },
        )
    except ActionRefusedError as refusal:
        return (
            environment.observe(),
            False,
            {
                "executed": False,
                "feedback": str(refusal),
                "physics_steps": getattr(environment, "_physics_steps", None),
                "action_refusal": refusal.as_feedback(),
            },
        )


def _observe_completed_tool(
    environment: Any,
    decision: dict[str, Any],
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    """Refresh observation after a tool that finished inside its handler."""

    _raise_for_tool_backend_failure(decision)
    observe = getattr(environment, "observe", None)
    if not callable(observe):
        raise RuntimeError(
            "environment must observe after a completed in-handler tool turn"
        )
    results = decision.get("tool_results") or []
    if not isinstance(results, list):
        raise TypeError("decision.tool_results must be a list when present")
    call_name = str(decision.get("call_name") or decision.get("action") or "")
    call_index = int(decision.get("tool_call_index") or 0)
    selected: Mapping[str, Any] | None = None
    if 0 <= call_index < len(results):
        candidate = results[call_index]
        if (
            isinstance(candidate, Mapping)
            and str(candidate.get("name") or "") == call_name
        ):
            selected = candidate
    if selected is None:
        matching = [
            row
            for row in results
            if (isinstance(row, Mapping) and str(row.get("name") or "") == call_name)
        ]
        if matching:
            # A failed suffix may be the only serialized result even though
            # its original tool_call_index is greater than zero.
            selected = matching[-1]
    is_error = bool(decision.get("is_error"))
    content = ""
    error_type = ""
    if selected is not None:
        is_error = is_error or bool(selected.get("is_error"))
        content = str(selected.get("content") or "").strip()
        error_type = _tool_result_error_type(selected)
    if is_error and not content:
        content = f"{call_name or 'tool'} failed without a result message."
    if is_error and error_type:
        content = f"[{error_type}] {content}"
    physics_step = int(getattr(environment, "_physics_steps", 0))
    return (
        observe(),
        False,
        {
            "executed": not is_error,
            "feedback": content,
            "physics_steps": physics_step,
            "action_physics_steps": 0,
        },
    )


def _raise_for_tool_backend_failure(decision: Mapping[str, Any]) -> None:
    """Reject any capability failure recorded during this decision.

    Information tools may run before the action ultimately returned by the
    policy kernel. Inspect the complete result batch rather than only the last
    call, otherwise an earlier backend failure can disappear behind a later
    successful call and be scored as model behaviour.
    """

    results = decision.get("tool_results") or []
    if not isinstance(results, list):
        raise TypeError("decision.tool_results must be a list when present")
    for result in results:
        if not isinstance(result, Mapping) or not bool(result.get("is_error")):
            continue
        error_type = _tool_result_error_type(result)
        if error_type not in BACKEND_FAILURES:
            continue
        name = str(result.get("name") or "tool").strip() or "tool"
        content = str(result.get("content") or "").strip()
        detail = f"[{error_type}] {content}" if content else f"[{error_type}]"
        raise EvaluationSupportServiceError(
            f"{name} evaluation capability failed: {detail}"
        )


def _execute_navigation_or_action(
    environment,
    agent,
    decision,
    observation,
    disruption: DisruptionRuntime | None = None,
):
    """Dispatch one policy action; NavDP owns its internal physical prefixes."""
    _raise_for_tool_backend_failure(decision)
    # A handler-side failure owns a tool-identity payload so it can occupy one
    # serial action step and return its feedback.  It is not a second attempt
    # at the semantic action.  In particular, a failed ``ask`` has no
    # recipient arguments, and a failed handover has no Human reply.  Preserve
    # the exact serialized ToolResult before consulting execution-time
    # disruptions or the environment.
    if bool(decision.get("is_error")):
        return _observe_completed_tool(environment, decision)
    physics_step = int(getattr(environment, "_physics_steps", 0))
    tool_name = str(decision.get("call_name") or "")
    if disruption is not None and disruption.component_unavailable(
        tool_name, physics_step
    ):
        return (
            observation,
            False,
            {
                "executed": False,
                "feedback": f"{tool_name} is temporarily unavailable.",
                "physics_steps": physics_step,
                "action_physics_steps": 0,
                "disruption": "component_outage",
            },
        )
    recipient = str(decision.get("recipient") or "")
    execution_decision = decision
    canonical_recipient: str | None = recipient
    if decision.get("action") == "handover_object":
        resolve_recipient = getattr(
            environment,
            "resolve_handover_human_id",
            None,
        )
        if not callable(resolve_recipient):
            raise RuntimeError(
                "handover_object requires environment.resolve_handover_human_id"
            )
        canonical_recipient = resolve_recipient(recipient)
        execution_decision = dict(decision)
        execution_decision["_handover_resolution_checked"] = True
        execution_decision["_resolved_handover_human_id"] = canonical_recipient
    can_communicate = getattr(environment, "can_communicate", None)
    recipient_can_communicate = False
    if callable(can_communicate) and canonical_recipient:
        try:
            recipient_can_communicate = bool(can_communicate(canonical_recipient))
        except KeyError:
            recipient_can_communicate = False
    unavailable = (
        disruption.unavailable_response(canonical_recipient, physics_step)
        if (
            disruption is not None
            and decision.get("action") == "handover_object"
            and recipient_can_communicate
        )
        else None
    )
    if unavailable is not None:
        return (
            observation,
            False,
            {
                "executed": False,
                "feedback": unavailable,
                "physics_steps": physics_step,
                "action_physics_steps": 0,
                "temporary_unavailability_human_id": canonical_recipient,
            },
        )
    if not requires_environment_execution(execution_decision):
        return _observe_completed_tool(environment, execution_decision)
    navigation = decision.get("navigation_command") or {}
    if navigation.get("backend") not in {"x_navdp", "x_navdp_fallback"}:
        return _execute_environment_action(environment, execution_decision)
    from tarsphere.infra.navigation.x_navdp_execution import execute_x_navdp_navigation

    handler = agent.tools.handler("navigate_to_pointgoal")
    backend = handler.x_navdp_backend
    if backend is None:
        raise RuntimeError("Closed-loop NavDP has no registered policy backend.")
    latest_observation = observation

    def execute_motion(inner):
        nonlocal latest_observation
        before = int(environment._physics_steps)
        updated, done, info = _execute_environment_action(environment, inner)
        latest_observation = updated
        info = dict(info)
        info["action_physics_steps"] = int(environment._physics_steps) - before
        return updated, done, info

    initial_physics_steps = int(environment._physics_steps)
    updated, done, info = execute_x_navdp_navigation(
        decision=decision,
        observation=observation,
        backend=backend,
        camera_views=agent.agent_harness_config.protocol.camera_views,
        execute_motion=execute_motion,
        remaining_episode_physics_steps=environment._remaining_physics_steps(),
    )
    info["physics_steps"] = int(environment._physics_steps)
    if (
        info["navigation_skill"]["physics_steps"]
        != int(environment._physics_steps) - initial_physics_steps
    ):
        raise RuntimeError("NavDP execution lost physical tick accounting.")
    return updated, done, info


def _observe_progress_watchdog(
    watchdog: ProgressWatchdog | None,
    heartbeat: dict[str, Any],
) -> dict[str, Any] | None:
    """Convert a measured locomotion stall into benchmark evidence."""

    if watchdog is None:
        return None
    try:
        return watchdog.observe(heartbeat)
    except EpisodeStallError as stall:
        return dict(stall.details)


def _runner_memory_diagnostic(
    environment: Any,
    label: str,
    decision_step: int,
) -> None:
    """Print sparse runner-boundary memory evidence for an opt-in rollout."""
    enabled = os.environ.get(
        "TARSPHERE_ROLLOUT_MEMORY_DIAGNOSTICS", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return
    physics_steps = int(getattr(environment, "_physics_steps", -1))
    try:
        minimum = int(
            os.environ.get("TARSPHERE_ROLLOUT_MEMORY_DIAGNOSTICS_MIN_STEP", "0")
        )
    except ValueError:
        minimum = 0
    if physics_steps < minimum:
        return
    rss_kib = -1
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    rss_kib = int(line.split()[1])
                    break
    except (OSError, ValueError, IndexError):
        pass
    print(
        "[RUNNER-MEM] "
        f"label={label} decision_step={int(decision_step)} "
        f"physics_steps={physics_steps} rss_kib={rss_kib}",
        flush=True,
    )


def sanitize_agent_observation(value: Any) -> Any:
    """Remove evaluator-only fields before an observation reaches the agent."""
    if isinstance(value, dict):
        return {
            key: sanitize_agent_observation(item)
            for key, item in value.items()
            if key not in FORBIDDEN_AGENT_KEYS
        }
    if isinstance(value, list):
        return [sanitize_agent_observation(item) for item in value]
    return value


def resolve_agent_action_space(
    episode: TARSphereEpisode,
    agent_harness_config: AgentHarnessConfig | None = None,
) -> List[str]:
    """Resolve the task-agnostic Agent surface plus explicit diagnostics.

    Official evaluated episodes always receive the same public tools, so tool
    presence cannot disclose the task mainline or capability slice. Component
    diagnostics may declare a deliberately restricted ``agent_action_space``.

    Relative-PointGoal navigation is a harness capability, not benchmark task
    semantics. The run-level navigation mode is resolved once before the
    episode and never changes with episode state. The same resolved list is
    used to build the registry, construct the prompt, publish the API tool
    schema, and authorize dispatch.
    """
    from tarsphere.bench.compilation.episode import (
        DEFAULT_NAVIGATION_ACTIONS,
        agent_action_space,
    )

    spec = (
        agent_action_space()
        if getattr(episode, "evaluation", None) is not None
        else episode.agent_action_space or agent_action_space()
    )
    actions = spec.cognitive + spec.perceptual + spec.communicative + spec.physical
    pass
    pass
    pass
    navigation_actions: tuple[str, ...] = ()
    navigation_actions = DEFAULT_NAVIGATION_ACTIONS
    pass
    for navigation_action in navigation_actions:
        if navigation_action not in actions:
            actions.append(navigation_action)
    return actions


def terminal_completion_human_ids(
    evaluation: Any,
    humans: Any = (),
    *,
    task_bindings: Any = None,
) -> tuple[str, ...]:
    """Return Humans who own terminal task outcomes in declaration order."""

    referenced = {
        str(condition.milestone_ref)
        for milestone in evaluation.milestones
        if milestone.kind == "scored"
        for condition in milestone.preconditions
        if isinstance(condition, MilestoneReference)
    }
    human_ids: list[str] = []
    for milestone in evaluation.milestones:
        if milestone.kind != "scored" or str(milestone.milestone_id) in referenced:
            continue
        truth = (
            task_bindings.truth(milestone.milestone_id)
            if task_bindings is not None
            else milestone.outcome.ground_truth
        )
        human_id = str(truth.get("human_id") or "").strip()
        if human_id and human_id not in human_ids:
            human_ids.append(human_id)
    cast_ids = tuple(
        dict.fromkeys(
            str(getattr(human, "id", "") or "").strip()
            for human in humans
            if str(getattr(human, "id", "") or "").strip()
        )
    )
    if not human_ids and len(cast_ids) == 1:
        return cast_ids
    if not human_ids:
        raise ValueError(
            "A completed evaluated task needs a terminal milestone whose "
            "ground_truth names human_id for positive completion feedback."
        )
    return tuple(human_ids)


def describe_locomotion_outcome(
    *,
    action: str,
    decision: Dict[str, Any],
    pose_before: Any,
    pose_after: Any,
    outcome: Optional[Dict[str, Any]] = None,
) -> str:
    """Report what the body actually did against what the agent asked for.

    ``move_by_speed`` is tracked by a learned gait. The simulator-adjacent
    executor applies it for the duration derived from the requested magnitude;
    acceleration, gait tracking, native contact, and final-tick dynamics can make
    achieved motion differ from the nominal request. This post-step measurement
    reports the physical result honestly to the Agent.

    Returns an empty clause for decisions carrying no magnitude — the internal
    approach refinements and the authoring oracle have their own completion
    conditions and nothing to compare against.
    """
    outcome = outcome or measure_locomotion_outcome(
        action=action,
        decision=decision,
        pose_before=pose_before,
        pose_after=pose_after,
    )
    if not outcome:
        return ""
    if not bool(outcome.get("available", False)):
        return (
            "How far you actually moved is unavailable: this step carries no "
            "robot pose."
        )
    if outcome["kind"] == "navigation_motion":
        achieved = outcome["achieved_body_delta"]
        return (
            "Navigation net body displacement: "
            f"forward={achieved['forward_m']:+.2f}m, "
            f"left={achieved['left_m']:+.2f}m, "
            f"turn={achieved['turn_deg']:+.1f}°."
        )
    if outcome["kind"] == "body_motion":
        requested = outcome["requested_body_delta"]
        achieved = outcome["achieved_body_delta"]
        return (
            "You requested body delta "
            f"(forward={float(requested['forward_m']):+.2f}m, "
            f"left={float(requested['left_m']):+.2f}m, "
            f"turn={float(requested['turn_deg']):+.1f}°); "
            "the measured delta was "
            f"(forward={float(achieved['forward_m']):+.2f}m, "
            f"left={float(achieved['left_m']):+.2f}m, "
            f"turn={float(achieved['turn_deg']):+.1f}°)."
        )
    requested = float(outcome["requested"])
    achieved = float(outcome["achieved_along_command"])
    unit = str(outcome["unit"])
    if outcome["kind"] == "translation":
        lateral = float(outcome["lateral_drift_m"])
        report = (
            f"You asked for {requested:.2f}{unit}; the body advanced "
            f"{achieved:.2f}{unit} along the requested direction with "
            f"{lateral:.2f}m lateral drift."
        )
        return report
    report = (
        f"You asked for {requested:.2f}{unit}; the body rotated "
        f"{achieved:.2f}{unit} in the requested direction."
    )
    return report


def measure_locomotion_outcome(
    *,
    action: str,
    decision: Dict[str, Any],
    pose_before: Any,
    pose_after: Any,
) -> Dict[str, Any]:
    """Measure achieved motion along the direction the primitive requested.

    Euclidean displacement is not action completion: a backward gait that
    drifts sideways or forward has not created the clearance its queued suffix
    assumes.  Translation is therefore projected onto the starting body axis;
    rotation is signed in the requested yaw direction.

    For a body-frame vector, ``translation_completion_ratio`` is the signed
    projection of achieved translation onto requested translation divided by
    the requested translation magnitude.  One matches the request, zero makes
    no progress along it, and a negative value moves oppositely.  The
    ``turn_error_deg`` is achieved turn minus requested turn, with positive
    values denoting excess leftward rotation.  Both are measurements only;
    neither changes execution or selects the next action.
    """

    navigation_motion = (decision.get("navigation_command") or {}).get(
        "phase"
    ) == "closed_loop_navigation"
    if action == "move_body" or navigation_motion:
        raw_delta = [0.0, 0.0, 0.0] if navigation_motion else decision.get("delta")
        if not isinstance(raw_delta, (list, tuple)) or len(raw_delta) != 3:
            return {}
        requested = {
            "forward_m": float(raw_delta[0]),
            "left_m": float(raw_delta[1]),
            "turn_deg": float(raw_delta[2]),
        }
        kind, unit = "body_motion", "body_frame"
    elif action in {
        "move_forward",
        "move_backward",
        "move_left",
        "move_right",
    }:
        requested = decision.get("distance_m")
        kind, unit = "translation", "m"
    elif action in {"turn_left", "turn_right"}:
        requested = decision.get("degrees")
        kind, unit = "rotation", "°"
    else:
        return {}
    if requested is None:
        return {}

    result: Dict[str, Any] = {
        "action": action,
        "kind": kind,
        "unit": unit,
        **(
            {"requested_body_delta": requested}
            if kind == "body_motion"
            else {"requested": float(requested)}
        ),
        "available": False,
    }
    before = pose_before if isinstance(pose_before, dict) else {}
    after = pose_after if isinstance(pose_after, dict) else {}
    start, end = before.get("position"), after.get("position")
    readable = (
        isinstance(start, (list, tuple))
        and len(start) >= 2
        and isinstance(end, (list, tuple))
        and len(end) >= 2
        and before.get("yaw_rad") is not None
        and after.get("yaw_rad") is not None
    )
    if not readable:
        return result

    if kind == "body_motion":
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        yaw = float(before["yaw_rad"])
        achieved_forward = dx * math.cos(yaw) + dy * math.sin(yaw)
        achieved_left = -dx * math.sin(yaw) + dy * math.cos(yaw)
        yaw_delta = math.atan2(
            math.sin(float(after["yaw_rad"]) - yaw),
            math.cos(float(after["yaw_rad"]) - yaw),
        )
        achieved_turn_deg = float(math.degrees(yaw_delta))
        requested_translation_m = math.hypot(
            float(requested["forward_m"]),
            float(requested["left_m"]),
        )
        result.update(
            {
                "available": True,
                "achieved_body_delta": {
                    "forward_m": float(achieved_forward),
                    "left_m": float(achieved_left),
                    "turn_deg": achieved_turn_deg,
                },
                "turn_error_deg": math.degrees(
                    math.atan2(
                        math.sin(
                            yaw_delta - math.radians(float(requested["turn_deg"]))
                        ),
                        math.cos(
                            yaw_delta - math.radians(float(requested["turn_deg"]))
                        ),
                    )
                ),
            }
        )
        if (decision.get("navigation_command") or {}).get(
            "phase"
        ) == "closed_loop_navigation":
            # The first controller prefix is not the MLLM's requested motion.
            # A curved, multi-turn path has no single requested body delta.
            result["kind"] = "navigation_motion"
            result.pop("requested_body_delta")
            result.pop("turn_error_deg")
            return result
        if requested_translation_m > 0.0:
            progress_along_request_m = (
                float(requested["forward_m"]) * achieved_forward
                + float(requested["left_m"]) * achieved_left
            ) / requested_translation_m
            result["translation_completion_ratio"] = (
                float(progress_along_request_m) / requested_translation_m
            )
        return result
    if kind == "translation":
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        yaw = float(before["yaw_rad"])
        along = dx * math.cos(yaw) + dy * math.sin(yaw)
        lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
        if action == "move_forward":
            achieved = along
            result["lateral_drift_m"] = abs(float(lateral))
        elif action == "move_backward":
            achieved = -along
            result["lateral_drift_m"] = abs(float(lateral))
        elif action == "move_left":
            achieved = lateral
            # Drift orthogonal to the commanded flank is along-heading slip.
            result["lateral_drift_m"] = abs(float(along))
        else:
            achieved = -lateral
            result["lateral_drift_m"] = abs(float(along))
    else:
        yaw_delta = math.atan2(
            math.sin(float(after["yaw_rad"]) - float(before["yaw_rad"])),
            math.cos(float(after["yaw_rad"]) - float(before["yaw_rad"])),
        )
        achieved = math.degrees(yaw_delta)
        if action == "turn_right":
            achieved = -achieved

    result.update(
        {
            "available": True,
            "achieved_along_command": float(achieved),
            "completion_ratio": float(achieved) / float(requested),
        }
    )
    return result


def navigation_execution_feedback(
    *,
    decision: Mapping[str, Any],
    motion_outcome: Mapping[str, Any],
    physical_contact: bool,
    navigation_skill: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compare one delegated navigation command with real body motion.

    Preserve the requested world destination while expressing its remaining
    offset in the post-action body frame, using measured translation and yaw.
    The original request remains in navigation_command. Frontier labels are
    provenance only. This is feedback, never a success gate or motion veto.
    """

    command = decision.get("navigation_command")
    if not isinstance(command, Mapping):
        return {}
    backend = str(command.get("backend") or "unknown")
    phase = str(command.get("phase") or "local_prefix")
    motion_available = (
        isinstance(motion_outcome, Mapping)
        and bool(motion_outcome.get("available", False))
        and motion_outcome.get("kind") in {"body_motion", "navigation_motion"}
    )
    raw_goal = command.get("pointgoal_body_xy")
    if not isinstance(raw_goal, (list, tuple)) or len(raw_goal) != 2:
        raise ValueError("navigation_command.pointgoal_body_xy must be [F, L].")
    goal = (float(raw_goal[0]), float(raw_goal[1]))
    if not all(math.isfinite(value) for value in goal):
        raise ValueError("navigation PointGoal must be finite.")
    before_m = math.hypot(*goal)
    remaining_goal = None
    after_m = None
    if motion_available:
        delta = motion_outcome["achieved_body_delta"]
        residual_forward = goal[0] - float(delta["forward_m"])
        residual_left = goal[1] - float(delta["left_m"])
        turn_rad = math.radians(float(delta["turn_deg"]))
        cosine, sine = math.cos(turn_rad), math.sin(turn_rad)
        # Residual is still in the request frame. Rotate by the inverse of
        # the actual body turn, not the commanded turn or planned heading.
        remaining_goal = [
            cosine * residual_forward + sine * residual_left,
            -sine * residual_forward + cosine * residual_left,
        ]
        after_m = math.hypot(*remaining_goal)
    progress_m = before_m - after_m if after_m is not None else None
    result = {
        "backend": backend,
        "target_description": str(command.get("target_description") or ""),
        "phase": phase,
        "remaining_pointgoal_body_xy": remaining_goal,
        "distance_before_m": before_m,
        "distance_after_estimate_m": after_m,
        "progress_m": progress_m,
        "physical_contact": bool(physical_contact),
    }
    if command.get("fallback_reason"):
        result["fallback_reason"] = str(command["fallback_reason"])
    if command.get("frontier_label"):
        result["frontier_label"] = str(command["frontier_label"])
    if command.get("frontier_id"):
        result["frontier_id"] = str(command["frontier_id"])
    if navigation_skill is not None:
        result.update(
            {
                key: navigation_skill[key]
                for key in (
                    "stop_reason",
                    "simulation_seconds",
                    "travelled_distance_m",
                    "motion_count",
                    "backend_error",
                    "physics_steps",
                    "physics_step_limit",
                    "goal_tolerance_m",
                    "position_heading_window_steps",
                    "position_window_radius_m",
                    "heading_window_span_rad",
                    "steps_without_new_coverage_or_closer_distance",
                )
                if key in navigation_skill
            }
        )
    return result


def build_agent_feedback(
    *,
    step: int,
    action: str,
    robot_message: str,
    execution: Dict[str, Any],
    observation: Dict[str, Any],
    decision: Optional[Dict[str, Any]] = None,
    motion_outcome: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build public feedback for an EmbodiedBench-style planner update.

    Carries what the agent's own action did. Human utterances reach the agent
    attributed through ``observation['public_events']`` and the dialogue
    history it feeds; action feedback does not duplicate them.

    ``body_stability`` exposes only post-action gravity tilt in degrees.  It is
    ordinary proprioception: zero degrees is upright, and larger values mean
    the body is farther from vertical.  Embodiments that do not report a
    gravity-aligned stability measurement omit the field rather than inventing
    one.  No stability value prescribes or blocks an Agent action.
    """
    safe_execution = sanitize_agent_observation(execution)
    safe_decision = sanitize_agent_observation(decision or {})
    envelope = safe_decision.get("envelope") if isinstance(safe_decision, dict) else {}
    if not isinstance(envelope, dict):
        envelope = {}

    robot_stability = safe_execution.get("robot_stability")
    body_stability: Dict[str, float] = {}
    if robot_stability is not None:
        if not isinstance(robot_stability, dict):
            raise TypeError("robot_stability must be a mapping when present.")
        gravity_tilt_deg = robot_stability.get("tilt_deg")
        if gravity_tilt_deg is None:
            raise ValueError("robot_stability must contain tilt_deg when present.")
        gravity_tilt_deg = float(gravity_tilt_deg)
        if not math.isfinite(gravity_tilt_deg):
            raise ValueError("robot_stability.tilt_deg must be finite.")
        body_stability = {"gravity_tilt_deg": gravity_tilt_deg}

    balance_control = safe_execution.get("balance_control") or {}
    if not isinstance(balance_control, dict):
        raise TypeError("balance_control must be a mapping when present.")

    grasp_failure = safe_execution.get("grasp_failure") or {}
    if not isinstance(grasp_failure, dict):
        raise TypeError("grasp_failure must be a mapping when present.")
    public_grasp_failure: Dict[str, str] = {}
    if grasp_failure:
        failure_code = str(grasp_failure.get("code") or "").strip()
        failure_message = str(grasp_failure.get("message") or "").strip()
        if failure_code not in GRASP_FAILURE_CODES:
            raise ValueError(
                "grasp_failure.code must use the closed runtime vocabulary; "
                f"got {failure_code!r}."
            )
        if not failure_message:
            raise ValueError("grasp_failure.message must be non-empty.")
        public_grasp_failure = {
            "code": failure_code,
            "message": failure_message,
        }

    grasp_reachability = safe_execution.get("grasp_reachability") or {}
    if not isinstance(grasp_reachability, dict):
        raise TypeError("grasp_reachability must be a mapping when present.")

    grasp_skill = safe_execution.get("grasp_skill") or {}
    if not isinstance(grasp_skill, dict):
        raise TypeError("grasp_skill must be a mapping when present.")

    placement_release = safe_execution.get("placement_release") or {}
    if not isinstance(placement_release, dict):
        raise TypeError("placement_release must be a mapping when present.")
    placed = safe_execution.get("placed", False)
    if not isinstance(placed, bool):
        raise TypeError("execution.placed must be a Boolean when present.")

    navigation_feedback = navigation_execution_feedback(
        # ``sanitize_agent_observation`` intentionally strips controller-only
        # fields, including the navigation phase. The command itself is public
        # Agent intent and this helper reads only that bounded submapping, so
        # preserve it from the executed decision. Otherwise A* heading-only
        # alignment is mislabeled as ordinary no-progress feedback.
        decision=decision or {},
        motion_outcome=motion_outcome or {},
        physical_contact=bool(safe_execution.get("physical_contact", False)),
        navigation_skill=execution.get("navigation_skill"),
    )
    feedback = {
        "step": step,
        "action_step": step,
        "policy_step": safe_decision.get("policy_step"),
        "action": action,
        "executed": bool(safe_execution.get("executed", False)),
        "physical_contact": bool(safe_execution.get("physical_contact", False)),
        "physical_contact_details": safe_execution.get("physical_contact_details")
        or [],
        "contact_reference": safe_execution.get("contact_reference"),
        "robot_message": robot_message,
        "feedback": str(safe_execution.get("feedback", "") or ""),
        # Runner-owned delivery/question outcome, independent of physical
        # feedback so contact or motion summaries cannot suppress it.
        "interaction_feedback": str(
            safe_execution.get("interaction_feedback", "") or ""
        ),
        "motion_outcome": dict(motion_outcome or {}),
        "body_stability": body_stability,
        "balance_control": dict(balance_control),
        "post_motion_settle_ticks": int(
            safe_execution.get("post_motion_settle_ticks") or 0
        ),
        "call_name": safe_decision.get("call_name"),
        "controller_backend": safe_decision.get("controller_backend"),
        # The current G1 manipulation contract has one certified working hand:
        # the right Dex3 hand. Successful execution remains authoritative.
        "hand": grasp_skill.get("hand") or safe_decision.get("hand"),
        "target": safe_decision.get("target"),
        "object_id": grasp_skill.get("object_id"),
        # The Agent receives the actionable outcome only. Controller phases,
        # contact manifolds, object ids, and retention internals stay in the
        # evaluator/debug trace below.
        "grasp_failure": public_grasp_failure,
        # Current ego RGB plus tool-private depth and cuRobo measurements from
        # a refused grasp. This is ego-relative reach evidence, never a world
        # coordinate or hidden destination, and lets the next Agent decision
        # improve its stance.
        "grasp_reachability": dict(grasp_reachability),
        # Successful one-call manipulation reports the completed internal
        # stages explicitly, so the next policy decision can begin transport
        # without guessing whether the arm is still extended or the body is
        # still crouched.
        "grasp_skill": dict(grasp_skill),
        # Dex3 hand-target contacts are target-specific manipulation evidence.
        # They are separate from locomotion ``physical_contact_details`` so
        # remembered grounding can invalidate only when the grasp object moved.
        "grasp_contact_details": list(
            safe_execution.get("grasp_contact_details") or []
        ),
        # Full placement geometry remains evaluator-private. The runtime owns
        # the public Boolean; report presence is not a diagnostic check.
        "placed": placed,
    }
    if navigation_feedback:
        feedback["navigation_execution"] = navigation_feedback
    displacement_m = safe_execution.get("grounding_displacement_m")
    if displacement_m is not None:
        feedback["grounding_displacement_m"] = float(displacement_m)
    dropped_objects = safe_execution.get("dropped_objects") or []
    if dropped_objects:
        if not isinstance(dropped_objects, list):
            raise TypeError("dropped_objects must be a list when present.")
        feedback["dropped_objects"] = [
            str(object_id) for object_id in dropped_objects if str(object_id)
        ]
    return feedback


def build_trace_observation(
    *,
    observation_id: str,
    step: int,
    timing: str,
    observation: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a compact public observation record for result serialization."""
    safe_observation = sanitize_agent_observation(observation)
    record = {
        "id": observation_id,
        "step": step,
        "timing": timing,
    }
    for key in OBSERVATION_LOG_KEYS:
        if key in safe_observation:
            value = safe_observation[key]
            if key == "text":
                value = str(value).strip()
            record[key] = value
    camera_params = safe_observation.get("camera_params")
    if not isinstance(camera_params, dict):
        ego = (safe_observation.get("views") or {}).get("ego") or {}
        camera_params = ego.get("camera_params")
    if isinstance(camera_params, dict):
        capture_clock = {
            key: int(camera_params[key])
            for key in ("capture_id", "physics_step")
            if camera_params.get(key) is not None
        }
        if capture_clock:
            record["camera_capture"] = capture_clock
    return record


class AutonomousEpisodeRunner:
    """Run an episode without phase scripts or target injection."""

    def __init__(
        self,
        episode: TARSphereEpisode,
        environment: Any,
        agent: Any,
        human: Any,
        event_bus: Optional[EventBus] = None,
        collector: Optional[Any] = None,
        abort_requested: Optional[Callable[[], bool]] = None,
        heartbeat_path: Optional[Path] = None,
        model_output_path: Optional[Path] = None,
        progress_watchdog: Optional[ProgressWatchdog] = None,
        agent_harness_config: AgentHarnessConfig | None = None,
    ):
        self.episode = episode
        self.environment = environment
        self.agent = agent
        self.human = human
        self._disruption = getattr(human, "_disruption", DisruptionRuntime(episode))
        self.event_bus = event_bus or EventBus()
        # Optional private side-channel for offline trajectory analysis.
        self.collector = collector
        self.abort_requested = abort_requested or (lambda: False)
        self.heartbeat_path = Path(heartbeat_path) if heartbeat_path else None
        self.model_output_path = Path(model_output_path) if model_output_path else None
        self.progress_watchdog = progress_watchdog
        agent_config = getattr(agent, "agent_harness_config", None)
        self.agent_harness_config = (
            agent_harness_config or agent_config or AgentHarnessConfig()
        )
        if agent_config is not None and agent_config != self.agent_harness_config:
            raise ValueError(
                "runner and agent received different AgentHarnessConfig values"
            )
        configure_agent_harness = getattr(environment, "configure_agent_harness", None)
        if callable(configure_agent_harness):
            configure_agent_harness(self.agent_harness_config)
        self.trace: List[Dict[str, Any]] = []
        self._policy_decisions = 0
        self.heartbeats: List[Dict[str, Any]] = []
        self.last_info: Dict[str, Any] = {}
        self.last_condition_results: Dict[str, bool] = {}
        self.safety_violations: List[Dict[str, Any]] = []
        self._evaluation_steps: List[TrajectoryStep] = []
        self._condition_history: List[Dict[str, bool]] = []
        # Raw pre-decision visual evidence is retained by action step until a
        # policy call declares which camera records were actually assembled
        # into its prompt. This prevents queued, model-unseen frames from
        # earning visual progress.
        self._pre_decision_reaction_samples: Dict[
            int,
            tuple[Dict[str, bool], Dict[str, Dict[str, Any]]],
        ] = {}
        self.termination_reason = "not_started"
        # Post-reset and post-action world-XY positions used only by hidden
        # diagnostic checks that require an ordered task stop to be visited.
        self._robot_position_history: List[tuple[float, float]] = []
        # Public utterances the Humans have produced since the agent last
        # observed. Delivered exactly once, on the next observation the agent
        # receives, so a line spoken mid-step is neither lost nor repeated on
        # every subsequent prompt.
        self._pending_public_events: List[Dict[str, Any]] = []
        self._terminal_boundary: dict[str, Any] | None = None
        self._completion_feedback_delivered = False
        # Model calls become billable/auditable as soon as policy inference
        # returns, before the selected physical action runs.  Keep that call
        # set here until the corresponding completed action is committed to
        # ``trace``.  If env.step(), evidence collection, or cancellation
        # fails first, partial_result must still count the real API traffic.
        self._uncommitted_model_calls: List[Dict[str, Any]] = []
        # Fail-loud friendly: collector/runner exceptions are recorded here
        # instead of being silently swallowed.
        if not callable(getattr(human, "tick", None)) or not hasattr(
            human, "human_ids"
        ):
            raise TypeError(
                "AutonomousEpisodeRunner requires a milestone-driven human "
                "runtime with tick() and human_ids."
            )

    def _stage_spatial_blocker_before_first_policy(
        self,
        observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Put an authored route blocker in place before the Agent can move.

        Spatial blocking is an initial world condition.  The Human still
        spawns at the collision-safe authored start pose and follows the real
        Human motion backend to the blocking pose, but that scene setup is
        completed before the first policy observation.  Otherwise a long
        navigation call on decision zero can cross the choke point before the
        blocker starts moving on decision one.
        """

        profile = getattr(self.episode.evaluation, "task_profile", None)
        if not isinstance(profile, SpatialBlockingProfile):
            return observation
        milestone = next(
            node
            for node in self.episode.evaluation.milestones
            if node.milestone_id == profile.blocking_milestone_id
        )
        actions = [
            dict(action)
            for action in milestone.event.human_actions
            if (
                action.get("command") == "goto"
                and str(action.get("human") or "") == profile.blocking_human_id
            )
        ]
        if len(actions) != 1:
            raise RuntimeError(
                "Spatial blocker setup requires exactly one authored goto."
            )
        apply_action = getattr(self.environment, "apply_human_action", None)
        if not callable(apply_action):
            raise RuntimeError("Spatial blocking requires a Human action runtime API.")
        setup_action = dict(actions[0])
        setup_action.pop("human", None)
        start_physics_step = getattr(self.environment, "_physics_steps", None)
        result = apply_action(profile.blocking_human_id, setup_action)
        if result is None or result.executed is not True:
            detail = "missing execution result" if result is None else result.feedback
            raise RuntimeError(
                f"Could not stage spatial blocker before the first policy: {detail}"
            )
        motion_state = result.state
        if motion_state is None:
            raise RuntimeError(
                "Spatial blocker setup did not return observable Human action state."
            )

        for _ in range(120):
            status = str(motion_state.action_status or "").lower()
            if status == "succeed" and not motion_state.pose_stale:
                break
            if status not in {"ongoing", "succeed"}:
                raise RuntimeError(
                    "Spatial blocker motion ended without success: "
                    f"status={status or 'missing'!r}, "
                    f"error={motion_state.last_error or 'none'!r}."
                )
            observation, done, _ = self.environment.step(
                {
                    "action": "wait",
                    "seconds": 0.5,
                }
            )
            if done:
                raise RuntimeError(
                    "Environment terminated while staging the spatial blocker."
                )
        else:
            raise RuntimeError(
                "Spatial blocker did not reach its authored blocking pose "
                "within 60 simulated seconds."
            )

        mark_prepared = getattr(
            self.human,
            "mark_prepared_milestone_completed",
            None,
        )
        if not callable(mark_prepared):
            raise RuntimeError(
                "Human runtime cannot record the prepared blocking event."
            )
        mark_prepared(
            milestone_id=profile.blocking_milestone_id,
            human_id=profile.blocking_human_id,
            step=0,
            start_physics_step=start_physics_step,
            physics_step=getattr(self.environment, "_physics_steps", None),
        )
        observe = getattr(self.environment, "observe", None)
        return observe() if callable(observe) else observation

    def run(
        self,
        max_decision_steps: Optional[int] = None,
        initial_observation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        action_space = resolve_agent_action_space(
            self.episode,
            self.agent_harness_config,
        )
        # ``None`` preserves the episode-authored interface, which may itself
        # be unbounded. A non-positive explicit override requests unbounded
        # decisions; simulated-time fairness is controlled independently by
        # ``runtime.max_physics_steps``.
        if max_decision_steps is None:
            authored_max_steps = self.episode.runtime.max_decision_steps
            max_steps = None if authored_max_steps is None else int(authored_max_steps)
        else:
            requested_steps = int(max_decision_steps)
            max_steps = requested_steps if requested_steps > 0 else None
        max_physics_steps = self.episode.runtime.max_physics_steps
        if getattr(self.agent, "name", None) == "harness_reference":
            if max_decision_steps is not None and max_decision_steps > 0:
                raise ValueError("Harness reference has no execution-step budget.")
            # Reference measures physical work, not policy efficiency. Episode
            # budgets belong to evaluated policies; safety/abort checks remain.
            max_steps = None
            max_physics_steps = None

        self.trace = []
        self._policy_decisions = 0
        self.heartbeats = []
        self.last_info = {}
        self.last_condition_results = {}
        self.safety_violations = []
        self._evaluation_steps = []
        self._condition_history = []
        self._pre_decision_reaction_samples = {}
        self.termination_reason = "running"
        self._robot_position_history = []
        if self.progress_watchdog is not None:
            self.progress_watchdog.reset()
        if self.heartbeat_path is not None:
            self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
            self.heartbeat_path.write_text("", encoding="utf-8")
        if self.model_output_path is not None:
            self.model_output_path.parent.mkdir(parents=True, exist_ok=True)
            self.model_output_path.write_text("", encoding="utf-8")
        set_abort_checker = getattr(self.environment, "set_abort_checker", None)
        if callable(set_abort_checker):
            set_abort_checker(self.abort_requested)
        set_physics_step_limit = getattr(
            self.environment, "set_physics_step_limit", None
        )
        if callable(set_physics_step_limit):
            set_physics_step_limit(max_physics_steps)
        self._raise_if_aborted()

        try:
            observation = (
                initial_observation
                if initial_observation is not None
                else self.environment.reset()
            )
        except Exception as exc:
            raise EpisodeInitializationError(
                f"Episode {self.episode.episode_id!r} initialization failed: {exc}"
            ) from exc
        self._record_robot_position(observation)
        self.agent.reset()
        configure_input_capture = getattr(
            self.agent, "configure_model_input_capture", None
        )
        if callable(configure_input_capture):
            configure_input_capture(
                self.model_output_path.with_suffix(".inputs")
                if self.model_output_path is not None
                else None
            )
        configure_episode = getattr(self.agent, "configure_episode", None)
        if callable(configure_episode):
            configure_episode(self.episode)
        tool_registry = getattr(self.agent, "tools", None)
        configure_disruption = getattr(tool_registry, "configure_disruption", None)
        if isinstance(self._disruption.profile, ComponentOutageProfile):
            if not callable(configure_disruption):
                raise RuntimeError(
                    "Component outage requires an Agent ToolRegistry so the "
                    "failed call is resolved before environment execution."
                )
            if not hasattr(self.environment, "_physics_steps"):
                raise RuntimeError(
                    "Component outage requires a simulator physics clock"
                )
            configure_disruption(
                lambda name: self._disruption.component_unavailable(
                    name,
                    int(self.environment._physics_steps),
                ),
            )
        elif callable(configure_disruption):
            configure_disruption(lambda _name: False)
        configure_mapping = getattr(
            self.environment,
            "configure_private_mapping_observer",
            None,
        )
        if callable(configure_mapping):
            mapping_observer = getattr(
                self.agent,
                "observe_private_mapping_frame",
                None,
            )
            configure_mapping(
                observer=(mapping_observer if callable(mapping_observer) else None),
                action_space=action_space,
            )
        self.agent.configure_episode_humans(list(self.episode.humans))
        self.human.reset()
        self._disruption.reset()
        observation = self._stage_spatial_blocker_before_first_policy(
            observation,
        )
        self.event_bus.clear()
        self._pending_public_events.clear()
        self._terminal_boundary = None
        self._completion_feedback_delivered = False
        self._uncommitted_model_calls.clear()

        trace = self.trace
        success = False
        condition_results: Dict[str, bool] = {}
        previous_decision: Dict[str, Any] = {}

        decision_step = 0
        # Trace/event indices count completed action outcomes; only a fresh
        # policy inference consumes the cognition budget. Drain the last
        # accepted block, including lazily resolved calls, before stopping.
        while (
            max_steps is None
            or self._policy_decisions < max_steps
            or agent_has_pending_actions(self.agent)
        ):
            self._raise_if_aborted()
            self.event_bus.step = decision_step
            # Milestone events are selected inside the Human tick from this
            # exact pre-tick world state. Sample their live primitive
            # preconditions before a newly scheduled Human action can alter
            # pose or visibility in the same decision step.
            event_condition_results: Dict[str, bool] = {}
            if getattr(self.episode, "evaluation", None) is not None:
                event_condition_results = self._sample_evaluation_conditions(
                    previous_decision,
                    observation,
                )
                self.human.observe_milestone_conditions(event_condition_results)
            observation, human_events = self._tick_human_actors(
                decision_step, observation
            )
            _runner_memory_diagnostic(
                self.environment, "after_human_tick", decision_step
            )
            pre_observation_id = f"step_{decision_step:03d}_pre_agent"
            perception_observation = observation
            perception_world_snapshot = self._get_world_snapshot(step=decision_step)
            pre_task_bindings = (
                self.human.task_bindings.snapshot()
                if self.human.task_bindings is not None
                else None
            )
            _runner_memory_diagnostic(
                self.environment, "after_world_snapshot", decision_step
            )
            self._update_private_oracle_snapshot(perception_world_snapshot)
            # Only the explicit GT reference policy receives task progress.
            # Ordinary RobotAgent inputs remain sanitized and unchanged.
            if str(getattr(self.agent, "name", "")) == "harness_reference":
                self.agent.update_reference_state(
                    snapshot=perception_world_snapshot,
                    opened=tuple(
                        dict.fromkeys(
                            (
                                *self._milestone_events_at_step(decision_step),
                                *(
                                    milestone.milestone_id
                                    for milestone, reach in zip(
                                        self.episode.evaluation.milestones,
                                        milestone_dag_reaches(
                                            self.episode.evaluation,
                                            self._evaluation_steps,
                                        ),
                                    )
                                    if reach.event_index is not None
                                ),
                            )
                        )
                    ),
                    completed=completed_milestone_ids(
                        self.episode.evaluation,
                        self._evaluation_steps,
                    ),
                )
            agent_observation = sanitize_agent_observation(observation)
            agent_observation["public_events"] = self._drain_public_events()
            agent_observation["visible_humans"] = (
                self._visible_humans_for_agent_memory()
            )
            reaction_evidence: dict[str, dict[str, Any]] = {}
            raw_pre_decision_reactions: Dict[str, bool] = {}
            if getattr(self.episode, "evaluation", None) is not None:
                raw_pre_decision_evidence: dict[str, dict[str, Any]] = {}
                raw_pre_decision_reactions = self._sample_evaluation_reactions(
                    {},
                    perception_observation,
                    phase="pre_decision",
                    evidence_sink=raw_pre_decision_evidence,
                )
                self._pre_decision_reaction_samples[int(decision_step)] = (
                    dict(raw_pre_decision_reactions),
                    dict(raw_pre_decision_evidence),
                )
            decision_requirements = self._event_decision_requirements(
                decision_step=decision_step,
            )
            requires_inference = getattr(self.agent, "requires_policy_inference", None)
            if (
                max_steps is not None
                and self._policy_decisions >= max_steps
                and callable(requires_inference)
                and requires_inference(
                    decision_requirements, observation=agent_observation
                )
            ):
                discarded = self._discard_agent_pending_actions(
                    "decision_budget_exhausted"
                )
                if trace:
                    prior_discarded = list(
                        trace[-1].get("discarded_action_details") or ()
                    )
                    trace[-1]["discarded_action_details"] = [
                        *prior_discarded,
                        *discarded,
                    ]
                    trace[-1]["discarded_actions"] = len(
                        trace[-1]["discarded_action_details"]
                    )
                # Delivery at the terminal boundary has no subsequent action
                # to score. Keep it separate from earlier action evidence.
                self._terminal_boundary = {
                    "action_step": decision_step,
                    "physics_step": int(self.environment._physics_steps),
                    "public_events": list(agent_observation["public_events"]),
                    "human_events": list(human_events),
                    "task_bindings": pre_task_bindings,
                }
                self.termination_reason = "decision_budget_exhausted"
                break
            update_agent_progress = getattr(
                self.agent,
                "update_episode_progress",
                None,
            )
            if callable(update_agent_progress):
                evaluation = getattr(self.episode, "evaluation", None)
                update_agent_progress(
                    completed_milestone_ids(
                        evaluation,
                        self._evaluation_steps,
                    )
                    if evaluation is not None
                    else ()
                )
            _runner_memory_diagnostic(
                self.environment, "before_agent_act", decision_step
            )
            decision = self.agent.act(
                observation=agent_observation,
                instruction=self.episode.system_prompt,
                action_space=action_space,
                decision_requirements=decision_requirements,
            )
            policy_inference = decision.get("policy_inference", True)
            if type(policy_inference) is not bool:
                raise TypeError("Agent policy_inference must be a boolean")
            if policy_inference:
                self._policy_decisions += 1
                if max_steps is not None and self._policy_decisions > max_steps:
                    raise RuntimeError(
                        "Agent inferred while draining its final action block"
                    )
            decision.setdefault(
                "agent_harness_config", self.agent_harness_config.as_dict()
            )
            decision.setdefault("diagnostic_only", False)
            _runner_memory_diagnostic(
                self.environment, "after_agent_act", decision_step
            )
            action = decision.get("action")

            if not action:
                raise ValueError(f"Agent decision missing action: {decision!r}")
            if (
                action not in action_space
                and not decision_has_compiled_controller(decision)
                and not bool(decision.get("is_error"))
            ):
                raise ValueError(
                    f"Agent decision action {action!r} is not in the episode "
                    f"action space: {action_space!r}. decision={decision!r}"
                )
            decision["action"] = action

            decision_reactions: Dict[str, bool] = {}
            pre_decision_reactions: Dict[str, bool] = {}
            if getattr(self.episode, "evaluation", None) is not None:
                pre_decision_reactions = (
                    self._commit_model_visible_pre_decision_reactions(
                        decision=decision,
                        decision_step=decision_step,
                        evidence_sink=reaction_evidence,
                    )
                )
                decision_reactions = self._sample_evaluation_reactions(
                    {} if bool(decision.get("is_error")) else decision,
                    perception_observation,
                    phase="decision",
                    evidence_sink=reaction_evidence,
                    preceding_reactions=pre_decision_reactions,
                    decision_step=decision_step,
                    event_conditions=event_condition_results,
                )

            raw_model_calls = decision.get("model_calls") or []
            if not isinstance(raw_model_calls, (list, tuple)):
                raise TypeError("decision.model_calls must be a list")
            self._uncommitted_model_calls = [dict(call) for call in raw_model_calls]

            # Persist the authoritative model output before the physical
            # action starts.  Heartbeats are necessarily post-action records;
            # this independent stream survives an environment-step crash and
            # exposes reasoning/tool calls while a long episode is active.
            if self.model_output_path is not None and bool(
                decision.get("policy_inference", True)
            ):
                envelope = decision.get("envelope")
                if not isinstance(envelope, dict):
                    raise TypeError(
                        "policy inference must expose its DecisionEnvelope as "
                        "a mapping before model-output persistence"
                    )
                write_heartbeat(
                    self.model_output_path,
                    {
                        "event": "policy_decision",
                        "decision_step": int(decision_step),
                        "action_step": int(decision.get("action_step", decision_step)),
                        "policy_step": decision.get("policy_step"),
                        "replan_reason": decision.get("replan_reason"),
                        "action_index": decision.get("action_index"),
                        "action_sequence_length": decision.get(
                            "action_sequence_length"
                        ),
                        "selected_action": action,
                        "call_name": decision.get("call_name"),
                        "envelope": envelope,
                        "tool_results": list(decision.get("tool_results") or []),
                        "model_calls": list(decision.get("model_calls") or []),
                    },
                )

            if not bool(decision.get("is_error")):
                self.event_bus.emit_action(action, "robot", **decision)
            _runner_memory_diagnostic(
                self.environment, "after_event_emit", decision_step
            )
            _runner_memory_diagnostic(
                self.environment, "before_environment_step", decision_step
            )
            # A measured fall is a benchmark outcome, not an infrastructure
            # crash. Preserve the measured terminal frame and contact window;
            # the pre-action frame would fabricate zero displacement/no contact.
            observation, env_done, info = _execute_navigation_or_action(
                self.environment,
                self.agent,
                decision,
                observation,
                self._disruption,
            )
            self.last_info = dict(info or {})
            raw_dropped_objects = self.last_info.get("dropped_objects", ())
            if not isinstance(raw_dropped_objects, (list, tuple)):
                raise TypeError("execution.dropped_objects must be a list")
            dropped_object_ids = tuple(
                str(object_id).strip() for object_id in raw_dropped_objects
            )
            if any(not object_id for object_id in dropped_object_ids):
                raise ValueError(
                    "execution.dropped_objects must contain non-empty object ids"
                )
            self._raise_if_aborted()
            _runner_memory_diagnostic(
                self.environment, "after_environment_step", decision_step
            )
            observe_robot_action = getattr(self.human, "observe_robot_action", None)
            if not callable(observe_robot_action):
                raise RuntimeError(
                    "Human runtime does not implement required Robot action "
                    "observation."
                )
            if not bool(decision.get("is_error")):
                observe_robot_action(decision, step=decision_step)

            decision_message = str(decision.get("message", "")).strip()
            spoken_message = ""
            robot_message_delivered = False
            # handover_object is communicative in the taxonomy but its public
            # line is the recipient's reply, emitted in its own branch below.
            if (
                not bool(decision.get("is_error"))
                and action in COMMUNICATIVE_ACTIONS
                and action != "handover_object"
                and decision_message
            ):
                spoken_message = decision_message
                self.event_bus.emit_language(action, "robot", spoken_message)
                # Reserve chronological order before the reply; attach the
                # actual delivery result before the next observation drains it.
                robot_utterance = {
                    "event_type": "language",
                    "speaker": "robot",
                    "text": spoken_message,
                    "step": int(decision_step),
                    "recipient": str(decision.get("recipient") or "").strip(),
                }
                self._pending_public_events.append(robot_utterance)
                (
                    observation,
                    reply_events,
                    interaction_feedback,
                    robot_message_delivered,
                ) = self._tick_human_actors_for_robot_message(
                    decision=decision,
                    spoken_message=spoken_message,
                    observation=observation,
                    step=decision_step,
                )
                robot_utterance["delivered"] = robot_message_delivered
                human_events.extend(reply_events)
                if interaction_feedback:
                    info["interaction_feedback"] = interaction_feedback
                    self.last_info = dict(info)
            if action == "handover_object" and not bool(decision.get("is_error")):
                # A handover is a verbal claim, never a physical transfer, so
                # the recipient's reply is the only evidence it happened. An
                # empty reply or an unattributed speaker would leave the
                # episode with no record at all rather than a failed one.
                handover_reply = str((info or {}).get("feedback") or "").strip()
                if not handover_reply:
                    raise RuntimeError(
                        "handover_object produced no feedback; a verbal "
                        "handover must yield the recipient reply that is its "
                        "only evidence."
                    )
                unavailable_speaker = str(
                    info.get("temporary_unavailability_human_id") or ""
                )
                utterance = getattr(self.environment, "last_utterance", None)
                handover_record = info.get("handover")
                resolved_human_id = (
                    str(handover_record.get("resolved_human_id") or "").strip()
                    if isinstance(handover_record, dict)
                    else ""
                )
                speaker = unavailable_speaker or (
                    str(utterance.get("speaker") or "").strip()
                    if (
                        resolved_human_id
                        and isinstance(utterance, dict)
                        and str(utterance.get("text") or "").strip() == handover_reply
                    )
                    else ""
                )
                if not speaker:
                    # The formal tool surface is task-agnostic. A model may
                    # offer an object to an unknown description, in which case
                    # the deterministic handover verdict is
                    # ``target_unresolved`` and no Human can truthfully own a
                    # reply. The execution feedback still records the refusal;
                    # do not fabricate a speaker or turn the model's bad target
                    # choice into an infrastructure-invalid episode.
                    pass
                elif unavailable_speaker:
                    reply_action = {
                        "command": "speak",
                        "message": handover_reply,
                        "interaction_response_mode": "temporarily_unavailable",
                    }
                    observation, emitted = self._apply_human_action(
                        reply_action,
                        observation,
                        speaker,
                    )
                    human_events.append(
                        self._human_event(
                            speaker,
                            reply_action,
                            emitted,
                            timing="after_agent",
                        )
                    )
                else:
                    # The environment already rendered the accepted/rejected
                    # handover reply; do not speak or append it a second time.
                    self._pending_public_events.append(
                        {
                            "event_type": "language",
                            "speaker": speaker,
                            "text": handover_reply,
                            "step": int(decision_step),
                        }
                    )
                    self.event_bus.emit_language(
                        "speak",
                        f"human:{speaker}",
                        handover_reply,
                    )
                    human_events.append(
                        {
                            "human_id": speaker,
                            "action": "speak",
                            "message": handover_reply,
                            "timing": "after_agent",
                        }
                    )

            self._record_robot_position(observation)
            if getattr(self.episode, "evaluation", None) is not None:
                post_action_conditions = self._sample_evaluation_conditions(
                    decision,
                    observation,
                )
                if set(event_condition_results) != set(post_action_conditions):
                    raise ValueError(
                        "evaluation condition schema changed within a decision step"
                    )
                # The trajectory row owns the event-time sample. The
                # post-action sample is published only to unlock the following
                # Human tick; OR-ing the two would permit future state to
                # retroactively authorize an earlier event.
                condition_results = dict(event_condition_results)
                post_action_reactions = self._sample_evaluation_reactions(
                    decision,
                    observation,
                    phase="post_action",
                    evidence_sink=reaction_evidence,
                    decision_step=decision_step,
                    event_conditions=event_condition_results,
                    preceding_reactions=_merge_reaction_samples(
                        pre_decision_reactions,
                        decision_reactions,
                    ),
                    human_events=human_events,
                )
                reaction_results = _merge_reaction_samples(
                    pre_decision_reactions,
                    decision_reactions,
                    post_action_reactions,
                )
                observe_conditions = getattr(
                    self.human,
                    "observe_milestone_conditions",
                    None,
                )
                if not callable(observe_conditions):
                    raise RuntimeError(
                        "Human runtime does not accept milestone conditions."
                    )
                observe_conditions(post_action_conditions)
                self._condition_history.append(dict(condition_results))
                # Success is resolved after this complete trajectory step is
                # appended, so event occurrence and reaction evidence share
                # one authoritative record.
                success = False
            else:
                success, condition_results = self._evaluate_diagnostics(action)
            if dropped_object_ids:
                # Dropping a transported object dominates any success
                # condition reached during the same physical action.
                success = False
            self.last_condition_results = dict(condition_results)

            # Record evaluator-private physical evidence after env.step and
            # any Human reply. The typed EpisodeTrajectory below remains the
            # official causal-progress and attribution input.
            post_world_snapshot = self._get_world_snapshot(step=decision_step)
            disruption_state = self._disruption.trace_state(
                int(post_world_snapshot.get("physics_steps") or 0),
            )
            safety_violation = (info or {}).get("safety_violation")
            if (
                isinstance(safety_violation, dict)
                and safety_violation.get("type") == "robot_fall"
            ):
                execution_safety = dict(
                    post_world_snapshot.get("execution_safety") or {}
                )
                execution_safety.update(
                    {
                        "fallen": True,
                        "robot_fall": dict(info["safety_violation"]),
                    }
                )
                post_world_snapshot["execution_safety"] = execution_safety
            if self.collector is not None:
                # A rollout without its required evidence is not a lower Agent
                # score; it is an invalid benchmark run. Let collection
                # failures terminate through the runner's ordinary fail-loud
                # result path instead of silently continuing with a partial
                # trace.
                self.collector.record_step(
                    step=decision_step,
                    observation=observation,
                    perception_observation=perception_observation,
                    decision=decision,
                    world_snapshot=post_world_snapshot,
                    perception_world_snapshot=perception_world_snapshot,
                    placement_release=info.get("placement_release"),
                    handover=info.get("handover"),
                    travelled_distance_m=(
                        info["navigation_skill"].get("travelled_distance_m")
                        if info.get("navigation_skill") is not None
                        else info.get("travelled_distance_m")
                    ),
                )
            motion_outcome = measure_locomotion_outcome(
                action=action,
                decision=decision,
                pose_before=perception_observation.get("pose"),
                pose_after=observation.get("pose"),
            )
            agent_feedback = build_agent_feedback(
                step=decision_step,
                action=action,
                robot_message=spoken_message,
                execution=info,
                observation=observation,
                decision=decision,
                motion_outcome=motion_outcome,
            )
            post_observation_id = f"step_{decision_step:03d}_post_step"
            # Persist the full structured decision envelope so the trace shows
            # the exact calls and event-window outputs the Agent emitted.
            envelope_dict = decision.get("envelope")
            trace.append(
                {
                    "step": decision_step,
                    "action_step": decision_step,
                    "policy_step": decision.get("policy_step"),
                    "policy_inference": bool(decision.get("policy_inference", True)),
                    "model_visible_action_steps": list(
                        decision.get("model_visible_action_steps") or []
                    ),
                    "action_index": decision.get("action_index"),
                    "action_sequence_length": decision.get("action_sequence_length"),
                    "replan_reason": decision.get("replan_reason"),
                    "discarded_actions": int(decision.get("discarded_actions") or 0),
                    "tool_call_index": int(
                        decision.get("tool_call_index", decision.get("action_index", 0))
                        or 0
                    ),
                    "discarded_action_details": list(
                        decision.get("discarded_action_details") or []
                    ),
                    "robot_action": action,
                    "robot_message": spoken_message or "",
                    "call_name": decision.get("call_name"),
                    "envelope": envelope_dict,
                    "accepted_event_output_batches": list(
                        decision.get("accepted_event_output_batches", [])
                    ),
                    "tool_results": decision.get("tool_results", []),
                    "model_calls": decision.get("model_calls", []),
                    "human_events": human_events,
                    "executed": bool(agent_feedback.get("executed", False)),
                    "physical_contact": bool(agent_feedback.get("physical_contact")),
                    "physical_contact_details": agent_feedback.get(
                        "physical_contact_details"
                    )
                    or [],
                    # Exact collider paths stay private, including on an aborted
                    # rollout; never inject this diagnostic evidence into prompts.
                    "native_contacts": list(info.get("native_contacts") or []),
                    "navigation_trace": list(info.get("navigation_trace") or []),
                    "navigation_skill": dict(info.get("navigation_skill") or {}),
                    "disruption_state": disruption_state,
                    "retention_contacts": list(info.get("retention_contacts") or []),
                    # Native hand-target contacts are evaluator-private. They are
                    # deliberately separate from public locomotion contact feedback
                    # consumed by Agent memory.
                    "grasp_contact_details": info.get("grasp_contact_details") or [],
                    "grasp_failure_evidence": dict(
                        (info.get("grasp_failure") or {}).get("evidence") or {}
                    ),
                    # Physical execution and interaction delivery are independent
                    # outcomes; neither may suppress the other's public evidence.
                    # Native PhysX contact is evidence from the completed motion;
                    # it never predicts, rejects, or shortens a requested leg.
                    "execution_feedback": str(info.get("feedback") or ""),
                    "interaction_feedback": agent_feedback["interaction_feedback"],
                    "motion_outcome": motion_outcome,
                    "navigation_execution": agent_feedback.get("navigation_execution")
                    or {},
                    "body_stability": agent_feedback.get("body_stability") or {},
                    "balance_control": agent_feedback.get("balance_control") or {},
                    "grasp_failure": agent_feedback.get("grasp_failure") or {},
                    "grasp_skill": agent_feedback.get("grasp_skill") or {},
                    "dropped_objects": list(dropped_object_ids),
                    "post_motion_settle_ticks": int(
                        agent_feedback.get("post_motion_settle_ticks") or 0
                    ),
                    "images": {
                        "pre": pre_observation_id,
                        "post": post_observation_id,
                    },
                }
            )
            # The same records now have an authoritative owner in ``trace``;
            # clearing here prevents normal runs from double-counting them.
            self._uncommitted_model_calls.clear()
            if getattr(self.episode, "evaluation", None) is not None:
                evaluation_observation = build_trace_observation(
                    observation_id=pre_observation_id,
                    step=decision_step,
                    timing="pre_agent",
                    observation=perception_observation,
                )
                evaluation_observation["public_events"] = list(
                    agent_observation.get("public_events") or []
                )
                self._evaluation_steps.append(
                    TrajectoryStep(
                        decision_step=decision_step,
                        policy_inference=policy_inference,
                        physics_step=int(info.get("physics_steps") or 0),
                        observation=evaluation_observation,
                        agent_state=_evaluation_agent_state(decision),
                        action={
                            "name": action,
                            "tool": decision.get("call_name"),
                            "args": dict(decision.get("tool_args") or {}),
                            "tool_call_index": int(
                                decision.get(
                                    "tool_call_index",
                                    decision.get("action_index", 0),
                                )
                                or 0
                            ),
                            "call_name": decision.get("call_name"),
                            "recipient": decision.get("recipient"),
                            "intent_id": decision.get("intent_id"),
                            "message": decision.get("message"),
                            "model_visible_action_steps": list(
                                decision.get("model_visible_action_steps") or []
                            ),
                            "calls": list((envelope_dict or {}).get("actions") or []),
                        },
                        feedback=dict(agent_feedback),
                        hidden_state={
                            "task_bindings": {
                                "pre_decision": pre_task_bindings,
                                "post_action": self.human.task_bindings.snapshot(),
                            },
                            "pre_decision_world": perception_world_snapshot,
                            # Actual addressed speech delivery at the post-action
                            # Human tick, distinct from pre-decision reachability.
                            "robot_message_delivered": robot_message_delivered,
                            "world": post_world_snapshot,
                            "disruption_state": disruption_state,
                            "robot": dict(observation.get("pose") or {}),
                            "dropped_objects": list(dropped_object_ids),
                            "resolved_action_targets": {
                                key: value
                                for key, value in {
                                    "object_id": _nested_value(
                                        info,
                                        "grasp_skill",
                                        "object_id",
                                    )
                                    or _nested_value(
                                        info,
                                        "grasp_failure",
                                        "evidence",
                                        "object_id",
                                    ),
                                    "human_ref": _nested_value(
                                        decision,
                                        "recipient",
                                    ),
                                }.items()
                                if value is not None
                            },
                            "human_events": list(human_events),
                            "placement_release": dict(
                                info.get("placement_release") or {}
                            ),
                        },
                        condition_values=dict(condition_results),
                        reaction_values=dict(reaction_results),
                        reaction_evidence=dict(reaction_evidence),
                        occurred_events=self._milestone_events_at_step(decision_step),
                        admitted_events=self._milestone_admissions_at_step(
                            decision_step
                        ),
                    )
                )
                evaluation = self.episode.evaluation
                # Internal phases have no NPC action to deliver. Record their
                # derived causal time in the same row, without a second tick.
                current_row = self._evaluation_steps[-1]
                internal_events = tuple(
                    milestone.milestone_id
                    for milestone, reach in zip(
                        evaluation.milestones,
                        milestone_dag_reaches(evaluation, self._evaluation_steps),
                    )
                    if is_internal_task_phase(milestone)
                    and reach.event_index == len(self._evaluation_steps) - 1
                )
                self._evaluation_steps[-1] = current_row.model_copy(
                    update={
                        "occurred_events": tuple(
                            dict.fromkeys(
                                (
                                    *current_row.occurred_events,
                                    *internal_events,
                                )
                            )
                        ),
                    }
                )
                completed = completed_milestone_ids(
                    evaluation,
                    self._evaluation_steps,
                )
                observe_progress = getattr(
                    self.human,
                    "observe_completed_milestones",
                    None,
                )
                if not callable(observe_progress):
                    raise RuntimeError(
                        "Human runtime does not accept milestone progress."
                    )
                observe_progress(completed)
                success = causal_graph_completed(
                    evaluation,
                    self._evaluation_steps,
                )
                robot_fell = (info.get("safety_violation") or {}).get(
                    "type"
                ) == "robot_fall"
                if dropped_object_ids or robot_fell:
                    success = False
                if (
                    success
                    and not dropped_object_ids
                    and not robot_fell
                    and not self._completion_feedback_delivered
                ):
                    observation, completion_events = (
                        self._emit_task_completion_feedback(observation)
                    )
                    human_events.extend(completion_events)
                    # Completion feedback is part of this action's observable
                    # causal closure, not an unrecorded postscript.
                    trace[-1]["human_events"] = list(human_events)
                    latest_step = self._evaluation_steps[-1]
                    latest_hidden = dict(latest_step.hidden_state)
                    latest_hidden["human_events"] = list(human_events)
                    self._evaluation_steps[-1] = latest_step.model_copy(
                        update={"hidden_state": latest_hidden}
                    )
                    self._completion_feedback_delivered = True
            heartbeat = self._build_heartbeat(
                decision_step=decision_step,
                decision=decision,
                observation=observation,
                info=info,
                agent_feedback=agent_feedback,
                world_snapshot=post_world_snapshot,
            )
            self.heartbeats.append(heartbeat)
            if self.heartbeat_path is not None:
                write_heartbeat(self.heartbeat_path, heartbeat)
            safety_violation = (info or {}).get("safety_violation")
            if safety_violation:
                violation = dict(safety_violation)
                trace[-1]["safety_violation"] = violation
                self.safety_violations.append(violation)
                # A fallen robot cannot execute a meaningful recovery action.
                # Human contact is different: PhysX has already resolved the
                # completed motion, so preserve the violation as sticky
                # evaluator evidence and let the Agent observe it, separate,
                # and continue the same uninterrupted rollout.
                if violation.get("type") == "robot_fall":
                    success = False
                    trace[-1]["note"] = "episode_ended: safety_violation"
                    trace[-1]["discarded_action_details"] = (
                        self._discard_agent_pending_actions("robot_fall")
                    )
                    self.termination_reason = "safety_violation"
                    break
            stall_details = _observe_progress_watchdog(
                self.progress_watchdog,
                heartbeat,
            )
            if stall_details is not None:
                trace[-1]["navigation_stuck"] = stall_details
                if self._evaluation_steps:
                    latest_step = self._evaluation_steps[-1]
                    enriched_feedback = dict(latest_step.feedback)
                    enriched_feedback["navigation_stuck"] = dict(stall_details)
                    self._evaluation_steps[-1] = latest_step.model_copy(
                        update={"feedback": enriched_feedback}
                    )
                if bool(stall_details.get("terminal")):
                    trace[-1]["note"] = "episode_ended: navigation_stuck"
                    trace[-1]["discarded_action_details"] = (
                        self._discard_agent_pending_actions("navigation_stuck")
                    )
                    self.termination_reason = "navigation_stuck"
                    break
                agent_feedback["navigation_stuck"] = dict(stall_details)
                current_feedback = str(agent_feedback.get("feedback") or "").strip()
                notice = (
                    "Navigation has made no effective progress for "
                    f"{int(stall_details['incident_decisions'])} consecutive "
                    "locomotion decisions. Change the approach."
                )
                agent_feedback["feedback"] = " ".join(
                    item for item in (current_feedback, notice) if item
                )
            self._update_agent_info(agent_feedback, observation)
            # Presentation-only archive of committed, agent-owned memory.
            if os.environ.get("TARSPHERE_PRESENTATION_CAPTURE") == "1":
                presentation_root = os.environ.get("TARSPHERE_RECORD_DIR")
                memory = getattr(self.agent, "memory", None)
                if presentation_root and memory is not None:
                    from tarsphere.infra.perception.presentation_memory import (
                        capture_agent_memory,
                    )

                    capture_agent_memory(
                        memory,
                        Path(presentation_root),
                        int(info.get("physics_steps") or 0),
                        decision_step,
                    )
            # Agent-action preconditions trigger Human events on the next
            # decision tick, matching HumanActorRuntime's one-step causal
            # boundary. Spatial predicates are re-sampled from the current
            # world and do not depend on this retained decision.
            previous_decision = dict(decision)

            if dropped_object_ids:
                trace[-1]["note"] = "episode_ended: object_dropped"
                trace[-1]["discarded_action_details"] = (
                    self._discard_agent_pending_actions("object_dropped")
                )
                self.termination_reason = "object_dropped"
                break

            if bool(decision.get("terminal_model_failure")):
                failure_reason = str(
                    decision.get("terminal_model_failure_reason")
                    or "model_request_failed"
                )
                trace[-1]["note"] = f"episode_ended: {failure_reason}"
                trace[-1]["discarded_action_details"] = (
                    self._discard_agent_pending_actions(failure_reason)
                )
                self.termination_reason = failure_reason
                break

            if success and getattr(self.episode, "evaluation", None) is not None:
                trace[-1]["note"] = "episode_ended: task_completed"
                trace[-1]["discarded_action_details"] = (
                    self._discard_agent_pending_actions("task_completed")
                )
                self.termination_reason = "task_completed"
                break

            # Early termination: robot reported done. Success was already
            # computed above via ``_evaluate_diagnostics(action)`` — do not
            # re-call with the observation dict (wrong type for last_action).
            if action == "report_done" and not bool(decision.get("is_error")):
                trace[-1]["note"] = "episode_ended: report_done"
                trace[-1]["discarded_action_details"] = (
                    self._discard_agent_pending_actions("report_done")
                )
                self.termination_reason = "report_done"
                break

            if agent_declared_plan_complete_and_idle(self.agent):
                # Freeze the complete physical trace at the Agent's own
                # terminal belief. Offline evaluation remains authoritative:
                # a hallucinated completion simply fails sooner instead of
                # spending the remaining budget on repeated wait calls.
                trace[-1]["note"] = "episode_ended: agent_declared_plan_complete"
                trace[-1]["discarded_action_details"] = (
                    self._discard_agent_pending_actions("agent_declared_plan_complete")
                )
                self.termination_reason = "agent_declared_plan_complete"
                break

            if max_physics_steps is not None and int(
                info.get("physics_steps") or 0
            ) >= int(max_physics_steps):
                trace[-1]["discarded_action_details"] = (
                    self._discard_agent_pending_actions("physics_budget_exhausted")
                )
                self.termination_reason = "physics_budget_exhausted"
                break

            if env_done:
                if getattr(self.episode, "evaluation", None) is not None:
                    raise RuntimeError(
                        "The simulator terminated or truncated a formal "
                        "episode before a harness-owned terminal condition."
                    )
                trace[-1]["discarded_action_details"] = (
                    self._discard_agent_pending_actions("environment_done")
                )
                self.termination_reason = "environment_done"
                break

            if success and getattr(self.episode, "evaluation", None) is None:
                self.termination_reason = "diagnostic_checks"
                break

            decision_step += 1

        if self.termination_reason == "running":
            if trace:
                trace[-1]["discarded_action_details"] = (
                    self._discard_agent_pending_actions("decision_budget_exhausted")
                )
            self.termination_reason = "decision_budget_exhausted"
        self.last_condition_results = dict(condition_results)
        return self.partial_result(success=success)

    def partial_result(
        self,
        *,
        success: bool = False,
        failure: Optional[BaseException] = None,
    ) -> Dict[str, Any]:
        """Materialize a result even when a fail-loud rollout was interrupted."""
        if isinstance(failure, EpisodeInitializationError):
            # Do not turn a broken scene into SR=0 or send it to a judge.
            # SceneServer returns the infrastructure error to the submitter.
            raise failure
        render_mode = getattr(self.environment, "render_mode", None)
        official_evaluation = getattr(self.episode, "evaluation", None) is not None
        causal_completion = bool(success)
        result: Dict[str, Any] = {
            "episode_id": self.episode.episode_id,
            # Official success includes semantic milestone scores and is set
            # only after the offline evaluator runs. The runner can establish
            # causal closure, but must not pre-empt that verdict.
            "success": (False if official_evaluation else causal_completion),
            "causal_graph_completed": (
                causal_completion if official_evaluation else None
            ),
            "outcome_status": (
                "pending_evaluation"
                if official_evaluation and failure is None
                else "rollout_failed"
                if failure is not None
                else "diagnostic"
            ),
            "decision_steps": self._policy_decisions,
            "action_steps": len(self.trace),
            # None means the environment publishes no counter, which must stay
            # distinguishable from an environment that genuinely ran zero steps.
            "physics_steps": _optional_counter(self.environment, "physics_steps"),
            "render_count": _optional_counter(self.environment, "render_count"),
            "render_mode": (
                render_mode.value
                if hasattr(render_mode, "value")
                else None
                if render_mode is None
                else str(render_mode)
            ),
            "termination_reason": self.termination_reason,
            "condition_results": dict(self.last_condition_results),
            "trace": list(self.trace),
            "human_replay_log": self._human_replay_log(),
            "terminal_boundary": self._terminal_boundary,
            "safety_violations": list(self.safety_violations),
            "agent_harness_config": self.agent_harness_config.as_dict(),
        }
        failed_model_calls = (
            getattr(failure, "model_calls", ()) if failure is not None else ()
        )
        if not isinstance(failed_model_calls, (list, tuple)):
            raise TypeError("failure.model_calls must be a list when present")
        result["agent_model_usage"] = aggregate_model_calls(
            [call for step in self.trace for call in (step.get("model_calls") or ())]
            + [dict(call) for call in self._uncommitted_model_calls]
            + [dict(call) for call in failed_model_calls]
        )
        human_usage = getattr(self.human, "model_usage", None)
        result["human_model_usage"] = (
            human_usage() if callable(human_usage) else aggregate_model_calls([])
        )
        stats = getattr(self.agent, "execution_stats", None)
        if isinstance(stats, dict):
            result["execution_stats"] = dict(stats)
        if self.heartbeat_path is not None:
            result["heartbeat_path"] = str(self.heartbeat_path)
        if failure is not None:
            details = getattr(failure, "details", None)
            result["failure"] = {
                "type": type(failure).__name__,
                "message": str(failure),
                "details": dict(details) if isinstance(details, dict) else {},
            }
            if isinstance(failure, EpisodeAbortError):
                result["termination_reason"] = "aborted"
            else:
                result["termination_reason"] = "failed"
            result["success"] = False
            result["outcome_status"] = "rollout_failed"
        return result

    def _raise_if_aborted(self) -> None:
        if self.abort_requested():
            raise EpisodeAbortError("Episode rollout aborted by the active request.")

    @staticmethod
    def _build_heartbeat(
        *,
        decision_step: int,
        decision: Dict[str, Any],
        observation: Dict[str, Any],
        info: Dict[str, Any],
        world_snapshot: Dict[str, Any],
        agent_feedback: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata = decision.get("controller_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        skill_action = metadata.get("skill_action")
        if not isinstance(skill_action, dict):
            skill_action = {}
        pose = observation.get("pose")
        if not isinstance(pose, dict):
            pose = {}
        position = pose.get("position")
        xy = (
            [float(position[0]), float(position[1])]
            if isinstance(position, (list, tuple)) and len(position) >= 2
            else None
        )
        humans: Dict[str, Any] = {}
        world_humans = world_snapshot.get("humans")
        if isinstance(world_humans, dict):
            for human_id, state in world_humans.items():
                if not isinstance(state, dict):
                    continue
                human_position = state.get("position")
                human_xy = (
                    [float(human_position[0]), float(human_position[1])]
                    if isinstance(human_position, (list, tuple))
                    and len(human_position) >= 2
                    else None
                )
                humans[str(human_id)] = {
                    "xy": human_xy,
                    "animation_state": state.get("animation_state"),
                    "action_status": state.get("action_status"),
                    "active_action": state.get("active_action"),
                    "target_position": state.get("target_position"),
                }
        precise = metadata.get("precise_stance_servo")
        if not isinstance(precise, dict):
            precise = {}
        heading_error = metadata.get("angular_error_rad")
        if heading_error is None:
            heading_error = metadata.get("path_entry_heading_error_rad")
        goal_distance = metadata.get("goal_distance_m")
        if goal_distance is None:
            goal_distance = precise.get("distance_m")
        proprio = observation.get("proprio")
        if not isinstance(proprio, dict):
            proprio = {}
        controllers = proprio.get("controllers")
        if not isinstance(controllers, dict):
            controllers = {}
        path_state = controllers.get("move_along_path")
        if not isinstance(path_state, dict):
            path_state = {}
        path_index = path_state.get("current_index")
        path_total = path_state.get("total_points")
        current_point = path_state.get("current_point")
        waypoint_distance = None
        if (
            xy is not None
            and isinstance(current_point, (list, tuple))
            and len(current_point) >= 2
        ):
            point_xy = [float(current_point[0]), float(current_point[1])]
            waypoint_distance = math.dist(xy, point_xy)

        controller_action = decision.get("controller_action")
        path_id = None
        if isinstance(controller_action, dict):
            path_payload = controller_action.get("move_along_path")
            try:
                points = path_payload[0]
                first_point = points[0]
                last_point = points[-1]
                path_id = (
                    f"{len(points)}:"
                    f"{float(first_point[0]):.3f},{float(first_point[1]):.3f}:"
                    f"{float(last_point[0]):.3f},{float(last_point[1]):.3f}"
                )
            except (IndexError, KeyError, TypeError, ValueError):
                path_id = None
        manipulation = world_snapshot.get("manipulation")
        objects = (
            manipulation.get("objects") if isinstance(manipulation, dict) else None
        )
        attached = bool(
            isinstance(objects, dict)
            and any(
                isinstance(state, dict) and bool(state.get("attached", False))
                for state in objects.values()
            )
        )
        physically_held = bool(
            isinstance(objects, dict)
            and any(
                isinstance(state, dict) and bool(state.get("physically_held", False))
                for state in objects.values()
            )
        )
        held = attached or physically_held
        phase = str(
            metadata.get("oracle_semantic_stop")
            or skill_action.get("action")
            or decision.get("action")
            or "unknown"
        )
        return {
            "decision_step": int(decision_step),
            "phase": phase,
            "action": str(decision.get("action") or ""),
            "policy_inference": bool(decision.get("policy_inference", True)),
            "replan_reason": decision.get("replan_reason"),
            "discarded_actions": int(decision.get("discarded_actions") or 0),
            "tool_call_index": int(
                decision.get("tool_call_index", decision.get("action_index", 0)) or 0
            ),
            "discarded_action_details": list(
                decision.get("discarded_action_details") or []
            ),
            "xy": xy,
            "yaw_rad": pose.get("yaw_rad"),
            "humans": humans,
            "goal_distance_m": goal_distance,
            "heading_error_rad": heading_error,
            "target_yaw_rad": decision.get("target_yaw_rad"),
            "rotation_objective": metadata.get("rotation_objective"),
            "controller_path_id": path_id,
            "controller_path_index": path_index,
            "controller_path_total": path_total,
            "controller_waypoint_distance_m": waypoint_distance,
            "attached": attached,
            "held": held,
            "executed": bool((info or {}).get("executed", False)),
            "physical_contact": bool((info or {}).get("physical_contact", False)),
            "physical_contact_details": (info or {}).get("physical_contact_details")
            or [],
            "execution_feedback": str((info or {}).get("feedback") or ""),
            "agent_feedback": str((agent_feedback or {}).get("feedback") or ""),
            "motion_outcome": (agent_feedback or {}).get("motion_outcome") or {},
            # Live audit of delegated navigation. Both records are public:
            # the command is the Agent-selected relative PointGoal and the
            # execution record is measured after PhysX. They never gate or
            # alter execution.
            "navigation_command": decision.get("navigation_command") or {},
            "navigation_execution": (agent_feedback or {}).get("navigation_execution")
            or {},
            "body_stability": (agent_feedback or {}).get("body_stability") or {},
            "balance_control": (agent_feedback or {}).get("balance_control") or {},
            "grasp_skill": (agent_feedback or {}).get("grasp_skill") or {},
            "grasp_reachability": (
                (agent_feedback or {}).get("grasp_reachability") or {}
            ),
            "post_motion_settle_ticks": int(
                (agent_feedback or {}).get("post_motion_settle_ticks") or 0
            ),
            "private_online_mapping": (
                (info or {}).get("private_online_mapping") or {}
            ),
            "physics_steps": (info or {}).get("physics_steps"),
        }

    def _update_agent_info(
        self,
        feedback: Dict[str, Any],
        observation: Optional[Dict[str, Any]] = None,
    ):
        update_info = getattr(self.agent, "update_info", None)
        if not callable(update_info):
            return
        payload = feedback
        pose = observation.get("pose") if isinstance(observation, dict) else None
        if isinstance(pose, dict) and "pose" not in feedback:
            payload = dict(feedback)
            payload["pose"] = pose
        update_info(payload)

    def _human_replay_log(self) -> List[Dict[str, Any]]:
        get_replay_log = getattr(self.human, "get_replay_log", None)
        events = list(get_replay_log()) if callable(get_replay_log) else []
        get_body_log = getattr(
            getattr(self, "environment", None),
            "get_human_body_replay_log",
            None,
        )
        if callable(get_body_log):
            events.extend(get_body_log())
        # Unknown clocks stay unknown and sort last; equal-tick transitions
        # preserve each producer's ordering. Consumers can distinguish native
        # body events from decision-frequency milestone commits by event type.
        events.sort(
            key=lambda item: (
                item.get("physics_step") is None,
                item.get("physics_step") or 0,
            )
        )
        return sanitize_agent_observation(events)

    def _human_ids(self) -> List[str]:
        return list(getattr(self.human, "human_ids"))

    def _visible_humans_for_agent_memory(self) -> Dict[str, tuple[float, float]]:
        """Return current RGB-D-grounded Human centres, and no hidden others.

        The environment owns named-Human visibility because it verifies the
        projected body volume against the current metric depth frame.  A
        position becomes public memory evidence only after that visibility
        predicate succeeds; invisible simulator poses never cross the Agent
        boundary.
        """

        human_ids = self._human_ids()
        if not human_ids:
            return {}
        is_human_visible = getattr(self.environment, "is_human_visible", None)
        get_human_position = getattr(self.environment, "get_human_position", None)
        if not callable(is_human_visible) or not callable(get_human_position):
            raise RuntimeError(
                "Human episodes require environment RGB-D visibility and "
                "position access for grounded Human memory."
            )
        visible: Dict[str, tuple[float, float]] = {}
        for human_id in human_ids:
            if not bool(is_human_visible(human_id)):
                continue
            position = get_human_position(human_id)
            if not isinstance(position, (list, tuple)) or len(position) < 2:
                raise ValueError(
                    f"Visible Human {human_id!r} has no valid world position."
                )
            xy = (float(position[0]), float(position[1]))
            if not all(math.isfinite(value) for value in xy):
                raise ValueError(f"Visible Human {human_id!r} has non-finite world XY.")
            visible[str(human_id)] = xy
        return visible

    def _get_human_observations(self) -> Dict[str, Dict[str, Any]]:
        human_ids = self._human_ids()
        get_observations = getattr(self.environment, "get_human_observations", None)
        observations: Dict[str, Dict[str, Any]] = {}
        if callable(get_observations):
            observations = get_observations(human_ids) or {}

        can_communicate = getattr(self.environment, "can_communicate", None)
        is_human_visible = getattr(self.environment, "is_human_visible", None)
        for human_id in human_ids:
            observation = observations.setdefault(human_id, {})
            if callable(can_communicate):
                observation["robot_can_communicate"] = bool(can_communicate(human_id))
            if callable(is_human_visible):
                # Visibility is a primitive current RGB-D fact.  Do not reuse
                # the stronger Agent-to-Human communication predicate here:
                # that predicate also requires the shared speech radius and
                # would silently over-constrain ``human_visible`` events.
                observation["human_visible_to_robot"] = bool(is_human_visible(human_id))
        return observations

    def _get_world_snapshot(
        self,
        human_observations: Optional[Dict[str, Dict[str, Any]]] = None,
        step: int = 0,
    ) -> Dict[str, Any]:
        human_ids = self._human_ids()
        get_snapshot = getattr(self.environment, "get_world_snapshot", None)
        snapshot = get_snapshot(human_ids) or {} if callable(get_snapshot) else {}
        if "step" in snapshot:
            snapshot.setdefault("physics_steps", snapshot.get("step"))
        # Human runtime triggers are evaluated at the runner decision-step
        # frequency.  Isaac snapshots may use ``step`` for physics ticks, so
        # preserve that value as ``physics_steps`` and expose the decision step
        # under the canonical key consumed by milestone conditions.
        snapshot["step"] = step
        humans = snapshot.setdefault("humans", {})
        for human_id, observation in (human_observations or {}).items():
            merged = dict(humans.get(human_id, {}))
            merged.update(observation or {})
            humans[human_id] = merged

        # Official interaction scoring consumes the same current simulator
        # predicates as the live Human runtime.  Environment world snapshots
        # own poses and body state but do not necessarily include these
        # Robot-relative facts, so record them explicitly on every trajectory
        # snapshot.  This is evidence capture only: it neither relaxes nor
        # changes the communication predicate.
        can_communicate = getattr(self.environment, "can_communicate", None)
        is_human_visible = getattr(self.environment, "is_human_visible", None)
        for human_id in human_ids:
            merged = dict(humans.get(human_id, {}))
            if callable(can_communicate):
                merged["robot_can_communicate"] = bool(can_communicate(human_id))
            if callable(is_human_visible):
                merged["human_visible_to_robot"] = bool(is_human_visible(human_id))
            humans[human_id] = merged
        return snapshot

    def _update_private_oracle_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Give live actor poses only to the benchmark-private Golden oracle.

        Dynamic Human perception must use the same simulator state later
        collected by the evaluator.  The hook is deliberately name-gated so
        an evaluated agent can never acquire private world coordinates merely
        by implementing a similarly named method.
        """
        if str(getattr(self.agent, "name", "")) != "golden_oracle":
            return
        update_snapshot = getattr(self.agent, "update_world_snapshot", None)
        if callable(update_snapshot):
            update_snapshot(snapshot)

    def _tick_human_actors_for_robot_message(
        self,
        *,
        decision: Dict[str, Any],
        spoken_message: str,
        observation: Dict[str, Any],
        step: int,
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]], str, bool]:
        """Return replies and delivery feedback on the public feedback path.

        Observation text is not part of the policy prompt. A failed exchange
        must reach the next decision through execution feedback instead.
        The last result is whether the nonempty Robot message reached its
        addressed recipient in the exact world sample used by the Human tick;
        it does not mean the Human answered or agreed.
        """
        human_observations = self._get_human_observations()
        recipient = str(decision.get("recipient") or "").strip()
        recipient_can_hear = (
            human_observations.get(recipient, {}).get("robot_can_communicate") is True
        )
        human_actions = self.human.tick(
            scene_context=observation.get("text", ""),
            robot_action=None,
            robot_message=spoken_message,
            human_observations=human_observations,
            step=step,
            world_snapshot=self._get_world_snapshot(human_observations, step),
        )
        observation, events = self._apply_human_actions(
            human_actions,
            observation,
            timing="after_agent",
        )
        question_feedback = ""
        if str(decision.get("action") or "") == "ask":
            feedback_for_question = getattr(self.human, "question_feedback", None)
            if not callable(feedback_for_question):
                raise RuntimeError(
                    "Human runtime does not implement required question feedback."
                )
            outcome = feedback_for_question(
                step=step,
                human_observations=human_observations,
                world_snapshot=self._get_world_snapshot(
                    human_observations,
                    step,
                ),
            )
            if not isinstance(outcome, QuestionOutcome):
                raise RuntimeError("ask produced no observable interaction feedback")
            question_feedback = str(outcome.feedback).strip()
            if not question_feedback:
                raise RuntimeError("ask produced empty interaction feedback")
            if outcome.reply_message:
                if not outcome.human_id:
                    raise RuntimeError(
                        "spoken question reply has no attributed Human id"
                    )
                reply_action = {
                    "command": "speak",
                    "message": outcome.reply_message,
                    "interaction_response_mode": outcome.response_mode,
                    "intent_id": outcome.intent_id,
                }
                observation, reply_message = self._apply_human_action(
                    reply_action,
                    observation,
                    outcome.human_id,
                )
                events.append(
                    self._human_event(
                        outcome.human_id,
                        reply_action,
                        reply_message,
                        timing="after_agent",
                    )
                )
                question_feedback = ""
        elif recipient_can_hear:
            disruption = getattr(self, "_disruption", None)
            unavailable = (
                disruption.unavailable_response(
                    recipient,
                    int(getattr(self.environment, "_physics_steps", 0)),
                )
                if disruption is not None
                else None
            )
            if unavailable is not None:
                reply_action = {
                    "command": "speak",
                    "message": unavailable,
                    "interaction_response_mode": "temporarily_unavailable",
                }
                observation, reply_message = self._apply_human_action(
                    reply_action,
                    observation,
                    recipient,
                )
                events.append(
                    self._human_event(
                        recipient,
                        reply_action,
                        reply_message,
                        timing="after_agent",
                    )
                )
                recipient_can_hear = False
        else:
            question_feedback = (
                f"Your message did not reach {recipient}: the recipient is "
                "not currently within visible speaking range."
            )
        return (
            observation,
            events,
            question_feedback,
            bool(spoken_message.strip() and recipient_can_hear),
        )

    def _emit_task_completion_feedback(
        self,
        observation: Dict[str, Any],
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Acknowledge the proven terminal outcome without overstating it."""

        evaluation = getattr(self.episode, "evaluation", None)
        if evaluation is None:
            raise RuntimeError(
                "Task completion feedback requires an official evaluation."
            )
        referenced = {
            str(condition.milestone_ref)
            for milestone in evaluation.milestones
            for condition in milestone.preconditions
            if isinstance(condition, MilestoneReference)
        }
        terminal_evaluators: dict[str, set[str]] = {}
        for milestone in evaluation.milestones:
            if milestone.kind != "scored" or str(milestone.milestone_id) in referenced:
                continue
            human_id = str(
                self.human.task_bindings.truth(milestone.milestone_id).get("human_id")
                or ""
            ).strip()
            if human_id:
                terminal_evaluators.setdefault(human_id, set()).add(
                    str(milestone.outcome.evaluator)
                )

        actions: dict[str, dict[str, str]] = {}
        for human_id in terminal_completion_human_ids(
            evaluation,
            self.episode.humans,
            task_bindings=self.human.task_bindings,
        ):
            evaluators = terminal_evaluators.get(human_id, set())
            if evaluators == {"handover_accepted"}:
                # Successful handover_object already emitted the recipient's
                # acceptance reply from the live handover verdict. Do not add
                # a second synthetic acknowledgement in the same step.
                continue
            elif evaluators == {"information_report_score"}:
                # The semantic judge runs after the rollout. Acknowledge that
                # the report was heard without pre-judging its correctness.
                message = "Thank you. I heard your report."
            else:
                message = "Thank you. The task is complete."
            actions[human_id] = {
                "command": "speak",
                "message": message,
            }
        return self._apply_human_actions(
            actions,
            observation,
            timing="after_agent",
        )

    def _tick_human_actors(
        self,
        step: int,
        observation: Dict[str, Any],
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        human_observations = self._get_human_observations()
        human_actions = self.human.tick(
            scene_context=observation.get("text", ""),
            robot_action=None,
            robot_message=None,
            human_observations=human_observations,
            step=step,
            world_snapshot=self._get_world_snapshot(human_observations, step),
        )
        return self._apply_human_actions(
            human_actions,
            observation,
            timing="before_agent",
        )

    def _apply_human_actions(
        self,
        human_actions: Dict[str, Dict[str, Any]],
        observation: Dict[str, Any],
        *,
        timing: str,
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        events: List[Dict[str, Any]] = []
        for human_id, human_action in human_actions.items():
            if not human_action:
                continue
            if self._is_spatial_clearance_action(human_id, human_action):
                plan_clearance = getattr(
                    self.environment,
                    "plan_spatial_blocker_clearance",
                    None,
                )
                if not callable(plan_clearance):
                    raise RuntimeError(
                        "Spatial blocking requires a live Robot-aware Human "
                        "clearance planner."
                    )
                plan = plan_clearance(human_id)
                if not isinstance(plan, dict) or not str(plan.get("status") or ""):
                    raise RuntimeError(
                        "Spatial blocker clearance planner returned an invalid result."
                    )
                if plan["status"] == "waiting_for_robot":
                    defer = getattr(
                        self.human,
                        "defer_spatial_clearance_action",
                        None,
                    )
                    if not callable(defer):
                        raise RuntimeError(
                            "Human runtime cannot defer a blocked clearance route."
                        )
                    defer(
                        human_id,
                        human_action,
                        step=self.event_bus.step,
                        reason=str(plan.get("reason") or plan["status"]),
                        physics_step=getattr(self.environment, "_physics_steps", None),
                    )
                    can_communicate = getattr(
                        self.environment,
                        "can_communicate",
                        None,
                    )
                    if not callable(can_communicate):
                        raise RuntimeError(
                            "Spatial clearance retries require the shared "
                            "communication predicate."
                        )
                    if can_communicate(human_id):
                        retry_action = {
                            "command": "speak",
                            "message": SPATIAL_CLEARANCE_RETRY_MESSAGE,
                            "interaction_response_mode": "clearance_retry",
                        }
                        observation, message = self._apply_human_action(
                            retry_action,
                            observation,
                            human_id,
                        )
                        events.append(
                            self._human_event(
                                human_id,
                                retry_action,
                                message,
                                timing=timing,
                            )
                        )
                    continue
                if plan["status"] != "ready":
                    raise RuntimeError(
                        "Spatial blocker clearance planner returned unsupported "
                        f"status {plan['status']!r}."
                    )
                bind_route = getattr(
                    self.human,
                    "bind_spatial_clearance_route",
                    None,
                )
                if not callable(bind_route):
                    raise RuntimeError(
                        "Human runtime cannot bind a live clearance route."
                    )
                bind_route(
                    human_id,
                    human_action,
                    target=plan.get("target") or (),
                    waypoints=plan.get("waypoints") or (),
                    step=self.event_bus.step,
                    robot_position=plan.get("robot_position") or (),
                    physics_step=getattr(self.environment, "_physics_steps", None),
                )
            action_name = str(human_action.get("command") or "")
            observation, message = self._apply_human_action(
                human_action, observation, human_id
            )
            if human_action.get("milestone_id"):
                self.human.acknowledge_action_applied(
                    human_id,
                    human_action,
                    step=self.event_bus.step,
                    physics_step=getattr(self.environment, "_physics_steps", None),
                )
            if action_name != "idle":
                events.append(
                    self._human_event(
                        human_id,
                        human_action,
                        message,
                        timing=timing,
                    )
                )
        return observation, events

    def _is_spatial_clearance_action(
        self,
        human_id: str,
        action: Mapping[str, Any],
    ) -> bool:
        evaluation = getattr(getattr(self, "episode", None), "evaluation", None)
        profile = getattr(evaluation, "task_profile", None)
        return bool(
            isinstance(profile, SpatialBlockingProfile)
            and str(human_id) == profile.blocking_human_id
            and str(action.get("milestone_id") or "") == profile.clear_milestone_id
            and str(action.get("command") or "") == "goto"
        )

    def _apply_human_action(
        self,
        human_action: Dict[str, Any],
        observation: Dict[str, Any],
        human_name: Optional[str] = None,
    ) -> tuple[Dict[str, Any], str]:
        if not human_action:
            return observation, ""

        human_name = human_name or getattr(self.human, "name", "human")
        apply_action = getattr(self.environment, "apply_human_action", None)
        if callable(apply_action):
            result = apply_action(human_name, human_action)
            if result is None or result.executed is not True:
                detail = (
                    "missing execution result" if result is None else result.feedback
                )
                raise RuntimeError(
                    f"Human action was not executed: {human_name} / {human_action.get('command')}: {detail}"
                )
        elif human_action.get("command") not in {"speak", "ask", "report_done"}:
            raise RuntimeError("Environment has no Human action executor")

        action_name = str(human_action.get("command") or "")
        message = str(human_action.get("message", "")).strip()
        if action_name in {"speak", "ask", "report_done"} and message:
            set_human_speech = getattr(
                self.environment,
                "set_human_speech",
                None,
            )
            if callable(set_human_speech):
                set_human_speech(message, speaker=human_name)
                delivery_tick = getattr(self.environment, "_physics_steps", None)
                observe = getattr(self.environment, "observe", None)
                if callable(observe):
                    observation = observe()
            else:
                # Minimal diagnostic/test adapters may expose no speech
                # channel. Preserve the same public evidence without making
                # the full simulator method part of their required surface.
                observation = dict(observation)
                prior_text = str(observation.get("text", "")).strip()
                utterance = f"{human_name}: {message}"
                observation["text"] = "\n".join(
                    part for part in (prior_text, utterance) if part
                )
                delivery_tick = getattr(self.environment, "_physics_steps", None)
            # Authored milestone speech already has its application receipt.
            # Interaction replies/retry speech have no milestone_id and need
            # their own delivery clock rather than a later action heartbeat.
            if not human_action.get("milestone_id"):
                record_delivery = getattr(self.human, "record_speech_delivery", None)
                if callable(record_delivery):
                    record_delivery(
                        human_name,
                        message,
                        step=self.event_bus.step,
                        physics_step=delivery_tick,
                    )
            self._pending_public_events.append(
                {
                    "event_type": "language",
                    "speaker": human_name,
                    "text": message,
                    "step": int(self.event_bus.step),
                }
            )
            self.event_bus.emit_language("speak", f"human:{human_name}", message)
            return observation, message

        return self.environment.observe(), ""

    def _drain_public_events(self) -> List[Dict[str, Any]]:
        """Hand the agent every public utterance since its last observation.

        Draining is what makes delivery exactly-once. The agent hears a line
        on the step after it was spoken and then only through
        ``dialogue_history``; leaving it in the queue would re-announce the
        same sentence as fresh on every later prompt, and dropping it when a
        second Human speaks in the same step would silently lose one of them.
        """
        events = self._pending_public_events
        self._pending_public_events = []
        return events

    @staticmethod
    def _human_event(
        human_id: str,
        human_action: Dict[str, Any],
        message: str = "",
        *,
        timing: str,
    ) -> Dict[str, Any]:
        """Serialize one Human action in its decision-step ordering phase.

        Trace steps are chronological. Within one step, ``before_agent`` events
        occur before the Agent observes and decides; ``after_agent`` events
        occur after the Robot action. Array order is authoritative within each
        phase, so no second sequence counter or flattened reply field is kept.
        """
        action_name = str(human_action.get("command") or "")
        event = {
            "human_id": human_id,
            "action": action_name,
            "timing": timing,
        }
        if message:
            event["message"] = message
        for key in (
            "target",
            "object",
            "recipient",
            "reasoning",
            "queue_name",
            "seat_prim",
            "target_anchor",
            "milestone_id",
            "interaction_response_mode",
            "intent_id",
        ):
            if key in human_action:
                event[key] = human_action[key]
        return event

    def _evaluate_diagnostics(self, last_action: str) -> tuple[bool, Dict[str, bool]]:
        checks = self.episode.diagnostics.checks if self.episode.diagnostics else []
        results: Dict[str, bool] = {}
        for index, check in enumerate(checks):
            key = f"{index}:{check.type}"
            results[key] = self._evaluate_diagnostic_check(
                check.type,
                check.params,
                last_action,
            )
        return bool(checks) and all(results.values()), results

    def _sample_evaluation_conditions(
        self,
        decision: Dict[str, Any],
        observation: Dict[str, Any] | None = None,
    ) -> Dict[str, bool]:
        """Sample every hidden TARSphere condition after one Agent action."""

        spec = getattr(self.episode, "evaluation", None)
        if spec is None:
            raise RuntimeError("episode has no TARSphere evaluation contract")
        return {
            condition.condition_id: self._evaluate_hidden_condition(
                condition.detector,
                condition.arguments,
                decision,
                observation=observation,
            )
            for condition in spec.conditions()
        }

    def _sample_evaluation_reactions(
        self,
        decision: Dict[str, Any],
        observation: Dict[str, Any] | None = None,
        *,
        phase: ReactionSamplingPhase,
        evidence_sink: dict[str, dict[str, Any]],
        preceding_reactions: Mapping[str, bool] | None = None,
        human_events: Sequence[Mapping[str, Any]] = (),
        decision_step: int | None = None,
        event_conditions: Mapping[str, bool] | None = None,
    ) -> Dict[str, bool]:
        """Sample primitive reactions only in their authoritative phase."""

        spec = getattr(self.episode, "evaluation", None)
        if spec is None:
            raise RuntimeError("episode has no TARSphere evaluation contract")
        results: Dict[str, bool] = {}
        completed = completed_milestone_ids(spec, self._evaluation_steps)
        if phase != "pre_decision" and decision_step is not None:
            if event_conditions is None:
                raise ValueError(
                    "Current-decision causal evidence needs its event-time conditions"
                )
            # The first plan can accompany an ask in the same decision. Reuse
            # the DAG's plan-submission semantics without grading it here or
            # pretending that a later physical action has already happened.
            state = _evaluation_agent_state(decision)
            current = SimpleNamespace(
                decision_step=decision_step,
                agent_state={"plan": state["plan"]}
                if isinstance(state.get("plan"), dict)
                else {},
                occurred_events=self._milestone_events_at_step(decision_step),
                condition_values=dict(event_conditions),
                reaction_values={},
                action={},
                hidden_state={},
            )
            current_completed = completed_milestone_ids(
                spec, [*self._evaluation_steps, current]
            )
            completed = completed.union(
                node.milestone_id
                for node in spec.milestones
                if node.outcome.evaluator == "structured_plan_submitted"
                and node.milestone_id in current_completed
            )
        # Earlier phases in this very action are valid causal evidence: seeing
        # the source and asking it must not require a redundant second ask.
        completed = completed.union(
            key for key, value in (preceding_reactions or {}).items() if value
        )
        for milestone in spec.milestones:
            evaluator = str(milestone.outcome.evaluator)
            if evaluator in DECLARATIVE_MILESTONE_EVALUATORS:
                continue
            expected_phase = PRIMITIVE_REACTION_SAMPLING_PHASE.get(evaluator)
            if expected_phase is None:
                raise ValueError(
                    f"primitive evaluator {evaluator!r} has no sampling phase"
                )
            if expected_phase != phase:
                continue
            milestone_id = str(milestone.milestone_id)
            parent_ids = {
                str(precondition.milestone_ref)
                for precondition in milestone.preconditions
                if getattr(precondition, "kind", None) == "milestone"
            }
            # A reaction is meaningful only after all causal parents have
            # completed. Besides matching the DAG semantics, this prevents an
            # evaluator-private semantic detector from running on every frame
            # before its observation phase is even open.
            if milestone_id in completed or not parent_ids.issubset(completed):
                results[milestone_id] = False
                continue
            results[milestone_id] = self._evaluate_hidden_condition(
                evaluator,
                (
                    self.human.task_bindings.truth(milestone_id)
                    if any(
                        update.outcome_updates
                        for node in spec.milestones
                        for update in node.event.task_updates
                    )
                    else milestone.outcome.ground_truth
                ),
                decision,
                evidence_key=milestone_id,
                evidence_sink=evidence_sink,
                observation=observation,
                human_events=human_events,
            )
            if evaluator == "human_event_completed" and results[milestone_id]:
                completed = completed.union({milestone_id})
        return results

    def _commit_model_visible_pre_decision_reactions(
        self,
        *,
        decision: Dict[str, Any],
        decision_step: int,
        evidence_sink: dict[str, dict[str, Any]],
    ) -> Dict[str, bool]:
        """Commit only visual evidence contained in the current model input.

        Pre-decision RGB-D is sampled before ``agent.act`` so it remains
        temporally exact. The Agent subsequently reports the action-step
        camera records that its kernel actually assembled into the model
        request. Queued actions have no model call and therefore cannot score
        an otherwise invisible intermediate frame. Object evidence may be
        aggregated across the explicitly supplied visual-history frames;
        current Human visibility is deliberately sampled from the current
        frame only.
        """

        current_sample = self._pre_decision_reaction_samples.get(int(decision_step))
        if current_sample is None:
            raise RuntimeError(
                "pre-decision reaction sample is missing for current step "
                f"{decision_step}"
            )
        current_results, current_evidence = current_sample
        event_completions = {
            str(node.milestone_id): bool(current_results[str(node.milestone_id)])
            for node in self.episode.evaluation.milestones
            if node.outcome.evaluator == "human_event_completed"
        }
        for milestone_id in event_completions:
            if milestone_id in current_evidence:
                evidence_sink[milestone_id] = dict(current_evidence[milestone_id])
        if not bool(decision.get("policy_inference", True)):
            return {
                milestone_id: event_completions.get(milestone_id, False)
                for milestone_id in current_results
            }

        raw_steps = decision.get("model_visible_action_steps")
        if not isinstance(raw_steps, (list, tuple)):
            raise TypeError("policy inference must publish model_visible_action_steps")
        visible_steps: list[int] = []
        for value in raw_steps:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    "model_visible_action_steps must contain integer action steps"
                )
            step = int(value)
            if step > int(decision_step):
                raise ValueError(
                    "model_visible_action_steps cannot reference a future "
                    f"step: {step} > {decision_step}"
                )
            if step not in visible_steps:
                visible_steps.append(step)
        if int(decision_step) not in visible_steps:
            raise ValueError(
                "the current pre-decision camera frame must be present in "
                "every policy model input"
            )

        missing = [
            step
            for step in visible_steps
            if step not in self._pre_decision_reaction_samples
        ]
        if missing:
            raise RuntimeError(
                "model input references visual action steps with no exact "
                f"evaluator sample: {missing}"
            )

        expected_ids = set(current_results)
        milestone_by_id = {
            str(milestone.milestone_id): milestone
            for milestone in self.episode.evaluation.milestones
        }
        completed = completed_milestone_ids(
            self.episode.evaluation,
            self._evaluation_steps,
        )
        completed = completed.union(
            key for key, value in event_completions.items() if value
        )
        committed: Dict[str, bool] = {}
        for milestone_id in sorted(expected_ids):
            if milestone_id in event_completions:
                committed[milestone_id] = event_completions[milestone_id]
                continue
            milestone = milestone_by_id.get(milestone_id)
            if milestone is None:
                raise ValueError(
                    "pre-decision reaction sample references an unknown "
                    f"milestone: {milestone_id!r}"
                )
            parent_ids = {
                str(precondition.milestone_ref)
                for precondition in milestone.preconditions
                if getattr(precondition, "kind", None) == "milestone"
            }
            if milestone_id in completed or not parent_ids.issubset(completed):
                committed[milestone_id] = False
                evidence_sink[milestone_id] = {
                    "matched": False,
                    "evidence_source": "causal_gate_closed",
                    "causal_gate_reason": (
                        "already_completed"
                        if milestone_id in completed
                        else "waiting_for_parents"
                    ),
                }
                continue
            evaluator = str(milestone.outcome.evaluator)
            if evaluator != "visual_targets_observed":
                matched = bool(current_results[milestone_id])
                _, current_evidence = current_sample
                evidence = current_evidence.get(milestone_id)
                committed[milestone_id] = matched
                evidence_sink[milestone_id] = {
                    "matched": matched,
                    "evidence_source": "current_model_visible_rgbd",
                    "model_visible_action_step": int(decision_step),
                    **(dict(evidence) if isinstance(evidence, dict) else {}),
                }
                continue
            per_frame: list[dict[str, Any]] = []
            matched_steps: list[int] = []
            contributing_steps: list[int] = []
            required_object_ids: tuple[str, ...] | None = None
            visible_object_ids: set[str] = set()
            for step in visible_steps:
                results, evidence = self._pre_decision_reaction_samples[step]
                if set(results) != expected_ids:
                    raise ValueError(
                        "pre-decision reaction schema changed within one "
                        "episode; visual evidence cannot be aligned safely"
                    )
                matched = bool(results[milestone_id])
                if matched:
                    matched_steps.append(step)
                item: dict[str, Any] = {
                    "action_step": int(step),
                    "matched": matched,
                }
                frame_evidence = evidence.get(milestone_id)
                if isinstance(frame_evidence, dict):
                    item["evidence"] = dict(frame_evidence)
                    frame_required_objects = frame_evidence.get("required_object_ids")
                    if isinstance(frame_required_objects, (list, tuple)):
                        normalized = tuple(
                            str(value) for value in frame_required_objects
                        )
                        if (
                            required_object_ids is not None
                            and normalized != required_object_ids
                        ):
                            raise ValueError(
                                "visual object requirements changed across "
                                "model-visible frames"
                            )
                        required_object_ids = normalized
                        frame_visible = {
                            str(value)
                            for value in (
                                frame_evidence.get("visible_object_ids") or ()
                            )
                        }
                        if frame_visible:
                            contributing_steps.append(step)
                        visible_object_ids.update(frame_visible)

                per_frame.append(item)
            if required_object_ids is None:
                raise RuntimeError(
                    "visual_targets_observed evidence must bind exact "
                    "episode object_ids"
                )
            aggregate_matched = set(required_object_ids).issubset(visible_object_ids)
            aggregate_evidence = {
                "required_object_ids": list(required_object_ids),
                "visible_object_ids": sorted(visible_object_ids),
            }
            committed[milestone_id] = aggregate_matched
            evidence_sink[milestone_id] = {
                "matched": aggregate_matched,
                "evidence_source": "model_visible_rgbd",
                "model_visible_action_steps": list(visible_steps),
                "matched_action_steps": matched_steps,
                "contributing_action_steps": sorted(set(contributing_steps)),
                **aggregate_evidence,
                "per_frame": per_frame,
            }
        return committed

    def _milestone_events_at_step(self, step: int) -> tuple[str, ...]:
        """Return evaluator-private milestone occurrences for one tick."""

        return tuple(
            str(item["milestone_id"])
            for item in self._human_replay_log()
            if isinstance(item, dict)
            and item.get("event") == "milestone_event_occurred"
            and item.get("step") is not None
            and int(item["step"]) == int(step)
            and str(item.get("milestone_id") or "").strip()
        )

    def _milestone_admissions_at_step(self, step: int) -> tuple[str, ...]:
        """Return one-shot Human event admissions for this evaluator row."""

        return tuple(
            str(item["milestone_id"])
            for item in self._human_replay_log()
            if isinstance(item, dict)
            and item.get("event") == "milestone_triggered"
            and item.get("step") is not None
            and int(item["step"]) == int(step)
            and str(item.get("milestone_id") or "").strip()
        )

    def evaluation_trajectory(self) -> EpisodeTrajectory:
        """Return the evaluator-private uninterrupted rollout artifact."""

        spec = self.episode.evaluation
        if spec is None:
            raise ValueError(
                "official TARSphere evaluation requires episode.evaluation"
            )
        if not self._evaluation_steps:
            raise ValueError("the rollout contains no evaluation trajectory steps")
        return EpisodeTrajectory(
            episode_id=self.episode.episode_id,
            task_profile=spec.task_profile,
            steps=tuple(self._evaluation_steps),
            termination_reason=self.termination_reason,
            controller_diagnostics=(
                ("certification_policy",)
                if getattr(self.agent, "certification_only", False)
                else ()
            ),
        )

    def _event_decision_requirements(
        self,
        *,
        decision_step: int,
    ) -> DecisionRequirements | None:
        """Require evaluator inputs when their public event first occurs.

        The mapping contains output shapes only. Condition ground truth and
        evaluator bindings remain private.
        """

        spec = getattr(self.episode, "evaluation", None)
        if spec is None:
            return None
        completed = completed_milestone_ids(spec, self._evaluation_steps)
        if len(completed) >= len(spec.milestones):
            return None
        occurred = set(self._milestone_events_at_step(decision_step))
        milestones = tuple(
            milestone
            for milestone in spec.milestones
            if milestone.milestone_id in occurred
        )
        if not milestones:
            return None
        evaluators = {milestone.outcome.evaluator for milestone in milestones}
        if not evaluators & {
            "structured_plan_submitted",
        }:
            return None
        return DecisionRequirements(
            planning=({} if "structured_plan_submitted" in evaluators else None),
            # Bind the required response to the delivered event.
            event_milestone_ids=tuple(
                milestone.milestone_id for milestone in milestones
            ),
        )

    def _discard_agent_pending_actions(self, reason: str) -> list[dict[str, Any]]:
        """Drop the queued suffix at episode end and surface it in the trace."""
        discard = getattr(self.agent, "discard_pending_actions", None)
        if not callable(discard):
            return []
        return list(discard(reason))

    def _evaluate_hidden_condition(
        self,
        detector: str,
        arguments: Dict[str, Any],
        decision: Dict[str, Any],
        *,
        evidence_key: str | None = None,
        evidence_sink: dict[str, dict[str, Any]] | None = None,
        observation: Dict[str, Any] | None = None,
        human_events: Sequence[Mapping[str, Any]] = (),
    ) -> bool:
        """Sample one closed-set primitive fact at the current decision step."""

        action = str(decision.get("action") or "")
        if detector == "human_event_completed":
            result = self.human.event_actions_completed(str(arguments["milestone_id"]))
            if evidence_sink is not None and evidence_key is not None:
                evidence_sink[evidence_key] = {
                    "completed": result,
                    "actions": [
                        row
                        for row in self.human.get_replay_log()
                        if row.get("milestone_id") == arguments["milestone_id"]
                    ],
                }
        elif detector == "agent_action_selected":
            result = action == str(arguments["action"])
        elif detector == "information_received":
            replies = [
                dict(event)
                for event in human_events
                if event.get("action") == "speak"
                and event.get("timing") == "after_agent"
                and event.get("human_id") == arguments["human_id"]
                and event.get("intent_id") == arguments["intent_id"]
                and event.get("interaction_response_mode") == "accept"
                and str(event.get("message") or "").strip()
            ]
            result = bool(
                action == "ask"
                and decision.get("recipient") == arguments["human_id"]
                and (
                    not decision.get("intent_id")
                    or decision["intent_id"] == arguments["intent_id"]
                )
                and replies
            )
            if evidence_sink is not None and evidence_key is not None:
                evidence_sink[evidence_key] = {"delivered_replies": replies}
        elif detector == "agent_interaction_selected":
            expected_human = str(arguments["human_id"])
            accepted_recipients = {expected_human}
            for human in self.episode.humans:
                if str(human.id) == expected_human:
                    accepted_recipients.add(str(human.id))
            result = (
                self._agent_can_address_human(expected_human)
                and action == "ask"
                and str(decision.get("recipient") or "").strip() in accepted_recipients
                and str(decision.get("intent_id") or "").strip()
                == str(arguments["intent_id"])
            )
        elif detector == "robot_in_place":
            current_position = (
                self._robot_position_history[-1]
                if self._robot_position_history
                else None
            )
            if current_position is None:
                result = False
            else:
                target = arguments["position"]
                target_xy = (float(target[0]), float(target[1]))
                result = math.dist(current_position, target_xy) <= float(
                    arguments["distance"]
                )
        elif detector == "human_visible":
            evaluate = getattr(self.environment, "is_human_visible", None)
            if not callable(evaluate):
                raise RuntimeError(
                    "human_visible requires environment.is_human_visible"
                )
            result = bool(evaluate(str(arguments["human_id"])))
        elif detector == "visual_targets_observed":
            object_ids = tuple(str(value) for value in arguments.get("object_ids", ()))
            evaluate = getattr(
                self.environment,
                "evaluate_visual_object_visibility",
                None,
            )
            if not callable(evaluate):
                raise RuntimeError(
                    "visual_targets_observed requires "
                    "environment.evaluate_visual_object_visibility"
                )
            evidence = evaluate(object_ids=object_ids)
            if not isinstance(evidence, dict) or "matched" not in evidence:
                raise TypeError(
                    "visual target visibility evaluation must return evidence "
                    "containing matched"
                )
            result = bool(evidence["matched"])
            if evidence_key is not None:
                if evidence_sink is None:
                    raise ValueError(
                        "visual_targets_observed requires an evidence sink"
                    )
                evidence_sink[evidence_key] = dict(evidence)
        elif detector == "object_dropped":
            object_id = str(arguments["object_id"])
            result = object_id in {
                str(item) for item in self.last_info.get("dropped_objects", ())
            }
        elif detector in {
            "robot_near_human",
            "robot_holding_object",
            "object_supported_by",
            "handover_accepted",
        }:
            result = self.environment.evaluate_success(
                SimpleNamespace(
                    type=detector,
                    params=dict(arguments),
                )
            )
        else:
            raise ValueError(f"unknown TARSphere detector {detector!r}")
        return bool(result)

    def _agent_can_address_human(self, human_id: str) -> bool:
        """Require current named-Human RGB-D plus the shared speech radius."""

        evaluate = getattr(self.environment, "can_communicate", None)
        if not callable(evaluate):
            raise RuntimeError(
                "Agent-to-Human interaction requires environment.can_communicate"
            )
        return bool(evaluate(str(human_id)))

    def _evaluate_diagnostic_check(
        self,
        evaluator: str,
        arguments: Dict[str, Any],
        last_action: str,
    ) -> bool:
        """Evaluate one non-TARSphere component/controller diagnostic check."""

        if evaluator == "agent_reported_done":
            result = last_action == "report_done"
        elif evaluator == "agent_action_occurred":
            expected = str(arguments.get("action") or "").strip()
            if not expected:
                raise ValueError("agent_action_occurred requires arguments.action")
            result = last_action == expected or any(
                item.get("robot_action") == expected for item in self.trace
            )
        elif evaluator == "object_dropped":
            object_id = str(arguments.get("object_id") or "").strip()
            if not object_id:
                raise ValueError("object_dropped requires arguments.object_id")
            result = object_id in {
                str(item) for item in self.last_info.get("dropped_objects", ())
            } or any(
                object_id in item.get("dropped_objects", ()) for item in self.trace
            )
        elif evaluator == "handover_accepted":
            human_id = str(arguments.get("human_id") or "").strip()
            object_id = str(arguments.get("object_id") or "").strip()
            if not human_id:
                raise ValueError("handover_accepted requires params.human_id")
            if not object_id:
                raise ValueError("handover_accepted requires params.object_id")
            was_accepted = getattr(
                self.environment,
                "handover_was_accepted",
                None,
            )
            if not callable(was_accepted):
                raise RuntimeError(
                    "handover_accepted requires environment.handover_was_accepted"
                )
            result = bool(was_accepted(human_id=human_id, object_id=object_id))
        else:
            result = self.environment.evaluate_success(
                SimpleNamespace(
                    type=evaluator,
                    params=dict(arguments),
                )
            )
        return bool(result)

    def _record_robot_position(self, observation: Dict[str, Any]) -> None:
        """Remember one public robot pose for trajectory diagnostic checks."""

        pose = observation.get("pose")
        if not isinstance(pose, dict):
            return
        position = pose.get("position")
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            return
        point = (float(position[0]), float(position[1]))
        if not all(math.isfinite(value) for value in point):
            return
        self._robot_position_history.append(point)
