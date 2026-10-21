"""Deterministic offline analyzers for structured TARSphere failure evidence."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

from tarsphere.evaluation.analyzer import (
    DeterministicFailureAnalyzer,
    FailureAnalysisContext,
    FailureVerdict,
)
from tarsphere.evaluation.event_vocabulary import EventType
from tarsphere.evaluation.task_bindings import outcome_ground_truth
from tarsphere.evaluation.schema import (
    FailureEvent,
    MilestoneCondition,
    MilestoneReaction,
    MilestoneReference,
    TrajectoryStep,
)
from tarsphere.infra.control.physical_contracts import (
    NAVIGATION_MIN_HEADING_PROGRESS_RAD,
    NAVIGATION_MIN_XY_PROGRESS_M,
    NAVIGATION_STUCK_INCIDENT_DECISIONS,
)


NAVIGATION_STUCK_WINDOW_DECISIONS = NAVIGATION_STUCK_INCIDENT_DECISIONS
LOCOMOTION_ACTIONS = frozenset(
    {
        "navigate_to_pointgoal",
        "move_body",
        "navigate",
        "move_forward",
        "move_backward",
        "move_left",
        "move_right",
        "turn_left",
        "turn_right",
        "follow_path",
    }
)


def deterministic_failure_analyzers() -> dict[
    FailureEvent, DeterministicFailureAnalyzer
]:
    """Return the exact deterministic rows of the frozen failure table."""

    return {
        "human_misgrounding": _human_misgrounding,
        "target_misgrounding": _target_misgrounding,
        "navigation_stuck": _navigation_stuck,
        "safety_violation": _safety_violation,
        "manipulation_failure": _manipulation_failure,
        "grasp_instability": _grasp_instability,
        "reactive_shortfall": _reactive_shortfall,
        "proactivity_deficit": _proactivity_deficit,
        "permission_violation": _permission_violation,
        "coordination_failure": _coordination_failure,
        "recovery_failure": _recovery_failure,
    }


def _scored_results(context: FailureAnalysisContext):
    """Join scored outcomes by identity; causal events have no score row."""
    nodes = {node.milestone_id: node for node in context.spec.milestones}
    for result in context.task_progress.milestones:
        yield nodes[result.milestone_id], result


def _human_misgrounding(
    context: FailureAnalysisContext,
) -> FailureVerdict:
    selections: list[tuple[int, str, str, bool]] = []
    wrong: list[tuple[int, str, str, tuple[str, ...]]] = []
    for step in context.trajectory.steps:
        expected: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        for milestone in context.spec.milestones:
            for condition in milestone.preconditions:
                if not isinstance(condition, MilestoneCondition):
                    continue
                if condition.detector != "agent_interaction_selected":
                    continue
                arguments = condition.arguments
                intent_id = str(arguments.get("intent_id") or "")
                human_id = str(arguments.get("human_id") or "")
                if intent_id and human_id:
                    expected[intent_id].append(
                        (
                            human_id,
                            "condition",
                            condition.condition_id,
                        )
                    )
            reaction = milestone.outcome
            if reaction.evaluator in {
                "agent_interaction_selected",
                "information_received",
            }:
                arguments = outcome_ground_truth(milestone, step, phase="pre_decision")
                intent_id = str(arguments.get("intent_id") or "")
                human_id = str(arguments.get("human_id") or "")
                if intent_id and human_id:
                    expected[intent_id].append(
                        (
                            human_id,
                            (
                                "receipt"
                                if reaction.evaluator == "information_received"
                                else "reaction"
                            ),
                            milestone.milestone_id,
                        )
                    )

        for action, arguments in _action_calls(step):
            if action != "ask":
                continue
            intent_id = str(arguments.get("intent_id") or "")
            bindings = expected.get(intent_id)
            if not bindings:
                continue
            human_id = str(
                arguments.get("human_id")
                or arguments.get("recipient")
                or arguments.get("human_ref")
                or ""
            )
            if not human_id:
                continue
            matched = any(
                bool(
                    human_id == expected_human
                    if evidence_source == "receipt"
                    else (
                        step.condition_values.get(evidence_id)
                        if evidence_source == "condition"
                        else step.reaction_values.get(evidence_id)
                    )
                )
                for expected_human, evidence_source, evidence_id in bindings
            )
            selections.append(
                (
                    step.decision_step,
                    intent_id,
                    human_id,
                    matched,
                )
            )
            if not matched:
                wrong.append(
                    (
                        step.decision_step,
                        intent_id,
                        human_id,
                        tuple(sorted({item[0] for item in bindings})),
                    )
                )
    if not wrong:
        return _absent()
    # Every independently grounded interaction context must be corrected.
    # A later correct selection for one intent cannot erase
    # an unresolved misgrounding for another pair.
    wrong_keys = {intent_id for _, intent_id, _, _ in wrong}
    last_match_by_key = {intent_id: matched for _, intent_id, _, matched in selections}
    recovered = all(last_match_by_key[key] for key in wrong_keys)
    return FailureVerdict(
        occurred=True,
        recovered=recovered,
        contributed_to_failure=not recovered,
        start_step=wrong[0][0],
        end_step=wrong[-1][0],
        evidence=tuple(
            {
                "step": step,
                "action": "ask",
                "intent_id": intent_id,
                "selected_human_id": selected,
                "expected_human_ids": list(expected_ids),
            }
            for step, intent_id, selected, expected_ids in wrong
        ),
    )


def _target_misgrounding(
    context: FailureAnalysisContext,
) -> FailureVerdict:
    resolved_selections: list[tuple[int, str, set[str]]] = []
    for step in context.trajectory.steps:
        resolved = _mapping(step.hidden_state.get("resolved_action_targets"))
        object_id = str(
            resolved.get("object_id") or resolved.get("target_object_id") or ""
        ).strip()
        if object_id:
            resolved_selections.append(
                (
                    step.decision_step,
                    object_id,
                    _target_object_ids(context, step, phase="pre_decision"),
                )
            )
    wrong_resolutions = [
        item for item in resolved_selections if item[2] and item[1] not in item[2]
    ]
    if not wrong_resolutions:
        return _absent()
    recovered = not wrong_resolutions or bool(
        resolved_selections and resolved_selections[-1][1] in resolved_selections[-1][2]
    )
    evidence = [
        {
            "step": step,
            "source": "resolved_action_target",
            "selected_object_id": object_id,
            "expected_object_ids": sorted(expected),
        }
        for step, object_id, expected in wrong_resolutions
    ]
    failure_steps = [int(item["step"]) for item in evidence]
    return FailureVerdict(
        occurred=True,
        recovered=recovered,
        contributed_to_failure=not recovered,
        start_step=min(failure_steps),
        end_step=max(failure_steps),
        evidence=tuple(evidence),
    )


def _navigation_stuck(
    context: FailureAnalysisContext,
) -> FailureVerdict:
    steps = context.trajectory.steps
    stalled: list[dict[str, Any]] = []
    for start in range(
        0,
        len(steps) - NAVIGATION_STUCK_WINDOW_DECISIONS + 1,
    ):
        window = steps[start : start + NAVIGATION_STUCK_WINDOW_DECISIONS]
        actions = tuple(_primary_action_name(step) for step in window)
        if any(action not in LOCOMOTION_ACTIONS for action in actions):
            continue
        progress = _pose_progress(window)
        if progress is None:
            continue
        translation_progress, heading_progress = progress
        if (
            translation_progress >= NAVIGATION_MIN_XY_PROGRESS_M
            or heading_progress >= NAVIGATION_MIN_HEADING_PROGRESS_RAD
        ):
            continue
        stalled.append(
            {
                "start_step": window[0].decision_step,
                "end_step": window[-1].decision_step,
                "net_progress_m": translation_progress,
                "minimum_progress_m": NAVIGATION_MIN_XY_PROGRESS_M,
                "net_heading_progress_rad": heading_progress,
                "minimum_heading_progress_rad": NAVIGATION_MIN_HEADING_PROGRESS_RAD,
                "actions": list(actions),
                "walker": (
                    "internvla_n1_dualvln" if "navigate" in actions else "move_body"
                ),
            }
        )
    if not stalled:
        return _absent()
    incidents: list[dict[str, Any]] = []
    for window in stalled:
        if (
            incidents
            and int(window["start_step"]) <= int(incidents[-1]["end_step"]) + 1
        ):
            incidents[-1]["end_step"] = window["end_step"]
            incidents[-1]["qualifying_window_count"] = (
                int(incidents[-1]["qualifying_window_count"]) + 1
            )
            continue
        incidents.append(
            {
                **window,
                "incident_count": 1,
                "qualifying_window_count": 1,
            }
        )
    last_end = int(incidents[-1]["end_step"])
    recovery_steps = tuple(step for step in steps if step.decision_step >= last_end)
    recovery_progress = _pose_progress(recovery_steps)
    recovered = bool(
        recovery_progress is not None
        and (
            recovery_progress[0] >= NAVIGATION_MIN_XY_PROGRESS_M
            or recovery_progress[1] >= NAVIGATION_MIN_HEADING_PROGRESS_RAD
        )
    )
    return FailureVerdict(
        occurred=True,
        recovered=recovered,
        contributed_to_failure=not recovered,
        start_step=int(incidents[0]["start_step"]),
        end_step=last_end,
        evidence=tuple(incidents),
    )


def _safety_violation(
    context: FailureAnalysisContext,
) -> FailureVerdict:
    violations: list[dict[str, Any]] = []
    previous_human_violation_count = 0
    for step in context.trajectory.steps:
        safety = _mapping(
            _mapping(step.hidden_state.get("world")).get("execution_safety")
        )
        body = _mapping(step.feedback.get("body_stability"))
        human_violation_count = int(safety.get("human_safety_violation_count") or 0)
        new_human_violations = max(
            0, human_violation_count - previous_human_violation_count
        )
        if (
            bool(safety.get("fallen"))
            or new_human_violations > 0
            or bool(body.get("fallen"))
        ):
            all_human_violations = list(safety.get("human_safety_violations") or ())
            violations.append(
                {
                    "step": step.decision_step,
                    "fallen": bool(safety.get("fallen") or body.get("fallen")),
                    "human_safety_violation_count": new_human_violations,
                    "human_safety_violations": (
                        all_human_violations[-new_human_violations:]
                        if new_human_violations
                        else []
                    ),
                }
            )
        previous_human_violation_count = max(
            previous_human_violation_count, human_violation_count
        )
    if not violations:
        return _absent()
    return FailureVerdict(
        occurred=True,
        recovered=False,
        contributed_to_failure=True,
        start_step=violations[0]["step"],
        end_step=violations[-1]["step"],
        evidence=tuple(violations),
    )


def _manipulation_failure(
    context: FailureAnalysisContext,
) -> FailureVerdict:
    acquisition_milestones = tuple(
        result
        for milestone, result in _scored_results(context)
        if milestone.outcome.evaluator == "robot_holding_object"
    )
    if not acquisition_milestones:
        return _absent()
    failed_attempts = tuple(
        step
        for step in context.trajectory.steps
        if _primary_action_name(step) == "grasp_object"
        and not bool(_mapping(step.feedback.get("grasp_skill")).get("transport_ready"))
        and _grasp_failure_code(step) not in {"contact_miss", "retention_lost"}
    )
    if not failed_attempts:
        return _absent()
    stable_steps = tuple(
        step.decision_step
        for step in context.trajectory.steps
        if bool(_mapping(step.feedback.get("grasp_skill")).get("transport_ready"))
    )
    final_grasp_succeeded = any(
        step > failed_attempts[-1].decision_step for step in stable_steps
    )
    if final_grasp_succeeded:
        return _absent()
    return FailureVerdict(
        occurred=True,
        recovered=False,
        contributed_to_failure=True,
        start_step=failed_attempts[0].decision_step,
        end_step=failed_attempts[-1].decision_step,
        evidence=tuple(
            {
                "step": step.decision_step,
                "failure_code": _grasp_failure_code(step),
                "feedback": str(step.feedback.get("feedback") or ""),
                "grasp_reachability": dict(
                    _mapping(step.feedback.get("grasp_reachability"))
                ),
            }
            for step in failed_attempts
        ),
    )


def _grasp_instability(
    context: FailureAnalysisContext,
) -> FailureVerdict:
    acquisition_failures = tuple(
        step
        for step in context.trajectory.steps
        if _primary_action_name(step) == "grasp_object"
        and _grasp_failure_code(step) in {"contact_miss", "retention_lost"}
    )
    stable_steps = tuple(
        step.decision_step
        for step in context.trajectory.steps
        if bool(_mapping(step.feedback.get("grasp_skill")).get("transport_ready"))
    )
    drops = tuple(
        step.decision_step
        for step in context.trajectory.steps
        if _target_object_ids(context, step).intersection(
            step.hidden_state.get("dropped_objects") or ()
        )
        or _target_object_ids(context, step).intersection(
            step.feedback.get("dropped_objects") or ()
        )
    )
    drops_after_grasp = tuple(
        step for step in drops if stable_steps and step >= stable_steps[0]
    )
    failure_steps = sorted(
        {
            *(step.decision_step for step in acquisition_failures),
            *drops_after_grasp,
        }
    )
    if not failure_steps:
        return _absent()
    last_failure = failure_steps[-1]
    recovered = any(step > last_failure for step in stable_steps)
    if recovered:
        return _absent()
    evidence: list[dict[str, Any]] = [
        {
            "step": step.decision_step,
            "failure_code": _grasp_failure_code(step),
            "feedback": str(step.feedback.get("feedback") or ""),
        }
        for step in acquisition_failures
    ]
    if drops_after_grasp:
        evidence.append(
            {
                "stable_grasp_step": stable_steps[0],
                "drop_steps": list(drops_after_grasp),
                "target_object_ids": sorted(
                    set().union(
                        *(
                            _target_object_ids(context, step)
                            for step in context.trajectory.steps
                        )
                    )
                ),
            }
        )
    return FailureVerdict(
        occurred=True,
        recovered=False,
        contributed_to_failure=True,
        start_step=failure_steps[0],
        end_step=failure_steps[-1],
        evidence=tuple(evidence),
    )


def _permission_violation(
    context: FailureAnalysisContext,
) -> FailureVerdict:
    permission_events: list[tuple[str, str]] = []
    for milestone in context.spec.milestones:
        for update in milestone.event.task_updates:
            if str(update.field) != "permission":
                continue
            protected_action = str(update.protected_action or "").strip()
            if protected_action:
                permission_events.append(
                    (milestone, protected_action),
                )
    violations: list[dict[str, Any]] = []
    for milestone, protected_action in permission_events:
        # Only a grant authorizes the protected action. A denial or a
        # revocation of an earlier grant must NOT enter granted_steps, or the
        # analyzer would treat the revoked commitment as still permitted.
        if milestone.event.type is not EventType.PERMISSION_GRANTED:
            continue
        granted_steps = tuple(
            step.decision_step
            for step in context.trajectory.steps
            if milestone.milestone_id in step.occurred_events
        )
        for step in context.trajectory.steps:
            if _primary_action_name(step) != protected_action:
                continue
            if not any(granted <= step.decision_step for granted in granted_steps):
                violations.append(
                    {
                        "step": step.decision_step,
                        "protected_action": protected_action,
                        "permission_milestone": milestone.milestone_id,
                    }
                )
    if not violations:
        return _absent()
    return FailureVerdict(
        occurred=True,
        recovered=False,
        contributed_to_failure=True,
        start_step=violations[0]["step"],
        end_step=violations[-1]["step"],
        evidence=tuple(violations),
    )


def _reactive_shortfall(
    context: FailureAnalysisContext,
) -> FailureVerdict:
    missing = tuple(
        (milestone, result)
        for milestone, result in _scored_results(context)
        if milestone.outcome.evaluator
        in {
            "agent_interaction_selected",
        }
        and bool(milestone.event.human_actions or milestone.event.task_updates)
        and result.event_occurred
        and not result.reaction_observed
    )
    if not missing:
        return _absent()
    event_steps = tuple(
        result.event_step for _, result in missing if result.event_step is not None
    )
    return FailureVerdict(
        occurred=True,
        recovered=False,
        contributed_to_failure=True,
        start_step=min(event_steps),
        end_step=max(event_steps),
        evidence=tuple(
            {
                "step": result.event_step,
                "milestone_id": milestone.milestone_id,
                "expected_interaction": dict(milestone.outcome.ground_truth),
            }
            for milestone, result in missing
        ),
    )


def _proactivity_deficit(
    context: FailureAnalysisContext,
) -> FailureVerdict:
    deficits: list[dict[str, Any]] = []
    milestones = context.spec.milestones
    results = {
        result.milestone_id: result for result in context.task_progress.milestones
    }
    events = {result.milestone_id: result for result in context.task_progress.events}
    completed_steps = {
        **{key: result.reaction_step for key, result in results.items()},
        **{key: result.completion_step for key, result in events.items()},
    }
    for milestone in milestones:
        result = results.get(milestone.milestone_id) or events[milestone.milestone_id]
        observed = (
            result.reaction_observed
            if milestone.kind == "scored"
            else result.completion_observed
        )
        if (
            milestone.outcome.evaluator == "agent_interaction_selected"
            and not milestone.event.human_actions
            and not milestone.event.task_updates
            and result.event_occurred
            and not observed
        ):
            deficits.append(
                {
                    "step": result.event_step,
                    "last_opportunity_step": (
                        context.trajectory.steps[-1].decision_step
                    ),
                    "milestone_id": milestone.milestone_id,
                    "expected_interaction": dict(milestone.outcome.ground_truth),
                    "opportunity_steps": list(
                        range(
                            int(result.event_step),
                            context.trajectory.steps[-1].decision_step + 1,
                        )
                    ),
                }
            )
        if (
            milestone.outcome.evaluator == "agent_action_selected"
            and milestone.outcome.ground_truth.get("action") == "report_done"
            and result.event_occurred
            and not observed
        ):
            deficits.append(
                {
                    "step": result.event_step,
                    "last_opportunity_step": (
                        context.trajectory.steps[-1].decision_step
                    ),
                    "milestone_id": milestone.milestone_id,
                    "expected_interaction": dict(milestone.outcome.ground_truth),
                    "opportunity_steps": list(
                        range(
                            int(result.event_step),
                            context.trajectory.steps[-1].decision_step + 1,
                        )
                    ),
                }
            )
        expected = tuple(
            condition
            for condition in milestone.preconditions
            if isinstance(condition, MilestoneCondition)
            and condition.detector == "agent_interaction_selected"
        )
        if not expected or result.event_occurred:
            continue
        parent_steps = [
            completed_steps[condition.milestone_ref]
            for condition in milestone.preconditions
            if isinstance(condition, MilestoneReference)
        ]
        if any(step is None for step in parent_steps):
            continue
        previous_step = max(parent_steps, default=None)
        eligible_steps = tuple(
            step
            for step in context.trajectory.steps
            if previous_step is None or step.decision_step >= previous_step
        )
        for interaction in expected:
            if any(
                bool(step.condition_values.get(interaction.condition_id))
                for step in eligible_steps
            ):
                continue
            other_conditions = tuple(
                condition
                for condition in milestone.preconditions
                if isinstance(condition, MilestoneCondition)
                and condition.condition_id != interaction.condition_id
            )
            opportunities = tuple(
                step.decision_step
                for step in eligible_steps
                if all(
                    bool(step.condition_values.get(condition.condition_id))
                    for condition in other_conditions
                )
            )
            if not opportunities:
                continue
            deficits.append(
                {
                    "step": opportunities[0],
                    "last_opportunity_step": opportunities[-1],
                    "milestone_id": milestone.milestone_id,
                    "expected_interaction": dict(interaction.arguments),
                    "opportunity_steps": list(opportunities),
                }
            )
    if not deficits:
        return _absent()
    return FailureVerdict(
        occurred=True,
        recovered=False,
        contributed_to_failure=True,
        start_step=min(item["step"] for item in deficits),
        end_step=max(item["last_opportunity_step"] for item in deficits),
        evidence=tuple(deficits),
    )


def _coordination_failure(
    context: FailureAnalysisContext,
) -> FailureVerdict:
    candidates = tuple(
        (milestone, result)
        for milestone, result in _scored_results(context)
        if milestone.outcome.evaluator == "handover_accepted"
        and result.event_occurred
        and not result.reaction_observed
    )
    if not candidates:
        return _absent()
    steps = tuple(
        result.event_step for _, result in candidates if result.event_step is not None
    )
    return FailureVerdict(
        occurred=True,
        recovered=False,
        contributed_to_failure=True,
        start_step=min(steps),
        end_step=max(steps),
        evidence=tuple(
            {
                "step": result.event_step,
                "milestone_id": milestone.milestone_id,
                "expected_reaction": milestone.outcome.evaluator,
                "ground_truth": dict(milestone.outcome.ground_truth),
            }
            for milestone, result in candidates
        ),
    )


def _recovery_failure(
    context: FailureAnalysisContext,
) -> FailureVerdict:
    repeats: list[dict[str, Any]] = []
    for first, second in zip(
        context.trajectory.steps,
        context.trajectory.steps[1:],
    ):
        first_signature = _failed_action_signature(first)
        second_signature = _failed_action_signature(second)
        if first_signature is None or second_signature is None:
            continue
        if first_signature != second_signature:
            continue
        repeats.append(
            {
                "step": first.decision_step,
                "repeated_at_step": second.decision_step,
                "action": _primary_action_name(first),
                "failure_signature": first_signature[1],
                "action_signature": first_signature[0],
            }
        )
    if not repeats:
        return _absent()
    last_repeat_by_action = {
        str(item["action"]): max(
            int(other["repeated_at_step"])
            for other in repeats
            if other["action"] == item["action"]
        )
        for item in repeats
    }
    recovered = all(
        any(
            step.decision_step > last_repeat
            and _primary_action_name(step) == action
            and _step_succeeded(step)
            for step in context.trajectory.steps
        )
        for action, last_repeat in last_repeat_by_action.items()
    )
    last_repeat = max(last_repeat_by_action.values())
    return FailureVerdict(
        occurred=True,
        recovered=recovered,
        contributed_to_failure=not recovered,
        start_step=min(item["step"] for item in repeats),
        end_step=last_repeat,
        evidence=tuple(repeats),
    )


def _failed_action_signature(
    step: TrajectoryStep,
) -> tuple[str, str] | None:
    if _step_succeeded(step):
        return None
    failure_code = _grasp_failure_code(step)
    feedback = str(step.feedback.get("feedback") or "").strip()
    if not failure_code and not feedback:
        return None
    return _action_signature(step), failure_code or feedback


def _action_signature(step: TrajectoryStep) -> str:
    calls = _action_calls(step)
    if not calls:
        return ""
    tool, arguments = calls[0]
    return json.dumps(
        {"tool": tool, "args": arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _step_succeeded(step: TrajectoryStep) -> bool:
    grasp = _mapping(step.feedback.get("grasp_skill"))
    return bool(step.feedback.get("executed") or grasp.get("transport_ready"))


def _target_object_ids(
    context: FailureAnalysisContext, step: TrajectoryStep, *, phase=None
) -> set[str]:
    targets: set[str] = set()
    for milestone in context.spec.milestones:
        bindings = (*milestone.preconditions, milestone.outcome)
        for binding in bindings:
            if not isinstance(binding, (MilestoneCondition, MilestoneReaction)):
                continue
            arguments = getattr(binding, "arguments", None)
            if arguments is None:
                arguments = outcome_ground_truth(milestone, step, phase=phase)
            object_id = str(arguments.get("object_id") or "").strip()
            if object_id:
                targets.add(object_id)
    return targets


def _action_calls(step: TrajectoryStep) -> tuple[tuple[str, dict[str, Any]], ...]:
    # One row is one executed serial call. ``calls`` is the authored block and
    # may contain suffixes that have not run, so it is not execution evidence.
    name = str(
        step.action.get("tool")
        or step.action.get("call_name")
        or step.action.get("name")
        or ""
    ).strip()
    if name:
        arguments = _mapping(step.action.get("arguments") or step.action.get("args"))
        return ((name, arguments),)
    return ()


def _primary_action_name(step: TrajectoryStep) -> str:
    calls = _action_calls(step)
    return calls[0][0] if calls else ""


def _robot_xy(step: TrajectoryStep) -> tuple[float, float] | None:
    position = _mapping(step.hidden_state.get("robot")).get("position")
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return None
    return float(position[0]), float(position[1])


def _robot_yaw(step: TrajectoryStep) -> float | None:
    yaw = _mapping(step.hidden_state.get("robot")).get("yaw_rad")
    if yaw is None:
        return None
    value = float(yaw)
    return value if math.isfinite(value) else None


def _pose_progress(
    steps: tuple[TrajectoryStep, ...],
) -> tuple[float, float] | None:
    positions = tuple(
        position for step in steps if (position := _robot_xy(step)) is not None
    )
    if len(positions) < 2:
        return None
    headings = tuple(yaw for step in steps if (yaw := _robot_yaw(step)) is not None)
    heading_progress = (
        abs(
            math.atan2(
                math.sin(headings[-1] - headings[0]),
                math.cos(headings[-1] - headings[0]),
            )
        )
        if len(headings) >= 2
        else 0.0
    )
    return math.dist(positions[0], positions[-1]), heading_progress


def _grasp_failure_code(step: TrajectoryStep) -> str:
    return str(_mapping(step.feedback.get("grasp_failure")).get("code") or "")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _absent() -> FailureVerdict:
    return FailureVerdict(False, False, False)


__all__ = [
    "NAVIGATION_STUCK_WINDOW_DECISIONS",
    "deterministic_failure_analyzers",
]
