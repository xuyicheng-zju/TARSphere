"""Evaluate one uninterrupted TARSphere trajectory."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Protocol, Sequence

from pydantic import ValidationError

from tarsphere.contracts.structured_plan import StructuredPlan
from tarsphere.evaluation.analyzer import (
    FailureAnalysisContext,
    FailureAttributionAnalyzer,
)
from tarsphere.evaluation.event_vocabulary import EventType
from tarsphere.evaluation.schema import (
    DECLARATIVE_MILESTONE_EVALUATORS,
    TARSPHERE_SCORING_VERSION,
    PRIMITIVE_REACTION_SAMPLING_PHASE,
    CausalEventResult,
    ConditionResult,
    EpisodeEvaluationReport,
    EpisodeEvaluationSpec,
    EpisodeTrajectory,
    MilestoneCondition,
    MilestoneReference,
    MilestoneResult,
    SemanticMilestone,
    TaskProgressResult,
    TrajectoryStep,
)
from tarsphere.evaluation.task_bindings import (
    outcome_ground_truth,
    validate_binding_snapshots,
)


@dataclass(frozen=True)
class MilestoneVerdict:
    """Progress quality and required-completion verdict from one evaluator."""

    score: float
    passed: bool
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class MilestoneScoreRequest:
    """Trajectory evidence supplied to one milestone evaluator."""

    milestone: SemanticMilestone
    event_step: TrajectoryStep
    reaction_step: TrajectoryStep
    conditions: Mapping[str, ConditionResult]
    public_context: tuple[Mapping[str, Any], ...]


class MilestoneEvaluator(Protocol):
    """Event-local quality and task-completion evaluator."""

    def __call__(self, request: MilestoneScoreRequest) -> MilestoneVerdict: ...


@dataclass(frozen=True)
class MilestoneReach:
    """Event/reaction indices for one node in the realized causal DAG."""

    event_index: int | None
    reaction_index: int | None


def milestone_dag_reaches(
    spec: EpisodeEvaluationSpec,
    steps: Sequence[TrajectoryStep],
    *,
    require_admissions: bool = False,
) -> tuple[MilestoneReach, ...]:
    """Resolve every required DAG node without invoking score judges."""

    reaches: list[MilestoneReach] = []
    reaches_by_id: dict[str, MilestoneReach] = {}
    milestones_by_id = {item.milestone_id: item for item in spec.milestones}
    for milestone in spec.milestones:
        references = tuple(
            item.milestone_ref
            for item in milestone.preconditions
            if isinstance(item, MilestoneReference)
        )
        reference_reactions = tuple(
            reaches_by_id[reference].reaction_index for reference in references
        )
        references_complete = all(index is not None for index in reference_reactions)
        first_allowed_index = (
            max(
                (
                    int(index)
                    + int(
                        _reaction_phase(milestones_by_id[reference])
                        > _reaction_phase(milestone)
                    )
                    for reference, index in zip(references, reference_reactions)
                ),
                default=0,
            )
            if references_complete
            else len(steps)
        )
        event_index = (
            next(
                (
                    index
                    for index in range(first_allowed_index, len(steps))
                    if (
                        is_internal_task_phase(milestone)
                        or _milestone_event_occurs(milestone, steps, index)
                    )
                    and _preconditions_supported(
                        milestone,
                        steps,
                        event_index=_event_admission_index(
                            milestone,
                            steps,
                            occurrence_index=index,
                            required=require_admissions,
                        ),
                    )
                ),
                None,
            )
            if references_complete
            else None
        )

        reaction_index = None
        if event_index is not None:
            evaluator = milestone.outcome.evaluator
            if evaluator == "structured_plan_submitted":
                # Same delayed-reaction rule as adaptation: the first
                # structured plan at or after the event is the scored one.
                reaction_index = next(
                    (
                        index
                        for index in range(event_index, len(steps))
                        if _has_structured_plan(steps[index])
                    ),
                    None,
                )
            elif evaluator == "inspection_submitted":
                # Opening an inspection phase does not demand an answer.
                # The first eligible submission is final. Repeated submissions
                # cannot replace its score or move the causal completion time.
                reaction_index = next(
                    (
                        index
                        for index in range(event_index, len(steps))
                        if isinstance(
                            result := steps[index].agent_state.get("inspection"),
                            str,
                        )
                        and result.strip()
                    ),
                    None,
                )
            elif evaluator == "information_report_score":
                reaction_index = next(
                    (
                        index
                        for index in range(event_index, len(steps))
                        if _information_report_selected(
                            steps[index],
                            milestone,
                        )
                    ),
                    None,
                )
            else:
                reaction_index = next(
                    (
                        index
                        for index in range(event_index, len(steps))
                        if bool(
                            steps[index].reaction_values.get(
                                milestone.milestone_id, False
                            )
                        )
                    ),
                    None,
                )

        reach = MilestoneReach(event_index, reaction_index)
        reaches.append(reach)
        reaches_by_id[milestone.milestone_id] = reach
    return tuple(reaches)


def _has_structured_plan(step: TrajectoryStep) -> bool:
    try:
        StructuredPlan.model_validate(step.agent_state.get("plan"))
    except ValidationError:
        return False
    return True


def _reaction_phase(milestone: SemanticMilestone) -> int:
    """Order actual evidence within a row, not evaluator invocation time."""

    if milestone.outcome.evaluator == "information_report_score":
        return 2  # The selected report must actually be delivered post-action.
    phase = PRIMITIVE_REACTION_SAMPLING_PHASE.get(
        milestone.outcome.evaluator,
        "decision",
    )
    return {"pre_decision": 0, "decision": 1, "post_action": 2}[phase]


def is_internal_task_phase(milestone: SemanticMilestone) -> bool:
    """A dependency-only phase opens on its causal facts, not a later NPC tick.

    Public utterances, world changes and live primitive preconditions still
    require their recorded event. No external event is synthesized here.
    """

    return (
        milestone.event.type
        in {EventType.TASK_PHASE_OPENED, EventType.CONTRIBUTION_REQUESTED}
        and not milestone.event.human_actions
        and not milestone.event.task_updates
        and bool(milestone.preconditions)
        and all(
            isinstance(item, MilestoneReference) for item in milestone.preconditions
        )
    )


def completed_milestone_ids(
    spec: EpisodeEvaluationSpec,
    steps: Sequence[TrajectoryStep],
) -> frozenset[str]:
    """Return evaluator-confirmed reaction IDs in the causal DAG."""

    return frozenset(
        milestone.milestone_id
        for milestone, reach in zip(spec.milestones, milestone_dag_reaches(spec, steps))
        if reach.reaction_index is not None
    )


def causal_graph_completed(
    spec: EpisodeEvaluationSpec,
    steps: Sequence[TrajectoryStep],
) -> bool:
    """Return whether every required milestone reaction was observed."""

    completed = completed_milestone_ids(spec, steps)
    scored = [node for node in spec.milestones if node.kind == "scored"]
    return bool(scored) and all(node.milestone_id in completed for node in scored)


def evaluate_trajectory(
    spec: EpisodeEvaluationSpec,
    trajectory: EpisodeTrajectory,
    *,
    failure_analyzer: FailureAttributionAnalyzer,
    milestone_evaluators: Mapping[str, MilestoneEvaluator],
) -> EpisodeEvaluationReport:
    """Produce causal progress and failure attribution from one trajectory."""

    progress, conditions = _score_trajectory(
        spec,
        trajectory,
        milestone_evaluators=milestone_evaluators,
    )
    failures = failure_analyzer.analyze(
        FailureAnalysisContext(
            spec=spec,
            trajectory=trajectory,
            task_progress=progress,
            conditions=conditions,
        )
    )
    return EpisodeEvaluationReport(
        episode_id=trajectory.episode_id,
        task_profile=trajectory.task_profile,
        task_progress=progress,
        failure_attribution=failures,
        decision_steps=trajectory.decision_steps,
        action_steps=len(trajectory.steps),
        physics_steps=trajectory.steps[-1].physics_step,
        termination_reason=trajectory.termination_reason,
    )


def score_trajectory(
    spec: EpisodeEvaluationSpec,
    trajectory: EpisodeTrajectory,
    *,
    milestone_evaluators: Mapping[str, MilestoneEvaluator],
) -> TaskProgressResult:
    """Return the official task score without running failure attribution."""

    progress, _ = _score_trajectory(
        spec,
        trajectory,
        milestone_evaluators=milestone_evaluators,
    )
    return progress


def _score_trajectory(
    spec: EpisodeEvaluationSpec,
    trajectory: EpisodeTrajectory,
    *,
    milestone_evaluators: Mapping[str, MilestoneEvaluator],
) -> tuple[TaskProgressResult, Mapping[str, ConditionResult]]:
    """Validate one trajectory and compute its causal task progress."""

    require_official_trajectory(trajectory)
    validate_binding_snapshots(spec, trajectory.steps)
    if trajectory.task_profile != spec.task_profile:
        raise ValueError("trajectory task profile differs from its evaluation contract")
    known_conditions = {condition.condition_id for condition in spec.conditions()}
    primitive_reactions = {
        milestone.milestone_id
        for milestone in spec.milestones
        if milestone.outcome.evaluator not in DECLARATIVE_MILESTONE_EVALUATORS
    }
    for step in trajectory.steps:
        provided = set(step.condition_values)
        if provided != known_conditions:
            missing = sorted(known_conditions - provided)
            unexpected = sorted(provided - known_conditions)
            raise ValueError(
                f"trajectory step {step.decision_step} condition sample differs "
                f"from contract; missing={missing}, unexpected={unexpected}"
            )
        provided_reactions = set(step.reaction_values)
        if provided_reactions != primitive_reactions:
            missing = sorted(primitive_reactions - provided_reactions)
            unexpected = sorted(provided_reactions - primitive_reactions)
            raise ValueError(
                f"trajectory step {step.decision_step} reaction sample differs "
                f"from contract; missing={missing}, unexpected={unexpected}"
            )

    _validate_trajectory_events(spec, trajectory.steps)

    conditions = _summarize_conditions(spec, trajectory.steps)
    progress = _evaluate_progress(
        spec,
        trajectory.steps,
        conditions,
        milestone_evaluators,
    )
    # These runtime termination reasons already mean unsafe task failure.
    # Preserve earned milestone points; ordinary contact attribution does not
    # override success. A final-row grasp/report must not erase a fall/drop.
    if trajectory.termination_reason in {"object_dropped", "safety_violation"}:
        progress = progress.model_copy(update={"success": False})
    return progress, conditions


def _validate_trajectory_events(
    spec: EpisodeEvaluationSpec,
    steps: Sequence[TrajectoryStep],
) -> None:
    """Reject missing, duplicate, unknown, or causally impossible event logs."""

    known = {milestone.milestone_id for milestone in spec.milestones}
    occurrences: dict[str, int] = {}
    admissions: dict[str, int] = {}
    for index, step in enumerate(steps):
        if len(step.occurred_events) != len(set(step.occurred_events)):
            raise ValueError(
                f"trajectory step {step.decision_step} repeats occurred_events"
            )
        if len(step.admitted_events) != len(set(step.admitted_events)):
            raise ValueError(
                f"trajectory step {step.decision_step} repeats admitted_events"
            )
        unknown = (set(step.occurred_events) | set(step.admitted_events)) - known
        if unknown:
            raise ValueError(
                f"trajectory step {step.decision_step} names unknown events: "
                f"{sorted(unknown)}"
            )
        for milestone_id in step.admitted_events:
            if milestone_id in admissions:
                raise ValueError(
                    f"milestone {milestone_id!r} was admitted more than once"
                )
            admissions[milestone_id] = index
        for milestone_id in step.occurred_events:
            if milestone_id in occurrences:
                raise ValueError(f"milestone {milestone_id!r} occurred more than once")
            occurrences[milestone_id] = index

    nodes = {node.milestone_id: node for node in spec.milestones}
    for milestone_id, occurrence_index in occurrences.items():
        node = nodes[milestone_id]
        if not is_internal_task_phase(node):
            admission_index = admissions.get(milestone_id)
            if admission_index is None or admission_index > occurrence_index:
                raise ValueError(
                    f"external milestone {milestone_id!r} occurrence lacks a "
                    "prior one-shot admission"
                )

    reaches = milestone_dag_reaches(spec, steps, require_admissions=True)
    for node, reach in zip(spec.milestones, reaches):
        occurrence_index = occurrences.get(node.milestone_id)
        if (
            occurrence_index is not None
            and not is_internal_task_phase(node)
            and reach.event_index != occurrence_index
        ):
            raise ValueError(
                f"milestone {node.milestone_id!r} occurred outside its causal "
                "preconditions or predecessor order"
            )


def require_official_trajectory(trajectory: EpisodeTrajectory) -> None:
    """Reject controller diagnostics before computing official metrics."""

    if trajectory.controller_diagnostics:
        raise ValueError(
            "official TARSphere evaluation rejects controller diagnostics: "
            f"{sorted(trajectory.controller_diagnostics)}"
        )


def _summarize_conditions(
    spec: EpisodeEvaluationSpec,
    steps: Sequence[TrajectoryStep],
) -> dict[str, ConditionResult]:
    results: dict[str, ConditionResult] = {}
    for definition in spec.conditions():
        true_steps = tuple(
            step.decision_step
            for step in steps
            if bool(step.condition_values[definition.condition_id])
        )
        results[definition.condition_id] = ConditionResult(
            condition_id=definition.condition_id,
            value=bool(true_steps),
            true_steps=true_steps,
        )
    return results


def _evaluate_progress(
    spec: EpisodeEvaluationSpec,
    steps: Sequence[TrajectoryStep],
    conditions: Mapping[str, ConditionResult],
    milestone_evaluators: Mapping[str, MilestoneEvaluator],
) -> TaskProgressResult:
    results: list[MilestoneResult] = []
    events: list[CausalEventResult] = []
    reaches = milestone_dag_reaches(spec, steps, require_admissions=True)
    # DAG admission is driven by observable completion, not by the later
    # scoring verdict. A submitted plan/inspection unlocks its child even when
    # the submission ultimately loses its own point. Otherwise one imperfect
    # early answer erases independently observable downstream performance.
    causally_completed: dict[str, bool] = {}
    for milestone, reach in zip(spec.milestones, reaches):
        event_index = reach.event_index
        reaction_index = reach.reaction_index
        reaction_observed = reaction_index is not None
        parent_ids = tuple(
            precondition.milestone_ref
            for precondition in milestone.preconditions
            if isinstance(precondition, MilestoneReference)
        )
        blocked_by = tuple(
            parent_id for parent_id in parent_ids if not causally_completed[parent_id]
        )
        if blocked_by:
            causal_status = "blocked"
        elif event_index is None:
            causal_status = "not_reached"
        elif not reaction_observed:
            causal_status = "failed"
        else:
            causal_status = "passed"
        if milestone.kind == "event":
            events.append(
                CausalEventResult(
                    milestone_id=milestone.milestone_id,
                    event_occurred=event_index is not None,
                    event_step=steps[event_index].decision_step
                    if event_index is not None
                    else None,
                    completion_observed=reaction_observed,
                    completion_step=(
                        steps[reaction_index].decision_step
                        if reaction_index is not None
                        else None
                    ),
                    causal_status=causal_status,
                    blocked_by=blocked_by,
                    evidence=(
                        dict(
                            steps[reaction_index].reaction_evidence.get(
                                milestone.milestone_id, {}
                            )
                        )
                        if reaction_index is not None
                        else {}
                    ),
                )
            )
            causally_completed[milestone.milestone_id] = causal_status == "passed"
            continue
        local_score = 0.0
        local_pass: bool | None = (
            False if event_index is not None and not reaction_observed else None
        )
        evidence: Mapping[str, Any] = (
            {"reason": "agent_reaction_not_observed"}
            if event_index is not None and not reaction_observed
            else {}
        )
        if reaction_observed:
            evaluator = milestone_evaluators.get(milestone.outcome.evaluator)
            if evaluator is None:
                raise ValueError(
                    f"no milestone evaluator registered for "
                    f"{milestone.outcome.evaluator!r}"
                )
            verdict = evaluator(
                MilestoneScoreRequest(
                    milestone=milestone,
                    event_step=steps[event_index],
                    reaction_step=steps[reaction_index],
                    conditions=conditions,
                    public_context=_public_task_context(steps[: reaction_index + 1]),
                )
            )
            if not isinstance(verdict.passed, bool):
                raise TypeError(
                    f"milestone {milestone.milestone_id!r} evaluator must "
                    "return a boolean passed verdict"
                )
            if (
                isinstance(verdict.score, bool)
                or not isinstance(verdict.score, (int, float))
                or not math.isfinite(verdict.score)
                or not 0.0 <= verdict.score <= 1.0
            ):
                raise ValueError("milestone score must be a finite number in [0,1]")
            local_score = float(verdict.score)
            local_pass = verdict.passed
            evidence = dict(verdict.evidence)
        if causal_status == "passed" and local_pass is not True:
            causal_status = "failed"
        causally_completed[milestone.milestone_id] = (
            not blocked_by and event_index is not None and reaction_observed
        )
        results.append(
            MilestoneResult(
                milestone_id=milestone.milestone_id,
                event_occurred=event_index is not None,
                event_step=(
                    steps[event_index].decision_step
                    if event_index is not None
                    else None
                ),
                reaction_observed=reaction_observed,
                reaction_step=(
                    steps[reaction_index].decision_step
                    if reaction_index is not None
                    else None
                ),
                local_pass=local_pass,
                causal_status=causal_status,
                blocked_by=blocked_by,
                earned_point=local_score
                if causally_completed[milestone.milestone_id]
                else 0.0,
                evidence=evidence,
            )
        )

    ordered_results = tuple(results)
    total_points = len(ordered_results)
    if not total_points:
        raise ValueError("Evaluation requires at least one scored task outcome")
    earned_points = sum(result.earned_point for result in ordered_results)
    return TaskProgressResult(
        scoring_version=TARSPHERE_SCORING_VERSION,
        success=all(result.causal_status == "passed" for result in ordered_results),
        completion_rate=earned_points / total_points,
        earned_points=earned_points,
        total_points=total_points,
        milestones=ordered_results,
        events=tuple(events),
    )


def _public_task_context(
    steps: Sequence[TrajectoryStep],
) -> tuple[Mapping[str, Any], ...]:
    """Public language delivered before the scored submission, without future feedback."""

    context: list[Mapping[str, Any]] = []
    for step in steps:
        text = step.observation.get("text")
        if isinstance(text, str) and text.strip():
            context.append({"step": step.decision_step, "text": text})
        for event in step.observation.get("public_events", ()):
            if (
                event.get("event_type") == "language"
                and str(event.get("text") or "").strip()
            ):
                context.append(
                    {
                        "step": step.decision_step,
                        "speaker": str(event.get("speaker") or ""),
                        "text": str(event["text"]),
                    }
                )
    return tuple(context)


def _milestone_event_occurs(
    milestone: SemanticMilestone,
    steps: Sequence[TrajectoryStep],
    index: int,
) -> bool:
    return milestone.milestone_id in steps[index].occurred_events


def _event_admission_index(
    milestone: SemanticMilestone,
    steps: Sequence[TrajectoryStep],
    *,
    occurrence_index: int,
    required: bool,
) -> int:
    """Use the consumed one-shot trigger, not delayed queued delivery time."""

    if is_internal_task_phase(milestone):
        return occurrence_index
    admitted = next(
        (
            index
            for index in range(occurrence_index + 1)
            if milestone.milestone_id in getattr(steps[index], "admitted_events", ())
        ),
        None,
    )
    if admitted is None and required:
        raise ValueError(
            f"external milestone {milestone.milestone_id!r} occurred without "
            "a recorded one-shot admission; causal preconditions cannot be "
            "reconstructed safely"
        )
    return occurrence_index if admitted is None else admitted


def _preconditions_supported(
    milestone: SemanticMilestone,
    steps: Sequence[TrajectoryStep],
    *,
    event_index: int,
) -> bool:
    """Require every live primitive precondition at the event itself.

    Historical task facts belong in predecessor milestone references.  A
    distance, visibility, or current-action predicate is a live condition and
    cannot authorize an event after it has become false.
    """

    primitive_conditions = tuple(
        condition
        for condition in milestone.preconditions
        if isinstance(condition, MilestoneCondition)
    )
    if not primitive_conditions:
        return True
    return all(
        steps[event_index].condition_values[condition.condition_id]
        for condition in primitive_conditions
    )


def _information_report_selected(
    step: TrajectoryStep,
    milestone: SemanticMilestone,
) -> bool:
    """Require selected and delivered speech; answer correctness is judge-owned."""

    human_id = str(outcome_ground_truth(milestone, step)["human_id"])
    if not (
        str(step.action.get("name") or "") == "speak"
        and str(step.action.get("recipient") or "") == human_id
        and bool(str(step.action.get("message") or "").strip())
    ):
        return False
    pre_decision_world = step.hidden_state.get("pre_decision_world")
    if not isinstance(pre_decision_world, dict):
        raise ValueError(
            "information_report_score requires evaluator-private "
            "pre_decision_world state from the exact observation used to "
            "select the report"
        )
    human_state = (pre_decision_world.get("humans") or {}).get(human_id) or {}
    delivered = step.hidden_state.get("robot_message_delivered")
    if not isinstance(delivered, bool):
        raise ValueError(
            "information_report_score requires evaluator-private "
            "robot_message_delivered from the actual post-action Human tick"
        )
    return human_state.get("robot_can_communicate") is True and delivered


__all__ = [
    "MilestoneVerdict",
    "MilestoneScoreRequest",
    "MilestoneEvaluator",
    "milestone_dag_reaches",
    "is_internal_task_phase",
    "completed_milestone_ids",
    "causal_graph_completed",
    "evaluate_trajectory",
    "require_official_trajectory",
    "score_trajectory",
]
