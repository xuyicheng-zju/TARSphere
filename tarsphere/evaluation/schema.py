"""Frozen TARSphere causal-event evaluation contracts.

Each milestone has primitive and causal preconditions, one event, and one
scored Agent reaction.  The milestone references form a monotone DAG;
required execution and interaction conditions determine success independently
of continuous plan and inspection quality.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Annotated, Any, Literal, Tuple, Union, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.contracts.human_interaction import (
    HUMAN_COMMUNICATION_RADIUS_M,
)
from tarsphere.contracts.task_profile import SCORED_OUTCOME_IDS, TaskProfile
from tarsphere.evaluation.event_vocabulary import (
    DETECTOR_ONLY_EVENT_TYPES,
    EVENT_TYPES_REQUIRING_HUMAN_GOTO,
    EventType,
)


TARSPHERE_EVALUATION_VERSION = "tarsphere.evaluation.v14"
TARSPHERE_SCORING_VERSION = "tarsphere.scoring.v1"

Identifier = Annotated[
    str,
    Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Stable lowercase identifier within one episode.",
    ),
]
NonEmptyText = Annotated[
    str,
    Field(min_length=1, description="Human-readable benchmark contract text."),
]
DetectorName = Literal[
    "human_event_completed",
    "agent_action_selected",
    "agent_interaction_selected",
    "information_received",
    "visual_targets_observed",
    "robot_in_place",
    "robot_near_human",
    "human_visible",
    "robot_holding_object",
    "object_supported_by",
    "handover_accepted",
    "object_dropped",
]
DETECTOR_NAMES = frozenset(get_args(DetectorName))
CapabilityName = Literal[
    "perception",
    "planning",
    "execution",
    "interaction",
    "adaptation",
]
FailureAnalysisMethod = Literal["deterministic", "semantic"]
MilestoneEvaluatorName = Literal[
    "human_event_completed",
    "structured_plan_submitted",
    "inspection_submitted",
    "information_report_score",
    "agent_action_selected",
    "agent_interaction_selected",
    "information_received",
    "visual_targets_observed",
    "robot_in_place",
    "robot_near_human",
    "human_visible",
    "robot_holding_object",
    "object_supported_by",
    "handover_accepted",
    "object_dropped",
]
DECLARATIVE_MILESTONE_EVALUATORS = frozenset(
    {
        "structured_plan_submitted",
        "inspection_submitted",
        "information_report_score",
    }
)

# Primitive reactions are sampled at the instant their evidence exists.  A
# visual fact belongs to the RGB-D observation supplied to the policy; a
# selected interaction belongs to the returned decision; physical state
# changes belong to the completed environment action.  Keeping this mapping
# exhaustive prevents a new evaluator from silently inheriting the wrong
# phase.
ReactionSamplingPhase = Literal["pre_decision", "decision", "post_action"]
PRIMITIVE_REACTION_SAMPLING_PHASE: dict[str, ReactionSamplingPhase] = {
    # Authored Human actions are applied by the pre-agent tick. A delivered
    # utterance or completed move is available before this decision's image.
    "human_event_completed": "pre_decision",
    "visual_targets_observed": "pre_decision",
    "human_visible": "pre_decision",
    "agent_action_selected": "decision",
    "agent_interaction_selected": "decision",
    # An accepted authored answer actually delivered after the question.
    "information_received": "post_action",
    "robot_in_place": "post_action",
    "robot_near_human": "post_action",
    "robot_holding_object": "post_action",
    "object_supported_by": "post_action",
    "handover_accepted": "post_action",
    "object_dropped": "post_action",
}
_PRIMITIVE_MILESTONE_EVALUATORS = (
    frozenset(get_args(MilestoneEvaluatorName)) - DECLARATIVE_MILESTONE_EVALUATORS
)
if set(PRIMITIVE_REACTION_SAMPLING_PHASE) != _PRIMITIVE_MILESTONE_EVALUATORS:
    raise RuntimeError(
        "primitive reaction sampling phases must cover the evaluator schema "
        "exactly; missing="
        f"{sorted(_PRIMITIVE_MILESTONE_EVALUATORS - set(PRIMITIVE_REACTION_SAMPLING_PHASE))}, "
        "unexpected="
        f"{sorted(set(PRIMITIVE_REACTION_SAMPLING_PHASE) - _PRIMITIVE_MILESTONE_EVALUATORS)}"
    )
FailureEvent = Literal[
    "human_misgrounding",
    "target_misgrounding",
    "information_omission",
    "goal_misalignment",
    "causal_violation",
    "plan_infeasibility",
    "navigation_stuck",
    "safety_violation",
    "manipulation_failure",
    "grasp_instability",
    "reactive_shortfall",
    "proactivity_deficit",
    "permission_violation",
    "coordination_failure",
    "change_unawareness",
    "update_delay",
    "obsolete_commitment",
    "recovery_failure",
]


class StrictModel(BaseModel):
    """Immutable current-release schema base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MilestoneCondition(StrictModel):
    """One primitive detector binding owned by exactly one milestone slot."""

    kind: Literal["detector"] = "detector"
    condition_id: Identifier = Field(
        description="Unique condition name within the episode milestone graph.",
    )
    detector: DetectorName = Field(
        description="Closed-set detector that samples only the current step.",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Episode-grounded arguments consumed by the detector.",
    )
    description: NonEmptyText = Field(
        description="Semantic meaning of this hidden fact in the task script.",
    )

    @model_validator(mode="after")
    def validate_detector_arguments(self) -> "MilestoneCondition":
        _validate_detector_arguments(self.detector, self.arguments)
        if self.detector == "information_received":
            raise ValueError(
                "information_received is a delivered action outcome; use it "
                "as a reaction and reference that milestone for later phases"
            )
        return self


class MilestoneReference(StrictModel):
    """One causal edge to an evaluator-confirmed milestone reaction."""

    kind: Literal["milestone"] = "milestone"
    milestone_ref: Identifier = Field(
        description="Earlier milestone whose reaction must be complete.",
    )


MilestonePrecondition = Annotated[
    Union[MilestoneCondition, MilestoneReference],
    Field(discriminator="kind"),
]


class PublicTaskUpdate(StrictModel):
    """One machine-readable task-state transition delivered by Human speech."""

    field: Identifier = Field(description="Stable public task-state field.")
    previous_value_ref: str | None = Field(
        default=None,
        description="Previously public symbolic value, or null when unknown.",
    )
    new_value_ref: NonEmptyText = Field(
        description="New public symbolic value activated by this event.",
    )
    source_human_id: NonEmptyText = Field(
        description="Human whose utterance authoritatively publishes the update.",
    )
    delivery: Literal["speech"] = Field(
        default="speech",
        description="Current runtime delivery channel.",
    )
    protected_action: Identifier | None = Field(
        default=None,
        description="Action controlled by a permission update, when applicable.",
    )
    outcome_updates: dict[Identifier, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Complete replacement GT bindings for not-yet-observed scored outcomes. "
            "Activated atomically when this source utterance is delivered; never public."
        ),
    )


class MilestoneEvent(StrictModel):
    """Concrete public or physical occurrence at one causal graph node."""

    type: EventType = Field(
        description=(
            "Typed public-state transition. Wording changes do not create a "
            "new causal event kind or diversity credit."
        ),
    )
    description: NonEmptyText = Field(
        description="Concrete event exposed at this graph node.",
    )

    human_actions: Tuple[dict[str, Any], ...] = Field(
        default=(),
        description="Public Human actions emitted when this event occurs.",
    )
    task_updates: Tuple[PublicTaskUpdate, ...] = Field(
        default=(),
        description="Public task facts published by this event.",
    )

    @model_validator(mode="after")
    def validate_event_payload(self) -> "MilestoneEvent":
        if self.type in EVENT_TYPES_REQUIRING_HUMAN_GOTO and not any(
            str(action.get("command") or "") == "goto" for action in self.human_actions
        ):
            raise ValueError(f"{self.type.value} requires a real Human goto action")
        if (
            self.type not in DETECTOR_ONLY_EVENT_TYPES
            and not self.human_actions
            and not self.task_updates
        ):
            raise ValueError(
                f"{self.type.value} requires a public Human action or task update"
            )
        speaking_humans = {
            str(action.get("human") or "")
            for action in self.human_actions
            if str(action.get("command") or "") in {"speak", "ask", "report_done"}
            and str(action.get("message") or "").strip()
        }
        missing_sources = sorted(
            {
                update.source_human_id
                for update in self.task_updates
                if update.source_human_id not in speaking_humans
            }
        )
        if missing_sources:
            raise ValueError(
                "task_updates must be delivered by a speaking Human in the "
                f"same event; missing sources={missing_sources!r}"
            )
        return self


class MilestoneReaction(StrictModel):
    """Agent behavior observed after an event and scored against episode GT."""

    description: NonEmptyText = Field(
        description="Agent result evaluated after this milestone event.",
    )
    evaluator: MilestoneEvaluatorName = Field(
        description="Single evaluator that owns this reaction score.",
    )
    ground_truth: dict[str, Any] = Field(
        default_factory=dict,
        description="Episode GT consumed only by the reaction evaluator.",
    )

    @model_validator(mode="after")
    def validate_evaluator_ground_truth(self) -> "MilestoneReaction":
        if self.evaluator == "structured_plan_submitted":
            _validate_planning_ground_truth(self.ground_truth)
        elif self.evaluator == "information_report_score":
            _validate_information_report_ground_truth(self.ground_truth)
        elif self.evaluator == "inspection_submitted":
            _validate_inspection_ground_truth(self.ground_truth)
        elif self.evaluator not in DECLARATIVE_MILESTONE_EVALUATORS:
            _validate_detector_arguments(self.evaluator, self.ground_truth)
        return self


class SemanticMilestone(StrictModel):
    """One causal graph node: preconditions, event, and Agent reaction."""

    milestone_id: Identifier = Field(
        description="Unique task-progress state name within the episode.",
    )
    kind: Literal["scored", "event"] = Field(
        default="scored",
        description="Scored task outcome or non-scoring causal event completion.",
    )
    preconditions: Tuple[MilestonePrecondition, ...] = Field(
        default=(),
        description=(
            "Conjunctive primitive detector bindings and milestone references. "
            "Fan-out and fan-in express parallel required branches."
        ),
    )
    event: MilestoneEvent = Field(
        description="Occurrence unlocked by the preconditions.",
    )
    reaction: MilestoneReaction | None = Field(
        default=None,
        description="Scored Agent result; present only on scored nodes.",
    )
    completion: MilestoneReaction | None = Field(
        default=None,
        description="Deterministic completion evidence; present only on event nodes and never scored.",
    )

    @property
    def outcome(self) -> MilestoneReaction:
        """The node's evidence contract, independently of scoring membership."""
        value = self.reaction if self.kind == "scored" else self.completion
        if value is None:
            raise ValueError(
                f"{self.milestone_id}: missing {self.kind} outcome contract"
            )
        return value

    @model_validator(mode="after")
    def validate_node_kind(self) -> "SemanticMilestone":
        if self.kind == "scored":
            if self.reaction is None or self.completion is not None:
                raise ValueError("scored nodes require reaction and forbid completion")
        else:
            if self.reaction is not None or self.completion is None:
                raise ValueError("event nodes require completion and forbid reaction")
            if self.completion.evaluator in DECLARATIVE_MILESTONE_EVALUATORS:
                raise ValueError(
                    "event completion must be deterministic, never an LLM score"
                )
        if self.event.type == "contribution_requested" and (
            self.kind != "event" or self.outcome.evaluator != "information_received"
        ):
            raise ValueError(
                "contribution_requested requires an unscored delivered acceptance receipt"
            )
        return self

    @model_validator(mode="after")
    def validate_directed_human_communication(self) -> "SemanticMilestone":
        """Keep Human delivery and Agent addressing directionally correct.

        Human-to-Agent public speech is unlocked by causal dependencies and
        the shared auditory radius only.  Agent-to-Human visibility and range
        are sampled by the reaction evaluator at the actual interaction step;
        placing ``human_visible`` here would incorrectly delay the Human event
        until the Robot happened to look at its speaker.
        """

        required_evaluators = {
            EventType.RESULT_REQUESTED: {"information_report_score"},
            EventType.INTERACTION_OPPORTUNITY_OPENED: {
                "agent_interaction_selected",
                "information_received",
            },
            EventType.CONTRIBUTION_REQUESTED: {
                "agent_interaction_selected",
                "information_received",
            },
        }
        allowed = required_evaluators.get(self.event.type)
        if allowed is not None and self.outcome.evaluator not in allowed:
            raise ValueError(
                f"event type {self.event.type.value!r} requires one of "
                f"{sorted(allowed)!r}, got {self.outcome.evaluator!r}"
            )

        speaking_humans: set[str] = set()
        for action in self.event.human_actions:
            if str(action.get("command") or "") not in {
                "ask",
                "speak",
                "report_done",
            }:
                continue
            human_id = str(action.get("human") or "").strip()
            if not human_id:
                raise ValueError(
                    f"milestone {self.milestone_id!r} has a directed Human "
                    "utterance without human"
                )
            speaking_humans.add(human_id)

        for human_id in sorted(speaking_humans):
            communication_gates = tuple(
                condition
                for condition in self.preconditions
                if isinstance(condition, MilestoneCondition)
                and condition.detector == "robot_near_human"
                and str(condition.arguments.get("human_id") or "") == human_id
            )
            if not communication_gates:
                # A rendezvous starts moving before the Robot is in range.
                # Speech still checks the shared radius at actual dispatch.
                actions = list(self.event.human_actions)
                speech_index = next(
                    i
                    for i, action in enumerate(actions)
                    if action.get("human") == human_id
                    and action.get("command") in {"speak", "ask", "report_done"}
                )
                if any(
                    action.get("human") == human_id and action.get("command") == "goto"
                    for action in actions[:speech_index]
                ):
                    continue
                raise ValueError(
                    f"milestone {self.milestone_id!r} communicates with "
                    f"Human {human_id!r} without a robot_near_human "
                    "communication "
                    "precondition for the same Human"
                )
            visibility_gates = tuple(
                condition
                for condition in self.preconditions
                if isinstance(condition, MilestoneCondition)
                and condition.detector == "human_visible"
                and str(condition.arguments.get("human_id") or "") == human_id
            )
            if visibility_gates:
                raise ValueError(
                    f"milestone {self.milestone_id!r} gates Human "
                    f"{human_id!r} speech on robot-camera visibility; "
                    "Human-to-Agent delivery must use causal dependencies "
                    "plus robot_near_human only"
                )
            if not any(
                math.isclose(
                    float(condition.arguments["distance"]),
                    HUMAN_COMMUNICATION_RADIUS_M,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for condition in communication_gates
            ):
                raise ValueError(
                    f"milestone {self.milestone_id!r} communication with "
                    f"Human {human_id!r} must use the shared "
                    f"{HUMAN_COMMUNICATION_RADIUS_M:g} m radius"
                )
        return self


class EpisodeEvaluationSpec(StrictModel):
    """Complete hidden evaluation contract for one uninterrupted episode."""

    schema_version: Literal["tarsphere.evaluation.v14"] = Field(
        default=TARSPHERE_EVALUATION_VERSION,
        description="Exact current TARSphere evaluation schema version.",
    )
    task_profile: TaskProfile = Field(
        description=(
            "One task mainline and one primary capability slice used for "
            "benchmark reporting and slice-specific validation."
        ),
    )
    milestones: Tuple[SemanticMilestone, ...] = Field(
        min_length=1,
        description=(
            "Topologically ordered causal DAG. Scored outcomes earn equal points; "
            "event completions have no score. Preconditions own required dependencies."
        ),
    )

    @model_validator(mode="after")
    def validate_references(self) -> "EpisodeEvaluationSpec":
        milestone_ids = tuple(item.milestone_id for item in self.milestones)
        duplicates = sorted(
            value for value, count in Counter(milestone_ids).items() if count > 1
        )
        if duplicates:
            raise ValueError(f"duplicate milestone ids: {duplicates}")

        known: set[str] = set()
        nodes = {node.milestone_id: node for node in self.milestones}
        scored_ids = tuple(
            node.milestone_id for node in self.milestones if node.kind == "scored"
        )
        if scored_ids != SCORED_OUTCOME_IDS:
            raise ValueError(
                "official evaluation requires exactly the five canonical "
                f"scored outcomes in order: {SCORED_OUTCOME_IDS!r}"
            )
        for milestone in self.milestones:
            references = tuple(
                item.milestone_ref
                for item in milestone.preconditions
                if isinstance(item, MilestoneReference)
            )
            duplicate_references = sorted(
                value for value, count in Counter(references).items() if count > 1
            )
            if duplicate_references:
                raise ValueError(
                    f"milestone {milestone.milestone_id!r} repeats milestone "
                    f"references: {duplicate_references}"
                )
            unknown = sorted(set(references) - known)
            if unknown:
                raise ValueError(
                    f"milestone {milestone.milestone_id!r} must reference only "
                    f"earlier declared milestones; unknown or forward={unknown}"
                )
            known.add(milestone.milestone_id)

            updated: set[str] = set()
            fields = [update.field for update in milestone.event.task_updates]
            if len(fields) != len(set(fields)):
                raise ValueError(
                    "An event must update each public task field at most once"
                )
            sources = {
                update.source_human_id
                for update in milestone.event.task_updates
                if update.outcome_updates
            }
            if len(sources) > 1:
                raise ValueError(
                    "An atomic task revision must have one authoritative speaker"
                )
            for update in milestone.event.task_updates:
                for outcome_id, truth in update.outcome_updates.items():
                    target = nodes.get(outcome_id)
                    if target is None or target.kind != "scored":
                        raise ValueError(
                            f"task update names unknown/non-scoring outcome {outcome_id!r}"
                        )
                    if (
                        outcome_id == "task_understanding"
                        or target.outcome.evaluator == "structured_plan_submitted"
                    ):
                        raise ValueError(
                            "task updates cannot rewrite the initial planning target"
                        )
                    if outcome_id in updated:
                        raise ValueError(
                            f"event updates outcome {outcome_id!r} more than once"
                        )
                    updated.add(outcome_id)
                    MilestoneReaction.model_validate(
                        {
                            "description": target.outcome.description,
                            "evaluator": target.outcome.evaluator,
                            "ground_truth": truth,
                        }
                    )

        parents = {
            node.milestone_id: {
                item.milestone_ref
                for item in node.preconditions
                if isinstance(item, MilestoneReference)
            }
            for node in self.milestones
        }

        def is_ancestor(ancestor: str, descendant: str) -> bool:
            frontier = list(parents[descendant])
            visited: set[str] = set()
            while frontier:
                current = frontier.pop()
                if current == ancestor:
                    return True
                if current not in visited:
                    visited.add(current)
                    frontier.extend(parents[current])
            return False

        for previous, current in zip(scored_ids, scored_ids[1:]):
            if not is_ancestor(previous, current):
                raise ValueError(
                    "canonical scored outcomes must remain causally downstream "
                    "in one direction; "
                    f"{previous!r} is not an ancestor of {current!r}"
                )

        conditions = tuple(
            condition
            for milestone in self.milestones
            for condition in milestone.preconditions
            if isinstance(condition, MilestoneCondition)
        )
        definitions: dict[str, MilestoneCondition] = {}
        for condition in conditions:
            previous = definitions.setdefault(condition.condition_id, condition)
            if previous != condition:
                raise ValueError(
                    "reused milestone condition ids must repeat the exact "
                    f"inline detector binding: {condition.condition_id!r}"
                )
        return self

    def conditions(self) -> Tuple[MilestoneCondition, ...]:
        """Return each inline condition once in first causal-use order."""

        unique: dict[str, MilestoneCondition] = {}
        for milestone in self.milestones:
            for condition in milestone.preconditions:
                if isinstance(condition, MilestoneCondition):
                    unique.setdefault(condition.condition_id, condition)
        return tuple(unique.values())


class TrajectoryStep(StrictModel):
    """One action outcome and its public, Agent-maintained, and hidden evidence."""

    decision_step: int = Field(
        ge=0,
        description=(
            "Zero-based action-outcome row index used for causal event timing; "
            "not the model-decision budget counter."
        ),
    )
    policy_inference: bool = Field(
        default=True,
        strict=True,
        description=(
            "Whether this row starts a fresh model decision. True for a "
            "single-action decision or rejected output; false for the queued "
            "suffix of the same decision. Diagnostic reference traces are "
            "not eligible for formal model-decision reporting. "
            "Transport retries belong to the same decision."
        ),
    )
    physics_step: int = Field(
        ge=0,
        description="Cumulative simulator step after the action completed.",
    )
    observation: dict[str, Any] = Field(
        default_factory=dict,
        description="Exact public multimodal observation supplied to the Agent.",
    )
    agent_state: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Agent-declared goal, plan, and task state at this decision. "
            "Event fields use the first accepted submission across successful "
            "same-decision tool rounds, not just the last raw model envelope."
        ),
    )
    action: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Executed Agent action. Runtime traces identify the exact call "
            "with tool, args, and tool_call_index; name records the resolved "
            "runtime action and calls retains the authored decision block."
        ),
    )
    feedback: dict[str, Any] = Field(
        default_factory=dict,
        description="Public execution feedback returned after the action.",
    )
    hidden_state: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Evaluator-only simulator and script state, explicitly separated "
            "into pre-decision and post-action snapshots where timing matters."
        ),
    )
    condition_values: dict[Identifier, bool] = Field(
        description=(
            "Inline milestone-condition values sampled at event selection "
            "before the Human tick; post-action values cannot retroactively "
            "authorize this event."
        ),
    )
    reaction_values: dict[Identifier, bool] = Field(
        default_factory=dict,
        description="Primitive milestone-reaction evaluator values at this step.",
    )
    reaction_evidence: dict[Identifier, dict[str, Any]] = Field(
        default_factory=dict,
        description="Evaluator-only metric evidence for primitive reactions.",
    )
    occurred_events: Tuple[Identifier, ...] = Field(
        default=(),
        description="Authored milestone events emitted on this decision step.",
    )
    admitted_events: Tuple[Identifier, ...] = Field(
        default=(),
        description=(
            "Authored Human events whose one-shot preconditions were consumed "
            "on this decision step, possibly before their queued actions completed."
        ),
    )


class EpisodeTrajectory(StrictModel):
    """Complete uninterrupted rollout consumed by offline evaluation."""

    episode_id: NonEmptyText = Field(description="Episode represented here.")
    task_profile: TaskProfile = Field(
        description="Episode task-mainline and capability-slice identity.",
    )
    steps: Tuple[TrajectoryStep, ...] = Field(
        min_length=1,
        description="Chronological action-outcome trace without resets or teleportation.",
    )
    termination_reason: NonEmptyText = Field(
        description="Runtime reason the single rollout ended.",
    )
    controller_diagnostics: Tuple[Identifier, ...] = Field(
        default=(),
        description=(
            "Private certification provenance or nonstandard controller dimensions; "
            "official evaluation requires this tuple to be empty."
        ),
    )

    @property
    def decision_steps(self) -> int:
        """Model decisions, independently of serial action expansion."""

        return sum(step.policy_inference for step in self.steps)

    @model_validator(mode="after")
    def validate_step_order(self) -> "EpisodeTrajectory":
        decision_steps = tuple(step.decision_step for step in self.steps)
        if decision_steps != tuple(range(len(self.steps))):
            raise ValueError(
                "trajectory decision steps must be contiguous and start at zero"
            )
        physics_steps = tuple(step.physics_step for step in self.steps)
        if any(
            second < first
            for first, second in zip(
                physics_steps,
                physics_steps[1:],
            )
        ):
            raise ValueError("trajectory physics steps must be non-decreasing")
        return self


class ConditionResult(StrictModel):
    """Temporal support for one inline milestone condition."""

    condition_id: Identifier = Field(description="Condition represented here.")
    value: bool = Field(description="Whether the condition was ever true.")
    true_steps: Tuple[int, ...] = Field(
        default=(),
        description="Decision steps at which the raw condition was true.",
    )


MilestoneCausalStatus = Literal["passed", "failed", "blocked", "not_reached"]


class MilestoneResult(StrictModel):
    """Event and reaction evidence for one causal graph node."""

    milestone_id: Identifier = Field(description="Milestone represented here.")
    event_occurred: bool = Field(description="Whether the node event occurred.")
    event_step: int | None = Field(
        default=None,
        ge=0,
        description="Decision step where the event occurred.",
    )
    reaction_observed: bool = Field(
        description="Whether the required Agent reaction followed the event.",
    )
    reaction_step: int | None = Field(
        default=None,
        ge=0,
        description="Decision step carrying the scored Agent reaction.",
    )
    local_pass: bool | None = Field(
        default=None,
        description=(
            "Required-completion verdict, independent of semantic quality. "
            "Valid plan/inspection submissions pass; reports require complete "
            "correctness. None means the stage was not reached."
        ),
    )
    causal_status: MilestoneCausalStatus = Field(
        description=(
            "Required-completion result after causal prerequisites. Parent "
            "event completion, not its quality score, admits descendants."
        ),
    )
    blocked_by: Tuple[Identifier, ...] = Field(
        default=(),
        description="Direct parent milestones that prevented this node from passing.",
    )
    earned_point: float = Field(
        ge=0.0,
        le=1.0,
        strict=True,
        allow_inf_nan=False,
        description=(
            "Eligible semantic quality or binary execution score. Zero when "
            "blocked or no reaction exists; partial reports can earn credit "
            "while failing the complete-correctness condition."
        ),
    )
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="Compact event-evaluator evidence retained for audit.",
    )


class TaskProgressResult(StrictModel):
    """Normalized semantic progress along one authored task trajectory."""

    scoring_version: Literal["tarsphere.scoring.v1"] = Field(
        description="Explicit quality-scoring contract; archived binary results require rescoring.",
    )
    success: bool = Field(
        description=(
            "Whether the active final task and necessary execution/interaction "
            "conditions were completed without terminal object loss or fall; "
            "initial plan and inspection quality need not be perfect."
        ),
    )
    completion_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Sum of eligible quality/execution scores divided by the stage count.",
    )
    earned_points: float = Field(
        ge=0.0,
        strict=True,
        allow_inf_nan=False,
        description="Sum of eligible stage scores, including partial semantic credit.",
    )
    total_points: int = Field(
        gt=0,
        description="Number of equally valued milestones in this episode.",
    )
    milestones: Tuple[MilestoneResult, ...] = Field(
        description="Only scored task outcomes; non-scoring events never enter the denominator.",
    )
    events: Tuple["CausalEventResult", ...] = Field(
        default=(),
        description="Non-scoring event occurrence and completion evidence.",
    )

    @model_validator(mode="after")
    def validate_score_accounting(self) -> "TaskProgressResult":
        """Reject internally inconsistent or hand-edited official scores."""

        if self.total_points != len(self.milestones):
            raise ValueError("total_points must equal the scored milestone count")
        milestone_ids = tuple(row.milestone_id for row in self.milestones)
        if len(milestone_ids) != len(set(milestone_ids)):
            raise ValueError("scored milestone ids must be unique")
        for row in self.milestones:
            eligible = (
                row.event_occurred
                and row.reaction_observed
                and not row.blocked_by
                and row.causal_status not in {"blocked", "not_reached"}
            )
            if not eligible and row.earned_point != 0:
                raise ValueError("ineligible milestones cannot earn progress")
            if row.causal_status == "passed" and (
                not eligible or row.local_pass is not True
            ):
                raise ValueError(
                    "passed milestones require eligible evidence and local_pass"
                )
        earned_points = sum(row.earned_point for row in self.milestones)
        if not math.isclose(
            self.earned_points, earned_points, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("earned_points must equal the milestone point sum")
        expected_rate = earned_points / self.total_points
        if not math.isclose(
            self.completion_rate,
            expected_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("completion_rate must equal earned_points / total_points")
        if self.success and any(
            row.causal_status != "passed" for row in self.milestones
        ):
            raise ValueError("success requires every task-completion condition")
        return self


class CausalEventResult(StrictModel):
    """Physical/public completion evidence with no score or earned point."""

    milestone_id: Identifier = Field(
        description="Authored non-scoring node owning this evidence."
    )
    event_occurred: bool = Field(
        description="Whether the public or physical event actually occurred."
    )
    event_step: int | None = Field(
        default=None, description="Decision step of the first event occurrence."
    )
    completion_observed: bool = Field(
        description="Whether deterministic completion evidence was present in the trajectory."
    )
    completion_step: int | None = Field(
        default=None, description="First decision step satisfying completion."
    )
    causal_status: MilestoneCausalStatus = Field(
        description="Topological completion result after direct parent verdicts are applied."
    )
    blocked_by: Tuple[Identifier, ...] = Field(
        default=(),
        description="Direct parent milestones that prevented this event from completing.",
    )
    evidence: dict[str, Any] = Field(
        default_factory=dict,
        description="Private detector evidence; never an earned score.",
    )


class FailureEventResult(StrictModel):
    """Auditable offline verdict for one frozen failure type."""

    failure_event: FailureEvent = Field(description="Frozen failure event name.")
    capability: CapabilityName = Field(
        description="Capability column owning this failure in the frozen table.",
    )
    analysis_method: FailureAnalysisMethod = Field(
        description="Offline trajectory mechanism used for this failure type.",
    )
    analysis: NonEmptyText = Field(
        description=(
            "Auditable application of this row rule, including decisive "
            "counterevidence when the event was not observed."
        ),
    )
    occurred: bool = Field(description="Whether the event happened in the interval.")
    recovered: bool = Field(description="Whether later behavior repaired the event.")
    contributed_to_failure: bool = Field(
        description="Whether the unrecovered event materially blocked completion.",
    )
    start_step: int | None = Field(
        default=None, ge=0, description="First supported event step."
    )
    end_step: int | None = Field(
        default=None, ge=0, description="Last supported event step."
    )
    evidence: Tuple[dict[str, Any], ...] = Field(
        default=(),
        description="Minimal simulator or Agent evidence supporting the verdict.",
    )
    caused_by: Tuple[FailureEvent, ...] = Field(
        default=(),
        description=(
            "Earlier failure types that fully explain this downstream "
            "symptom and therefore prevent duplicate attribution."
        ),
    )
    excluded_reason: str | None = Field(
        default=None,
        description="Why an observed candidate is absent from final attribution.",
    )


class FailureAttributionResult(StrictModel):
    """Final failures plus excluded candidates retained for audit."""

    attributed: Tuple[FailureEventResult, ...] = Field(
        description="Unrecovered, contributing, non-duplicated failure events.",
    )
    analyzed: Tuple[FailureEventResult, ...] = Field(
        description=(
            "All 18 table-row verdicts before filtering for a failed episode; "
            "empty only when the episode succeeded."
        ),
    )


class EpisodeEvaluationReport(StrictModel):
    """Official TARSphere report for one uninterrupted rollout."""

    schema_version: Literal["tarsphere.evaluation.v15"] = Field(
        default="tarsphere.evaluation.v15",
        description="Report version for continuous quality and independent task success.",
    )
    episode_id: NonEmptyText = Field(description="Evaluated episode identifier.")
    task_profile: TaskProfile = Field(
        description="Evaluated task-mainline and capability-slice identity.",
    )
    task_progress: TaskProgressResult = Field(description="Semantic progress result.")
    failure_attribution: FailureAttributionResult = Field(
        description="Trajectory-level diagnosis for unsuccessful episodes.",
    )
    decision_steps: int = Field(
        ge=0,
        description="Model decisions consumed, counting each serial action block once.",
    )
    action_steps: int = Field(
        ge=0,
        description=(
            "Completed action-outcome rows, including non-executed policy "
            "rejections; not a model-decision budget."
        ),
    )
    physics_steps: int = Field(
        ge=0, description="Simulator steps consumed by the rollout."
    )
    termination_reason: NonEmptyText = Field(
        description="Why the uninterrupted rollout ended."
    )


class OnlineEpisodeScore(StrictModel):
    """Task score emitted immediately, before failure attribution."""

    schema_version: Literal["tarsphere.online_score.v2"] = "tarsphere.online_score.v2"
    episode_id: NonEmptyText
    task_profile: TaskProfile
    task_progress: TaskProgressResult
    decision_steps: int = Field(ge=0)
    action_steps: int = Field(ge=0)
    physics_steps: int = Field(ge=0)
    termination_reason: NonEmptyText


__all__ = [
    "EpisodeEvaluationReport",
    "OnlineEpisodeScore",
    "EpisodeEvaluationSpec",
    "EpisodeTrajectory",
    "FailureAttributionResult",
    "FailureEvent",
    "FailureEventResult",
    "CapabilityName",
    "FailureAnalysisMethod",
    "TARSPHERE_EVALUATION_VERSION",
    "TARSPHERE_SCORING_VERSION",
    "MilestoneResult",
    "MilestoneEvent",
    "MilestoneReaction",
    "MilestoneEvaluatorName",
    "DECLARATIVE_MILESTONE_EVALUATORS",
    "MilestoneCausalStatus",
    "DetectorName",
    "DETECTOR_NAMES",
    "ConditionResult",
    "SemanticMilestone",
    "TaskProgressResult",
    "MilestoneCondition",
    "MilestoneReference",
    "MilestonePrecondition",
    "PRIMITIVE_REACTION_SAMPLING_PHASE",
    "PublicTaskUpdate",
    "ReactionSamplingPhase",
    "TrajectoryStep",
]


def _validate_detector_arguments(
    detector: DetectorName,
    arguments: dict[str, Any],
) -> None:
    """Reject malformed bindings before an episode reaches the runtime."""

    required = {
        "human_event_completed": {"milestone_id"},
        "agent_action_selected": {"action"},
        "robot_near_human": {"human_id", "distance"},
        "human_visible": {"human_id"},
        "robot_holding_object": {"object_id"},
        "object_supported_by": {"object_id", "destination_object_id", "distance"},
        "handover_accepted": {"human_id", "object_id"},
        "object_dropped": {"object_id"},
    }
    if detector == "visual_targets_observed":
        expected = {"object_ids", "scene_anchor_id"}
    elif detector == "robot_in_place":
        expected = {"position", "distance"}
    elif detector in {"agent_interaction_selected", "information_received"}:
        expected = {"human_id", "intent_id"}
    else:
        expected = required.get(detector, set())
    actual = set(arguments)
    if actual != expected:
        raise ValueError(
            f"{detector} arguments must be exactly {sorted(expected)}; "
            f"got {sorted(actual)}"
        )
    for key in expected - {
        "position",
        "distance",
        "tolerance_m",
        "object_ids",
    }:
        if not str(arguments[key]).strip():
            raise ValueError(f"{detector} arguments.{key} must be non-empty")
    if "object_ids" in arguments:
        object_ids = arguments["object_ids"]
        if not isinstance(object_ids, (list, tuple)) or not object_ids:
            raise ValueError(
                f"{detector} arguments.object_ids must be a non-empty list"
            )
        normalized = [str(value).strip() for value in object_ids]
        if any(not value for value in normalized):
            raise ValueError(
                f"{detector} arguments.object_ids entries must be non-empty"
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                f"{detector} arguments.object_ids must not contain duplicates"
            )
    if "position" in arguments:
        position = arguments["position"]
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            raise ValueError(f"{detector} arguments.position must be exactly [x, y]")
        if not all(math.isfinite(float(value)) for value in position):
            raise ValueError(f"{detector} arguments.position values must be finite")
    if "distance" in arguments:
        value = float(arguments["distance"])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{detector} arguments.distance must be finite and positive"
            )
    if "tolerance_m" in arguments:
        value = float(arguments["tolerance_m"])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{detector} arguments.tolerance_m must be finite and positive"
            )


def _validate_information_report_ground_truth(
    ground_truth: dict[str, Any],
) -> None:
    expected = {"human_id", "answer"}
    actual = set(ground_truth)
    if actual != expected:
        raise ValueError(
            "information_report_score ground_truth fields must be exactly "
            f"{sorted(expected)}; got {sorted(actual)}"
        )
    for field in sorted(expected):
        if not str(ground_truth[field]).strip():
            raise ValueError(
                f"information_report_score ground_truth.{field} must be non-empty"
            )


def _validate_planning_ground_truth(ground_truth: dict[str, Any]) -> None:
    """Validate the event-local reference shared by plan quality and attribution."""

    expected = {"goal", "required_subgoals"}
    actual = set(ground_truth)
    if actual != expected:
        raise ValueError(
            "structured_plan_submitted ground_truth fields must be exactly "
            f"{sorted(expected)}; got {sorted(actual)}"
        )
    if not str(ground_truth["goal"]).strip():
        raise ValueError(
            "structured_plan_submitted ground_truth.goal must be non-empty"
        )
    subgoals = ground_truth["required_subgoals"]
    if (
        not isinstance(subgoals, (list, tuple))
        or not subgoals
        or any(not isinstance(item, str) or not item.strip() for item in subgoals)
    ):
        raise ValueError(
            "structured_plan_submitted ground_truth.required_subgoals must "
            "be a non-empty string list"
        )


def _validate_inspection_ground_truth(ground_truth: dict[str, Any]) -> None:
    """Require a concise authored criterion and its correct inspected result."""

    expected = {"criterion", "answer"}
    actual = set(ground_truth)
    if actual != expected:
        raise ValueError(
            "inspection_submitted ground_truth fields must be exactly "
            f"{sorted(expected)}; got {sorted(actual)}"
        )
    for field in sorted(expected):
        if not str(ground_truth[field]).strip():
            raise ValueError(
                f"inspection_submitted ground_truth.{field} must be non-empty"
            )
