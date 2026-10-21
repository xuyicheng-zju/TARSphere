"""Canonical benchmark task-mainline and capability-slice contract."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskMainline(str, Enum):
    """The physical task outcome that organizes an episode."""

    NAVIGATE_AND_REPORT = "navigate_and_report"
    FETCH_AND_DELIVER = "fetch_and_deliver"


class CapabilitySlice(str, Enum):
    """The one primary capability isolated by an episode."""

    BASE = "base"
    PERCEPTUAL_ROBUSTNESS = "perceptual_robustness"
    DISRUPTION_ADAPTATION = "disruption_adaptation"
    TASK_UPDATE = "task_update"
    HUMAN_COORDINATION = "human_coordination"
    INTENT_UNDERSTANDING = "intent_understanding"


SCORED_OUTCOME_IDS = (
    "task_understanding",
    "onsite_target_observed",
    "onsite_task_completed",
    "recipient_observed",
    "task_completed",
)

# Evaluator-private duration for every call-triggered temporary disruption:
# five canonical 0.5-second waits at 240 Hz. These values are deliberately
# absent from episode profiles and model feedback.
TEMPORARY_UNAVAILABILITY_DURATION_PHYSICS_STEPS = 600
COMPONENT_OUTAGE_DURATION_PHYSICS_STEPS = 600

# PointGoal and frontier failures have navigation fallbacks. A grasp_object
# failure can be bypassed with direct end-effector control; doing so may be
# difficult, but the benchmark contract only requires that it remains possible.
# Communication, handover, and direct body motion are never outage targets.
COMPONENT_OUTAGE_TOOL_ALTERNATIVES = {
    "navigate_to_frontier": frozenset(
        {
            "move_body",
            "navigate_to_pointgoal",
        }
    ),
    "navigate_to_pointgoal": frozenset(
        {
            "move_body",
            "navigate_to_frontier",
        }
    ),
    "grasp_object": frozenset({"move_end_effector"}),
}


COMPONENT_OUTAGE_AFFECTED_OUTCOMES = {
    "navigate_to_frontier": "onsite_target_observed",
    "navigate_to_pointgoal": "onsite_target_observed",
    "grasp_object": "onsite_task_completed",
}


def require_supported_component_outage_tool(value: str) -> str:
    """Validate one outage target against the three-tool contract."""

    tool_name = str(value).strip()
    if tool_name not in COMPONENT_OUTAGE_TOOL_ALTERNATIVES:
        raise ValueError(
            "Component outage requires one supported tool; expected one of "
            f"{sorted(COMPONENT_OUTAGE_TOOL_ALTERNATIVES)}, got "
            f"{tool_name!r}"
        )
    return tool_name


class _TaskProfile(BaseModel):
    """Fields shared by every current benchmark reporting cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mainline: TaskMainline = Field(
        description=(
            "Primary physical task outcome: navigation/inspection without "
            "object transport, or grasp, carry, and obtain Human acceptance "
            "of an offered handover."
        ),
    )


class BaseTaskProfile(_TaskProfile):
    """Minimum complete task without another capability challenge."""

    capability_slice: Literal[CapabilitySlice.BASE] = CapabilitySlice.BASE


class PerceptualRobustnessTaskProfile(_TaskProfile):
    """One controlled observation condition over an unchanged Base task."""

    capability_slice: Literal[CapabilitySlice.PERCEPTUAL_ROBUSTNESS] = (
        CapabilitySlice.PERCEPTUAL_ROBUSTNESS
    )
    subtype: Literal["low_light", "reduced_resolution", "depth_noise"] = Field(
        description=(
            "The sole episode-wide observation intervention: RTX capture "
            "exposure, effective RGB-D spatial detail, or depth measurement "
            "noise. Task, geometry, and five scored outcomes remain unchanged."
        ),
    )


class _DisruptionAdaptationProfile(_TaskProfile):
    """An execution contingency; the five original task outcomes do not change."""

    capability_slice: Literal[CapabilitySlice.DISRUPTION_ADAPTATION] = (
        CapabilitySlice.DISRUPTION_ADAPTATION
    )
    affected_outcome_id: Literal[
        "onsite_target_observed",
        "onsite_task_completed",
        "recipient_observed",
        "task_completed",
    ] = Field(description="Existing task outcome whose execution is disrupted.")


class SpatialBlockingProfile(_DisruptionAdaptationProfile):
    """A Human holds a real choke point until a locally accepted request."""

    subtype: Literal["spatial_blocking"] = "spatial_blocking"
    blocking_human_id: str = Field(min_length=1)
    blocking_milestone_id: str = Field(min_length=1)
    clearance_request_milestone_id: str = Field(min_length=1)
    clear_milestone_id: str = Field(min_length=1)
    passage_request_intent_id: str = Field(min_length=1)


class TemporaryUnavailabilityProfile(_DisruptionAdaptationProfile):
    """One task-relevant Human briefly refuses the first valid interaction."""

    subtype: Literal["temporary_unavailability"] = "temporary_unavailability"
    human_id: str = Field(min_length=1)
    unavailable_response: str = Field(min_length=1)


class ComponentOutageProfile(_DisruptionAdaptationProfile):
    """One of three supported tools fails on its first valid call."""

    subtype: Literal["component_outage"] = "component_outage"
    component_tool_name: str = Field(
        min_length=1,
        description=(
            "Temporarily unavailable PointGoal, frontier, or grasp tool. "
            "Navigation failures have navigation fallbacks; grasp can be "
            "replaced by direct end-effector control."
        ),
    )

    @model_validator(mode="after")
    def validate_navigation_outage(self) -> "ComponentOutageProfile":
        tool_name = require_supported_component_outage_tool(self.component_tool_name)
        expected_outcome = COMPONENT_OUTAGE_AFFECTED_OUTCOMES[tool_name]
        if self.affected_outcome_id != expected_outcome:
            raise ValueError(
                f"{tool_name} component outage must affect {expected_outcome}"
            )
        return self


DisruptionAdaptationTaskProfile = Annotated[
    Union[
        SpatialBlockingProfile, TemporaryUnavailabilityProfile, ComponentOutageProfile
    ],
    Field(discriminator="subtype"),
]


class TaskUpdateTaskProfile(_TaskProfile):
    """A Human revision changes the authoritative task outcome mid-execution."""

    capability_slice: Literal[CapabilitySlice.TASK_UPDATE] = CapabilitySlice.TASK_UPDATE
    subtype: Literal["task_content_update", "delivery_requirement_update"] = Field(
        description="Which still-pending task commitment is revised by delivered speech.",
    )
    revision_milestone_id: str = Field(
        min_length=1,
        description="Milestone whose typed revision event changes the task.",
    )
    obsolete_commitment: str = Field(
        min_length=1,
        description="Previously authoritative commitment made wrong by the revision.",
    )
    revised_commitment: str = Field(
        min_length=1,
        description="Authoritative task commitment required after the revision.",
    )


def coordination_affected_outcomes(subtype: str, pattern: str) -> tuple[str, ...]:
    """Allowed dependent outcomes; early information need not gate observation.

    An enroute allocation may identify the target, whereas an enroute criterion
    only constrains execution. The authored affected outcome selects that meaning.
    """
    information = {
        "enroute_information": ("onsite_target_observed", "onsite_task_completed"),
        "onsite_information": ("onsite_task_completed",),
        "delivery_information": ("recipient_observed",),
    }
    collaboration = {
        "preparatory_collaboration": ("onsite_target_observed",),
        "parallel_collaboration": ("task_completed",),
        "completion_collaboration": ("task_completed",),
    }
    patterns = {
        "information_coordination": information,
        "task_collaboration": collaboration,
    }
    if subtype not in patterns or pattern not in patterns[subtype]:
        raise ValueError("coordination_pattern does not match the HC subtype")
    return patterns[subtype][pattern]


class HumanCoordinationTaskProfile(_TaskProfile):
    """A Human, including the requester, makes a necessary task contribution."""

    capability_slice: Literal[CapabilitySlice.HUMAN_COORDINATION] = (
        CapabilitySlice.HUMAN_COORDINATION
    )
    subtype: Literal["information_coordination", "task_collaboration"] = Field(
        description="An authorized factual contribution, or physically situated symbolic auxiliary work.",
    )
    coordination_pattern: Literal[
        "enroute_information",
        "onsite_information",
        "delivery_information",
        "preparatory_collaboration",
        "parallel_collaboration",
        "completion_collaboration",
    ] = Field(
        description="Causal stage at which the Human contribution is produced and used."
    )
    supporting_human_id: str = Field(
        min_length=1,
        description=(
            "Human providing the necessary contribution; may also be the requester or final recipient."
        ),
    )
    coordination_milestone_id: str = Field(
        min_length=1,
        description="The one required milestone owned by the supporting Human.",
    )
    affected_outcome_id: Literal[
        "onsite_target_observed",
        "onsite_task_completed",
        "recipient_observed",
        "task_completed",
    ] = Field(
        description="Existing task outcome that needs this contribution; independent earlier work remains possible.",
    )
    contribution: str = Field(
        min_length=1,
        description="The non-substitutable contribution required from that Human.",
    )

    @model_validator(mode="after")
    def validate_coordination_pattern(self):
        allowed = coordination_affected_outcomes(
            self.subtype, self.coordination_pattern
        )
        if self.affected_outcome_id not in allowed:
            raise ValueError("affected_outcome_id does not match coordination_pattern")
        return self


class IntentResolutionOption(BaseModel):
    """One plausible interpretation, optionally grounded in a scene entity."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    option_id: str = Field(
        pattern=r"^[a-z][a-z0-9_]*$", description="Local interpretation identifier."
    )
    description: str = Field(
        min_length=1, description="The task interpretation this option represents."
    )
    entity_ref: str | None = Field(
        default=None,
        min_length=1,
        description="Existing episode object, anchor or Human when the interpretation selects an entity.",
    )


class _IntentUnderstandingProfile(_TaskProfile):
    """Private authoring evidence for resolving the paired Base task."""

    capability_slice: Literal[CapabilitySlice.INTENT_UNDERSTANDING] = (
        CapabilitySlice.INTENT_UNDERSTANDING
    )
    subtype: Literal["preference_confirmation", "requirement_clarification"] = Field(
        description="Whether the ambiguity concerns a preference or a task requirement.",
    )
    resolution_options: tuple[IntentResolutionOption, ...] = Field(
        min_length=2,
        description="Genuinely competing task interpretations, checked during semantic review.",
    )
    resolved_option_id: str = Field(
        min_length=1,
        description="Option matching the paired Base outcome; never disclosed as evaluator state.",
    )
    gap_cues: tuple[str, ...] = Field(
        min_length=1,
        description="Public cues that make interpreting the request necessary.",
    )
    affected_outcome_id: Literal["onsite_task_completed", "task_completed"] = Field(
        description="Existing scored outcome whose task interpretation needs resolution; no additional score.",
    )

    @model_validator(mode="after")
    def validate_options(self):
        ids = [option.option_id for option in self.resolution_options]
        if len(ids) != len(set(ids)) or self.resolved_option_id not in ids:
            raise ValueError(
                "IU options must be distinct and include resolved_option_id"
            )
        return self


class ClarificationRequiredProfile(_IntentUnderstandingProfile):
    """The requested outcome requires an actually delivered clarification."""

    resolution_mode: Literal["clarification_required"] = Field(
        description="Missing information must be obtained from the declared Human."
    )
    fact_id: str = Field(
        min_length=1,
        description="Task-critical fact absent from the initial public request.",
    )
    source_human_id: str = Field(
        min_length=1,
        description="Human the Agent must identify as the appropriate source.",
    )
    interaction_milestone_id: str = Field(
        min_length=1,
        description="Non-scoring receipt of the necessary clarification answer.",
    )


class IntentPublicEvidence(BaseModel):
    """Exact public information supporting an inferable interpretation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source: Literal["dialogue_context", "public_profile"] = Field(
        description="Owner of the public evidence."
    )
    owner_id: str = Field(
        min_length=1,
        description="Milestone id for dialogue, Human id for profile evidence.",
    )
    fact_id: str | None = Field(
        default=None,
        description="Human-local public fact id; required only for profile evidence.",
    )

    @model_validator(mode="after")
    def validate_source(self):
        if (self.source == "public_profile") != bool(self.fact_id):
            raise ValueError("Only public_profile evidence requires fact_id")
        return self


class ContextInferableProfile(_IntentUnderstandingProfile):
    """Public context resolves the request without a required question."""

    resolution_mode: Literal["context_inferable"] = Field(
        description="The public evidence suffices to resolve the request."
    )
    public_evidence: tuple[IntentPublicEvidence, ...] = Field(
        min_length=1,
        description="Exact public dialogue or profile facts supporting the interpretation.",
    )
    discriminating_cues: tuple[str, ...] = Field(
        min_length=1,
        description="Semantic justification separating the correct option from alternatives; reviewed, not a proof.",
    )


IntentUnderstandingTaskProfile = Annotated[
    Union[ClarificationRequiredProfile, ContextInferableProfile],
    Field(discriminator="resolution_mode"),
]


TaskProfile = Annotated[
    Union[
        BaseTaskProfile,
        PerceptualRobustnessTaskProfile,
        DisruptionAdaptationTaskProfile,
        TaskUpdateTaskProfile,
        HumanCoordinationTaskProfile,
        IntentUnderstandingTaskProfile,
    ],
    Field(discriminator="capability_slice"),
]


__all__ = [
    "BaseTaskProfile",
    "CapabilitySlice",
    "COMPONENT_OUTAGE_AFFECTED_OUTCOMES",
    "COMPONENT_OUTAGE_TOOL_ALTERNATIVES",
    "DisruptionAdaptationTaskProfile",
    "SpatialBlockingProfile",
    "TemporaryUnavailabilityProfile",
    "ComponentOutageProfile",
    "HumanCoordinationTaskProfile",
    "PerceptualRobustnessTaskProfile",
    "IntentUnderstandingTaskProfile",
    "ClarificationRequiredProfile",
    "ContextInferableProfile",
    "TaskMainline",
    "TaskProfile",
    "TaskUpdateTaskProfile",
    "require_supported_component_outage_tool",
]
