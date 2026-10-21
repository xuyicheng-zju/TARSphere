"""Typed causal-event vocabulary for benchmark episode graphs.

The vocabulary classifies what changes in the public task state.  It is not a
list of prose templates, a difficulty scale, or an actor-count taxonomy.
Two events count as different causal designs only when their state transition
differs; changing names or wording does not create diversity.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class CausalEventRole(str, Enum):
    """Role played by an event in the milestone DAG."""

    TASK_LIFECYCLE = "task_lifecycle"
    INFORMATION = "information"
    AUTHORIZATION = "authorization"
    REVISION = "revision"
    SPATIAL = "spatial"
    AVAILABILITY = "availability"
    COORDINATION = "coordination"
    TRANSFER = "transfer"
    RESULT = "result"


class EventType(str, Enum):
    """Closed set of semantically distinct, currently authorable events."""

    # Task lifecycle.
    TASK_REQUESTED = "task_requested"
    TASK_ADDED = "task_added"
    TASK_WITHDRAWN = "task_withdrawn"
    TASK_PAUSED = "task_paused"
    TASK_RESUMED = "task_resumed"
    TASK_DECLARED_INFEASIBLE = "task_declared_infeasible"
    TASK_COMPLETED_EXTERNALLY = "task_completed_externally"
    ALTERNATIVE_PROPOSED = "alternative_proposed"
    TASK_OWNERSHIP_TRANSFERRED = "task_ownership_transferred"

    # Information state.
    INFORMATION_DISCLOSED = "information_disclosed"
    INFORMATION_CORRECTED = "information_corrected"
    AMBIGUITY_CLARIFIED = "ambiguity_clarified"
    INFORMATION_SOURCE_REDIRECTED = "information_source_redirected"
    CORRECTION_REQUESTED = "correction_requested"

    # Authorization state.
    AUTHORIZATION_REQUIRED = "authorization_required"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    PERMISSION_REVOKED = "permission_revoked"
    AUTHORITY_TRANSFERRED = "authority_transferred"

    # Existing task semantics change.
    TARGET_REVISED = "target_revised"
    DESTINATION_REVISED = "destination_revised"
    CONSTRAINT_ADDED = "constraint_added"
    CONSTRAINT_REMOVED = "constraint_removed"
    PRIORITY_REVISED = "priority_revised"
    CONFLICTING_INSTRUCTION_ISSUED = "conflicting_instruction_issued"
    RECIPIENT_REASSIGNED = "recipient_reassigned"

    # Spatial/public availability transitions supported by Human motion or
    # public updates. Physical world mutation must still be independently
    # realized and certified; this label never fabricates it.
    INTERACTION_ENDPOINT_RELOCATED = "interaction_endpoint_relocated"
    PASSAGE_BLOCKED_BY_HUMAN = "passage_blocked_by_human"
    PASSAGE_CLEARED_BY_HUMAN = "passage_cleared_by_human"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    RESOURCE_AVAILABLE = "resource_available"
    HUMAN_BECAME_UNAVAILABLE = "human_became_unavailable"
    HUMAN_BECAME_AVAILABLE = "human_became_available"

    # Coordination and phase boundaries.
    TASK_PHASE_OPENED = "task_phase_opened"
    INTERACTION_OPPORTUNITY_OPENED = "interaction_opportunity_opened"
    CONTRIBUTION_REQUESTED = "contribution_requested"
    CONTRIBUTION_COMPLETED = "contribution_completed"

    # Transfer protocol.
    HANDOVER_OFFERED = "handover_offered"
    RECIPIENT_READY = "recipient_ready"
    RECIPIENT_NOT_READY = "recipient_not_ready"
    HANDOVER_DECLINED = "handover_declined"

    # Result solicitation/acknowledgement.
    RESULT_REQUESTED = "result_requested"
    COMPLETION_ACKNOWLEDGED = "completion_acknowledged"


@dataclass(frozen=True)
class EventTypeSpec:
    """Semantic contract for one selectable event kind."""

    event_type: EventType
    role: CausalEventRole
    transition: str
    authoring_rule: str


def _spec(
    event_type: EventType,
    role: CausalEventRole,
    transition: str,
    authoring_rule: str,
) -> EventTypeSpec:
    return EventTypeSpec(
        event_type=event_type,
        role=role,
        transition=transition,
        authoring_rule=authoring_rule,
    )


_EVENT_SPECS = (
    _spec(
        EventType.TASK_REQUESTED,
        CausalEventRole.TASK_LIFECYCLE,
        "no task -> one public task line",
        "A Human publicly assigns a previously absent task objective.",
    ),
    _spec(
        EventType.TASK_ADDED,
        CausalEventRole.TASK_LIFECYCLE,
        "active task set -> active task set plus an independent task line",
        "The added objective must have its own target and closure.",
    ),
    _spec(
        EventType.TASK_WITHDRAWN,
        CausalEventRole.TASK_LIFECYCLE,
        "active task line -> withdrawn task line",
        "The withdrawn line must previously have been authoritative.",
    ),
    _spec(
        EventType.TASK_PAUSED,
        CausalEventRole.TASK_LIFECYCLE,
        "active task line -> temporarily inactive task line",
        "The pause must change which action is currently valid.",
    ),
    _spec(
        EventType.TASK_RESUMED,
        CausalEventRole.TASK_LIFECYCLE,
        "paused task line -> active task line",
        "A previously public pause must exist and the resumed task must remain feasible.",
    ),
    _spec(
        EventType.TASK_DECLARED_INFEASIBLE,
        CausalEventRole.TASK_LIFECYCLE,
        "apparently feasible task -> publicly known infeasible task",
        "Scene evidence must support infeasibility; the Agent must refuse, redirect, or report it.",
    ),
    _spec(
        EventType.TASK_COMPLETED_EXTERNALLY,
        CausalEventRole.TASK_LIFECYCLE,
        "pending task line -> already completed by another actor",
        "The external completion must be observable or authoritatively disclosed.",
    ),
    _spec(
        EventType.ALTERNATIVE_PROPOSED,
        CausalEventRole.TASK_LIFECYCLE,
        "blocked or rejected task -> explicit feasible alternative",
        "Both the original limitation and replacement objective must be public.",
    ),
    _spec(
        EventType.TASK_OWNERSHIP_TRANSFERRED,
        CausalEventRole.TASK_LIFECYCLE,
        "current task owner -> different task owner for the same obligation",
        "Both Humans must be task-grounded and the new owner must accept responsibility.",
    ),
    _spec(
        EventType.INFORMATION_DISCLOSED,
        CausalEventRole.INFORMATION,
        "unknown task-relevant fact -> known fact",
        "The fact must be unavailable from the initial request and affect a later reaction.",
    ),
    _spec(
        EventType.INFORMATION_CORRECTED,
        CausalEventRole.INFORMATION,
        "known task-relevant fact -> corrected fact",
        "Both the obsolete value and authoritative corrected value must be explicit.",
    ),
    _spec(
        EventType.AMBIGUITY_CLARIFIED,
        CausalEventRole.INFORMATION,
        "multiple valid referents -> one grounded referent",
        "The world must contain the competing referents; wording alone is insufficient.",
    ),
    _spec(
        EventType.INFORMATION_SOURCE_REDIRECTED,
        CausalEventRole.INFORMATION,
        "current source lacks required fact -> another grounded source identified",
        "The first source must genuinely lack the fact and the named source must be reachable.",
    ),
    _spec(
        EventType.CORRECTION_REQUESTED,
        CausalEventRole.INFORMATION,
        "accepted result -> result explicitly challenged for re-verification",
        "The request must identify a checkable discrepancy rather than demand repetition.",
    ),
    _spec(
        EventType.AUTHORIZATION_REQUIRED,
        CausalEventRole.AUTHORIZATION,
        "unrestricted-looking action -> action requiring permission",
        "The episode must identify the controlled action and the authority holder.",
    ),
    _spec(
        EventType.PERMISSION_GRANTED,
        CausalEventRole.AUTHORIZATION,
        "permission pending -> permitted",
        "Permission must be granted by the task-grounded authority.",
    ),
    _spec(
        EventType.PERMISSION_DENIED,
        CausalEventRole.AUTHORIZATION,
        "permission pending -> prohibited",
        "Denial must change the valid plan or terminal report.",
    ),
    _spec(
        EventType.PERMISSION_REVOKED,
        CausalEventRole.AUTHORIZATION,
        "permitted action -> prohibited action",
        "A prior grant must exist and continuing the old commitment must become invalid.",
    ),
    _spec(
        EventType.AUTHORITY_TRANSFERRED,
        CausalEventRole.AUTHORIZATION,
        "current permission authority -> different permission authority",
        "The transfer must identify the controlled action and both grounded authorities.",
    ),
    _spec(
        EventType.TARGET_REVISED,
        CausalEventRole.REVISION,
        "authoritative target -> different authoritative target",
        "The old and new targets must both be explicit and the old commitment becomes wrong.",
    ),
    _spec(
        EventType.DESTINATION_REVISED,
        CausalEventRole.REVISION,
        "authoritative destination -> different authoritative destination",
        "The old and new destinations must both be explicit and spatially distinct.",
    ),
    _spec(
        EventType.CONSTRAINT_ADDED,
        CausalEventRole.REVISION,
        "valid plan set -> stricter valid plan set",
        "The new public constraint must exclude at least one formerly valid plan.",
    ),
    _spec(
        EventType.CONSTRAINT_REMOVED,
        CausalEventRole.REVISION,
        "restricted plan set -> expanded valid plan set",
        "The removed constraint must previously have been authoritative.",
    ),
    _spec(
        EventType.PRIORITY_REVISED,
        CausalEventRole.REVISION,
        "task ordering or priority -> different ordering or priority",
        "At least two active obligations must make the revision consequential.",
    ),
    _spec(
        EventType.CONFLICTING_INSTRUCTION_ISSUED,
        CausalEventRole.REVISION,
        "consistent public instructions -> explicit conflict",
        "A public authority or recency rule must make the conflict resolvable.",
    ),
    _spec(
        EventType.RECIPIENT_REASSIGNED,
        CausalEventRole.REVISION,
        "authoritative report or handover recipient -> different recipient",
        "The old and new named recipients must be distinct and task-grounded.",
    ),
    _spec(
        EventType.INTERACTION_ENDPOINT_RELOCATED,
        CausalEventRole.SPATIAL,
        "Human interaction endpoint -> different visible physical endpoint",
        "A real Human goto must complete; a textual location change is insufficient.",
    ),
    _spec(
        EventType.PASSAGE_BLOCKED_BY_HUMAN,
        CausalEventRole.SPATIAL,
        "traversable passage -> Human-occupied passage",
        "A real Human goto and occupancy evidence must prove the changed route consequence.",
    ),
    _spec(
        EventType.PASSAGE_CLEARED_BY_HUMAN,
        CausalEventRole.SPATIAL,
        "Human-occupied passage -> traversable passage",
        "A real Human goto must clear the body disk from the certified passage.",
    ),
    _spec(
        EventType.RESOURCE_UNAVAILABLE,
        CausalEventRole.AVAILABILITY,
        "usable task resource -> unavailable resource",
        "The loss must be publicly observable or announced and alter a later reaction.",
    ),
    _spec(
        EventType.RESOURCE_AVAILABLE,
        CausalEventRole.AVAILABILITY,
        "unavailable task resource -> usable resource",
        "The availability transition must open a previously invalid action.",
    ),
    _spec(
        EventType.HUMAN_BECAME_UNAVAILABLE,
        CausalEventRole.AVAILABILITY,
        "reachable task-relevant Human -> temporarily unreachable Human",
        "A real Human relocation or explicit departure must invalidate the old interaction endpoint.",
    ),
    _spec(
        EventType.HUMAN_BECAME_AVAILABLE,
        CausalEventRole.AVAILABILITY,
        "unreachable task-relevant Human -> reachable Human",
        "A real Human arrival must create a visible, in-range interaction opportunity.",
    ),
    _spec(
        EventType.TASK_PHASE_OPENED,
        CausalEventRole.COORDINATION,
        "phase closed -> next task phase active",
        "The phase boundary must follow detector or predecessor evidence, not hidden time.",
    ),
    _spec(
        EventType.INTERACTION_OPPORTUNITY_OPENED,
        CausalEventRole.COORDINATION,
        "required interaction unavailable -> locally available",
        "The relevant Human must be visible and within communication range when used.",
    ),
    _spec(
        EventType.CONTRIBUTION_REQUESTED,
        CausalEventRole.COORDINATION,
        "unrequested contribution -> requested contribution",
        "The contribution must be necessary for a shared terminal goal.",
    ),
    _spec(
        EventType.CONTRIBUTION_COMPLETED,
        CausalEventRole.COORDINATION,
        "required contribution pending -> completed",
        "The completed contribution must supply evidence used downstream.",
    ),
    _spec(
        EventType.HANDOVER_OFFERED,
        CausalEventRole.TRANSFER,
        "held object -> transfer offered",
        "The correct held object and intended recipient must be grounded.",
    ),
    _spec(
        EventType.RECIPIENT_READY,
        CausalEventRole.TRANSFER,
        "recipient not ready -> ready to receive",
        "Readiness must open a previously invalid handover.",
    ),
    _spec(
        EventType.RECIPIENT_NOT_READY,
        CausalEventRole.TRANSFER,
        "recipient ready or assumed ready -> temporarily not ready",
        "The Agent must delay or revise handover rather than force transfer.",
    ),
    _spec(
        EventType.HANDOVER_DECLINED,
        CausalEventRole.TRANSFER,
        "offered transfer -> recipient refuses the transfer",
        "The refusal must require a different recipient, destination, or infeasibility report.",
    ),
    _spec(
        EventType.RESULT_REQUESTED,
        CausalEventRole.RESULT,
        "result available but unreported -> report solicited",
        "The requested result must be grounded in completed upstream evidence.",
    ),
    _spec(
        EventType.COMPLETION_ACKNOWLEDGED,
        CausalEventRole.RESULT,
        "reported result -> Human acknowledgement",
        "Acknowledgement may close interaction but cannot substitute for physical success.",
    ),
)


EVENT_TYPE_CATALOG: Mapping[EventType, EventTypeSpec] = MappingProxyType(
    {spec.event_type: spec for spec in _EVENT_SPECS}
)

EVENT_TYPES_REQUIRING_HUMAN_GOTO = frozenset(
    {
        EventType.INTERACTION_ENDPOINT_RELOCATED,
        EventType.PASSAGE_BLOCKED_BY_HUMAN,
        EventType.PASSAGE_CLEARED_BY_HUMAN,
    }
)

DETECTOR_ONLY_EVENT_TYPES = frozenset(
    {
        EventType.TASK_PHASE_OPENED,
        EventType.INTERACTION_OPPORTUNITY_OPENED,
        # Acceptance is delivered by the interaction router, not a second queued
        # utterance. The node must bind the actual information_received receipt.
        EventType.CONTRIBUTION_REQUESTED,
        EventType.INTERACTION_OPPORTUNITY_OPENED,
    }
)

if len(EVENT_TYPE_CATALOG) != len(EventType):
    raise RuntimeError("causal event vocabulary contains duplicate or missing kinds")


__all__ = [
    "EVENT_TYPE_CATALOG",
    "DETECTOR_ONLY_EVENT_TYPES",
    "EVENT_TYPES_REQUIRING_HUMAN_GOTO",
    "EventType",
    "CausalEventRole",
    "EventTypeSpec",
]
