"""Frozen observable failure-event vocabulary for TARSPHERE."""

from __future__ import annotations

from dataclasses import dataclass

from tarsphere.evaluation.schema import (
    CapabilityName,
    FailureAnalysisMethod,
    FailureEvent,
)


@dataclass(frozen=True)
class SemanticFailureProtocol:
    """Release-fixed decision protocol for one semantic failure event."""

    evidence_sources: tuple[str, ...]
    occurrence_rule: str
    counterexamples: tuple[str, ...]
    recovery_rule: str
    contribution_rule: str
    causal_rule: str


@dataclass(frozen=True)
class FailureEventDefinition:
    """One row in the frozen capability--failure attribution table."""

    event: FailureEvent
    capability: CapabilityName
    analysis_method: FailureAnalysisMethod
    definition: str
    semantic_protocol: SemanticFailureProtocol | None = None


CAPABILITY_FAILURE_TABLE: tuple[FailureEventDefinition, ...] = (
    FailureEventDefinition(
        "human_misgrounding",
        "perception",
        "deterministic",
        "The task, request, or later behavior is bound to the wrong person.",
    ),
    FailureEventDefinition(
        "target_misgrounding",
        "perception",
        "deterministic",
        "The task target is bound to the wrong object, place, or instance.",
    ),
    FailureEventDefinition(
        "information_omission",
        "perception",
        "semantic",
        "Exposed task-relevant information never enters later task reasoning.",
        SemanticFailureProtocol(
            evidence_sources=(
                "Public Human utterances and task_updates available before the "
                "information was needed.",
                "Later Agent goal, plan, task state, key memory, reasoning, tool "
                "arguments, and physical behavior.",
                "Milestone event/reaction timing showing whether the omitted fact "
                "could materially affect progress.",
            ),
            occurrence_rule=(
                "Identify one concrete, unambiguous, task-relevant fact that was "
                "publicly exposed before it was needed. The later Agent state and "
                "behavior must both fail to incorporate that fact, and the omission "
                "must explain a wrong decision or blocked reaction."
            ),
            counterexamples=(
                "The fact was never publicly exposed, was ambiguous, or arrived "
                "after the relevant decision.",
                "The Agent did not repeat the wording in memory but acted "
                "consistently with the fact.",
                "The episode failed before the fact became relevant.",
            ),
            recovery_rule=(
                "Recovered only if later explicit state or behavior demonstrably "
                "incorporates the omitted fact and repairs its task consequence."
            ),
            contribution_rule=(
                "Contributing only if the omission caused an incorrect target, "
                "constraint, interaction, plan, or action that materially blocked "
                "terminal completion."
            ),
            causal_rule=(
                "When this omission fully explains a later wrong goal or obsolete "
                "behavior, those downstream rows should cite information_omission "
                "in caused_by rather than duplicate the root cause."
            ),
        ),
    ),
    FailureEventDefinition(
        "goal_misalignment",
        "planning",
        "semantic",
        "The maintained goal, target, or constraint conflicts with sufficient information.",
        SemanticFailureProtocol(
            evidence_sources=(
                "The Agent current goal, target, constraints, plan, and commitments "
                "at each decision step.",
                "All public task information and milestone ground truth available "
                "by that step.",
                "Actions showing which maintained goal actually controlled behavior.",
            ),
            occurrence_rule=(
                "After sufficient information is available, the Agent explicitly "
                "maintains or acts toward a goal, target, recipient, or constraint "
                "that conflicts with the valid task state."
            ),
            counterexamples=(
                "The goal is correct but execution fails.",
                "The goal text is incomplete while the ordered plan and actions "
                "remain task-aligned.",
                "The Agent lacks the necessary information because it was never "
                "exposed or was genuinely ambiguous.",
            ),
            recovery_rule=(
                "Recovered only after the Agent explicitly replaces the wrong goal "
                "and subsequent behavior follows the corrected goal."
            ),
            contribution_rule=(
                "Contributing only if the wrong goal directs consequential behavior "
                "or prevents a required milestone reaction."
            ),
            causal_rule=(
                "Use an earlier information_omission or change_unawareness as "
                "caused_by only when it fully explains the misaligned goal; do not "
                "hide an independently chosen wrong goal."
            ),
        ),
    ),
    FailureEventDefinition(
        "causal_violation",
        "planning",
        "semantic",
        "The plan violates a required causal dependency or partial order.",
        SemanticFailureProtocol(
            evidence_sources=(
                "The authored milestone DAG, primitive preconditions, and event "
                "occurrence steps.",
                "The Agent ordered subgoals, plan revisions, and executed actions.",
                "Reaction evidence showing whether a prerequisite was actually "
                "completed before its dependent action.",
            ),
            occurrence_rule=(
                "The Agent commits to or executes a dependent action before a "
                "required prerequisite reaction has been completed, or reverses an "
                "explicitly required causal order."
            ),
            counterexamples=(
                "A later milestone is merely missing because the episode ended.",
                "Independent actions are reordered without violating a prerequisite.",
                "The causal order is valid but a controller or skill fails.",
            ),
            recovery_rule=(
                "Recovered only if the prerequisite is subsequently established and "
                "the dependent consequence is redone or otherwise repaired without "
                "leaving an invalid task state."
            ),
            contribution_rule=(
                "Contributing only if the order violation invalidates a reaction, "
                "causes irreversible action, or materially blocks the task graph."
            ),
            causal_rule=(
                "A wrong goal may cause the ordering error; cite goal_misalignment "
                "only when the same evidence completely explains the violation."
            ),
        ),
    ),
    FailureEventDefinition(
        "plan_infeasibility",
        "planning",
        "semantic",
        "The plan cannot be realized by the environment, embodiment, or exposed tools.",
        SemanticFailureProtocol(
            evidence_sources=(
                "The explicit Agent plan and intended subgoals.",
                "The public action schema, embodiment limits, environment facts, and "
                "measured tool feedback available when the plan was maintained.",
                "Repeated attempts showing reliance on the infeasible step.",
            ),
            occurrence_rule=(
                "A necessary plan step contradicts a known embodiment, tool, or "
                "environment constraint, so the plan has no realizable route to its "
                "stated goal even under correct execution."
            ),
            counterexamples=(
                "A feasible plan suffers a controller, navigation, or manipulation "
                "failure.",
                "The Agent is still exploring among plausible alternatives.",
                "Infeasibility is visible only in evaluator-private state that was "
                "not available to the Agent.",
            ),
            recovery_rule=(
                "Recovered when the Agent abandons the impossible dependency, adopts "
                "a realizable plan, and acts consistently with it."
            ),
            contribution_rule=(
                "Contributing only if reliance on the infeasible plan consumes "
                "meaningful progress or prevents terminal completion."
            ),
            causal_rule=(
                "Do not duplicate a pure execution failure. Cite a grounding or goal "
                "error only when it is the complete reason the infeasible step was "
                "selected."
            ),
        ),
    ),
    FailureEventDefinition(
        "navigation_stuck",
        "execution",
        "deterministic",
        "The semantic destination is correct but locomotion stalls or oscillates.",
    ),
    FailureEventDefinition(
        "safety_violation",
        "execution",
        "deterministic",
        "The trajectory contains a robot fall or classified unsafe Human contact.",
    ),
    FailureEventDefinition(
        "manipulation_failure",
        "execution",
        "deterministic",
        "The correct target never reaches a manipulation-ready successful acquisition attempt.",
    ),
    FailureEventDefinition(
        "grasp_instability",
        "execution",
        "deterministic",
        "A valid acquisition attempt fails to establish or retain a stable grasp.",
    ),
    FailureEventDefinition(
        "reactive_shortfall",
        "interaction",
        "deterministic",
        "The Agent fails to answer an interaction initiated by a Human.",
    ),
    FailureEventDefinition(
        "proactivity_deficit",
        "interaction",
        "deterministic",
        "The Agent fails to initiate required clarification, confirmation, permission, or help.",
    ),
    FailureEventDefinition(
        "permission_violation",
        "interaction",
        "deterministic",
        "A restricted action occurs without, or contrary to, required permission.",
    ),
    FailureEventDefinition(
        "coordination_failure",
        "interaction",
        "deterministic",
        "A handover, wait, synchronization, or cross-person dependency is not completed.",
    ),
    FailureEventDefinition(
        "change_unawareness",
        "adaptation",
        "semantic",
        "An exposed change is not recognized as affecting the current task.",
        SemanticFailureProtocol(
            evidence_sources=(
                "Actually delivered change and subsequent plan, memory and actions.",
            ),
            occurrence_rule="A delivered relevant change is ignored and later behavior contradicts it.",
            counterexamples=(
                "No explicit plan update, but subsequent behavior follows the change.",
                "The change was queued but never delivered.",
            ),
            recovery_rule="Later behavior incorporates the change and repairs its consequences.",
            contribution_rule="The ignored change must materially block a task outcome.",
            causal_rule="Do not infer unawareness solely from missing structured output.",
        ),
    ),
    FailureEventDefinition(
        "update_delay",
        "adaptation",
        "semantic",
        "Behavioral adoption of a delivered change is delayed with task consequences.",
        SemanticFailureProtocol(
            evidence_sources=(
                "Delivered change time and subsequent decisions and actions.",
            ),
            occurrence_rule="After recognizing a change, the Agent continues obsolete behavior before adapting.",
            counterexamples=(
                "Immediate behavioral adaptation without a structured plan update.",
                "An already executing physical action finishes before replanning.",
            ),
            recovery_rule="The Agent adopts the new requirement and repairs the delayed behavior.",
            contribution_rule="Delay must cause observable wasted execution or a blocked task outcome.",
            causal_rule="Do not charge ordinary evidence or communication latency as Agent delay.",
        ),
    ),
    FailureEventDefinition(
        "obsolete_commitment",
        "adaptation",
        "semantic",
        "The Agent continues an invalidated goal, plan, subgoal, or behavior.",
        SemanticFailureProtocol(
            evidence_sources=(
                "The exposed task change and the exact goal, plan, subgoal, target, "
                "or promise it invalidates.",
                "Post-change Agent commitments and actions, especially after the "
                "Agent acknowledges the change.",
                "Later revisions showing whether the obsolete commitment was "
                "explicitly abandoned.",
            ),
            occurrence_rule=(
                "After the change is available—and especially after acknowledging "
                "it—the Agent explicitly retains, renews, or acts on a commitment "
                "that the new task state invalidates."
            ),
            counterexamples=(
                "A stale sentence remains in history but no longer controls state "
                "or behavior.",
                "The Agent is completing a necessary safe stop before switching.",
                "The Agent never recognizes the change; analyze change_unawareness "
                "rather than inferring an explicit commitment.",
            ),
            recovery_rule=(
                "Recovered when the obsolete commitment is explicitly replaced and "
                "subsequent behavior follows the updated task state."
            ),
            contribution_rule=(
                "Contributing only if acting on the obsolete commitment causes "
                "wrong progress, loss, or failure to complete the revised task."
            ),
            causal_rule=(
                "Do not duplicate change_unawareness unless independent evidence "
                "shows the Agent both failed to notice and later explicitly "
                "recommitted to the obsolete state."
            ),
        ),
    ),
    FailureEventDefinition(
        "recovery_failure",
        "adaptation",
        "deterministic",
        "Explicit failure feedback is followed by the same ineffective strategy.",
    ),
)

FAILURE_EVENT_BY_NAME = {
    definition.event: definition for definition in CAPABILITY_FAILURE_TABLE
}

if len(FAILURE_EVENT_BY_NAME) != 18:
    raise RuntimeError("TARSphere must define exactly 18 unique failure events")

for definition in CAPABILITY_FAILURE_TABLE:
    if (
        definition.analysis_method == "semantic"
        and definition.semantic_protocol is None
    ):
        raise RuntimeError(
            f"Semantic failure {definition.event!r} has no analysis protocol"
        )
    if (
        definition.analysis_method == "deterministic"
        and definition.semantic_protocol is not None
    ):
        raise RuntimeError(
            f"Deterministic failure {definition.event!r} cannot own a semantic protocol"
        )


def require_failure_event(event: FailureEvent) -> FailureEventDefinition:
    """Return the release-frozen definition of ``event``."""

    try:
        return FAILURE_EVENT_BY_NAME[event]
    except KeyError as error:
        raise KeyError(f"Unknown TARSphere failure event: {event!r}") from error


__all__ = [
    "CAPABILITY_FAILURE_TABLE",
    "FAILURE_EVENT_BY_NAME",
    "FailureEventDefinition",
    "SemanticFailureProtocol",
    "require_failure_event",
]
