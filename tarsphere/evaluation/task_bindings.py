"""Event-local task bindings shared by live detectors and offline judges."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from tarsphere.evaluation.schema import (
    EpisodeEvaluationSpec,
    PRIMITIVE_REACTION_SAMPLING_PHASE,
    SemanticMilestone,
    TrajectoryStep,
)


class TaskBindings:
    """Own the current private outcome GT; no evaluator score mutates this state."""

    def __init__(self, spec: EpisodeEvaluationSpec):
        self._spec = spec
        self.reset()

    def reset(self) -> None:
        self.version = 0
        self._outcomes = {
            node.milestone_id: deepcopy(node.outcome.ground_truth)
            for node in self._spec.milestones
        }

    def activate(
        self, replacements: dict[str, dict[str, Any]], *, completed: set[str]
    ) -> None:
        unknown = set(replacements) - set(self._outcomes)
        if unknown:
            raise ValueError(f"Unknown task binding outcomes: {sorted(unknown)}")
        already_observed = set(replacements) & completed
        if already_observed:
            raise ValueError(
                f"Task update would rewrite observed outcomes: {sorted(already_observed)}"
            )
        self._outcomes.update(deepcopy(replacements))
        if replacements:
            self.version += 1

    def truth(self, outcome_id: str) -> dict[str, Any]:
        return deepcopy(self._outcomes[outcome_id])

    def snapshot(self) -> dict[str, Any]:
        return {"version": self.version, "outcomes": deepcopy(self._outcomes)}


def outcome_ground_truth(
    node: SemanticMilestone,
    step: TrajectoryStep,
    *,
    phase: Literal["pre_decision", "post_action"] | None = None,
) -> dict[str, Any]:
    """Select the binding at the evidence instant, not the rollout's final state.

    Static trajectories need no binding snapshots. The trajectory evaluator
    separately requires both snapshots whenever its spec declares replacements.
    """
    snapshots = step.hidden_state.get("task_bindings")
    if snapshots is None:
        return deepcopy(node.outcome.ground_truth)
    if phase is None:
        phase = PRIMITIVE_REACTION_SAMPLING_PHASE.get(
            node.outcome.evaluator, "decision"
        )
        if node.outcome.evaluator == "information_report_score":
            phase = "post_action"
    key = "post_action" if phase == "post_action" else "pre_decision"
    return deepcopy(snapshots[key]["outcomes"][node.milestone_id])


def final_outcome_ground_truth(
    spec: EpisodeEvaluationSpec, outcome_id: str
) -> dict[str, Any]:
    """Authored eventual bindings for route review only, never live evidence."""
    node = next(node for node in spec.milestones if node.milestone_id == outcome_id)
    truth = deepcopy(node.outcome.ground_truth)
    for event in spec.milestones:
        for update in event.event.task_updates:
            if outcome_id in update.outcome_updates:
                truth = deepcopy(update.outcome_updates[outcome_id])
    return truth


def validate_binding_snapshots(
    spec: EpisodeEvaluationSpec, steps: tuple[TrajectoryStep, ...]
) -> None:
    """Rebuild versions from actual speech, rejecting stale or future task GT."""
    if not any(
        update.outcome_updates
        for node in spec.milestones
        for update in node.event.task_updates
    ):
        return
    bindings = TaskBindings(spec)
    nodes = {node.milestone_id: node for node in spec.milestones}
    delivered: set[tuple[str, str]] = set()
    for step in steps:
        snapshots = step.hidden_state.get("task_bindings")
        if not isinstance(snapshots, dict) or set(snapshots) != {
            "pre_decision",
            "post_action",
        }:
            raise ValueError(
                "Dynamic task trajectory requires phase-local task binding snapshots"
            )
        for timing, phase in (
            ("before_agent", "pre_decision"),
            ("after_agent", "post_action"),
        ):
            for speech in step.hidden_state.get("human_events", ()):
                node = nodes.get(speech.get("milestone_id"))
                if (
                    node is None
                    or speech.get("timing") != timing
                    or speech.get("action") not in {"speak", "ask", "report_done"}
                    or not str(speech.get("message") or "").strip()
                ):
                    continue
                source = speech.get("human_id")
                replacements = {}
                for update in node.event.task_updates:
                    key = (node.milestone_id, update.field)
                    if update.source_human_id == source and key not in delivered:
                        replacements.update(update.outcome_updates)
                        delivered.add(key)
                if replacements:
                    bindings.activate(replacements, completed=set())
            if snapshots[phase] != bindings.snapshot():
                raise ValueError(
                    f"Step {step.decision_step} {phase}: bindings differ from delivered task updates"
                )
