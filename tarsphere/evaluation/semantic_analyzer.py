"""Constrained offline analysis for semantic TARSphere failure types."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.evaluation.analyzer import (
    FailureAnalysisContext,
    FailureVerdict,
)
from tarsphere.evaluation.failure_events import FailureEventDefinition
from tarsphere.evaluation.judge import JudgeClient
from tarsphere.evaluation.schema import FailureEvent


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SemanticEvidence(_StrictModel):
    """One trajectory observation or behavior supporting a verdict."""

    step: int = Field(ge=0)
    evidence: str = Field(min_length=1)


class SemanticVerdict(_StrictModel):
    """Exact constrained result for one semantic failure-table row."""

    failure_event: FailureEvent
    analysis: str = Field(min_length=1)
    occurred: bool
    recovered: bool
    contributed_to_failure: bool
    start_step: int | None = Field(default=None, ge=0)
    end_step: int | None = Field(default=None, ge=0)
    evidence: Tuple[SemanticEvidence, ...] = ()
    caused_by: Tuple[FailureEvent, ...] = ()

    @model_validator(mode="after")
    def validate_evidence(self) -> "SemanticVerdict":
        if self.occurred:
            if self.start_step is None or self.end_step is None:
                raise ValueError("an observed failure requires a step interval")
            if self.end_step < self.start_step:
                raise ValueError("failure end_step cannot precede start_step")
            if not self.evidence:
                raise ValueError("an observed failure requires evidence")
        elif (
            self.recovered
            or self.contributed_to_failure
            or self.start_step is not None
            or self.end_step is not None
            or self.evidence
            or self.caused_by
        ):
            raise ValueError("an unobserved failure cannot carry a verdict")
        return self


class SemanticAnalysisResponse(_StrictModel):
    """One batch response covering every semantic failure mechanism."""

    verdicts: Tuple[SemanticVerdict, ...]


class ConstrainedSemanticFailureAnalyzer:
    """Apply the frozen semantic rubrics to a completed failed trajectory."""

    SYSTEM_PROMPT = (
        "You are the offline Failure Attribution Analyzer for TARSPHERE. Analyze "
        "one completed failed embodied trajectory after task-progress scoring. "
        "Work chronologically and apply each requested row independently using "
        "its exact semantic_protocol. For each row: (1) locate the earliest "
        "eligible evidence window; (2) test the occurrence_rule against the "
        "listed counterexamples; (3) inspect all later steps for the exact "
        "recovery_rule; (4) decide final-task contribution using the "
        "contribution_rule; and (5) apply the causal_rule. Return exactly one "
        "verdict for every requested failure_event. Cite concrete steps only "
        "when occurred=true. When occurred=false, put decisive counterevidence "
        "only in analysis and return recovered=false, "
        "contributed_to_failure=false, start_step=null, end_step=null, "
        "evidence=[], and caused_by=[]. A "
        "missing or zero-score milestone is an outcome, never by itself a "
        "failure cause. Absence of explicit evidence means occurred=false. "
        "Do not infer hidden intent from model identity, writing style, or an "
        "eventual failure. Use caused_by only when an earlier observed "
        "failure-table row fully explains the downstream symptom; shared "
        "timing or correlation is insufficient. Do not invent events, steps, "
        "evidence, affordances, or Agent-visible knowledge."
    )

    def __init__(self, client: JudgeClient) -> None:
        self.client = client

    def __call__(
        self,
        context: FailureAnalysisContext,
        definitions: tuple[FailureEventDefinition, ...],
    ) -> dict[FailureEvent, FailureVerdict]:
        shared_payload = {
            "milestones": [
                milestone.model_dump(mode="json")
                for milestone in context.spec.milestones
            ],
            "task_progress": context.task_progress.model_dump(mode="json"),
            "trajectory": [
                self._compact_step(step) for step in context.trajectory.steps
            ],
            "response_schema": {
                "verdicts": [
                    {
                        "failure_event": "one requested failure_event",
                        "analysis": (
                            "concise chronological application of this row protocol, "
                            "including decisive counterevidence when occurred=false"
                        ),
                        "occurred": "boolean",
                        "recovered": "boolean",
                        "contributed_to_failure": "boolean",
                        "start_step": "integer or null",
                        "end_step": "integer or null",
                        "evidence": (
                            "[{step: integer, evidence: specific observed output "
                            "or behavior}] when occurred=true; otherwise []"
                        ),
                        "caused_by": ["earlier failure_event"],
                    }
                ],
            },
            "response_invariants": {
                "occurred_false": {
                    "recovered": False,
                    "contributed_to_failure": False,
                    "start_step": None,
                    "end_step": None,
                    "evidence": [],
                    "caused_by": [],
                    "counterevidence_location": "analysis only",
                },
                "occurred_true": (
                    "start_step/end_step and at least one concrete evidence "
                    "item are required"
                ),
            },
        }
        trajectory_steps = {step.decision_step for step in context.trajectory.steps}
        requested = {definition.event for definition in definitions}
        payload = {
            **shared_payload,
            "requested_semantic_analyses": [
                {
                    "failure_event": definition.event,
                    "capability": definition.capability,
                    "definition": definition.definition,
                    "semantic_protocol": asdict(definition.semantic_protocol),
                }
                for definition in definitions
            ],
        }
        response = self.client.complete_model(
            SemanticAnalysisResponse,
            system_prompt=self.SYSTEM_PROMPT,
            payload=payload,
            max_tokens=32768,
        )
        if len(response.verdicts) != len(definitions):
            raise ValueError(
                "semantic analyzer requires exactly one verdict per "
                f"failure row, got {len(response.verdicts)} for "
                f"{len(definitions)} requested rows"
            )

        verdicts: dict[FailureEvent, FailureVerdict] = {}
        for item in response.verdicts:
            if item.failure_event not in requested:
                raise ValueError(
                    f"semantic analyzer returned unrequested row {item.failure_event!r}"
                )
            if item.failure_event in verdicts:
                raise ValueError(
                    f"duplicate semantic verdict for {item.failure_event!r}"
                )
            cited_steps = {
                step for step in (item.start_step, item.end_step) if step is not None
            }
            cited_steps.update(evidence.step for evidence in item.evidence)
            unknown_steps = cited_steps - trajectory_steps
            if unknown_steps:
                raise ValueError(
                    f"semantic verdict {item.failure_event!r} cites trajectory "
                    f"steps that do not exist: {sorted(unknown_steps)}"
                )
            verdicts[item.failure_event] = FailureVerdict(
                occurred=item.occurred,
                recovered=item.recovered,
                contributed_to_failure=item.contributed_to_failure,
                start_step=item.start_step,
                end_step=item.end_step,
                evidence=tuple(
                    {
                        "step": evidence.step,
                        "evidence": evidence.evidence,
                    }
                    for evidence in item.evidence
                ),
                caused_by=item.caused_by,
                analysis=item.analysis,
            )
        missing = requested - set(verdicts)
        if missing:
            raise ValueError(
                f"semantic analyzer omitted requested rows: {sorted(missing)}"
            )
        return verdicts

    @staticmethod
    def _compact_step(step: Any) -> dict:
        """Keep semantic evidence while excluding bulky repeated payloads."""

        agent_state = {
            key: step.agent_state[key]
            for key in (
                "thinking",
                "memory",
                "plan",
                "subgoal",
                "inspection",
            )
            if key in step.agent_state
            and step.agent_state[key] not in (None, "", [], {})
        }
        feedback = {
            key: step.feedback[key]
            for key in (
                "action",
                "call_name",
                "executed",
                "physical_contact",
                "physical_contact_details",
                "robot_message",
                "motion_outcome",
                "body_stability",
                "balance_control",
                "grasp_failure",
                "grasp_reachability",
                "grasp_skill",
                "grasp_contact_details",
                "placed",
            )
            if key in step.feedback and step.feedback[key] not in (None, "", [], {})
        }
        return {
            "decision_step": step.decision_step,
            "physics_step": step.physics_step,
            "observation": {
                key: step.observation[key]
                for key in ("pose", "public_events")
                if key in step.observation
                and step.observation[key] not in (None, "", [], {})
            },
            "agent_state": agent_state,
            "action": step.action,
            "feedback": feedback,
            "condition_values": step.condition_values,
            "reaction_values": step.reaction_values,
            "occurred_events": list(step.occurred_events),
        }


__all__ = [
    "ConstrainedSemanticFailureAnalyzer",
    "SemanticAnalysisResponse",
    "SemanticEvidence",
    "SemanticVerdict",
]
