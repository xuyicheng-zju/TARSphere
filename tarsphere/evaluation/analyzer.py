"""Offline failure attribution over one completed TARSphere trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from tarsphere.evaluation.failure_events import (
    CAPABILITY_FAILURE_TABLE,
    FailureEventDefinition,
)
from tarsphere.evaluation.schema import (
    ConditionResult,
    EpisodeEvaluationSpec,
    EpisodeTrajectory,
    FailureAttributionResult,
    FailureEvent,
    FailureEventResult,
    TaskProgressResult,
)


@dataclass(frozen=True)
class FailureAnalysisContext:
    """Complete evaluator-private evidence available after one rollout."""

    spec: EpisodeEvaluationSpec
    trajectory: EpisodeTrajectory
    task_progress: TaskProgressResult
    conditions: Mapping[str, ConditionResult]


@dataclass(frozen=True)
class FailureVerdict:
    """One offline mechanism's verdict for its frozen failure type."""

    occurred: bool
    recovered: bool
    contributed_to_failure: bool
    start_step: int | None = None
    end_step: int | None = None
    evidence: tuple[dict[str, Any], ...] = ()
    caused_by: tuple[FailureEvent, ...] = ()
    analysis: str = ""

    def __post_init__(self) -> None:
        if self.occurred:
            if self.start_step is None or self.end_step is None:
                raise ValueError("an observed failure requires a step interval")
            if self.end_step < self.start_step:
                raise ValueError("failure end_step cannot precede start_step")
            if not self.evidence:
                raise ValueError("an observed failure requires evidence")
            return
        if (
            self.recovered
            or self.contributed_to_failure
            or self.start_step is not None
            or self.end_step is not None
            or self.evidence
            or self.caused_by
        ):
            raise ValueError("an unobserved failure cannot carry a verdict")


class DeterministicFailureAnalyzer(Protocol):
    """Analyze one failure type from structured trajectory evidence."""

    def __call__(self, context: FailureAnalysisContext) -> FailureVerdict: ...


class SemanticFailureAnalyzer(Protocol):
    """Analyze every semantic table row in one fixed offline request."""

    def __call__(
        self,
        context: FailureAnalysisContext,
        definitions: tuple[FailureEventDefinition, ...],
    ) -> Mapping[FailureEvent, FailureVerdict]: ...


class FailureAttributionAnalyzer:
    """Run the frozen capability--failure table when completion is not safe."""

    def __init__(
        self,
        *,
        deterministic_analyzers: Mapping[FailureEvent, DeterministicFailureAnalyzer],
        semantic_analyzer: SemanticFailureAnalyzer,
    ) -> None:
        expected = {
            item.event
            for item in CAPABILITY_FAILURE_TABLE
            if item.analysis_method == "deterministic"
        }
        provided = set(deterministic_analyzers)
        if provided != expected:
            raise ValueError(
                "deterministic failure analyzer registry differs from the "
                f"frozen table; missing={sorted(expected - provided)}, "
                f"unexpected={sorted(provided - expected)}"
            )
        self._deterministic = dict(deterministic_analyzers)
        self._semantic = semantic_analyzer

    def analyze(
        self,
        context: FailureAnalysisContext,
    ) -> FailureAttributionResult:
        """Return one audited verdict per row for an incomplete or unsafe rollout."""

        safety_verdict = self._deterministic["safety_violation"](context)
        if context.task_progress.success and not safety_verdict.occurred:
            return FailureAttributionResult(attributed=(), analyzed=())

        semantic_definitions = tuple(
            item
            for item in CAPABILITY_FAILURE_TABLE
            if item.analysis_method == "semantic"
        )
        semantic_verdicts = dict(
            self._semantic(
                context,
                semantic_definitions,
            )
        )
        expected_semantic = {item.event for item in semantic_definitions}
        if set(semantic_verdicts) != expected_semantic:
            raise ValueError(
                "semantic failure analyzer result differs from the frozen "
                f"table; missing={sorted(expected_semantic - set(semantic_verdicts))}, "
                f"unexpected={sorted(set(semantic_verdicts) - expected_semantic)}"
            )

        analyzed: list[FailureEventResult] = []
        for definition in CAPABILITY_FAILURE_TABLE:
            verdict = (
                safety_verdict
                if definition.event == "safety_violation"
                else self._deterministic[definition.event](context)
                if definition.analysis_method == "deterministic"
                else semantic_verdicts[definition.event]
            )
            if definition.event in verdict.caused_by:
                raise ValueError(f"{definition.event!r} cannot cause itself")
            analyzed.append(
                FailureEventResult(
                    failure_event=definition.event,
                    capability=definition.capability,
                    analysis_method=definition.analysis_method,
                    occurred=verdict.occurred,
                    recovered=verdict.recovered,
                    contributed_to_failure=verdict.contributed_to_failure,
                    start_step=verdict.start_step,
                    end_step=verdict.end_step,
                    evidence=verdict.evidence,
                    caused_by=verdict.caused_by,
                    analysis=(
                        verdict.analysis.strip()
                        or _deterministic_analysis(definition, verdict)
                    ),
                )
            )
        ordered = tuple(analyzed)
        _validate_causal_graph(ordered)
        return _attribute_failures(ordered)


def _attribute_failures(
    analyzed: tuple[FailureEventResult, ...],
) -> FailureAttributionResult:
    qualifying = {
        result.failure_event
        for result in analyzed
        if result.occurred and not result.recovered and result.contributed_to_failure
    }
    attributed: list[FailureEventResult] = []
    audited: list[FailureEventResult] = []
    for result in analyzed:
        reason: str | None = None
        if not result.occurred:
            reason = "event_not_observed"
        elif result.recovered:
            reason = "recovered_before_rollout_end"
        elif not result.contributed_to_failure:
            reason = "did_not_contribute_to_final_failure"
        else:
            causal_sources = tuple(
                event for event in result.caused_by if event in qualifying
            )
            if causal_sources:
                reason = "downstream_symptom_of:" + ",".join(causal_sources)
        reviewed = result.model_copy(update={"excluded_reason": reason})
        audited.append(reviewed)
        if reason is None:
            attributed.append(reviewed)
    return FailureAttributionResult(
        attributed=tuple(attributed),
        analyzed=tuple(audited),
    )


def _validate_causal_graph(
    analyzed: tuple[FailureEventResult, ...],
) -> None:
    results = {result.failure_event: result for result in analyzed}
    graph = {result.failure_event: tuple(result.caused_by) for result in analyzed}
    visiting: set[FailureEvent] = set()
    visited: set[FailureEvent] = set()

    def visit(event: FailureEvent) -> None:
        if event in visiting:
            raise ValueError("failure attribution causal graph contains a cycle")
        if event in visited:
            return
        visiting.add(event)
        for cause in graph[event]:
            visit(cause)
        visiting.remove(event)
        visited.add(event)

    for event in graph:
        visit(event)
    for result in analyzed:
        for cause_name in result.caused_by:
            cause = results[cause_name]
            if not cause.occurred:
                raise ValueError(
                    f"{result.failure_event!r} cites unobserved cause {cause_name!r}"
                )
            if (
                result.start_step is not None
                and cause.start_step is not None
                and cause.start_step > result.start_step
            ):
                raise ValueError(
                    f"{result.failure_event!r} cites later cause {cause_name!r}"
                )


def _deterministic_analysis(
    definition: FailureEventDefinition,
    verdict: FailureVerdict,
) -> str:
    if definition.analysis_method != "deterministic":
        raise ValueError(f"Semantic verdict {definition.event!r} returned no analysis")
    if verdict.occurred:
        return (
            f"Structured trajectory evidence satisfied the deterministic "
            f"{definition.event} rule; recovery and contribution were resolved "
            "from later structured state."
        )
    return (
        f"No structured trajectory evidence satisfied the deterministic "
        f"{definition.event} rule."
    )


__all__ = [
    "DeterministicFailureAnalyzer",
    "FailureAnalysisContext",
    "FailureAttributionAnalyzer",
    "FailureVerdict",
    "SemanticFailureAnalyzer",
]
