"""The single official trajectory evaluation path used by every runtime."""

from __future__ import annotations

from typing import Any

from tarsphere.evaluation.analyzer import FailureAttributionAnalyzer
from tarsphere.evaluation.deterministic_analyzers import (
    deterministic_failure_analyzers,
)
from tarsphere.evaluation.engine import evaluate_trajectory, score_trajectory
from tarsphere.evaluation.judge import JudgeClient
from tarsphere.evaluation.progress import (
    InitialPlanJudge,
    InspectionJudge,
    InformationReportJudge,
    milestone_evaluators,
)
from tarsphere.evaluation.schema import (
    EpisodeEvaluationReport,
    EpisodeTrajectory,
    OnlineEpisodeScore,
)
from tarsphere.evaluation.semantic_analyzer import ConstrainedSemanticFailureAnalyzer


# One model output (including a rejected output) consumes one decision;
# its serial action suffix and internal transport retries do not consume more.
FORMAL_EVALUATION_MAX_DECISION_STEPS = 100


def validate_formal_trajectory_budget(trajectory: EpisodeTrajectory) -> None:
    """Reject missing decision accounting instead of guessing from row count."""

    if any(
        "policy_inference" not in step.model_fields_set for step in trajectory.steps
    ):
        raise ValueError(
            "Formal trajectory is missing explicit policy_inference evidence"
        )
    if not trajectory.steps[0].policy_inference:
        raise ValueError("Formal trajectory must begin with a model decision")
    if not 1 <= trajectory.decision_steps <= FORMAL_EVALUATION_MAX_DECISION_STEPS:
        raise ValueError("Formal trajectory exceeds the 100-model-decision budget")
    if (
        trajectory.termination_reason == "decision_budget_exhausted"
        and trajectory.decision_steps != FORMAL_EVALUATION_MAX_DECISION_STEPS
    ):
        raise ValueError(
            "Formal decision budget was exhausted before 100 model decisions"
        )


def validate_formal_episode_runtime(episode: Any) -> None:
    """Reject episode-authored limits that would change the formal protocol."""

    if episode.runtime.max_decision_steps is not None:
        raise ValueError(
            f"Formal episode {episode.episode_id!r} must leave "
            "runtime.max_decision_steps null; formal evaluation applies the "
            f"fixed {FORMAL_EVALUATION_MAX_DECISION_STEPS}-decision limit."
        )
    if episode.runtime.max_physics_steps is not None:
        raise ValueError(
            f"Formal episode {episode.episode_id!r} must leave "
            "runtime.max_physics_steps null; formal evaluation has no "
            "episode-wide physics-step limit."
        )


class EvaluationRunError(RuntimeError):
    """An evaluator failure with the Judge usage spent before it failed."""

    def __init__(self, message: str, *, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = dict(usage)


def evaluate_official_trajectory(
    episode: Any,
    trajectory: EpisodeTrajectory,
) -> tuple[EpisodeEvaluationReport, dict[str, Any]]:
    """Evaluate one frozen trajectory without re-running physical actions."""

    if episode.evaluation is None:
        raise ValueError("official TARSphere rollout has no evaluation contract")
    validate_formal_trajectory_budget(trajectory)
    judge_client = JudgeClient.from_environment()
    needs_information_report_judge = any(
        milestone.outcome.evaluator == "information_report_score"
        for milestone in episode.evaluation.milestones
    )
    information_report_judge = (
        InformationReportJudge(judge_client) if needs_information_report_judge else None
    )
    try:
        report = evaluate_trajectory(
            episode.evaluation,
            trajectory,
            failure_analyzer=FailureAttributionAnalyzer(
                deterministic_analyzers=deterministic_failure_analyzers(),
                semantic_analyzer=ConstrainedSemanticFailureAnalyzer(judge_client),
            ),
            milestone_evaluators=milestone_evaluators(
                initial_plan_judge=InitialPlanJudge(judge_client),
                inspection_judge=InspectionJudge(judge_client),
                information_report_judge=information_report_judge,
            ),
        )
    except Exception as exc:
        raise EvaluationRunError(str(exc), usage=judge_client.usage_report()) from exc
    return report, judge_client.usage_report()


def score_official_trajectory(
    episode: Any,
    trajectory: EpisodeTrajectory,
    *,
    judge_client: JudgeClient | None = None,
) -> tuple[OnlineEpisodeScore, dict[str, Any]]:
    """Compute the official online score without failure attribution."""

    if episode.evaluation is None:
        raise ValueError("official TARSphere rollout has no evaluation contract")
    validate_formal_trajectory_budget(trajectory)
    judge_client = (
        judge_client if judge_client is not None else JudgeClient.from_environment()
    )
    needs_information_report_judge = any(
        milestone.outcome.evaluator == "information_report_score"
        for milestone in episode.evaluation.milestones
    )
    information_report_judge = (
        InformationReportJudge(judge_client) if needs_information_report_judge else None
    )
    try:
        progress = score_trajectory(
            episode.evaluation,
            trajectory,
            milestone_evaluators=milestone_evaluators(
                initial_plan_judge=InitialPlanJudge(judge_client),
                inspection_judge=InspectionJudge(judge_client),
                information_report_judge=information_report_judge,
            ),
        )
        score = OnlineEpisodeScore(
            episode_id=trajectory.episode_id,
            task_profile=trajectory.task_profile,
            task_progress=progress,
            decision_steps=trajectory.decision_steps,
            action_steps=len(trajectory.steps),
            physics_steps=trajectory.steps[-1].physics_step,
            termination_reason=trajectory.termination_reason,
        )
    except Exception as exc:
        raise EvaluationRunError(str(exc), usage=judge_client.usage_report()) from exc
    return score, judge_client.usage_report()


def apply_official_evaluation_outcome(
    result: dict[str, Any],
    report: EpisodeEvaluationReport | dict[str, Any],
) -> None:
    """Publish the official success verdict from one validated report."""

    payload = (
        report.model_dump(mode="json")
        if isinstance(report, EpisodeEvaluationReport)
        else report
    )
    task_progress = payload.get("task_progress")
    if not isinstance(task_progress, dict):
        raise ValueError("evaluation report is missing task_progress")
    evaluated_success = task_progress.get("success")
    if not isinstance(evaluated_success, bool):
        raise TypeError("evaluation task_progress.success must be a boolean")
    result["success"] = bool(evaluated_success and "failure" not in result)
    result["outcome_status"] = "evaluated"


__all__ = [
    "EvaluationRunError",
    "FORMAL_EVALUATION_MAX_DECISION_STEPS",
    "apply_official_evaluation_outcome",
    "evaluate_official_trajectory",
    "score_official_trajectory",
    "validate_formal_episode_runtime",
    "validate_formal_trajectory_budget",
]
