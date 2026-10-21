"""Non-blocking online scoring of already-frozen physical trajectories."""

from __future__ import annotations

import json
import hashlib
import os
import threading
from pathlib import Path
from typing import Any

from tarsphere.api_usage import aggregate_model_calls, merge_usage_reports
from tarsphere.bench.schema import TARSphereEpisode
from tarsphere.evaluation.official import (
    EvaluationRunError,
    score_official_trajectory,
)
from tarsphere.evaluation.schema import (
    EpisodeEvaluationReport,
    EpisodeTrajectory,
    FailureAttributionResult,
    TARSPHERE_SCORING_VERSION,
)


def _atomic_json(path: Path, value: Any) -> None:
    pending = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.pending"
    )
    pending.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def archive_previous_evaluation(path: Path) -> Path | None:
    """Preserve exact prior scored-result bytes before a versioned rescore."""

    raw = path.read_bytes()
    result = json.loads(raw)
    if not (
        isinstance(result.get("online_score"), dict)
        or isinstance(
            (result.get("evaluation_report") or {}).get("task_progress"), dict
        )
    ):
        return None
    archive = (
        path.parent
        / ".evaluation_history"
        / (f"{path.stem}-{hashlib.sha256(raw).hexdigest()}.json")
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        with archive.open("xb") as handle:
            handle.write(raw)
    except FileExistsError:
        if archive.read_bytes() != raw:
            raise ValueError(f"Prior-score archive hash mismatch: {archive}")
    return archive


def has_archived_binary_score(result: dict[str, Any]) -> bool:
    """Recognize the explicit archived protocol without reinterpreting its scores."""

    return any(
        isinstance(score := result.get(key), dict)
        and score.get("schema_version") == version
        and isinstance(score.get("task_progress"), dict)
        and "scoring_version" not in score["task_progress"]
        for key, version in (
            ("online_score", "tarsphere.online_score.v1"),
            ("evaluation_report", "tarsphere.evaluation.v14"),
        )
    )


def _attempt_identity(result: dict[str, Any]) -> tuple[Any, ...]:
    request_id = result.get("request_id")
    if request_id is not None:
        return ("request", request_id)
    return (
        "artifact",
        result.get("evaluation_identity"),
        result.get("physical_attempt"),
        result.get("episode_artifact_sha256"),
        result.get("rollout_artifact_sha256"),
        result.get("model_output_trace_sha256"),
    )


def score_saved_result(
    *,
    authoritative_result_path: Path,
    latest_result_path: Path | None = None,
) -> dict[str, Any]:
    """Score one immutable attempt and conditionally refresh its latest view."""

    result = json.loads(authoritative_result_path.read_text(encoding="utf-8"))
    archive = archive_previous_evaluation(authoritative_result_path)
    if archive is not None:
        result["previous_evaluation_archive"] = str(archive)
    identity = _attempt_identity(result)
    result["evaluation_attempts"] = int(result.get("evaluation_attempts") or 0) + 1
    attempt = result["evaluation_attempts"]
    try:
        if result.get("diagnostic_only"):
            raise ValueError(
                "Private certification results are ineligible for official scoring"
            )
        episode = TARSphereEpisode.model_validate_json(
            Path(result["episode_artifact_path"]).read_text(encoding="utf-8")
        )
        trajectory = EpisodeTrajectory.model_validate_json(
            Path(result["evaluation_trajectory_path"]).read_text(encoding="utf-8")
        )
        score, usage = score_official_trajectory(
            episode,
            trajectory,
        )
        result["evaluation_scoring_mode"] = "official"
        result["online_score"] = score.model_dump(mode="json")
        score_succeeded = bool(score.task_progress.success and "failure" not in result)
        result["success"] = score_succeeded
        if score_succeeded:
            report = EpisodeEvaluationReport(
                episode_id=score.episode_id,
                task_profile=score.task_profile,
                task_progress=score.task_progress,
                failure_attribution=FailureAttributionResult(
                    attributed=(),
                    analyzed=(),
                ),
                decision_steps=score.decision_steps,
                action_steps=score.action_steps,
                physics_steps=score.physics_steps,
                termination_reason=score.termination_reason,
            )
            result["evaluation_report"] = report.model_dump(mode="json")
            result["outcome_status"] = "evaluated"
            result["failure_attribution_status"] = "not_required"
        else:
            result.pop("evaluation_report", None)
            result["outcome_status"] = "scored_pending_attribution"
            result["failure_attribution_status"] = "pending"
        result.setdefault("evaluation_attempt_history", []).append(
            {
                "attempt": attempt,
                "scoring_version": TARSPHERE_SCORING_VERSION,
                "status": result["outcome_status"],
                "usage": usage,
            }
        )
        result.setdefault("evaluation_model_usage_attempts", []).append(usage)
        result["evaluation_model_usage"] = merge_usage_reports(
            result["evaluation_model_usage_attempts"]
        )
    except Exception as exc:  # the frozen physical attempt remains reusable
        usage = (
            exc.usage
            if isinstance(exc, EvaluationRunError)
            else aggregate_model_calls([])
        )
        result.pop("online_score", None)
        result["evaluation_report"] = {"error": str(exc)}
        result.setdefault("evaluation_attempt_history", []).append(
            {
                "attempt": attempt,
                "scoring_version": TARSPHERE_SCORING_VERSION,
                "status": "evaluation_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "usage": usage,
            }
        )
        result.setdefault("evaluation_model_usage_attempts", []).append(usage)
        result["evaluation_model_usage"] = merge_usage_reports(
            result["evaluation_model_usage_attempts"]
        )
        result["success"] = False
        result["outcome_status"] = "evaluation_failed"
    result["api_usage"] = merge_usage_reports(
        (
            result.get("agent_model_usage") or aggregate_model_calls([]),
            result.get("human_model_usage") or aggregate_model_calls([]),
            result.get("evaluation_model_usage") or aggregate_model_calls([]),
        )
    )
    _atomic_json(authoritative_result_path, result)
    if (
        latest_result_path is not None
        and latest_result_path != authoritative_result_path
    ):
        try:
            latest = json.loads(latest_result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            latest = None
        if isinstance(latest, dict) and _attempt_identity(latest) == identity:
            _atomic_json(latest_result_path, result)
    return result


__all__ = [
    "score_saved_result",
    "archive_previous_evaluation",
    "has_archived_binary_score",
]
