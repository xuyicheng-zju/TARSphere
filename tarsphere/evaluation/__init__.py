"""TARSphere causal progress and offline failure attribution."""

from tarsphere.evaluation.analyzer import FailureAttributionAnalyzer
from tarsphere.evaluation.engine import evaluate_trajectory
from tarsphere.evaluation.failure_events import (
    CAPABILITY_FAILURE_TABLE,
    FAILURE_EVENT_BY_NAME,
)
from tarsphere.evaluation.event_vocabulary import (
    EVENT_TYPE_CATALOG,
    EventType,
    CausalEventRole,
)
from tarsphere.evaluation.schema import (
    EpisodeEvaluationReport,
    EpisodeEvaluationSpec,
    EpisodeTrajectory,
)

__all__ = [
    "EpisodeEvaluationReport",
    "EpisodeEvaluationSpec",
    "EpisodeTrajectory",
    "EVENT_TYPE_CATALOG",
    "EventType",
    "CausalEventRole",
    "CAPABILITY_FAILURE_TABLE",
    "FAILURE_EVENT_BY_NAME",
    "FailureAttributionAnalyzer",
    "evaluate_trajectory",
]
