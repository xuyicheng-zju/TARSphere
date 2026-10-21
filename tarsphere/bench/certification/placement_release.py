"""Dataset certification view of the shared physical placement verdict."""

from __future__ import annotations

from typing import Any, Literal, Sequence

from tarsphere.evaluation.placement_transition import (
    PlacementTransitionVerdict,
    evaluate_placement_trace,
)
from tarsphere.infra.control.physical_contracts import (
    G1_PLACEMENT_RELATION_TOLERANCE_M,
    G1_PLACEMENT_SETTLE_SAMPLE_COUNT,
    G1_PLACEMENT_SETTLE_MOTION_M,
)


# The Dataset certification report and formal evaluator intentionally expose
# the same immutable verdict type. A second schema would duplicate derived
# state and allow the two gates to disagree.
PlacementReleaseEvidence = PlacementTransitionVerdict


def measure_placement_release(
    steps: Sequence[Any],
    *,
    target_object_id: str,
    destination_object_id: str,
    goal_relation: Literal["on"] = "on",
    required_settle_samples: int = G1_PLACEMENT_SETTLE_SAMPLE_COUNT,
    relation_tolerance_m: float = G1_PLACEMENT_RELATION_TOLERANCE_M,
    motion_tolerance_m: float = G1_PLACEMENT_SETTLE_MOTION_M,
) -> PlacementReleaseEvidence:
    """Measure placement with the same reducer used for benchmark scoring."""

    if goal_relation != "on":
        raise ValueError(f"unsupported placement release relation: {goal_relation!r}")
    return evaluate_placement_trace(
        steps,
        target_object_id=target_object_id,
        destination_object_id=destination_object_id,
        required_settle_samples=required_settle_samples,
        relation_tolerance_m=relation_tolerance_m,
        motion_tolerance_m=motion_tolerance_m,
    )


__all__ = [
    "PlacementReleaseEvidence",
    "measure_placement_release",
]
