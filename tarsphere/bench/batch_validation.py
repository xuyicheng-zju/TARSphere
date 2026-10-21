"""Cross-episode semantic and structural validation for candidate batches."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from tarsphere.bench.difficulty import assign_batch_difficulty
from tarsphere.bench.schema import TARSphereEpisode
from tarsphere.evaluation.event_vocabulary import EVENT_TYPE_CATALOG
from tarsphere.evaluation.schema import MilestoneCondition, MilestoneReference


@dataclass(frozen=True)
class BatchValidationReport:
    """Auditable batch properties that cannot be checked episode-by-episode."""

    episode_count: int
    scene_counts: dict[str, int]
    mainline_counts: dict[str, int]
    capability_slice_counts: dict[str, int]
    reporting_cell_counts: dict[str, int]
    event_kind_counts: dict[str, int]
    event_role_counts: dict[str, int]
    causal_signature_counts: dict[str, int]
    causal_signature_entropy_bits: float
    semantic_review_warnings: tuple[str, ...]


def validate_candidate_batch(
    episodes: Iterable[TARSphereEpisode],
) -> BatchValidationReport:
    """Reject cross-episode contradictions and summarize diversity evidence.

    Causal signature counts are review evidence, not an arbitrary pass
    threshold. Semantic equivalence remains a review question; a hash must not
    pretend to settle it.
    """

    values = tuple(episodes)
    if not values:
        raise ValueError("candidate batch must contain at least one episode")
    ids = [episode.episode_id for episode in values]
    duplicates = sorted(
        episode_id for episode_id, count in Counter(ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"candidate batch has duplicate episode ids: {duplicates}")

    # Difficulty is a Base-task property.  This also proves that every variant
    # has exactly one same-scene, same-mainline Base to inherit from.
    assign_batch_difficulty(values)

    scene_counts = Counter(str(episode.scene.asset_path) for episode in values)
    mainline_counts = Counter(
        episode.evaluation.task_profile.mainline.value
        for episode in values
        if episode.evaluation is not None
    )
    capability_slice_counts = Counter(
        episode.evaluation.task_profile.capability_slice.value
        for episode in values
        if episode.evaluation is not None
    )
    reporting_cell_counts = Counter(
        f"{episode.evaluation.task_profile.mainline.value}/"
        f"{episode.evaluation.task_profile.capability_slice.value}"
        for episode in values
        if episode.evaluation is not None
    )
    signatures = Counter(_causal_signature(episode) for episode in values)
    event_kind_counts = Counter(
        milestone.event.type.value
        for episode in values
        for milestone in episode.evaluation.milestones
        if episode.evaluation is not None
    )
    event_role_counts = Counter(
        EVENT_TYPE_CATALOG[milestone.event.type].role.value
        for episode in values
        for milestone in episode.evaluation.milestones
        if episode.evaluation is not None
    )
    entropy = -sum(
        (count / len(values)) * math.log2(count / len(values))
        for count in signatures.values()
    )
    warnings = tuple(
        warning for episode in values for warning in _semantic_review_warnings(episode)
    )
    return BatchValidationReport(
        episode_count=len(values),
        scene_counts=dict(sorted(scene_counts.items())),
        mainline_counts=dict(sorted(mainline_counts.items())),
        capability_slice_counts=dict(sorted(capability_slice_counts.items())),
        reporting_cell_counts=dict(sorted(reporting_cell_counts.items())),
        event_kind_counts=dict(sorted(event_kind_counts.items())),
        event_role_counts=dict(sorted(event_role_counts.items())),
        causal_signature_counts={
            signature: count for signature, count in sorted(signatures.items())
        },
        causal_signature_entropy_bits=round(entropy, 6),
        semantic_review_warnings=warnings,
    )


def _causal_signature(episode: TARSphereEpisode) -> str:
    evaluation = episode.evaluation
    if evaluation is None:
        raise ValueError(f"{episode.episode_id}: candidate has no evaluation")
    nodes: list[str] = []
    node_indices = {
        milestone.milestone_id: index
        for index, milestone in enumerate(evaluation.milestones)
    }
    for milestone in evaluation.milestones:
        parents = sorted(
            node_indices[item.milestone_ref]
            for item in milestone.preconditions
            if isinstance(item, MilestoneReference)
        )
        detectors = sorted(
            item.detector
            for item in milestone.preconditions
            if isinstance(item, MilestoneCondition)
        )
        event_modes = sorted(
            {
                str(action.get("command") or "")
                for action in milestone.event.human_actions
                if action.get("command")
            }
        )
        nodes.append(
            "|".join(
                (
                    milestone.event.type.value,
                    milestone.outcome.evaluator,
                    ",".join(detectors),
                    ",".join(str(index) for index in parents),
                    ",".join(event_modes),
                    str(bool(milestone.event.task_updates)),
                )
            )
        )
    return " -> ".join(nodes)


def _semantic_review_warnings(episode: TARSphereEpisode) -> tuple[str, ...]:
    """Surface potentially guessable reports without pretending to judge them."""

    evaluation = episode.evaluation
    if evaluation is None:
        return ()
    initial_messages = " ".join(
        str(action.get("message") or "")
        for milestone in evaluation.milestones[:1]
        for action in milestone.event.human_actions
    ).lower()
    warnings: list[str] = []
    for milestone in evaluation.milestones:
        if milestone.outcome.evaluator != "information_report_score":
            continue
        answer = str(milestone.outcome.ground_truth.get("answer") or "").lower()
        answer_tokens = set(re.findall(r"[a-z0-9]+", answer))
        request_tokens = set(re.findall(r"[a-z0-9]+", initial_messages))
        if answer_tokens and len(answer_tokens - request_tokens) <= 2:
            warnings.append(
                f"{episode.episode_id}:{milestone.milestone_id}: report GT has "
                "little lexical information beyond the initial request; review "
                "whether a non-embodied language-only policy can guess it"
            )
    return tuple(warnings)


__all__ = ["BatchValidationReport", "validate_candidate_batch"]
