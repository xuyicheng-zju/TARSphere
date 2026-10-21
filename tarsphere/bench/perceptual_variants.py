"""Compile one sensor-condition variant over an unchanged production Base."""

from __future__ import annotations

from copy import deepcopy

from tarsphere.bench.schema import TARSphereEpisode
from tarsphere.infra.perception.sensor_conditions import PerceptualCondition


def compile_perceptual_variant(base: dict, condition: PerceptualCondition) -> dict:
    parsed = TARSphereEpisode.model_validate(base)
    if parsed.evaluation.task_profile.capability_slice.value != "base":
        raise ValueError("Perceptual robustness requires a paired Base episode")
    if condition not in {"low_light", "reduced_resolution", "depth_noise"}:
        raise ValueError(f"Unknown perceptual robustness condition: {condition!r}")
    variant = deepcopy(base)
    variant["episode_id"] = (
        base["episode_id"].removesuffix("_base") + "_perceptual_robustness"
    )
    variant["title"] = base["title"] + f" / {condition.replace('_', ' ')}"
    variant["evaluation"]["task_profile"] = {
        "mainline": parsed.evaluation.task_profile.mainline.value,
        "capability_slice": "perceptual_robustness",
        "subtype": condition,
    }
    variant["authoring"]["achieved_gates"] = ["authored", "schema_valid"]
    variant["authoring"]["difficulty_factors"] = []
    variant["metadata"]["paired_base_episode_id"] = base["episode_id"]
    variant["tags"] = [tag for tag in base.get("tags", ()) if tag != "base"] + [
        "perceptual_robustness"
    ]
    validate_perceptual_pair(base, variant)
    return variant


def validate_perceptual_pair(base: dict, variant: dict) -> None:
    """Reject changes to task semantics, geometry, scoring, or interaction."""

    original = TARSphereEpisode.model_validate(base)
    candidate = TARSphereEpisode.model_validate(variant)
    if original.evaluation.task_profile.capability_slice.value != "base":
        raise ValueError("Paired source is not Base")
    if (
        candidate.evaluation.task_profile.capability_slice.value
        != "perceptual_robustness"
    ):
        raise ValueError("Candidate is not perceptual robustness")
    for field in (
        "scene",
        "robot",
        "humans",
        "interaction",
        "runtime",
        "world_knowledge",
        "asset_adaptation",
    ):
        if base.get(field) != variant.get(field):
            raise ValueError(f"Perceptual condition changed paired Base {field}")
    if base["evaluation"]["milestones"] != variant["evaluation"]["milestones"]:
        raise ValueError("Perceptual condition changed the five scored outcomes")
    if len(candidate.evaluation.milestones) != 5:
        raise ValueError("Perceptual condition cannot add a milestone")
    if base["authoring"]["route_overlay"] != variant["authoring"]["route_overlay"]:
        raise ValueError("Perceptual condition changed the paired route overlay")
    if candidate.authoring.difficulty_factors:
        raise ValueError("Perceptual condition must inherit Base difficulty")


__all__ = ["compile_perceptual_variant", "validate_perceptual_pair"]
