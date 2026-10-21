"""Benchmark authoring, validation, certification and metric APIs."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    "TARSphereEpisode": ".schema",
    "EpisodeAuthoringSpec": ".schema",
    "AuthoringEvidenceRecord": ".schema",
    "load_asset_source_registry": ".inputs.sources",
    "resolve_active_grscene": ".inputs.sources",
    "resolve_scene": ".grounding",
    "resolve_static_navigation": ".spatial",
    "validate_candidate_geometry": ".authoring",
    "validate_candidate_payload": ".authoring",
    "validate_candidate_publication": ".authoring",
    "PRODUCTION_NAVIGATION_CLEARANCE_RADIUS_M": ".production_accessibility",
    "PRODUCTION_ROUTE_LENGTH_REVIEW_THRESHOLD_M": ".production_accessibility",
    "validate_production_navigation_accessibility": ".production_accessibility",
    "SceneServerClient": ".certification.scene_client",
    "certify_physical_task": ".certification.physical_search",
    "DIFFICULTY_CATALOG": ".difficulty",
    "DIFFICULTY_TIER_RANGES_BY_MAINLINE": ".difficulty",
    "BASE_DIFFICULTY_CALIBRATION": ".difficulty",
    "DifficultyCalibration": ".difficulty",
    "DifficultyDimension": ".difficulty",
    "DifficultyFactor": ".difficulty",
    "DifficultyScore": ".difficulty",
    "BaseDifficultyMeasurements": ".difficulty",
    "DifficultyTier": ".difficulty",
    "DifficultyAssignment": ".difficulty",
    "assign_batch_difficulty": ".difficulty",
    "base_difficulty_key": ".difficulty",
    "derive_episode_difficulty": ".difficulty",
    "measure_base_difficulty": ".difficulty",
    "route_extent_factor": ".difficulty",
    "score_difficulty": ".difficulty",
    "BENCHMARK_CELLS": ".strata",
    "BenchmarkCell": ".strata",
    "CellMetrics": ".reporting",
    "EpisodeCost": ".reporting",
    "EpisodeMeasurement": ".reporting",
    "summarize_reporting_cells": ".reporting",
    "PaperCellMetrics": ".reporting",
    "PaperFamilyAverage": ".reporting",
    "PaperPairedDifference": ".reporting",
    "PaperMetrics": ".reporting",
    "summarize_paper_metrics": ".reporting",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
