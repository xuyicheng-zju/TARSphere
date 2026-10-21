"""Factor task geometry into staged physical rollout trials."""

from __future__ import annotations

from collections import Counter
from typing import Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.bench.certification.physical_trial import (
    PhysicalTrialSelection,
)
from tarsphere.bench.certification.task_geometry import (
    TaskGeometryCandidates,
)


class StrictModel(BaseModel):
    """Immutable current-release trial-plan schema base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GraspTrial(StrictModel):
    """One Stage-A target approach and near-field grasp entry."""

    target_approach_path_index: int = Field(
        ge=0,
        description=(
            "Official target path whose exact G1-clear endpoint precedes "
            "near-field grasp motion."
        ),
    )
    grasp_entry_index: int = Field(
        ge=0,
        description=("Zero-based pose within that path’s ordered grasp-entry set."),
    )


class TransportTrial(StrictModel):
    """One Stage-B loaded route to an official destination approach."""

    destination_approach_path_index: int = Field(
        ge=0,
        description=(
            "Official destination path whose exact G1-clear endpoint closes "
            "the loaded transport stage."
        ),
    )


class PlacementTrial(StrictModel):
    """One Stage-C surface patch and release stance."""

    placement_patch_index: int = Field(
        ge=0,
        description=(
            "Zero-based live destination patch index within the shared task "
            "geometry candidates."
        ),
    )
    destination_approach_path_index: int = Field(
        ge=0,
        description=(
            "Official destination path from which this placement stance is "
            "occupancy-reachable."
        ),
    )
    placement_stance_index: int = Field(
        ge=0,
        description=("Zero-based pose within the selected patch/path stance set."),
    )


class PhysicalTrialPlan(StrictModel):
    """Staged finite search plan without a grasp×placement Cartesian product."""

    scene_id: str = Field(
        min_length=1,
        description="Grounded scene shared by every planned rollout trial.",
    )
    scene_server_id: str = Field(
        min_length=1,
        description=(
            "Unique baked Scene Server process shared by every planned trial."
        ),
    )
    scene_server_git_sha: str = Field(
        min_length=1,
        description=(
            "Scene Server revision that produced the live geometry underlying "
            "every planned trial."
        ),
    )
    target_object_id: str = Field(
        min_length=1,
        description="Grounded pickable exercised by all three trial stages.",
    )
    destination_object_id: str = Field(
        min_length=1,
        description="Grounded placement destination exercised by Stages B/C.",
    )
    goal_relation: Literal["on"] = Field(
        description="Final relation that successful Stage C must establish.",
    )
    grasp_trials: Tuple[GraspTrial, ...] = Field(
        min_length=1,
        description=(
            "Stage-A trials ordered first by official target path and then by "
            "the deterministic grasp-entry ranking."
        ),
    )
    transport_trials: Tuple[TransportTrial, ...] = Field(
        min_length=1,
        description=(
            "Stage-B trials containing only destination paths that retain at "
            "least one placement stance."
        ),
    )
    placement_trials: Tuple[PlacementTrial, ...] = Field(
        min_length=1,
        description=(
            "Stage-C trials ordered by live patch, destination path, and "
            "deterministic stance ranking."
        ),
    )

    @model_validator(mode="after")
    def validate_plan(self) -> "PhysicalTrialPlan":
        if self.target_object_id == self.destination_object_id:
            raise ValueError("physical trial target and destination must be distinct")
        _require_unique(self.grasp_trials, label="grasp trials")
        _require_unique(self.transport_trials, label="transport trials")
        _require_unique(self.placement_trials, label="placement trials")
        transport_paths = {
            trial.destination_approach_path_index for trial in self.transport_trials
        }
        placement_paths = {
            trial.destination_approach_path_index for trial in self.placement_trials
        }
        if placement_paths != transport_paths:
            raise ValueError(
                "transport and placement trials must cover the same destination paths"
            )
        return self

    def placements_for(
        self,
        transport: TransportTrial,
    ) -> tuple[PlacementTrial, ...]:
        """Return the ordered Stage-C choices reachable from one Stage B."""
        if transport not in self.transport_trials:
            raise ValueError("transport trial is absent from this physical trial plan")
        matches = tuple(
            placement
            for placement in self.placement_trials
            if (
                placement.destination_approach_path_index
                == transport.destination_approach_path_index
            )
        )
        if not matches:
            raise ValueError("transport trial has no compatible placement trial")
        return matches


def plan_physical_trials(
    candidates: TaskGeometryCandidates,
) -> PhysicalTrialPlan:
    """Create three finite ordered searches instead of one Cartesian search."""
    grasp_trials = tuple(
        sorted(
            (
                GraspTrial(
                    target_approach_path_index=entry_set.target_approach_path_index,
                    grasp_entry_index=entry_index,
                )
                for entry_set in candidates.grasp_entry_sets
                for entry_index in range(len(entry_set.poses))
            ),
            key=lambda trial: (
                trial.target_approach_path_index,
                trial.grasp_entry_index,
            ),
        )
    )

    patch_indexes = {
        patch.center_xyz: patch_index
        for patch_index, patch in enumerate(candidates.placement_patches)
    }
    placement_trials = tuple(
        sorted(
            (
                PlacementTrial(
                    placement_patch_index=patch_indexes[stance_set.surface_center_xyz],
                    destination_approach_path_index=(
                        stance_set.destination_approach_path_index
                    ),
                    placement_stance_index=stance_index,
                )
                for stance_set in candidates.placement_stance_sets
                for stance_index in range(len(stance_set.candidates))
            ),
            key=lambda trial: (
                trial.placement_patch_index,
                trial.destination_approach_path_index,
                trial.placement_stance_index,
            ),
        )
    )
    successful_destination_paths = {
        trial.destination_approach_path_index for trial in placement_trials
    }
    transport_trials = tuple(
        TransportTrial(
            destination_approach_path_index=path_index,
        )
        for path_index in candidates.examined_destination_path_indexes
        if path_index in successful_destination_paths
    )
    return PhysicalTrialPlan(
        scene_id=candidates.scene_id,
        scene_server_id=candidates.scene_server_id,
        scene_server_git_sha=candidates.scene_server_git_sha,
        target_object_id=candidates.target_object_id,
        destination_object_id=candidates.destination_object_id,
        goal_relation=candidates.goal_relation,
        grasp_trials=grasp_trials,
        transport_trials=transport_trials,
        placement_trials=placement_trials,
    )


def select_physical_trial(
    grasp: GraspTrial,
    transport: TransportTrial,
    placement: PlacementTrial,
) -> PhysicalTrialSelection:
    """Compose one full rollout only from causally compatible stage choices."""
    if (
        transport.destination_approach_path_index
        != placement.destination_approach_path_index
    ):
        raise ValueError(
            "transport and placement trials use different destination paths"
        )
    return PhysicalTrialSelection(
        target_approach_path_index=grasp.target_approach_path_index,
        grasp_entry_index=grasp.grasp_entry_index,
        placement_patch_index=placement.placement_patch_index,
        destination_approach_path_index=(transport.destination_approach_path_index),
        placement_stance_index=placement.placement_stance_index,
    )


def _require_unique(values: tuple[object, ...], *, label: str) -> None:
    duplicates = tuple(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} must be unique: {duplicates}")


__all__ = [
    "GraspTrial",
    "PhysicalTrialPlan",
    "PlacementTrial",
    "TransportTrial",
    "plan_physical_trials",
    "select_physical_trial",
]
