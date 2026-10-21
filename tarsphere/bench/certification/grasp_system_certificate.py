"""Certify the shared G1 grasp stack before judging scene assets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.bench.certification.grasp_system import (
    GraspSystemTrial,
    resolve_grasp_system_trial,
)
from tarsphere.bench.certification.grasp_system_episode import (
    compile_grasp_system_episode,
)
from tarsphere.bench.certification.physical_rollout import (
    PhysicalTrialRolloutResult,
    run_grasp_system_trial,
)
from tarsphere.bench.certification.scene_client import (
    SceneServerClient,
)
from tarsphere.bench.grounding.schema import GroundedScene


class StrictModel(BaseModel):
    """Immutable current-release grasp-system certificate base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GraspSystemCertificate(StrictModel):
    """Consecutive known-positive passes proving the grasp stack is healthy."""

    trial: GraspSystemTrial = Field(
        description="Exact reference geometry and grasp stance executed every time.",
    )
    required_consecutive_passes: int = Field(
        ge=2,
        description="Frozen uninterrupted pass count required for calibration.",
    )
    consecutive_passes: Tuple[PhysicalTrialRolloutResult, ...] = Field(
        min_length=2,
        description=(
            "Ordered live passes proving GT-guided IK, native contact, "
            "attachment, and carry-away on the known reference object."
        ),
    )

    @model_validator(mode="after")
    def validate_certificate(self) -> "GraspSystemCertificate":
        if len(self.consecutive_passes) != self.required_consecutive_passes:
            raise ValueError(
                "grasp-system certificate has the wrong consecutive pass count"
            )
        request_ids = tuple(rollout.request_id for rollout in self.consecutive_passes)
        reference_id = self.trial.reference_object.object_id
        if len(set(request_ids)) != len(request_ids) or any(
            rollout.stage != "A"
            or not rollout.passed
            or not rollout.grasp_certification_evidence.passed
            or rollout.grasp_certification_evidence.target_object_id != reference_id
            or rollout.scene_server_id != self.trial.scene_server_id
            or rollout.scene_server_git_sha != self.trial.scene_server_git_sha
            for rollout in self.consecutive_passes
        ):
            raise ValueError(
                "grasp-system certificate lacks uninterrupted reference passes"
            )
        return self


class GraspSystemCalibrationFailed(RuntimeError):
    """Raised when the known-positive reference fails the shared grasp stack."""

    def __init__(
        self,
        *,
        trial: GraspSystemTrial,
        rollouts: Tuple[PhysicalTrialRolloutResult, ...],
    ) -> None:
        self.trial = trial
        self.rollouts = rollouts
        latest = rollouts[-1]
        super().__init__(
            "Grasp system is not calibrated; candidate-object outcomes are "
            "inconclusive. Latest reference failure: "
            f"{latest.failure or latest.termination_reason}"
        )


def certify_grasp_system(
    scene: GroundedScene,
    *,
    scene_server: SceneServerClient,
    required_consecutive_passes: int = 3,
    maximum_decision_steps_per_run: Optional[int] = None,
    assets_root: Optional[Path | str] = None,
    record_video: bool = False,
) -> GraspSystemCertificate:
    """Require the same known-positive trial to pass without interruption."""
    if required_consecutive_passes < 2:
        raise ValueError(
            "grasp-system calibration requires at least two consecutive passes"
        )
    identity = scene_server.identity()
    if identity.active_request_id is not None:
        raise RuntimeError(
            "grasp-system calibration requires an idle single-flight server"
        )
    if scene.scene_id not in identity.scene:
        raise ValueError(
            "grasp-system calibration scene differs from the loaded server: "
            f"{scene.scene_id!r} not in {identity.scene!r}"
        )
    trial = resolve_grasp_system_trial(
        scene,
        scene_server_id=identity.scene_server_id,
        scene_server_git_sha=identity.scene_server_git_sha,
        assets_root=assets_root,
    )
    rollouts = []
    for pass_number in range(1, required_consecutive_passes + 1):
        episode = compile_grasp_system_episode(
            f"grasp_system_calibration_pass_{pass_number}",
            trial=trial,
            scene=scene,
        )
        rollout = run_grasp_system_trial(
            scene_server,
            episode,
            record_video=record_video,
            max_steps=maximum_decision_steps_per_run,
        )
        rollouts.append(rollout)
        if not rollout.passed:
            raise GraspSystemCalibrationFailed(
                trial=trial,
                rollouts=tuple(rollouts),
            )
    return GraspSystemCertificate(
        trial=trial,
        required_consecutive_passes=required_consecutive_passes,
        consecutive_passes=tuple(rollouts),
    )


__all__ = [
    "GraspSystemCalibrationFailed",
    "GraspSystemCertificate",
    "certify_grasp_system",
]
