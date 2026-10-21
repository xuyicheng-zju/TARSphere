"""Search grasp, loaded-route, and placement candidates through live trials."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.bench.task_candidates import (
    PhysicalTaskShortlist,
)
from tarsphere.bench.certification.physical_rollout import (
    PhysicalTrialRolloutResult,
    rollout_rejects_candidate,
    run_physical_trial,
)
from tarsphere.bench.certification.physical_trial import (
    PhysicalTrial,
    PhysicalTrialSelection,
    resolve_physical_trial,
)
from tarsphere.bench.certification.grasp_system_certificate import (
    GraspSystemCertificate,
)
from tarsphere.bench.certification.scene_client import (
    SceneServerClient,
)
from tarsphere.bench.certification.task_geometry import (
    TaskGeometryCandidates,
    TaskGeometryResolutionError,
    resolve_task_geometry_candidates,
)
from tarsphere.bench.certification.trial_plan import (
    GraspTrial,
    PlacementTrial,
    TransportTrial,
    plan_physical_trials,
    select_physical_trial,
)
from tarsphere.bench.certification.trial_episode import (
    compile_physical_trial_episode,
)
from tarsphere.bench.grounding.schema import GroundedScene
from tarsphere.bench.spatial.physical_task.target_screening import (
    ManipulationTargetScreening,
)


class StrictModel(BaseModel):
    """Immutable current-release physical-search schema base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GraspTrialAttempt(StrictModel):
    """One Stage-A candidate and its live cumulative rollout."""

    trial: GraspTrial = Field(
        description="Target approach and grasp entry tested by this attempt.",
    )
    rollout: PhysicalTrialRolloutResult = Field(
        description="Live Stage-A result returned for that exact candidate.",
    )

    @model_validator(mode="after")
    def validate_stage(self) -> "GraspTrialAttempt":
        if self.rollout.stage != "A":
            raise ValueError("grasp trial attempt requires a Stage-A rollout")
        return self


class TransportTrialAttempt(StrictModel):
    """One Stage-B destination path and its live cumulative rollout."""

    trial: TransportTrial = Field(
        description="Loaded destination path tested by this attempt.",
    )
    rollout: PhysicalTrialRolloutResult = Field(
        description=(
            "Live Stage-B result after repeating the already selected grasp "
            "as the physical setup for this route."
        ),
    )

    @model_validator(mode="after")
    def validate_stage(self) -> "TransportTrialAttempt":
        if self.rollout.stage != "B":
            raise ValueError("transport trial attempt requires a Stage-B rollout")
        return self


class PlacementTrialAttempt(StrictModel):
    """One Stage-C placement choice and its live cumulative rollout."""

    trial: PlacementTrial = Field(
        description="Live surface patch and release stance tested by this attempt.",
    )
    rollout: PhysicalTrialRolloutResult = Field(
        description=(
            "Live Stage-C result after repeating the selected grasp and "
            "loaded destination path."
        ),
    )

    @model_validator(mode="after")
    def validate_stage(self) -> "PlacementTrialAttempt":
        if self.rollout.stage != "C":
            raise ValueError("placement trial attempt requires a Stage-C rollout")
        return self


class PhysicalTaskCertification(StrictModel):
    """First live-certified grasp, loaded route, and dynamic placement."""

    certified_task: PhysicalTrial = Field(
        description=(
            "Exact nav–grasp–transport–place task assembled from the three "
            "passing stage choices and executed by the Stage-C rollout."
        ),
    )
    scene_id: str = Field(
        min_length=1,
        description="Grounded scene in which all three stages were certified.",
    )
    proposal_index: int = Field(
        ge=0,
        description="Semantic shortlist proposal under physical certification.",
    )
    scene_server_id: str = Field(
        min_length=1,
        description="Unique baked Scene Server used by every attempt.",
    )
    scene_server_git_sha: str = Field(
        min_length=1,
        description="Repository revision loaded by that Scene Server.",
    )
    target_object_id: str = Field(
        min_length=1,
        description="Grounded object attached by the certified grasp choice.",
    )
    destination_object_id: str = Field(
        min_length=1,
        description=(
            "Grounded destination reached and dynamically supporting the "
            "released target."
        ),
    )
    certified_grasp: GraspTrial = Field(
        description="First ordered Stage-A candidate that passed live rollout.",
    )
    certified_transport: TransportTrial = Field(
        description="First ordered Stage-B loaded path that passed live rollout.",
    )
    certified_placement: PlacementTrial = Field(
        description=(
            "First ordered Stage-C surface and stance that passed dynamic "
            "release and settle certification."
        ),
    )
    grasp_attempts: Tuple[GraspTrialAttempt, ...] = Field(
        min_length=1,
        description="Ordered Stage-A attempts through the first passing grasp.",
    )
    transport_attempts: Tuple[TransportTrialAttempt, ...] = Field(
        min_length=1,
        description="Ordered Stage-B attempts through the first passing path.",
    )
    placement_attempts: Tuple[PlacementTrialAttempt, ...] = Field(
        min_length=1,
        description=(
            "Ordered Stage-C attempts through the first passing dynamic placement."
        ),
    )

    @model_validator(mode="after")
    def validate_certification(self) -> "PhysicalTaskCertification":
        expected_selection = select_physical_trial(
            self.certified_grasp,
            self.certified_transport,
            self.certified_placement,
        )
        if self.certified_task.selection != expected_selection:
            raise ValueError(
                "certified_task must contain the three passing stage choices"
            )
        if (
            self.certified_task.scene_id != self.scene_id
            or self.certified_task.scene_server_id != self.scene_server_id
            or self.certified_task.scene_server_git_sha != self.scene_server_git_sha
            or self.certified_task.grasp.target_object_id != self.target_object_id
            or self.certified_task.place.destination_object_id
            != self.destination_object_id
        ):
            raise ValueError("certified_task does not match its certification identity")
        _require_unique(
            tuple(item.trial for item in self.grasp_attempts),
            label="grasp attempts",
        )
        _require_unique(
            tuple(item.trial for item in self.transport_attempts),
            label="transport attempts",
        )
        _require_unique(
            tuple(item.trial for item in self.placement_attempts),
            label="placement attempts",
        )
        if self.grasp_attempts[-1].trial != self.certified_grasp:
            raise ValueError("certified_grasp must be the final Stage-A attempt")
        if self.transport_attempts[-1].trial != self.certified_transport:
            raise ValueError("certified_transport must be the final Stage-B attempt")
        if self.placement_attempts[-1].trial != self.certified_placement:
            raise ValueError("certified_placement must be the final Stage-C attempt")
        if not self.grasp_attempts[-1].rollout.passed:
            raise ValueError("certified grasp rollout did not pass")
        if not self.transport_attempts[-1].rollout.passed:
            raise ValueError("certified transport rollout did not pass")
        if not self.placement_attempts[-1].rollout.passed:
            raise ValueError("certified placement rollout did not pass")
        if any(
            item.rollout.passed
            for item in (
                *self.grasp_attempts[:-1],
                *self.transport_attempts[:-1],
                *self.placement_attempts[:-1],
            )
        ):
            raise ValueError("physical search must stop at the first passing candidate")
        if any(
            attempt.rollout.scene_server_id != self.scene_server_id
            or attempt.rollout.scene_server_git_sha != self.scene_server_git_sha
            for attempt in (
                *self.grasp_attempts,
                *self.transport_attempts,
                *self.placement_attempts,
            )
        ):
            raise ValueError(
                "every physical search attempt must use the bound Scene Server"
            )
        return self


class PhysicalTrialCandidatesExhausted(RuntimeError):
    """Raised after every ordered candidate for one stage was rejected."""

    def __init__(
        self,
        *,
        stage: Literal["A", "B", "C"],
        attempts: tuple[
            GraspTrialAttempt | TransportTrialAttempt | PlacementTrialAttempt,
            ...,
        ],
    ) -> None:
        self.stage = stage
        self.attempts = attempts
        super().__init__(
            f"Physical trial Stage {stage} exhausted "
            f"{len(attempts)} ordered candidates without a pass"
        )


class PhysicalTrialSearchInterrupted(RuntimeError):
    """Raised when a system failure cannot be interpreted as candidate evidence."""

    def __init__(
        self,
        *,
        stage: Literal["A", "B", "C"],
        attempt: (GraspTrialAttempt | TransportTrialAttempt | PlacementTrialAttempt),
    ) -> None:
        self.stage = stage
        self.attempt = attempt
        failure = attempt.rollout.failure
        failure_name = failure.type if failure is not None else "unknown"
        super().__init__(
            f"Physical trial Stage {stage} stopped on non-candidate failure "
            f"{failure_name!r}; no later candidate was attempted"
        )


class PhysicalTaskProposalRejection(StrictModel):
    """Why one reviewed semantic task failed physical admission."""

    proposal_index: int = Field(
        ge=0,
        description="Zero-based rank of the rejected semantic task proposal.",
    )
    rejection_phase: Literal["geometry", "physical_rollout"] = Field(
        description=(
            "Whether rejection occurred before Robot motion or during staged "
            "live certification."
        ),
    )
    message: str = Field(
        min_length=1,
        description="Candidate-specific failure reported by the physical stage.",
    )


class PhysicalTaskShortlistCertification(StrictModel):
    """First reviewed physical-task proposal passing complete Stage C."""

    shortlist: PhysicalTaskShortlist = Field(
        description="Reviewed semantic proposals considered in ranked order.",
    )
    rejected_proposals: Tuple[PhysicalTaskProposalRejection, ...] = Field(
        default=(),
        description="Ordered candidate failures preceding the accepted task.",
    )
    accepted_task: PhysicalTaskCertification = Field(
        description=(
            "First proposal with certified grasp, loaded transport, dynamic "
            "release, and settled placement."
        ),
    )

    @model_validator(mode="after")
    def validate_ranked_search(self) -> "PhysicalTaskShortlistCertification":
        accepted_index = self.accepted_task.proposal_index
        if accepted_index >= len(self.shortlist.proposals):
            raise ValueError("accepted physical task is outside its semantic shortlist")
        rejected_indexes = tuple(
            rejection.proposal_index for rejection in self.rejected_proposals
        )
        if rejected_indexes != tuple(range(accepted_index)):
            raise ValueError(
                "physical-task rejections must precede acceptance in rank order"
            )
        proposal = self.shortlist.proposals[accepted_index]
        if (
            self.accepted_task.scene_id != self.shortlist.scene_id
            or self.accepted_task.target_object_id != proposal.target_object_id
            or self.accepted_task.destination_object_id
            != proposal.destination_object_id
        ):
            raise ValueError(
                "accepted physical certification differs from its proposal"
            )
        return self


def certify_physical_task_shortlist(
    scene: GroundedScene,
    shortlist: PhysicalTaskShortlist,
    target_screening: ManipulationTargetScreening,
    *,
    scene_server: SceneServerClient,
    grasp_system_certificate: GraspSystemCertificate,
    assets_root: Optional[Path | str] = None,
    record_video: bool = False,
    max_steps: Optional[int] = None,
) -> PhysicalTaskShortlistCertification:
    """Return the first ranked proposal that passes complete physical trials."""
    if shortlist.scene_id != scene.scene_id:
        raise ValueError("physical-task shortlist and grounded scene must match")
    identity = scene_server.identity()
    calibration = grasp_system_certificate
    if (
        calibration.trial.scene_id != scene.scene_id
        or calibration.trial.scene_server_id != identity.scene_server_id
        or calibration.trial.scene_server_git_sha != identity.scene_server_git_sha
        or scene.scene_id not in identity.scene
        or identity.active_request_id is not None
    ):
        raise ValueError(
            "physical-task certification requires a matching grasp-system "
            "certificate from this idle Scene Server process and revision"
        )
    rejections = []
    for proposal_index in range(len(shortlist.proposals)):
        try:
            candidates = resolve_task_geometry_candidates(
                scene,
                shortlist,
                target_screening,
                proposal_index=proposal_index,
                scene_server=scene_server,
                assets_root=assets_root,
            )
        except TaskGeometryResolutionError as exc:
            rejections.append(
                PhysicalTaskProposalRejection(
                    proposal_index=proposal_index,
                    rejection_phase="geometry",
                    message=str(exc),
                )
            )
            continue
        try:
            accepted = certify_physical_task(
                scene,
                candidates,
                scene_server=scene_server,
                record_video=record_video,
                max_steps=max_steps,
            )
        except PhysicalTrialCandidatesExhausted as exc:
            rejections.append(
                PhysicalTaskProposalRejection(
                    proposal_index=proposal_index,
                    rejection_phase="physical_rollout",
                    message=str(exc),
                )
            )
            continue
        return PhysicalTaskShortlistCertification(
            shortlist=shortlist,
            rejected_proposals=tuple(rejections),
            accepted_task=accepted,
        )
    raise PhysicalTaskCandidatesExhausted(
        shortlist=shortlist,
        rejections=tuple(rejections),
    )


class PhysicalTaskCandidatesExhausted(RuntimeError):
    """Raised after every reviewed physical-task proposal was rejected."""

    def __init__(
        self,
        *,
        shortlist: PhysicalTaskShortlist,
        rejections: tuple[PhysicalTaskProposalRejection, ...],
    ) -> None:
        self.shortlist = shortlist
        self.rejections = rejections
        super().__init__(
            f"Physical task shortlist exhausted {len(rejections)} reviewed "
            "proposals without a complete Stage-C pass"
        )


def certify_physical_task(
    scene: GroundedScene,
    candidates: TaskGeometryCandidates,
    *,
    scene_server: SceneServerClient,
    required_grasp: Optional[GraspTrial] = None,
    record_video: bool = False,
    max_steps: Optional[int] = None,
) -> PhysicalTaskCertification:
    """Certify the first viable grasp, loaded path, and dynamic placement.

    ``required_grasp`` is used only when another already-certified task must
    continue from the exact same target-side manipulation choice.
    """
    if scene.scene_id != candidates.scene_id:
        raise ValueError("physical search candidates and grounded scene must match")
    plan = plan_physical_trials(candidates)
    grasp_trials = (
        plan.grasp_trials
        if required_grasp is None
        else tuple(grasp for grasp in plan.grasp_trials if grasp == required_grasp)
    )
    if not grasp_trials:
        raise ValueError("required grasp is absent from the task geometry candidates")
    representative_transport = plan.transport_trials[0]
    representative_placement = plan.placements_for(representative_transport)[0]

    grasp_attempts = []
    certified_grasp = None
    for grasp in grasp_trials:
        selection = select_physical_trial(
            grasp,
            representative_transport,
            representative_placement,
        )
        episode = _compile_attempt(
            scene,
            candidates,
            selection,
            stage="A",
        )
        attempt = GraspTrialAttempt(
            trial=grasp,
            rollout=run_physical_trial(
                scene_server,
                episode,
                stage="A",
                record_video=record_video,
                max_steps=max_steps,
            ),
        )
        grasp_attempts.append(attempt)
        if attempt.rollout.passed:
            certified_grasp = grasp
            break
        if not rollout_rejects_candidate(attempt.rollout):
            raise PhysicalTrialSearchInterrupted(
                stage="A",
                attempt=attempt,
            )
    if certified_grasp is None:
        raise PhysicalTrialCandidatesExhausted(
            stage="A",
            attempts=tuple(grasp_attempts),
        )

    transport_attempts = []
    certified_transport = None
    for transport in plan.transport_trials:
        placement = plan.placements_for(transport)[0]
        selection = select_physical_trial(
            certified_grasp,
            transport,
            placement,
        )
        episode = _compile_attempt(
            scene,
            candidates,
            selection,
            stage="B",
        )
        attempt = TransportTrialAttempt(
            trial=transport,
            rollout=run_physical_trial(
                scene_server,
                episode,
                stage="B",
                record_video=record_video,
                max_steps=max_steps,
            ),
        )
        transport_attempts.append(attempt)
        if attempt.rollout.passed:
            certified_transport = transport
            break
        if not rollout_rejects_candidate(attempt.rollout):
            raise PhysicalTrialSearchInterrupted(
                stage="B",
                attempt=attempt,
            )
    if certified_transport is None:
        raise PhysicalTrialCandidatesExhausted(
            stage="B",
            attempts=tuple(transport_attempts),
        )

    placement_attempts = []
    certified_placement = None
    for placement in plan.placements_for(certified_transport):
        selection = select_physical_trial(
            certified_grasp,
            certified_transport,
            placement,
        )
        episode = _compile_attempt(
            scene,
            candidates,
            selection,
            stage="C",
        )
        attempt = PlacementTrialAttempt(
            trial=placement,
            rollout=run_physical_trial(
                scene_server,
                episode,
                stage="C",
                record_video=record_video,
                max_steps=max_steps,
            ),
        )
        placement_attempts.append(attempt)
        if attempt.rollout.passed:
            certified_placement = placement
            break
        if not rollout_rejects_candidate(attempt.rollout):
            raise PhysicalTrialSearchInterrupted(
                stage="C",
                attempt=attempt,
            )
    if certified_placement is None:
        raise PhysicalTrialCandidatesExhausted(
            stage="C",
            attempts=tuple(placement_attempts),
        )
    certified_selection = select_physical_trial(
        certified_grasp,
        certified_transport,
        certified_placement,
    )
    certified_task = resolve_physical_trial(
        scene,
        candidates,
        certified_selection,
    )

    return PhysicalTaskCertification(
        certified_task=certified_task,
        scene_id=candidates.scene_id,
        proposal_index=candidates.proposal_index,
        scene_server_id=candidates.scene_server_id,
        scene_server_git_sha=candidates.scene_server_git_sha,
        target_object_id=candidates.target_object_id,
        destination_object_id=candidates.destination_object_id,
        certified_grasp=certified_grasp,
        certified_transport=certified_transport,
        certified_placement=certified_placement,
        grasp_attempts=tuple(grasp_attempts),
        transport_attempts=tuple(transport_attempts),
        placement_attempts=tuple(placement_attempts),
    )


def _compile_attempt(
    scene: GroundedScene,
    candidates: TaskGeometryCandidates,
    selection: PhysicalTrialSelection,
    *,
    stage: Literal["A", "B", "C"],
):
    trial = resolve_physical_trial(scene, candidates, selection)
    trial_id = (
        f"physical_proposal_{candidates.proposal_index}_stage_{stage.lower()}"
        f"_target_path_{selection.target_approach_path_index}"
        f"_grasp_entry_{selection.grasp_entry_index}"
        f"_destination_path_{selection.destination_approach_path_index}"
        f"_patch_{selection.placement_patch_index}"
        f"_stance_{selection.placement_stance_index}"
    )
    return compile_physical_trial_episode(
        trial_id,
        trial=trial,
        scene=scene,
    )


def _require_unique(values: tuple[object, ...], *, label: str) -> None:
    duplicates = tuple(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} must be unique: {duplicates}")


__all__ = [
    "GraspTrialAttempt",
    "PhysicalTaskCertification",
    "PhysicalTaskCandidatesExhausted",
    "PhysicalTaskProposalRejection",
    "PhysicalTaskShortlistCertification",
    "PhysicalTrialCandidatesExhausted",
    "PhysicalTrialSearchInterrupted",
    "PlacementTrialAttempt",
    "TransportTrialAttempt",
    "certify_physical_task",
    "certify_physical_task_shortlist",
    "rollout_rejects_candidate",
]
