"""Validate live Scene Server results for Human-free physical trials."""

from __future__ import annotations

import uuid
import math
from typing import Any, Literal, Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.bench.certification.grasp_reach import (
    GraspReachEvidence,
    PhysXGraspContactEvidence,
)
from tarsphere.bench.certification.placement_release import (
    PlacementReleaseEvidence,
)
from tarsphere.bench.certification.physical_trial import (
    PhysicalTrial,
)
from tarsphere.bench.certification.grasp_system import (
    GraspSystemTrial,
)
from tarsphere.embodiment.dex3 import (
    DEX3_MINIMUM_CLOSURE_L2_RAD,
    dex3_closure_l2,
)
from tarsphere.infra.control.physical_contracts import G1_GRASP_CONTACT_GAP_M
from tarsphere.bench.certification.scene_client import (
    SceneServerClient,
    SceneServerError,
)
from tarsphere.bench.schema import TARSphereEpisode


class StrictModel(BaseModel):
    """Immutable current-release physical-rollout schema base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PhysicalTrialGates(StrictModel):
    """Cumulative grasp, transport, and placement gate outcomes."""

    grasp: bool = Field(
        description="Whether Stage A proved attachment and carry preparation.",
    )
    transport: bool = Field(
        description="Whether Stage B also proved loaded navigation.",
    )
    placement: bool = Field(
        description="Whether Stage C also proved the final placement relation.",
    )


class PhysicalTrialFinalState(StrictModel):
    """Robot and attachment state at the end of a physical trial."""

    upright: bool = Field(
        description="Whether G1 satisfied the final pelvis stability contract.",
    )
    ever_attached: bool = Field(
        description=(
            "Whether the target was held by a fixed joint or a verified "
            "physical hold during the run."
        ),
    )
    attached: bool = Field(
        description="Whether the target remained attached at termination.",
    )
    xy: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Final G1 world XY position when a heartbeat was recorded.",
    )
    yaw_rad: Optional[float] = Field(
        default=None,
        description="Final G1 world yaw when a heartbeat was recorded.",
    )


class Dex3GraspAttachmentEvidence(StrictModel):
    """Finger closure and opposing native contacts authorizing attachment."""

    hand: Literal["right"] = Field(
        description="Certified right Dex3 hand that established the grasp.",
    )
    target_object_id: str = Field(
        min_length=1,
        description="Exact grounded object contacted by this hand.",
    )
    finger_joint_positions: Tuple[float, ...] = Field(
        min_length=7,
        max_length=7,
        description="Measured positions of the seven Dex3 joints.",
    )
    finger_closure_l2_rad: float = Field(
        ge=0.0,
        description="Measured L2 displacement from the open-hand pose.",
    )
    minimum_finger_closure_l2_rad: float = Field(
        ge=0.0,
        description="Frozen minimum closure required before attachment.",
    )
    finger_contacts: Tuple[PhysXGraspContactEvidence, ...] = Field(
        min_length=2,
        description="All native finger contacts used by the grasp gate.",
    )
    opposing_finger_contact: bool = Field(
        description=("Whether thumb and index-or-middle touched the same target."),
    )

    @model_validator(mode="after")
    def validate_attachment(self) -> "Dex3GraspAttachmentEvidence":
        measured_closure = dex3_closure_l2(self.finger_joint_positions)
        if not math.isclose(
            measured_closure,
            self.finger_closure_l2_rad,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "declared Dex3 closure disagrees with measured joint state"
            )
        if not math.isclose(
            self.minimum_finger_closure_l2_rad,
            DEX3_MINIMUM_CLOSURE_L2_RAD,
            abs_tol=1e-9,
        ):
            raise ValueError("grasp certificate does not use the frozen Dex3 closure")
        if measured_closure < self.minimum_finger_closure_l2_rad:
            raise ValueError("Dex3 hand did not close enough to grasp")

        prefix = f"{self.hand}_hand_"
        contacts = tuple(
            contact
            for contact in self.finger_contacts
            if (
                contact.hand == self.hand
                and contact.target_object_id == self.target_object_id
                and contact.hand_link.startswith(prefix)
                and contact.separation_m <= G1_GRASP_CONTACT_GAP_M
            )
        )
        has_thumb = any("_thumb_" in contact.hand_link for contact in contacts)
        has_finger = any(
            ("_index_" in contact.hand_link or "_middle_" in contact.hand_link)
            for contact in contacts
        )
        opposing = bool(has_thumb and has_finger)
        if self.opposing_finger_contact != opposing or not opposing:
            raise ValueError(
                "Dex3 attachment requires thumb and index-or-middle contact "
                "on the same target"
            )
        return self


class GraspCertificationEvidence(StrictModel):
    """Live evidence required before one object can enter the pickable pool."""

    target_object_id: str = Field(
        min_length=1,
        description="Grounded object tested by this certification rollout.",
    )
    contact: bool = Field(
        description="Whether live hand and target geometry reached contact.",
    )
    contact_evidence: Optional[PhysXGraspContactEvidence] = Field(
        default=None,
        description=("Closest native PhysX contact for the exact hand–target pair."),
    )
    attachment_evidence: Optional[Dex3GraspAttachmentEvidence] = Field(
        default=None,
        description=(
            "Measured Dex3 closure and opposing contacts that authorized the "
            "retention joint."
        ),
    )
    maximum_contact_separation_m: float = Field(
        ge=0.0,
        description=("Largest PhysX collider separation accepted as contact."),
    )
    minimum_aabb_surface_gap_m: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=("Smallest diagnostic hand–target AABB gap; never a pass gate."),
    )
    carried_away: bool = Field(
        description="Whether a valid attachment displaced the object enough.",
    )
    maximum_attached_displacement_m: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Largest displacement while the FixedJoint was valid.",
    )
    required_displacement_m: float = Field(
        gt=0.0,
        description="Minimum displacement proving departure from support.",
    )
    carried_state: Optional[Mapping[str, Any]] = Field(
        default=None,
        description="Trace state attaining the largest valid displacement.",
    )
    passed: bool = Field(
        description="Whether native contact and carry-away both passed.",
    )

    @model_validator(mode="after")
    def validate_evidence(self) -> "GraspCertificationEvidence":
        if not math.isclose(
            self.maximum_contact_separation_m,
            G1_GRASP_CONTACT_GAP_M,
            abs_tol=1e-9,
        ):
            raise ValueError("grasp certificate does not use the frozen contact gap")
        closest_contact_is_valid = bool(
            self.contact_evidence is not None
            and self.contact_evidence.target_object_id == self.target_object_id
            and self.contact_evidence.separation_m <= self.maximum_contact_separation_m
        )
        attachment_is_valid = bool(
            self.attachment_evidence is not None
            and self.attachment_evidence.target_object_id == self.target_object_id
            and self.attachment_evidence.opposing_finger_contact
        )
        contact = bool(closest_contact_is_valid and attachment_is_valid)
        carried = bool(
            self.maximum_attached_displacement_m is not None
            and self.maximum_attached_displacement_m >= self.required_displacement_m
            and self.carried_state is not None
        )
        if (
            self.contact != contact
            or self.carried_away != carried
            or self.passed != (contact and carried)
        ):
            raise ValueError(
                "grasp certificate booleans disagree with their live evidence"
            )
        return self


class PhysicalTrialFailure(StrictModel):
    """Fail-loud exception materialized by the Scene Server."""

    type: str = Field(
        min_length=1,
        description="Python exception class raised by the rollout.",
    )
    message: str = Field(
        min_length=1,
        description="Actionable exception message from the failed rollout.",
    )
    details: Mapping[str, Any] = Field(
        default_factory=dict,
        description="Structured exception details when the failure supplied them.",
    )

    @model_validator(mode="after")
    def validate_candidate_rejection(self) -> "PhysicalTrialFailure":
        if self.type != "PhysicalTrialCandidateError":
            return self
        required = {
            "physical_trial_stage",
            "candidate_rejection_reason",
            "evidence",
        }
        missing = sorted(required - set(self.details))
        if missing:
            raise ValueError(
                f"PhysicalTrialCandidateError is missing structured details: {missing}"
            )
        if self.details["physical_trial_stage"] not in {"A", "B", "C"}:
            raise ValueError("physical_trial_stage must be A, B, or C")
        if not str(self.details["candidate_rejection_reason"]).strip():
            raise ValueError("candidate_rejection_reason must not be empty")
        if not isinstance(self.details["evidence"], dict):
            raise ValueError("candidate rejection evidence must be a mapping")
        return self

    @property
    def rejects_candidate(self) -> bool:
        """Whether this failure is an expected bounded-candidate rejection."""
        return self.type == "PhysicalTrialCandidateError"


class PhysicalTrialRolloutResult(StrictModel):
    """Typed result for one cumulative Human-free certification stage."""

    request_id: str = Field(
        min_length=1,
        description="Single-flight request identifier sent by this client.",
    )
    trial_id: str = Field(
        min_length=1,
        description="Runtime episode identifier of the executed physical trial.",
    )
    stage: Literal["A", "B", "C"] = Field(
        description=(
            "Cumulative certification prefix executed: A=grasp, B=grasp plus "
            "transport, and C=grasp plus transport plus dynamic placement."
        ),
    )
    scene_server_id: str = Field(
        min_length=1,
        description=("Unique baked Scene Server process that executed the rollout."),
    )
    scene_server_git_sha: str = Field(
        min_length=1,
        description="Repository revision loaded by that Scene Server process.",
    )
    passed: bool = Field(
        description="Whether the requested cumulative stage passed.",
    )
    gates: PhysicalTrialGates = Field(
        description="Cumulative outcomes for all three physical gates.",
    )
    decision_steps: int = Field(
        ge=0,
        description="Agent decisions consumed by this rollout.",
    )
    termination_reason: str = Field(
        min_length=1,
        description="Harness termination condition for this rollout.",
    )
    final_state: PhysicalTrialFinalState = Field(
        description="Final G1 stability, pose, and attachment state.",
    )
    grasp_reach_evidence: Tuple[GraspReachEvidence, ...] = Field(
        default=(),
        description=(
            "Live hand–target geometry measured during bounded Stage-A arm reaches."
        ),
    )
    grasp_certification_evidence: GraspCertificationEvidence = Field(
        description=("Native contact and carry-away evidence for Stage A."),
    )
    placement_release_evidence: Optional[PlacementReleaseEvidence] = Field(
        default=None,
        description=(
            "Dynamic release and fixed settle-window evidence for Stage C; "
            "omitted for Stages A and B."
        ),
    )
    failure: Optional[PhysicalTrialFailure] = Field(
        default=None,
        description="Fail-loud rollout exception, omitted for normal completion.",
    )

    @model_validator(mode="after")
    def validate_cumulative_gates(self) -> "PhysicalTrialRolloutResult":
        if self.gates.placement and not self.gates.transport:
            raise ValueError("placement gate cannot pass before transport")
        if self.gates.transport and not self.gates.grasp:
            raise ValueError("transport gate cannot pass before grasp")
        requested_gate = {
            "A": self.gates.grasp,
            "B": self.gates.transport,
            "C": self.gates.placement,
        }[self.stage]
        if self.passed != requested_gate:
            raise ValueError("passed must equal the requested cumulative stage gate")
        if self.gates.grasp and (
            not self.final_state.upright
            or not self.final_state.ever_attached
            or not self.grasp_reach_evidence
            or not self.grasp_certification_evidence.passed
        ):
            raise ValueError(
                "a passing grasp gate requires an upright final state, a "
                "prior hold, and live grasp-reach evidence"
            )
        if self.passed and self.stage in {"A", "B"} and not self.final_state.attached:
            raise ValueError("passing Stage A/B requires the target to remain attached")
        if self.stage == "C":
            evidence = self.placement_release_evidence
            if evidence is None:
                raise ValueError("Stage C requires placement-release evidence")
            if evidence.passed != self.gates.placement:
                raise ValueError(
                    "placement gate disagrees with placement-release evidence"
                )
            if self.passed and self.final_state.attached:
                raise ValueError("passing Stage C requires the target to be released")
        elif self.placement_release_evidence is not None:
            raise ValueError("Stages A/B cannot contain placement-release evidence")
        if self.passed and self.failure is not None:
            raise ValueError("a passed physical trial cannot contain a failure")
        return self


def run_physical_trial(
    scene_server: SceneServerClient,
    episode: TARSphereEpisode,
    *,
    stage: Literal["A", "B", "C"],
    request_id: Optional[str] = None,
    record_video: bool = False,
    max_steps: Optional[int] = None,
) -> PhysicalTrialRolloutResult:
    """Run and validate one cumulative physical-trial prefix."""
    if stage not in {"A", "B", "C"}:
        raise ValueError("physical trial stage must be A, B, or C")
    world = episode.world_knowledge
    if (
        not isinstance(world, dict)
        or "physical_trial" not in world
        or any(
            key in world
            for key in (
                "physical_task",
                "grasp_system_trial",
            )
        )
    ):
        raise ValueError(
            "run_physical_trial requires one Human-free physical-trial episode"
        )
    trial = PhysicalTrial.model_validate(world["physical_trial"])

    result = _run_certification_episode(
        scene_server,
        episode,
        stage=stage,
        request_id=request_id,
        record_video=record_video,
        max_steps=max_steps,
    )
    if (
        result.scene_server_id != trial.scene_server_id
        or result.scene_server_git_sha != trial.scene_server_git_sha
    ):
        raise SceneServerError(
            "Physical-trial rollout did not run on the Scene Server instance "
            "that produced its live geometry"
        )
    return result


def run_grasp_system_trial(
    scene_server: SceneServerClient,
    episode: TARSphereEpisode,
    *,
    request_id: Optional[str] = None,
    record_video: bool = False,
    max_steps: Optional[int] = None,
) -> PhysicalTrialRolloutResult:
    """Run the known-positive reference through the normal grasp stack."""
    world = episode.world_knowledge
    if (
        not isinstance(world, dict)
        or "grasp_system_trial" not in world
        or any(
            key in world
            for key in (
                "physical_trial",
                "physical_task",
                "episode_route",
            )
        )
    ):
        raise ValueError("run_grasp_system_trial requires one grasp-system episode")
    trial = GraspSystemTrial.model_validate(world["grasp_system_trial"])
    result = _run_certification_episode(
        scene_server,
        episode,
        stage="A",
        request_id=request_id,
        record_video=record_video,
        max_steps=max_steps,
    )
    if (
        result.scene_server_id != trial.scene_server_id
        or result.scene_server_git_sha != trial.scene_server_git_sha
    ):
        raise SceneServerError(
            "Grasp-system rollout ran on a different Scene Server instance"
        )
    if (
        result.grasp_certification_evidence.target_object_id
        != trial.reference_object.object_id
    ):
        raise SceneServerError("Grasp-system rollout certified a different target")
    return result


def _run_certification_episode(
    scene_server: SceneServerClient,
    episode: TARSphereEpisode,
    *,
    stage: Literal["A", "B", "C"],
    request_id: Optional[str],
    record_video: bool,
    max_steps: Optional[int],
) -> PhysicalTrialRolloutResult:
    """Execute and parse one strict Golden physical certification response."""
    resolved_request_id = (
        str(uuid.uuid4()) if request_id is None else str(request_id).strip()
    )
    if not resolved_request_id:
        raise ValueError("physical trial request_id must not be empty")
    try:
        response = scene_server.run_certification_episode(
            episode,
            certification_stage=stage,
            request_id=resolved_request_id,
            record_video=record_video,
            max_steps=max_steps,
        )
    except KeyboardInterrupt as exc:
        try:
            scene_server.abort(resolved_request_id)
        except SceneServerError as abort_error:
            exc.add_note(f"Exact Scene Server abort also failed: {abort_error}")
        raise
    except SceneServerError as exc:
        try:
            scene_server.abort(resolved_request_id)
        except SceneServerError as abort_error:
            raise SceneServerError(
                f"{exc}; exact request abort also failed: {abort_error}"
            ) from exc
        raise
    try:
        pass_gates = response["pass_gates"]
        status = response["certification_status"]
        result = PhysicalTrialRolloutResult(
            request_id=response["request_id"],
            trial_id=response["episode_id"],
            stage=response["certification_stage"],
            scene_server_id=response["scene_server_id"],
            scene_server_git_sha=response["scene_server_git_sha"],
            passed=response["certification_passed"],
            gates=PhysicalTrialGates(
                grasp=pass_gates["A"],
                transport=pass_gates["B"],
                placement=pass_gates["C"],
            ),
            decision_steps=response["decision_steps"],
            termination_reason=response["termination_reason"],
            final_state=response["final_state"],
            grasp_reach_evidence=status["grasp_reach_evidence"],
            grasp_certification_evidence=(response["grasp_certification_evidence"]),
            placement_release_evidence=response.get("placement_release_evidence"),
            failure=response.get("failure"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SceneServerError(
            f"Scene Server returned an invalid physical-trial result: {exc}"
        ) from exc
    if result.trial_id != episode.episode_id or result.stage != stage:
        raise SceneServerError(
            "Physical-trial result does not match the requested episode and stage"
        )
    return result


def rollout_rejects_candidate(
    result: PhysicalTrialRolloutResult,
) -> bool:
    """Return whether one failed rollout is evidence against its exact pose."""
    if result.passed:
        return False
    previous_stage_passed = {
        "A": True,
        "B": result.gates.grasp,
        "C": result.gates.transport,
    }[result.stage]
    failure = result.failure
    if failure is None:
        return previous_stage_passed and result.termination_reason in {
            "environment_done",
            "report_done",
            "diagnostic_checks",
        }
    if failure.type == "PhysicalTrialCandidateError":
        return failure.details["physical_trial_stage"] == result.stage
    if failure.type == "RobotFallError":
        return previous_stage_passed
    return failure.type == "EpisodeStallError" and previous_stage_passed


__all__ = [
    "Dex3GraspAttachmentEvidence",
    "PhysicalTrialFailure",
    "PhysicalTrialFinalState",
    "GraspCertificationEvidence",
    "PhysicalTrialGates",
    "PhysicalTrialRolloutResult",
    "run_grasp_system_trial",
    "run_physical_trial",
    "rollout_rejects_candidate",
]
