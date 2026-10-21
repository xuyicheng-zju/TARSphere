"""TARSphere-Bench case schema definitions.

These Pydantic models define the declarative specification for a TARSphere
episode: scene, robot, humans, and one causal milestone graph. Optional
component/controller diagnostics are kept explicitly separate from official
evaluation. An episode YAML is loaded into these models, then compiled into an
TARSphere Config at runtime.
"""

from __future__ import annotations

from tarsphere.evaluation.schema import MilestoneReference

import math
import re
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.bench.capability_validation import (
    validate_episode_task_profile,
)
from tarsphere.bench.difficulty import DifficultyFactor
from tarsphere.contracts.human_interaction import (
    HUMAN_COMMUNICATION_RADIUS_M,
)
from tarsphere.infra.control.physical_contracts import (
    HUMAN_BODY_RADIUS_M,
)
from tarsphere.evaluation.schema import (
    EpisodeEvaluationSpec,
    MilestoneCondition,
)


# Episode-authoring clearance between every pair of spawned actors.  This is
# intentionally stronger than the runtime collision disks: initial layouts
# must leave enough room for perception, turning, and natural interaction.
INITIAL_ACTOR_SPAWN_SEPARATION_M = 2.0
INITIAL_REQUEST_FACING_TOLERANCE_RAD = math.radians(15.0)


class ActionLevel(str, Enum):
    """Action evaluation mode."""

    HIGH_LEVEL = "high_level"
    LOW_LEVEL = "low_level"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# Scene & Robot
# ---------------------------------------------------------------------------


class SceneSpec(BaseModel):
    """Physical scene configuration."""

    model_config = ConfigDict(extra="forbid")

    asset_path: str
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mock_setup: Optional[Dict[str, Any]] = None


class RobotSpec(BaseModel):
    """Evaluated embodied robot configuration.

    The evaluated robot uses the locomotion-certified Unitree G1 body with its
    native Dex3 hands. Evaluated grasping is right-hand-only. A retention
    FixedJoint is legal only after measured finger closure and native physical
    contact; the physical setting never creates one. The standing base (pelvis)
    height is ~0.8 m off the ground —
    Isaac's root link is the pelvis, not the foot, so start_position.z must
    match the robot's pelvis height or the robot will spawn inside the floor.
    """

    model_config = ConfigDict(extra="forbid")

    id: Optional[int | str] = None
    name: Optional[str] = None
    model: Optional[str] = None
    task: str = ""
    start_position: Tuple[float, float, float] = (0.0, 0.0, 0.8)
    start_orientation: Optional[Tuple[float, float, float, float]] = None
    action_level: ActionLevel = ActionLevel.HIGH_LEVEL
    controllers: Optional[List[Dict[str, Any]]] = None
    sensors: Optional[List[Dict[str, Any]]] = None

    @model_validator(mode="after")
    def validate_start_pose(self) -> "RobotSpec":
        if not all(math.isfinite(float(value)) for value in self.start_position):
            raise ValueError("Robot start_position must be finite.")
        if self.start_orientation is not None:
            quaternion = tuple(float(value) for value in self.start_orientation)
            if not all(math.isfinite(value) for value in quaternion):
                raise ValueError("Robot start_orientation must be finite.")
            norm = math.sqrt(sum(value * value for value in quaternion))
            if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-4):
                raise ValueError(
                    "Robot start_orientation must be a unit WXYZ quaternion; "
                    f"got norm {norm:.6f}."
                )
        return self

    @property
    def entity_name(self) -> str:
        """Runtime actor name derived from the single authored identity."""

        if self.name:
            return str(self.name)
        if self.id is not None:
            return f"robot_{self.id}"
        return "robot_0"

    @property
    def robot_type(self) -> str:
        """TARSphere robot profile derived from the authored model."""

        return str(self.model or "G1Robot")


# ---------------------------------------------------------------------------
# Human
# ---------------------------------------------------------------------------


class AuthoredInteractionAnswer(BaseModel):
    """One deterministic literal answer selected by an authored intent."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["authored_text"]
    text: str

    @model_validator(mode="after")
    def validate_text(self) -> "AuthoredInteractionAnswer":
        self.text = str(self.text).strip()
        if not self.text:
            raise ValueError("authored_text interaction answer cannot be empty")
        return self


class RelativeDirectionInteractionAnswer(BaseModel):
    """Live bearing to a named anchor, with authored access information."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["relative_direction"]
    target_anchor_id: str = Field(
        description="Episode spatial anchor being located; room directions use its actual entrance, not an interior inspection stance.",
    )
    target_name: str = Field(
        description="Public name of that exact anchor, such as the corridor doorway into the study."
    )
    approach_description: str = Field(
        description="Private authored access/landmark explanation disclosed with the bearing; valid independently of the Robot current heading.",
    )

    @model_validator(mode="after")
    def validate_target(self) -> "RelativeDirectionInteractionAnswer":
        self.target_anchor_id = str(self.target_anchor_id).strip()
        self.target_name = str(self.target_name).strip()
        self.approach_description = self.approach_description.strip()
        if (
            not self.target_anchor_id
            or not self.target_name
            or not self.approach_description
        ):
            raise ValueError(
                "relative_direction answer requires target_anchor_id and "
                "target_name and approach_description"
            )
        return self


InteractionAnswer = Annotated[
    Union[AuthoredInteractionAnswer, RelativeDirectionInteractionAnswer],
    Field(discriminator="kind"),
]


class InteractionIntentSpec(BaseModel):
    """One semantic question scope and its deterministic episode answer."""

    model_config = ConfigDict(extra="forbid")

    intent_id: str
    description: str = Field(
        description=(
            "Semantic information need covered by this intent, not a canonical "
            "question string. Natural-language routing must accept direct, "
            "indirect, and elliptical paraphrases within this scope."
        ),
    )
    public: bool = Field(
        default=True,
        description="Expose this intent in the Agent tool prompt and allow Agent-supplied exact IDs. Private topics can be selected only through authorized natural-language routing.",
    )
    answer: InteractionAnswer
    eligible_human_ids: List[str] = Field(default_factory=list)
    preconditions: tuple[MilestoneReference, ...] = Field(
        default=(),
        description="Required causal facts before this private answer may be delivered.",
    )

    @model_validator(mode="after")
    def validate_intent(self) -> "InteractionIntentSpec":
        self.intent_id = str(self.intent_id).strip()
        if not self.intent_id:
            raise ValueError("interaction intent_id cannot be empty")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.intent_id):
            raise ValueError("interaction intent_id must be a lowercase snake_case id")
        self.description = str(self.description).strip()
        if not self.description:
            raise ValueError("interaction description cannot be empty")
        eligible = [str(human_id).strip() for human_id in self.eligible_human_ids]
        if any(not human_id for human_id in eligible):
            raise ValueError("eligible_human_ids cannot contain empty identities")
        if len(eligible) != len(set(eligible)):
            raise ValueError("eligible_human_ids must be unique")
        self.eligible_human_ids = eligible
        return self


class EpisodeInteractionSpec(BaseModel):
    """Optional deterministic question intents enabled for this episode."""

    model_config = ConfigDict(extra="forbid")

    intents: List[InteractionIntentSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_intents(self) -> "EpisodeInteractionSpec":
        intent_ids = [intent.intent_id for intent in self.intents]
        if len(intent_ids) != len(set(intent_ids)):
            raise ValueError("interaction intent_id values must be unique")
        return self


class HumanFact(BaseModel):
    """One authored fact owned by a named Human."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    fact_id: str = Field(
        pattern=r"^[a-z][a-z0-9_]*$", description="Human-local fact identifier."
    )
    text: str = Field(
        min_length=1,
        description="Complete authored fact; no evaluator identifiers or hidden coordinates.",
    )


class HumanSpec(BaseModel):
    """Specification for a benchmark human actor.

    Attributes:
        id: Globally stable human-readable identity used everywhere in the
            episode, including prompts, runtime state, and evaluator bindings.
        name: Runtime-derived display alias. It must equal ``id``; episode YAML
            must not author a second identity field.
        role: Narrative role (target, staff, bystander, family).
        asset: IsaacPeople character asset name.
        character_name: omni.anim.people character name (None = random).
        start_position: Initial world position.
        persona: Personality description injected into LLM prompt.
        goals: What this human wants (drives LLM decisions).
        public_profile_facts: Episode-specific facts explicitly known to the Agent.
        answer_facts: Private facts this Human may disclose in supplementary answers.
        appearance_attributes_gt: Optional atomic appearance goldens when the
            human has no characters.json asset entry (e.g. mock).
        scene_person_prim_path: Exact authored GRScene person prim. When set,
            the runtime binds this already-rendered person into the Human FSM
            instead of spawning a duplicate IsaacPeople character.
        static: If True, the human stands at start_position for the whole
            episode and must not be given NavMesh locomotion.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    name: Optional[str] = None
    role: str = "target"
    asset: Optional[str] = None
    character_name: Optional[str] = None
    start_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    persona: str = ""
    goals: List[str] = Field(default_factory=list)
    start_yaw: float = 0.0  # initial facing direction in radians (0 = +X, π = -X)
    public_profile_facts: List[HumanFact] = Field(
        default_factory=list,
        description="Public prior knowledge rendered in the Agent Human directory.",
    )
    answer_facts: List[HumanFact] = Field(
        default_factory=list,
        description="Facts authorized for supplementary generated replies, never scoring receipts. Do not put gated answers here.",
    )
    appearance_attributes_gt: List[str] = Field(default_factory=list)
    public_description: str = Field(
        default="",
        description=(
            "Public episode-specific role/location description. For catalog Humans it "
            "supplements the fixed visual caption; for native Humans it supplies the "
            "complete reviewed description. Never include private answers or future events."
        ),
    )
    public_job: str = Field(
        default="",
        description=(
            "Public occupation or responsibility in this episode, taking precedence over "
            "a catalog occupation when supplied. Native Humans must declare it."
        ),
    )
    scene_person_prim_path: Optional[str] = None
    scene_person_bounds_min_m: Optional[Tuple[float, float, float]] = None
    scene_person_bounds_max_m: Optional[Tuple[float, float, float]] = None
    static: bool = False

    @model_validator(mode="after")
    def validate_named_identity(self) -> "HumanSpec":
        for facts in (self.public_profile_facts, self.answer_facts):
            identifiers = [fact.fact_id for fact in facts]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(
                    "Human fact identifiers must be unique within their disclosure channel"
                )
        identity = str(self.id).strip()
        if not identity:
            raise ValueError("Human id must be a non-empty global name.")
        if identity.casefold().startswith("human_") and identity[6:].isdigit():
            raise ValueError(
                "Human id must be the character name, not a numbered alias: "
                f"{identity!r}"
            )
        if self.name is not None and str(self.name).strip() != identity:
            raise ValueError(
                f"Human name must equal id; got id={identity!r}, name={self.name!r}"
            )
        self.id = identity
        self.name = identity
        if self.character_name and self.asset and self.character_name != self.asset:
            raise ValueError(
                "Human character_name is runtime-derived and must match asset; "
                f"got {self.character_name!r} vs {self.asset!r}"
            )
        if self.character_name is None and self.asset:
            self.character_name = str(self.asset)
        if self.scene_person_prim_path and (self.asset or self.character_name):
            raise ValueError(
                "A scene-bound Human cannot also declare a spawned "
                "IsaacPeople asset/character_name."
            )
        if self.asset:
            from tarsphere.humans.character_catalog import character_record

            fixed_name = str(character_record(str(self.asset))["name"])
            if identity != fixed_name:
                raise ValueError(
                    "Human id must equal the global character-catalog name; "
                    f"got {identity!r} vs {fixed_name!r} for asset {self.asset!r}"
                )
        if self.scene_person_prim_path:
            path = str(self.scene_person_prim_path).strip()
            from tarsphere.contracts.scene_humans import (
                is_audited_scene_human_prim,
            )

            if not is_audited_scene_human_prim(path):
                raise ValueError(
                    "scene_person_prim_path must bind an exact GRScene person "
                    f"prim or an explicitly audited source-taxonomy exception; got {path!r}"
                )
            if not self.static:
                raise ValueError("A scene-bound Human must be static.")
            if not self.public_description.strip() or not self.public_job.strip():
                raise ValueError(
                    "A scene-bound Human requires public_description and "
                    "public_job so the Agent can ground the authored identity."
                )
            bounds = (
                self.scene_person_bounds_min_m,
                self.scene_person_bounds_max_m,
            )
            if any(value is None for value in bounds):
                raise ValueError(
                    "A scene-bound Human requires exact min/max bounds from "
                    "the pinned scene geometry."
                )
            lower, upper = bounds
            assert lower is not None and upper is not None
            if not all(math.isfinite(float(value)) for value in (*lower, *upper)):
                raise ValueError("Scene-bound Human bounds must be finite.")
            if any(float(lower[index]) >= float(upper[index]) for index in range(3)):
                raise ValueError("Scene-bound Human bounds must have positive extent.")
            center_xy = tuple(
                (float(lower[index]) + float(upper[index])) * 0.5 for index in range(2)
            )
            if math.dist(center_xy, tuple(map(float, self.start_position[:2]))) > 0.05:
                raise ValueError(
                    "Scene-bound Human start_position must equal the authored "
                    "person-bounds XY center within 0.05 m."
                )
        elif any(
            value is not None
            for value in (
                self.scene_person_bounds_min_m,
                self.scene_person_bounds_max_m,
            )
        ):
            raise ValueError(
                "scene_person_bounds_min_m/max_m require scene_person_prim_path."
            )
        if not all(math.isfinite(float(value)) for value in self.start_position):
            raise ValueError(f"Human {identity} start_position must be finite.")
        if not math.isfinite(float(self.start_yaw)):
            raise ValueError(f"Human {identity} start_yaw must be finite.")
        return self


# ---------------------------------------------------------------------------
# Component diagnostics
# ---------------------------------------------------------------------------


class DiagnosticCheck(BaseModel):
    """A simulator check used only by component/controller diagnostics."""

    model_config = ConfigDict(extra="forbid")

    type: str
    params: Dict[str, Any] = Field(default_factory=dict)


class DiagnosticSpec(BaseModel):
    """Non-official diagnostic completion checks (all must be satisfied)."""

    model_config = ConfigDict(extra="forbid")

    description: Optional[str] = None
    checks: List[DiagnosticCheck] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Action Specification
# ---------------------------------------------------------------------------

AGENT_ACTION_FAMILY_VOCAB = {
    "cognitive": frozenset({"wait", "tilt_camera"}),
    "perceptual": frozenset(
        {
            "locate_object",
            "remember_entity",
            "inspect_entity",
        }
    ),
    "communicative": frozenset(
        {
            "ask",
            "speak",
            "report_done",
            "handover_object",
        }
    ),
    "physical": frozenset(
        {
            "move_body",
            "move_forward",
            "move_backward",
            "move_left",
            "move_right",
            "turn_left",
            "turn_right",
            "crouch",
            "stand_up",
            "move_end_effector",
            "grasp_object",
        }
    ),
}


class ActionSpec(BaseModel):
    """Defines the action space for the evaluated robot.

    One list per family of decisions the episode admits. The runner rejects any
    decision whose action is absent from these lists, so an episode that omits
    a family withholds that capability rather than merely discouraging it.
    """

    model_config = ConfigDict(extra="forbid")

    cognitive: List[str] = Field(
        default_factory=lambda: ["wait", "tilt_camera"],
        description=(
            "Stationary timing and camera-gaze actions available to the agent. "
            "Camera frames themselves arrive automatically on every policy turn."
        ),
    )
    perceptual: List[str] = Field(
        default_factory=list,
        description=(
            "Read-only structured visual grounding tools explicitly required "
            "by this episode."
        ),
    )
    communicative: List[str] = Field(
        default_factory=lambda: ["ask", "speak"],
        description="Decisions whose whole effect is on the dialogue.",
    )
    physical: List[str] = Field(
        default_factory=lambda: ["move_body"],
        description=(
            "Locomotion, posture, and manipulation decisions this episode "
            "admits. The default grants continuous body-frame locomotion and "
            "camera observations arrive automatically. PointGoal navigation "
            "is supplied by the default harness, "
            "while manipulation must be declared by an episode with a reachable "
            "target."
        ),
    )

    @model_validator(mode="after")
    def validate_current_action_vocabulary(self) -> "ActionSpec":
        """Reject removed, unknown, misplaced, and duplicate agent actions."""

        for family, allowed in AGENT_ACTION_FAMILY_VOCAB.items():
            actions = list(getattr(self, family))
            if len(actions) != len(set(actions)):
                raise ValueError(
                    f"agent_action_space.{family} contains duplicate actions: "
                    f"{actions!r}"
                )
            invalid = sorted(set(actions) - allowed)
            if invalid:
                raise ValueError(
                    f"agent_action_space.{family} contains unavailable actions "
                    f"{invalid!r}; allowed actions are {sorted(allowed)!r}"
                )
        return self


class RuntimeSpec(BaseModel):
    """Episode runtime cadence and optional benchmark-wide budgets."""

    model_config = ConfigDict(extra="forbid")

    # ``None`` means unbounded. A decision budget and a physics budget remain
    # separate interfaces because cognition count and simulated time answer
    # different experimental questions; neither is silently derived from the
    # other.
    max_decision_steps: Optional[int] = Field(
        default=None,
        description=(
            "Maximum model decisions, not executed actions. One serial action "
            "block counts once and may drain at the limit. Null is unbounded; "
            "the formal worker supplies the fixed protocol limit."
        ),
    )
    max_physics_steps: Optional[int] = None

    @model_validator(mode="after")
    def validate_runtime_budget(self) -> "RuntimeSpec":
        for field_name in ("max_decision_steps", "max_physics_steps"):
            value = getattr(self, field_name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"runtime.{field_name} must be positive or null.")
        return self


class AuthoringGate(str, Enum):
    """Monotonic evidence gates for candidate promotion."""

    AUTHORED = "authored"
    SCHEMA_VALID = "schema_valid"
    GEOMETRY_VALID = "geometry_valid"
    OVERLAY_REVIEWED = "overlay_reviewed"
    ORACLE_VALID = "oracle_valid"
    RUNTIME_VALID = "runtime_valid"
    GOLDEN = "golden"


AUTHORING_GATE_ORDER = tuple(AuthoringGate)


class AuthoringEvidenceRecord(BaseModel):
    """One immutable external artifact bound to a promotion gate."""

    model_config = ConfigDict(extra="forbid")

    artifact_path: str
    artifact_sha256: str
    episode_id: str
    scene_id: str
    git_sha: str
    gate: AuthoringGate
    passed: Literal[True]

    @model_validator(mode="after")
    def validate_evidence_identity(self) -> "AuthoringEvidenceRecord":
        for field_name in (
            "artifact_path",
            "episode_id",
            "scene_id",
            "git_sha",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"authoring evidence {field_name} must be non-empty")
        digest = str(self.artifact_sha256).strip().lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                "authoring evidence artifact_sha256 must be lowercase SHA-256"
            )
        self.artifact_sha256 = digest
        git_sha = str(self.git_sha).strip().lower()
        if len(git_sha) != 40 or any(
            character not in "0123456789abcdef" for character in git_sha
        ):
            raise ValueError(
                "authoring evidence git_sha must be a full lowercase git SHA"
            )
        self.git_sha = git_sha
        return self


class EpisodeAuthoringSpec(BaseModel):
    """Publication evidence owned by the coding-agent authoring workflow."""

    model_config = ConfigDict(extra="forbid")

    achieved_gates: List[AuthoringGate]
    difficulty_factors: List[DifficultyFactor] = Field(default_factory=list)
    route_overlay: str
    external_evidence: Dict[
        AuthoringGate,
        AuthoringEvidenceRecord,
    ] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_monotonic_gates(self) -> "EpisodeAuthoringSpec":
        if len(self.difficulty_factors) != len(set(self.difficulty_factors)):
            raise ValueError("authoring.difficulty_factors cannot contain duplicates")
        expected = list(AUTHORING_GATE_ORDER[: len(self.achieved_gates)])
        if self.achieved_gates != expected:
            raise ValueError(
                "authoring.achieved_gates must be an exact monotonic prefix; "
                f"expected {[gate.value for gate in expected]!r}"
            )
        for gate in self.achieved_gates:
            if (
                gate
                in {
                    AuthoringGate.ORACLE_VALID,
                    AuthoringGate.RUNTIME_VALID,
                    AuthoringGate.GOLDEN,
                }
                and gate not in self.external_evidence
            ):
                raise ValueError(
                    f"authoring.external_evidence requires an artifact for {gate.value}"
                )
        for gate, record in self.external_evidence.items():
            if record.gate is not gate:
                raise ValueError(
                    "authoring evidence map key must equal record.gate; "
                    f"got {gate.value!r} and {record.gate.value!r}"
                )
            if gate not in self.achieved_gates:
                raise ValueError(
                    f"authoring evidence for unachieved gate {gate.value!r}"
                )
        return self


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


class TARSphereEpisode(BaseModel):
    """Complete specification for one TARSphere episode.

    This is the top-level model that a YAML episode file is parsed into.
    It is then compiled into an TARSphere ``Config`` at runtime.

    Task-assignment boundary: at evaluation time the robot receives only
    ``system_prompt`` and public runtime events. Human episodes disclose the
    concrete task through Human speech or observable behaviour; Human-free
    component episodes may place their concrete task in ``system_prompt``.
    ``robot.task`` is kept for developer/evaluator reference and must NOT be
    injected into the agent's prompt.
    """

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    title: Optional[str] = None
    description: Optional[str] = None
    # Public task instruction injected into the agent at every act() call.
    # Human episodes normally keep it generic because the concrete request
    # arrives through dialogue; Human-free component episodes state the task
    # here. Benchmark-private world_knowledge is never interpolated into it.
    system_prompt: str = (
        "Help people complete their public requests in this shared physical "
        "environment. Navigate proactively when needed. Human speech may reach "
        f"you within {HUMAN_COMMUNICATION_RADIUS_M:g} m without camera "
        "visibility, but you may address a Human only when current RGB shows "
        f"them and they are within {HUMAN_COMMUNICATION_RADIUS_M:g} m. Keep about "
        "1 m of personal space. Use only the public observations and tools "
        "explicitly exposed by the Agent harness."
    )

    # Scene
    scene: SceneSpec

    # Robot
    robot: RobotSpec

    # Humans
    humans: List[HumanSpec] = Field(default_factory=list)

    # Deterministic, episode-specific answers available through ``ask``.
    # These are optional public affordances, not scored milestones. An empty
    # eligible_human_ids list means any Human in this episode may provide the
    # shared fact (for example, the direction of a public room).
    interaction: EpisodeInteractionSpec = Field(
        default_factory=EpisodeInteractionSpec,
    )

    agent_action_space: Optional[ActionSpec] = Field(
        default=None,
        description=(
            "Restricted action surface for non-benchmark component diagnostics. "
            "Official evaluated episodes must omit this field and use the one "
            "task-agnostic Agent harness action surface."
        ),
    )

    # Non-benchmark component/controller checks. These never define official
    # task progress or benchmark success; that authority belongs exclusively
    # to ``evaluation.milestones``.
    diagnostics: Optional[DiagnosticSpec] = None
    # Official causal-progress and failure-attribution contract. Component and
    # controller-diagnostic episodes may omit it; an official TARSphere rollout
    # may not.
    evaluation: Optional[EpisodeEvaluationSpec] = None
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)

    # Candidate/golden publication evidence. Component diagnostics may omit
    # this, while the strict benchmark-candidate gate requires it.
    authoring: Optional[EpisodeAuthoringSpec] = None

    # Benchmark-private scene/task annotations consumed by human actors and
    # evaluators. This must not be exposed to the evaluated robot agent.
    world_knowledge: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Metadata
    tags: Optional[List[str]] = None

    def model_post_init(self, __context: Any) -> None:
        """Reject invalid starts, task graphs, and reporting coordinates."""
        self._validate_human_identities()
        self._validate_interaction_bindings()
        self._validate_initial_actor_separation()
        self._validate_human_spatial_separation()
        if self.evaluation is not None:
            if self.agent_action_space is not None:
                raise ValueError(
                    "Official evaluated episodes must omit agent_action_space; "
                    "the Agent harness owns one task-agnostic action surface."
                )
            self._validate_evaluation_bindings()
            self._validate_task_evaluation_contract()
            if self.authoring is not None or self.metadata.get("scene_context"):
                self._validate_initial_request_facing()
                validate_episode_task_profile(self)
        self._validate_authoring_evidence_bindings()

    def _validate_authoring_evidence_bindings(self) -> None:
        if self.authoring is None:
            return
        scene_id = str(self.world_knowledge.get("scene_id") or "").strip()
        for gate, record in self.authoring.external_evidence.items():
            if record.episode_id != self.episode_id:
                raise ValueError(
                    f"authoring evidence {gate.value} is bound to episode "
                    f"{record.episode_id!r}, expected {self.episode_id!r}"
                )
            if record.scene_id != scene_id:
                raise ValueError(
                    f"authoring evidence {gate.value} is bound to scene "
                    f"{record.scene_id!r}, expected {scene_id!r}"
                )

    def _validate_human_identities(self) -> None:
        identities = [str(human.id) for human in self.humans]
        if len(identities) != len(set(identities)):
            raise ValueError("Episode Human ids must be globally unique names.")

    def _validate_interaction_bindings(self) -> None:
        """Resolve public intent sources, scored bindings, and answers."""

        humans_by_id = {str(human.id): human for human in self.humans}
        human_ids = set(humans_by_id)
        anchors = self.world_knowledge.get("spatial_anchors") or {}
        if not isinstance(anchors, dict):
            raise ValueError("world_knowledge.spatial_anchors must be a mapping")
        for intent in self.interaction.intents:
            unknown_humans = set(intent.eligible_human_ids) - human_ids
            if unknown_humans:
                raise ValueError(
                    f"Interaction intent {intent.intent_id!r} references "
                    f"unknown eligible Humans: {sorted(unknown_humans)}"
                )
            answer = intent.answer
            if not isinstance(answer, RelativeDirectionInteractionAnswer):
                continue
            anchor = anchors.get(answer.target_anchor_id)
            if not (
                isinstance(anchor, (list, tuple))
                and len(anchor) >= 2
                and all(math.isfinite(float(value)) for value in anchor[:2])
            ):
                raise ValueError(
                    f"Interaction intent {intent.intent_id!r} relative "
                    f"direction anchor {answer.target_anchor_id!r} must "
                    "resolve to a finite XY spatial anchor"
                )
        if self.evaluation is None:
            return
        intents = {intent.intent_id: intent for intent in self.interaction.intents}
        for milestone in self.evaluation.milestones:
            for action in milestone.event.human_actions:
                human_id = str(action.get("human") or "")
                human = humans_by_id.get(human_id)
                if (
                    human is not None
                    and human.scene_person_prim_path
                    and str(action.get("command") or "")
                    not in {
                        "speak",
                        "ask",
                        "report_done",
                        "idle",
                        "listen",
                        "perform_task",
                    }
                ):
                    raise ValueError(
                        f"Scene-bound Human {human_id!r} cannot execute body "
                        f"action {action.get('command')!r}."
                    )
                if action.get("command") == "perform_task":
                    from tarsphere.contracts.human_task import (
                        HumanTaskAction,
                        human_task_stance_error,
                    )

                    task = HumanTaskAction.model_validate(action)
                    if human is None:
                        raise ValueError("Human task references an unknown Human")
                    anchor = anchors.get(task.target_anchor)
                    if human_task_stance_error(
                        anchor, anchor, moving=False, stale=False
                    ):
                        raise ValueError(
                            "Human task requires a finite XYZ spatial anchor"
                        )
                    if human.scene_person_prim_path and human_task_stance_error(
                        human.start_position,
                        anchor,
                        moving=False,
                        stale=False,
                    ):
                        raise ValueError(
                            "Scene-bound Human task must use its actual static operation stance"
                        )
                    if task.partner_human_id and (
                        task.partner_human_id not in human_ids
                        or task.partner_human_id == human_id
                    ):
                        raise ValueError(
                            "Human task partner must be a different declared Human"
                        )
            if milestone.outcome.evaluator not in {
                "agent_interaction_selected",
                "information_received",
            }:
                continue
            truths = [
                milestone.outcome.ground_truth,
                *(
                    update.outcome_updates[milestone.milestone_id]
                    for event in self.evaluation.milestones
                    for update in event.event.task_updates
                    if milestone.milestone_id in update.outcome_updates
                ),
            ]
            for truth in truths:
                intent_id = str(truth["intent_id"])
                intent = intents.get(intent_id)
                if intent is None:
                    raise ValueError(
                        f"Milestone {milestone.milestone_id!r} references "
                        f"undeclared interaction intent {intent_id!r}"
                    )
                human_id = str(truth["human_id"])
                if (
                    intent.eligible_human_ids
                    and human_id not in intent.eligible_human_ids
                ):
                    raise ValueError(
                        f"Milestone {milestone.milestone_id!r} scores Human "
                        f"{human_id!r}, but intent {intent_id!r} is eligible only "
                        f"for {intent.eligible_human_ids!r}"
                    )

    def _validate_evaluation_bindings(self) -> None:
        """Prove every graph binding resolves to an episode-owned entity."""

        assert self.evaluation is not None
        human_ids = {str(human.id) for human in self.humans}
        object_ids = {
            str(item["object_id"])
            for item in (self.world_knowledge.get("objects") or ())
            if isinstance(item, dict) and item.get("object_id") is not None
        }

        def validate_arguments(
            owner: str,
            arguments: Dict[str, Any],
        ) -> None:
            human_id = arguments.get("human_id")
            if human_id is not None and str(human_id) not in human_ids:
                raise ValueError(f"{owner} references unknown Human id {human_id!r}")
            for argument_name in ("object_id", "destination_object_id"):
                object_id = arguments.get(argument_name)
                if object_id is not None and str(object_id) not in object_ids:
                    raise ValueError(
                        f"{owner} references unknown object id {object_id!r}"
                    )
            referenced_object_ids = arguments.get("object_ids") or ()
            unknown_object_ids = [
                str(value)
                for value in referenced_object_ids
                if str(value) not in object_ids
            ]
            if unknown_object_ids:
                raise ValueError(
                    f"{owner} references unknown object ids {unknown_object_ids!r}"
                )

        for milestone in self.evaluation.milestones:
            for action in milestone.event.human_actions:
                human_id = action.get("human")
                if human_id is None or str(human_id) not in human_ids:
                    raise ValueError(
                        f"evaluation milestone {milestone.milestone_id!r} "
                        "event references unknown Human id "
                        f"{human_id!r}"
                    )
                if str(action.get("command") or "") == "speak":
                    message = str(action.get("message") or "")
                    if re.search(
                        rf"\bfacing\s+{re.escape(str(human_id))}\b",
                        message,
                        flags=re.IGNORECASE,
                    ):
                        raise ValueError(
                            f"evaluation milestone {milestone.milestone_id!r} "
                            f"has {human_id!r} refer to themself in route speech; "
                            'use "facing me"'
                        )
            for update in milestone.event.task_updates:
                for outcome_id, truth in update.outcome_updates.items():
                    validate_arguments(f"task update for {outcome_id!r}", truth)
                if str(update.source_human_id) not in human_ids:
                    raise ValueError(
                        f"evaluation milestone {milestone.milestone_id!r} "
                        "task update references unknown Human id "
                        f"{update.source_human_id!r}"
                    )
            for condition in milestone.preconditions:
                if not isinstance(condition, MilestoneCondition):
                    continue
                validate_arguments(
                    f"evaluation condition {condition.condition_id!r}",
                    condition.arguments,
                )
            validate_arguments(
                f"evaluation milestone {milestone.milestone_id!r} reaction",
                milestone.outcome.ground_truth,
            )

    def _validate_task_evaluation_contract(self) -> None:
        """Require official graphs to prove their declared world/task results."""

        assert self.evaluation is not None
        if self.authoring is None and not self.metadata.get("scene_context"):
            return
        tool_grounding_nodes = [
            milestone.milestone_id
            for milestone in self.evaluation.milestones
            if milestone.outcome.evaluator == "object_localized"
        ]
        if tool_grounding_nodes:
            raise ValueError(
                "object_localized is private grasp/perception trace evidence, "
                "not a required benchmark milestone; score the resulting world "
                f"or task state instead: {tool_grounding_nodes!r}"
            )

        held: set[str] = set()
        for milestone in self.evaluation.milestones:
            reaction = milestone.outcome
            object_id = str(reaction.ground_truth.get("object_id") or "")
            if reaction.evaluator == "robot_holding_object" and object_id:
                held.add(object_id)
            elif (
                reaction.evaluator
                in {
                    "handover_accepted",
                    "object_supported_by",
                }
                and object_id
                and object_id not in held
            ):
                raise ValueError(
                    f"physical closure for {object_id!r} must follow a "
                    "successful robot_holding_object reaction"
                )

    def _validate_initial_actor_separation(self) -> None:
        """Require every Robot/Human spawn pair to be strictly over 2 m apart."""

        robot_xy = self.robot.start_position[:2]
        for human in self.humans:
            human_xy = human.start_position[:2]
            distance = math.hypot(
                float(robot_xy[0]) - float(human_xy[0]),
                float(robot_xy[1]) - float(human_xy[1]),
            )
            if distance <= INITIAL_ACTOR_SPAWN_SEPARATION_M:
                raise ValueError(
                    f"Initial actor spawn separation: robot and Human "
                    f"{human.id!r} are {distance:.2f}m apart; require > "
                    f"{INITIAL_ACTOR_SPAWN_SEPARATION_M:.2f}m. Move the episode "
                    "actors to distinct real route/object-derived points."
                )

    def _validate_initial_request_facing(self) -> None:
        """Require the initial task requester and Robot to face each other."""

        assert self.evaluation is not None
        task_requests = [
            milestone
            for milestone in self.evaluation.milestones
            if str(getattr(milestone.event.type, "value", milestone.event.type))
            == "task_requested"
        ]
        if not task_requests:
            return
        first_request = task_requests[0]
        requester_ids = {
            str(action.get("human") or "").strip()
            for action in first_request.event.human_actions
            if str(action.get("command") or "") in {"ask", "speak"}
        }
        requester_ids.discard("")
        if not requester_ids:
            raise ValueError(
                "Initial task_requested event has no speaking Human requester."
            )
        if self.robot.start_orientation is None:
            raise ValueError(
                "Initial Human request requires an authored Robot orientation."
            )

        w, x, y, z = (float(value) for value in self.robot.start_orientation)
        robot_yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        robot_xy = tuple(float(value) for value in self.robot.start_position[:2])
        humans = {str(human.id): human for human in self.humans}

        def angular_error(actual: float, expected: float) -> float:
            return abs(
                (float(actual) - float(expected) + math.pi) % (2.0 * math.pi) - math.pi
            )

        for requester_id in sorted(requester_ids):
            requester = humans[requester_id]
            human_xy = tuple(float(value) for value in requester.start_position[:2])
            opening_distance = math.dist(robot_xy, human_xy)
            if opening_distance > HUMAN_COMMUNICATION_RADIUS_M:
                raise ValueError(
                    f"Initial requester range: Robot and Human "
                    f"{requester_id!r} are {opening_distance:.2f}m apart; "
                    f"require <= {HUMAN_COMMUNICATION_RADIUS_M:g}m for the "
                    "opening task request."
                )
            expected_robot_yaw = math.atan2(
                human_xy[1] - robot_xy[1],
                human_xy[0] - robot_xy[0],
            )
            expected_human_yaw = math.atan2(
                robot_xy[1] - human_xy[1],
                robot_xy[0] - human_xy[0],
            )
            robot_error = angular_error(robot_yaw, expected_robot_yaw)
            human_error = angular_error(
                float(requester.start_yaw),
                expected_human_yaw,
            )
            human_facing_required = not bool(requester.scene_person_prim_path)
            if robot_error > INITIAL_REQUEST_FACING_TOLERANCE_RAD or (
                human_facing_required
                and human_error > INITIAL_REQUEST_FACING_TOLERANCE_RAD
            ):
                requirement = (
                    f"Robot must face scene-bound Human {requester_id!r}"
                    if not human_facing_required
                    else f"Robot and Human {requester_id!r} must face each other"
                )
                raise ValueError(
                    f"Initial requester facing: {requirement} within 15 degrees; "
                    f"errors are Robot={math.degrees(robot_error):.1f} degrees "
                    f"and Human={math.degrees(human_error):.1f} degrees."
                )

    def _validate_human_spatial_separation(self) -> None:
        """Reject authored Human poses whose physical body disks overlap."""

        runtime_minimum = 2.0 * HUMAN_BODY_RADIUS_M
        positions = {
            human.id: tuple(float(value) for value in human.start_position[:2])
            for human in self.humans
        }

        def check(label: str, *, minimum: float, strict: bool = False) -> None:
            identities = tuple(positions)
            for index, first in enumerate(identities):
                for second in identities[index + 1 :]:
                    distance = math.dist(positions[first], positions[second])
                    invalid = distance <= minimum if strict else distance < minimum
                    if invalid:
                        comparator = ">" if strict else ">="
                        raise ValueError(
                            f"Human spatial separation {label}: {first!r} and "
                            f"{second!r} are {distance:.2f}m apart; require "
                            f"{comparator} {minimum:.2f}m center separation"
                        )

        check(
            "at episode start",
            minimum=INITIAL_ACTOR_SPAWN_SEPARATION_M,
            strict=True,
        )
        if self.evaluation is None:
            return
        for milestone in self.evaluation.milestones:
            for action in milestone.event.human_actions:
                if str(action.get("command") or "") != "goto":
                    continue
                human_id = str(action.get("human") or "")
                target = action.get("target")
                if (
                    human_id in positions
                    and isinstance(target, (list, tuple))
                    and len(target) >= 2
                ):
                    positions[human_id] = (
                        float(target[0]),
                        float(target[1]),
                    )
                    check(
                        f"after milestone {milestone.milestone_id!r}",
                        minimum=runtime_minimum,
                    )
