"""Typed records retrieved from episode-scoped Robot memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tarsphere.harness.memory.key import KeyMemoryRecord
from tarsphere.harness.memory.semantic import SemanticEntitySummary


@dataclass
class TextStepRecord:
    """Textual memory for one primitive action step.

    A record starts before the primitive is selected and becomes completed only
    after the environment returns execution feedback.  Intermediate actions of
    a multi-action policy proposal therefore receive their own observation and
    feedback rows even though only the first row carries a new policy proposal.
    """

    action_step: int
    policy_step: int | None = None
    # Runtime-authored public note attached to this observation, such as
    # interaction feedback or a depth-confidence warning. This remains
    # authoritative and separate from the agent's thinking.
    observation_note: str = ""
    plan_update: dict[str, Any] | None = None
    proposed_actions: list[dict[str, Any]] = field(default_factory=list)
    executed_action: dict[str, Any] | None = None
    execution_feedback: dict[str, Any] | None = None

    @property
    def completed(self) -> bool:
        return self.execution_feedback is not None


@dataclass(frozen=True)
class HumanDialogueTurn:
    """One public language turn retained for the full episode."""

    speaker: str
    text: str
    action_step: int
    sequence: int = 0
    # Addressed Human for Robot speech; empty for received Human language.
    recipient: str = ""
    # Runtime-confirmed delivery to that recipient, not agreement or task
    # success. None denotes an incoming Human turn already heard by Robot.
    delivered: bool | None = None


@dataclass
class HumanRecord:
    """Episode-long public identity, dialogue, and grounded sightings."""

    speaker: str
    # Asset-fixed public identity description from assets/characters/characters.json.
    # The Agent may use it to bind a visible person to a display name, but
    # cannot overwrite it with a model-authored observation.
    caption: str = ""
    # Asset-fixed public occupation/category from the same character catalog.
    job: str = ""
    public_profile_facts: list[dict[str, str]] = field(default_factory=list)
    # Latest public utterance step. Language has no spatial record because the
    # harness exposes text rather than spatialized microphone observations.
    last_utterance_action_step: int | None = None
    # Historical Human position admitted only while the named body was
    # supported by the current ego RGB-D frame.  It is not refreshed while the
    # Human is outside view and therefore never represents live tracking.
    last_seen_action_step: int | None = None
    last_seen_position_xy: tuple[float, float] | None = None
    last_seen_place: str = ""
    dialogue: list[HumanDialogueTurn] = field(default_factory=list)


@dataclass(frozen=True)
class VisualFrameRecord:
    """All configured camera views captured at one action step."""

    action_step: int
    views: dict[str, Any]


@dataclass(frozen=True)
class GraspedHand:
    """The certified right hand while its grasp remains occupied."""

    # Physical hand whose live retention was confirmed by runtime feedback.
    hand: str
    # Human-readable visual phrase retained in model-facing memory.
    target_label: str
    # Primitive action step at which retention began.
    since_action_step: int
    # Private runtime identifier used to place exactly the retained object.
    object_id: str = ""

    def __post_init__(self) -> None:
        if self.hand != "right":
            raise ValueError("Agent memory accepts only the certified right work hand.")

    def describe(self) -> str:
        return (
            f'right hand has been holding "{self.target_label}" since '
            "action step "
            f"{self.since_action_step}"
        )


@dataclass
class MemoryState:
    """Retrieved public state for one policy inference."""

    actor_id: str
    action_step: int = 0
    text_history: list[TextStepRecord] = field(default_factory=list)
    current_text: TextStepRecord | None = None
    visual_history: list[VisualFrameRecord] = field(default_factory=list)
    camera_gaze_tilt_deg: float = 0.0
    body_posture: str = "standing"
    known_objects: list[Any] = field(default_factory=list)
    # Agent-authored episode facts deliberately retained beyond bounded text
    # history. Runtime-owned ledgers (Humans, holding, grounding, plan) remain
    # separate authorities and must not be duplicated here.
    key_memories: list[KeyMemoryRecord] = field(default_factory=list)
    grasped_hands: list[GraspedHand] = field(default_factory=list)
    humans: list[HumanRecord] = field(default_factory=list)
    # Sparse landmarks explicitly chosen by the Agent. Coordinates and
    # keyframe pixels stay behind semantic-map tools.
    semantic_entities: list[SemanticEntitySummary] = field(default_factory=list)


__all__ = [
    "GraspedHand",
    "HumanDialogueTurn",
    "HumanRecord",
    "MemoryState",
    "TextStepRecord",
    "VisualFrameRecord",
]
