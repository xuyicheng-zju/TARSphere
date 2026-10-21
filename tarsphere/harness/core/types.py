"""Core type definitions for the robot agent framework.

Aligned with ACTIO's tool type system. These types form the vocabulary for
tool interactions, agent observations, and module contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


# ---------------------------------------------------------------------------
# Tool types (aligned with actio-core/tool_types.py)
# ---------------------------------------------------------------------------

ToolRiskLevel = Literal["low", "medium", "high"]
ToolExecutionKind = Literal["primitive", "information", "bookkeeping"]


class DeferredToolEffect(Protocol):
    """A reversible non-physical effect staged during tool resolution.

    Handlers must not mutate durable Agent state while an action block is
    merely being validated.  The kernel commits these effects only after
    every call in the block has resolved successfully, and rolls committed
    effects back if a later commit fails.
    """

    def commit(self) -> None:
        """Apply the staged effect."""

    def rollback(self) -> None:
        """Restore the exact state that preceded :meth:`commit`."""


class CompositeDeferredToolEffect:
    """Commit several reversible state updates as one tool-owned effect."""

    def __init__(self, *effects: DeferredToolEffect) -> None:
        if not effects:
            raise ValueError("CompositeDeferredToolEffect requires an effect.")
        self._effects = tuple(effects)
        self._attempted: list[DeferredToolEffect] = []
        self._committed = False

    def commit(self) -> None:
        if self._committed or self._attempted:
            raise RuntimeError("Composite deferred effect was committed twice.")
        try:
            for effect in self._effects:
                # Include the current effect so its rollback can undo a
                # mutation made before commit itself raises.
                self._attempted.append(effect)
                effect.commit()
        except Exception:
            self.rollback()
            raise
        self._committed = True

    def rollback(self) -> None:
        rollback_errors: list[Exception] = []
        for effect in reversed(self._attempted):
            try:
                effect.rollback()
            except Exception as exc:  # noqa: BLE001 - preserve all rollback attempts
                rollback_errors.append(exc)
        self._attempted.clear()
        self._committed = False
        if rollback_errors:
            raise RuntimeError(
                "Composite deferred effect rollback failed: "
                + "; ".join(f"{type(exc).__name__}: {exc}" for exc in rollback_errors)
            )


@dataclass(frozen=True)
class ToolSpec:
    """Model-visible declaration of one atomic callable capability."""

    # Stable identifier the model writes as `{"name": ...}`. Must match the
    # name its handler is registered under.
    name: str
    # One sentence: what this call does. Rendered beside the name, so it is the
    # only line guaranteed to be read when the model is scanning the list.
    summary: str
    # JSON Schema for ``args``, and the sole authority for this call's
    # parameters: the prompt's parameter list is derived from it, so every
    # property needs its own ``description``, and ``required`` here is what the
    # model is told is mandatory.
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    # Runtime risk tier. Governs handler-side gating, not prompt text.
    risk_level: ToolRiskLevel = "medium"
    # Semantic tags for filtering a registry (e.g. 'locomotion', 'perception').
    tags: frozenset[str] = field(default_factory=frozenset)
    # Runtime classification: information tools return facts/images, bookkeeping
    # tools write semantic state, primitives move the robot. Every kind is an
    # ordinary turn whose result becomes the next Current observation.
    execution_kind: ToolExecutionKind = "primitive"


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation requested by the VLM."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""


def tool_turn_action(tool: str) -> dict[str, Any]:
    """Runtime payload of a tool that finished inside its handler.

    Observation, bookkeeping, failure, already-at-goal, and no-route results
    still own one action: the tool that ran. That identity is not a
    controller command. The runner observes the current state and does not
    step physics.
    """

    name = str(tool or "").strip()
    if not name:
        raise ValueError("tool_turn_action requires the tool name")
    return {"action": name}


@dataclass(frozen=True)
class ToolResult:
    """Result envelope for one tool invocation.

    Every completed call owns a runtime payload: exactly one of ``action`` or
    ``actions``. ``action`` is the tool's own payload — a controller command
    when the environment must move the body, or ``tool_turn_action(name)``
    when the handler already finished. It is never omitted and never a forged
    wait. Failures use the same identity payload; the error text is the turn
    result and becomes the next Current observation.

    ``images`` carries zero or more base64 data URLs (e.g.
    ``data:image/jpeg;base64,...``) that the turn loop forwards to the VLM as
    ``image_url`` content blocks. Empty by default so existing text-only tools
    are unaffected. Observation tools can use this field to deliver visual
    evidence the model can actually inspect alongside the text result.
    """

    call_id: str
    name: str
    content: str = ""
    action: dict[str, Any] | None = None
    # A model-visible macro may resolve to several ordered physical
    # primitives while still costing one policy call. Most tools use
    # ``action``; ``actions`` is reserved for an ordered, handler-owned scan or
    # manipulation sequence and must never be supplied together with it.
    actions: tuple[dict[str, Any], ...] = ()
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    images: tuple[str, ...] = ()
    # Optional reversible state mutation. It is never serialized or exposed
    # to the model; EnvelopeKernel owns its commit boundary.
    deferred_effect: DeferredToolEffect | None = None

    def __post_init__(self) -> None:
        if self.action is not None and self.actions:
            raise ValueError(
                f"{self.name} returned both action and actions; a tool owns "
                "one runtime payload form"
            )
        if self.action is None and not self.actions:
            raise ValueError(
                f"{self.name} must return action or actions; a completed tool "
                "call has a runtime payload"
            )


# ---------------------------------------------------------------------------
# Agent observation / action types
# ---------------------------------------------------------------------------


@dataclass
class CameraViewObservation:
    """One calibrated camera view in a normalized agent observation."""

    name: str
    rgb: Any = None  # np.ndarray [H,W,3]
    # np.ndarray [H,W] metric depth, consumed privately by perception tools.
    depth: Any = None
    camera_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentObservation:
    """Normalized multi-view observation passed between agent modules."""

    actor_id: str = "robot"
    views: dict[str, CameraViewObservation] = field(default_factory=dict)
    pose: dict[str, Any] | None = None
    text: str = ""
    public_events: list[dict[str, Any]] = field(default_factory=list)
    # Named Human centres are exposed only for bodies grounded in the current
    # ego RGB-D frame.  They are historical perception evidence for memory,
    # never a live simulator directory.
    visible_humans: dict[str, tuple[float, float]] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentAction:
    """One resolved action to execute."""

    actor_id: str = "robot"
    name: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass
class TurnResult:
    """Outcome of one bounded turn."""

    status: str = "completed"  # completed / rejected / error
    actions: list[dict[str, Any]] = field(default_factory=list)
    # Model-authored suffix calls that must be resolved one by one against
    # each later action step's fresh observation. No suffix handler, schema
    # validation, or availability check runs before that call's own step.
    pending_calls: list[ToolCall] = field(default_factory=list)
    # Model-authored suffix calls intentionally cut after the first tool whose
    # result must be observed before another policy choice.
    discarded_calls: list[ToolCall] = field(default_factory=list)
    # Harness-owned failure detail when the turn did not complete a tool.
    diagnostic: str = ""
    # Machine-readable infrastructure category for ``status == 'error'``.
    # Policy rejections intentionally leave this empty.
    failure_kind: str = ""
    message: str = ""
    tool_results: list[ToolResult] = field(default_factory=list)
    # The single logical policy call for this turn, including provider-level
    # network retries and token usage for empty responses.
    model_calls: list[dict[str, Any]] = field(default_factory=list)
    # Complete structured decision object. Present when the turn loop runs
    # under the envelope protocol.
    envelope: "DecisionEnvelope | None" = None
    # Ordered, validated memory batches from successful model calls in this
    # turn. The memory owner commits them with the executable decision.
    # Boundaries permit a later call to revise a fact accepted earlier.
    accepted_memory_update_batches: list[list[dict[str, Any]]] = field(
        default_factory=list
    )
    # Explicit event outputs accepted from successful calls in this policy
    # turn, in submission order. The first submission is evaluator evidence;
    # plan memory applies revisions in order. The final raw envelope is intact.
    accepted_event_output_batches: list[dict[str, Any]] = field(default_factory=list)
    plan_updates_accepted: bool = True


# ---------------------------------------------------------------------------
# Decision envelope — the unified VLM output contract
# ---------------------------------------------------------------------------
#
# Each decision is one JSON object parsed into ``DecisionEnvelope``. Thinking
# and actions are required. Memory, plan, subgoal, and inspection fields are
# optional and are committed at their decision-step event boundary.

#: Envelope fields whose meaning is bound to the decision step that authored
#: them. Queued action rows replay execution bookkeeping only: copying these
#: onto later rows would let a submission written BEFORE a task event count as
#: the first reaction AT that event.
EVENT_WINDOW_OUTPUT_FIELDS = (
    "plan",
    "inspection",
)


@dataclass
class DecisionEnvelope:
    """One parsed agent decision with actions and event-driven predictions."""

    thinking: str
    actions: list[ToolCall] = field(default_factory=list)
    # Memory mutations authored by the model: each entry carries ``op`` and
    # ``note`` only. The kernel resolves ids before the ledger validates them.
    memory: list[dict[str, Any]] = field(default_factory=list)
    plan: dict[str, Any] | None = None
    # True when the model declared the active plan step complete.
    subgoal_checked: bool = False
    # Explicit task conclusion; omission means the Agent is still inspecting.
    inspection: str | None = None

    def validate(self, *, action_horizon: int) -> None:
        """Validate the registry-independent envelope structure.

        Exact Action Horizon semantics depend on whether a registered call is
        executable, informational, or terminal and are therefore enforced by
        ``EnvelopeKernel`` before any tool handler runs.
        """
        if not isinstance(self.thinking, str) or not self.thinking.strip():
            raise ValueError("DecisionEnvelope thinking must be a non-empty string.")
        if not isinstance(self.memory, list):
            raise ValueError(
                "DecisionEnvelope memory must be a list; omit the field when "
                "no memory change is needed."
            )
        if not isinstance(self.subgoal_checked, bool):
            raise ValueError("DecisionEnvelope subgoal_checked must be a boolean.")
        if self.inspection is not None and (
            not isinstance(self.inspection, str) or not self.inspection.strip()
        ):
            raise ValueError("DecisionEnvelope inspection must be a non-empty string.")
        if not self.actions:
            raise ValueError(
                "DecisionEnvelope has no actions; every policy decision "
                "must propose at least one tool call."
            )
        if len(self.actions) > int(action_horizon):
            raise ValueError(
                f"DecisionEnvelope proposed {len(self.actions)} calls but "
                f"Action Horizon is {int(action_horizon)}."
            )

    @property
    def call_names(self) -> list[str]:
        return [call.tool for call in self.actions]

    def as_dict(self) -> dict[str, Any]:
        """Serialise for trace persistence / post-hoc evaluation."""
        out: dict[str, Any] = {
            "thinking": self.thinking,
            "actions": [
                {"tool": call.tool, "args": dict(call.args)} for call in self.actions
            ],
        }
        if self.memory:
            out["memory"] = [dict(update) for update in self.memory]
        if self.plan is not None:
            out["plan"] = self.plan
        if self.subgoal_checked:
            out["subgoal"] = "checked"
        if self.inspection is not None:
            out["inspection"] = self.inspection
        return out


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class ToolHandler(Protocol):
    """Protocol for tool execution handlers.

    One handler per tool. It validates the model's arguments and returns a
    :class:`ToolResult` with a runtime payload: a controller command when the
    environment must move the body, or ``tool_turn_action(name)`` when the
    handler already finished. Multi-step execution behind a single call
    (``grasp_object``'s approach and closure) belongs to the runtime, not here.
    """

    def execute(self, call: ToolCall, context: dict[str, Any]) -> ToolResult: ...


__all__ = [
    "ToolSpec",
    "ToolCall",
    "ToolResult",
    "tool_turn_action",
    "CameraViewObservation",
    "AgentObservation",
    "AgentAction",
    "TurnResult",
    "DecisionEnvelope",
    "ToolHandler",
]
