"""Single typed authority for the evaluated Agent harness settings.

The public names here are shared by the paper, command-line entry points,
trace metadata, and result tables.  A run has two independent groups:

``ProtocolConfig``
    What observations and action proposals cross the Agent boundary.

``ControllerConfig``
    Private online mapping cadence for the physical controller.

There are no environment-variable readers or compatibility aliases in this
module.  Callers construct one :class:`AgentHarnessConfig`, pass that same instance
through the agent, runner, and environment, and persist :meth:`as_dict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping


TEXT_HISTORY_ALL = "all"
MAIN_TEXT_HISTORY_STEPS = 10
# Number of completed observations that may accompany the always-present
# current observation.  This is deliberately a history count: zero means the
# model sees only current RGB; four means current RGB plus four older frames in
# one image.
MAX_VISUAL_HISTORY_FRAMES = 4
DEFAULT_VISUAL_HISTORY_FRAMES = 0
MAX_ACTION_HORIZON = 5
DYNAMIC_ACTION_HORIZON = "dynamic"


class CameraView(str, Enum):
    """Agent-visible camera names in their canonical prompt order."""

    EGO = "ego"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"


CAMERA_VIEW_ORDER = (
    CameraView.EGO,
    CameraView.LEFT_WRIST,
    CameraView.RIGHT_WRIST,
)


@dataclass(frozen=True)
class ProtocolConfig:
    """Observation/action Protocol used by one evaluated-agent run.

    ``text_history_horizon`` counts completed action-step text records.  The
    literal ``"all"`` retains all completed records; zero removes past records
    while leaving task semantics, active plan/progress, current observation,
    and spatial/current state intact.

    ``visual_history_horizon`` counts completed action-step frames before the
    always-present current observation and is bounded by
    :data:`MAX_VISUAL_HISTORY_FRAMES`. Zero sends current only; four sends the
    current frame plus four older frames. They are composed without overlap as
    one chronological mosaic per camera view, rather than separate messages.
    An integer ``action_horizon`` is the exact number of executable primitives
    proposed by every ordinary policy inference. ``"dynamic"`` lets the model
    choose one through :data:`MAX_ACTION_HORIZON` primitives from current
    evidence. Calls in one block execute serially without another model
    inference; each handler receives the fresh observation produced by its
    predecessor. Every completed call remains one action step. Model-output
    errors consume that policy decision; only transient transport failures
    retry inside it.

    Every field is fixed when this frozen object is created and remains
    unchanged for the complete run.  History horizons select exactly the most
    recent configured number of existing records; the beginning of an episode
    is not padded with fabricated history.
    """

    text_history_horizon: int | str = MAIN_TEXT_HISTORY_STEPS
    visual_history_horizon: int = DEFAULT_VISUAL_HISTORY_FRAMES
    action_horizon: int | str = DYNAMIC_ACTION_HORIZON
    camera_views: tuple[CameraView, ...] = (CameraView.EGO,)

    def __post_init__(self) -> None:
        text_horizon = self.text_history_horizon
        if text_horizon != TEXT_HISTORY_ALL:
            if isinstance(text_horizon, bool) or not isinstance(text_horizon, int):
                raise TypeError(
                    "text_history_horizon must be a non-negative integer or "
                    f"{TEXT_HISTORY_ALL!r}, got {text_horizon!r}"
                )
            if text_horizon < 0:
                raise ValueError("text_history_horizon must be non-negative")

        if isinstance(self.visual_history_horizon, bool) or not isinstance(
            self.visual_history_horizon, int
        ):
            raise TypeError("visual_history_horizon must be an integer")
        if not 0 <= self.visual_history_horizon <= MAX_VISUAL_HISTORY_FRAMES:
            raise ValueError(
                "visual_history_horizon must be between 0 and "
                f"{MAX_VISUAL_HISTORY_FRAMES}"
            )

        action_horizon = self.action_horizon
        if action_horizon != DYNAMIC_ACTION_HORIZON:
            if isinstance(action_horizon, bool) or not isinstance(action_horizon, int):
                raise TypeError(
                    f"action_horizon must be an integer or {DYNAMIC_ACTION_HORIZON!r}"
                )
            if not 1 <= action_horizon <= MAX_ACTION_HORIZON:
                raise ValueError(
                    f"action_horizon must be between 1 and {MAX_ACTION_HORIZON}"
                )

        views = tuple(CameraView(view) for view in self.camera_views)
        if not views:
            raise ValueError("camera_views must contain at least ego")
        if len(set(views)) != len(views):
            raise ValueError(f"camera_views contains duplicates: {views!r}")
        if CameraView.EGO not in views:
            raise ValueError("camera_views must include ego")
        expected = tuple(view for view in CAMERA_VIEW_ORDER if view in views)
        if views != expected:
            raise ValueError(
                "camera_views must use canonical order "
                f"{tuple(view.value for view in CAMERA_VIEW_ORDER)}, got "
                f"{tuple(view.value for view in views)}"
            )
        object.__setattr__(self, "camera_views", views)

    @property
    def dynamic_action_horizon(self) -> bool:
        """Whether each ordinary policy inference selects its block length."""

        return self.action_horizon == DYNAMIC_ACTION_HORIZON

    @property
    def action_horizon_limit(self) -> int:
        """Maximum executable calls accepted from one policy inference."""

        if self.dynamic_action_horizon:
            return MAX_ACTION_HORIZON
        return int(self.action_horizon)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable resolved Protocol metadata."""

        return {
            "text_history_horizon": self.text_history_horizon,
            "visual_history_horizon": self.visual_history_horizon,
            "action_horizon": self.action_horizon,
            "camera_views": [view.value for view in self.camera_views],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProtocolConfig":
        """Parse the current mapping shape and reject unknown/missing aliases."""

        allowed = {
            "text_history_horizon",
            "visual_history_horizon",
            "action_horizon",
            "camera_views",
        }
        extras = set(value) - allowed
        if extras:
            raise ValueError(f"unknown Protocol fields: {sorted(extras)}")
        kwargs = dict(value)
        if "camera_views" in kwargs:
            raw_views = kwargs["camera_views"]
            if not isinstance(raw_views, (list, tuple)):
                raise TypeError("camera_views must be an ordered list")
            kwargs["camera_views"] = tuple(CameraView(item) for item in raw_views)
        return cls(**kwargs)


@dataclass(frozen=True)
class ControllerConfig:
    """Timing parameters for the physical controller and private online map."""

    # Private synchronized ego RGB-D mapping samples per simulated second.
    # Zero samples only at decision boundaries; policy inference is independent.
    online_mapping_hz: float = 1.0

    def __post_init__(self) -> None:
        mapping_hz = float(self.online_mapping_hz)
        if not math.isfinite(mapping_hz) or not 0.0 <= mapping_hz <= 30.0:
            raise ValueError("online_mapping_hz must be finite and between 0 and 30 Hz")
        object.__setattr__(self, "online_mapping_hz", mapping_hz)

    def as_dict(self) -> dict[str, Any]:
        return {"online_mapping_hz": self.online_mapping_hz}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ControllerConfig":
        extras = set(value) - {"online_mapping_hz"}
        if extras:
            raise ValueError(f"unknown Controller fields: {sorted(extras)}")
        return cls(**dict(value))


@dataclass(frozen=True)
class AgentHarnessConfig:
    """Fully resolved default harness configuration shared by the whole run."""

    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol.as_dict(),
            "controller": self.controller.as_dict(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AgentHarnessConfig":
        allowed = {"protocol", "controller"}
        extras = set(value) - allowed
        if extras:
            raise ValueError(f"unknown AgentHarnessConfig fields: {sorted(extras)}")
        protocol_value = value.get("protocol", {})
        controller_value = value.get("controller", {})
        if not isinstance(protocol_value, Mapping):
            raise TypeError("protocol must be an object")
        if not isinstance(controller_value, Mapping):
            raise TypeError("controller must be an object")
        return cls(
            protocol=ProtocolConfig.from_mapping(protocol_value),
            controller=ControllerConfig.from_mapping(controller_value),
        )


__all__ = [
    "CAMERA_VIEW_ORDER",
    "CameraView",
    "ControllerConfig",
    "DEFAULT_VISUAL_HISTORY_FRAMES",
    "DYNAMIC_ACTION_HORIZON",
    "AgentHarnessConfig",
    "MAIN_TEXT_HISTORY_STEPS",
    "MAX_ACTION_HORIZON",
    "MAX_VISUAL_HISTORY_FRAMES",
    "ProtocolConfig",
    "TEXT_HISTORY_ALL",
]
