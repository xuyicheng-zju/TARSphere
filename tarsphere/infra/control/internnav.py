"""InternVLA-N1 DualVLN client and RGB-D action adapter.

The client sends the navigation instruction unchanged. Its observation and HTTP
contracts follow InternNav's agent server. DualVLN with ``continuous_traj=True``
returns discrete actions after converting predicted waypoints: stop (0), forward
0.25 m (1), left 15 degrees (2), right 15 degrees (3), or stand still (-1).
The adapter maps these IDs to ``move_body`` deltas; the locomotion primitive
owns duration and velocity. This interface rejects velocity payloads.

The default navigation skill uses the online map and X-NavDP; this adapter serves
explicit InternVLA integrations."""

from __future__ import annotations

import base64
import json
import math
import os
import pickle
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np


INTERNNAV_HOST_ENV = "TARSPHERE_INTERNNAV_HOST"
INTERNNAV_PORT_ENV = "TARSPHERE_INTERNNAV_PORT"
INTERNNAV_MODEL_NAME_ENV = "TARSPHERE_INTERNNAV_MODEL_NAME"
INTERNNAV_MODEL_PATH_ENV = "TARSPHERE_INTERNNAV_MODEL_PATH"
INTERNNAV_TIMEOUT_S_ENV = "TARSPHERE_INTERNNAV_TIMEOUT_S"


# DualVLN consumes the same physical RGB-D render product as the evaluated
# Agent.  Resolution and optical calibration therefore follow the G1 ego
# camera instead of creating a second synthetic view.  The model server uses
# these values for depth projection, so they must match Isaac exactly.
DUALVLN_RGB_WIDTH = 640
DUALVLN_RGB_HEIGHT = 480
DUALVLN_HORIZONTAL_FOV_DEG = float(
    os.environ.get("TARSPHERE_ROBOT_CAMERA_FOV_DEG", "100.0")
)
if not 0.0 < DUALVLN_HORIZONTAL_FOV_DEG < 180.0:
    raise ValueError(
        "TARSPHERE_ROBOT_CAMERA_FOV_DEG must be in (0, 180); got "
        f"{DUALVLN_HORIZONTAL_FOV_DEG!r}."
    )
_DUALVLN_FOCAL_LENGTH_PX = (DUALVLN_RGB_WIDTH / 2.0) / math.tan(
    math.radians(DUALVLN_HORIZONTAL_FOV_DEG) / 2.0
)
DUALVLN_CAMERA_INTRINSICS = (
    (_DUALVLN_FOCAL_LENGTH_PX, 0.0, DUALVLN_RGB_WIDTH / 2.0),
    (0.0, _DUALVLN_FOCAL_LENGTH_PX, DUALVLN_RGB_HEIGHT / 2.0),
    (0.0, 0.0, 1.0),
)
DUALVLN_HISTORY_FRAMES = 8
DUALVLN_NUM_FRAMES = 32
DUALVLN_DEFAULT_PORT = 8087
DUALVLN_DEFAULT_MODEL_NAME = "internvla_n1"
DUALVLN_DEFAULT_MODEL_PATH = "checkpoints/InternVLA-N1-DualVLN"
DUALVLN_DEFAULT_TIMEOUT_S = 120.0
DUALVLN_MAX_INNER_STEPS = 120

# Official DiscreteController spatial meaning in vln_default_config.py.
# Duration and twist belong to the evaluated ``move_body`` primitive.
DUALVLN_FORWARD_DISTANCE_M = 0.25
DUALVLN_TURN_DEG = 15.0
DUALVLN_DEPTH_MAX_M = 10.0
# InternNav keeps the official Habitat LOOKUP/LOOKDOWN ids. During one
# ``navigate`` call, a 10° step temporarily tilts the shared ego camera
# (harness −15° → LOOKDOWN −25°). The normal Agent gaze is restored
# before control returns. Repeated LOOKDOWN is idempotent.
DUALVLN_LOOK_TILT_STEP_DEG = 10.0
DUALVLN_LOOK_DOWN_EXTRA_DEG = -DUALVLN_LOOK_TILT_STEP_DEG
DUALVLN_LOOK_UP_EXTRA_DEG = DUALVLN_LOOK_TILT_STEP_DEG
_ZERO_BODY_DELTA = (0.0, 0.0, 0.0)

# Habitat-style ids from InternVLAN1Agent / DiscreteController.
_DISCRETE_STOP = 0
_DISCRETE_FORWARD = 1
_DISCRETE_TURN_LEFT = 2
_DISCRETE_TURN_RIGHT = 3
_DISCRETE_LOOK_UP = 4
_DISCRETE_LOOK_DOWN = 5
_DISCRETE_STAND_STILL = -1


_dualvln_client: DualVLNAgentServerBackend | None = None


# Representative R2R phrasing used by component tests only. It is not injected
# into the evaluated Agent prompt.
DUALVLN_ROUTE_INSTRUCTION_EXAMPLE = (
    "Walk past the dining table on your right. "
    "Go through the kitchen doorway. "
    "Wait near the refrigerator."
)


class InternNavServerError(RuntimeError):
    """The DualVLN agent-server rejected or failed a request."""


@dataclass(frozen=True)
class InternNavSettings:
    """Connection and DualVLN observation limits for one process.

    Attributes:
        host: Agent-server hostname. Required for DualVLN.
        port: Agent-server port. Required for DualVLN.
        model_name: InternNav agent route name, used as ``/agent/{name}/step``.
        model_path: DualVLN checkpoint the server loads at ``/agent/init``.
        request_timeout_s: Maximum time allowed for one HTTP call.
    """

    host: str
    port: int
    model_name: str
    model_path: str
    request_timeout_s: float

    @classmethod
    def from_environ(
        cls,
        environ: dict[str, str] | None = None,
    ) -> InternNavSettings:
        source = os.environ if environ is None else environ
        host = str(source.get(INTERNNAV_HOST_ENV, "")).strip()
        raw_port = str(source.get(INTERNNAV_PORT_ENV, "")).strip()
        model_name = (
            str(source.get(INTERNNAV_MODEL_NAME_ENV, "")).strip()
            or DUALVLN_DEFAULT_MODEL_NAME
        )
        model_path = (
            str(source.get(INTERNNAV_MODEL_PATH_ENV, "")).strip()
            or DUALVLN_DEFAULT_MODEL_PATH
        )
        raw_timeout = str(source.get(INTERNNAV_TIMEOUT_S_ENV, "")).strip()
        if host:
            port = int(raw_port or DUALVLN_DEFAULT_PORT)
        elif raw_port:
            port = int(raw_port)
        else:
            port = 0
        timeout_s = float(raw_timeout or DUALVLN_DEFAULT_TIMEOUT_S)
        if host and not (1 <= port <= 65535):
            raise ValueError(
                f"{INTERNNAV_PORT_ENV} must lie in [1, 65535]; got {port!r}."
            )
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError(
                f"{INTERNNAV_TIMEOUT_S_ENV} must be a positive timeout; "
                f"got {timeout_s!r}."
            )
        return cls(
            host=host,
            port=port,
            model_name=model_name,
            model_path=model_path,
            request_timeout_s=timeout_s,
        )


def dualvln_server_configured(settings: InternNavSettings) -> bool:
    """True when DualVLN has a hostname and port to open."""

    return bool(settings.host) and 1 <= int(settings.port) <= 65535


def dualvln_agent_init_payload(settings: InternNavSettings) -> dict[str, Any]:
    """Return the authoritative official DualVLN ``/agent/init`` request."""

    return {
        "agent_config": {
            "server_host": settings.host,
            "server_port": settings.port,
            "model_name": settings.model_name,
            "ckpt_path": "",
            "model_settings": {
                "policy_name": "InternVLAN1_Policy",
                "state_encoder": None,
                "env_num": 1,
                "sim_num": 1,
                "model_path": settings.model_path,
                "camera_intrinsic": [list(row) for row in DUALVLN_CAMERA_INTRINSICS],
                "width": DUALVLN_RGB_WIDTH,
                "height": DUALVLN_RGB_HEIGHT,
                "hfov": DUALVLN_HORIZONTAL_FOV_DEG,
                "resize_w": 384,
                "resize_h": 384,
                "max_new_tokens": 1024,
                "num_frames": DUALVLN_NUM_FRAMES,
                "num_history": DUALVLN_HISTORY_FRAMES,
                "num_future_steps": 4,
                "device": "cuda:0",
                "predict_step_nums": 32,
                "continuous_traj": True,
                "infer_mode": "partial_async",
                "vis_debug": False,
            },
        }
    }


def official_norm_depth(depth_m: Any) -> np.ndarray:
    """Match ``internnav.evaluator.utils.common.norm_depth``.

    Official ``InternVLAN1Agent.step`` indexes ``obs['depth']``. DualVLN
    training is RGB-primary relative to NavDP, but the agent-server still
    requires this channel. Metric metres are clipped at 10 m and scaled
    to ``[0, 1]``, then given a trailing singleton axis.
    """

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(
            f"DualVLN metric depth must be HxW metres; got shape {tuple(depth.shape)}."
        )
    clipped = np.minimum(depth, DUALVLN_DEPTH_MAX_M)
    clipped = np.maximum(clipped, 0.0)
    normalized = clipped / DUALVLN_DEPTH_MAX_M
    return normalized[..., np.newaxis]


@dataclass(frozen=True)
class DualVLNObservation:
    """One official InternVLAN1Agent observation. Never shown to the VLM.

    Attributes:
        rgb: Egocentric RGB at the official 640x480 resolution. Pose
            matches the G1 harness ``d435_link`` camera (mount pitch
            -15°). HFOV 79 is InternNav-official and differs from the
            G1 ego default of 100°.
        depth: Official normalized depth ``HxWx1`` in ``[0, 1]``.
        instruction: The instruction string posted to DualVLN.
        globalgps: Robot base XYZ in metres, tool-private.
        globalrotation: Robot base quaternion ``[x, y, z, w]``, tool-private.
    """

    rgb: np.ndarray
    depth: np.ndarray
    instruction: str
    globalgps: tuple[float, float, float]
    globalrotation: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        frame = np.asarray(self.rgb)
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError(
                f"DualVLN RGB must be an HxWx3 image; got shape {tuple(frame.shape)}."
            )
        if frame.dtype != np.uint8:
            raise ValueError(
                "DualVLN RGB must be uint8; official InternVLAN1_Policy "
                f"calls Image.fromarray(rgb). Got dtype {frame.dtype}."
            )
        if frame.shape[0] != DUALVLN_RGB_HEIGHT or frame.shape[1] != DUALVLN_RGB_WIDTH:
            raise ValueError(
                "DualVLN RGB must be "
                f"{DUALVLN_RGB_WIDTH}x{DUALVLN_RGB_HEIGHT}; "
                f"got {frame.shape[1]}x{frame.shape[0]}. DualVLN consumes "
                "the evaluated G1 ego RGB-D frame without resizing."
            )
        depth = np.asarray(self.depth)
        if depth.shape[:2] != (DUALVLN_RGB_HEIGHT, DUALVLN_RGB_WIDTH):
            raise ValueError(
                f"DualVLN depth must match 640x480 RGB; got shape {tuple(depth.shape)}."
            )
        if depth.ndim != 3 or depth.shape[2] != 1:
            raise ValueError(
                "DualVLN depth must be HxWx1 after official norm_depth; "
                f"got shape {tuple(depth.shape)}."
            )
        if len(self.globalgps) != 3 or not all(
            math.isfinite(float(value)) for value in self.globalgps
        ):
            raise ValueError("DualVLN globalgps must be three finite metres.")
        if len(self.globalrotation) != 4 or not all(
            math.isfinite(float(value)) for value in self.globalrotation
        ):
            raise ValueError("DualVLN globalrotation must be a finite xyzw quaternion.")

    def as_internnav_obs(self) -> dict[str, Any]:
        """Return the official InternNav ``step`` observation dict."""

        rgb = np.asarray(self.rgb)
        if rgb.shape[2] > 3:
            rgb = rgb[:, :, :3]
        return {
            "rgb": np.ascontiguousarray(rgb),
            "depth": np.ascontiguousarray(self.depth),
            "instruction": self.instruction,
            "globalgps": [float(value) for value in self.globalgps],
            "globalrotation": [float(value) for value in self.globalrotation],
        }


@dataclass(frozen=True)
class NavigationCommand:
    """One DualVLN inner-loop spatial decision for the ``move_body`` primitive.

    Attributes:
        status: ``continue`` applies ``delta`` through ``move_body``;
            ``hold`` is zero motion through the same primitive stack;
            ``look_down`` is official Habitat action 5: zero body
            motion plus a temporary shared-ego-camera tilt of
            ``DUALVLN_LOOK_DOWN_EXTRA_DEG`` (−10°);
            ``look_up`` is official Habitat action 4: the matching
            +10° nav tilt toward the harness;
            ``stop`` ends the ``navigate`` skill.
        delta: Body-frame ``(forward_m, left_m, turn_deg)``. Official
            DualVLN ids use 0.25 m forward or ±15° yaw. ``None`` when
            ``status`` is ``stop``.
        raw_action: The official agent-server payload, telemetry only.
    """

    status: Literal["continue", "hold", "look_down", "look_up", "stop"]
    delta: tuple[float, float, float] | None
    raw_action: Any

    def __post_init__(self) -> None:
        if self.status == "stop":
            if self.delta is not None:
                raise ValueError("A DualVLN stop has no body delta.")
            return
        delta = self.delta
        if self.status in ("hold", "look_down", "look_up"):
            delta = _ZERO_BODY_DELTA if delta is None else delta
            object.__setattr__(self, "delta", delta)
        if delta is None or len(delta) != 3:
            raise ValueError(
                "A DualVLN locomotion command needs a three-number "
                "body-frame delta [forward_m, left_m, turn_deg]."
            )
        if not all(math.isfinite(float(value)) for value in delta):
            raise ValueError("DualVLN body delta components must be finite.")
        magnitude = max(abs(float(value)) for value in delta)
        if self.status == "continue" and magnitude <= 0.0:
            raise ValueError("A DualVLN continue command needs a non-zero body delta.")
        if self.status in ("hold", "look_down", "look_up") and magnitude > 0.0:
            raise ValueError(f"A DualVLN {self.status} must be a zero body delta.")


class InternNavBackend(Protocol):
    """Process DualVLN observations into spatial body deltas."""

    name: str

    def is_ready(self) -> bool:
        """True when the inner loop can consume an observation."""

    def reset(self) -> None:
        """Clear DualVLN episode state before a new ``navigate`` call."""

    def step(self, observation: DualVLNObservation) -> NavigationCommand:
        """Return the next spatial delta, hold, or stop."""


class DualVLNAgentServerBackend:
    """Official InternNav agent-server client for DualVLN.

    Speaks ``InternNav/internnav/utils/comm_utils/client.py``:
    ``POST /agent/init``, pickled observations on
    ``POST /agent/{name}/step``, and ``POST /agent/{name}/reset``.
    """

    name = "internvla_n1_dualvln"

    def __init__(self, settings: InternNavSettings) -> None:
        if not dualvln_server_configured(settings):
            raise InternNavServerError(
                f"DualVLN requires {INTERNNAV_HOST_ENV} and {INTERNNAV_PORT_ENV}."
            )
        self.settings = settings
        self._agent_name: str | None = None

    def is_ready(self) -> bool:
        try:
            self._ensure_agent()
        except InternNavServerError:
            return False
        return True

    def require_ready(self) -> None:
        self._ensure_agent()

    def reset(self) -> None:
        agent_name = self._ensure_agent()
        self._post(
            f"/agent/{agent_name}/reset",
            {"reset_index": None},
        )

    def step(self, observation: DualVLNObservation) -> NavigationCommand:
        agent_name = self._ensure_agent()
        encoded = base64.b64encode(
            pickle.dumps([observation.as_internnav_obs()])
        ).decode("utf-8")
        payload = self._post(
            f"/agent/{agent_name}/step",
            {"observation": encoded},
        )
        if "action" not in payload:
            raise InternNavServerError(
                "DualVLN agent-server step response is missing `action`."
            )
        return parse_internnav_action(payload["action"])

    def _ensure_agent(self) -> str:
        if self._agent_name:
            return self._agent_name
        # The production launcher eagerly initializes the resident model.
        # Reuse it instead of posting /agent/init again (which would construct
        # a second policy and can exhaust GPU memory). A stock, non-eager
        # official server returns 404 here and falls through to initialization.
        existing_name = self.settings.model_name
        existing = self._post(
            f"/agent/{existing_name}/reset",
            {"reset_index": None},
            missing_ok=True,
        )
        if existing is not None:
            self._agent_name = existing_name
            return existing_name
        payload = self._post(
            "/agent/init",
            dualvln_agent_init_payload(self.settings),
        )
        assert payload is not None
        agent_name = str(payload.get("agent_name") or "").strip()
        if not agent_name:
            raise InternNavServerError(
                "DualVLN agent-server init did not return agent_name."
            )
        self._agent_name = agent_name
        return agent_name

    def _post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        missing_ok: bool = False,
    ) -> dict[str, Any] | None:
        url = f"http://{self.settings.host}:{self.settings.port}{path}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.settings.request_timeout_s,
            ) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if missing_ok and exc.code == 404:
                return None
            detail = exc.read().decode("utf-8", errors="replace")
            raise InternNavServerError(
                f"DualVLN agent-server {path} failed with HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise InternNavServerError(
                f"DualVLN agent-server {url} is unreachable: {exc.reason}"
            ) from exc
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise InternNavServerError(
                f"DualVLN agent-server {path} returned non-JSON: {body!r}"
            ) from exc
        if not isinstance(decoded, dict):
            raise InternNavServerError(
                f"DualVLN agent-server {path} returned {type(decoded).__name__}, "
                "not an object."
            )
        return decoded


def parse_internnav_action(raw: Any) -> NavigationCommand:
    """Map one official InternVLAN1Agent ``step`` payload onto a body delta.

    Official DualVLN returns ``[{'action': [id], 'ideal_flag': True}]``.
    ``continuous_traj=True`` does not put a twist on the wire; S1
    waypoints are already Habitat ids after ``traj_to_actions``. Duration
    and ``move_by_speed`` are not this module's contract.
    """

    action_id = _official_discrete_id(raw)
    if action_id == _DISCRETE_STOP:
        return NavigationCommand(
            status="stop",
            delta=None,
            raw_action=raw,
        )
    if action_id == _DISCRETE_STAND_STILL:
        return NavigationCommand(
            status="hold",
            delta=_ZERO_BODY_DELTA,
            raw_action=raw,
        )
    if action_id == _DISCRETE_LOOK_DOWN:
        return NavigationCommand(
            status="look_down",
            delta=_ZERO_BODY_DELTA,
            raw_action=raw,
        )
    if action_id == _DISCRETE_LOOK_UP:
        return NavigationCommand(
            status="look_up",
            delta=_ZERO_BODY_DELTA,
            raw_action=raw,
        )
    if action_id == _DISCRETE_FORWARD:
        delta = (DUALVLN_FORWARD_DISTANCE_M, 0.0, 0.0)
    elif action_id == _DISCRETE_TURN_LEFT:
        delta = (0.0, 0.0, DUALVLN_TURN_DEG)
    elif action_id == _DISCRETE_TURN_RIGHT:
        delta = (0.0, 0.0, -DUALVLN_TURN_DEG)
    else:
        raise InternNavServerError(
            f"DualVLN returned an action this harness cannot execute: {raw!r}"
        )
    return NavigationCommand(
        status="continue",
        delta=delta,
        raw_action=raw,
    )


def _official_discrete_id(raw: Any) -> int:
    payload = _official_step_object(raw)
    if payload.get("ideal_flag") is not True:
        raise InternNavServerError(
            "InternVLA-N1 DualVLN returns Habitat discrete ids with "
            "ideal_flag True. A velocity / ideal_flag False payload is "
            f"not this model's agent-server contract. Got {raw!r}."
        )
    if "action" not in payload:
        raise InternNavServerError(
            f"DualVLN action object is missing `action`: {raw!r}"
        )
    current = payload["action"]
    while isinstance(current, (list, tuple)):
        if len(current) != 1:
            raise InternNavServerError(
                "DualVLN action must be one Habitat discrete id, not a "
                f"velocity vector. Got {raw!r}."
            )
        current = current[0]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise InternNavServerError(f"DualVLN action is not a discrete id: {raw!r}")
    if isinstance(current, float) and not current.is_integer():
        raise InternNavServerError(f"DualVLN action is not a discrete id: {raw!r}")
    return int(current)


def _official_step_object(raw: Any) -> dict[str, Any]:
    current = raw
    if isinstance(current, list):
        if len(current) != 1 or not isinstance(current[0], dict):
            raise InternNavServerError(
                f"DualVLN agent-server step must return one action object. Got {raw!r}."
            )
        current = current[0]
    if not isinstance(current, dict):
        raise InternNavServerError(
            "DualVLN action is not an official step object "
            f"{{action, ideal_flag}}. Got {raw!r}."
        )
    return current


def internav_backend_available(
    environ: dict[str, str] | None = None,
) -> bool:
    """True when this process can construct the configured backend."""

    settings = InternNavSettings.from_environ(environ)
    return dualvln_server_configured(settings)


def build_internnav_backend(
    environ: dict[str, str] | None = None,
) -> InternNavBackend:
    """Construct the configured InternNav backend, or fail loudly."""

    settings = InternNavSettings.from_environ(environ)
    global _dualvln_client
    if _dualvln_client is None:
        _dualvln_client = DualVLNAgentServerBackend(settings)
    elif _dualvln_client.settings != settings:
        raise RuntimeError(
            "DualVLN settings changed after the agent-server client was opened."
        )
    return _dualvln_client


def reset_internnav_backend_for_tests() -> None:
    """Drop the process DualVLN client. Tests only."""

    global _dualvln_client
    _dualvln_client = None


__all__ = [
    "DUALVLN_CAMERA_INTRINSICS",
    "DUALVLN_FORWARD_DISTANCE_M",
    "DUALVLN_LOOK_DOWN_EXTRA_DEG",
    "DUALVLN_LOOK_TILT_STEP_DEG",
    "DUALVLN_LOOK_UP_EXTRA_DEG",
    "DUALVLN_HISTORY_FRAMES",
    "DUALVLN_HORIZONTAL_FOV_DEG",
    "DUALVLN_MAX_INNER_STEPS",
    "DUALVLN_NUM_FRAMES",
    "DUALVLN_RGB_HEIGHT",
    "DUALVLN_ROUTE_INSTRUCTION_EXAMPLE",
    "DUALVLN_RGB_WIDTH",
    "DUALVLN_TURN_DEG",
    "INTERNNAV_HOST_ENV",
    "INTERNNAV_MODEL_PATH_ENV",
    "INTERNNAV_PORT_ENV",
    "DualVLNAgentServerBackend",
    "DualVLNObservation",
    "InternNavBackend",
    "InternNavServerError",
    "InternNavSettings",
    "NavigationCommand",
    "build_internnav_backend",
    "dualvln_server_configured",
    "internav_backend_available",
    "official_norm_depth",
    "parse_internnav_action",
    "reset_internnav_backend_for_tests",
]
