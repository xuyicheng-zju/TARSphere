"""Official Psi0 HTTP ``POST /act`` client. Isaac never loads the weights."""

from __future__ import annotations

import json
import os
from base64 import b64decode, b64encode
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from numpy.lib.format import descr_to_dtype, dtype_to_descr

from tarsphere.policies.psi0.contract import (
    DEFAULT_PSI0_PORT,
    DEFAULT_PSI0_TIMEOUT_S,
    TARSPHERE_PSI0_ENDPOINT_ENV,
    TARSPHERE_PSI0_TIMEOUT_S_ENV,
    PSI0_ACTION_DIM,
    PSI0_DATASET_NAME,
    PSI0_IMAGE_KEY,
    Psi0ContractError,
)


class Psi0ServerError(RuntimeError):
    """The official serve_psi0 HTTP server is missing or malformed."""


@dataclass(frozen=True)
class Psi0ServerEndpoint:
    """TCP address of one out-of-band ``serve_psi0`` process.

    Attributes:
        host: Hostname or IPv4 address of the FastAPI server.
        port: HTTP port bound by official ``psi0_serve_simple``.
    """

    host: str
    port: int

    def __post_init__(self) -> None:
        if not str(self.host).strip():
            raise Psi0ServerError("Psi0 serve host cannot be empty.")
        if not 1 <= int(self.port) <= 65535:
            raise Psi0ServerError("Psi0 serve port must lie in [1, 65535].")


def require_psi0_endpoint(raw: str | None = None) -> Psi0ServerEndpoint:
    """Parse ``host:port``. Empty input is an unavailable backend, not a default."""

    text = (
        raw if raw is not None else os.environ.get(TARSPHERE_PSI0_ENDPOINT_ENV, "")
    ).strip()
    if not text:
        raise Psi0ServerError(
            f"{TARSPHERE_PSI0_ENDPOINT_ENV} is unset. Start official "
            f"serve_psi0 on host:{DEFAULT_PSI0_PORT} and point this "
            "process at that host:port."
        )
    host, separator, port_text = text.rpartition(":")
    if separator != ":" or not host or not port_text:
        raise Psi0ServerError(f"Psi0 endpoint must be host:port, got {text!r}.")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise Psi0ServerError(
            f"Psi0 endpoint port must be an integer, got {text!r}."
        ) from exc
    return Psi0ServerEndpoint(host=host, port=port)


def numpy_serialize(value: object) -> object:
    """Encode one NumPy array the way official RequestMessage does."""

    if isinstance(value, dict) and "__numpy__" in value:
        return value
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        data = array.data if array.flags["C_CONTIGUOUS"] else array.tobytes()
        return {
            "__numpy__": b64encode(data).decode(),
            "dtype": dtype_to_descr(array.dtype),
            "shape": array.shape,
        }
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def numpy_deserialize(value: object) -> object:
    """Decode one official ``__numpy__`` blob."""

    if not isinstance(value, dict) or "__numpy__" not in value:
        return value
    array = np.frombuffer(
        b64decode(value["__numpy__"]),
        descr_to_dtype(value["dtype"]),
    )
    shape = value["shape"]
    return array.reshape(shape) if shape else array[0]


def convert_numpy_in_dict(data: object, func: Any) -> Any:
    """Walk the official Request/Response JSON tree."""

    if isinstance(data, dict):
        if "__numpy__" in data:
            return func(data)
        return {key: convert_numpy_in_dict(item, func) for key, item in data.items()}
    if isinstance(data, list):
        return [convert_numpy_in_dict(item, func) for item in data]
    if isinstance(data, (np.ndarray, np.generic)):
        return func(data)
    return data


def serialize_request(
    *,
    image: dict[str, np.ndarray],
    instruction: str,
    history: dict[str, Any],
    state: dict[str, np.ndarray],
    condition: dict[str, Any] | None = None,
    gt_action: object | None = None,
    dataset_name: str = PSI0_DATASET_NAME,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build the official RequestMessage JSON body."""

    if PSI0_IMAGE_KEY not in image:
        raise Psi0ContractError(
            f"Psi0 request image must use official key {PSI0_IMAGE_KEY!r}."
        )
    if "states" not in state:
        raise Psi0ContractError('Psi0 request state must contain "states".')
    payload = {
        "image": image,
        "instruction": instruction,
        "history": history,
        "state": state,
        "condition": {} if condition is None else condition,
        "gt_action": [] if gt_action is None else gt_action,
        "dataset_name": dataset_name,
        "timestamp": timestamp or _official_timestamp(),
    }
    return convert_numpy_in_dict(payload, numpy_serialize)


def deserialize_response(payload: dict[str, Any]) -> np.ndarray:
    """Return the official denormalized ``(Ta, 36)`` action chunk."""

    decoded = convert_numpy_in_dict(payload, numpy_deserialize)
    if not isinstance(decoded, dict) or "action" not in decoded:
        raise Psi0ServerError(f"Psi0 /act response is missing action: {payload!r}.")
    action = np.asarray(decoded["action"], dtype=np.float32)
    if action.ndim != 2 or action.shape[-1] != PSI0_ACTION_DIM:
        raise Psi0ServerError(
            f"Psi0 /act action must be (Ta, {PSI0_ACTION_DIM}), got {action.shape}."
        )
    if action.shape[0] < 1:
        raise Psi0ServerError("Psi0 /act returned an empty action chunk.")
    if not np.isfinite(action).all():
        raise Psi0ServerError("Psi0 /act returned a non-finite action.")
    return action


class Psi0ActSession:
    """Stateful HTTP client for one Psi0 ``/act`` session."""

    def __init__(
        self,
        endpoint: Psi0ServerEndpoint,
        *,
        timeout_s: float | None = None,
    ) -> None:
        self.endpoint = endpoint
        if timeout_s is None:
            raw = os.environ.get(TARSPHERE_PSI0_TIMEOUT_S_ENV, "").strip()
            timeout_s = float(raw) if raw else DEFAULT_PSI0_TIMEOUT_S
        if timeout_s <= 0.0:
            raise Psi0ServerError("Psi0 /act timeout must be positive.")
        self.timeout_s = float(timeout_s)
        self._reset_next = True
        self._step_index = 0
        self._session_index = 0

    @property
    def base_url(self) -> str:
        return f"http://{self.endpoint.host}:{self.endpoint.port}"

    def health(self) -> dict[str, Any]:
        """Return a validated ``GET /health`` response."""

        try:
            with urlopen(f"{self.base_url}/health", timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode())
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise Psi0ServerError(
                f"Psi0 /health failed at {self.base_url}/health: {exc}"
            ) from exc
        if not isinstance(body, dict) or body.get("status") != "ok":
            raise Psi0ServerError(f"Psi0 /health is not ok: {body!r}")
        return body

    def reset(self) -> None:
        """Mark the next /act as a new SIMPLE episode (history.reset)."""

        self._reset_next = True
        self._step_index = 0
        self._session_index += 1

    def predict_action(
        self,
        *,
        image: dict[str, np.ndarray],
        instruction: str,
        state: dict[str, np.ndarray],
    ) -> np.ndarray:
        """POST official /act and return the denormalized action chunk."""

        history: dict[str, Any] = {
            "session_id": f"psi0-{os.getpid()}-{self._session_index}",
            "episode_index": 0,
            "step_index": int(self._step_index),
        }
        if self._reset_next:
            history["reset"] = True
            self._reset_next = False
        payload = serialize_request(
            image=image,
            instruction=instruction,
            history=history,
            state=state,
        )
        request = Request(
            f"{self.base_url}/act",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode()
                status = getattr(response, "status", 200)
        except HTTPError as exc:
            detail = exc.read().decode() if exc.fp is not None else str(exc)
            raise Psi0ServerError(f"Psi0 /act HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            raise Psi0ServerError(
                f"Psi0 /act failed at {self.base_url}/act: {exc}"
            ) from exc
        if status != 200:
            raise Psi0ServerError(f"Psi0 /act HTTP {status}: {raw}")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise Psi0ServerError(f"Psi0 /act returned non-JSON: {raw!r}") from exc
        if isinstance(body, str):
            raise Psi0ServerError(f"Psi0 /act returned an error string: {body}")
        action = deserialize_response(body)
        self._step_index += 1
        return action


def _official_timestamp() -> str:
    # The official server expects this local wall-clock string shape.
    return str(datetime.now()).replace(" ", "_").replace(":", "-")  # noqa: DTZ005
