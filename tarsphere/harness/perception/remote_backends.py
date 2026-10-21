"""Network clients for shared perception model services."""

from __future__ import annotations

import os
import socket
import struct
import threading
from typing import Any

import numpy as np

from tarsphere.harness.perception.contracts import (
    Detection2D,
    PerceptionInput,
    PerceptionOutput,
)
from tarsphere.harness.perception.service_protocol import (
    PROTOCOL_VERSION,
    decode_message,
    encode_message,
)

_HEADER = struct.Struct("!Q")
_DEFAULT_SAM3_HOST = "127.0.0.1"
_DEFAULT_SAM3_PORT = 5558
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 300.0
_MAX_MESSAGE_BYTES = 256 * 1024 * 1024


def _read_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError(
                "SAM3 service connection closed before the response completed"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv(connection: socket.socket) -> dict[str, Any]:
    size = _HEADER.unpack(_read_exact(connection, _HEADER.size))[0]
    if size > _MAX_MESSAGE_BYTES:
        raise ValueError(
            f"SAM3 service response is {size} bytes; maximum is {_MAX_MESSAGE_BYTES}"
        )
    return decode_message(_read_exact(connection, size))


def _send(connection: socket.socket, message: dict[str, Any]) -> None:
    payload = encode_message(message)
    if len(payload) > _MAX_MESSAGE_BYTES:
        raise ValueError(
            f"SAM3 service request is {len(payload)} bytes; maximum is "
            f"{_MAX_MESSAGE_BYTES}"
        )
    connection.sendall(_HEADER.pack(len(payload)) + payload)


class SAM3ServiceClient:
    """Synchronous client for the one host-level SAM3 service.

    Every Isaac process owns a lightweight client. The service owns the only
    model instance and serializes GPU inference, so fifteen episode workers do
    not create fifteen checkpoint copies.
    """

    def __init__(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.host = host or os.environ.get("TARSPHERE_SAM3_HOST", _DEFAULT_SAM3_HOST)
        self.port = int(
            port
            if port is not None
            else os.environ.get("TARSPHERE_SAM3_PORT", _DEFAULT_SAM3_PORT)
        )
        self.timeout_seconds = float(
            timeout_seconds
            if timeout_seconds is not None
            else os.environ.get(
                "TARSPHERE_SAM3_REQUEST_TIMEOUT_SECONDS",
                _DEFAULT_REQUEST_TIMEOUT_SECONDS,
            )
        )
        if not self.host.strip():
            raise ValueError("TARSPHERE_SAM3_HOST must not be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("TARSPHERE_SAM3_PORT must be in 1..65535")
        if self.timeout_seconds <= 0:
            raise ValueError("TARSPHERE_SAM3_REQUEST_TIMEOUT_SECONDS must be positive")
        self._lock = threading.Lock()

    def request(self, message: dict[str, Any]) -> Any:
        with self._lock:
            try:
                with socket.create_connection(
                    (self.host, self.port), timeout=self.timeout_seconds
                ) as connection:
                    connection.settimeout(self.timeout_seconds)
                    _send(connection, message)
                    response = _recv(connection)
            except OSError as exc:
                raise RuntimeError(
                    f"SAM3 service unavailable at {self.host}:{self.port}: {exc}"
                ) from exc

        if not response.get("ok"):
            trace = response.get("traceback") or ""
            error_type = response.get("error_type") or "RuntimeError"
            error = response.get("error") or "SAM3 service request failed"
            raise RuntimeError(f"SAM3 service {error_type}: {error}\n{trace}")
        return response.get("result")


class RemoteSAM3PerceptionBackend:
    """SAM3 perception backend backed by the shared host service."""

    name = "sam3"

    def __init__(self, *, client: SAM3ServiceClient | None = None) -> None:
        self._client = client or SAM3ServiceClient()

    def reset(self) -> None:
        # SAM3 has no episode state. X-NavDP, not SAM3, owns trajectory history.
        return None

    def require_ready(self) -> None:
        """Require the server to have loaded the real checkpoint."""
        identity = self._client.request({"cmd": "sam3_require_ready"})
        if not isinstance(identity, dict) or identity.get("status") != "ok":
            raise RuntimeError(f"SAM3 returned invalid readiness: {identity!r}")
        if identity.get("service") != "tarsphere.sam3":
            raise RuntimeError(f"SAM3 service identity mismatch: {identity!r}")
        if identity.get("protocol") != PROTOCOL_VERSION:
            raise RuntimeError(f"SAM3 protocol identity mismatch: {identity!r}")
        if int(identity.get("checkpoint_size") or 0) <= 0:
            raise RuntimeError(f"SAM3 checkpoint identity is invalid: {identity!r}")

    def process(self, data: PerceptionInput) -> PerceptionOutput:
        observation = data.observation
        ego = observation.views.get("ego")
        rgb = ego.rgb if ego is not None else None
        if rgb is None:
            raise ValueError("SAM3 perception requires observation.views['ego'].rgb.")

        prompts = list(data.text_prompts or [])
        if not prompts:
            raise ValueError(
                "SAM3 perception requires at least one text prompt; the caller "
                "must name what to detect."
            )

        return self._client.request(
            {
                "cmd": "sam3_process",
                "actor_id": getattr(observation, "actor_id", "robot"),
                "rgb": np.asarray(rgb),
                "depth": ego.depth,
                "camera_params": ego.camera_params,
                "pose": getattr(observation, "pose", None),
                "text": getattr(observation, "text", ""),
                "text_prompts": prompts,
                "metadata": dict(data.metadata or {}),
            }
        )


__all__ = [
    "Detection2D",
    "RemoteSAM3PerceptionBackend",
    "SAM3ServiceClient",
]
