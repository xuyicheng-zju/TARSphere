"""One process-wide SAM3 model shared by all local Isaac workers."""

from __future__ import annotations

import os
import socketserver
import struct
import threading
import traceback
from typing import Any

from tarsphere.harness.core.types import AgentObservation, CameraViewObservation
from tarsphere.harness.perception.service_protocol import (
    PROTOCOL_VERSION,
    decode_message,
    encode_message,
)

_HEADER = struct.Struct("!Q")
_MAX_MESSAGE_BYTES = 256 * 1024 * 1024


def _read_exact(stream: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("SAM3 client disconnected before the request completed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv(stream: Any) -> dict[str, Any]:
    size = _HEADER.unpack(_read_exact(stream, _HEADER.size))[0]
    if size > _MAX_MESSAGE_BYTES:
        raise ValueError(
            f"SAM3 service request is {size} bytes; maximum is {_MAX_MESSAGE_BYTES}"
        )
    return decode_message(_read_exact(stream, size))


def _send(stream: Any, message: dict[str, Any]) -> None:
    payload = encode_message(message)
    if len(payload) > _MAX_MESSAGE_BYTES:
        raise ValueError(
            f"SAM3 service response is {len(payload)} bytes; maximum is "
            f"{_MAX_MESSAGE_BYTES}"
        )
    stream.write(_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def _observation_from_request(request: dict[str, Any]) -> AgentObservation:
    ego = CameraViewObservation(
        name="ego",
        rgb=request.get("rgb"),
        depth=request.get("depth"),
        camera_params=dict(request.get("camera_params") or {}),
    )
    return AgentObservation(
        actor_id=request.get("actor_id") or "robot",
        views={"ego": ego},
        pose=request.get("pose"),
        text=request.get("text") or "",
    )


class SAM3ModelState:
    """The service-owned checkpoint configuration and loaded backend."""

    def __init__(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path
        self.backend: Any | None = None

    def require_ready(self) -> Any:
        if self.backend is None:
            from tarsphere.harness.perception.backends.sam3 import SAM3PerceptionBackend

            self.backend = SAM3PerceptionBackend(
                model_kwargs={
                    "checkpoint_path": self.checkpoint_path,
                    "load_from_HF": False,
                    "enable_segmentation": True,
                    "enable_inst_interactivity": False,
                }
            )
        self.backend.require_ready()
        return self.backend


def handle_request(state: SAM3ModelState, request: dict[str, Any]) -> Any:
    cmd = request.get("cmd")
    if cmd == "ping":
        return {
            "status": "ok",
            "service": "tarsphere.sam3",
            "protocol": PROTOCOL_VERSION,
            "pid": os.getpid(),
        }
    if cmd == "sam3_require_ready":
        state.require_ready()
        checkpoint = os.path.realpath(state.checkpoint_path)
        return {
            "status": "ok",
            "service": "tarsphere.sam3",
            "protocol": PROTOCOL_VERSION,
            "checkpoint_path": checkpoint,
            "checkpoint_size": os.path.getsize(checkpoint),
        }
    if cmd == "sam3_process":
        from tarsphere.harness.perception.contracts import PerceptionInput

        backend = state.require_ready()
        return backend.process(
            PerceptionInput(
                observation=_observation_from_request(request),
                text_prompts=list(request.get("text_prompts") or []),
                metadata=dict(request.get("metadata") or {}),
            )
        )
    raise ValueError(f"Unknown SAM3 service command: {cmd!r}")


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            request = _recv(self.rfile)
            with self.server.inference_lock:  # type: ignore[attr-defined]
                result = handle_request(
                    self.server.model_state,
                    request,  # type: ignore[attr-defined]
                )
        except Exception as exc:  # noqa: BLE001 - returned to the local client
            response = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        else:
            response = {"ok": True, "result": result}
        _send(self.wfile, response)


class SAM3RequestServer(socketserver.ThreadingTCPServer):
    """Threaded admission with one serialized GPU inference section."""

    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 64

    def __init__(self, address: tuple[str, int], model_state: SAM3ModelState):
        self.model_state = model_state
        self.inference_lock = threading.Lock()
        super().__init__(address, _RequestHandler)


__all__ = ["SAM3ModelState", "SAM3RequestServer", "handle_request"]
