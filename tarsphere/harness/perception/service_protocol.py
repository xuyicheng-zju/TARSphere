"""Non-executable wire codec for the local SAM3 service."""

from __future__ import annotations

import base64
import json
import math
from typing import Any

import numpy as np

from tarsphere.harness.perception.contracts import Detection2D, PerceptionOutput

PROTOCOL_VERSION = "tarsphere.sam3.json.v1"
_ARRAY_TAG = "ndarray"
_DETECTION_TAG = "detection2d"
_OUTPUT_TAG = "perception_output"


def encode_message(value: dict[str, Any]) -> bytes:
    envelope = {
        "protocol": PROTOCOL_VERSION,
        "payload": _encode(value),
    }
    return json.dumps(
        envelope,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def decode_message(payload: bytes) -> dict[str, Any]:
    try:
        envelope = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("SAM3 message is not valid UTF-8 JSON") from exc
    if not isinstance(envelope, dict) or set(envelope) != {"protocol", "payload"}:
        raise ValueError("SAM3 message envelope has invalid fields")
    if envelope["protocol"] != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported SAM3 protocol: {envelope['protocol']!r}")
    value = _decode(envelope["payload"])
    if not isinstance(value, dict):
        raise TypeError("SAM3 message payload must be an object")
    return value


def _encode(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return _encode(value.item())
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.kind not in {"b", "i", "u", "f"}:
            raise TypeError(f"Unsupported SAM3 ndarray dtype: {array.dtype}")
        return {
            "__type__": _ARRAY_TAG,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": base64.b64encode(array.tobytes()).decode("ascii"),
        }
    if isinstance(value, Detection2D):
        return {
            "__type__": _DETECTION_TAG,
            "label": value.label,
            "score": value.score,
            "box_xyxy": list(value.box_xyxy),
            "mask": _encode(value.mask),
            "source": value.source,
        }
    if isinstance(value, PerceptionOutput):
        return {
            "__type__": _OUTPUT_TAG,
            "actor_id": value.actor_id,
            "detections": _encode(value.detections),
            "semantic_facts": _encode(value.semantic_facts),
            "tool_trace": _encode(value.tool_trace),
            "metadata": _encode(value.metadata),
        }
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("SAM3 protocol object keys must be strings")
        return {key: _encode(item) for key, item in value.items()}
    raise TypeError(f"Unsupported SAM3 protocol value: {type(value).__name__}")


def _decode(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        raise TypeError(f"Invalid SAM3 protocol value: {type(value).__name__}")
    tag = value.get("__type__")
    if tag is None:
        return {str(key): _decode(item) for key, item in value.items()}
    if tag == _ARRAY_TAG:
        if set(value) != {"__type__", "dtype", "shape", "data"}:
            raise ValueError("SAM3 ndarray payload has invalid fields")
        dtype = np.dtype(value["dtype"])
        if dtype.kind not in {"b", "i", "u", "f"} or dtype.hasobject:
            raise ValueError(f"Unsupported SAM3 ndarray dtype: {dtype}")
        shape = tuple(int(size) for size in value["shape"])
        if len(shape) > 4 or any(size < 0 for size in shape):
            raise ValueError(f"Invalid SAM3 ndarray shape: {shape!r}")
        raw = base64.b64decode(value["data"], validate=True)
        expected = math.prod(shape) * dtype.itemsize
        if len(raw) != expected:
            raise ValueError(
                f"SAM3 ndarray byte count mismatch: expected {expected}, got {len(raw)}"
            )
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if tag == _DETECTION_TAG:
        if set(value) != {"__type__", "label", "score", "box_xyxy", "mask", "source"}:
            raise ValueError("SAM3 detection payload has invalid fields")
        box = tuple(float(item) for item in value["box_xyxy"])
        if len(box) != 4:
            raise ValueError("SAM3 detection box must contain four coordinates")
        return Detection2D(
            label=str(value["label"]),
            score=float(value["score"]),
            box_xyxy=box,
            mask=_decode(value["mask"]),
            source=str(value["source"]),
        )
    if tag == _OUTPUT_TAG:
        if set(value) != {
            "__type__",
            "actor_id",
            "detections",
            "semantic_facts",
            "tool_trace",
            "metadata",
        }:
            raise ValueError("SAM3 perception output has invalid fields")
        return PerceptionOutput(
            actor_id=str(value["actor_id"]),
            detections=list(_decode(value["detections"])),
            semantic_facts=[str(item) for item in _decode(value["semantic_facts"])],
            tool_trace=list(_decode(value["tool_trace"])),
            metadata=dict(_decode(value["metadata"])),
        )
    raise ValueError(f"Unknown SAM3 protocol type tag: {tag!r}")


__all__ = ["PROTOCOL_VERSION", "decode_message", "encode_message"]
