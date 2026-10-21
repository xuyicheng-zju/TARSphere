"""Lossless, credential-free snapshots of the exact provider request.

Text and request controls stay in JSON; inline image bytes are stored once by
SHA-256. References are local to one physical attempt and never fetch a URL.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any


def _encoded(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def save_model_input_snapshot(
    directory: Path,
    *,
    index: int,
    request: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Persist before inference; return the hash-bound trace reference.

    ``request`` is the actual SDK create() payload, not client credentials.
    ``context`` binds the request to its policy/action indices. A failed write
    raises before any model call; a repeated index is never overwritten.
    """

    if set(request) - {
        "model",
        "messages",
        "max_tokens",
        "timeout",
        "temperature",
        "seed",
        "response_format",
        "extra_body",
    }:
        raise ValueError("Unexpected provider request fields in model-input snapshot")
    images: dict[str, dict[str, str]] = {}
    directory.mkdir(parents=True, exist_ok=True)

    def externalize(value):
        if isinstance(value, dict):
            if value.get("type") == "image_url":
                image_url = dict(value["image_url"])
                url = image_url["url"]
                if not isinstance(url, str) or not url.startswith("data:image/"):
                    raise ValueError("Model-input snapshots require inline image bytes")
                header, encoded = url.split(",", 1)
                if not header.endswith(";base64"):
                    raise ValueError("Model-input snapshot image must use base64")
                raw = base64.b64decode(encoded, validate=True)
                if base64.b64encode(raw).decode("ascii") != encoded:
                    raise ValueError(
                        "Model-input snapshot image is not canonical base64"
                    )
                digest = hashlib.sha256(raw).hexdigest()
                image_path = directory / "images" / digest
                image_path.parent.mkdir(exist_ok=True)
                if image_path.exists():
                    if image_path.read_bytes() != raw:
                        raise ValueError(
                            "Existing model-input image does not match its hash"
                        )
                else:
                    with image_path.open("xb") as stream:
                        stream.write(raw)
                reference = f"sha256:{digest}"
                # Keep MIME at the occurrence, so equal bytes with distinct
                # headers still reconstruct the original payload exactly.
                image_url["url"] = {"image_ref": reference, "header": header}
                images[reference] = {"path": f"images/{digest}", "sha256": digest}
                return {**value, "image_url": image_url}
            return {k: externalize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [externalize(v) for v in value]
        return value

    snapshot = {
        "schema_version": "tarsphere.model_input.v1",
        "context": context,
        "request_sha256": hashlib.sha256(_encoded(request)).hexdigest(),
        "request": externalize(request),
        "images": images,
    }
    raw = _encoded(snapshot)
    path = directory / f"{index:04d}.request.json"
    with path.open("xb") as stream:
        stream.write(raw)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "request_sha256": snapshot["request_sha256"],
    }


def load_model_input_snapshot(
    reference: dict[str, str], *, artifact_root: Path
) -> dict:
    """Verify all hashes/containment and reconstruct the SDK request offline."""

    root = artifact_root.resolve()
    path = Path(reference["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Model-input snapshot escapes its physical attempt")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("Model-input snapshot hash mismatch")
    snapshot = json.loads(raw)
    if snapshot["schema_version"] != "tarsphere.model_input.v1":
        raise ValueError("Unknown model-input snapshot schema")

    def restore(value):
        if isinstance(value, dict):
            if value.get("type") == "image_url":
                image_url = dict(value["image_url"])
                occurrence = image_url["url"]
                item = snapshot["images"][occurrence["image_ref"]]
                image_path = (path.parent / item["path"]).resolve()
                if not image_path.is_relative_to(path.parent):
                    raise ValueError("Model-input image escapes its snapshot")
                data = image_path.read_bytes()
                if hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise ValueError("Model-input image hash mismatch")
                image_url["url"] = (
                    occurrence["header"] + "," + base64.b64encode(data).decode("ascii")
                )
                return {**value, "image_url": image_url}
            return {k: restore(v) for k, v in value.items()}
        if isinstance(value, list):
            return [restore(v) for v in value]
        return value

    request = restore(snapshot["request"])
    digest = hashlib.sha256(_encoded(request)).hexdigest()
    if digest != snapshot["request_sha256"] or digest != reference["request_sha256"]:
        raise ValueError("Reconstructed provider request hash mismatch")
    return {"request": request, "context": snapshot["context"]}


def validate_model_input_trace(
    model_output_path: Path,
    *,
    trajectory: Any,
    artifact_root: Path,
    model_id: str,
) -> None:
    """Require one reconstructable input for each model decision, in order."""

    rows = [
        json.loads(line)
        for line in model_output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    policy_rows = [step for step in trajectory.steps if step.policy_inference]
    if len(rows) != len(policy_rows):
        raise ValueError(
            "Model-input/output trace does not cover every policy decision"
        )
    for policy_step, (row, step) in enumerate(zip(rows, policy_rows)):
        if (
            row.get("event") != "policy_decision"
            or row.get("policy_step") != policy_step
            or row.get("action_step") != step.decision_step
        ):
            raise ValueError(
                "Model-input/output trace has inconsistent policy/action indices"
            )
        calls = row.get("model_calls")
        if not isinstance(calls, list) or len(calls) != 1:
            raise ValueError(
                "Each policy decision must own exactly one logical model call"
            )
        restored = load_model_input_snapshot(
            calls[0]["input_snapshot"], artifact_root=artifact_root
        )
        if restored["context"] != {
            "policy_step": policy_step,
            "action_step": step.decision_step,
        }:
            raise ValueError(
                "Model-input snapshot has inconsistent policy/action indices"
            )
        if restored["request"].get("model") != model_id:
            raise ValueError(
                "Model-input snapshot belongs to a different evaluated model"
            )
