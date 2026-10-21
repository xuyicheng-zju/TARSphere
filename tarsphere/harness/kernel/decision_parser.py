"""Strict parser for one model-authored decision envelope.

The model receives the complete JSON contract in its prompt. This boundary
does not repair fences, prose, aliases, flattened arguments, unknown fields,
or malformed optional state. A format error is one rejected policy decision
and its diagnostic becomes feedback for the next decision.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from tarsphere.contracts.structured_plan import StructuredPlan

from tarsphere.harness.core.types import DecisionEnvelope, ToolCall
from tarsphere.harness.memory.key import MAX_KEY_MEMORY_CONTENT_CHARS


class DecisionParseError(ValueError):
    """Raised when a model output violates the decision contract."""


_TOP_LEVEL_FIELDS = frozenset(
    {
        "thinking",
        "actions",
        "memory",
        "plan",
        "subgoal",
        "inspection",
    }
)
_MEMORY_OPERATIONS = frozenset({"remember", "revise", "forget"})


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Require the complete response to be exactly one JSON object."""

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DecisionParseError(
            "decision must be exactly one valid JSON object: "
            f"{exc.msg} at character {exc.pos}; "
            f"response_start={raw[:160]!r}; response_end={raw[-160:]!r}"
        ) from exc
    if not isinstance(value, dict):
        raise DecisionParseError(
            f"decision must be a JSON object, got {type(value).__name__}"
        )
    unknown = set(value) - _TOP_LEVEL_FIELDS
    if unknown:
        raise DecisionParseError(f"decision has unknown fields: {sorted(unknown)}")
    return value


def _parse_action(value: Any, *, index: int) -> ToolCall:
    if not isinstance(value, dict):
        raise DecisionParseError(
            f'actions[{index}] must be an object {{"tool","args"}}, '
            f"got {type(value).__name__}"
        )
    fields = set(value)
    if fields != {"tool", "args"}:
        missing = {"tool", "args"} - fields
        unknown = fields - {"tool", "args"}
        details = []
        if missing:
            details.append(f"missing fields {sorted(missing)}")
        if unknown:
            details.append(f"unknown fields {sorted(unknown)}")
        raise DecisionParseError(
            f'actions[{index}] must contain exactly "tool" and "args": '
            + "; ".join(details)
        )
    tool = value["tool"]
    if not isinstance(tool, str) or not tool.strip():
        raise DecisionParseError(f"actions[{index}].tool must be a non-empty string")
    args = value["args"]
    if not isinstance(args, dict):
        raise DecisionParseError(
            f"actions[{index}].args must be an object, got {type(args).__name__}"
        )
    return ToolCall(tool=tool.strip(), args=dict(args))


def _parse_memory(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise DecisionParseError('memory must be a list of {"op","note"} objects')
    updates: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise DecisionParseError(
                f'memory[{index}] must be an object {{"op","note"}}'
            )
        if set(item) != {"op", "note"}:
            raise DecisionParseError(
                f'memory[{index}] must contain exactly "op" and "note"'
            )
        op = item["op"]
        if op not in _MEMORY_OPERATIONS:
            raise DecisionParseError(
                f"memory[{index}].op must be one of {sorted(_MEMORY_OPERATIONS)}"
            )
        note = item["note"]
        if not isinstance(note, str) or not note.strip():
            raise DecisionParseError(f"memory[{index}].note must be a non-empty string")
        note = " ".join(note.split())
        if len(note) > MAX_KEY_MEMORY_CONTENT_CHARS:
            raise DecisionParseError(
                f"memory[{index}].note exceeds "
                f"{MAX_KEY_MEMORY_CONTENT_CHARS} characters"
            )
        updates.append({"op": op, "note": note})
    return updates


def _parse_plan(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        return StructuredPlan.model_validate(value).as_public_dict()
    except ValidationError as exc:
        raise DecisionParseError(f"invalid plan: {exc.errors()[0]['msg']}") from exc


def parse_decision(
    raw: str,
    *,
    action_horizon: int,
    tool_argument_names: Mapping[str, frozenset[str]] | None = None,
) -> DecisionEnvelope:
    """Parse one strict decision without repairing model output."""

    if not raw or not raw.strip():
        raise DecisionParseError("empty model output")
    value = _parse_json_object(raw)

    thinking = value.get("thinking")
    if not isinstance(thinking, str) or not thinking.strip():
        raise DecisionParseError("decision.thinking must be a non-empty string")
    actions = value.get("actions")
    if not isinstance(actions, list) or not actions:
        raise DecisionParseError("decision.actions must be a non-empty list")
    calls = [_parse_action(item, index=index) for index, item in enumerate(actions)]
    if tool_argument_names is not None:
        for index, call in enumerate(calls):
            allowed = tool_argument_names.get(call.tool)
            if allowed is None:
                continue
            unknown = set(call.args) - set(allowed)
            if unknown:
                raise DecisionParseError(
                    f"actions[{index}].args has unknown fields for "
                    f"{call.tool}: {sorted(unknown)}"
                )
    if len(calls) > int(action_horizon):
        raise DecisionParseError(
            f"actions length {len(calls)} exceeds Action Horizon {int(action_horizon)}"
        )

    subgoal = value.get("subgoal")
    if subgoal is not None and subgoal != "checked":
        raise DecisionParseError('subgoal must be exactly "checked"')
    inspection = value.get("inspection")
    if inspection is not None and (
        not isinstance(inspection, str) or not inspection.strip()
    ):
        raise DecisionParseError("inspection must be a non-empty string when supplied")

    envelope = DecisionEnvelope(
        thinking=thinking.strip(),
        actions=calls,
        memory=_parse_memory(value.get("memory")),
        plan=_parse_plan(value.get("plan")),
        subgoal_checked=subgoal == "checked",
        inspection=inspection.strip() if inspection is not None else None,
    )
    try:
        envelope.validate(action_horizon=action_horizon)
    except ValueError as exc:
        raise DecisionParseError(str(exc)) from exc
    return envelope


__all__ = ["DecisionParseError", "parse_decision"]
