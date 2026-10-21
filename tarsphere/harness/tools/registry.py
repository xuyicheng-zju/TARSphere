"""Tool registry: dispatches VLM tool calls to handlers."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from tarsphere.harness.core.failures import (
    FailureReason,
    tool_failure,
    unexpected_tool_failure,
)
from tarsphere.harness.core.types import ToolCall, ToolHandler, ToolResult, ToolSpec


class ToolRegistry:
    """Registers tool specs and dispatches calls to handlers.

    Each tool name maps to (spec, handler). The registry exposes specs to the
    VLM prompt and executes calls during the turn loop.
    """

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, ToolHandler] = {}
        self._temporarily_unavailable: Callable[[str], bool] | None = None

    def configure_disruption(self, unavailable: Callable[[str], bool]) -> None:
        """Bind the live simulator clock without changing the visible tool set."""
        self._temporarily_unavailable = unavailable

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Duplicate tool registration: {spec.name!r}")
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    @property
    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def spec_names(self) -> list[str]:
        return list(self._specs.keys())

    def spec(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def handler(self, name: str) -> ToolHandler | None:
        """The handler registered for ``name``, or None when the tool is unwired.

        Entrypoint bootstrap uses this to reach a tool's backend through the
        same object the model's calls are dispatched to, so warmup cannot
        certify a backend the model would not actually reach.
        """
        return self._handlers.get(name)

    def execute(self, call: ToolCall, context: dict[str, Any]) -> ToolResult:
        spec = self._specs.get(call.tool)
        handler = self._handlers.get(call.tool)
        if spec is None or handler is None:
            return tool_failure(
                call,
                FailureReason.UNKNOWN_VALUE,
                f"Unknown tool: {call.tool}.",
            )
        invalid = _validate_call_arguments(call, spec, all_specs=self._specs)
        if invalid is not None:
            return invalid
        if self._temporarily_unavailable is not None and self._temporarily_unavailable(
            call.tool
        ):
            return tool_failure(
                call,
                FailureReason.TEMPORARILY_UNAVAILABLE,
                f"{call.tool} is temporarily unavailable.",
            )
        try:
            return handler.execute(call, context)
        except Exception as exc:  # noqa: BLE001 - contain one broken capability
            # Preserve the tool identity and exception type at this boundary.
            # The episode runner treats this closed-vocabulary internal_error
            # as evaluation infrastructure failure, so it cannot be scored as
            # model behaviour or silently consume the decision budget.
            # SystemExit/KeyboardInterrupt deliberately remain fatal.
            return unexpected_tool_failure(call, exc)

    def validate(self, call: ToolCall) -> ToolResult | None:
        """Validate one call without invoking its handler.

        The kernel uses this preflight for every member of an action block
        before any handler can stage or commit state. This prevents a malformed
        suffix from partially resolving an earlier call.
        """

        spec = self._specs.get(call.tool)
        handler = self._handlers.get(call.tool)
        if spec is None or handler is None:
            return tool_failure(
                call,
                FailureReason.UNKNOWN_VALUE,
                f"Unknown tool: {call.tool}.",
            )
        return _validate_call_arguments(call, spec, all_specs=self._specs)


def _validate_call_arguments(
    call: ToolCall,
    spec: ToolSpec,
    *,
    all_specs: dict[str, ToolSpec] | None = None,
) -> ToolResult | None:
    """Enforce the same argument schema that the model reads in the prompt."""

    schema = spec.input_schema
    properties = schema.get("properties") or {}
    required = tuple(schema.get("required") or ())
    missing = [name for name in required if name not in call.args]
    if missing:
        return tool_failure(
            call,
            FailureReason.MISSING_ARGUMENT,
            f"{call.tool} requires argument(s): {', '.join(missing)}.",
        )
    if schema.get("additionalProperties") is False:
        unexpected = sorted(set(call.args) - set(properties))
        if unexpected:
            owners = sorted(
                {
                    other.name
                    for other in (all_specs or {}).values()
                    for arg in unexpected
                    if arg in (other.input_schema.get("properties") or {})
                }
            )
            hint = f" (use {', '.join(owners)} instead)" if owners else ""
            return tool_failure(
                call,
                FailureReason.MALFORMED_ARGUMENT,
                f"{call.tool} does not accept argument(s): "
                f"{', '.join(unexpected)}{hint}.",
            )

    for name, value in call.args.items():
        rule = properties.get(name)
        if not isinstance(rule, dict):
            continue
        violation = _schema_violation(value, rule, path=f"`{name}`")
        if violation is not None:
            reason, message = violation
            return tool_failure(call, reason, f"{call.tool} {message}")
    return None


def _schema_violation(
    value: Any,
    rule: dict[str, Any],
    *,
    path: str,
) -> tuple[str, str] | None:
    """Validate the JSON-Schema subset used by public tool specifications."""

    expected = str(rule.get("type") or "")
    if not _matches_json_type(value, expected):
        return (
            FailureReason.MALFORMED_ARGUMENT,
            f"{path} must be {expected}; got {type(value).__name__}.",
        )
    allowed_values = rule.get("enum")
    if allowed_values is not None and value not in allowed_values:
        return (
            FailureReason.UNKNOWN_VALUE,
            f"{path} must be one of {list(allowed_values)}; got {value!r}.",
        )
    if expected in {"number", "integer"}:
        number = float(value)
        if not math.isfinite(number):
            return FailureReason.MALFORMED_ARGUMENT, f"{path} must be finite."
        minimum = rule.get("minimum")
        maximum = rule.get("maximum")
        if minimum is not None and number < float(minimum):
            return (
                FailureReason.MALFORMED_ARGUMENT,
                f"{path} must be at least {minimum}; got {number}.",
            )
        if maximum is not None and number > float(maximum):
            return (
                FailureReason.MALFORMED_ARGUMENT,
                f"{path} must be at most {maximum}; got {number}.",
            )
    if expected == "string":
        minimum_length = rule.get("minLength")
        maximum_length = rule.get("maxLength")
        if minimum_length is not None and len(value) < int(minimum_length):
            return (
                FailureReason.MALFORMED_ARGUMENT,
                f"{path} needs at least {minimum_length} character(s).",
            )
        if maximum_length is not None and len(value) > int(maximum_length):
            return (
                FailureReason.MALFORMED_ARGUMENT,
                f"{path} allows at most {maximum_length} character(s).",
            )
    if expected == "array":
        minimum_items = rule.get("minItems")
        maximum_items = rule.get("maxItems")
        if minimum_items is not None and len(value) < int(minimum_items):
            return (
                FailureReason.MALFORMED_ARGUMENT,
                f"{path} needs at least {minimum_items} item(s).",
            )
        if maximum_items is not None and len(value) > int(maximum_items):
            return (
                FailureReason.MALFORMED_ARGUMENT,
                f"{path} allows at most {maximum_items} item(s).",
            )
        item_rule = rule.get("items") or {}
        if isinstance(item_rule, dict):
            for index, item in enumerate(value):
                violation = _schema_violation(
                    item,
                    item_rule,
                    path=f"{path}[{index}]",
                )
                if violation is not None:
                    return violation
    return None


def _matches_json_type(value: Any, expected: str) -> bool:
    """Match the JSON types used by the shipped tool schemas."""

    if expected == "number":
        return not isinstance(value, bool) and isinstance(value, (int, float))
    if expected == "integer":
        return not isinstance(value, bool) and isinstance(value, int)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "boolean":
        return isinstance(value, bool)
    return True


__all__ = ["ToolRegistry"]
