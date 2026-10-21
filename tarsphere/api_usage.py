"""Canonical OpenAI-compatible API usage records for evaluated calls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)


def normalize_openai_usage(usage: Any) -> dict[str, int]:
    """Normalize one OpenAI-compatible usage object into Lingzhi token fields."""

    if usage is None:
        payload: Mapping[str, Any] = {}
    elif isinstance(usage, Mapping):
        payload = usage
    else:
        model_dump = getattr(usage, "model_dump", None)
        if not callable(model_dump):
            raise TypeError("API usage must be a mapping or expose model_dump().")
        dumped = model_dump()
        if not isinstance(dumped, Mapping):
            raise TypeError("API usage model_dump() must return a mapping.")
        payload = dumped

    prompt_details = payload.get("prompt_tokens_details") or {}
    completion_details = payload.get("completion_tokens_details") or {}
    if not isinstance(prompt_details, Mapping):
        raise TypeError("prompt_tokens_details must be a mapping when present.")
    if not isinstance(completion_details, Mapping):
        raise TypeError("completion_tokens_details must be a mapping when present.")

    normalized = {
        "input_tokens": _nonnegative_int(payload.get("prompt_tokens", 0)),
        "cached_input_tokens": _nonnegative_int(prompt_details.get("cached_tokens", 0)),
        "output_tokens": _nonnegative_int(payload.get("completion_tokens", 0)),
        "reasoning_tokens": _nonnegative_int(
            completion_details.get("reasoning_tokens", 0)
        ),
        "total_tokens": _nonnegative_int(payload.get("total_tokens", 0)),
    }
    expected_total = normalized["input_tokens"] + normalized["output_tokens"]
    if normalized["total_tokens"] == 0 and expected_total:
        normalized["total_tokens"] = expected_total
    if normalized["total_tokens"] != expected_total:
        raise ValueError(
            "Lingzhi token usage is inconsistent: total_tokens must equal "
            "prompt_tokens + completion_tokens."
        )
    if normalized["cached_input_tokens"] > normalized["input_tokens"]:
        raise ValueError("cached input tokens cannot exceed input tokens.")
    if normalized["reasoning_tokens"] > normalized["output_tokens"]:
        raise ValueError("reasoning tokens cannot exceed output tokens.")
    return normalized


def aggregate_model_calls(
    calls: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate auditable model-call records without inventing prices."""

    report: dict[str, Any] = {
        "gateway": "",
        "logical_calls": 0,
        "request_attempts": 0,
        "provider_responses": 0,
        # Responses excluded from token subtotals because their usage was
        # absent or invalid. Request/response counts still include them.
        "unavailable_usage_responses": 0,
        "latency_s": 0.0,
        **{field: 0 for field in _TOKEN_FIELDS},
        "by_model": {},
    }
    gateways: set[str] = set()
    for call in calls:
        gateway = str(call.get("gateway") or "").strip()
        if not gateway:
            raise ValueError(
                "Every model call in an official run must name its gateway."
            )
        gateways.add(gateway)
        model = str(call.get("model") or "").strip()
        if not model:
            raise ValueError("Every model call must name its model.")
        usage = call.get("usage") or {}
        if not isinstance(usage, Mapping):
            raise TypeError("model call usage must be a mapping.")
        per_model = report["by_model"].setdefault(
            model,
            {
                "logical_calls": 0,
                "request_attempts": 0,
                "provider_responses": 0,
                "unavailable_usage_responses": 0,
                "latency_s": 0.0,
                **{field: 0 for field in _TOKEN_FIELDS},
            },
        )
        for target in (report, per_model):
            target["logical_calls"] += 1
            target["request_attempts"] += _nonnegative_int(
                call.get("request_attempts", 0)
            )
            target["provider_responses"] += _nonnegative_int(
                call.get("provider_responses", 0)
            )
            target["unavailable_usage_responses"] += _nonnegative_int(
                call.get("unavailable_usage_responses", 0)
            )
            target["latency_s"] += _nonnegative_float(call.get("latency_s", 0.0))
            for field in _TOKEN_FIELDS:
                target[field] += _nonnegative_int(usage.get(field, 0))
    report["gateway"] = (
        "none"
        if not gateways
        else next(iter(gateways))
        if len(gateways) == 1
        else "mixed"
    )
    report["latency_s"] = round(report["latency_s"], 3)
    for per_model in report["by_model"].values():
        per_model["latency_s"] = round(per_model["latency_s"], 3)
    return report


def sum_token_usage(usages: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Sum canonical token dictionaries from provider responses."""

    total = {field: 0 for field in _TOKEN_FIELDS}
    for usage in usages:
        for field in _TOKEN_FIELDS:
            total[field] += _nonnegative_int(usage.get(field, 0))
    return total


def merge_usage_reports(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge Agent and evaluator aggregates into one episode API total."""

    for report in reports:
        if not str(report.get("gateway") or "").strip():
            raise ValueError("Every official usage report must name its gateway.")
        by_model = report.get("by_model") or {}
        if not isinstance(by_model, Mapping):
            raise TypeError("usage report by_model must be a mapping.")
        for usage in by_model.values():
            if not isinstance(usage, Mapping):
                raise TypeError("per-model usage must be a mapping.")

    # Aggregate each model subtotal once, then restore the authoritative
    # logical-call count because the subtotal is not itself one API call.
    compact: list[dict[str, Any]] = []
    for report in reports:
        by_model = report.get("by_model") or {}
        for model, usage in by_model.items():
            compact.append(
                {
                    "gateway": str(report.get("gateway")),
                    "model": str(model),
                    "request_attempts": int(usage.get("request_attempts", 0)),
                    "provider_responses": int(usage.get("provider_responses", 0)),
                    "unavailable_usage_responses": int(
                        usage.get("unavailable_usage_responses", 0)
                    ),
                    "latency_s": float(usage.get("latency_s", 0.0)),
                    "usage": {
                        field: int(usage.get(field, 0)) for field in _TOKEN_FIELDS
                    },
                }
            )
    merged = aggregate_model_calls(compact)
    merged["logical_calls"] = sum(
        int(report.get("logical_calls", 0)) for report in reports
    )
    for model, per_model in merged["by_model"].items():
        per_model["logical_calls"] = sum(
            int((report.get("by_model") or {}).get(model, {}).get("logical_calls", 0))
            for report in reports
        )
    return merged


def _nonnegative_int(value: Any) -> int:
    result = int(value or 0)
    if result < 0:
        raise ValueError("API usage counters must be non-negative.")
    return result


def _nonnegative_float(value: Any) -> float:
    result = float(value or 0.0)
    if result < 0.0:
        raise ValueError("API usage durations must be non-negative.")
    return result


__all__ = [
    "aggregate_model_calls",
    "merge_usage_reports",
    "normalize_openai_usage",
    "sum_token_usage",
]
