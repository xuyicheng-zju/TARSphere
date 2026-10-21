"""Shared OpenAI-compatible request rules for configured model endpoints."""

from __future__ import annotations

# These routes reject or do not reliably honor provider-side JSON mode. Local
# schema validation remains authoritative; only the transport hint is omitted.
PROMPT_ONLY_JSON_MODELS = frozenset(
    {
        "qwen3.7-max-2026-06-08",
        "qwen3.7-plus",
    }
)

# The Lingzhi Bedrock route rejects any explicit temperature for this model.
PROVIDER_DEFAULT_TEMPERATURE_MODELS = frozenset(
    {
        "claude-sonnet-5",
    }
)

# These hybrid models default to a large reasoning budget. Evaluation support
# is constrained classification/Judge work, so disable hidden reasoning to
# bound latency and token spend while retaining provider JSON mode.
NON_THINKING_EVALUATION_MODELS = frozenset(
    {
        "qwen3.8-flash",
    }
)


def add_json_request_controls(
    request: dict,
    *,
    model: str,
    temperature: float,
    seed: int | None = None,
) -> None:
    """Apply only provider-supported deterministic JSON request controls."""

    if model not in PROVIDER_DEFAULT_TEMPERATURE_MODELS:
        request["temperature"] = float(temperature)
        if seed is not None:
            request["seed"] = int(seed)
    if model not in PROMPT_ONLY_JSON_MODELS:
        request["response_format"] = {"type": "json_object"}
    if model in NON_THINKING_EVALUATION_MODELS:
        request["extra_body"] = {
            **dict(request.get("extra_body") or {}),
            "enable_thinking": False,
        }


__all__ = [
    "PROMPT_ONLY_JSON_MODELS",
    "PROVIDER_DEFAULT_TEMPERATURE_MODELS",
    "NON_THINKING_EVALUATION_MODELS",
    "add_json_request_controls",
]
