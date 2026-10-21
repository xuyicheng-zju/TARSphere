"""One deterministic semantic-judge client shared by TARSphere evaluators."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from tarsphere.api_usage import aggregate_model_calls, normalize_openai_usage
from tarsphere.configs.secrets import EvaluationSupportConfig
from tarsphere.provider_compat import add_json_request_controls

JUDGE_TRANSIENT_MAX_ATTEMPTS = 2
# One initial request plus two bounded correction attempts.  A retry that does
# not explain the schema error merely asks the model to repeat the same
# mistake, so ``complete_model`` feeds the exact local validation failure back
# without changing the underlying evidence payload.
JUDGE_SCHEMA_MAX_ATTEMPTS = 3
JUDGE_SEED = 0
ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
JSON_RESPONSE_INSTRUCTION = (
    "Return exactly one JSON object matching the requested response schema."
)


class JudgeTruncationError(RuntimeError):
    """Raised when a provider stops before returning the complete JSON."""


JudgeConfig = EvaluationSupportConfig


class JudgeClient:
    """Return strict JSON from one fixed model under a constrained schema."""

    def __init__(self, config: JudgeConfig) -> None:
        if not config.api_key.strip():
            raise ValueError("TARSphere judge requires a non-empty API key")
        if not config.model.strip():
            raise ValueError("TARSphere judge requires a fixed model name")
        self.config = config
        self.model_calls: list[dict[str, Any]] = []

    @classmethod
    def from_environment(cls) -> JudgeClient:
        return cls(JudgeConfig.from_environment())

    def complete_json(
        self,
        *,
        system_prompt: str,
        payload: Mapping[str, Any],
        max_tokens: int,
    ) -> str:
        """Run one deterministic judge request and return its raw JSON text."""

        from openai import (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            OpenAI,
            RateLimitError,
        )

        client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url or None,
            timeout=self.config.timeout_seconds,
            max_retries=0,
        )
        started = time.monotonic()
        messages = (
            {
                "role": "system",
                "content": f"{system_prompt}\n{JSON_RESPONSE_INSTRUCTION}",
            },
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        )
        transient_errors: list[str] = []
        response = None
        attempts = 0
        for attempts in range(1, JUDGE_TRANSIENT_MAX_ATTEMPTS + 1):
            try:
                request: dict[str, Any] = {
                    "model": self.config.model,
                    "max_tokens": max_tokens,
                    "messages": messages,
                }
                add_json_request_controls(
                    request,
                    model=self.config.model,
                    temperature=0.0,
                    seed=JUDGE_SEED,
                )
                response = client.chat.completions.create(**request)
                break
            except (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            ) as exc:
                transient_errors.append(type(exc).__name__)
                if attempts >= JUDGE_TRANSIENT_MAX_ATTEMPTS:
                    self.model_calls.append(
                        {
                            "gateway": "lingzhi",
                            "model": self.config.model,
                            "request_attempts": attempts,
                            "provider_responses": 0,
                            "latency_s": round(time.monotonic() - started, 3),
                            "usage": normalize_openai_usage(None),
                            "responses": [],
                            "transport_failures": transient_errors,
                        }
                    )
                    raise
                time.sleep(float(attempts))
        if response is None:
            raise RuntimeError("TARSphere judge completed without a response.")
        usage = normalize_openai_usage(getattr(response, "usage", None))
        self.model_calls.append(
            {
                "gateway": "lingzhi",
                "model": self.config.model,
                "request_attempts": attempts,
                "provider_responses": 1,
                "latency_s": round(time.monotonic() - started, 3),
                "usage": usage,
                "transport_failures": transient_errors,
                "responses": [
                    {
                        "finish_reason": getattr(
                            response.choices[0], "finish_reason", None
                        ),
                        "content_empty": not bool(
                            (response.choices[0].message.content or "").strip()
                        ),
                        "usage": usage,
                    }
                ],
            }
        )
        finish_reason = str(getattr(response.choices[0], "finish_reason", "") or "")
        if finish_reason in {"length", "max_tokens"}:
            raise JudgeTruncationError(
                f"TARSphere judge response was truncated at max_tokens={max_tokens}"
            )
        content = response.choices[0].message.content or ""
        if not content.strip():
            raise ValueError("TARSphere judge returned empty content")
        return _unwrap_json_code_fence(content)

    def complete_model(
        self,
        response_model: type[ResponseModel],
        *,
        system_prompt: str,
        payload: Mapping[str, Any],
        max_tokens: int,
    ) -> ResponseModel:
        """Validate strict JSON with two bounded, error-informed retries."""

        failures: list[str] = []
        retry_prompt = system_prompt
        retry_payload = dict(payload)
        for attempt in range(1, JUDGE_SCHEMA_MAX_ATTEMPTS + 1):
            try:
                raw = self.complete_json(
                    system_prompt=retry_prompt,
                    payload=retry_payload,
                    max_tokens=max_tokens,
                )
                return response_model.model_validate_json(raw)
            except JudgeTruncationError as exc:
                # A truncated response means the output limit is genuinely
                # too small for this judgment. Retrying with the same limit
                # and a longer prompt cannot succeed; per protocol this is an
                # explicit evaluator failure, not something to re-spend on.
                failures.append(str(exc))
                break
            except (ValidationError, ValueError) as exc:
                failures.append(str(exc))
                if attempt >= JUDGE_SCHEMA_MAX_ATTEMPTS:
                    break
                retry_prompt = (
                    system_prompt
                    + "\nYour previous response failed the required local "
                    "JSON schema. Return a corrected complete JSON object only; "
                    "do not repeat fields that the schema requires to be empty."
                )
                retry_payload = {
                    **dict(payload),
                    "response_validation_retry": {
                        "attempt": attempt + 1,
                        "previous_error": str(exc),
                    },
                }
        raise ValueError(
            "TARSphere judge failed strict response validation after "
            f"{attempt} attempts: {failures}"
        )

    def usage_report(self) -> dict[str, Any]:
        """Aggregate every Lingzhi request made by this evaluation client."""

        return aggregate_model_calls(self.model_calls)


def _unwrap_json_code_fence(content: str) -> str:
    """Extract one complete Markdown JSON fence without repairing its body."""

    stripped = content.strip()
    match = re.fullmatch(
        r"```(?:json)?[ \t]*\r?\n(?P<body>(?:(?!```)[\s\S])*)\r?\n```",
        stripped,
        flags=re.IGNORECASE,
    )
    if match is not None:
        return match.group("body").strip()
    fenced = tuple(
        re.finditer(
            r"```(?:json)?[ \t]*\r?\n(?P<body>(?:(?!```)[\s\S])*?)\r?\n```",
            stripped,
            flags=re.IGNORECASE,
        )
    )
    if len(fenced) == 1:
        return fenced[0].group("body").strip()
    return stripped


__all__ = [
    "JSON_RESPONSE_INSTRUCTION",
    "JUDGE_SCHEMA_MAX_ATTEMPTS",
    "JUDGE_TRANSIENT_MAX_ATTEMPTS",
    "JudgeClient",
    "JudgeConfig",
    "JudgeTruncationError",
]
