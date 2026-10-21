"""Thin OpenAI-compatible VLM client wrapper.

Owns transport and multimodal message formatting only. DecisionEnvelope
protocol lives in EnvelopeKernel + prompt_builder.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import math
import random
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from openai import APIConnectionError, APITimeoutError

from tarsphere.api_usage import normalize_openai_usage, sum_token_usage
from tarsphere.harness.kernel.model_input_snapshot import save_model_input_snapshot
from tarsphere.configs.secrets import DEFAULT_AGENT_MAX_TOKENS, DEFAULT_LLM_MODEL
from tarsphere.provider_compat import (
    PROMPT_ONLY_JSON_MODELS,
    add_json_request_controls,
)

# This deployment emits its chain of thought into ``message.content``, breaking
# the single-strict-JSON DecisionEnvelope gate and inflating latency; the local
# decision parser remains the same strict gate.
PROVIDER_THINKING_IN_CONTENT_MODELS = frozenset(
    {
        "Qwen/Qwen3.5-397B-A17B-FP8",
    }
)
# GLM 4.5+ enables native thinking by default.  The evaluated policy already
# owns an explicit reasoning field inside its DecisionEnvelope, so provider
# thinking only adds hidden latency and output tokens without adding a second
# benchmark-visible capability.
GLM_NATIVE_THINKING_MODELS = frozenset(
    {
        "glm-5.3-flash",
    }
)
# Formal local endpoints share a context-window completion cap. Provider JSON
# support is listed separately in ``PROMPT_ONLY_JSON_MODELS``.
LOCAL_EMBODIED_VLLM_MODELS = frozenset(
    {
        "Qwen/Qwen3.8-27B",
        "tencent/Hy-Embodied-VLM-1.0",
        "BAAI/RoboBrain2.0-32B",
        "google/gemma-4-31B-it",
    }
)
# The formal Qwen, Hy-Embodied, and Gemma endpoints turn thinking off through
# their chat-template ``enable_thinking`` kwarg. The benchmark-visible
# DecisionEnvelope already contains its own concise reasoning field, so a
# second provider reasoning stream only consumes latency and completion tokens.
LOCAL_EMBODIED_THINKING_KWARG_MODELS = frozenset(
    {
        "Qwen/Qwen3.8-27B",
        "tencent/Hy-Embodied-VLM-1.0",
        "google/gemma-4-31B-it",
    }
)
# RoboBrain2.0 publishes this decoding value in its checkpoint
# ``generation_config.json``. Without it, the 32B model can repeat its
# ``thinking`` string until the local 2,048-token ceiling and leave an
# unterminated DecisionEnvelope.
LOCAL_EMBODIED_REPETITION_PENALTIES = {
    "BAAI/RoboBrain2.0-32B": 1.05,
}
# The official Hy-Embodied chat template renders only ``user`` messages. The
# harness owns one system instruction plus one current user observation, so
# transport must preserve both by placing the system text first in that user
# turn. Sending the system role unchanged silently discards the entire decision
# contract before tokenization.
SYSTEM_PROMPT_IN_USER_MODELS = frozenset(
    {
        "tencent/Hy-Embodied-VLM-1.0",
    }
)
# Queued reasoning models through the gateway can take ~40 s to the first
# token and then generate at single-digit tokens/step; a full JSON envelope
# therefore needs up to ~140 s end-to-end. 120 s read as a gateway outage.
VLM_REQUEST_TIMEOUT_S = 300.0
VLM_TRANSIENT_TRANSPORT_RETRIES = 5
# The Lingzhi Claude route has returned a generic Bedrock 400 for a request
# whose byte-identical saved payload succeeded on replay. Give that precise
# route error one retry; a repeated rejection remains an infrastructure error.
VLM_BEDROCK_VALIDATION_RETRIES = 1
VLM_TRANSIENT_BACKOFF_INITIAL_S = 1.0
VLM_TRANSIENT_BACKOFF_CAP_S = 20.0
# The completion budget covers the structured decision and any provider
# reasoning tokens. The strict envelope parser validates the completed output.
VLM_DEFAULT_MAX_TOKENS = DEFAULT_AGENT_MAX_TOKENS
# Wall-clock budget for one ``chat`` call's transient-failure recovery, spanning
# the requests themselves and not just the sleeps between them. A budget that
# counted only sleeps would be bounded by attempt count times the request
# timeout instead, which is tens of minutes and reads as a hang.
#
# The budget is expressed in whole request timeouts because a timeout is itself
# a retryable transport failure. It must cover the initial request plus every
# configured retry; otherwise the nominal retry count is silently unreachable.
VLM_TRANSIENT_RETRY_BUDGET_TIMEOUTS = float(VLM_TRANSIENT_TRANSPORT_RETRIES + 1)


def vlm_transient_retry_budget_s(request_timeout_s: float) -> float:
    """Recovery budget that can outlast full-length request timeouts."""

    return float(request_timeout_s) * VLM_TRANSIENT_RETRY_BUDGET_TIMEOUTS


LOGGER = logging.getLogger(__name__)


def _with_provider_json_hint(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Put the json-object contract in the final user message.

    Some OpenAI-compatible routes validate ``response_format=json_object``
    against user content only, even when the system prompt already states the
    same contract.  Copy just the final user message so transport formatting
    never mutates the kernel's within-turn conversation.
    """

    provider_messages = list(messages)
    for index in range(len(provider_messages) - 1, -1, -1):
        message = provider_messages[index]
        if message.get("role") != "user":
            continue
        updated = dict(message)
        content = updated.get("content")
        hint = "Return one valid json object."
        if isinstance(content, list):
            updated["content"] = [*content, {"type": "text", "text": hint}]
        elif isinstance(content, str):
            updated["content"] = content + "\n\n" + hint
        else:
            updated["content"] = hint
        provider_messages[index] = updated
        return provider_messages
    raise ValueError("response_format=json_object requires a user message")


def _fold_system_prompt_into_user_message(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve the harness system prompt for a user-only chat template."""

    if len(messages) != 2 or [message.get("role") for message in messages] != [
        "system",
        "user",
    ]:
        raise ValueError(
            "Hy-Embodied requires exactly one system message followed by one "
            "user message; its server chat template renders user messages only."
        )
    system_content = messages[0].get("content")
    if not isinstance(system_content, str) or not system_content.strip():
        raise ValueError("Hy-Embodied requires a non-empty system prompt.")

    user_message = dict(messages[1])
    user_content = user_message.get("content")
    if isinstance(user_content, list):
        user_message["content"] = [
            {"type": "text", "text": system_content},
            *user_content,
        ]
    elif isinstance(user_content, str):
        user_message["content"] = system_content + "\n\n" + user_content
    else:
        raise ValueError(
            "Hy-Embodied user content must be text or multimodal content blocks."
        )
    return [user_message]


def _validate_image_urls(messages: list[dict[str, Any]]) -> int:
    """Validate every multimodal image URL before provider submission."""

    image_count = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "image_url":
                continue
            image_count += 1
            image = block.get("image_url")
            url = image.get("url") if isinstance(image, dict) else None
            if not isinstance(url, str) or not url:
                raise ValueError("image_url block must contain a non-empty URL.")
            if url.startswith("data:"):
                header, separator, encoded = url.partition(",")
                if separator != "," or header not in {
                    "data:image/jpeg;base64",
                    "data:image/png;base64",
                }:
                    raise ValueError(
                        "Embedded image must be a base64 JPEG or PNG data URL."
                    )
                try:
                    decoded = base64.b64decode(encoded, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise ValueError(
                        "Embedded image contains invalid base64 data."
                    ) from exc
                if not decoded:
                    raise ValueError("Embedded image payload is empty.")
                continue
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("Remote image must use an absolute HTTP(S) URL.")
    return image_count


def _prompt_profile(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize model-visible request size without retaining prompt content.

    Provider token usage does not separate text from vision. This structural
    profile makes that split diagnosable from the private trace while keeping
    images, dialogue, and model-authored text out of the usage record itself.
    Embedded-image bytes are decoded-size estimates from the validated base64
    payload; remote image sizes are intentionally unknown.
    """

    text_characters_by_role: Counter[str] = Counter()
    text_blocks_by_role: Counter[str] = Counter()
    text_block_characters_by_role: dict[str, list[int]] = {}
    text_block_labels_by_role: dict[str, list[str]] = {}
    text_section_characters_by_role: dict[str, list[dict[str, Any]]] = {}
    text_sha256 = hashlib.sha256()
    embedded_image_bytes = 0
    embedded_image_count = 0
    remote_image_count = 0
    for message in messages:
        role = str(message.get("role") or "unknown")
        content = message.get("content")
        blocks = content if isinstance(content, list) else [content]
        for block in blocks:
            if isinstance(block, str):
                value = block
                text_characters_by_role[role] += len(block)
                text_blocks_by_role[role] += 1
                text_block_characters_by_role.setdefault(role, []).append(len(block))
                text_block_labels_by_role.setdefault(role, []).append(
                    _prompt_text_label(value)
                )
                text_section_characters_by_role.setdefault(role, []).extend(
                    _prompt_text_sections(value)
                )
                text_sha256.update(role.encode("utf-8"))
                text_sha256.update(b"\0")
                text_sha256.update(value.encode("utf-8"))
                continue
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                value = str(block.get("text") or "")
                text_characters_by_role[role] += len(value)
                text_blocks_by_role[role] += 1
                text_block_characters_by_role.setdefault(role, []).append(len(value))
                text_block_labels_by_role.setdefault(role, []).append(
                    _prompt_text_label(value)
                )
                text_section_characters_by_role.setdefault(role, []).extend(
                    _prompt_text_sections(value)
                )
                text_sha256.update(role.encode("utf-8"))
                text_sha256.update(b"\0")
                text_sha256.update(value.encode("utf-8"))
                continue
            if block.get("type") != "image_url":
                continue
            image = block.get("image_url")
            url = image.get("url") if isinstance(image, dict) else None
            if not isinstance(url, str):
                continue
            if url.startswith("data:"):
                encoded = url.partition(",")[2]
                # Base64 carries four characters for each three source bytes,
                # minus the one or two padding bytes at the end.
                embedded_image_bytes += max(
                    0,
                    (len(encoded) * 3 // 4) - encoded[-2:].count("="),
                )
                embedded_image_count += 1
            else:
                remote_image_count += 1
    return {
        "message_count": len(messages),
        "text_characters": sum(text_characters_by_role.values()),
        "text_characters_by_role": dict(text_characters_by_role),
        "text_blocks_by_role": dict(text_blocks_by_role),
        "text_block_characters_by_role": text_block_characters_by_role,
        "text_block_labels_by_role": text_block_labels_by_role,
        "text_section_characters_by_role": text_section_characters_by_role,
        "text_sha256": text_sha256.hexdigest(),
        "image_count": embedded_image_count + remote_image_count,
        "embedded_image_count": embedded_image_count,
        "embedded_image_bytes": embedded_image_bytes,
        "remote_image_count": remote_image_count,
    }


def _prompt_text_label(value: str) -> str:
    """Return a content-free stable label for one model-visible text block."""

    first = next(
        (line.strip() for line in value.splitlines() if line.strip()),
        "(empty)",
    )
    return first[:96]


def _prompt_text_sections(value: str) -> list[dict[str, Any]]:
    """Return content-free heading/size records for prompt-budget audits."""

    lines = value.splitlines(keepends=True)
    sections: list[dict[str, Any]] = []
    label = "(preamble)"
    characters = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("## ", "### ")):
            if characters:
                sections.append({"label": label[:96], "characters": characters})
            label = stripped
            characters = 0
        characters += len(line)
    if characters or not sections:
        sections.append({"label": label[:96], "characters": characters})
    return sections


def _is_transient_vlm_transport_failure(error: Exception) -> bool:
    """Return True only for retryable gateway/network transport failures."""

    if isinstance(error, (APIConnectionError, APITimeoutError)):
        return True
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True
    return isinstance(status_code, int) and status_code >= 500


def _is_unavailable_model_channel(error: Exception) -> bool:
    """A provider-specific 503 that another credential/group can resolve.

    An ordinary 503 stays on the bounded transport retry path. Lingzhi's
    ``model_not_found``/``No available channel`` response instead identifies
    the current credential's group, so retrying that same group cannot help.
    """

    if getattr(error, "status_code", None) != 503:
        return False
    body = getattr(error, "body", None)
    if not isinstance(body, Mapping):
        return False
    detail = body.get("error", body)
    return (
        isinstance(detail, Mapping)
        and detail.get("code") == "model_not_found"
        and "No available channel for model" in str(detail.get("message", ""))
    )


def _is_inconsistent_bedrock_validation(
    error: Exception,
    *,
    model: str,
    api_base: str,
) -> bool:
    """Recognize the replay-proven, generic Claude route rejection only."""

    if (
        model != "claude-sonnet-5"
        or urlparse(api_base).hostname != "lingzhi.agibot.com"
        or getattr(error, "status_code", None) != 400
    ):
        return False
    body = getattr(error, "body", None)
    detail = body.get("error", body) if isinstance(body, Mapping) else None
    message = (
        str(detail.get("message", "")) if isinstance(detail, Mapping) else str(error)
    )
    return (
        "Bedrock Runtime" in message
        and "ValidationException: The input fails to satisfy the constraints "
        "specified by Amazon Bedrock."
        in message
    )


def _retry_after_seconds(error: Exception) -> float | None:
    """Return a throttler's requested wait, when it states one."""

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        retry_after = float(headers.get("Retry-After"))
    except (AttributeError, TypeError, ValueError):
        return None
    if not math.isfinite(retry_after) or retry_after < 0.0:
        return None
    return retry_after


def _transient_transport_backoff_sleep(
    *,
    error: Exception,
    transport_retries: int,
    deadline: float,
) -> float:
    """Exponential backoff with jitter, clamped to the remaining budget.

    A throttler that names its own wait is obeyed instead: retrying a 429
    sooner than asked only deepens the rate limit.
    """

    base = min(
        VLM_TRANSIENT_BACKOFF_INITIAL_S * (2**transport_retries),
        VLM_TRANSIENT_BACKOFF_CAP_S,
    )
    requested = _retry_after_seconds(error)
    sleep_s = (
        requested if requested is not None else base + random.uniform(0.0, base * 0.1)
    )
    remaining = deadline - time.monotonic()
    return min(sleep_s, max(0.0, remaining))


class VLMClient:
    """Thin OpenAI-compatible VLM client wrapper."""

    def __init__(
        self,
        *,
        api_key: str = "",
        fallback_api_keys: tuple[str, ...] = (),
        api_base: str = "",
        model: str = DEFAULT_LLM_MODEL,
        temperature: float = 0.0,
        max_tokens: int = VLM_DEFAULT_MAX_TOKENS,
        request_timeout_s: float = VLM_REQUEST_TIMEOUT_S,
    ) -> None:
        self.api_key = api_key
        self._api_keys = tuple(
            dict.fromkeys(key for key in (api_key, *fallback_api_keys) if key)
        )
        self._api_key_index = 0
        self.api_base = api_base
        self.model = model
        self.temperature = temperature
        self.max_tokens = int(max_tokens)
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive.")
        self.request_timeout_s = float(request_timeout_s)
        if not math.isfinite(self.request_timeout_s) or (self.request_timeout_s <= 0.0):
            raise ValueError("request_timeout_s must be finite and positive.")
        self._client = None
        self._input_directory: Path | None = None
        self._input_index = 0
        self._request_context: dict[str, Any] = {}

    def configure_request_capture(self, directory: Path | None) -> None:
        """Start one physical attempt; never overwrite an earlier request log."""

        if directory is not None and directory.exists() and any(directory.iterdir()):
            raise FileExistsError(
                f"Model-input capture directory is not empty: {directory}"
            )
        self._input_directory = directory
        self._input_index = 0
        self._request_context = {}

    def set_request_context(self, *, policy_step: int, action_step: int) -> None:
        self._request_context = {"policy_step": policy_step, "action_step": action_step}

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError(
                "No LLM API key. Set TARSPHERE_AGENT_API_KEY or OPENAI_API_KEY."
            )
        # Respect explicit proxy configuration. The server launcher adds local
        # model ports to NO_PROXY; never silently delete process-wide settings.
        for logger_name in ("openai", "httpx", "httpcore"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        from openai import OpenAI

        self._client = OpenAI(
            api_key=self._api_keys[self._api_key_index],
            base_url=self.api_base or None,
            timeout=self.request_timeout_s,
            max_retries=0,
        )
        return self._client

    def _switch_to_fallback_key(self) -> bool:
        """Move this worker to its next credential without exposing key data."""

        if self._api_key_index + 1 >= len(self._api_keys):
            return False
        self._api_key_index += 1
        self._client = None
        self._ensure_client()
        return True

    def chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Call Lingzhi and return content plus auditable request/token usage.

        The agent's decision protocol is the DecisionEnvelope carried in the
        assistant text (exactly one JSON object), parsed by
        ``kernel.decision_parser``. Native OpenAI tool-calling is deliberately
        not used, so no ``tools`` are advertised on the request.
        """
        client = self._ensure_client()
        call_started = time.monotonic()
        _validate_image_urls(messages)
        input_snapshot = None
        response_records: list[dict[str, Any]] = []
        request_attempts = 0

        def model_call_record() -> dict[str, Any]:
            return {
                "gateway": urlparse(self.api_base).netloc or "lingzhi",
                "model": self.model,
                "request_attempts": request_attempts,
                "provider_responses": len(response_records),
                # Token totals below are verified subtotals. Missing or invalid
                # provider usage is counted explicitly, never priced as zero.
                "unavailable_usage_responses": sum(
                    "usage" not in record for record in response_records
                ),
                "latency_s": round(time.monotonic() - call_started, 3),
                "prompt_profile": dict(prompt_profile),
                **(
                    {"input_snapshot": input_snapshot}
                    if input_snapshot is not None
                    else {}
                ),
                "usage": sum_token_usage(
                    [
                        record["usage"]
                        for record in response_records
                        if "usage" in record
                    ]
                ),
                "responses": list(response_records),
            }

        def attach_failure_context(
            error: Exception,
            *,
            failure_kind: str,
        ) -> None:
            try:
                error.model_call = model_call_record()
                error.vlm_failure_kind = failure_kind
            except (AttributeError, TypeError):
                pass

        request_max_tokens = self.max_tokens
        if self.model in LOCAL_EMBODIED_VLLM_MODELS:
            # Local vLLM endpoints enforce prompt+completion against one
            # context window; a 16k completion budget on top of a 16k-window
            # deployment is an instant 400. A DecisionEnvelope fits in ~2k
            # tokens; the strict parser is the real output guard.
            request_max_tokens = min(request_max_tokens, 2048)
        # Only recoverable transport failures and the replay-proven Bedrock
        # route inconsistency retry. A model response, including empty
        # content, is returned once and charged as one policy decision.
        retry_budget_s = vlm_transient_retry_budget_s(self.request_timeout_s)
        transport_deadline = time.monotonic() + retry_budget_s
        transport_retries = 0
        bedrock_validation_retries = 0
        provider_messages = messages
        if self.model in SYSTEM_PROMPT_IN_USER_MODELS:
            provider_messages = _fold_system_prompt_into_user_message(
                provider_messages,
            )
        if self.model not in PROMPT_ONLY_JSON_MODELS:
            provider_messages = _with_provider_json_hint(provider_messages)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": provider_messages,
            "max_tokens": request_max_tokens,
            "timeout": self.request_timeout_s,
        }
        add_json_request_controls(
            request,
            model=self.model,
            temperature=self.temperature,
        )
        extra_body: dict[str, Any] = {}
        if self.model in PROVIDER_THINKING_IN_CONTENT_MODELS or (
            self.model in LOCAL_EMBODIED_THINKING_KWARG_MODELS
        ):
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        repetition_penalty = LOCAL_EMBODIED_REPETITION_PENALTIES.get(
            self.model,
        )
        if repetition_penalty is not None:
            extra_body["repetition_penalty"] = repetition_penalty
        if self.model in GLM_NATIVE_THINKING_MODELS:
            extra_body["thinking"] = {"type": "disabled"}
        if extra_body:
            request["extra_body"] = extra_body
        # Profile and snapshot the actual outgoing payload after provider
        # adaptation (especially Hy's system-to-user fold), once per decision.
        prompt_profile = _prompt_profile(provider_messages)
        if self._input_directory is not None:
            input_snapshot = save_model_input_snapshot(
                self._input_directory,
                index=self._input_index,
                request=request,
                context=self._request_context,
            )
            self._input_index += 1
        while True:
            try:
                request_attempts += 1
                response = client.chat.completions.create(**request)
                break
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                if (
                    _is_inconsistent_bedrock_validation(
                        exc,
                        model=self.model,
                        api_base=self.api_base,
                    )
                    and bedrock_validation_retries < VLM_BEDROCK_VALIDATION_RETRIES
                ):
                    bedrock_validation_retries += 1
                    LOGGER.warning(
                        "Claude Bedrock route rejected a replay-valid request "
                        "with generic HTTP 400; retrying the identical "
                        "request once."
                    )
                    time.sleep(1.0)
                    continue
                if (
                    status_code in {401, 402, 403, 429}
                    or _is_unavailable_model_channel(exc)
                ) and (self._switch_to_fallback_key()):
                    client = self._ensure_client()
                    LOGGER.warning(
                        "VLM credential rejected, throttled, or its model "
                        "channel unavailable (HTTP %s); "
                        "switched to the configured fallback credential.",
                        status_code,
                    )
                    continue
                if _is_transient_vlm_transport_failure(exc):
                    failure_name = (
                        f"HTTP {exc.status_code}"
                        if getattr(exc, "status_code", None) is not None
                        else type(exc).__name__
                    )
                    sleep_s = _transient_transport_backoff_sleep(
                        error=exc,
                        transport_retries=transport_retries,
                        deadline=transport_deadline,
                    )
                    exhausted = (
                        transport_retries >= VLM_TRANSIENT_TRANSPORT_RETRIES
                        or sleep_s <= 0.0
                    )
                    if exhausted:
                        LOGGER.error(
                            "Giving up on the VLM after %d transient "
                            "transport retries and %.1f s; last failure %s.",
                            transport_retries,
                            retry_budget_s
                            - max(0.0, transport_deadline - time.monotonic()),
                            failure_name,
                        )
                        attach_failure_context(
                            exc,
                            failure_kind="transport_retry_exhausted",
                        )
                        raise
                    transport_retries += 1
                    LOGGER.warning(
                        "Transient VLM transport failure (%s); retrying "
                        "attempt %d/%d after %.2f s.",
                        failure_name,
                        transport_retries,
                        VLM_TRANSIENT_TRANSPORT_RETRIES,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                attach_failure_context(
                    exc,
                    failure_kind="provider_request_failed",
                )
                raise
        # The request and response happened even if its payload is unusable.
        # Register it first. Malformed usage metadata stays explicitly
        # unavailable, while a missing choice becomes empty model output for
        # the strict decision parser to reject on this policy step.
        response_record: dict[str, Any] = {}
        response_records.append(response_record)
        raw_usage = getattr(response, "usage", None)
        dump_usage = getattr(raw_usage, "model_dump", None)
        response_record["raw_usage"] = (
            dump_usage() if callable(dump_usage) else raw_usage
        )
        if raw_usage is not None:
            try:
                response_record["usage"] = normalize_openai_usage(
                    response_record["raw_usage"],
                )
            except (TypeError, ValueError) as exc:
                response_record["usage_error"] = f"{type(exc).__name__}: {exc}"
        try:
            choice = response.choices[0]
            content = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", None)
        except (AttributeError, IndexError, TypeError) as exc:
            content = ""
            finish_reason = "provider_response_invalid"
            response_record["response_error"] = f"{type(exc).__name__}: {exc}"
        response_record.update(
            {
                "finish_reason": finish_reason,
                "content_empty": not bool(content.strip()),
                # Preserve the exact provider text for forensic trajectory review,
                # especially when strict JSON parsing rejects the decision.
                "content": content,
            }
        )
        return {
            "content": content,
            "finish_reason": finish_reason,
            "model_call": model_call_record(),
        }


__all__ = [
    "GLM_NATIVE_THINKING_MODELS",
    "PROVIDER_THINKING_IN_CONTENT_MODELS",
    "VLM_DEFAULT_MAX_TOKENS",
    "VLM_REQUEST_TIMEOUT_S",
    "VLM_TRANSIENT_TRANSPORT_RETRIES",
    "VLMClient",
]
