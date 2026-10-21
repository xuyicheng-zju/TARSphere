"""Local project LLM defaults.

Real local values live in ``tarsphere/configs/local_secrets.py``, which is
gitignored. This keeps runs convenient without committing secrets.
"""

from __future__ import annotations

import importlib
import math
import os
from dataclasses import dataclass

DISABLE_LOCAL_SECRETS_ENV = "TARSPHERE_DISABLE_LOCAL_SECRETS"
DEFAULT_LLM_MODEL = "gpt-5.6-sol"
DEFAULT_AGENT_MAX_TOKENS = 16384
DEFAULT_EVALUATION_SUPPORT_MODEL = "qwen3.8-flash"
DEFAULT_EVALUATION_SUPPORT_TIMEOUT_SECONDS = 180.0


@dataclass(frozen=True)
class LLMDefaults:
    api_key: str = ""
    api_base: str = ""
    model: str = ""


@dataclass(frozen=True)
class EvaluationSupportConfig:
    """One top-level configuration for Judge and Human/NPC inference."""

    api_key: str
    base_url: str
    model: str
    timeout_seconds: float = DEFAULT_EVALUATION_SUPPORT_TIMEOUT_SECONDS

    @classmethod
    def from_environment(cls) -> EvaluationSupportConfig:
        """Resolve the shared support service without Agent-model fallbacks."""

        timeout_text = _first(
            os.environ.get("TARSPHERE_EVALUATION_SUPPORT_TIMEOUT_SECONDS", ""),
            _local_value("TARSPHERE_EVALUATION_SUPPORT_TIMEOUT_SECONDS"),
            str(DEFAULT_EVALUATION_SUPPORT_TIMEOUT_SECONDS),
        )
        try:
            timeout_seconds = float(timeout_text)
        except ValueError as exc:
            raise ValueError(
                "TARSPHERE_EVALUATION_SUPPORT_TIMEOUT_SECONDS must be numeric"
            ) from exc
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError(
                "TARSPHERE_EVALUATION_SUPPORT_TIMEOUT_SECONDS must be finite and positive"
            )

        return cls(
            api_key=_first(
                os.environ.get("TARSPHERE_EVALUATION_SUPPORT_API_KEY", ""),
                os.environ.get("TARSPHERE_PROPRIETARY_API_KEY", ""),
                _local_value("TARSPHERE_EVALUATION_SUPPORT_API_KEY"),
                _local_value("TARSPHERE_PROPRIETARY_API_KEY"),
            ),
            base_url=_first(
                os.environ.get("TARSPHERE_EVALUATION_SUPPORT_API_BASE", ""),
                os.environ.get("TARSPHERE_PROPRIETARY_API_BASE", ""),
                _local_value("TARSPHERE_EVALUATION_SUPPORT_API_BASE"),
                _local_value("TARSPHERE_PROPRIETARY_API_BASE"),
            ),
            model=_first(
                os.environ.get("TARSPHERE_EVALUATION_SUPPORT_MODEL", ""),
                _local_value("TARSPHERE_EVALUATION_SUPPORT_MODEL"),
                DEFAULT_EVALUATION_SUPPORT_MODEL,
            ),
            timeout_seconds=timeout_seconds,
        )


class EvaluationSupportServiceError(RuntimeError):
    """The private Judge/Human support service could not produce a result."""


def _first(*values: str | None) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _local_value(name: str) -> str:
    if os.environ.get(DISABLE_LOCAL_SECRETS_ENV, "").strip():
        return ""
    try:
        module = importlib.import_module("tarsphere.configs.local_secrets")
    except ModuleNotFoundError as exc:
        if exc.name == "tarsphere.configs.local_secrets":
            return ""
        raise
    return str(getattr(module, name, "") or "").strip()


def agent_llm_defaults(
    *,
    api_key: str | None = None,
    model: str | None = None,
    api_base: str | None = None,
    api_key_env: str = "TARSPHERE_AGENT_API_KEY",
    default_model: str = DEFAULT_LLM_MODEL,
) -> LLMDefaults:
    return LLMDefaults(
        api_key=_first(
            api_key,
            os.environ.get(api_key_env, ""),
            os.environ.get("TARSPHERE_AGENT_API_KEY", ""),
            os.environ.get("OPENAI_API_KEY", ""),
            _local_value("TARSPHERE_AGENT_API_KEY"),
        ),
        model=_first(
            model,
            os.environ.get("TARSPHERE_AGENT_MODEL", ""),
            os.environ.get("OPENAI_MODEL", ""),
            _local_value("TARSPHERE_AGENT_MODEL"),
            default_model,
        ),
        api_base=_first(
            api_base,
            os.environ.get("TARSPHERE_AGENT_API_BASE", ""),
            os.environ.get("OPENAI_BASE_URL", ""),
            os.environ.get("OPENAI_API_BASE", ""),
            _local_value("TARSPHERE_AGENT_API_BASE"),
        ),
    )
