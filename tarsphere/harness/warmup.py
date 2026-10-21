"""Explicit startup warmup for heavyweight robot perception backends.

This module is only for entrypoint-level validation/bootstrap. It does not
change the planner-visible tool architecture: it simply asks
the already-wired agent backends to load using the first real ego observation,
after Isaac owns the CUDA context.
"""

from __future__ import annotations

import os
from typing import Any


TRUE_VALUES = {"1", "true", "yes", "on"}


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUE_VALUES


def heavy_backend_warmup_requested() -> bool:
    return env_flag("TARSPHERE_AGENT_WARMUP_HEAVY_BACKENDS")


def _split_prompts(value: str | None) -> list[str]:
    prompts = [item.strip() for item in (value or "").split(",")]
    return [item for item in prompts if item]


def _to_agent_observation(agent: Any, observation: Any) -> Any:
    if hasattr(agent, "_to_agent_observation") and isinstance(observation, dict):
        return agent._to_agent_observation(observation)
    return observation


def _semantic_backend(agent: Any) -> Any:
    """The SAM3 backend held by the agent's ``locate_object`` tool, if wired.

    The tool is the single owner of the backend, so warming up through it
    exercises exactly the path the model will take at decision time.
    """
    registry = getattr(agent, "tools", None)
    if registry is None:
        return None
    return getattr(registry.handler("locate_object"), "backend", None)


def warmup_configured_heavy_backends(
    agent: Any,
    observation: Any,
    *,
    sam3_prompts: list[str] | None = None,
) -> dict[str, Any]:
    """Force-load configured SAM3 backend using one ego observation.

    Mapping and navigation backends require no heavyweight model warmup here.
    The remaining heavy perception backend is SAM3. It is reached through the
    ``locate_object`` tool that owns it — the same path the model uses. If
    the tool is absent, this raises so validation cannot silently pass on the
    wrong architecture.
    """
    agent_obs = _to_agent_observation(agent, observation)
    summary: dict[str, Any] = {}

    backend = _semantic_backend(agent)
    if backend is None:
        raise RuntimeError(
            "The configured agent has no locate_object "
            "tool backed by a semantic backend. Wire SAM3 via "
            "build_robot_agent_envelope."
        )
    prompts = (
        sam3_prompts
        or _split_prompts(os.environ.get("TARSPHERE_AGENT_SAM3_WARMUP_PROMPTS"))
        or ["person"]
    )
    from tarsphere.harness.perception.contracts import PerceptionInput

    output = backend.process(
        PerceptionInput(observation=agent_obs, text_prompts=prompts)
    )
    summary["sam3"] = {
        "status": "loaded",
        "prompts": prompts,
        "detections": len(getattr(output, "detections", []) or []),
    }

    return summary


__all__ = [
    "env_flag",
    "heavy_backend_warmup_requested",
    "warmup_configured_heavy_backends",
]
