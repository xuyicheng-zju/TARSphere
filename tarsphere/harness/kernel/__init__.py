"""Kernel package for the evaluated agent.

The supported kernel is ``EnvelopeKernel``: a bounded VLM turn loop using the
strict DecisionEnvelope JSON protocol. ``VLMClient`` is kept as a standalone
OpenAI-compatible chat wrapper.
"""

from __future__ import annotations

from typing import Any

__all__ = ["VLMClient"]


def __getattr__(name: str) -> Any:
    if name != "VLMClient":
        raise AttributeError(name)
    from tarsphere.harness.kernel.vlm_client import VLMClient

    globals()[name] = VLMClient
    return VLMClient
