"""Expected action refusals that remain part of an evaluated rollout."""

from __future__ import annotations

from typing import Any, Mapping


class ActionRefusedError(RuntimeError):
    """A valid public action cannot execute in the current physical state."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "action_refused",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.details = dict(details or {})

    def as_feedback(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "details": dict(self.details),
        }


__all__ = ["ActionRefusedError"]
