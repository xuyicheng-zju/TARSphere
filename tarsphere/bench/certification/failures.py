"""Structured failures that distinguish invalid geometry from system faults."""

from __future__ import annotations

from typing import Any, Literal


class PhysicalTrialCandidateError(RuntimeError):
    """Reject one bounded physical candidate without hiding system failures."""

    def __init__(
        self,
        message: str,
        *,
        stage: Literal["A", "B", "C"],
        reason: str,
        evidence: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.details = {
            "physical_trial_stage": stage,
            "candidate_rejection_reason": reason,
            "evidence": dict(evidence),
        }


__all__ = ["PhysicalTrialCandidateError"]
