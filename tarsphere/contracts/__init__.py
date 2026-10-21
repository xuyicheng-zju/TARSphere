"""Shared contracts between the Agent harness and episode runtime.

Keep this package free of agent/eval implementation imports so either side
can depend on it without creating a cycle.
"""

from tarsphere.contracts.decision_requirements import DecisionRequirements
from tarsphere.contracts.task_profile import (
    CapabilitySlice,
    TaskMainline,
    TaskProfile,
)

__all__ = [
    "CapabilitySlice",
    "DecisionRequirements",
    "TaskMainline",
    "TaskProfile",
]
