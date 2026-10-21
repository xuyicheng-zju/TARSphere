"""Evaluated embodied agent implementation.

Architecture (envelope protocol: tools + StrategyPlanner):
    VLM -> EnvelopeKernel -> tools -> ActionRouter -> Robot runtime

Only the evaluated embodied agent belongs in this package. Scene grounding,
route planning for data, and certification utilities live under
``tarsphere.bench``.
"""

from tarsphere.harness.agent import RobotAgent
from tarsphere.harness.factory import build_robot_agent_envelope

__all__ = [
    "RobotAgent",
    "build_robot_agent_envelope",
]
