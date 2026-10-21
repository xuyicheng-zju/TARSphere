"""Robot perception module contracts and backends."""

from tarsphere.harness.perception.contracts import (
    Detection2D,
    PerceptionInput,
    PerceptionOutput,
)
from tarsphere.harness.perception.base import PerceptionBackend
from tarsphere.harness.perception.module import RobotPerceptionModule

__all__ = [
    "Detection2D",
    "PerceptionBackend",
    "PerceptionInput",
    "PerceptionOutput",
    "RobotPerceptionModule",
]
