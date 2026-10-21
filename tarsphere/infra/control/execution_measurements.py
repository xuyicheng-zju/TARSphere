"""Read-only whole-episode physics measurements, after reset warmup."""

from __future__ import annotations

import math
from typing import Sequence


class PhysicsPathAccumulator:
    """Integrate live pelvis XY at every native tick, including non-navigation.

    The reset endpoint is the first sample. No action-local distance is added;
    an out-and-back motion is therefore counted once and never becomes a chord.
    Runtime counter agreement proves all counted execution ticks were sampled.
    """

    def __init__(self, *, position: Sequence[float], starting_physics_step: int):
        self._previous_xy = self._xy(position)
        self._starting_physics_step = int(starting_physics_step)
        self._sampled_physics_steps = 0
        self._path_length_m = 0.0

    @staticmethod
    def _xy(position: Sequence[float]) -> tuple[float, float]:
        xy = (float(position[0]), float(position[1]))
        if not all(math.isfinite(value) for value in xy):
            raise ValueError("Execution path requires finite live pelvis XY.")
        return xy

    def record_tick(self, position: Sequence[float]) -> None:
        xy = self._xy(position)
        self._path_length_m += math.dist(self._previous_xy, xy)
        self._previous_xy = xy
        self._sampled_physics_steps += 1

    def snapshot(self, *, total_physics_steps: int) -> dict:
        execution_ticks = int(total_physics_steps) - self._starting_physics_step
        complete = execution_ticks == self._sampled_physics_steps
        return {
            "source": "native_physics_tick_pelvis_xy",
            "status": "available" if complete else "unavailable",
            "missing_evidence": [] if complete else ["physics_tick_coverage_mismatch"],
            # Warmup is outside the Agent/Oracle execution interval.
            "starting_physics_step": self._starting_physics_step,
            "sampled_physics_steps": self._sampled_physics_steps,
            "physics_steps": execution_ticks if complete else None,
            "path_length_m": self._path_length_m if complete else None,
        }
