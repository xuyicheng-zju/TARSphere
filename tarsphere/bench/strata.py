"""Canonical benchmark reporting cells: task mainline x capability slice."""

from __future__ import annotations

from dataclasses import dataclass

from tarsphere.contracts.task_profile import CapabilitySlice, TaskMainline


@dataclass(frozen=True, order=True)
class BenchmarkCell:
    """One of 12 reporting cells; difficulty is inherited Base metadata."""

    mainline: TaskMainline
    capability_slice: CapabilitySlice

    @property
    def key(self) -> str:
        return f"{self.mainline.value}/{self.capability_slice.value}"


BENCHMARK_CELLS: tuple[BenchmarkCell, ...] = tuple(
    BenchmarkCell(mainline=mainline, capability_slice=capability)
    for mainline in TaskMainline
    for capability in CapabilitySlice
)


__all__ = ["BENCHMARK_CELLS", "BenchmarkCell"]
