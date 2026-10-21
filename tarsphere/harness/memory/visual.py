"""Bounded action-step visual memory."""

from __future__ import annotations

from tarsphere.harness.memory.contracts import VisualFrameRecord


class VisualMemory:
    """Keep exactly the configured number of current/recent visual frames."""

    def __init__(self, *, horizon: int) -> None:
        if horizon < 1:
            raise ValueError("Visual-History Horizon must be at least 1")
        self.horizon = int(horizon)
        self._frames: list[VisualFrameRecord] = []

    def record(self, *, action_step: int, views: dict[str, object]) -> None:
        if self._frames and self._frames[-1].action_step == int(action_step):
            raise RuntimeError(f"visual frame already recorded for {action_step}")
        self._frames.append(
            VisualFrameRecord(
                action_step=int(action_step),
                views=dict(views),
            )
        )
        self._frames = self._frames[-self.horizon :]

    def retrieved_frames(self) -> list[VisualFrameRecord]:
        return [
            VisualFrameRecord(action_step=frame.action_step, views=dict(frame.views))
            for frame in self._frames
        ]

    def reset(self) -> None:
        self._frames.clear()


__all__ = ["VisualMemory"]
