"""Fail-loud control primitives for long simulator rollouts."""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Optional

from tarsphere.infra.control.physical_contracts import (
    NAVIGATION_MIN_HEADING_PROGRESS_RAD,
    NAVIGATION_MIN_XY_PROGRESS_M,
    NAVIGATION_STUCK_INCIDENT_DECISIONS,
    NAVIGATION_STUCK_TERMINATION_DECISIONS,
)


class EpisodeAbortError(RuntimeError):
    """Raised when the active scene-server request is explicitly cancelled."""


class EpisodeStallError(RuntimeError):
    """Raised when a controller phase makes no measurable physical progress."""

    def __init__(self, message: str, *, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = dict(details)


class ProgressWatchdog:
    """Detect repeated navigation decisions without task-relevant progress.

    Path following is judged against the active controller waypoint: advancing
    its monotone index or reducing its distance is progress. This rejects a
    robot orbiting one waypoint even though it accumulates plenty of raw XY
    motion. Euclidean distance to the final semantic goal remains diagnostic
    because a collision-free A* route may legitimately move away from it.
    Local translation falls back to net XY displacement, while rotation is
    judged by reduction of absolute heading error.
    """

    _TRANSLATION_ACTIONS = {
        "navigate_to_pointgoal",
        "move_body",
        "follow_path",
        "precise_approach",
        "move_forward",
        "move_backward",
        "move_left",
        "move_right",
    }
    _ROTATION_ACTIONS = {
        "precise_rotate",
        "turn_left",
        "turn_right",
    }
    _AGENT_LOCOMOTION_ACTIONS = {
        "navigate_to_pointgoal",
        "move_body",
        "move_forward",
        "move_backward",
        "move_left",
        "move_right",
        "turn_left",
        "turn_right",
    }

    def __init__(
        self,
        *,
        incident_decisions: int = NAVIGATION_STUCK_INCIDENT_DECISIONS,
        termination_decisions: int = NAVIGATION_STUCK_TERMINATION_DECISIONS,
        min_xy_progress_m: float = NAVIGATION_MIN_XY_PROGRESS_M,
        min_heading_progress_rad: float = NAVIGATION_MIN_HEADING_PROGRESS_RAD,
    ) -> None:
        if int(incident_decisions) < 2:
            raise ValueError("incident_decisions must be >= 2.")
        if int(termination_decisions) <= int(incident_decisions):
            raise ValueError("termination_decisions must exceed incident_decisions.")
        if float(min_xy_progress_m) <= 0.0:
            raise ValueError("Progress watchdog min_xy_progress_m must be positive.")
        if float(min_heading_progress_rad) <= 0.0:
            raise ValueError(
                "Progress watchdog min_heading_progress_rad must be positive."
            )
        self.incident_decisions = int(incident_decisions)
        self.termination_decisions = int(termination_decisions)
        self._incident_emitted = False
        self.min_xy_progress_m = float(min_xy_progress_m)
        self.min_heading_progress_rad = float(min_heading_progress_rad)
        self._key: tuple[str, str, str, str] | None = None
        self._records: Deque[dict[str, Any]] = deque(maxlen=self.termination_decisions)

    def reset(self) -> None:
        self._reset_progress_window()

    def _reset_progress_window(self) -> None:
        self._key = None
        self._records.clear()
        self._incident_emitted = False

    def observe(self, heartbeat: dict[str, Any]) -> dict[str, Any] | None:
        action = str(heartbeat.get("action") or "")
        if action not in self._TRANSLATION_ACTIONS | self._ROTATION_ACTIONS:
            # A non-locomotion decision (camera tilt, wait, grasp, ...) says
            # nothing about locomotion progress. Clearing the whole window
            # here let a stuck robot evade 5-record/10-termination forever by
            # interleaving one camera re-aim between motions; the no-progress
            # streak must survive across such decisions instead.
            return None
        agent_locomotion = action in self._AGENT_LOCOMOTION_ACTIONS
        phase = (
            "agent_locomotion"
            if agent_locomotion
            else str(heartbeat.get("phase") or action)
        )
        path_id = (
            str(heartbeat.get("controller_path_id") or "")
            if action == "follow_path"
            else ""
        )
        rotation_objective = (
            str(heartbeat.get("rotation_objective") or "")
            if action in self._ROTATION_ACTIONS
            else ""
        )
        key_action = "agent_locomotion" if agent_locomotion else action
        key = (phase, key_action, path_id, rotation_objective)
        if key != self._key:
            self._key = key
            self._records.clear()
            self._incident_emitted = False
        self._records.append(dict(heartbeat))
        if len(self._records) < self.incident_decisions:
            return None

        first = self._records[0]
        xy_progress = self._xy_progress(first, self._records)
        heading_progress = self._heading_progress(first, self._records)
        path_progress = self._path_progress(first, self._records)
        if agent_locomotion:
            rotation_progress = self._rotation_displacement(first, self._records)
            translation_requested, rotation_requested = (
                self._requested_agent_objectives(self._records)
            )
            translation_stalled = (
                not translation_requested or xy_progress < self.min_xy_progress_m
            )
            rotation_stalled = (
                not rotation_requested
                or rotation_progress < self.min_heading_progress_rad
            )
            stalled = translation_stalled and rotation_stalled
        elif action in self._ROTATION_ACTIONS:
            rotation_progress = (
                heading_progress
                if action == "precise_rotate"
                else self._rotation_displacement(first, self._records)
            )
            stalled = rotation_progress < self.min_heading_progress_rad
        elif action == "follow_path" and path_progress is not None:
            rotation_progress = 0.0
            stalled = path_progress < self.min_xy_progress_m
        else:
            rotation_progress = 0.0
            stalled = (
                xy_progress < self.min_xy_progress_m
                and heading_progress < self.min_heading_progress_rad
            )
        if not stalled:
            latest = dict(self._records[-1])
            self._records.clear()
            self._records.append(latest)
            self._incident_emitted = False
            return None

        last = self._records[-1]
        details = {
            "stuck_phase": phase,
            "action": action,
            "streak_decisions": len(self._records),
            "incident_decisions": self.incident_decisions,
            "termination_decisions": self.termination_decisions,
            "xy": last.get("xy"),
            "yaw_rad": last.get("yaw_rad"),
            "goal_distance_m": last.get("goal_distance_m"),
            "heading_error_rad": last.get("heading_error_rad"),
            "attached": bool(last.get("attached", False)),
            "xy_progress_m": round(float(xy_progress), 6),
            "heading_progress_rad": round(float(heading_progress), 6),
            "rotation_displacement_rad": round(float(rotation_progress), 6),
            "translation_requested": (
                bool(translation_requested) if agent_locomotion else None
            ),
            "rotation_requested": (
                bool(rotation_requested) if agent_locomotion else None
            ),
            "path_progress_m": (
                round(float(path_progress), 6) if path_progress is not None else None
            ),
            "controller_path_id": last.get("controller_path_id"),
            "controller_path_index": last.get("controller_path_index"),
            "controller_path_total": last.get("controller_path_total"),
            "controller_waypoint_distance_m": last.get(
                "controller_waypoint_distance_m"
            ),
        }
        if len(self._records) >= self.termination_decisions:
            details["terminal"] = True
            raise EpisodeStallError(
                "Controller phase made no measurable physical progress for "
                f"{self.termination_decisions} decisions: "
                f"phase={phase!r}, action={action!r}.",
                details=details,
            )
        if not self._incident_emitted:
            self._incident_emitted = True
            details["terminal"] = False
            return details
        return None

    @staticmethod
    def _requested_agent_objectives(
        records: Deque[dict[str, Any]],
    ) -> tuple[bool, bool]:
        """Return which physical objectives the recent public calls requested.

        A collision can rotate a body that requested pure translation. That
        incidental yaw is measured honestly, but it is not progress toward the
        requested objective and therefore cannot keep a stalled rollout alive.
        """

        translation = False
        rotation = False
        for record in records:
            action = str(record.get("action") or "")
            if action in {
                "move_forward",
                "move_backward",
                "move_left",
                "move_right",
            }:
                translation = True
                continue
            if action in {"turn_left", "turn_right"}:
                rotation = True
                continue
            if action != "move_body":
                continue
            outcome = record.get("motion_outcome")
            requested = (
                outcome.get("requested_body_delta")
                if isinstance(outcome, dict)
                else None
            )
            if not isinstance(requested, dict):
                continue
            forward = ProgressWatchdog._finite_float(requested.get("forward_m"))
            left = ProgressWatchdog._finite_float(requested.get("left_m"))
            turn = ProgressWatchdog._finite_float(requested.get("turn_deg"))
            if math.hypot(forward or 0.0, left or 0.0) > 1e-6:
                translation = True
            if abs(turn or 0.0) > 1e-6:
                rotation = True
        if not translation and not rotation:
            # Older internal heartbeats may not carry the body-vector request.
            # Preserve fail-loud monitoring without pretending either axis was
            # specifically requested.
            return True, True
        return translation, rotation

    @staticmethod
    def _finite_float(value: Any) -> Optional[float]:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    def _xy_progress(
        self,
        first: dict[str, Any],
        records: Deque[dict[str, Any]],
    ) -> float:
        first_xy = first.get("xy")
        if not isinstance(first_xy, (list, tuple)) or len(first_xy) < 2:
            return 0.0
        last_xy = records[-1].get("xy")
        if not isinstance(last_xy, (list, tuple)) or len(last_xy) < 2:
            return 0.0
        return math.dist(
            [float(first_xy[0]), float(first_xy[1])],
            [float(last_xy[0]), float(last_xy[1])],
        )

    def _path_progress(
        self,
        first: dict[str, Any],
        records: Deque[dict[str, Any]],
    ) -> Optional[float]:
        """Return monotone progress toward the active path waypoint.

        An index transition is authoritative controller progress. It returns
        at least the configured minimum so the watchdog never mistakes a
        completed waypoint for a stall. Within one index, progress is the
        reduction in measured distance to that exact waypoint.
        """
        first_index = self._finite_float(first.get("controller_path_index"))
        first_distance = self._finite_float(first.get("controller_waypoint_distance_m"))
        if first_index is None or first_distance is None:
            return None
        indices = [
            value
            for value in (
                self._finite_float(record.get("controller_path_index"))
                for record in records
            )
            if value is not None
        ]
        if indices and max(indices) > first_index:
            return self.min_xy_progress_m
        last_distance = self._finite_float(
            records[-1].get("controller_waypoint_distance_m")
        )
        if last_distance is None:
            return None
        return max(0.0, first_distance - last_distance)

    def _heading_progress(
        self,
        first: dict[str, Any],
        records: Deque[dict[str, Any]],
    ) -> float:
        first_error = self._finite_float(first.get("heading_error_rad"))
        errors = [
            abs(value)
            for value in (
                self._finite_float(record.get("heading_error_rad"))
                for record in records
            )
            if value is not None
        ]
        if first_error is None or not errors:
            return 0.0
        return max(0.0, abs(first_error) - min(errors))

    def _rotation_displacement(
        self,
        first: dict[str, Any],
        records: Deque[dict[str, Any]],
    ) -> float:
        """Measure achieved yaw for repeated public turn primitives."""

        first_yaw = self._finite_float(first.get("yaw_rad"))
        last_yaw = self._finite_float(records[-1].get("yaw_rad"))
        if first_yaw is None or last_yaw is None:
            return 0.0
        delta = math.atan2(
            math.sin(last_yaw - first_yaw),
            math.cos(last_yaw - first_yaw),
        )
        return abs(delta)


def write_heartbeat(path: Path, heartbeat: dict[str, Any]) -> None:
    """Append one durable decision record without buffering rollout state."""
    import json
    import os

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(heartbeat, sort_keys=True, default=str))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


AbortChecker = Callable[[], bool]


__all__ = [
    "AbortChecker",
    "EpisodeAbortError",
    "EpisodeStallError",
    "ProgressWatchdog",
    "write_heartbeat",
]
