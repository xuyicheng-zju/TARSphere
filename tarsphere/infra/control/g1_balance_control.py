"""Proprioceptive balance intervention for G1 locomotion.

The evaluated agent owns the requested body motion.  This module observes only
G1's own IMU and pelvis state while that motion is executing.  It never reads
scene geometry, contact reports, occupancy, a target, or a route.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class G1BodyDynamics:
    """One physics-tick sample of G1's balance-relevant proprioception."""

    gravity_tilt_deg: float
    roll_pitch_rate_rad_s: float
    pelvis_height_m: float

    def __post_init__(self) -> None:
        values = (
            self.gravity_tilt_deg,
            self.roll_pitch_rate_rad_s,
            self.pelvis_height_m,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"G1 body dynamics must be finite: {self!r}")
        if self.gravity_tilt_deg < 0.0:
            raise ValueError("gravity_tilt_deg must be non-negative.")
        if self.roll_pitch_rate_rad_s < 0.0:
            raise ValueError("roll_pitch_rate_rad_s must be non-negative.")


@dataclass(frozen=True)
class G1BalanceLimits:
    """Frozen signal and hysteresis limits for the G1 balance controller."""

    trend_window_ticks: int = 16
    trigger_confirmation_ticks: int = 3
    recovery_confirmation_ticks: int = 24
    recovery_timeout_ticks: int = 360
    elevated_tilt_deg: float = 8.0
    minimum_tilt_rise_deg: float = 1.5
    minimum_tick_tilt_rise_deg: float = 0.15
    elevated_roll_pitch_rate_rad_s: float = 1.0
    minimum_pelvis_drop_m: float = 0.04
    minimum_recent_pelvis_fall_m: float = 0.015
    critical_tilt_deg: float = 20.0
    critical_pelvis_drop_m: float = 0.10
    recovered_tilt_deg: float = 7.0
    recovered_roll_pitch_rate_rad_s: float = 0.65
    recovered_pelvis_drop_m: float = 0.03

    def __post_init__(self) -> None:
        integer_limits = (
            self.trend_window_ticks,
            self.trigger_confirmation_ticks,
            self.recovery_confirmation_ticks,
            self.recovery_timeout_ticks,
        )
        if any(int(value) <= 0 for value in integer_limits):
            raise ValueError("G1 balance timing limits must be positive.")
        if self.recovery_timeout_ticks < self.recovery_confirmation_ticks:
            raise ValueError(
                "G1 balance recovery timeout must admit one stable window."
            )
        if self.recovered_tilt_deg >= self.elevated_tilt_deg:
            raise ValueError(
                "Recovered tilt must be below the intervention tilt threshold."
            )
        if self.recovered_roll_pitch_rate_rad_s >= self.elevated_roll_pitch_rate_rad_s:
            raise ValueError(
                "Recovered roll/pitch rate must be below the intervention rate."
            )
        if self.recovered_pelvis_drop_m >= self.minimum_pelvis_drop_m:
            raise ValueError(
                "Recovered pelvis drop must be below the intervention drop."
            )


@dataclass(frozen=True)
class G1BalanceReport:
    """Measured outcome of balance control during one locomotion action."""

    intervened: bool
    trigger_action_tick: int | None
    trigger_signals: tuple[str, ...]
    trigger_gravity_tilt_deg: float | None
    trigger_roll_pitch_rate_rad_s: float | None
    trigger_pelvis_height_m: float | None
    trigger_tilt_rise_deg: float | None
    trigger_pelvis_drop_m: float | None
    recovery_ticks: int
    recovered: bool
    peak_gravity_tilt_deg: float
    peak_roll_pitch_rate_rad_s: float
    minimum_pelvis_height_m: float

    def as_dict(self) -> dict[str, object]:
        return {
            "intervened": self.intervened,
            "trigger_action_tick": self.trigger_action_tick,
            "trigger_signals": list(self.trigger_signals),
            "trigger_gravity_tilt_deg": self.trigger_gravity_tilt_deg,
            "trigger_roll_pitch_rate_rad_s": (self.trigger_roll_pitch_rate_rad_s),
            "trigger_pelvis_height_m": self.trigger_pelvis_height_m,
            "trigger_tilt_rise_deg": self.trigger_tilt_rise_deg,
            "trigger_pelvis_drop_m": self.trigger_pelvis_drop_m,
            "recovery_ticks": self.recovery_ticks,
            "recovered": self.recovered,
            "peak_gravity_tilt_deg": self.peak_gravity_tilt_deg,
            "peak_roll_pitch_rate_rad_s": self.peak_roll_pitch_rate_rad_s,
            "minimum_pelvis_height_m": self.minimum_pelvis_height_m,
        }

    def agent_feedback(self) -> str:
        if not self.intervened:
            return ""
        signals = ", ".join(self.trigger_signals)
        return (
            "Proprioceptive balance control interrupted the body motion at "
            f"action-local physics tick {self.trigger_action_tick} after "
            "sustained instability "
            f"({signals}). At intervention, gravity tilt was "
            f"{self.trigger_gravity_tilt_deg:.1f} deg, roll/pitch rate was "
            f"{self.trigger_roll_pitch_rate_rad_s:.2f} rad/s, and pelvis drop "
            f"was {self.trigger_pelvis_drop_m:.3f} m. Zero-speed recovery took "
            f"{self.recovery_ticks} ticks. "
            f"Peak gravity tilt was {self.peak_gravity_tilt_deg:.1f} deg, peak "
            f"roll/pitch rate was {self.peak_roll_pitch_rate_rad_s:.2f} rad/s, "
            f"and minimum pelvis height was {self.minimum_pelvis_height_m:.3f} m. "
            "The unfinished part of the old motion was discarded; decide again "
            "from the fresh observation and measured displacement."
        )


class G1BalanceRecoveryError(RuntimeError):
    """Raised when zero-speed control cannot restore G1 within its bound."""

    def __init__(self, report: G1BalanceReport):
        self.report = report
        self.details = report.as_dict()
        super().__init__(
            "G1 proprioceptive balance recovery timed out: "
            f"trigger_action_tick={report.trigger_action_tick}, "
            f"trigger_signals={list(report.trigger_signals)!r}, "
            f"trigger_gravity_tilt_deg={report.trigger_gravity_tilt_deg:.2f}, "
            "trigger_roll_pitch_rate_rad_s="
            f"{report.trigger_roll_pitch_rate_rad_s:.3f}, "
            f"trigger_pelvis_drop_m={report.trigger_pelvis_drop_m:.3f}, "
            f"recovery_ticks={report.recovery_ticks}, "
            f"peak_gravity_tilt_deg={report.peak_gravity_tilt_deg:.2f}, "
            "peak_roll_pitch_rate_rad_s="
            f"{report.peak_roll_pitch_rate_rad_s:.3f}, "
            f"minimum_pelvis_height_m={report.minimum_pelvis_height_m:.3f}."
        )


class G1BalanceController:
    """Interrupt a locomotion action only when body dynamics keep worsening."""

    def __init__(self, limits: G1BalanceLimits | None = None) -> None:
        self.limits = limits or G1BalanceLimits()
        self._history: deque[G1BodyDynamics] = deque(
            maxlen=self.limits.trend_window_ticks,
        )
        self._baseline_pelvis_height_m: float | None = None
        self._trigger_candidate_ticks = 0
        self._candidate_signals: set[str] = set()
        self._trigger_action_tick: int | None = None
        self._trigger_signals: tuple[str, ...] = ()
        self._trigger_sample: G1BodyDynamics | None = None
        self._trigger_tilt_rise_deg: float | None = None
        self._trigger_pelvis_drop_m: float | None = None
        self._recovery_ticks = 0
        self._recovered = False
        self._stable_recovery_ticks = 0
        self._peak_tilt_deg = 0.0
        self._peak_roll_pitch_rate_rad_s = 0.0
        self._minimum_pelvis_height_m = math.inf

    @property
    def intervening(self) -> bool:
        return self._trigger_action_tick is not None and not self._recovered

    @property
    def recovered(self) -> bool:
        return self._recovered

    def begin(self, sample: G1BodyDynamics) -> None:
        """Establish the action-local upright reference from live proprioception."""

        if self._baseline_pelvis_height_m is not None:
            raise RuntimeError("G1 balance controller already began an action.")
        self._baseline_pelvis_height_m = float(sample.pelvis_height_m)
        self._record(sample)

    def observe(self, sample: G1BodyDynamics, *, tick_index: int) -> None:
        """Consume one post-physics-tick sample and advance the state machine."""

        if self._baseline_pelvis_height_m is None:
            raise RuntimeError("G1 balance controller must begin before observe.")
        if self._recovered:
            raise RuntimeError("Recovered G1 balance controller cannot resume.")
        self._record(sample)
        if self._trigger_action_tick is None:
            self._observe_motion(sample, tick_index=int(tick_index))
            return
        self._observe_recovery(sample)

    def report(self) -> G1BalanceReport:
        if self._baseline_pelvis_height_m is None:
            raise RuntimeError("G1 balance controller has no action report.")
        return G1BalanceReport(
            intervened=self._trigger_action_tick is not None,
            trigger_action_tick=self._trigger_action_tick,
            trigger_signals=self._trigger_signals,
            trigger_gravity_tilt_deg=(
                self._trigger_sample.gravity_tilt_deg
                if self._trigger_sample is not None
                else None
            ),
            trigger_roll_pitch_rate_rad_s=(
                self._trigger_sample.roll_pitch_rate_rad_s
                if self._trigger_sample is not None
                else None
            ),
            trigger_pelvis_height_m=(
                self._trigger_sample.pelvis_height_m
                if self._trigger_sample is not None
                else None
            ),
            trigger_tilt_rise_deg=self._trigger_tilt_rise_deg,
            trigger_pelvis_drop_m=self._trigger_pelvis_drop_m,
            recovery_ticks=self._recovery_ticks,
            recovered=self._recovered,
            peak_gravity_tilt_deg=self._peak_tilt_deg,
            peak_roll_pitch_rate_rad_s=self._peak_roll_pitch_rate_rad_s,
            minimum_pelvis_height_m=self._minimum_pelvis_height_m,
        )

    def _record(self, sample: G1BodyDynamics) -> None:
        self._history.append(sample)
        self._peak_tilt_deg = max(
            self._peak_tilt_deg,
            float(sample.gravity_tilt_deg),
        )
        self._peak_roll_pitch_rate_rad_s = max(
            self._peak_roll_pitch_rate_rad_s,
            float(sample.roll_pitch_rate_rad_s),
        )
        self._minimum_pelvis_height_m = min(
            self._minimum_pelvis_height_m,
            float(sample.pelvis_height_m),
        )

    def _observe_motion(self, sample: G1BodyDynamics, *, tick_index: int) -> None:
        signals = self._deterioration_signals(sample)
        if self._is_deteriorating(sample, signals):
            self._trigger_candidate_ticks += 1
            self._candidate_signals.update(signals)
        else:
            self._trigger_candidate_ticks = 0
            self._candidate_signals.clear()
        if self._trigger_candidate_ticks < self.limits.trigger_confirmation_ticks:
            return
        self._trigger_action_tick = tick_index
        self._trigger_signals = tuple(sorted(self._candidate_signals))
        self._trigger_sample = sample
        oldest = self._history[0]
        self._trigger_tilt_rise_deg = float(
            sample.gravity_tilt_deg - oldest.gravity_tilt_deg
        )
        self._trigger_pelvis_drop_m = float(
            float(self._baseline_pelvis_height_m) - sample.pelvis_height_m
        )
        self._stable_recovery_ticks = 0

    def _deterioration_signals(
        self,
        sample: G1BodyDynamics,
    ) -> tuple[str, ...]:
        oldest = self._history[0]
        previous = self._history[-2] if len(self._history) >= 2 else oldest
        baseline = float(self._baseline_pelvis_height_m)
        tilt_rise = float(sample.gravity_tilt_deg - oldest.gravity_tilt_deg)
        tick_tilt_rise = float(sample.gravity_tilt_deg - previous.gravity_tilt_deg)
        pelvis_drop = float(baseline - sample.pelvis_height_m)
        recent_pelvis_fall = float(oldest.pelvis_height_m - sample.pelvis_height_m)
        tilt_is_still_rising = tick_tilt_rise >= self.limits.minimum_tick_tilt_rise_deg
        signals: list[str] = []
        if (
            sample.gravity_tilt_deg >= self.limits.elevated_tilt_deg
            and tilt_rise >= self.limits.minimum_tilt_rise_deg
            and tilt_is_still_rising
        ):
            signals.append("gravity_tilt_rising")
        if (
            sample.gravity_tilt_deg >= self.limits.elevated_tilt_deg
            and sample.roll_pitch_rate_rad_s
            >= self.limits.elevated_roll_pitch_rate_rad_s
            and tilt_is_still_rising
        ):
            signals.append("roll_pitch_rate_high")
        if (
            pelvis_drop >= self.limits.minimum_pelvis_drop_m
            and recent_pelvis_fall >= self.limits.minimum_recent_pelvis_fall_m
        ):
            signals.append("pelvis_height_falling")
        return tuple(signals)

    def _is_deteriorating(
        self,
        sample: G1BodyDynamics,
        signals: Iterable[str],
    ) -> bool:
        signal_set = set(signals)
        if len(signal_set) >= 2:
            return True
        baseline = float(self._baseline_pelvis_height_m)
        critical_tilt = (
            sample.gravity_tilt_deg >= self.limits.critical_tilt_deg
            and bool(signal_set)
        )
        critical_height = (
            baseline - sample.pelvis_height_m >= self.limits.critical_pelvis_drop_m
            and bool(signal_set)
        )
        return critical_tilt or critical_height

    def _observe_recovery(self, sample: G1BodyDynamics) -> None:
        self._recovery_ticks += 1
        baseline = float(self._baseline_pelvis_height_m)
        recovered_now = (
            sample.gravity_tilt_deg <= self.limits.recovered_tilt_deg
            and sample.roll_pitch_rate_rad_s
            <= self.limits.recovered_roll_pitch_rate_rad_s
            and baseline - sample.pelvis_height_m <= self.limits.recovered_pelvis_drop_m
        )
        self._stable_recovery_ticks = (
            self._stable_recovery_ticks + 1 if recovered_now else 0
        )
        if self._stable_recovery_ticks >= self.limits.recovery_confirmation_ticks:
            self._recovered = True
            return
        if self._recovery_ticks >= self.limits.recovery_timeout_ticks:
            raise G1BalanceRecoveryError(self.report())


def measure_g1_body_dynamics(
    *,
    imu_orientation_wxyz: Iterable[float],
    imu_angular_velocity_world_rad_s: Iterable[float],
    pelvis_height_m: float,
) -> G1BodyDynamics:
    """Convert live IMU pose/velocity into balance signals in the IMU frame."""

    quaternion = np.asarray(tuple(imu_orientation_wxyz), dtype=np.float64).reshape(-1)
    angular_velocity_world = np.asarray(
        tuple(imu_angular_velocity_world_rad_s),
        dtype=np.float64,
    ).reshape(-1)
    if quaternion.size != 4 or not np.all(np.isfinite(quaternion)):
        raise ValueError(
            f"G1 IMU orientation must be a finite wxyz quaternion: {quaternion!r}"
        )
    if angular_velocity_world.size != 3 or not np.all(
        np.isfinite(angular_velocity_world)
    ):
        raise ValueError(
            "G1 IMU angular velocity must be a finite world-frame vector: "
            f"{angular_velocity_world!r}"
        )
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-8:
        raise ValueError("G1 IMU orientation quaternion has zero norm.")
    w, x, y, z = (quaternion / norm).tolist()
    imu_to_world = np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )
    angular_velocity_imu = imu_to_world.T @ angular_velocity_world
    up_alignment = float(np.clip(imu_to_world[2, 2], -1.0, 1.0))
    return G1BodyDynamics(
        gravity_tilt_deg=float(math.degrees(math.acos(up_alignment))),
        roll_pitch_rate_rad_s=float(np.linalg.norm(angular_velocity_imu[:2])),
        pelvis_height_m=float(pelvis_height_m),
    )


__all__ = [
    "G1BalanceController",
    "G1BalanceLimits",
    "G1BalanceRecoveryError",
    "G1BalanceReport",
    "G1BodyDynamics",
    "measure_g1_body_dynamics",
]
