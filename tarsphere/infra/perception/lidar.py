"""Body-frame range scans from live PhysX collision geometry.

Each bearing samples a vertical ray fan and reports the minimum horizontal
range, ``slant * cos(elevation)``. Floor hits and returns above the robot's
collision envelope are excluded. Bearings are denser in the forward sector;
measurements remain discrete and contain no object identities.

The scan has a finite range and no reflectivity, dropout, or range-noise model.
Scene meshes need collision geometry to produce returns, so the runtime audits
collider coverage before sampling. ``IsaacAutonomousEnv`` supplies raycasts;
this module defines beam geometry, validation, and model-facing text.

The private online map fuses the same scan with synchronized RGB-D. Its sector
interpolation estimates free space between measured rays. Policy observations
receive the current scan and the supported sensor-derived map visualization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

from tarsphere.infra.control.physical_contracts import (
    G1_STANDING_BODY_TOP_M,
    LOCOMOTION_MAX_STEP_DISTANCE_M,
    LIDAR_COARSE_AZIMUTH_STEP_DEG,
    LIDAR_FINE_AZIMUTH_STEP_DEG,
    LIDAR_FORWARD_SECTOR_DEG,
    LIDAR_FULL_HEIGHT_REFERENCE_M,
    LIDAR_MAX_RANGE_M,
    LIDAR_ORIGIN_HEIGHT_M,
    LIDAR_VERTICAL_GAP_AT_MAX_LEG_M,
)


def beam_azimuths_deg() -> tuple[float, ...]:
    """Body-frame bearings of the fan, in the agent's own sign convention.

    ``0`` is straight ahead and positive is left (counter-clockwise), matching
    ``move_body.delta[2]`` and the bearing already reported for physical
    contact. Bearings are returned in ``(-180, 180]`` and ordered from left to
    right, so reading the list top to bottom sweeps the way the robot would turn
    for a negative yaw command.

    The grid is deliberately **not** uniform. Angular spacing buys spatial
    resolution in inverse proportion to distance, and the distance that matters
    is the one the agent commits a forward leg to, so the forward sector is
    sampled finely enough that the beams stay closer together than the body is
    wide out to the longest leg. Behind the shoulders the coarse spacing is kept:
    backward motion is capped far shorter, and every extra bearing is another
    line the model has to read.
    """

    fine_step = float(LIDAR_FINE_AZIMUTH_STEP_DEG)
    coarse_step = float(LIDAR_COARSE_AZIMUTH_STEP_DEG)
    sector = float(LIDAR_FORWARD_SECTOR_DEG)
    for name, step in (("fine", fine_step), ("coarse", coarse_step)):
        if step <= 0.0 or not math.isfinite(step):
            raise ValueError(f"LiDAR {name} azimuth step must be positive.")
    # Forward, both flanks, and dead astern are each sampled exactly, so "can I
    # move this way" has one unambiguous reading rather than an interpolation
    # between two beams. That requires both steps to divide the quadrant.
    for name, step in (("fine", fine_step), ("coarse", coarse_step)):
        if abs(90.0 / step - round(90.0 / step)) > 1e-9:
            raise ValueError(
                f"LiDAR {name} azimuth step {step} deg does not divide 90 "
                "deg, so forward/left/right/rear would not all be sampled."
            )
    if abs(sector / fine_step - round(sector / fine_step)) > 1e-9:
        raise ValueError(
            f"LiDAR forward sector {sector} deg is not a whole number of "
            f"{fine_step} deg steps, so the two grids would not meet."
        )
    if abs(sector / coarse_step - round(sector / coarse_step)) > 1e-9:
        raise ValueError(
            f"LiDAR forward sector {sector} deg is not a whole number of "
            f"{coarse_step} deg steps, so the sector edge would fall between "
            "two coarse bearings and leave a gap wider than either grid."
        )

    bearings: set[float] = set()
    steps_fine = int(round(sector / fine_step))
    for index in range(-steps_fine, steps_fine + 1):
        bearings.add(round(index * fine_step, 6))
    steps_coarse = int(round(360.0 / coarse_step))
    for index in range(steps_coarse):
        bearing = index * coarse_step
        if bearing > 180.0:
            bearing -= 360.0
        bearings.add(round(bearing, 6))
    return tuple(sorted(bearings, reverse=True))


def beam_elevations_deg() -> tuple[float, ...]:
    """Elevation angles of the vertical fan cast at every bearing.

    The fan is defined by what it has to cover rather than by an authored angle,
    so it tracks the body it protects: at
    ``LIDAR_FULL_HEIGHT_REFERENCE_M`` the lowest beam is at floor level and
    the highest is at the top of the head, and the layers are spaced evenly in
    *height* there. Spacing in height rather than in angle is what makes the
    guarantee statable — the layer count is then whatever keeps the spacing at
    the longest forward leg under ``LIDAR_VERTICAL_GAP_AT_MAX_LEG_M``.

    Angles are degrees, positive up, and returned lowest first.
    """

    reference = float(LIDAR_FULL_HEIGHT_REFERENCE_M)
    body_top = float(G1_STANDING_BODY_TOP_M)
    origin = float(LIDAR_ORIGIN_HEIGHT_M)
    if not 0.0 < origin < body_top:
        raise ValueError(
            f"LiDAR origin {origin} m must sit between the floor and the "
            f"top of the body ({body_top} m)."
        )
    # Heights spread over the reference distance scale linearly with range, so
    # the gap at the longest leg is the gap at the reference times their ratio.
    stretch = float(LOCOMOTION_MAX_STEP_DISTANCE_M) / reference
    intervals = math.ceil(body_top * stretch / float(LIDAR_VERTICAL_GAP_AT_MAX_LEG_M))
    intervals = max(1, int(intervals))
    return tuple(
        round(
            math.degrees(math.atan2(body_top * index / intervals - origin, reference)),
            6,
        )
        for index in range(intervals + 1)
    )


def vertical_gap_m(distance_m: float) -> float:
    """Height between two adjacent fan layers at one distance.

    This is the sensor's own honest statement of what it can straddle: a rise
    shorter than this, standing clear of the floor, can pass between two layers.
    """

    elevations = beam_elevations_deg()
    heights = [
        float(distance_m) * math.tan(math.radians(angle)) for angle in elevations
    ]
    return (
        max(upper - lower for lower, upper in zip(heights, heights[1:]))
        if len(heights) > 1
        else 0.0
    )


def azimuth_gap_m(distance_m: float, *, forward_only: bool = True) -> float:
    """Arc between two adjacent beams at one distance.

    ``forward_only`` measures the refined sector the agent commits legs into;
    the full circle includes the coarse rear grid and is correspondingly wider.
    """

    step = (
        float(LIDAR_FINE_AZIMUTH_STEP_DEG)
        if forward_only
        else float(LIDAR_COARSE_AZIMUTH_STEP_DEG)
    )
    return 2.0 * float(distance_m) * math.sin(math.radians(step) / 2.0)


@dataclass(frozen=True)
class LidarScan:
    """One instantaneous scan, folded to horizontal distance in body frame.

    Attributes:
        bearings_deg: Body-frame azimuths, ordered left to right. Carried
            explicitly rather than recomputed because the grid is non-uniform
            and a replayed trace must mean what it meant when it was recorded.
        ranges_m: One entry per bearing. Each is the smallest horizontal
            distance measured by a sampled elevation layer inside the standing
            body's height band. ``None`` means no layer struck anything the body
            could collide with, which is reported as such rather than as a large
            number the agent would read as a measurement.
        max_range_m: Sensor limit. Distances are clamped to it, never beyond.
        origin_height_m: Height of the fan's apex above the floor.
        elevations_deg: The layers actually cast, lowest first.
    """

    bearings_deg: tuple[float, ...]
    ranges_m: tuple[Optional[float], ...]
    max_range_m: float = LIDAR_MAX_RANGE_M
    origin_height_m: float = LIDAR_ORIGIN_HEIGHT_M
    elevations_deg: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        bearings = tuple(round(float(value), 6) for value in self.bearings_deg)
        if not bearings:
            raise ValueError("LiDAR needs at least one bearing.")
        if len(bearings) != len(self.ranges_m):
            raise ValueError(
                f"LiDAR has {len(bearings)} bearings but {len(self.ranges_m)} returns."
            )
        if len(set(bearings)) != len(bearings):
            raise ValueError(f"LiDAR bearings repeat: {bearings}")
        if list(bearings) != sorted(bearings, reverse=True):
            raise ValueError(f"LiDAR bearings must run left to right: {bearings}")
        if not all(-180.0 < value <= 180.0 for value in bearings):
            raise ValueError(f"LiDAR bearings must lie in (-180, 180]: {bearings}")
        # Forward is the bearing every translation magnitude is judged against,
        # so a grid that does not sample it exactly cannot answer the question
        # the sensor exists for without interpolating between two beams.
        if 0.0 not in bearings:
            raise ValueError(f"LiDAR must sample straight ahead exactly: {bearings}")
        if not math.isfinite(self.max_range_m) or self.max_range_m <= 0.0:
            raise ValueError(
                f"LiDAR max_range_m must be positive: {self.max_range_m!r}"
            )
        cleaned: list[Optional[float]] = []
        for bearing, value in zip(bearings, self.ranges_m):
            if value is None:
                cleaned.append(None)
                continue
            distance = float(value)
            if not math.isfinite(distance) or distance < 0.0:
                raise ValueError(
                    f"LiDAR return at {bearing:+.1f} deg must be a "
                    f"non-negative finite distance or None: {value!r}"
                )
            # A beam that reached the limit carries no information about what is
            # beyond it, so it is recorded as no-return rather than as a
            # measurement exactly at the boundary.
            cleaned.append(None if distance >= self.max_range_m else distance)
        object.__setattr__(self, "bearings_deg", bearings)
        object.__setattr__(self, "ranges_m", tuple(cleaned))
        object.__setattr__(
            self,
            "elevations_deg",
            tuple(round(float(value), 6) for value in self.elevations_deg),
        )

    def as_mapping(self) -> dict[float, Optional[float]]:
        """Bearing-keyed view, for traces and tests."""

        return dict(zip(self.bearings_deg, self.ranges_m))

    def detections(self) -> tuple[tuple[float, float], ...]:
        """Bearings that struck something, with their distance, in sweep order."""

        return tuple(
            (bearing, value)
            for bearing, value in zip(self.bearings_deg, self.ranges_m)
            if value is not None
        )

    def open_bearings_deg(self) -> tuple[float, ...]:
        """Bearings where no layer of the fan struck anything measurable."""

        return tuple(
            bearing
            for bearing, value in zip(self.bearings_deg, self.ranges_m)
            if value is None
        )

    def nearest(self) -> Optional[tuple[float, float]]:
        """The closest detection and its bearing, or ``None`` if all are clear.

        Ties resolve to the bearing nearest straight ahead, because that is the
        one a forward leg would be commanded along.
        """

        detections = self.detections()
        if not detections:
            return None
        return min(detections, key=lambda item: (item[1], abs(item[0])))

    def range_at(self, bearing_deg: float) -> Optional[float]:
        """Return the beam nearest one bearing, wrapping across ±180."""

        def separation(candidate: float) -> float:
            return abs((candidate - float(bearing_deg) + 180.0) % 360.0 - 180.0)

        nearest = min(self.bearings_deg, key=separation)
        return self.as_mapping()[nearest]

    def as_dict(self) -> dict[str, object]:
        """JSON-compatible evidence for the result trace."""

        return {
            "sensor": "lidar",
            "geometry": "azimuth_elevation_fan",
            "max_range_m": float(self.max_range_m),
            "origin_height_m": float(self.origin_height_m),
            "elevations_deg": [float(value) for value in self.elevations_deg],
            "frame": "body",
            "bearing_convention": "zero_forward_positive_left",
            "distance_convention": "horizontal_from_body_surface",
            "returns": [
                {
                    "bearing_deg": float(bearing),
                    "distance_m": None if value is None else float(value),
                }
                for bearing, value in zip(self.bearings_deg, self.ranges_m)
            ],
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "LidarScan":
        """Rebuild a scan from :meth:`as_dict`, for replay and tests."""

        returns = payload.get("returns")
        if not isinstance(returns, Sequence) or not returns:
            raise ValueError("LiDAR payload needs a non-empty returns list.")
        ordered = sorted(
            (dict(entry) for entry in returns),  # type: ignore[arg-type]
            key=lambda entry: -float(entry["bearing_deg"]),
        )
        elevations = payload.get("elevations_deg") or ()
        return cls(
            bearings_deg=tuple(float(entry["bearing_deg"]) for entry in ordered),
            ranges_m=tuple(
                None if entry.get("distance_m") is None else float(entry["distance_m"])
                for entry in ordered
            ),
            max_range_m=float(
                payload.get("max_range_m", LIDAR_MAX_RANGE_M)  # type: ignore[arg-type]
            ),
            origin_height_m=float(
                payload.get("origin_height_m", LIDAR_ORIGIN_HEIGHT_M)  # type: ignore[arg-type]
            ),
            elevations_deg=tuple(
                float(value)
                for value in elevations  # type: ignore[union-attr]
            ),
        )


def _bearing_label(bearing_deg: float) -> str:
    """Print a bearing exactly.

    Beam spacing need not be a whole number of degrees — the refined forward
    grid is 7.5 deg apart — and rounding the label would name a bearing no beam
    was cast at, which is a misreported measurement rather than a cosmetic
    difference.
    """

    return f"{bearing_deg:+.1f}".removesuffix(".0") + "°"


def describe_lidar(scan: LidarScan | None) -> str:
    """Render the scan as the text block the model reads.

    Text, not an image, and this is the point rather than an implementation
    convenience: the RGB-D condition established that a VLM does not recover
    numbers from a colour ramp, while it uses a labelled number directly. Each
    line is a bearing in the same convention as the yaw command and a distance
    in the same metres as the translation command, so choosing a magnitude is a
    comparison rather than an estimate.

    Only bearings that struck something are listed, in sweep order. The rest are
    named on one line as returning nothing, so the report's length follows how
    cluttered the surroundings are instead of how densely the sensor samples —
    which is what allows the fan to be dense enough to resolve the body width.
    """

    if scan is None:
        return ""
    limit = f"{scan.max_range_m:.1f} m"
    # Static frame/unit semantics live in the system prompt. Repeating that
    # paragraph every turn makes the current numbers harder to find.
    lines: list[str] = []

    detections = scan.detections()
    if detections:
        nearest = scan.nearest()
        assert nearest is not None
        lines.append(
            f"Nearest obstacle: {nearest[1]:.2f} m at {_bearing_label(nearest[0])}."
        )
        lines.append(
            "Measured returns (left to right): "
            + "; ".join(
                f"{_bearing_label(bearing)} {value:.2f} m"
                for bearing, value in detections
            )
            + "."
        )
    else:
        lines.append(f"No obstacle detected at any bearing within {limit}.")

    clear = scan.open_bearings_deg()
    if clear:
        lines.append(
            f"No return within {limit} at: "
            + ", ".join(_bearing_label(bearing) for bearing in clear)
            + ". No sampled ray returned; this does not prove a body-width path."
        )
    return "\n".join(lines)


def build_lidar(
    distances_m: Iterable[Optional[float]],
    *,
    bearings_deg: Sequence[float] | None = None,
    max_range_m: float = LIDAR_MAX_RANGE_M,
    origin_height_m: float = LIDAR_ORIGIN_HEIGHT_M,
    elevations_deg: Sequence[float] | None = None,
) -> LidarScan:
    """Validate per-bearing horizontal distances in contract bearing order."""

    return LidarScan(
        bearings_deg=tuple(
            beam_azimuths_deg() if bearings_deg is None else bearings_deg
        ),
        ranges_m=tuple(distances_m),
        max_range_m=float(max_range_m),
        origin_height_m=float(origin_height_m),
        elevations_deg=tuple(
            beam_elevations_deg() if elevations_deg is None else elevations_deg
        ),
    )


__all__ = [
    "LidarScan",
    "azimuth_gap_m",
    "beam_azimuths_deg",
    "beam_elevations_deg",
    "build_lidar",
    "describe_lidar",
    "vertical_gap_m",
]
