"""Measured execution quality, independent of milestone scores and control.

Efficiency is actual/reference, never success-weighted or clipped at one.
References must be supplied explicitly for the exact frozen episode; a pending
Oracle review is not automatically admitted. The paper has not fixed severity
weights, so safety preserves raw evidence and publishes no invented scalar.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping


def episode_content_sha256(episode: Any) -> str:
    """Bind reference denominators to the complete authored episode content."""
    payload = episode.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ExecutionReference:
    """Explicit reviewed denominators; this type does not certify an Oracle."""

    # Authored episode identity and content digest, not a scene-only match.
    episode_id: str
    episode_sha256: str
    # Physics ticks from the admitted same-episode Oracle execution, from its
    # first policy observation to termination; reset/warmup ticks are excluded.
    oracle_physics_steps: int
    # Total shortest admissible route length for this complete episode, metres.
    shortest_admissible_route_m: float
    # Auditable artifact identities for the two independently measured sources.
    oracle_source: str
    route_source: str

    def __post_init__(self) -> None:
        for name in ("episode_id", "oracle_source", "route_source"):
            if (
                not isinstance(getattr(self, name), str)
                or not getattr(self, name).strip()
            ):
                raise ValueError(f"{name} must identify a nonempty reference source.")
        if len(self.episode_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.episode_sha256
        ):
            raise ValueError("episode_sha256 must be a lowercase SHA-256 digest.")
        _integer(self.oracle_physics_steps, "oracle_physics_steps", positive=True)
        _number(
            self.shortest_admissible_route_m,
            "shortest_admissible_route_m",
            positive=True,
        )


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < (1 if positive else 0)
    ):
        raise ValueError(
            f"{name} must be a {'positive' if positive else 'nonnegative'} integer."
        )
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        raise ValueError(
            f"{name} must be finite and {'positive' if positive else 'nonnegative'}."
        )
    return number


def execution_quality(
    *,
    episode: Any,
    result: Mapping[str, Any],
    measurements: Mapping[str, Any],
    reference: ExecutionReference | None = None,
) -> dict[str, Any]:
    """Build JSON-ready metrics from recorded physics, with explicit gaps.

    ``path_length_m`` must be a complete physics-sampled path integral.
    A final displacement, sum of decision-frame chords, action count, wall time,
    or incomplete subset is not a substitute for a missing physical measure.
    """
    if result.get("episode_id") != episode.episode_id:
        raise ValueError("Execution metrics result belongs to a different episode.")
    if reference is not None and (
        reference.episode_id != episode.episode_id
        or reference.episode_sha256 != episode_content_sha256(episode)
    ):
        raise ValueError("Efficiency reference belongs to different episode content.")
    source = measurements.get("source")
    physics_steps = measurements.get("physics_steps")
    travelled = measurements.get("path_length_m")
    if (physics_steps is not None or travelled is not None) and (
        source != "native_physics_tick_pelvis_xy"
        or measurements.get("status") != "available"
    ):
        raise ValueError(
            "Efficiency actuals require complete native physics measurements."
        )
    if physics_steps is not None:
        physics_steps = _integer(physics_steps, "physics_steps")
    if travelled is not None:
        travelled = _number(travelled, "path_length_m")
    missing = list(measurements.get("missing_evidence") or [])
    if physics_steps is None:
        missing.append("physics_steps")
    if travelled is None:
        missing.append("complete_physics_sampled_distance")
    if reference is None:
        missing.append("reviewed_same_episode_reference")
    efficiency = {
        "status": "available" if not missing else "unavailable",
        "missing_evidence": missing,
        "physics_steps": physics_steps,
        "travelled_distance_m": travelled,
        "measurement_source": source,
        "starting_physics_step": measurements.get("starting_physics_step"),
        "sampled_physics_steps": measurements.get("sampled_physics_steps"),
        "physics_steps_ratio": (
            physics_steps / reference.oracle_physics_steps
            if reference is not None and physics_steps is not None
            else None
        ),
        "distance_ratio": (
            travelled / reference.shortest_admissible_route_m
            if reference is not None and travelled is not None
            else None
        ),
        "reference": asdict(reference) if reference is not None else None,
    }

    latest = measurements.get("execution_safety") or {}
    totals = latest.get("contact_totals")
    measured_contacts = None
    if totals is not None:
        if latest.get("source") != "native_physx_contact_reports":
            raise ValueError("Safety contact metrics require native PhysX evidence.")
        measured_contacts = {}
        for target in ("scene", "human"):
            group = totals[target]
            measured_contacts[target] = {
                "collider_pair_events": _integer(
                    group["collider_pair_events"], "collider_pair_events"
                ),
                "contact_reports": _integer(
                    group["contact_reports"], "contact_reports"
                ),
                "normal_impulse_ns": _number(
                    group["normal_impulse_ns"], "normal_impulse_ns"
                ),
                "pair_report_duration_s": _number(
                    group["pair_report_duration_s"], "pair_report_duration_s"
                ),
                "touch_pair_events": _integer(
                    group["touch_pair_events"], "touch_pair_events"
                ),
                "touch_reports": _integer(group["touch_reports"], "touch_reports"),
                "unknown_touch_reports": _integer(
                    group["unknown_touch_reports"], "unknown_touch_reports"
                ),
                "pair_touch_duration_s": _number(
                    group["pair_touch_duration_s"], "pair_touch_duration_s"
                ),
            }
    trace = result.get("trace")
    drops = None
    instability = None
    if isinstance(trace, list) and trace:
        if all(isinstance(row.get("dropped_objects"), list) for row in trace):
            # A drop terminates this uninterrupted episode. Repeated terminal
            # feedback for one object is not another physical loss onset.
            drops = len(
                {object_id for row in trace for object_id in row["dropped_objects"]}
            )
        if all(isinstance(row.get("grasp_failure"), dict) for row in trace):
            instability = sum(
                row["grasp_failure"].get("code") == "retention_lost" for row in trace
            )
    violations = result.get("safety_violations")
    falls = (
        sum(item.get("type") == "robot_fall" for item in violations)
        if isinstance(violations, list)
        else None
    )
    safety = {
        "status": "unavailable",
        "score": None,
        "score_status": "weights_not_defined",
        "missing_evidence": [],
        "contacts": measured_contacts,
        "contact_duration_basis": latest.get("contact_duration_basis"),
        "contact_physics_hz": latest.get("contact_physics_hz"),
        "contact_coverage": latest.get("coverage"),
        "falls": falls,
        "object_drops": drops,
        "grasp_retention_failures": instability,
    }
    for name, value in (
        ("native_contact_impulse_and_duration", measured_contacts),
        ("falls", falls),
        ("object_drops", drops),
        ("grasp_retention_failures", instability),
    ):
        if value is None:
            safety["missing_evidence"].append(name)
    if measured_contacts is not None and any(
        group["unknown_touch_reports"] > 0 for group in measured_contacts.values()
    ):
        safety["missing_evidence"].append("physical_touch_classification")
    safety["status"] = (
        "available"
        if not safety["missing_evidence"]
        else "unavailable"
        if all(
            value is None
            for value in (
                measured_contacts,
                falls,
                drops,
                instability,
            )
        )
        else "partial"
    )
    return {"efficiency": efficiency, "safety": safety}


def attach_execution_quality(
    result: dict[str, Any],
    *,
    episode: Any,
    environment: Any,
    reference: ExecutionReference | None = None,
) -> None:
    """Runtime result entry point; never invokes models or modifies success."""
    measure = getattr(environment, "execution_measurements", None)
    measurements = measure() if callable(measure) else {}
    result["execution_measurements"] = measurements
    result["execution_quality"] = execution_quality(
        episode=episode,
        result=result,
        measurements=measurements,
        reference=reference,
    )
