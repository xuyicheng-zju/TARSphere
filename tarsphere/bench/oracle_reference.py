"""Build auditable Oracle execution references from native result evidence."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping


CONTACT_FIELDS = (
    "collider_pair_events",
    "contact_reports",
    "normal_impulse_ns",
    "pair_report_duration_s",
    "touch_pair_events",
    "touch_reports",
    "unknown_touch_reports",
    "pair_touch_duration_s",
)


def _contact_group(totals: Mapping[str, Any], target: str) -> dict[str, int | float]:
    group = totals.get(target)
    if not isinstance(group, Mapping):
        raise ValueError(f"Oracle result is missing {target} native contact totals.")
    return {field: group[field] for field in CONTACT_FIELDS}


def build_oracle_reference_row(
    result: Mapping[str, Any],
    *,
    result_path: Path | str,
    episode_path: Path | str,
    episode_sha256: str,
    task_mainline: str,
    grasp_executions: int | None = None,
) -> dict[str, Any]:
    """Extract one lossless collision reference plus table-ready summaries."""

    execution = result.get("execution_measurements")
    if not isinstance(execution, Mapping) or execution.get("status") != "available":
        raise ValueError("Oracle result has no complete native execution measurements.")
    safety = execution.get("execution_safety")
    if not isinstance(safety, Mapping):
        raise ValueError("Oracle result has no native execution safety evidence.")
    if safety.get("source") != "native_physx_contact_reports":
        raise ValueError(
            "Oracle collision reference must come from native PhysX reports."
        )
    totals = safety.get("contact_totals")
    if not isinstance(totals, Mapping):
        raise ValueError("Oracle result has no native contact totals.")

    quality_safety = (result.get("execution_quality") or {}).get("safety") or {}
    trace = result.get("trace")
    if not isinstance(trace, list):
        raise ValueError("Oracle result has no decision trace.")
    contact_decisions = []
    phase_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    contact_objects: set[str] = set()
    for decision in trace:
        native = decision.get("native_contacts") or []
        public = decision.get("physical_contact_details") or []
        if not decision.get("physical_contact") and not native and not public:
            continue
        phase_value = decision.get("reference_phase")
        if not phase_value:
            thinking = str((decision.get("envelope") or {}).get("thinking") or "")
            prefix = "GT reference phase:"
            phase_value = (
                thinking.split(prefix, 1)[1].strip()
                if prefix in thinking
                else "unknown"
            )
        phase = str(phase_value)
        action = str(
            decision.get("robot_action") or decision.get("call_name") or "unknown"
        )
        phase_counts[phase] += 1
        action_counts[action] += 1
        for contact in native:
            other_path = contact.get("other_path")
            if other_path:
                contact_objects.add(str(other_path))
        contact_decisions.append(
            {
                "step": decision.get("step"),
                "reference_phase": decision.get("reference_phase"),
                "robot_action": decision.get("robot_action"),
                "call_name": decision.get("call_name"),
                "executed": decision.get("executed"),
                "physical_contact_details": public,
                "native_contacts": native,
            }
        )

    incidents = safety.get("contact_incidents")
    if not isinstance(incidents, list):
        raise ValueError("Oracle result has no native contact incident ledger.")
    for incident in incidents:
        other_path = incident.get("other_path")
        if other_path:
            contact_objects.add(str(other_path))
    violations = result.get("safety_violations")
    if not isinstance(violations, list):
        raise ValueError("Oracle result has no safety-violation ledger.")

    result_path = Path(result_path)
    episode_path = Path(episode_path)
    return {
        "episode_id": result["episode_id"],
        "task_mainline": task_mainline,
        "capability_slice": "base",
        "oracle_closed": result.get("reference_status") == "closed_pending_review",
        "reference_status": result.get("reference_status"),
        "causal_graph_completed": bool(result.get("causal_graph_completed")),
        "termination_reason": result.get("termination_reason"),
        "outcome_status": result.get("outcome_status"),
        "decision_steps": result.get("decision_steps"),
        "physics_steps": execution.get("physics_steps"),
        "travelled_distance_m": execution.get("path_length_m"),
        "wall_time_s": result.get("wall_clock_s"),
        "grasp_executions": grasp_executions,
        "collision_count": safety.get("collision_count"),
        "contact_decision_count": len(contact_decisions),
        "contact_incident_count": len(incidents),
        "scene_contacts": _contact_group(totals, "scene"),
        "human_contacts": _contact_group(totals, "human"),
        "falls": quality_safety.get("falls"),
        "object_drops": quality_safety.get("object_drops"),
        "grasp_retention_failures": quality_safety.get("grasp_retention_failures"),
        "safety_violation_count": len(violations),
        "safety_violation_types": sorted(str(item.get("type")) for item in violations),
        "contact_phase_counts": dict(sorted(phase_counts.items())),
        "contact_action_counts": dict(sorted(action_counts.items())),
        "contact_object_paths": sorted(contact_objects),
        "contact_coverage": safety.get("coverage"),
        "contact_duration_basis": safety.get("contact_duration_basis"),
        "contact_physics_hz": safety.get("contact_physics_hz"),
        # The detailed ledgers make the table auditable without changing the
        # immutable source result. They are not collapsed into a safety score.
        "contact_incidents": incidents,
        "contact_decisions": contact_decisions,
        "safety_violations": violations,
        "result_path": str(result_path),
        "result_sha256": sha256(result_path.read_bytes()).hexdigest(),
        "episode_path": str(episode_path),
        "episode_sha256": episode_sha256,
    }


def aggregate_oracle_reference_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Return the Oracle rows that belong beside model cells in the main table."""

    values = list(rows)
    groups = [("all/base", values)]
    for mainline in sorted({str(row["task_mainline"]) for row in values}):
        groups.append(
            (
                f"{mainline}/base",
                [row for row in values if row["task_mainline"] == mainline],
            )
        )
    output = []
    for cell, group in groups:
        if not group:
            continue
        output.append(
            {
                "model": "Oracle (GT-guided physical reference)",
                "cell": cell,
                "episode_count": len(group),
                "success_rate": fmean(float(row["oracle_closed"]) for row in group),
                "mean_decision_steps": fmean(row["decision_steps"] for row in group),
                "mean_physics_steps": fmean(row["physics_steps"] for row in group),
                "mean_travelled_distance_m": fmean(
                    row["travelled_distance_m"] for row in group
                ),
                "mean_wall_time_s": fmean(row["wall_time_s"] for row in group),
                "mean_collision_count": fmean(row["collision_count"] for row in group),
                "mean_scene_normal_impulse_ns": fmean(
                    row["scene_contacts"]["normal_impulse_ns"] for row in group
                ),
                "mean_scene_pair_touch_duration_s": fmean(
                    row["scene_contacts"]["pair_touch_duration_s"] for row in group
                ),
                "mean_human_collision_count": fmean(
                    row["human_contacts"]["touch_pair_events"] for row in group
                ),
                "fall_episode_rate": fmean(
                    float((row["falls"] or 0) > 0) for row in group
                ),
                "object_drop_episode_rate": fmean(
                    float((row["object_drops"] or 0) > 0) for row in group
                ),
                "retention_failure_episode_rate": fmean(
                    float((row["grasp_retention_failures"] or 0) > 0) for row in group
                ),
            }
        )
    return tuple(output)


def json_cell(value: Any) -> str:
    """Stable compact JSON for nested CSV cells."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "CONTACT_FIELDS",
    "aggregate_oracle_reference_rows",
    "build_oracle_reference_row",
    "json_cell",
]
