"""Uniform task, execution-quality and cost aggregation for benchmark cells."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable

from tarsphere.contracts.task_profile import CapabilitySlice
from tarsphere.evaluation.schema import EpisodeEvaluationReport


PAPER_CONDITIONS = tuple(item.value for item in CapabilitySlice)


@dataclass(frozen=True)
class EpisodeCost:
    """Provider/runtime costs recorded beside one official evaluation report."""

    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time_s: float = 0.0
    api_cost_usd: float | None = None
    policy_usage_complete: bool = False


@dataclass(frozen=True)
class EpisodeMeasurement:
    report: EpisodeEvaluationReport
    cost: EpisodeCost = EpisodeCost()
    execution_quality: dict[str, Any] | None = None
    pair_id: str | None = None
    included_in_metrics: bool = True


@dataclass(frozen=True)
class ContactMetrics:
    """Per-episode mean native evidence for one Human/scene contact category."""

    # Collider-pair onsets, not unique people and not report/tick counts.
    mean_collider_pair_events: float
    # Sum over contact points, in newton-seconds, divided by measured episodes.
    mean_normal_impulse_ns: float
    # Pair-report duration; concurrent pairs add, not wall-clock contact time.
    mean_pair_report_duration_s: float
    # Measured touch requires nonpositive separation or positive impulse.
    mean_touch_pair_events: float
    mean_pair_touch_duration_s: float
    # Missing separation and zero impulse, not classifiable as touch/no-touch.
    mean_unknown_touch_reports: float


@dataclass(frozen=True)
class SafetyMetrics:
    """Raw safety aggregate; each denominator reports its evidence coverage."""

    scene_contacts: ContactMetrics | None
    human_contacts: ContactMetrics | None
    contact_episode_count: int
    # Fraction of measured episodes with at least one physical occurrence.
    fall_episode_rate: float | None
    fall_episode_count: int
    object_drop_episode_rate: float | None
    object_drop_episode_count: int
    grasp_retention_failure_episode_rate: float | None
    grasp_retention_failure_episode_count: int


@dataclass(frozen=True)
class CellMetrics:
    """Raw operational means, retained separately from paper metrics."""

    cell: str
    episode_count: int
    success_rate: float
    milestone_progress: float
    mean_decision_steps: float
    # Completed action/feedback rows, distinct from the model-decision budget.
    mean_action_steps: float
    # Measured execution ticks after reset, never global trajectory timestamps.
    mean_physics_steps: float | None
    mean_model_calls: float
    mean_input_tokens: float
    mean_output_tokens: float
    mean_wall_time_s: float
    mean_api_cost_usd: float | None
    physics_steps_episode_count: int = 0
    # Whole-episode native pelvis XY integral, including non-navigation actions.
    mean_travelled_distance_m: float | None = None
    travelled_distance_episode_count: int = 0
    # Actual/reference ratios; missing references are excluded and counted.
    mean_physics_steps_ratio: float | None = None
    physics_steps_ratio_episode_count: int = 0
    mean_distance_ratio: float | None = None
    distance_ratio_episode_count: int = 0
    # Raw scene/Human severity evidence; no undefined weighted scalar.
    safety: SafetyMetrics | None = None


@dataclass(frozen=True)
class PaperCellMetrics:
    """Published cell metrics and the evidence denominator for each quantity."""

    cell: str
    valid_episode_count: int
    success_rate: float
    progress: float
    mean_decisions: float
    mean_tokens_per_decision: float | None
    token_episode_count: int
    efficiency_distance_ratio: float | None
    efficiency_episode_count: int
    mean_touch_pair_onsets: float | None
    contact_episode_count: int
    fall_episode_rate: float | None
    fall_episode_count: int


@dataclass(frozen=True)
class PaperFamilyAverage:
    """One paper-table row across the six complete cells in a task family."""

    mainline: str
    condition_count: int
    valid_episode_count: int
    success_rate: float
    progress: float
    mean_decisions: float
    mean_tokens_per_decision: float | None
    token_episode_count: int
    efficiency_distance_ratio: float | None
    efficiency_episode_count: int
    mean_touch_pair_onsets: float | None
    contact_episode_count: int
    fall_episode_rate: float | None
    fall_episode_count: int


@dataclass(frozen=True)
class PaperPairedDifference:
    """Mean variant-minus-Base difference over explicitly matched scored pairs."""

    mainline: str
    condition: str
    pair_count: int
    success_delta: float
    progress_delta: float


@dataclass(frozen=True)
class PaperMetrics:
    """Paper-aligned metrics for one evaluated model."""

    cells: tuple[PaperCellMetrics, ...]
    family_averages: tuple[PaperFamilyAverage, ...]
    paired_differences: tuple[PaperPairedDifference, ...]


def summarize_reporting_cells(
    measurements: Iterable[EpisodeMeasurement],
) -> tuple[CellMetrics, ...]:
    """Group official results by task-mainline/capability reporting cell."""

    groups: dict[str, list[EpisodeMeasurement]] = {}
    for measurement in measurements:
        profile = measurement.report.task_profile
        key = f"{profile.mainline.value}/{profile.capability_slice.value}"
        groups.setdefault(key, []).append(measurement)

    summaries: list[CellMetrics] = []
    for key, values in sorted(groups.items()):
        priced = [
            value.cost.api_cost_usd
            for value in values
            if value.cost.api_cost_usd is not None
        ]
        efficiency_values = {}
        for field in (
            "physics_steps",
            "travelled_distance_m",
            "physics_steps_ratio",
            "distance_ratio",
        ):
            efficiency_values[field] = [
                (value.execution_quality or {}).get("efficiency", {}).get(field)
                for value in values
                if (value.execution_quality or {}).get("efficiency", {}).get(field)
                is not None
            ]
        safety = [(value.execution_quality or {}).get("safety", {}) for value in values]
        contacts = [
            value["contacts"] for value in safety if value.get("contacts") is not None
        ]
        contact_means = {
            target: ContactMetrics(
                mean_collider_pair_events=fmean(
                    value[target]["collider_pair_events"] for value in contacts
                ),
                mean_normal_impulse_ns=fmean(
                    value[target]["normal_impulse_ns"] for value in contacts
                ),
                mean_pair_report_duration_s=fmean(
                    value[target]["pair_report_duration_s"] for value in contacts
                ),
                mean_touch_pair_events=fmean(
                    value[target]["touch_pair_events"] for value in contacts
                ),
                mean_pair_touch_duration_s=fmean(
                    value[target]["pair_touch_duration_s"] for value in contacts
                ),
                mean_unknown_touch_reports=fmean(
                    value[target]["unknown_touch_reports"] for value in contacts
                ),
            )
            if contacts
            else None
            for target in ("scene", "human")
        }
        incidents = {
            field: [
                value[field] > 0 for value in safety if value.get(field) is not None
            ]
            for field in ("falls", "object_drops", "grasp_retention_failures")
        }
        summaries.append(
            CellMetrics(
                cell=key,
                episode_count=len(values),
                success_rate=fmean(
                    float(value.report.task_progress.success) for value in values
                ),
                milestone_progress=fmean(
                    value.report.task_progress.completion_rate for value in values
                ),
                mean_decision_steps=fmean(
                    value.report.decision_steps for value in values
                ),
                mean_action_steps=fmean(value.report.action_steps for value in values),
                mean_physics_steps=(
                    fmean(efficiency_values["physics_steps"])
                    if efficiency_values["physics_steps"]
                    else None
                ),
                physics_steps_episode_count=len(efficiency_values["physics_steps"]),
                mean_travelled_distance_m=(
                    fmean(efficiency_values["travelled_distance_m"])
                    if efficiency_values["travelled_distance_m"]
                    else None
                ),
                travelled_distance_episode_count=len(
                    efficiency_values["travelled_distance_m"]
                ),
                mean_model_calls=fmean(value.cost.model_calls for value in values),
                mean_input_tokens=fmean(value.cost.input_tokens for value in values),
                mean_output_tokens=fmean(value.cost.output_tokens for value in values),
                mean_wall_time_s=fmean(value.cost.wall_time_s for value in values),
                mean_api_cost_usd=fmean(priced) if priced else None,
                mean_physics_steps_ratio=(
                    fmean(efficiency_values["physics_steps_ratio"])
                    if efficiency_values["physics_steps_ratio"]
                    else None
                ),
                physics_steps_ratio_episode_count=len(
                    efficiency_values["physics_steps_ratio"]
                ),
                mean_distance_ratio=(
                    fmean(efficiency_values["distance_ratio"])
                    if efficiency_values["distance_ratio"]
                    else None
                ),
                distance_ratio_episode_count=len(efficiency_values["distance_ratio"]),
                safety=SafetyMetrics(
                    scene_contacts=contact_means["scene"],
                    human_contacts=contact_means["human"],
                    contact_episode_count=len(contacts),
                    fall_episode_rate=fmean(incidents["falls"])
                    if incidents["falls"]
                    else None,
                    fall_episode_count=len(incidents["falls"]),
                    object_drop_episode_rate=fmean(incidents["object_drops"])
                    if incidents["object_drops"]
                    else None,
                    object_drop_episode_count=len(incidents["object_drops"]),
                    grasp_retention_failure_episode_rate=(
                        fmean(incidents["grasp_retention_failures"])
                        if incidents["grasp_retention_failures"]
                        else None
                    ),
                    grasp_retention_failure_episode_count=len(
                        incidents["grasp_retention_failures"]
                    ),
                ),
            )
        )
    return tuple(summaries)


def summarize_paper_metrics(
    measurements: Iterable[EpisodeMeasurement],
) -> PaperMetrics:
    """Calculate the metric contract reported by the paper for one model.

    Callers identify Base/variant pairs with ``pair_id`` and mark excluded or
    invalid rows with ``included_in_metrics=False``. Missing cost, path, contact,
    or fall evidence is excluded only from that metric and remains visible in
    its denominator.
    """

    values = tuple(row for row in measurements if row.included_in_metrics)
    groups: dict[str, list[EpisodeMeasurement]] = {}
    for row in values:
        profile = row.report.task_profile
        key = f"{profile.mainline.value}/{profile.capability_slice.value}"
        groups.setdefault(key, []).append(row)

    cells = tuple(
        _paper_cell(key, rows) for key, rows in sorted(groups.items()) if rows
    )
    cell_by_key = {cell.cell: cell for cell in cells}
    mainlines = sorted({row.report.task_profile.mainline.value for row in values})
    averages: list[PaperFamilyAverage] = []
    for mainline in mainlines:
        family_cells = [
            cell_by_key.get(f"{mainline}/{condition}") for condition in PAPER_CONDITIONS
        ]
        if any(cell is None for cell in family_cells):
            continue
        complete_cells = [cell for cell in family_cells if cell is not None]
        family_rows = [
            row for row in values if row.report.task_profile.mainline.value == mainline
        ]
        execution = _paper_cell(f"{mainline}/all", family_rows)
        averages.append(
            PaperFamilyAverage(
                mainline=mainline,
                condition_count=len(complete_cells),
                valid_episode_count=sum(
                    cell.valid_episode_count for cell in complete_cells
                ),
                success_rate=fmean(cell.success_rate for cell in complete_cells),
                progress=fmean(cell.progress for cell in complete_cells),
                mean_decisions=execution.mean_decisions,
                mean_tokens_per_decision=execution.mean_tokens_per_decision,
                token_episode_count=execution.token_episode_count,
                efficiency_distance_ratio=execution.efficiency_distance_ratio,
                efficiency_episode_count=execution.efficiency_episode_count,
                mean_touch_pair_onsets=execution.mean_touch_pair_onsets,
                contact_episode_count=execution.contact_episode_count,
                fall_episode_rate=execution.fall_episode_rate,
                fall_episode_count=execution.fall_episode_count,
            )
        )

    paired_rows: dict[tuple[str, str, str], EpisodeMeasurement] = {}
    for row in values:
        if row.pair_id is None:
            continue
        pair_id = str(row.pair_id).strip()
        if not pair_id:
            raise ValueError("pair_id must be non-empty when provided")
        profile = row.report.task_profile
        key = (profile.mainline.value, pair_id, profile.capability_slice.value)
        if key in paired_rows:
            raise ValueError(f"duplicate paired measurement: {key!r}")
        paired_rows[key] = row

    differences: list[PaperPairedDifference] = []
    for mainline in mainlines:
        pair_ids = sorted(
            pair_id
            for family, pair_id, condition in paired_rows
            if family == mainline and condition == CapabilitySlice.BASE.value
        )
        for condition in PAPER_CONDITIONS:
            if condition == CapabilitySlice.BASE.value:
                continue
            matched = [
                (
                    paired_rows[(mainline, pair_id, CapabilitySlice.BASE.value)],
                    paired_rows[(mainline, pair_id, condition)],
                )
                for pair_id in pair_ids
                if (mainline, pair_id, condition) in paired_rows
            ]
            if not matched:
                continue
            differences.append(
                PaperPairedDifference(
                    mainline=mainline,
                    condition=condition,
                    pair_count=len(matched),
                    success_delta=fmean(
                        float(variant.report.task_progress.success)
                        - float(base.report.task_progress.success)
                        for base, variant in matched
                    ),
                    progress_delta=fmean(
                        variant.report.task_progress.completion_rate
                        - base.report.task_progress.completion_rate
                        for base, variant in matched
                    ),
                )
            )

    return PaperMetrics(
        cells=cells,
        family_averages=tuple(averages),
        paired_differences=tuple(differences),
    )


def _paper_cell(key: str, values: list[EpisodeMeasurement]) -> PaperCellMetrics:
    tokens_per_decision = [
        (row.cost.input_tokens + row.cost.output_tokens) / row.report.decision_steps
        for row in values
        if row.cost.policy_usage_complete and row.report.decision_steps > 0
    ]
    eligible_efficiency_rows = [
        row
        for row in values
        if row.report.task_profile.capability_slice == CapabilitySlice.BASE
        and row.report.task_progress.success
    ]
    distance_ratios = [
        ratio
        for row in eligible_efficiency_rows
        if (quality := row.execution_quality or {}).get("efficiency", {}).get("status")
        == "available"
        and (ratio := quality["efficiency"].get("distance_ratio")) is not None
    ]
    touch_onsets: list[float] = []
    falls: list[bool] = []
    for row in values:
        safety = (row.execution_quality or {}).get("safety", {})
        contacts = safety.get("contacts")
        if isinstance(contacts, dict) and all(
            isinstance(contacts.get(target), dict) for target in ("scene", "human")
        ):
            scene = contacts["scene"].get("touch_pair_events")
            human = contacts["human"].get("touch_pair_events")
            if scene is not None and human is not None:
                touch_onsets.append(float(scene) + float(human))
        if safety.get("falls") is not None:
            falls.append(bool(safety["falls"] > 0))

    return PaperCellMetrics(
        cell=key,
        valid_episode_count=len(values),
        success_rate=fmean(float(row.report.task_progress.success) for row in values),
        progress=fmean(row.report.task_progress.completion_rate for row in values),
        mean_decisions=fmean(row.report.decision_steps for row in values),
        mean_tokens_per_decision=(
            fmean(tokens_per_decision)
            if len(tokens_per_decision) == len(values)
            else None
        ),
        token_episode_count=len(tokens_per_decision),
        efficiency_distance_ratio=(
            fmean(distance_ratios)
            if eligible_efficiency_rows
            and len(distance_ratios) == len(eligible_efficiency_rows)
            else None
        ),
        efficiency_episode_count=len(distance_ratios),
        mean_touch_pair_onsets=(
            fmean(touch_onsets) if len(touch_onsets) == len(values) else None
        ),
        contact_episode_count=len(touch_onsets),
        fall_episode_rate=fmean(falls) if len(falls) == len(values) else None,
        fall_episode_count=len(falls),
    )


__all__ = [
    "CellMetrics",
    "ContactMetrics",
    "EpisodeCost",
    "EpisodeMeasurement",
    "PaperCellMetrics",
    "PaperFamilyAverage",
    "PaperMetrics",
    "PaperPairedDifference",
    "SafetyMetrics",
    "summarize_paper_metrics",
    "summarize_reporting_cells",
]
