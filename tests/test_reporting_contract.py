from types import SimpleNamespace

import pytest

from tarsphere.bench import summarize_paper_metrics as exported_paper_metrics
from tarsphere.bench.reporting import (
    EpisodeCost,
    EpisodeMeasurement,
    summarize_paper_metrics,
)
from tarsphere.contracts.task_profile import CapabilitySlice, TaskMainline


def _measurement(
    condition: CapabilitySlice,
    *,
    success: bool,
    progress: float,
    decisions: int = 10,
    pair_id: str | None = None,
    cost: EpisodeCost = EpisodeCost(),
    execution_quality=None,
    included: bool = True,
) -> EpisodeMeasurement:
    report = SimpleNamespace(
        task_profile=SimpleNamespace(
            mainline=TaskMainline.NAVIGATE_AND_REPORT,
            capability_slice=condition,
        ),
        task_progress=SimpleNamespace(success=success, completion_rate=progress),
        decision_steps=decisions,
        action_steps=decisions,
    )
    return EpisodeMeasurement(
        report=report,
        cost=cost,
        execution_quality=execution_quality,
        pair_id=pair_id,
        included_in_metrics=included,
    )


def test_paper_average_is_unweighted_and_pairs_require_explicit_matches():
    rows = [
        _measurement(
            CapabilitySlice.BASE,
            success=True,
            progress=1.0,
            pair_id="pair-1",
        ),
        _measurement(
            CapabilitySlice.PERCEPTUAL_ROBUSTNESS,
            success=False,
            progress=0.2,
            pair_id="pair-1",
        ),
        _measurement(
            CapabilitySlice.PERCEPTUAL_ROBUSTNESS,
            success=True,
            progress=0.8,
            pair_id="unmatched-variant",
        ),
    ]
    rows.extend(
        _measurement(condition, success=False, progress=0.0)
        for condition in CapabilitySlice
        if condition
        not in {CapabilitySlice.BASE, CapabilitySlice.PERCEPTUAL_ROBUSTNESS}
    )

    metrics = summarize_paper_metrics(rows)
    average = metrics.family_averages[0]
    assert average.condition_count == 6
    assert average.valid_episode_count == 7
    assert average.success_rate == pytest.approx(0.25)
    assert average.progress == pytest.approx(0.25)
    assert average.mean_decisions == 10.0
    assert average.mean_tokens_per_decision is None
    assert average.token_episode_count == 0
    assert average.efficiency_distance_ratio is None
    assert average.efficiency_episode_count == 0

    paired = metrics.paired_differences[0]
    assert paired.condition == CapabilitySlice.PERCEPTUAL_ROBUSTNESS.value
    assert paired.pair_count == 1
    assert paired.success_delta == -1.0
    assert paired.progress_delta == pytest.approx(-0.8)


def test_paper_metrics_apply_per_metric_evidence_rules():
    complete_safety = {
        "efficiency": {"status": "available", "distance_ratio": 1.5},
        "safety": {
            "contacts": {
                "scene": {"touch_pair_events": 2},
                "human": {"touch_pair_events": 1},
            },
            "falls": 0,
        },
    }
    rows = [
        _measurement(
            CapabilitySlice.BASE,
            success=True,
            progress=1.0,
            decisions=10,
            cost=EpisodeCost(
                input_tokens=80,
                output_tokens=20,
                policy_usage_complete=True,
            ),
            execution_quality=complete_safety,
        ),
        _measurement(
            CapabilitySlice.BASE,
            success=False,
            progress=0.4,
            decisions=30,
            cost=EpisodeCost(
                input_tokens=60,
                output_tokens=30,
                policy_usage_complete=True,
            ),
            execution_quality={
                "efficiency": {"status": "available", "distance_ratio": 9.0},
                "safety": {"contacts": None, "falls": 1},
            },
        ),
        _measurement(
            CapabilitySlice.PERCEPTUAL_ROBUSTNESS,
            success=True,
            progress=1.0,
            execution_quality={
                "efficiency": {"status": "available", "distance_ratio": 7.0},
                "safety": {},
            },
        ),
    ]

    metrics = summarize_paper_metrics(rows)
    base = next(cell for cell in metrics.cells if cell.cell.endswith("/base"))
    variant = next(
        cell for cell in metrics.cells if cell.cell.endswith("/perceptual_robustness")
    )
    assert base.efficiency_distance_ratio == 1.5
    assert base.efficiency_episode_count == 1
    assert variant.efficiency_distance_ratio is None
    assert variant.efficiency_episode_count == 0
    assert base.mean_tokens_per_decision == pytest.approx(6.5)
    assert base.token_episode_count == 2
    assert base.mean_touch_pair_onsets is None
    assert base.contact_episode_count == 1
    assert base.fall_episode_rate == 0.5
    assert base.fall_episode_count == 2

    complete = summarize_paper_metrics(
        [
            _measurement(
                CapabilitySlice.BASE,
                success=True,
                progress=1.0,
                execution_quality=complete_safety,
            )
        ]
    ).cells[0]
    assert complete.mean_touch_pair_onsets == 3.0
    assert complete.contact_episode_count == 1


def test_family_execution_metrics_pool_valid_runs_across_conditions():
    execution = {
        "efficiency": {"status": "available", "distance_ratio": 1.25},
        "safety": {
            "contacts": {
                "scene": {"touch_pair_events": 2},
                "human": {"touch_pair_events": 1},
            },
            "falls": 0,
        },
    }
    rows = [
        _measurement(
            condition,
            success=True,
            progress=1.0,
            decisions=10,
            cost=EpisodeCost(
                input_tokens=80,
                output_tokens=20,
                policy_usage_complete=True,
            ),
            execution_quality=execution,
        )
        for condition in CapabilitySlice
    ]
    rows.append(
        _measurement(
            CapabilitySlice.PERCEPTUAL_ROBUSTNESS,
            success=False,
            progress=0.0,
            decisions=20,
            cost=EpisodeCost(
                input_tokens=300,
                output_tokens=100,
                policy_usage_complete=True,
            ),
            execution_quality=execution,
        )
    )

    family = summarize_paper_metrics(rows).family_averages[0]
    assert family.valid_episode_count == 7
    assert family.mean_decisions == pytest.approx(80 / 7)
    assert family.mean_tokens_per_decision == pytest.approx(80 / 7)
    assert family.token_episode_count == 7
    assert family.efficiency_distance_ratio == 1.25
    assert family.efficiency_episode_count == 1
    assert family.mean_touch_pair_onsets == 3.0
    assert family.contact_episode_count == 7
    assert family.fall_episode_rate == 0.0
    assert family.fall_episode_count == 7


def test_paper_metrics_leave_missing_evidence_undefined():
    assert exported_paper_metrics is summarize_paper_metrics
    row = _measurement(
        CapabilitySlice.BASE,
        success=False,
        progress=0.0,
        included=False,
    )
    assert summarize_paper_metrics([row]).cells == ()

    included = _measurement(
        CapabilitySlice.BASE,
        success=False,
        progress=0.0,
    )
    metrics = summarize_paper_metrics([included])
    cell = metrics.cells[0]
    assert cell.mean_tokens_per_decision is None
    assert cell.token_episode_count == 0
    assert cell.efficiency_distance_ratio is None
    assert cell.efficiency_episode_count == 0
    assert cell.mean_touch_pair_onsets is None
    assert cell.contact_episode_count == 0
    assert cell.fall_episode_rate is None
    assert cell.fall_episode_count == 0
    assert metrics.family_averages == ()
    assert metrics.paired_differences == ()
