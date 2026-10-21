"""Contract tests for the audited event/reaction and watchdog fixes.

Covers: delayed adaptation reactions are scored at the first post-event
submission; event requirements accumulate without interrupting a serial
action block; queued action rows do not replay event-window outputs; and the
stuck watchdog survives interleaved non-locomotion decisions.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tarsphere.harness.core.types import (
    AgentObservation,
    CameraViewObservation,
)
from tarsphere.contracts.decision_requirements import DecisionRequirements
from tarsphere.evaluation.engine import milestone_dag_reaches
from tarsphere.evaluation.event_vocabulary import EventType
from tarsphere.evaluation.deterministic_analyzers import (
    _action_calls,
    _recovery_failure,
)
from tarsphere.infra.episode.rollout_control import ProgressWatchdog


# ---------------------------------------------------------------------------
# Delayed adaptation/planning reactions (engine)
# ---------------------------------------------------------------------------


def _evaluation_spec(milestone_id: str, evaluator: str):
    """A duck-typed one-node spec; milestone_dag_reaches only indexes steps."""
    return SimpleNamespace(
        milestones=[
            SimpleNamespace(
                milestone_id=milestone_id,
                preconditions=(),
                event=SimpleNamespace(type=EventType.TARGET_REVISED),
                outcome=SimpleNamespace(
                    evaluator=evaluator,
                    ground_truth={},
                ),
            )
        ]
    )


def _step(index: int, agent_state: dict | None = None, milestone_id: str = "adapt"):
    return SimpleNamespace(
        decision_step=index,
        physics_step=index,
        observation={},
        agent_state=dict(agent_state or {}),
        action={"name": "move_body", "calls": []},
        feedback={},
        hidden_state={},
        condition_values={},
        reaction_values={},
        reaction_evidence={},
        occurred_events=((milestone_id,) if index == 1 else ()),
    )


def test_change_event_waits_for_actual_completion_without_a_model_update():
    spec = _evaluation_spec("adapt", "human_event_completed")
    steps = [_step(index) for index in range(4)]
    steps[3].reaction_values["adapt"] = True
    reaches = milestone_dag_reaches(spec, steps)
    assert reaches[0].event_index == 1
    assert reaches[0].reaction_index == 3


def test_failure_attribution_uses_only_the_executed_serial_call():
    first = SimpleNamespace(
        decision_step=0,
        action={
            "name": "move_body",
            "tool": "move_body",
            "args": {"delta": [1.0, 0.0, 0.0]},
            "tool_call_index": 0,
            "calls": [
                {"tool": "move_body", "args": {"delta": [1.0, 0.0, 0.0]}},
                {"tool": "ask", "args": {"recipient": "Nora", "message": "Where?"}},
            ],
        },
        feedback={"executed": False, "feedback": "front contact"},
    )
    second = SimpleNamespace(
        decision_step=1,
        action={
            "name": "move_body",
            "tool": "move_body",
            "args": {"delta": [1.0, 0.0, 0.0]},
            "tool_call_index": 0,
            "calls": [{"tool": "move_body", "args": {"delta": [1.0, 0.0, 0.0]}}],
            "model_visible_action_steps": [0, 1],
        },
        feedback={"executed": False, "feedback": "front contact"},
    )
    recovered = SimpleNamespace(
        decision_step=2,
        action={
            "name": "move_body",
            "tool": "move_body",
            "args": {"delta": [0.0, 0.2, 0.0]},
            "tool_call_index": 0,
        },
        feedback={"executed": True, "feedback": ""},
    )

    assert _action_calls(first) == (
        (
            "move_body",
            {"delta": [1.0, 0.0, 0.0]},
        ),
    )
    verdict = _recovery_failure(
        SimpleNamespace(
            trajectory=SimpleNamespace(steps=(first, second, recovered)),
        )
    )
    assert verdict.occurred is True
    assert verdict.recovered is True


def test_planning_reaction_is_scored_at_first_post_event_submission() -> None:
    spec = _evaluation_spec("plan", "structured_plan_submitted")
    steps = [
        _step(0, milestone_id="plan"),
        _step(1, milestone_id="plan"),
        _step(
            2, {"plan": {"goal": "study", "steps": ["doorway"]}}, milestone_id="plan"
        ),
    ]

    reaches = milestone_dag_reaches(spec, steps)

    assert reaches[0].event_index == 1
    assert reaches[0].reaction_index == 2


def test_inspection_reaction_is_scored_at_the_first_submission() -> None:
    """A later conclusion cannot replace the first eligible submission."""
    spec = _evaluation_spec("inspect", "inspection_submitted")
    steps = [
        _step(0, milestone_id="inspect"),
        _step(1, milestone_id="inspect"),
        _step(2, {"inspection": "first guess"}, milestone_id="inspect"),
        _step(3, milestone_id="inspect"),
        _step(
            4,
            {"inspection": "closer look: the beige bottle is upright in the tray"},
            milestone_id="inspect",
        ),
    ]

    reaches = milestone_dag_reaches(spec, steps)

    assert reaches[0].event_index == 1
    assert reaches[0].reaction_index == 2


@pytest.mark.parametrize("malformed", ["", "  ", 7])
def test_malformed_inspection_does_not_consume_first_submission(malformed) -> None:
    spec = _evaluation_spec("inspect", "inspection_submitted")
    steps = [
        _step(0, milestone_id="inspect"),
        _step(1, milestone_id="inspect"),
        _step(2, {"inspection": malformed}, milestone_id="inspect"),
        _step(3, {"inspection": "The surface is blue."}, milestone_id="inspect"),
    ]
    reaches = milestone_dag_reaches(spec, steps)
    assert reaches[0].reaction_index == 3


def test_repeated_inspection_preserves_completed_descendants() -> None:
    from tarsphere.evaluation.schema import MilestoneReference

    spec = _evaluation_spec("inspect", "inspection_submitted")
    spec.milestones.append(
        SimpleNamespace(
            milestone_id="recipient_seen",
            preconditions=(MilestoneReference(milestone_ref="inspect"),),
            event=SimpleNamespace(
                type="task_phase_opened", human_actions=(), task_updates=()
            ),
            outcome=SimpleNamespace(evaluator="human_visible", ground_truth={}),
        )
    )
    steps = [
        _step(0, milestone_id="inspect"),
        _step(1, milestone_id="inspect"),
        _step(2, {"inspection": "The bottle is upright."}, milestone_id="inspect"),
        _step(3, milestone_id="recipient_seen"),
    ]
    steps[-1].occurred_events = ("recipient_seen",)
    steps[-1].reaction_values = {"recipient_seen": True}
    completed = milestone_dag_reaches(spec, steps)
    assert completed[1].reaction_index == 3

    steps.append(
        _step(
            4,
            {"inspection": "The bottle is upright in the tray."},
            milestone_id="inspect",
        )
    )

    assert milestone_dag_reaches(spec, steps) == completed


# ---------------------------------------------------------------------------
# Requirements survive until the next policy inference
# ---------------------------------------------------------------------------


def test_event_requirements_merge_without_interrupting_the_action_block() -> None:
    first = DecisionRequirements(
        planning={},
        event_milestone_ids=("plan_revision_1",),
    )
    second = DecisionRequirements(
        planning={},
        event_milestone_ids=("plan_revision_2",),
    )
    from tarsphere.harness.agent import RobotAgent

    merged = RobotAgent._merge_decision_requirements(first, second)
    assert merged is not None
    assert merged.planning == {}
    assert merged.event_milestone_ids == (
        "plan_revision_1",
        "plan_revision_2",
    )


# ---------------------------------------------------------------------------
# Queued rows do not replay event-window outputs
# ---------------------------------------------------------------------------


def _observation_fixture() -> AgentObservation:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    return AgentObservation(
        views={
            "ego": CameraViewObservation(
                name="ego",
                rgb=rgb,
                depth=None,
                camera_params={},
            )
        },
        pose={"position": [0.0, 0.0, 0.8], "yaw_rad": 0.0},
        visible_humans={},
    )


def test_queued_action_rows_strip_event_window_outputs() -> None:
    from tarsphere.harness.core.types import EVENT_WINDOW_OUTPUT_FIELDS

    parent_envelope = {
        "thinking": "Walk, then report the inspection conclusion.",
        "actions": [{"tool": "move_body", "args": {"delta": [1, 0, 0]}}],
        "memory": [],
        "inspection": "pre-open guess",
        "plan": {"goal": "study"},
    }

    stripped = {
        key: value
        for key, value in parent_envelope.items()
        if key not in EVENT_WINDOW_OUTPUT_FIELDS
    }

    assert "inspection" not in stripped
    assert "plan" not in stripped
    assert stripped["thinking"] == "Walk, then report the inspection conclusion."


# ---------------------------------------------------------------------------
# Watchdog survives non-locomotion decisions
# ---------------------------------------------------------------------------


def _heartbeat(action: str) -> dict:
    return {
        "action": action,
        "xy": (0.0, 0.0),
        "yaw_rad": 0.0,
        "goal_distance_m": 5.0,
        "heading_error_rad": 0.5,
    }


def test_watchdog_streak_survives_interleaved_camera_tilt() -> None:
    watchdog = ProgressWatchdog()
    move = dict(_heartbeat("move_body"), phase="agent_locomotion")

    # Four no-progress locomotion decisions...
    for _ in range(4):
        assert watchdog.observe(dict(move)) is None

    # ...an interleaved camera re-aim must not clear the window...
    assert watchdog.observe(_heartbeat("tilt_camera")) is None

    # ...so the fifth locomotion decision emits the incident.
    details = watchdog.observe(dict(move))
    assert details is not None
    assert details["streak_decisions"] == 5
    assert details.get("terminal") is False


def test_watchdog_agent_locomotion_window_spans_motion_primitives() -> None:
    watchdog = ProgressWatchdog()
    move = dict(_heartbeat("move_body"), phase="agent_locomotion")
    for _ in range(4):
        watchdog.observe(dict(move))

    # All agent locomotion shares one window key, so switching the motion
    # primitive does NOT start a fresh window: alternating primitives is
    # exactly how a stuck robot would otherwise evade the streak.
    turn = dict(_heartbeat("turn_left"), phase="agent_locomotion")
    details = watchdog.observe(turn)
    assert details is not None
    assert details["streak_decisions"] == 5
