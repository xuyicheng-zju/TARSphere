"""Completed tools own a runtime payload; the runner does not forge wait."""

from __future__ import annotations

import pytest

from tarsphere.harness.actions.router import requires_environment_execution
from tarsphere.harness.core.failures import FailureReason, tool_failure
from tarsphere.harness.core.types import ToolCall, ToolResult, tool_turn_action
from tarsphere.infra.episode.episode_runner import _execute_navigation_or_action


def test_tool_result_requires_exactly_one_payload_form() -> None:
    with pytest.raises(ValueError, match="runtime payload"):
        ToolResult(call_id="x", name="locate_object")
    with pytest.raises(ValueError, match="one runtime payload form"):
        ToolResult(
            call_id="x",
            name="locate_object",
            action={"action": "locate_object"},
            actions=({"action": "locate_object"},),
        )


def test_tool_failure_owns_the_tool_identity() -> None:
    result = tool_failure(
        ToolCall(tool="locate_object", args={"target": "cup"}),
        FailureReason.NOT_IN_VIEW,
        "not in view",
    )
    assert result.action == {"action": "locate_object"}
    assert result.is_error is True
    assert not requires_environment_execution(
        {
            **result.action,
            "is_error": True,
            "call_name": "locate_object",
        }
    )


def test_requires_environment_execution_is_payload_complete() -> None:
    assert requires_environment_execution(
        {
            "action": "move_forward",
            "distance_m": 0.3,
        }
    )
    assert requires_environment_execution(
        {
            "action": "wait",
            "seconds": 0.5,
        }
    )
    assert requires_environment_execution(
        {
            "action": "grasp_object",
            "hand": "right",
            "preset": "grasp",
        }
    )
    assert requires_environment_execution(
        {
            "action": "navigate_to_pointgoal",
            "navigation_command": {"backend": "x_navdp"},
        }
    )
    assert not requires_environment_execution(tool_turn_action("locate_object"))
    assert not requires_environment_execution(tool_turn_action("remember_entity"))
    assert not requires_environment_execution(
        {
            "action": "grasp_object",
            "is_error": True,
            "hand": "right",
            "preset": "grasp",
        }
    )
    assert not requires_environment_execution(
        {
            "action": "navigate_to_pointgoal",
            "navigation_command": {
                "backend": "astar_pointgoal",
                "phase": "already_reached",
            },
        }
    )
    with pytest.raises(ValueError, match="no action"):
        requires_environment_execution({})


def test_runner_observes_completed_in_handler_tools_without_stepping() -> None:
    class Environment:
        def __init__(self) -> None:
            self._physics_steps = 12
            self.steps: list[dict] = []
            self.observations = 0

        def step(self, decision):
            self.steps.append(decision)
            raise AssertionError("in-handler tools must not step physics")

        def observe(self):
            self.observations += 1
            return {"pose": {"position": [0.0, 0.0, 0.8]}}

    environment = Environment()
    observation, done, info = _execute_navigation_or_action(
        environment,
        None,
        {
            "action": "locate_object",
            "call_name": "locate_object",
            "tool_results": [
                {
                    "name": "locate_object",
                    "content": "saved",
                    "is_error": False,
                }
            ],
        },
        {"pose": {"position": [0.0, 0.0, 0.8]}},
    )
    assert observation == {"pose": {"position": [0.0, 0.0, 0.8]}}
    assert done is False
    assert environment.observations == 1
    assert environment.steps == []
    assert info["executed"] is True
    assert info["feedback"] == "saved"
    assert info["action_physics_steps"] == 0
    assert info["physics_steps"] == 12


@pytest.mark.parametrize(
    "decision",
    [
        {"action": "precise_rotate", "controller_action": {"move_by_speed": [0, 0, 1]}},
        {
            "action": "precise_approach",
            "controller_action": {"move_by_speed": [0.3, 0, 0]},
        },
        {
            "action": "follow_path",
            "controller_action": {"move_along_path": [[[0, 0, 0], [1, 0, 0]]]},
        },
        {
            "action": "place_object",
            "controller_metadata": {"g1_place_ik": {"target": [1, 0, 0.8]}},
        },
    ],
)
def test_runner_executes_compiled_private_controller_exactly_once(decision):
    from tarsphere.harness.actions.router import decision_has_compiled_controller

    class Environment:
        _physics_steps = 12

        def __init__(self):
            self.steps = []

        def step(self, payload):
            self.steps.append(payload)
            self._physics_steps += 60
            return (
                {"after": True},
                False,
                {
                    "executed": True,
                    "physics_steps": self._physics_steps,
                },
            )

        def observe(self):
            raise AssertionError(
                "An unexecuted compiled command is not a completed tool"
            )

    assert decision_has_compiled_controller(decision)
    assert requires_environment_execution(decision)
    environment = Environment()
    observation, done, info = _execute_navigation_or_action(
        environment,
        None,
        decision,
        {"before": True},
    )
    assert environment.steps == [decision]
    assert observation == {"after": True}
    assert not done
    assert info["executed"]
    assert info["physics_steps"] == 72


@pytest.mark.parametrize(
    "error",
    [
        {"is_error": True},
        {"tool_results": [{"name": "navigation", "is_error": True}]},
    ],
)
def test_compiled_controller_never_overrides_a_failed_tool(error):
    assert not requires_environment_execution(
        {
            "action": "precise_rotate",
            "controller_action": {"move_by_speed": [0, 0, 1]},
            **error,
        }
    )


def test_runner_preserves_failed_in_handler_tool_feedback() -> None:
    class Environment:
        _physics_steps = 12

        def step(self, decision):
            raise AssertionError("failed in-handler tools must not step physics")

        def observe(self):
            return {"pose": {"position": [0.0, 0.0, 0.8]}}

    _observation, done, info = _execute_navigation_or_action(
        Environment(),
        None,
        {
            "action": "grasp_object",
            "call_name": "grasp_object",
            "is_error": True,
            "tool_call_index": 1,
            "tool_results": [
                {
                    # A failed suffix can be the only returned result even though
                    # its original action index is greater than zero.
                    "name": "grasp_object",
                    "content": "target is out of reach",
                    "is_error": True,
                    "error_type": "out_of_reach",
                }
            ],
        },
        {"pose": {"position": [0.0, 0.0, 0.8]}},
    )
    assert done is False
    assert info["executed"] is False
    assert info["feedback"] == "[out_of_reach] target is out of reach"
    assert info["action_physics_steps"] == 0


def test_in_handler_grasp_feedback_does_not_claim_an_object() -> None:
    from tarsphere.harness.memory.text import TextMemory

    class Observation:
        text = ""

    memory = TextMemory()
    memory.begin_action_step(
        action_step=0,
        observation=Observation(),
        perception=None,
    )
    memory.record_executed_action(
        policy_step=0,
        action={"action": "grasp_object"},
    )
    memory.complete_action_step(
        {
            "executed": True,
            "action": "grasp_object",
            "feedback": "out of reach",
        }
    )
    assert memory.grasped_hands == []


def test_physical_grasp_feedback_still_requires_the_object() -> None:
    from tarsphere.harness.memory.text import TextMemory

    class Observation:
        text = ""

    memory = TextMemory()
    memory.begin_action_step(
        action_step=0,
        observation=Observation(),
        perception=None,
    )
    memory.record_executed_action(
        policy_step=0,
        action={"action": "grasp_object", "hand": "right", "preset": "grasp"},
    )
    with pytest.raises(ValueError, match="target or object_id"):
        memory.complete_action_step(
            {
                "executed": True,
                "action": "grasp_object",
                "hand": "right",
                "preset": "grasp",
            }
        )


def test_information_hold_wait_is_gone() -> None:
    import tarsphere.harness.tools.handlers as handlers

    assert not hasattr(handlers, "INFORMATION_HOLD_ACTION")
    assert tool_turn_action("inspect_entity") != {
        "action": "wait",
        "seconds": 0.5,
    }
