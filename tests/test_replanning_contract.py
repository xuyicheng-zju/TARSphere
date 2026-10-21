"""Queued actions stop when new evidence invalidates their assumptions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tarsphere.contracts import DecisionRequirements
from tarsphere.harness.agent import RobotAgent
from tarsphere.harness.core.types import DecisionEnvelope, ToolCall, TurnResult
from tarsphere.harness.harness import AgentHarnessConfig
from tarsphere.harness.tools.handlers import GraspObjectHandler, RememberEntityHandler


class _Perception:
    def process(self, observation, metadata):
        return None

    def reset(self):
        pass


class _Memory:
    def __init__(self):
        self.completed_feedback = []
        self.semantic = SimpleNamespace(landmarks=())
        self.strategy_planner = object()

    def begin_action_step(self, **kwargs):
        return SimpleNamespace(visual_history=[])

    def complete_action_step(self, feedback):
        self.completed_feedback.append(dict(feedback))

    def record_policy_decision(self, **kwargs):
        pass

    def record_executed_action(self, **kwargs):
        pass

    def reset(self):
        pass


class _Kernel:
    def __init__(self):
        self.run_calls = []
        self.vlm = SimpleNamespace()

    def run(self, **kwargs):
        self.run_calls.append(kwargs)
        call = ToolCall(call_id="fresh", tool="wait", args={"seconds": 0.5})
        return TurnResult(
            status="completed",
            actions=[
                {
                    "action": "wait",
                    "call_name": "wait",
                    "tool_args": {"seconds": 0.5},
                    "tool_call_index": 0,
                }
            ],
            envelope=DecisionEnvelope(thinking="replan", actions=[call]),
        )

    def reset(self):
        pass


class _Tools:
    def spec_names(self):
        return ()


def _observation(*, public_events=None):
    observation = {
        "views": {
            "ego": {
                "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                "depth": np.ones((8, 8), dtype=np.float32),
                "camera_params": {},
            }
        },
        "pose": {"body_posture": "standing"},
    }
    if public_events is not None:
        observation["public_events"] = public_events
    return observation


def _agent() -> tuple[RobotAgent, _Kernel]:
    kernel = _Kernel()
    agent = RobotAgent(
        agent_harness_config=AgentHarnessConfig(),
        perception=_Perception(),
        memory=_Memory(),
        kernel=kernel,
        tools=_Tools(),
    )
    return agent, kernel


def _queue(agent: RobotAgent, *names: str) -> None:
    agent._pending_actions = [
        {
            "action": name,
            "call_name": name,
            "policy_step": 0,
            "tool_call_index": index,
            "action_index": index,
            "action_sequence_length": len(names) + 1,
        }
        for index, name in enumerate(names, start=1)
    ]
    agent._queued_policy_step = 0
    agent._queued_sequence_length = len(names) + 1


def test_public_event_discards_queue_and_preserves_requirement_for_replan():
    agent, kernel = _agent()
    _queue(agent, "move_forward", "turn_left")
    requirement = DecisionRequirements(
        planning={},
        event_milestone_ids=("task_update",),
    )

    action = agent.act(
        _observation(),
        "help",
        [],
        decision_requirements=requirement,
    )

    assert action["call_name"] == "wait"
    assert action["replan_reason"] == "public_event"
    assert [row["call_name"] for row in action["discarded_action_details"]] == [
        "move_forward",
        "turn_left",
    ]
    assert kernel.run_calls[0]["decision_requirements"] == requirement


def test_failed_tool_feedback_discards_suffix_before_next_inference():
    agent, kernel = _agent()
    _queue(agent, "turn_left", "move_forward")
    agent._active_memory_step = 0

    agent.update_info(
        {
            "executed": False,
            "action": "locate_object",
            "feedback": "target was not visible",
        }
    )

    assert agent.execution_stats["pending_actions"] == 0
    action = agent.act(_observation(), "help", [])
    assert len(kernel.run_calls) == 1
    assert action["replan_reason"] == "execution_failure"
    assert action["discarded_actions"] == 2
    assert {row["reason"] for row in action["discarded_action_details"]} == {
        "execution_failure_replan"
    }


def test_measured_executed_false_feedback_uses_same_replan_boundary():
    agent, _kernel = _agent()
    _queue(agent, "move_forward")
    agent._active_memory_step = 0

    agent.update_info(
        {
            "executed": False,
            "action": "move_body",
            "feedback": "requested motion was refused",
            "motion_outcome": {"available": True},
        }
    )

    assert agent.execution_stats["pending_actions"] == 0
    assert agent.execution_stats["discarded_actions"] == 1


def test_successful_feedback_keeps_queue_without_extra_inference():
    agent, kernel = _agent()
    _queue(agent, "turn_left")
    agent._active_memory_step = 0
    agent.update_info({"executed": True, "action": "move_forward"})

    action = agent.act(_observation(), "help", [])

    assert action["call_name"] == "turn_left"
    assert action["policy_inference"] is False
    assert kernel.run_calls == []
    assert action["discarded_actions"] == 0


def test_event_at_policy_cap_requires_inference_before_queue_is_mutated():
    agent, kernel = _agent()
    _queue(agent, "turn_left", "move_forward")
    requirement = DecisionRequirements(
        planning={},
        event_milestone_ids=("late_task_update",),
    )

    assert agent.requires_policy_inference(requirement) is True
    assert agent.execution_stats["pending_actions"] == 2
    assert kernel.run_calls == []

    # This is the runner's existing terminal path when the admission check
    # finds that the policy budget is already exhausted.
    discarded = agent.discard_pending_actions("decision_budget_exhausted")
    assert len(discarded) == 2
    assert agent.execution_stats["pending_actions"] == 0
    assert kernel.run_calls == []


def test_policy_cap_still_admits_an_unchanged_successful_suffix():
    agent, kernel = _agent()
    _queue(agent, "turn_left")

    assert agent.requires_policy_inference(None) is False
    action = agent.act(_observation(), "help", [])

    assert action["call_name"] == "turn_left"
    assert action["policy_inference"] is False
    assert kernel.run_calls == []


def test_human_task_update_without_decision_requirements_replans():
    agent, kernel = _agent()
    _queue(agent, "turn_left", "move_forward")
    observation = _observation(
        public_events=[
            {
                "event_type": "language",
                "speaker": "Nora",
                "text": "Please take the bottle to Alex instead.",
                "step": 7,
            }
        ]
    )

    assert agent.requires_policy_inference(observation=observation) is True
    action = agent.act(observation, "help", [])

    assert action["replan_reason"] == "public_event"
    assert action["discarded_actions"] == 2
    assert len(kernel.run_calls) == 1


def test_robot_utterance_echo_does_not_interrupt_queue():
    agent, kernel = _agent()
    _queue(agent, "turn_left")
    observation = _observation(
        public_events=[
            {
                "event_type": "language",
                "speaker": "robot",
                "text": "I will check it now.",
                "step": 3,
                "recipient": "Nora",
                "delivered": True,
            }
        ]
    )

    assert agent.requires_policy_inference(observation=observation) is False
    action = agent.act(observation, "help", [])

    assert action["call_name"] == "turn_left"
    assert action["policy_inference"] is False
    assert kernel.run_calls == []


def test_terminal_discard_does_not_infer_or_defer_trace_accounting():
    agent, kernel = _agent()
    _queue(agent, "turn_left", "move_forward")

    details = agent.discard_pending_actions("task_completed")

    assert len(details) == 2
    assert agent.execution_stats["discarded_actions"] == 2
    assert agent.execution_stats["pending_actions"] == 0
    assert kernel.run_calls == []
    assert agent._deferred_discarded_action_details == []


def test_planning_shape_without_event_identity_does_not_cancel_queue():
    agent, kernel = _agent()
    _queue(agent, "turn_left")

    action = agent.act(
        _observation(),
        "help",
        [],
        decision_requirements=DecisionRequirements(planning={}),
    )

    assert action["call_name"] == "turn_left"
    assert action["policy_inference"] is False
    assert kernel.run_calls == []


def test_removed_multicall_grasp_and_memory_arguments_are_rejected():
    grasp = GraspObjectHandler(
        semantic_backend=object(),
        grasp_backend=object(),
    ).execute(
        ToolCall(
            tool="grasp_object",
            args={"target": "cup", "confirm_preview_id": "old-preview"},
        ),
        {},
    )
    remember = RememberEntityHandler(
        memory=object(), semantic_backend=object()
    ).execute(
        ToolCall(
            tool="remember_entity",
            args={
                "entity_id": "cup",
                "entity_type": "object",
                "label": "cup",
                "select": "A",
            },
        ),
        {},
    )

    assert grasp.is_error
    assert "no `confirm_preview_id`" in grasp.content
    assert remember.is_error
    assert "no `select`" in remember.content


def test_runner_stops_before_inference_when_final_queue_receives_event(monkeypatch):
    from tarsphere.infra.episode import episode_runner as runner_module

    class _BudgetAgent:
        def __init__(self):
            self.agent_harness_config = AgentHarnessConfig()
            self.pending = 1
            self.act_calls = 0
            self.discarded = []

        @property
        def execution_stats(self):
            return {"pending_actions": self.pending}

        def reset(self):
            # Model decision 100 has already admitted one queued suffix.
            runner._policy_decisions = 1
            runner.trace.append(
                {
                    "discarded_actions": 1,
                    "discarded_action_details": [
                        {
                            "call_name": "grasp_object",
                            "reason": "feedback_boundary",
                        }
                    ],
                }
            )

        def configure_episode(self, episode):
            pass

        def configure_episode_humans(self, humans):
            pass

        def requires_policy_inference(self, requirements=None, *, observation=None):
            return RobotAgent._decision_boundary_requires_replan(
                observation=observation,
                requirements=requirements,
            )

        def discard_pending_actions(self, reason):
            self.pending = 0
            self.discarded = [{"call_name": "turn_left", "reason": reason}]
            return list(self.discarded)

        def act(self, *args, **kwargs):
            self.act_calls += 1
            raise AssertionError("the runner admitted policy decision 101")

    class _Human:
        human_ids = ()
        task_bindings = None
        _disruption = SimpleNamespace(profile=None, reset=lambda: None)

        def tick(self, *args, **kwargs):
            return []

        def reset(self):
            pass

        def get_replay_log(self):
            return []

    class _Environment:
        _physics_steps = 240
        physics_steps = 240
        render_count = 1
        render_mode = None

        def reset(self):
            return _observation()

    episode = SimpleNamespace(
        episode_id="budget_boundary",
        runtime=SimpleNamespace(max_decision_steps=1, max_physics_steps=None),
        evaluation=None,
        humans=(),
    )
    agent = _BudgetAgent()
    runner = runner_module.AutonomousEpisodeRunner(
        episode=episode,
        environment=_Environment(),
        agent=agent,
        human=_Human(),
    )
    monkeypatch.setattr(runner_module, "resolve_agent_action_space", lambda *args: [])
    monkeypatch.setattr(runner, "_record_robot_position", lambda observation: None)
    monkeypatch.setattr(runner, "_get_world_snapshot", lambda **kwargs: {})
    monkeypatch.setattr(
        runner, "_update_private_oracle_snapshot", lambda snapshot: None
    )

    def deliver_update(step, observation):
        runner._pending_public_events.append(
            {
                "event_type": "language",
                "speaker": "Nora",
                "text": "The destination is now Alex.",
                "step": step,
            }
        )
        return observation, [{"human_id": "Nora", "action": "speak"}]

    monkeypatch.setattr(runner, "_tick_human_actors", deliver_update)

    result = runner.run(max_decision_steps=1)

    assert result["decision_steps"] == 1
    assert result["termination_reason"] == "decision_budget_exhausted"
    assert result["terminal_boundary"]["public_events"][0]["speaker"] == "Nora"
    assert result["terminal_boundary"]["human_events"] == [
        {"human_id": "Nora", "action": "speak"}
    ]
    assert agent.act_calls == 0
    assert agent.discarded == [
        {"call_name": "turn_left", "reason": "decision_budget_exhausted"}
    ]
    assert result["trace"][0]["discarded_actions"] == 2
    assert result["trace"][0]["discarded_action_details"] == [
        {"call_name": "grasp_object", "reason": "feedback_boundary"},
        {"call_name": "turn_left", "reason": "decision_budget_exhausted"},
    ]
