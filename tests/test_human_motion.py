"""Grounded symbolic work; synthetic unit evidence, not simulator certification."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tarsphere.infra.control.human_body import HumanBodyResult, HumanBodyState
from tarsphere.infra.control.human_task_execution import execute_human_task
from tarsphere.infra.episode.episode_runner import AutonomousEpisodeRunner
from tarsphere.bench.case_loader import BenchmarkCaseLoader
from tarsphere.bench.schema import TARSphereEpisode


@pytest.mark.parametrize(
    "position,status,active,stale,accepted",
    [
        ([1.0, 2.0, 0.0], "idle", "", False, True),
        ([1.0, 2.0, 0.0], "succeed", "", False, True),
        ([1.0, 2.0, 0.0], "succeed", "goto", False, True),
        ([1.0, 2.0, 0.0], "ongoing", "goto", False, False),
        ([1.0, 2.0, 0.0], "failed", "", False, False),
        ([1.0, 2.0, 0.0], "succeed", "", True, False),
        ([3.0, 2.0, 0.0], "succeed", "", False, False),
    ],
)
def test_work_requires_current_operation_stance(
    position, status, active, stale, accepted
):
    episode = SimpleNamespace(
        world_knowledge={"spatial_anchors": {"stove": [1.0, 2.0, 0.0]}}
    )
    state = HumanBodyState(
        "Nora", position, action_status=status, active_action=active, pose_stale=stale
    )
    action = {
        "command": "perform_task",
        "target_anchor": "stove",
        "task_description": "Start the stove for preparation.",
        "milestone_id": "work_completed",
    }
    result = execute_human_task(episode, "Nora", action, lambda _: state)
    assert result.executed is accepted
    assert (
        state.position == position
    )  # No teleport, animation or fabricated scene mutation.


@pytest.mark.parametrize("partner_x,accepted", [(2.0, True), (4.0, False)])
def test_peer_work_uses_partner_pose_not_robot_pose(partner_x, accepted):
    states = {
        "Nora": HumanBodyState("Nora", [0.0, 0.0, 0.0]),
        "Marcus": HumanBodyState("Marcus", [partner_x, 0.0, 0.0]),
    }
    episode = SimpleNamespace(
        world_knowledge={"spatial_anchors": {"counter": [0.0, 0.0, 0.0]}}
    )
    result = execute_human_task(
        episode,
        "Nora",
        {
            "command": "perform_task",
            "target_anchor": "counter",
            "task_description": "Confirm preparation with Marcus.",
            "partner_human_id": "Marcus",
        },
        states.get,
    )
    assert result.executed is accepted


@pytest.mark.parametrize("command", ["speak", "perform_task"])
def test_rejected_action_cannot_acknowledge_or_publish(command):
    runner = object.__new__(AutonomousEpisodeRunner)
    acknowledged = []
    runner.human = SimpleNamespace(
        acknowledge_action_applied=lambda *a, **kw: acknowledged.append(a)
    )
    runner.environment = SimpleNamespace(
        apply_human_action=lambda *a: HumanBodyResult(
            "Nora", command, False, "not at operation stance"
        )
    )
    runner.event_bus = SimpleNamespace(step=3)
    runner._pending_public_events = []
    with pytest.raises(RuntimeError, match="not at operation stance"):
        runner._apply_human_actions(
            {
                "Nora": {
                    "command": command,
                    "milestone_id": "work",
                    "message": "Done.",
                }
            },
            {},
            timing="before_agent",
        )
    assert acknowledged == []
    assert runner._pending_public_events == []


def collaboration_payload(affected="onsite_target_observed"):
    payload = (
        BenchmarkCaseLoader()
        .load_file(
            Path(__file__).resolve().parents[1]
            / "episodes/case/navigate_and_report/human_coordination.yaml"
        )
        .model_dump(mode="json")
    )
    profile = payload["evaluation"]["task_profile"]
    source = profile["supporting_human_id"]
    receipt_id = profile["coordination_milestone_id"]
    profile.update(
        subtype="task_collaboration",
        coordination_pattern=(
            "preparatory_collaboration"
            if affected == "onsite_target_observed"
            else "parallel_collaboration"
        ),
        coordination_milestone_id="contribution_result_delivered",
        affected_outcome_id=affected,
    )
    human = next(h for h in payload["humans"] if h["id"] == source)
    payload["world_knowledge"]["spatial_anchors"]["work_stance"] = human[
        "start_position"
    ]
    nodes = payload["evaluation"]["milestones"]
    receipt = next(n for n in nodes if n["milestone_id"] == receipt_id)
    receipt["event"]["type"] = "contribution_requested"
    receipt["preconditions"] = [
        {"kind": "milestone", "milestone_ref": "task_understanding"}
    ]
    for node in nodes:
        node["preconditions"] = [
            ref
            for ref in node["preconditions"]
            if ref.get("milestone_ref") != receipt_id
        ]
    nodes.remove(receipt)
    nodes.insert(1, receipt)
    work = {
        "milestone_id": "auxiliary_work_completed",
        "kind": "event",
        "preconditions": [{"kind": "milestone", "milestone_ref": receipt_id}],
        "event": {
            "type": "contribution_completed",
            "description": "Complete auxiliary work.",
            "human_actions": [
                {
                    "human": source,
                    "command": "perform_task",
                    "target_anchor": "work_stance",
                    "task_description": "Check the preparation record.",
                }
            ],
        },
        "completion": {
            "evaluator": "human_event_completed",
            "description": "Actual work completion.",
            "ground_truth": {"milestone_id": "auxiliary_work_completed"},
        },
    }
    nodes.insert(2, work)
    result = {
        "milestone_id": "contribution_result_delivered",
        "kind": "event",
        "preconditions": [
            {"kind": "milestone", "milestone_ref": "auxiliary_work_completed"},
            {
                "kind": "detector",
                "condition_id": "contributor_within_auditory_range",
                "detector": "robot_near_human",
                "arguments": {"human_id": source, "distance": 2.5},
                "description": "The Robot is close enough to hear the result.",
            },
        ],
        "event": {
            "type": "information_disclosed",
            "description": "Deliver the local result.",
            "human_actions": [
                {
                    "human": source,
                    "command": "speak",
                    "message": "The preparation check is complete.",
                }
            ],
        },
        "completion": {
            "evaluator": "human_event_completed",
            "description": "Actual local result.",
            "ground_truth": {"milestone_id": "contribution_result_delivered"},
        },
    }
    nodes.insert(3, result)
    next(n for n in nodes if n["milestone_id"] == affected)["preconditions"].append(
        {"kind": "milestone", "milestone_ref": "contribution_result_delivered"}
    )
    return payload


@pytest.mark.parametrize("affected", ["onsite_target_observed", "task_completed"])
def test_collaboration_can_gate_onsite_or_final_outcome(affected):
    episode = TARSphereEpisode.model_validate(collaboration_payload(affected))
    assert sum(n.kind == "scored" for n in episode.evaluation.milestones) == 5
    if affected == "task_completed":
        onsite = next(
            n
            for n in episode.evaluation.milestones
            if n.milestone_id == "onsite_task_completed"
        )
        assert [p.milestone_ref for p in onsite.preconditions] == [
            "onsite_target_observed"
        ]


def test_collaboration_must_not_include_return_in_work_completion():
    payload = collaboration_payload()
    work = next(
        n
        for n in payload["evaluation"]["milestones"]
        if n["milestone_id"] == "auxiliary_work_completed"
    )
    work["event"]["human_actions"].append(
        {"human": "Marcus", "command": "goto", "target": [0.0, 0.0, 0.0]}
    )
    with pytest.raises(ValueError, match="return motion or result speech"):
        TARSphereEpisode.model_validate(payload)


def test_collaboration_requires_received_acceptance_not_just_selected_ask():
    payload = collaboration_payload()
    work = next(
        n
        for n in payload["evaluation"]["milestones"]
        if n["milestone_id"] == "auxiliary_work_completed"
    )
    work["preconditions"] = [
        {"kind": "milestone", "milestone_ref": "task_understanding"}
    ]
    with pytest.raises(ValueError, match="delivered request acceptance"):
        TARSphereEpisode.model_validate(payload)


def test_unrelated_factual_receipt_does_not_accept_auxiliary_work():
    payload = collaboration_payload()
    receipt = next(
        n
        for n in payload["evaluation"]["milestones"]
        if n["milestone_id"] == "required_information_received"
    )
    receipt["event"]["type"] = "interaction_opportunity_opened"
    with pytest.raises(ValueError, match="delivered request acceptance"):
        TARSphereEpisode.model_validate(payload)


def test_animation_speech_preserves_ongoing_body_motion():
    from tarsphere.infra.control.anim_people_executor import AnimPeopleBodyExecutor

    executor = object.__new__(AnimPeopleBodyExecutor)
    state = HumanBodyState(
        "Nora",
        [0.0, 0.0, 0.0],
        action_status="ongoing",
        active_action="goto",
        target_position=[4.0, 0.0, 0.0],
    )
    executor._states = {"Nora": state}
    executor._sync = lambda _: None
    result = executor.apply_action(
        "Nora", {"command": "speak", "message": "Yes, I am on my way."}
    )
    assert result.executed
    assert state.action_status == "ongoing"
    assert state.active_action == "goto"
    assert state.target_position == [4.0, 0.0, 0.0]


def test_kinematic_route_must_reach_endpoint_before_symbolic_work():
    from tarsphere.infra.control.kinematic_human_executor import KinematicHumanExecutor

    characters = SimpleNamespace(
        clear=lambda: None,
        spawn=lambda human: None,
        is_scene_bound=lambda name: False,
        set_pose=lambda *args: None,
    )
    body = KinematicHumanExecutor(
        characters=characters,
        pose_sync=lambda *args: None,
        route_resolver=lambda name, start, target, action: [[1.0, 0.0, 0.0], target],
    )
    body.reset(
        [SimpleNamespace(id="Nora", start_position=[0.0, 0.0, 0.0], start_yaw=0.0)]
    )
    target = [1.0, 1.0, 0.0]
    episode = SimpleNamespace(world_knowledge={"spatial_anchors": {"work": target}})
    action = {
        "command": "perform_task",
        "target_anchor": "work",
        "task_description": "Check the preparation record.",
    }
    body.apply_action("Nora", {"command": "goto", "target": target})
    body.step(1.99)
    assert not execute_human_task(episode, "Nora", action, body.get_state).executed
    body.apply_action("Nora", {"command": "speak", "message": "Still walking."})
    assert body.get_state("Nora").action_status == "ongoing"
    body.step(0.01)
    assert execute_human_task(episode, "Nora", action, body.get_state).executed
    assert body.position("Nora") == target
