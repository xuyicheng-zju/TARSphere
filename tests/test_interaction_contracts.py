from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tarsphere.humans.llm_policy import HumanDialoguePolicy, TopicSelection
from tarsphere.infra.episode.episode_runner import _execute_navigation_or_action
from tarsphere.infra.manipulation.handover import (
    HandoverHuman,
    REASON_TARGET_UNRESOLVED,
    evaluate_handover_claim,
)
from tarsphere.infra.simulation.isaac_env import IsaacAutonomousEnv


class _UnavailableHandoverEnvironment:
    _physics_steps = 120

    def __init__(self) -> None:
        self.communicability_queries = []

    @staticmethod
    def resolve_handover_human_id(recipient):
        return {
            "Nora": "Nora",
            "household coordinator": "Nora",
            "woman in a pale green top": "Nora",
        }.get(recipient)

    def can_communicate(self, human_id):
        self.communicability_queries.append(human_id)
        return human_id == "Nora"

    def step(self, decision):  # pragma: no cover - the disruption must intercept
        raise AssertionError(
            f"unavailable handover reached the environment: {decision}"
        )


class _ActiveHumanDisruption:
    def __init__(self) -> None:
        self.queries = []

    @staticmethod
    def component_unavailable(tool_name, physics_step):
        return False

    def unavailable_response(self, human_id, physics_step):
        self.queries.append((human_id, physics_step))
        return "One moment, I cannot take it yet."


@pytest.mark.parametrize(
    "recipient",
    ["Nora", "household coordinator", "woman in a pale green top"],
)
def test_handover_aliases_share_canonical_unavailability(recipient):
    environment = _UnavailableHandoverEnvironment()
    disruption = _ActiveHumanDisruption()

    _, done, info = _execute_navigation_or_action(
        environment,
        SimpleNamespace(),
        {
            "action": "handover_object",
            "call_name": "handover_object",
            "recipient": recipient,
        },
        {"frame": "current"},
        disruption,
    )

    assert done is False
    assert info["executed"] is False
    assert info["temporary_unavailability_human_id"] == "Nora"
    assert environment.communicability_queries == ["Nora"]
    assert disruption.queries == [("Nora", 120)]


def test_handover_resolver_rejects_unresolved_and_ambiguous_descriptions():
    humans = [
        HandoverHuman(
            id="Nora",
            role="coordinator",
            descriptions=("adult woman in a pale green top",),
        ),
        HandoverHuman(
            id="Elena",
            role="coordinator",
            descriptions=("adult woman in a blue top",),
        ),
    ]
    environment = SimpleNamespace(_handover_human_identities=lambda: humans)

    assert IsaacAutonomousEnv.resolve_handover_human_id(environment, "pale") == "Nora"
    assert (
        IsaacAutonomousEnv.resolve_handover_human_id(environment, "coordinator") is None
    )
    assert IsaacAutonomousEnv.resolve_handover_human_id(environment, "pilot") is None

    verdict = evaluate_handover_claim(
        target="",
        humans=humans,
        held_object_id="bottle",
        object_transport_ready=True,
        robot_xy=(0.0, 0.0),
        human_positions={"Nora": (1.0, 0.0), "Elena": (1.0, 0.0)},
        visible_human_ids={"Nora", "Elena"},
        accepted_handovers={},
    )
    assert verdict.accepted is False
    assert verdict.reason == REASON_TARGET_UNRESOLVED


def _dialogue_policy_with_selection(fact_ids, *, unsupported_text=None):
    policy = object.__new__(HumanDialoguePolicy)
    policy._npc_context = {}
    policy._infer = lambda instruction, payload, schema: SimpleNamespace(
        fact_ids=list(fact_ids),
        message=unsupported_text,
    )
    return policy


def test_supplementary_reply_uses_authored_fact_text_only():
    policy = _dialogue_policy_with_selection(
        ["entrance"],
        unsupported_text="The locked red door is beside the lift.",
    )

    answer = policy.answer(
        question="Where is the entrance?",
        facts=[
            {
                "fact_id": "entrance",
                "text": "The entrance is through the corridor doorway.",
            }
        ],
        history=[],
    )

    assert answer.fact_ids == ["entrance"]
    assert answer.message == "The entrance is through the corridor doorway."


def test_supplementary_fact_selection_rejects_unknown_and_duplicate_ids():
    facts = [{"fact_id": "entrance", "text": "Use the corridor doorway."}]

    with pytest.raises(ValueError, match="unauthorized fact"):
        _dialogue_policy_with_selection(["unknown"]).answer(
            question="Where?", facts=facts, history=[]
        )
    with pytest.raises(ValueError, match="more than once"):
        _dialogue_policy_with_selection(["entrance", "entrance"]).answer(
            question="Where?", facts=facts, history=[]
        )


def test_supplementary_fact_contract_handles_empty_and_duplicate_facts():
    answer = _dialogue_policy_with_selection([]).answer(
        question="Where?",
        facts=[{"fact_id": "entrance", "text": "Use the corridor doorway."}],
        history=[],
    )
    assert answer.message == ""
    assert answer.fact_ids == []

    with pytest.raises(ValueError, match="Duplicate NPC supplementary fact id"):
        _dialogue_policy_with_selection(["entrance"]).answer(
            question="Where?",
            facts=[
                {"fact_id": "entrance", "text": "Use the corridor doorway."},
                {"fact_id": "entrance", "text": "Use the side entrance."},
            ],
            history=[],
        )


def test_topic_router_cannot_supply_public_refusal_text():
    with pytest.raises(ValidationError):
        TopicSelection(
            intent_id=None,
            unsupported_reply="The office moved upstairs.",
        )
