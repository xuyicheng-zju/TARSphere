"""Compile one execution disruption over a frozen five-outcome Base task."""

from __future__ import annotations

from copy import deepcopy
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from tarsphere.bench.schema import TARSphereEpisode
from tarsphere.contracts.task_profile import (
    COMPONENT_OUTAGE_AFFECTED_OUTCOMES,
    SCORED_OUTCOME_IDS,
    require_supported_component_outage_tool,
)


class _DisruptionDesign(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    affected_outcome_id: Literal[
        "onsite_target_observed",
        "onsite_task_completed",
        "recipient_observed",
        "task_completed",
    ] = Field(description="Existing task outcome affected by this contingency.")
    rationale: str = Field(
        min_length=1,
        description="Scene-grounded reason this disruption belongs in the task.",
    )
    counterfactual: str = Field(
        min_length=1,
        description="Consequence of ignoring the disruption while pursuing the same task.",
    )


class SpatialBlockingDesign(_DisruptionDesign):
    subtype: Literal["spatial_blocking"]
    human_id: str = Field(
        min_length=1,
        description="Movable catalog Human whose body closes the required passage.",
    )
    asset: str = Field(
        min_length=1,
        description="Catalog character asset matching the authored Human identity.",
    )
    role: str = Field(
        min_length=1,
        description="Scene-natural reason for this Human to be at the passage.",
    )
    public_description: str = Field(
        min_length=1,
        description="Visible identity cue, without private route or score information.",
    )
    public_job: str = Field(
        min_length=1, description="Public social role of the blocker."
    )
    clear_xyz: tuple[float, float, float] = Field(
        description=(
            "Offline geometry witness and collision-safe initial spawn outside "
            "the passage. Runtime clearance selects a nearby live target with "
            "Robot-aware A* rather than replaying this coordinate."
        ),
    )
    block_xyz: tuple[float, float, float] = Field(
        description="World pose whose Human body closes every valid task route."
    )
    contact_xy: tuple[float, float] = Field(
        description="Reachable Robot stance on the approach side for a local passage request."
    )
    start_yaw: float = Field(
        description="Initial Human facing direction in world-frame radians."
    )
    passage_description: str = Field(
        min_length=1, description="Public physical passage occupied by this Human."
    )
    request_intent_id: str = Field(
        min_length=1,
        description="Authorized local question topic for asking this Human to clear.",
    )
    request_reply: str = Field(
        min_length=1,
        description="Literal agreement that causally releases the Human clearing action.",
    )


class TemporaryUnavailabilityDesign(_DisruptionDesign):
    subtype: Literal["temporary_unavailability"]
    human_id: str = Field(
        min_length=1,
        description="Existing Human whose required interaction temporarily fails.",
    )
    unavailable_response: str = Field(
        min_length=1,
        description="Scene-natural refusal delivered on an attempted interaction.",
    )


class ComponentOutageDesign(_DisruptionDesign):
    subtype: Literal["component_outage"]
    component_tool_name: str = Field(
        min_length=1,
        description=(
            "PointGoal, frontier, or grasp is temporarily unavailable but "
            "still visible. Navigation or direct EEF alternatives remain "
            "available; no other tool may be the outage target."
        ),
    )

    @model_validator(mode="after")
    def validate_navigation_outage(self) -> "ComponentOutageDesign":
        tool_name = require_supported_component_outage_tool(self.component_tool_name)
        expected_outcome = COMPONENT_OUTAGE_AFFECTED_OUTCOMES[tool_name]
        if self.affected_outcome_id != expected_outcome:
            raise ValueError(
                f"{tool_name} component outage must affect {expected_outcome}"
            )
        return self


DISRUPTION_DESIGN = TypeAdapter(
    Annotated[
        SpatialBlockingDesign | TemporaryUnavailabilityDesign | ComponentOutageDesign,
        Field(discriminator="subtype"),
    ]
)


def _ref(node_id: str) -> dict:
    return {"kind": "milestone", "milestone_ref": node_id}


def _event(
    node_id: str,
    predecessor: str,
    event_type: str,
    description: str,
    evaluator: str,
    truth: dict,
    actions: list[dict] | None = None,
) -> dict:
    return {
        "milestone_id": node_id,
        "kind": "event",
        "preconditions": [_ref(predecessor)],
        "event": {
            "type": event_type,
            "description": description,
            "human_actions": actions or [],
        },
        "completion": {
            "description": description,
            "evaluator": evaluator,
            "ground_truth": truth,
        },
    }


def _validate_component_outage(
    original: TARSphereEpisode,
    profile: object,
) -> None:
    if getattr(profile, "subtype", None) != "component_outage":
        return
    tool_name = require_supported_component_outage_tool(profile.component_tool_name)
    expected_outcome = COMPONENT_OUTAGE_AFFECTED_OUTCOMES[tool_name]
    if profile.affected_outcome_id != expected_outcome:
        raise ValueError(f"{tool_name} component outage must affect {expected_outcome}")
    if tool_name == "grasp_object":
        if original.evaluation.task_profile.mainline.value != "fetch_and_deliver":
            raise ValueError(
                "grasp_object component outage requires a fetch-and-deliver task"
            )
        return
    route = original.world_knowledge.get("episode_route") or {}
    outbound = route.get("navigate_to_target") or {}
    if not outbound.get("points"):
        raise ValueError(
            "Navigation component outage requires an authored outbound route"
        )


def compile_disruption_variant(base: dict, design: _DisruptionDesign) -> dict:
    """Keep the Base objective fixed while adding one real execution condition."""
    original = TARSphereEpisode.model_validate(base)
    if original.evaluation.task_profile.capability_slice.value != "base":
        raise ValueError("Disruption compilation requires a paired Base episode")
    _validate_component_outage(original, design)
    data = deepcopy(base)
    data["episode_id"] = (
        base["episode_id"].removesuffix("_base") + "_disruption_adaptation"
    )
    data["title"] = base["title"] + f" / {design.subtype.replace('_', ' ')}"
    data["evaluation"]["task_profile"] = {
        "mainline": original.evaluation.task_profile.mainline.value,
        "capability_slice": "disruption_adaptation",
        **design.model_dump(
            exclude={
                "rationale",
                "counterfactual",
                "asset",
                "role",
                "public_description",
                "public_job",
                "clear_xyz",
                "block_xyz",
                "contact_xy",
                "start_yaw",
                "passage_description",
                "request_intent_id",
                "request_reply",
            }
        ),
    }
    profile = data["evaluation"]["task_profile"]
    data["authoring"]["achieved_gates"] = ["authored", "schema_valid"]
    data["authoring"]["difficulty_factors"] = []
    data["metadata"]["paired_base_episode_id"] = base["episode_id"]
    data["tags"] = [tag for tag in base.get("tags", ()) if tag != "base"] + [
        "disruption_adaptation",
        design.subtype,
    ]
    if isinstance(design, SpatialBlockingDesign):
        profile.pop("human_id")
        if design.human_id in {human["id"] for human in data["humans"]}:
            raise ValueError("Spatial blocker must be a distinct authored Human")
        data["humans"].append(
            {
                "id": design.human_id,
                "role": design.role,
                "asset": design.asset,
                "start_position": list(design.clear_xyz),
                "start_yaw": design.start_yaw,
                "static": False,
                "persona": f"{design.human_id} responds only to the authorized local passage request.",
                "goals": ["Continue the scene activity and move aside when asked."],
                "public_description": design.public_description,
                "public_job": design.public_job,
            }
        )
        profile.update(
            blocking_human_id=design.human_id,
            blocking_milestone_id="passage_blocked",
            clearance_request_milestone_id="passage_request_accepted",
            clear_milestone_id="passage_cleared",
            passage_request_intent_id=design.request_intent_id,
        )
        data.setdefault("interaction", {}).setdefault("intents", []).append(
            {
                "intent_id": design.request_intent_id,
                "description": f"Ask {design.human_id} to make room at {design.passage_description}.",
                "public": True,
                "eligible_human_ids": [design.human_id],
                "preconditions": [_ref("passage_blocked")],
                "answer": {"kind": "authored_text", "text": design.request_reply},
            }
        )
        nodes = data["evaluation"]["milestones"]
        nodes[1:1] = [
            _event(
                "passage_blocked",
                "task_understanding",
                "passage_blocked_by_human",
                f"{design.human_id} occupies {design.passage_description}.",
                "human_event_completed",
                {"milestone_id": "passage_blocked"},
                [
                    {
                        "human": design.human_id,
                        "command": "goto",
                        "target": list(design.block_xyz),
                    }
                ],
            ),
            _event(
                "passage_request_accepted",
                "passage_blocked",
                "interaction_opportunity_opened",
                f"{design.human_id} answers a locally delivered request to pass.",
                "information_received",
                {"human_id": design.human_id, "intent_id": design.request_intent_id},
            ),
            _event(
                "passage_cleared",
                "passage_request_accepted",
                "passage_cleared_by_human",
                f"{design.human_id} moves clear of {design.passage_description}.",
                "human_event_completed",
                {"milestone_id": "passage_cleared"},
                [
                    {
                        "human": design.human_id,
                        "command": "goto",
                        "target": list(design.clear_xyz),
                        "route_planner": "live_robot_aware_astar",
                    }
                ],
            ),
        ]
        # The blocker is physically staged before the first policy frame.  Its
        # public event therefore shares the opening world preconditions rather
        # than waiting for the Agent's plan reaction; otherwise a decision-zero
        # navigation macro could cross the choke before the Human starts.
        nodes[1]["preconditions"] = deepcopy(nodes[0]["preconditions"])
        # Blocking is an observable contingency, not a procedural receipt
        # lock.  Keep block -> accepted request -> clear as one side branch in
        # the episode DAG, but do not make any canonical score depend on it.
        # An Agent may ask the blocker to move; if it safely finds another
        # physical route, the ordinary task evidence remains scoreable.
        route = data["world_knowledge"]["episode_route"]
        route["navigate_to_target"]["points"].insert(
            1,
            {
                "point_id": "passage_contact",
                "position_xy": list(design.contact_xy),
            },
        )
        stop_map = {
            stop["milestone_id"]: stop["route_point_id"]
            for stop in route["milestone_stops"]
        }
        stop_map.update(
            passage_blocked="task_start",
            passage_request_accepted="passage_contact",
            passage_cleared="passage_contact",
        )
        route["milestone_stops"] = [
            {
                "milestone_id": node["milestone_id"],
                "route_point_id": stop_map[node["milestone_id"]],
            }
            for node in nodes
        ]
    validate_disruption_pair(base, data)
    return data


def validate_disruption_pair(base: dict, variant: dict) -> None:
    """Reject task drift and additional scored outcomes in the pilot pair."""
    original = TARSphereEpisode.model_validate(base)
    candidate = TARSphereEpisode.model_validate(variant)
    if (
        candidate.evaluation.task_profile.capability_slice.value
        != "disruption_adaptation"
    ):
        raise ValueError("Candidate is not a Disruption Adaptation episode")
    for field in ("scene", "robot", "asset_adaptation"):
        if base.get(field) != variant.get(field):
            raise ValueError(f"Disruption changed paired Base {field}")
    for field in ("objects", "physical_task"):
        if base["world_knowledge"].get(field) != variant["world_knowledge"].get(field):
            raise ValueError(f"Disruption changed paired Base {field}")
    original_scored = [
        node for node in original.evaluation.milestones if node.kind == "scored"
    ]
    candidate_scored = [
        node for node in candidate.evaluation.milestones if node.kind == "scored"
    ]
    if (
        tuple(node.milestone_id for node in original_scored) != SCORED_OUTCOME_IDS
        or tuple(node.milestone_id for node in candidate_scored) != SCORED_OUTCOME_IDS
    ):
        raise ValueError("Disruption must preserve exactly the five Base scored IDs")
    for before, after in zip(original_scored, candidate_scored):
        if (
            before.outcome.evaluator != after.outcome.evaluator
            or before.outcome.ground_truth != after.outcome.ground_truth
        ):
            raise ValueError(f"Disruption changed task score: {before.milestone_id}")
    if candidate.authoring.difficulty_factors:
        raise ValueError("Disruption must inherit Base difficulty")


__all__ = [
    "DISRUPTION_DESIGN",
    "ComponentOutageDesign",
    "SpatialBlockingDesign",
    "TemporaryUnavailabilityDesign",
    "compile_disruption_variant",
    "validate_disruption_pair",
]
