"""Compile task update variants with speech-delivered outcome revisions.

The opening commitment is a decoy; the delivered revision converges to Base GT.
"""

from __future__ import annotations

from copy import deepcopy
from math import dist
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.bench.asset_adaptation import (
    EpisodeAssetAdaptation,
    EpisodeAssetInstance,
)
from tarsphere.bench.authoring import CandidateEpisodeRoute
from tarsphere.bench.certification.certified_asset_pool import (
    certified_asset,
    load_certified_asset_pool,
)
from tarsphere.bench.schema import HumanSpec, InteractionIntentSpec, TARSphereEpisode
from tarsphere.contracts.human_interaction import HUMAN_COMMUNICATION_RADIUS_M
from tarsphere.evaluation.task_bindings import final_outcome_ground_truth


class TaskUpdateDesign(BaseModel):
    """Authored decoy commitment and delivered revision to paired Base GT."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    subtype: Literal["task_content_update", "delivery_requirement_update"] = Field(
        description="Content updates change what/where/who at the onsite phase; delivery updates change the final recipient."
    )

    obsolete_commitment: str = Field(
        min_length=1,
        description="Opening decoy commitment that will be superseded by the revision.",
    )

    revised_commitment: str = Field(
        min_length=1,
        description="Authoritative commitment after revision; must match Base final GT.",
    )

    revision_speaker: str = Field(
        min_length=1,
        description="Declared Human who physically delivers the revision within auditory range.",
    )

    revision_message: str = Field(
        min_length=1,
        description="Actual revision speech delivered after the decoy investment.",
    )

    initial_request: str = Field(
        min_length=1,
        description="Complete opening request, including truthful directions to the initial task locus.",
    )
    initial_plan_goal: str = Field(
        min_length=1,
        description="Private plan rubric for the initial commitment only; it must not anticipate the revision.",
    )
    initial_plan_subgoals: tuple[str, ...] = Field(
        min_length=1,
        description="Ordered initial-task subgoals, before any revision is known to the Agent.",
    )
    initial_observation_gt: dict[str, Any] | None = Field(
        default=None,
        description="Initial target/source observation GT for a content revision.",
    )
    initial_onsite_gt: dict[str, Any] | None = Field(
        default=None, description="Initial onsite completion GT for a content revision."
    )
    initial_recipient_gt: dict[str, Any] | None = Field(
        default=None,
        description="Initial recipient observation GT for a delivery revision.",
    )
    initial_final_gt: dict[str, Any] = Field(
        description="Initial final-outcome GT, replaced only when the revision speech is delivered."
    )
    episode_route: CandidateEpisodeRoute = Field(
        description="Fresh full-route evidence through the decoy and Base final objective."
    )
    initial_task_reference_xy: tuple[tuple[float, float], ...] = Field(
        min_length=2,
        description="Authoring-BEV waypoints for the initial intended task route, never Agent input.",
    )
    revised_task_reference_xy: tuple[tuple[float, float], ...] = Field(
        min_length=2,
        description="Authoring-BEV waypoints for the revised intended task route, never Agent input.",
    )
    rendezvous_xyz: tuple[float, float, float] | None = Field(
        default=None,
        description="Reviewed reachable Human meeting pose before revision speech; omitted only if already in range.",
    )
    speaker_return_xyz: tuple[float, float, float] | None = Field(
        default=None,
        description="Optional post-speech return pose; it may gate recipient discovery but not onsite execution.",
    )
    added_humans: tuple[HumanSpec, ...] = Field(
        default=(),
        description="Fully authored extra Humans, each with a distinct identity and verified physical placement.",
    )
    added_intents: tuple[InteractionIntentSpec, ...] = Field(
        default=(),
        description="Authored initial inquiry topics, distinct from Base topics and delivered only by eligible Humans.",
    )
    added_world_objects: tuple[dict[str, Any], ...] = Field(
        default=(),
        description="Exact metadata for existing evidenced scene objects not already declared by Base.",
    )
    added_asset_instances: tuple[EpisodeAssetInstance, ...] = Field(
        default=(),
        description="Certified same-category clarification objects introduced for an initial fetch target revision.",
    )
    added_spatial_anchors: dict[str, tuple[float, float]] = Field(
        default_factory=dict,
        description="Reviewed additional anchor coordinates; Base anchor values cannot be modified.",
    )
    delivery_change: Literal["recipient", "destination"] | None = Field(
        default=None,
        description="For delivery revisions, distinguish a new person from a new physical endpoint for the same person.",
    )
    destination_human_id: str | None = Field(
        default=None,
        description="Same named final recipient for a destination revision; this Human must actually move.",
    )
    previous_destination_anchor_id: str | None = Field(
        default=None,
        description="Initially authoritative delivery location anchor for a destination revision.",
    )
    revised_destination_anchor_id: str | None = Field(
        default=None,
        description="New physically distinct delivery location anchor reached by the named recipient.",
    )

    @model_validator(mode="after")
    def validate_subtype_fields(self):
        named_facing = re.search(
            r"\bfacing\s+(?!me\b)[A-Z][\w-]*", self.initial_request
        )
        if named_facing:
            raise ValueError(
                f"Initial request refers to its speaker in the third person: "
                f"{named_facing.group(0)!r}"
            )
        if "ask me" in self.revision_message.casefold():
            raise ValueError(
                "A revision speaker must state the changed need naturally; "
                "it cannot tell the Agent to ask the same speaker again"
            )
        content = self.subtype == "task_content_update"
        delivery = self.subtype == "delivery_requirement_update"

        if content:
            if self.initial_observation_gt is None or self.initial_onsite_gt is None:
                raise ValueError(
                    "Content revision requires initial observation and onsite GT"
                )
            if self.initial_recipient_gt is not None or any(
                (
                    self.delivery_change,
                    self.destination_human_id,
                    self.previous_destination_anchor_id,
                    self.revised_destination_anchor_id,
                )
            ):
                raise ValueError("Content revision cannot declare delivery bindings")
        if delivery:
            if self.initial_recipient_gt is None or self.delivery_change is None:
                raise ValueError(
                    "Delivery revision requires initial recipient GT and delivery_change"
                )
            if (
                self.initial_observation_gt is not None
                or self.initial_onsite_gt is not None
            ):
                raise ValueError("Delivery revision cannot change the onsite phase")
            if self.delivery_change == "destination" and not all(
                (
                    self.destination_human_id,
                    self.previous_destination_anchor_id,
                    self.revised_destination_anchor_id,
                )
            ):
                raise ValueError(
                    "Destination revision requires named Human and both location anchors"
                )
            if self.delivery_change == "recipient" and any(
                (
                    self.destination_human_id,
                    self.previous_destination_anchor_id,
                    self.revised_destination_anchor_id,
                )
            ):
                raise ValueError(
                    "Recipient revision cannot declare destination bindings"
                )
            if self.initial_final_gt.get("human_id") != self.initial_recipient_gt.get(
                "human_id"
            ):
                raise ValueError(
                    "Initial final recipient must equal initial recipient observation"
                )

        if self.initial_task_reference_xy == self.revised_task_reference_xy:
            raise ValueError("Initial and revised reference routes must differ")

        return self


def _ref(node_id: str) -> dict:
    return {"kind": "milestone", "milestone_ref": node_id}


def _binding_ref(observation: dict[str, Any], onsite: dict[str, Any]) -> str:
    if "object_ids" in observation:
        return ",".join(str(value) for value in observation["object_ids"])
    if "human_id" in observation:
        return ":".join(
            (str(observation["human_id"]), str(onsite.get("intent_id", "")))
        ).rstrip(":")
    raise ValueError("Task observation has no typed object or Human reference")


def _auditory_gate(human_id: str) -> dict[str, Any]:
    return {
        "kind": "detector",
        "condition_id": "revision_speaker_in_auditory_range",
        "detector": "robot_near_human",
        "arguments": {"human_id": human_id, "distance": HUMAN_COMMUNICATION_RADIUS_M},
        "description": "The revision speaker is physically within auditory range.",
    }


def compile_task_update_variant(base: dict, design: TaskUpdateDesign) -> dict:
    """Compile one candidate with five unchanged score roles and delivered GT updates."""

    parsed_base = TARSphereEpisode.model_validate(base)
    if parsed_base.evaluation.task_profile.capability_slice.value != "base":
        raise ValueError("TA compilation requires a paired Base episode")

    mainline = parsed_base.evaluation.task_profile.mainline.value
    data = deepcopy(base)
    onsite = next(
        n
        for n in base["evaluation"]["milestones"]
        if n["milestone_id"] == "onsite_task_completed"
    )["reaction"]
    if (
        design.subtype == "task_content_update"
        and onsite["evaluator"] == "inspection_submitted"
    ):
        criterion = onsite["ground_truth"]["criterion"].strip()
        initial_text = " ".join(
            (
                design.initial_request,
                design.initial_plan_goal,
                *design.initial_plan_subgoals,
            )
        )
        if criterion.casefold() in initial_text.casefold():
            raise ValueError("TA initial task leaks the revised inspection criterion")

    if design.added_asset_instances:
        if mainline != "fetch_and_deliver" or design.subtype != "task_content_update":
            raise ValueError(
                "Additional pickables are allowed only for fetch content revisions"
            )
        initial_object_id = str((design.initial_onsite_gt or {}).get("object_id") or "")
        if (
            len(design.added_asset_instances) != 1
            or design.added_asset_instances[0].object_id != initial_object_id
            or design.initial_observation_gt
            != {
                "object_ids": [initial_object_id],
                "scene_anchor_id": "initial_task_observation",
            }
            or design.initial_final_gt.get("object_id") != initial_object_id
        ):
            raise ValueError(
                "Fetch content revision must initially bind one added certified object"
            )
        adaptation = data["world_knowledge"]["asset_adaptation"]
        target = next(
            item for item in adaptation["instances"] if item["role"] == "target"
        )
        certified_pool = load_certified_asset_pool()
        for instance in design.added_asset_instances:
            certified = certified_asset(certified_pool, instance.asset_key)
            if (
                instance.role != "clarification"
                or instance.category != target["category"]
                or instance.asset_path != certified["usd_path"]
                or instance.category != certified["category"]
                or list(instance.size_xyz_m) != certified["size_xyz_m"]
                or instance.asset_meters_per_unit != certified["meters_per_unit"]
            ):
                raise ValueError(
                    "TA clarification instance differs from its certified pool entry"
                )
            adaptation["instances"].append(instance.model_dump(mode="json"))
            data["world_knowledge"]["physical_task"]["distractor_object_ids"].append(
                instance.object_id
            )

    data["episode_id"] = base["episode_id"].removesuffix("_base") + "_task_update"
    data["title"] = base["title"] + " / task update"
    data["description"] = f"{design.obsolete_commitment} → {design.revised_commitment}"

    known_humans = {human["id"] for human in base["humans"]}
    for human in design.added_humans:
        if human.id in known_humans:
            raise ValueError(f"Extra Human duplicates a Base identity: {human.id}")
        data["humans"].append(
            human.model_dump(
                mode="json",
                exclude_none=True,
                exclude={"name", "character_name"},
            )
        )
        known_humans.add(human.id)
    if design.revision_speaker not in known_humans:
        raise ValueError(f"Unknown revision speaker: {design.revision_speaker}")
    known_intents = {intent["intent_id"] for intent in data["interaction"]["intents"]}
    for intent in design.added_intents:
        if intent.intent_id in known_intents:
            raise ValueError(
                f"Extra intent duplicates a Base topic: {intent.intent_id}"
            )
        data["interaction"]["intents"].append(
            intent.model_dump(mode="json", exclude_none=True)
        )
        known_intents.add(intent.intent_id)
    known_objects = {item["object_id"] for item in base["world_knowledge"]["objects"]}
    for item in design.added_world_objects:
        object_id = str(item.get("object_id") or "")
        if not object_id or object_id in known_objects:
            raise ValueError(
                f"Extra world object has missing or duplicate ID: {object_id!r}"
            )
        data["world_knowledge"]["objects"].append(deepcopy(item))
        known_objects.add(object_id)
    for anchor_id, xy in design.added_spatial_anchors.items():
        if anchor_id in data["world_knowledge"]["spatial_anchors"]:
            raise ValueError(f"Extra spatial anchor duplicates Base: {anchor_id}")
        data["world_knowledge"]["spatial_anchors"][anchor_id] = list(xy)

    nodes = data["evaluation"]["milestones"]
    by_id = {node["milestone_id"]: node for node in nodes}

    opening = by_id["task_understanding"]["event"]["human_actions"][0]
    opening["message"] = design.initial_request
    plan = by_id["task_understanding"]["reaction"]["ground_truth"]
    plan["goal"] = design.initial_plan_goal
    plan["required_subgoals"] = list(design.initial_plan_subgoals)

    base_observation = deepcopy(
        by_id["onsite_target_observed"]["reaction"]["ground_truth"]
    )
    base_onsite = deepcopy(by_id["onsite_task_completed"]["reaction"]["ground_truth"])
    base_recipient = deepcopy(by_id["recipient_observed"]["reaction"]["ground_truth"])
    base_final = deepcopy(by_id["task_completed"]["reaction"]["ground_truth"])
    if (
        design.subtype == "task_content_update"
        and by_id["onsite_task_completed"]["reaction"]["evaluator"]
        == "information_received"
    ):
        revised_intent_id = str(base_onsite["intent_id"])
        revised_intent = next(
            (
                intent
                for intent in data["interaction"]["intents"]
                if intent["intent_id"] == revised_intent_id
            ),
            None,
        )
        if revised_intent is None:
            raise ValueError(f"Missing revised inquiry intent: {revised_intent_id}")
        revised_intent["public"] = False
        revised_intent.setdefault("preconditions", []).append(
            _ref("task_change_delivered")
        )
    data["world_knowledge"]["episode_route"] = design.episode_route.model_dump(
        mode="json"
    )
    data["world_knowledge"]["task_update_route_comparison"] = {
        "initial_task_reference_xy": [
            list(point) for point in design.initial_task_reference_xy
        ],
        "revised_task_reference_xy": [
            list(point) for point in design.revised_task_reference_xy
        ],
    }

    if design.subtype == "task_content_update":
        trigger_after = "onsite_target_observed"
        event_type = "target_revised"
        assert design.initial_observation_gt is not None
        assert design.initial_onsite_gt is not None
        by_id["onsite_target_observed"]["reaction"]["ground_truth"] = deepcopy(
            design.initial_observation_gt
        )
        by_id["onsite_task_completed"]["reaction"]["ground_truth"] = deepcopy(
            design.initial_onsite_gt
        )
        by_id["task_completed"]["reaction"]["ground_truth"] = deepcopy(
            design.initial_final_gt
        )
        outcome_updates = {
            "onsite_task_completed": base_onsite,
            "task_completed": base_final,
        }
        previous_value_ref = _binding_ref(
            design.initial_observation_gt, design.initial_onsite_gt
        )
        new_value_ref = _binding_ref(base_observation, base_onsite)
        by_id["onsite_target_observed"]["event"]["description"] = (
            "The initially requested onsite target can now be observed."
        )
        by_id["onsite_target_observed"]["reaction"]["description"] = (
            "Observe the initially requested onsite target."
        )
    else:
        trigger_after = "onsite_task_completed"
        assert design.initial_recipient_gt is not None
        by_id["recipient_observed"]["reaction"]["ground_truth"] = deepcopy(
            design.initial_recipient_gt
        )
        by_id["task_completed"]["reaction"]["ground_truth"] = deepcopy(
            design.initial_final_gt
        )
        outcome_updates = {
            "recipient_observed": base_recipient,
            "task_completed": base_final,
        }
        if design.delivery_change == "recipient":
            event_type = "recipient_reassigned"
            previous_value_ref = str(design.initial_recipient_gt["human_id"])
            new_value_ref = str(base_recipient["human_id"])
            if previous_value_ref == new_value_ref:
                raise ValueError("Recipient revision must name two different Humans")
        else:
            event_type = "destination_revised"
            previous_value_ref = str(design.previous_destination_anchor_id)
            new_value_ref = str(design.revised_destination_anchor_id)
            anchors = data["world_knowledge"]["spatial_anchors"]
            if previous_value_ref not in anchors or new_value_ref not in anchors:
                raise ValueError(
                    "Destination revision references an undeclared spatial anchor"
                )
            if dist(anchors[previous_value_ref], anchors[new_value_ref]) <= 0.002:
                raise ValueError(
                    "Destination revision requires spatially distinct endpoints"
                )
            if (
                design.destination_human_id != base_recipient["human_id"]
                or design.initial_recipient_gt["human_id"] != base_recipient["human_id"]
            ):
                raise ValueError(
                    "Destination revision must keep the same Base recipient"
                )
            if (
                design.revision_speaker != design.destination_human_id
                or design.speaker_return_xyz is None
            ):
                raise ValueError(
                    "Destination recipient must announce and physically reach the new endpoint"
                )
            if dist(design.speaker_return_xyz[:2], anchors[new_value_ref]) > 0.5:
                raise ValueError(
                    "Recipient return pose must match the revised destination anchor"
                )
            base_human = next(
                human
                for human in base["humans"]
                if human["id"] == design.destination_human_id
            )
            if dist(design.speaker_return_xyz, base_human["start_position"]) > 0.002:
                raise ValueError(
                    "Revised recipient location must return to the paired Base endpoint"
                )

    # The binding, not Base prose, is authoritative. Neutral descriptions stay
    # truthful both before and after the delivered revision and cannot leak the
    # future answer through review or diagnostic surfaces.
    affected_ids = (
        ("onsite_task_completed", "task_completed")
        if design.subtype == "task_content_update"
        else ("recipient_observed", "task_completed")
    )
    for outcome_id in affected_ids:
        node = by_id[outcome_id]
        node["event"]["description"] = (
            f"The {outcome_id} phase uses the currently active task binding."
        )
        node["reaction"]["description"] = (
            f"Evaluate {outcome_id} against the binding active at this evidence step."
        )

    speaker = next(
        human for human in data["humans"] if human["id"] == design.revision_speaker
    )
    if design.rendezvous_xyz is not None or design.speaker_return_xyz is not None:
        if speaker.get("scene_person_prim_path"):
            raise ValueError(
                "A scene-native static Human cannot move to deliver a revision"
            )
        speaker["static"] = False
    arrival_event = None
    if design.rendezvous_xyz is not None:
        arrival_event = {
            "milestone_id": "revision_speaker_arrived",
            "kind": "event",
            "preconditions": [_ref("onsite_target_observed")],
            "event": {
                "type": "interaction_endpoint_relocated",
                "description": "The revision speaker moves toward a reachable meeting place.",
                "human_actions": [
                    {
                        "human": design.revision_speaker,
                        "command": "goto",
                        "target": list(design.rendezvous_xyz),
                    }
                ],
                "task_updates": [],
            },
            "completion": {
                "description": "The revision speaker physically reaches the meeting place.",
                "evaluator": "human_event_completed",
                "ground_truth": {"milestone_id": "revision_speaker_arrived"},
            },
        }
    human_actions = [
        {
            "human": design.revision_speaker,
            "command": "speak",
            "message": design.revision_message,
        }
    ]

    revision_preconditions = [_ref(trigger_after)]
    if arrival_event is not None:
        revision_preconditions.append(_ref("revision_speaker_arrived"))
    revision_preconditions.append(_auditory_gate(design.revision_speaker))
    revision_event = {
        "milestone_id": "task_change_delivered",
        "kind": "event",
        "preconditions": revision_preconditions,
        "event": {
            "type": event_type,
            "description": "Human physically delivers the authoritative task revision.",
            "human_actions": human_actions,
            "task_updates": [
                {
                    "field": (
                        "target"
                        if design.subtype == "task_content_update"
                        else "recipient"
                        if design.delivery_change == "recipient"
                        else "delivery_destination"
                    ),
                    "previous_value_ref": previous_value_ref,
                    "new_value_ref": new_value_ref,
                    "source_human_id": design.revision_speaker,
                    "delivery": "speech",
                    "outcome_updates": outcome_updates,
                }
            ],
        },
        "completion": {
            "description": "Revision speech actually reaches the Robot.",
            "evaluator": "human_event_completed",
            "ground_truth": {"milestone_id": "task_change_delivered"},
        },
    }

    if arrival_event is not None:
        observed_idx = next(
            i
            for i, n in enumerate(nodes)
            if n["milestone_id"] == "onsite_target_observed"
        )
        nodes.insert(observed_idx + 1, arrival_event)
    trigger_idx = next(
        i for i, n in enumerate(nodes) if n["milestone_id"] == trigger_after
    )
    if arrival_event is not None:
        arrival_idx = next(
            i
            for i, n in enumerate(nodes)
            if n["milestone_id"] == "revision_speaker_arrived"
        )
        trigger_idx = max(trigger_idx, arrival_idx)
    nodes.insert(trigger_idx + 1, revision_event)
    insert_idx = trigger_idx + 2

    if (
        design.subtype == "task_content_update"
        and design.initial_observation_gt != base_observation
    ):
        revised_observed = {
            "milestone_id": "revised_target_observed",
            "kind": "event",
            "preconditions": [_ref("task_change_delivered")],
            "event": {
                "type": "task_phase_opened",
                "description": "The revised target must be observed before onsite execution.",
                "human_actions": [],
                "task_updates": [],
            },
            "completion": {
                "description": "Observe the revised target in actual RGB-D evidence.",
                "evaluator": by_id["onsite_target_observed"]["reaction"]["evaluator"],
                "ground_truth": base_observation,
            },
        }
        nodes.insert(insert_idx, revised_observed)
        insert_idx += 1
        by_id["onsite_task_completed"]["preconditions"].append(
            _ref("revised_target_observed")
        )

    # Add task_change_delivered as precondition to affected outcomes
    if design.subtype == "task_content_update":
        by_id["onsite_task_completed"]["preconditions"].append(
            _ref("task_change_delivered")
        )
        by_id["task_completed"]["preconditions"].append(_ref("task_change_delivered"))
    else:
        by_id["recipient_observed"]["preconditions"].append(
            _ref("task_change_delivered")
        )
        by_id["task_completed"]["preconditions"].append(_ref("task_change_delivered"))

    if design.speaker_return_xyz is not None:
        return_event = {
            "milestone_id": "revision_speaker_relocated",
            "kind": "event",
            "preconditions": [_ref("task_change_delivered")],
            "event": {
                "type": "interaction_endpoint_relocated",
                "description": "The revision speaker moves to the later interaction endpoint.",
                "human_actions": [
                    {
                        "human": design.revision_speaker,
                        "command": "goto",
                        "target": list(design.speaker_return_xyz),
                    }
                ],
                "task_updates": [],
            },
            "completion": {
                "description": "The Human reaches the later interaction endpoint.",
                "evaluator": "human_event_completed",
                "ground_truth": {"milestone_id": "revision_speaker_relocated"},
            },
        }
        nodes.insert(insert_idx, return_event)
        if base_recipient["human_id"] == design.revision_speaker:
            by_id["recipient_observed"]["preconditions"].append(
                _ref("revision_speaker_relocated")
            )

    # Update task_profile
    data["evaluation"]["task_profile"] = {
        "mainline": mainline,
        "capability_slice": "task_update",
        "subtype": design.subtype,
        "revision_milestone_id": "task_change_delivered",
        "obsolete_commitment": design.obsolete_commitment,
        "revised_commitment": design.revised_commitment,
    }

    data["metadata"].update(
        paired_base_episode_id=base["episode_id"],
        task_source="paired_task_update_variants",
    )
    data["authoring"].update(
        achieved_gates=["authored", "schema_valid"],
        external_evidence={},
        difficulty_factors=[],
    )
    data["tags"] = [tag for tag in data.get("tags", []) if tag != "base"] + [
        "task_update"
    ]

    # Validate pairing
    validate_task_update_pair(base, data)

    return data


def validate_task_update_pair(base: dict, variant: dict) -> None:
    """Reject altered Base facts, extra scores, and stale task bindings."""

    parsed = TARSphereEpisode.model_validate(variant)
    for key in ("scene", "robot", "runtime"):
        if base.get(key) != variant.get(key):
            raise ValueError(f"TA changed paired Base {key}")

    base_world = base["world_knowledge"]
    variant_world = variant["world_knowledge"]
    if base_world.get("asset_adaptation") is None:
        if variant_world.get("asset_adaptation") is not None:
            raise ValueError("TA added asset adaptation to a non-fetch Base")
    else:
        original = base_world["asset_adaptation"]
        current = variant_world["asset_adaptation"]
        EpisodeAssetAdaptation.model_validate(current)
        if {key: value for key, value in current.items() if key != "instances"} != {
            key: value for key, value in original.items() if key != "instances"
        } or current["instances"][: len(original["instances"])] != original[
            "instances"
        ]:
            raise ValueError("TA changed paired Base asset adaptation")
        added_instances = current["instances"][len(original["instances"]) :]
        is_fetch_content = (
            variant["evaluation"]["task_profile"]["mainline"] == "fetch_and_deliver"
            and variant["evaluation"]["task_profile"]["subtype"]
            == "task_content_update"
        )
        if len(added_instances) != (1 if is_fetch_content else 0):
            raise ValueError(
                "Only fetch content revision may add one clarification asset"
            )
        if any(instance["role"] != "clarification" for instance in added_instances):
            raise ValueError("TA may add only clarification asset instances")
        new_objects = {
            item["object_id"]: item
            for item in variant_world["objects"]
            if item["object_id"]
            not in {base_item["object_id"] for base_item in base_world["objects"]}
        }
        for instance in added_instances:
            obj = new_objects.get(instance["object_id"])
            if (
                obj is None
                or obj.get("runtime_prim") != instance["runtime_prim"]
                or obj.get("category") != instance["category"]
                or obj.get("position_xyz") != instance["position_xyz_m"]
                or obj.get("interaction") != "pickable"
            ):
                raise ValueError(
                    "TA clarification world object differs from injected asset"
                )
            lower = obj.get("bounds_min_xyz") or ()
            upper = obj.get("bounds_max_xyz") or ()
            if (
                len(lower) != 3
                or len(upper) != 3
                or any(
                    abs((upper[i] - lower[i]) - instance["size_xyz_m"][i]) > 1e-6
                    for i in range(3)
                )
            ):
                raise ValueError(
                    "TA clarification world bounds differ from injected asset"
                )
        original_physical = base_world["physical_task"]
        current_physical = variant_world["physical_task"]
        if {
            key: value
            for key, value in current_physical.items()
            if key != "distractor_object_ids"
        } != {
            key: value
            for key, value in original_physical.items()
            if key != "distractor_object_ids"
        }:
            raise ValueError("TA changed paired Base physical task")
        if current_physical["distractor_object_ids"] != [
            *original_physical["distractor_object_ids"],
            *(instance["object_id"] for instance in added_instances),
        ]:
            raise ValueError(
                "TA physical distractors do not match added clarification assets"
            )

    base_humans = {human["id"]: human for human in base["humans"]}
    variant_humans = {human["id"]: human for human in variant["humans"]}
    if len(variant_humans) != len(variant["humans"]):
        raise ValueError("TA Human identities must be unique")
    movement_ids = {
        str(action["human"])
        for node in variant["evaluation"]["milestones"]
        for action in node["event"]["human_actions"]
        if action.get("command") == "goto"
    }
    for human_id, original in base_humans.items():
        current = variant_humans.get(human_id)
        if current is None:
            raise ValueError(f"TA removed Base Human {human_id}")
        expected = deepcopy(original)
        if human_id in movement_ids and not expected.get("scene_person_prim_path"):
            expected["static"] = False
        if current != expected:
            raise ValueError(
                f"TA changed Base Human {human_id} beyond authored mobility"
            )
    if len(variant_humans) - len(base_humans) > 1:
        raise ValueError("TA may add at most one necessary Human")

    for key, value in base["world_knowledge"].items():
        if key == "episode_route":
            route = CandidateEpisodeRoute.model_validate(
                variant["world_knowledge"][key]
            )
            start = route.navigate_to_target.points[0].position_xy
            if dist(start, base["robot"]["start_position"][:2]) > 0.002:
                raise ValueError("TA route must start at the paired Base Robot spawn")
            if route.occupancy_sha256 != value["occupancy_sha256"]:
                raise ValueError("TA route must use the paired Base occupancy")
        elif key == "objects":
            original = {item["object_id"]: item for item in value}
            current = {
                item["object_id"]: item for item in variant["world_knowledge"][key]
            }
            if len(current) != len(variant["world_knowledge"][key]):
                raise ValueError("TA world object IDs must be unique")
            if any(
                current.get(object_id) != item for object_id, item in original.items()
            ):
                raise ValueError("TA changed or removed a Base world object")
        elif key == "spatial_anchors":
            current = variant["world_knowledge"][key]
            if any(current.get(anchor_id) != xy for anchor_id, xy in value.items()):
                raise ValueError("TA changed or removed a Base spatial anchor")
        elif key in {"asset_adaptation", "physical_task"}:
            continue
        elif variant["world_knowledge"].get(key) != value:
            raise ValueError(f"TA changed paired Base world field {key}")

    base_intents = {item["intent_id"]: item for item in base["interaction"]["intents"]}
    variant_intents = {
        item["intent_id"]: item for item in variant["interaction"]["intents"]
    }
    if len(variant_intents) != len(variant["interaction"]["intents"]):
        raise ValueError("TA interaction intent IDs must be unique")
    revised_inquiry_id = None
    if variant["evaluation"]["task_profile"]["subtype"] == "task_content_update":
        onsite = next(
            node
            for node in base["evaluation"]["milestones"]
            if node["milestone_id"] == "onsite_task_completed"
        )
        if onsite["reaction"]["evaluator"] == "information_received":
            revised_inquiry_id = str(onsite["reaction"]["ground_truth"]["intent_id"])
    for intent_id, original in base_intents.items():
        expected = deepcopy(original)
        if intent_id == revised_inquiry_id:
            expected["public"] = False
            expected.setdefault("preconditions", []).append(
                _ref("task_change_delivered")
            )
        if variant_intents.get(intent_id) != expected:
            raise ValueError(
                "TA changed or removed a Base interaction intent outside its revision gate"
            )

    old = base["evaluation"]["milestones"]
    comparison = variant["world_knowledge"].get("task_update_route_comparison")
    if not isinstance(comparison, dict) or set(comparison) != {
        "initial_task_reference_xy",
        "revised_task_reference_xy",
    }:
        raise ValueError(
            "TA requires initial and revised authoring-BEV reference routes"
        )
    for name, points in comparison.items():
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError(f"TA {name} needs at least two authored waypoints")
        if dist(points[0], base["robot"]["start_position"][:2]) > 0.002:
            raise ValueError(f"TA {name} must start at the paired Base Robot spawn")
    if (
        comparison["initial_task_reference_xy"]
        == comparison["revised_task_reference_xy"]
    ):
        raise ValueError("TA initial and revised reference routes must differ")
    route = variant["world_knowledge"]["episode_route"]
    objective_xy = route["navigate_to_target"]["points"][-1]["position_xy"]
    terminal_xy = route["transport_to_destination"]["points"][-1]["position_xy"]
    if dist(comparison["revised_task_reference_xy"][-1], terminal_xy) > 0.002:
        raise ValueError("TA revised reference must end at its authored final stance")
    if not any(
        dist(point, objective_xy) <= 0.002
        for point in comparison["revised_task_reference_xy"]
    ):
        raise ValueError("TA revised reference must pass through the final task stance")
    scored = [
        n
        for n in variant["evaluation"]["milestones"]
        if n.get("kind", "scored") == "scored"
    ]
    if len(scored) != 5 or [n["milestone_id"] for n in scored] != [
        n["milestone_id"] for n in old
    ]:
        raise ValueError("TA must retain the same five scored outcome IDs")
    for a, b in zip(old, scored):
        if a["reaction"]["evaluator"] != b["reaction"]["evaluator"]:
            raise ValueError("TA changed a Base evaluator")

    revision = next(
        (
            n
            for n in variant["evaluation"]["milestones"]
            if n["milestone_id"] == "task_change_delivered"
        ),
        None,
    )
    if revision is None or len(revision["event"]["task_updates"]) != 1:
        raise ValueError("TA needs one delivered task update")
    arrival = next(
        (
            n
            for n in variant["evaluation"]["milestones"]
            if n["milestone_id"] == "revision_speaker_arrived"
        ),
        None,
    )
    if arrival is not None:
        if arrival["preconditions"] != [_ref("onsite_target_observed")]:
            raise ValueError(
                "TA speaker may start moving only after initial target observation"
            )
        if _ref("revision_speaker_arrived") not in revision["preconditions"]:
            raise ValueError("TA revision speech must wait for speaker arrival")
    trigger_after = (
        "onsite_target_observed"
        if variant["evaluation"]["task_profile"]["subtype"] == "task_content_update"
        else "onsite_task_completed"
    )
    if _ref(trigger_after) not in revision["preconditions"]:
        raise ValueError(f"TA revision speech must follow {trigger_after}")
    actions = revision["event"]["human_actions"]
    update = revision["event"]["task_updates"][0]
    if not actions or actions[-1].get("command") != "speak":
        raise ValueError("TA revision must end with actual Human speech")
    if update["source_human_id"] != actions[-1].get("human"):
        raise ValueError("TA task update must be sourced by the actual speaker")
    updates = revision["event"]["task_updates"][0]["outcome_updates"]
    subtype = variant["evaluation"]["task_profile"]["subtype"]
    expected = (
        {"onsite_task_completed", "task_completed"}
        if subtype == "task_content_update"
        else {"recipient_observed", "task_completed"}
    )
    if set(updates) != expected:
        raise ValueError("TA must replace exactly the two remaining scored outcomes")
    old_by_id = {node["milestone_id"]: node for node in old}
    for outcome_id in expected:
        if updates[outcome_id] != old_by_id[outcome_id]["reaction"]["ground_truth"]:
            raise ValueError(f"TA outcome {outcome_id} does not converge to Base")
    for outcome_id in ("onsite_task_completed", "recipient_observed", "task_completed"):
        if (
            final_outcome_ground_truth(parsed.evaluation, outcome_id)
            != old_by_id[outcome_id]["reaction"]["ground_truth"]
        ):
            raise ValueError(f"TA eventual outcome {outcome_id} differs from Base")
    if subtype == "task_content_update":
        observed = next(
            node for node in scored if node["milestone_id"] == "onsite_target_observed"
        )
        base_truth = old_by_id["onsite_target_observed"]["reaction"]["ground_truth"]
        revised = next(
            (
                node
                for node in variant["evaluation"]["milestones"]
                if node["milestone_id"] == "revised_target_observed"
            ),
            None,
        )
        if observed["reaction"]["ground_truth"] != base_truth and (
            revised is None or revised["completion"]["ground_truth"] != base_truth
        ):
            raise ValueError("TA revised observation must bind the actual Base target")
        if (
            scored[2]["reaction"]["ground_truth"]
            == old_by_id["onsite_task_completed"]["reaction"]["ground_truth"]
        ):
            raise ValueError("TA content revision has no changed onsite commitment")
    elif update["field"] == "recipient":
        if (
            scored[3]["reaction"]["ground_truth"]["human_id"]
            == old_by_id["recipient_observed"]["reaction"]["ground_truth"]["human_id"]
        ):
            raise ValueError("TA recipient revision has no changed Human")
        initial_id = scored[3]["reaction"]["ground_truth"]["human_id"]
        initial_human = variant_humans[initial_id]
        if (
            dist(
                comparison["initial_task_reference_xy"][-1],
                initial_human["start_position"][:2],
            )
            > HUMAN_COMMUNICATION_RADIUS_M
        ):
            raise ValueError(
                "TA initial task route does not reach the initial recipient"
            )
    elif update["field"] == "delivery_destination":
        relocation = next(
            (
                node
                for node in variant["evaluation"]["milestones"]
                if node["milestone_id"] == "revision_speaker_relocated"
            ),
            None,
        )
        if relocation is None or not any(
            action.get("command") == "goto"
            for action in relocation["event"]["human_actions"]
        ):
            raise ValueError(
                "TA destination revision requires actual recipient relocation"
            )
        old_anchor = variant["world_knowledge"]["spatial_anchors"][
            update["previous_value_ref"]
        ]
        if (
            dist(comparison["initial_task_reference_xy"][-1], old_anchor)
            > HUMAN_COMMUNICATION_RADIUS_M
        ):
            raise ValueError(
                "TA initial task route does not reach the old meeting location"
            )
    else:
        raise ValueError("Unknown delivery revision field")

    if parsed.authoring.difficulty_factors:
        raise ValueError("TA variants inherit Base difficulty without authored factors")


__all__ = [
    "TaskUpdateDesign",
    "compile_task_update_variant",
    "validate_task_update_pair",
]
