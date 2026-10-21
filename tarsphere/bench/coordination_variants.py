"""Bind authored Human contributions to the unchanged five Base outcomes."""

from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from tarsphere.bench.asset_adaptation import EpisodeAssetInstance
from tarsphere.bench.certification.certified_asset_pool import (
    certified_asset,
    load_certified_asset_pool,
)
from tarsphere.bench.schema import HumanSpec, TARSphereEpisode
from tarsphere.contracts.human_interaction import HUMAN_COMMUNICATION_RADIUS_M
from tarsphere.contracts.task_profile import coordination_affected_outcomes


class CoordinationDesign(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    human_id: str = Field(
        min_length=1,
        description="Existing Base Human who contributes; requester is allowed.",
    )
    interaction_route_point_id: Literal[
        "task_start", "task_target", "completion_stance"
    ] = Field(
        description="Existing A/B/C interaction stance near the contributor, for route evidence rather than a forced question time."
    )
    interaction_position_xy: tuple[float, float] | None = Field(
        default=None,
        description=(
            "Reviewed conversation stance in world metres when the existing route point "
            "cannot support visible conversation. Inserts one contributor visit after the "
            "start/target or before completion, without moving any Base anchor or Human."
        ),
    )
    request: str = Field(
        min_length=1,
        description="Complete public opening request including reviewed route cues.",
    )
    contribution: str = Field(
        min_length=1,
        description="Necessary auxiliary contribution, distinct from the Robot outcome.",
    )
    affected_outcome_id: Literal[
        "onsite_target_observed",
        "onsite_task_completed",
        "recipient_observed",
        "task_completed",
    ] = Field(
        description="Base outcome requiring the contribution; other branches remain independent."
    )
    topic: str = Field(
        min_length=1,
        description="Public topic for requesting the contribution; never disclose its private answer.",
    )
    answer: str = Field(
        min_length=1,
        description="Delivered factual information or affirmative agreement to the auxiliary work.",
    )
    additional_humans: tuple[HumanSpec, ...] = Field(
        default=(),
        description=(
            "Static supporting roles bound to a reviewed native person or canonical "
            "catalog asset when the paired Base has no suitable contributor."
        ),
    )


class InformationCoordinationDesign(CoordinationDesign):
    subtype: Literal["information_coordination"]
    private_base_intent_ids: tuple[str, ...] = Field(
        default=(),
        description=(
            "Existing Base inquiry topics whose public names or descriptions would reveal "
            "the factual choice supplied by the contributor. Hides only directory exposure; "
            "authorized natural-language routing, answers and scored bindings stay unchanged."
        ),
    )
    coordination_pattern: Literal[
        "enroute_information", "onsite_information", "delivery_information"
    ] = "onsite_information"
    added_world_objects: tuple[dict[str, Any], ...] = Field(
        default=(),
        description="Exact bounds and prim bindings for evidenced allocation alternatives; Base objects stay immutable.",
    )
    added_asset_instances: tuple[EpisodeAssetInstance, ...] = Field(
        default=(),
        description="Certified clarification items sharing the Base target support; their physical placement needs separate visual and runtime review.",
    )
    allocation_candidate_object_ids: tuple[str, ...] = Field(
        default=(),
        description="Distinct real candidates for the private factual allocation, including the unchanged Base target.",
    )


class TaskCollaborationDesign(CoordinationDesign):
    subtype: Literal["task_collaboration"]
    coordination_pattern: Literal[
        "preparatory_collaboration",
        "parallel_collaboration",
        "completion_collaboration",
    ] = "parallel_collaboration"
    work_position: tuple[float, float, float] = Field(
        description="Evidence-backed world XYZ operation stance; not a guessed scene location."
    )
    work_site: str = Field(
        min_length=1,
        description="Real semantic work locus and its source evidence for human review.",
    )
    outbound_waypoints: tuple[tuple[float, float, float], ...] = Field(
        default=(),
        description="Optional evidenced corridor waypoints for Human travel to the work site.",
    )
    return_waypoints: tuple[tuple[float, float, float], ...] = Field(
        default=(),
        description="Optional evidenced corridor waypoints for Human travel back to its initial location.",
    )
    task_description: str = Field(
        min_length=1,
        description="Symbolic auxiliary work performed at the real stance.",
    )
    result_message: str = Field(
        min_length=1,
        description="Work result delivered locally after returning to the original Human position.",
    )


COORDINATION_DESIGN = TypeAdapter(
    Annotated[
        InformationCoordinationDesign | TaskCollaborationDesign,
        Field(discriminator="subtype"),
    ]
)


def _ref(node_id: str) -> dict:
    return {"kind": "milestone", "milestone_ref": node_id}


def _event(node_id, parents, event_type, description, evaluator, truth, actions=()):
    return {
        "milestone_id": node_id,
        "kind": "event",
        "preconditions": [_ref(p) for p in parents],
        "event": {
            "type": event_type,
            "description": description,
            "human_actions": list(actions),
        },
        "completion": {
            "description": description,
            "evaluator": evaluator,
            "ground_truth": truth,
        },
    }


def _add_allocation_candidates(
    data: dict, design: InformationCoordinationDesign
) -> None:
    world = data["world_knowledge"]
    candidates = design.allocation_candidate_object_ids
    if not candidates:
        if design.added_world_objects or design.added_asset_instances:
            raise ValueError(
                "Added objects require an explicit factual allocation candidate set"
            )
        return
    data.setdefault("metadata", {})["coordination_candidate_object_ids"] = list(
        candidates
    )
    if len(candidates) < 2 or len(set(candidates)) != len(candidates):
        raise ValueError("An allocation requires at least two distinct real candidates")
    onsite = next(
        n
        for n in data["evaluation"]["milestones"]
        if n["milestone_id"] == "onsite_target_observed"
    )
    targets = onsite["reaction"]["ground_truth"].get("object_ids", [])
    if not targets or not set(targets).issubset(candidates):
        raise ValueError(
            "Allocation must retain the Base observed target among its candidates"
        )
    known = {o["object_id"] for o in world["objects"]}
    for item in design.added_world_objects:
        if not item.get("object_id") or item["object_id"] in known:
            raise ValueError("Allocation object duplicates an existing identity")
        if item["object_id"] not in candidates:
            raise ValueError("An added allocation object must be a declared candidate")
        world["objects"].append(deepcopy(item))
        known.add(item["object_id"])
    if not set(candidates).issubset(known):
        raise ValueError("Allocation candidates must resolve to real world objects")
    if design.added_asset_instances:
        adaptation = world["asset_adaptation"]
        target = next(i for i in adaptation["instances"] if i["role"] == "target")
        pool = load_certified_asset_pool()
        for instance in design.added_asset_instances:
            certified = certified_asset(pool, instance.asset_key)
            if (
                instance.role != "clarification"
                or instance.object_id not in candidates
                or instance.support_object_id != target["support_object_id"]
                or instance.asset_path != certified["usd_path"]
                or instance.category != certified["category"]
                or list(instance.size_xyz_m) != certified["size_xyz_m"]
                or instance.asset_meters_per_unit != certified["meters_per_unit"]
            ):
                raise ValueError(
                    "HC allocation instance differs from its certified pool or Base support"
                )
            objects = {obj["object_id"]: obj for obj in world["objects"]}
            matching = [
                obj
                for obj in design.added_world_objects
                if obj["object_id"] == instance.object_id
            ]
            if len(matching) != 1:
                raise ValueError(
                    "An allocation instance requires exactly one added world object"
                )
            obj = matching[0]
            if instance.orientation_xyzw != (0.0, 0.0, 0.0, 1.0):
                raise ValueError(
                    "HC allocation currently requires the certified unrotated placement"
                )
            lower = [
                p - size / 2
                for p, size in zip(instance.position_xyz_m, instance.size_xyz_m)
            ]
            upper = [
                p + size / 2
                for p, size in zip(instance.position_xyz_m, instance.size_xyz_m)
            ]
            if (
                obj.get("runtime_prim") != instance.runtime_prim
                or obj.get("category") != instance.category
                or any(
                    not math.isclose(a, b, abs_tol=1e-6)
                    for a, b in zip(obj["position_xyz"], instance.position_xyz_m)
                )
                or any(
                    not math.isclose(a, b, abs_tol=1e-6)
                    for a, b in zip(obj["bounds_min_xyz"], lower)
                )
                or any(
                    not math.isclose(a, b, abs_tol=1e-6)
                    for a, b in zip(obj["bounds_max_xyz"], upper)
                )
            ):
                raise ValueError(
                    "Allocation world geometry differs from its physical instance"
                )
            support = objects[instance.support_object_id]
            if any(
                lower[axis] < support["bounds_min_xyz"][axis] - 1e-6
                or upper[axis] > support["bounds_max_xyz"][axis] + 1e-6
                for axis in (0, 1)
            ):
                raise ValueError(
                    "Allocation instance extends beyond its Base support footprint"
                )
            # These alternatives reuse the exact horizontal support plane of
            # the certified Base placement; local mesh support remains a live gate.
            target_bottom = target["position_xyz_m"][2] - target["size_xyz_m"][2] / 2
            if not math.isclose(lower[2], target_bottom, abs_tol=1e-6):
                raise ValueError(
                    "Allocation instance must reuse the Base support plane"
                )
            for other in adaptation["instances"]:
                other_lower = objects[other["object_id"]]["bounds_min_xyz"]
                other_upper = objects[other["object_id"]]["bounds_max_xyz"]
                if all(
                    lower[axis] < other_upper[axis] and upper[axis] > other_lower[axis]
                    for axis in range(3)
                ):
                    raise ValueError("Authored allocation instances overlap")
            adaptation["instances"].append(instance.model_dump(mode="json"))
            world["physical_task"]["distractor_object_ids"].append(instance.object_id)


def compile_coordination_variant(base: dict, design: CoordinationDesign) -> dict:
    original = TARSphereEpisode.model_validate(base)
    if original.evaluation.task_profile.capability_slice.value != "base":
        raise ValueError("HC compilation requires a paired Base")
    data = deepcopy(base)
    if isinstance(design, InformationCoordinationDesign):
        _add_allocation_candidates(data, design)
    existing_ids = {person["id"] for person in data["humans"]}
    existing_prims = {
        person["scene_person_prim_path"]
        for person in data["humans"]
        if person.get("scene_person_prim_path")
    }
    for additional in design.additional_humans:
        if not additional.static or not (
            additional.scene_person_prim_path or additional.asset
        ):
            raise ValueError(
                "An additional coordination Human must be static and bind a native person or catalog asset"
            )
        if additional.id in existing_ids or (
            additional.scene_person_prim_path
            and additional.scene_person_prim_path in existing_prims
        ):
            raise ValueError(
                "An additional coordination Human duplicates an existing identity or scene prim"
            )
        data["humans"].append(
            additional.model_dump(
                mode="json", exclude_none=True, exclude={"name", "character_name"}
            )
        )
        existing_ids.add(additional.id)
        if additional.scene_person_prim_path:
            existing_prims.add(additional.scene_person_prim_path)
    human = next((h for h in data["humans"] if h["id"] == design.human_id), None)
    if human is None:
        raise ValueError("Coordination contribution must bind an authored Human")
    opening_requester_id = data["evaluation"]["milestones"][0]["event"][
        "human_actions"
    ][0]["human"]
    if isinstance(design, InformationCoordinationDesign):
        if design.human_id == opening_requester_id:
            raise ValueError(
                "Information coordination source must differ from the opening requester"
            )
        base_onsite = next(
            node
            for node in base["evaluation"]["milestones"]
            if node["milestone_id"] == "onsite_task_completed"
        )
        if (
            base_onsite["reaction"]["evaluator"] == "information_received"
            and design.human_id == base_onsite["reaction"]["ground_truth"]["human_id"]
        ):
            raise ValueError(
                "Information coordination for an inquiry requires a second Human, "
                "distinct from the paired Base inquiry source"
            )
        base_opening = base["evaluation"]["milestones"][0]["event"]["human_actions"][0][
            "message"
        ]
        if design.request.strip() == base_opening.strip():
            raise ValueError(
                "Information coordination must rewrite the Base opening to expose the missing fact"
            )
        if design.answer.strip().lower() in design.request.lower():
            raise ValueError(
                "Information coordination opening leaks the authored answer"
            )
        explicit_addressee = re.match(
            r"^\s*Ask(?:\s+the)?\s+([A-Z][A-Za-z0-9_]*)\b", design.topic
        )
        if (
            explicit_addressee
            and explicit_addressee.group(1) in existing_ids
            and explicit_addressee.group(1) != design.human_id
        ):
            raise ValueError(
                "Information coordination topic addresses a Human other than its bound source"
            )
    nodes = data["evaluation"]["milestones"]
    by_id = {node["milestone_id"]: node for node in nodes}
    data["episode_id"] = (
        base["episode_id"].removesuffix("_base") + "_human_coordination"
    )
    data["title"] = base["title"] + " / human coordination"
    data["description"] = design.contribution
    by_id["task_understanding"]["event"]["human_actions"][0]["message"] = design.request
    plan = by_id["task_understanding"]["reaction"]["ground_truth"]
    plan["goal"] = design.request
    plan["required_subgoals"] = [
        f"Obtain the necessary contribution from {design.human_id}; a conditional plan is correct before the reply.",
        "Use the public route cues and perform the Robot onsite task; independent work may proceed in parallel.",
        "Complete the original report or handover with the named recipient after the necessary contribution.",
    ]
    intents = data.setdefault("interaction", {}).setdefault("intents", [])
    if isinstance(design, InformationCoordinationDesign):
        private_ids = design.private_base_intent_ids
        if len(private_ids) != len(set(private_ids)):
            raise ValueError("Private Base intent IDs must be unique")
        base_intents = {intent["intent_id"]: intent for intent in intents}
        if set(private_ids) - base_intents.keys():
            raise ValueError(
                "Private Base intent IDs must reference existing Base topics"
            )
        for private_id in private_ids:
            base_intents[private_id]["public"] = False
    intent_id = "coordination_contribution"
    if any(i["intent_id"] == intent_id for i in intents):
        raise ValueError("Base already declares the HC contribution topic")
    intents.append(
        {
            "intent_id": intent_id,
            "description": design.topic,
            "eligible_human_ids": [design.human_id],
            "answer": {"kind": "authored_text", "text": design.answer},
        }
    )
    collaboration = isinstance(design, TaskCollaborationDesign)
    if isinstance(design, InformationCoordinationDesign):
        expected_point = {
            "enroute_information": "task_start",
            "onsite_information": "task_target",
            "delivery_information": "task_target",
        }[design.coordination_pattern]
        if design.interaction_route_point_id != expected_point:
            raise ValueError(
                f"{design.coordination_pattern} must use the existing "
                f"{expected_point} interaction point"
            )
        contribution_parent = {
            "enroute_information": "task_understanding",
            "onsite_information": "onsite_target_observed",
            "delivery_information": "onsite_task_completed",
        }[design.coordination_pattern]
    else:
        if design.coordination_pattern == "completion_collaboration":
            contribution_parent = (
                "onsite_task_completed"
                if design.interaction_route_point_id == "task_target"
                else "recipient_observed"
            )
            if design.interaction_route_point_id == "task_start":
                raise ValueError("Completion collaboration cannot start at task_start")
        else:
            contribution_parent = "task_understanding"
    expected_affected = coordination_affected_outcomes(
        design.subtype, design.coordination_pattern
    )
    if design.affected_outcome_id not in expected_affected:
        raise ValueError(
            f"{design.coordination_pattern} must affect {' or '.join(expected_affected)}, "
            f"not {design.affected_outcome_id}"
        )
    receipt_id = (
        "contribution_accepted"
        if collaboration
        else "coordination_information_received"
    )
    added = [
        _event(
            receipt_id,
            [contribution_parent],
            "contribution_requested" if collaboration else "task_phase_opened",
            f"{design.human_id} actually delivers the authorized contribution response.",
            "information_received",
            {"human_id": design.human_id, "intent_id": intent_id},
        )
    ]
    if isinstance(design, InformationCoordinationDesign):
        onsite = by_id["onsite_task_completed"]["reaction"]
        if (
            onsite["evaluator"] == "information_received"
            and design.coordination_pattern != "delivery_information"
        ):
            onsite_intent_id = onsite["ground_truth"]["intent_id"]
            onsite_intent = next(
                (
                    intent
                    for intent in intents
                    if intent["intent_id"] == onsite_intent_id
                ),
                None,
            )
            if onsite_intent is None:
                raise ValueError(f"Missing onsite inquiry intent: {onsite_intent_id}")
            onsite_intent.setdefault("preconditions", []).append(_ref(receipt_id))
    contribution_id = receipt_id
    prerequisite = receipt_id
    if collaboration:
        point = list(design.work_position)
        if not all(math.isfinite(x) for x in point):
            raise ValueError("Operation stance must be finite")
        moving = math.dist(human["start_position"], point) > 0.001
        if moving and human.get("scene_person_prim_path"):
            raise ValueError("A scene-native contributor cannot move")
        data["world_knowledge"]["spatial_anchors"]["coordination_work"] = point
        actions = []
        if moving:
            human["static"] = False
            travel = {"human": design.human_id, "command": "goto", "target": point}
            if design.outbound_waypoints:
                travel["waypoints"] = [
                    *[list(p) for p in design.outbound_waypoints],
                    point,
                ]
            actions.append(travel)
        elif design.outbound_waypoints or design.return_waypoints:
            raise ValueError("In-place work cannot declare unused travel waypoints")
        actions.append(
            {
                "human": design.human_id,
                "command": "perform_task",
                "target_anchor": "coordination_work",
                "task_description": design.task_description,
            }
        )
        contribution_id = "auxiliary_work_completed"
        added.append(
            _event(
                contribution_id,
                [receipt_id],
                "contribution_completed",
                design.task_description,
                "human_event_completed",
                {"milestone_id": contribution_id},
                actions,
            )
        )
        delivery_parent = contribution_id
        if moving:
            delivery_parent = "contributor_returned"
            travel = {
                "human": design.human_id,
                "command": "goto",
                "target": human["start_position"],
            }
            if design.return_waypoints:
                travel["waypoints"] = [
                    *[list(p) for p in design.return_waypoints],
                    human["start_position"],
                ]
            added.append(
                _event(
                    delivery_parent,
                    [contribution_id],
                    "human_became_available",
                    "The contributor returns to the original interaction location.",
                    "human_event_completed",
                    {"milestone_id": delivery_parent},
                    [travel],
                )
            )
        prerequisite = "contribution_result_delivered"
        result = _event(
            prerequisite,
            [delivery_parent],
            "information_disclosed",
            "The auxiliary result is spoken locally, not transmitted remotely.",
            "human_event_completed",
            {"milestone_id": prerequisite},
            [
                {
                    "human": design.human_id,
                    "command": "speak",
                    "message": design.result_message,
                }
            ],
        )
        result["preconditions"].append(
            {
                "kind": "detector",
                "condition_id": "contributor_within_auditory_range",
                "detector": "robot_near_human",
                "arguments": {
                    "human_id": design.human_id,
                    "distance": HUMAN_COMMUNICATION_RADIUS_M,
                },
                "description": "The Robot can hear this actual local result delivery.",
            }
        )
        added.append(result)
    parent_index = next(
        index
        for index, node in enumerate(nodes)
        if node["milestone_id"] == contribution_parent
    )
    nodes[parent_index + 1 : parent_index + 1] = added
    interaction_point_id = design.interaction_route_point_id
    if design.interaction_position_xy is not None:
        from tarsphere.bench.spatial.coordination_route import (
            resolve_coordination_detour,
            CONVERSATION_POINT_ID,
        )

        resolve_coordination_detour(
            data,
            human,
            design.interaction_route_point_id,
            design.interaction_position_xy,
        )
        interaction_point_id = CONVERSATION_POINT_ID
    route = data["world_knowledge"]["episode_route"]
    points = [
        p
        for leg in ("navigate_to_target", "transport_to_destination")
        for p in route[leg]["points"]
    ]
    stance = next(p for p in points if p["point_id"] == interaction_point_id)
    if (
        math.dist(stance["position_xy"], human["start_position"][:2])
        > HUMAN_COMMUNICATION_RADIUS_M
    ):
        raise ValueError("Contribution route evidence must actually approach its Human")
    existing_stops = {stop["milestone_id"]: stop for stop in route["milestone_stops"]}
    added_ids = {node["milestone_id"] for node in added}
    route["milestone_stops"] = [
        (
            existing_stops[node["milestone_id"]]
            if node["milestone_id"] in existing_stops
            else {
                "milestone_id": node["milestone_id"],
                "route_point_id": interaction_point_id,
            }
        )
        for node in nodes
        if node["milestone_id"] in existing_stops or node["milestone_id"] in added_ids
    ]
    by_id[design.affected_outcome_id]["preconditions"].append(_ref(prerequisite))
    data["evaluation"]["task_profile"] = {
        "mainline": original.evaluation.task_profile.mainline.value,
        "capability_slice": "human_coordination",
        "subtype": design.subtype,
        "coordination_pattern": design.coordination_pattern,
        "supporting_human_id": design.human_id,
        "coordination_milestone_id": prerequisite,
        "affected_outcome_id": design.affected_outcome_id,
        "contribution": design.contribution,
    }
    data["metadata"].update(
        paired_base_episode_id=base["episode_id"],
        task_source="paired_coordination_variants",
    )
    data["authoring"].update(
        achieved_gates=["authored", "schema_valid"],
        external_evidence={},
        difficulty_factors=[],
    )
    data["tags"] = [tag for tag in data.get("tags", []) if tag != "base"] + [
        "human_coordination"
    ]
    validate_coordination_pair(base, data)
    return data


def _validate_allocation_world_field(key: str, value: Any, variant: dict) -> None:
    current = variant["world_knowledge"][key]
    information = (
        variant["evaluation"]["task_profile"]["subtype"] == "information_coordination"
    )
    if key == "objects":
        valid = current[: len(value)] == value and (information or current == value)
    else:
        collection = (
            "instances" if key == "asset_adaptation" else "distractor_object_ids"
        )
        left, right = deepcopy(value), deepcopy(current)
        original, expanded = left.pop(collection), right.pop(collection)
        valid = (
            left == right
            and expanded[: len(original)] == original
            and (information or expanded == original)
        )
        if key == "asset_adaptation" and any(
            i["role"] != "clarification" for i in expanded[len(original) :]
        ):
            valid = False
    if not valid:
        raise ValueError(
            f"HC changed a Base world field instead of adding allocation alternatives: {key}"
        )


def validate_coordination_pair(base: dict, variant: dict) -> None:
    TARSphereEpisode.model_validate(variant)
    for key in ("scene", "robot", "runtime", "asset_adaptation"):
        if base.get(key) != variant.get(key):
            raise ValueError(f"HC changed paired Base {key}")
    for key, value in base["world_knowledge"].items():
        if key == "spatial_anchors":
            if any(
                variant["world_knowledge"][key].get(k) != v for k, v in value.items()
            ):
                raise ValueError("HC changed a Base spatial anchor")
        elif key in {"objects", "asset_adaptation", "physical_task"}:
            _validate_allocation_world_field(key, value, variant)
        elif key == "episode_route":
            from tarsphere.bench.spatial.coordination_route import (
                validate_coordination_route_pair,
            )

            validate_coordination_route_pair(base, variant)
        elif variant["world_knowledge"].get(key) != value:
            raise ValueError(f"HC changed paired Base world field {key}")
    extra_objects = variant["world_knowledge"]["objects"][
        len(base["world_knowledge"]["objects"]) :
    ]
    if extra_objects:
        candidates = variant["metadata"].get("coordination_candidate_object_ids", [])
        existing = {obj["object_id"] for obj in variant["world_knowledge"]["objects"]}
        observed = next(
            node
            for node in base["evaluation"]["milestones"]
            if node["milestone_id"] == "onsite_target_observed"
        )["reaction"]["ground_truth"].get("object_ids", [])
        if (
            len(candidates) < 2
            or len(set(candidates)) != len(candidates)
            or not set(candidates).issubset(existing)
            or not set(observed).issubset(candidates)
            or any(obj["object_id"] not in candidates for obj in extra_objects)
        ):
            raise ValueError(
                "HC added objects must support the declared Base-target allocation"
            )
    human_bindings = lambda people: [
        {key: value for key, value in human.items() if key != "static"}
        for human in people
    ]
    original_humans = variant["humans"][: len(base["humans"])]
    if human_bindings(base["humans"]) != human_bindings(original_humans):
        raise ValueError("HC changed a Base Human identity or initial pose")
    additional_humans = variant["humans"][len(base["humans"]) :]
    existing_ids = {person["id"] for person in base["humans"]}
    existing_prims = {
        person["scene_person_prim_path"]
        for person in base["humans"]
        if person.get("scene_person_prim_path")
    }
    for additional in additional_humans:
        if not additional.get("static") or not (
            additional.get("scene_person_prim_path") or additional.get("asset")
        ):
            raise ValueError(
                "Additional coordination Humans must be static and bind a native person or catalog asset"
            )
        if additional["id"] in existing_ids or (
            additional.get("scene_person_prim_path")
            and additional["scene_person_prim_path"] in existing_prims
        ):
            raise ValueError(
                "Additional coordination Human duplicates an identity or scene prim"
            )
        existing_ids.add(additional["id"])
        if additional.get("scene_person_prim_path"):
            existing_prims.add(additional["scene_person_prim_path"])
    old = base["evaluation"]["milestones"]
    scored = [
        n
        for n in variant["evaluation"]["milestones"]
        if n.get("kind", "scored") == "scored"
    ]
    if len(scored) != 5 or [n["milestone_id"] for n in scored] != [
        n["milestone_id"] for n in old
    ]:
        raise ValueError("HC must retain the same five scored outcomes")
    for a, b in zip(old, scored):
        if a["reaction"]["evaluator"] != b["reaction"]["evaluator"] or (
            a["milestone_id"] != "task_understanding" and a["reaction"] != b["reaction"]
        ):
            raise ValueError("HC changed a Base outcome standard")
