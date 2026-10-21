"""Machine gates for the 2 x 6 benchmark reporting matrix.

The task profile is a reporting coordinate, not a bag of difficulty labels.
Each official episode declares exactly one physical mainline and one primary
capability slice. This module proves that declaration against the milestone
DAG and episode-owned world state. Semantic quality is reviewed later.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

from tarsphere.contracts.task_profile import (
    COMPONENT_OUTAGE_TOOL_ALTERNATIVES,
    SCORED_OUTCOME_IDS,
    CapabilitySlice,
    TaskMainline,
)
from tarsphere.evaluation.event_vocabulary import EVENT_TYPE_CATALOG, CausalEventRole
from tarsphere.evaluation.schema import MilestoneCondition, MilestoneReference
from tarsphere.evaluation.task_bindings import final_outcome_ground_truth

_PHYSICAL_FETCH_EVALUATORS = frozenset(
    {
        "robot_holding_object",
        "handover_accepted",
        "object_supported_by",
    }
)
_CHALLENGE_CHANGE_ROLES = frozenset(
    {
        CausalEventRole.SPATIAL,
        CausalEventRole.AVAILABILITY,
        CausalEventRole.REVISION,
    }
)


def validate_episode_task_profile(episode: Any) -> None:
    """Five task points, with explicit unscored causal interference."""
    if episode.evaluation is None:
        return
    profile = episode.evaluation.task_profile
    nodes = tuple(episode.evaluation.milestones)
    scored = tuple(node for node in nodes if node.kind == "scored")
    by_id = {node.milestone_id: node for node in nodes}
    _validate_mainline(episode, profile.mainline)
    _validate_progress_structure(profile.mainline, profile.capability_slice, nodes)
    active_nodes = nodes
    for event in nodes:
        replacements = {
            key: truth
            for update in event.event.task_updates
            for key, truth in update.outcome_updates.items()
        }
        if not replacements:
            continue
        if profile.capability_slice is not CapabilitySlice.TASK_UPDATE:
            raise ValueError("Only task update may revise scored task bindings")
        if event.milestone_id != profile.revision_milestone_id:
            raise ValueError(
                "Task bindings may change only at the declared revision event"
            )
        active_nodes = tuple(
            node.model_copy(
                update={
                    "reaction": node.outcome.model_copy(
                        update={"ground_truth": replacements[node.milestone_id]}
                    )
                }
            )
            if node.milestone_id in replacements
            else node
            for node in active_nodes
        )
        _validate_progress_structure(
            profile.mainline, profile.capability_slice, active_nodes
        )
        if profile.mainline is TaskMainline.FETCH_AND_DELIVER:
            active = {
                node.milestone_id: node.outcome.ground_truth for node in active_nodes
            }
            target = active["onsite_task_completed"]["object_id"]
            observed = active["onsite_target_observed"]["object_ids"]
            if target not in observed:
                revised = by_id.get("revised_target_observed")
                if (
                    revised is None
                    or revised.kind != "event"
                    or revised.outcome.evaluator != "visual_targets_observed"
                ):
                    raise ValueError(
                        "Revised fetch target needs actual revised observation"
                    )
                observed = revised.outcome.ground_truth["object_ids"]
            if (
                active["task_completed"]["object_id"] != target
                or target not in observed
            ):
                raise ValueError(
                    "Revised fetch discovery, hold and handover must bind the same target"
                )
    for intent in episode.interaction.intents:
        if any(ref.milestone_ref not in by_id for ref in intent.preconditions):
            raise ValueError("Intent references an unknown causal node")
        for node in nodes:
            truths = [
                node.outcome.ground_truth,
                *(
                    update.outcome_updates[node.milestone_id]
                    for event in nodes
                    for update in event.event.task_updates
                    if node.milestone_id in update.outcome_updates
                ),
            ]
            if (
                node.outcome.evaluator == "information_received"
                and any(truth["intent_id"] == intent.intent_id for truth in truths)
                and any(
                    not _is_ancestor(ref.milestone_ref, node.milestone_id, nodes)
                    for ref in intent.preconditions
                )
            ):
                raise ValueError(
                    "Answer routing must depend on completed ancestors of its receipt"
                )
    capability = profile.capability_slice
    if capability is CapabilitySlice.BASE:
        _validate_base(profile.mainline, scored)
        if len(nodes) != 5:
            raise ValueError("Base has no auxiliary required causal events")
        return
    if capability is CapabilitySlice.PERCEPTUAL_ROBUSTNESS:
        if len(nodes) != 5:
            raise ValueError(
                "Perceptual robustness changes the observation condition, "
                "not the five-outcome causal graph"
            )
        return
    if capability is CapabilitySlice.DISRUPTION_ADAPTATION:
        _validate_disruption_adaptation(episode, profile, nodes, by_id)
        return
    if capability is CapabilitySlice.TASK_UPDATE:
        event_id = profile.revision_milestone_id
        change = by_id.get(event_id)
        if (
            change is None
            or change.kind != "event"
            or change.outcome.evaluator != "human_event_completed"
        ):
            raise ValueError(
                "Adaptation requires non-scoring actual Human event completion"
            )
        if change.outcome.ground_truth != {"milestone_id": event_id}:
            raise ValueError("Human event completion must bind its own event")
        role = EVENT_TYPE_CATALOG[change.event.type].role
        if role is not CausalEventRole.REVISION or not change.event.task_updates:
            raise ValueError("Task update requires a typed delivered revision")
        affected = {
            key
            for update in change.event.task_updates
            for key in update.outcome_updates
        }
        expected = (
            {"onsite_task_completed", "task_completed"}
            if profile.subtype == "task_content_update"
            else {"recipient_observed", "task_completed"}
        )
        if affected != expected:
            raise ValueError(f"Task revision must atomically rebind {sorted(expected)}")
        predecessor = (
            "onsite_target_observed"
            if profile.subtype == "task_content_update"
            else "onsite_task_completed"
        )
        if not _is_ancestor(predecessor, event_id, nodes):
            raise ValueError("Task revision is outside its allowed causal phase")
        if any(not _is_ancestor(event_id, key, nodes) for key in affected):
            raise ValueError("Revision must precede each affected outcome")
        if not any(_is_ancestor(event_id, node.milestone_id, nodes) for node in scored):
            raise ValueError("Adaptation must affect a downstream task outcome")
        updates = tuple(update for update in change.event.task_updates)
        sources = {str(update.source_human_id) for update in updates}
        speeches = {
            str(action.get("human") or "")
            for action in change.event.human_actions
            if action.get("command") == "speak"
            and str(action.get("message") or "").strip()
        }
        auditory_sources = {
            str(condition.arguments.get("human_id") or "")
            for condition in change.preconditions
            if isinstance(condition, MilestoneCondition)
            and condition.detector == "robot_near_human"
        }
        moving_sources = {
            str(action.get("human") or "")
            for action in change.event.human_actions
            if action.get("command") == "goto"
        }
        if (
            not sources
            or sources != speeches
            or not sources.issubset(auditory_sources | moving_sources)
            or any(update.delivery != "speech" for update in updates)
        ):
            raise ValueError(
                "Every task revision must be audibly spoken by its declared "
                "source before the new task binding activates"
            )
        return
    if capability in {
        CapabilitySlice.HUMAN_COORDINATION,
        CapabilitySlice.INTENT_UNDERSTANDING,
    }:
        if (
            capability is CapabilitySlice.HUMAN_COORDINATION
            and profile.subtype == "task_collaboration"
        ):
            _validate_task_collaboration(episode, profile, nodes)
            return
        if capability is CapabilitySlice.INTENT_UNDERSTANDING:
            _validate_intent_resolution(episode, profile, nodes)
            if profile.resolution_mode == "context_inferable":
                return
        event_id = (
            profile.coordination_milestone_id
            if capability is CapabilitySlice.HUMAN_COORDINATION
            else profile.interaction_milestone_id
        )
        human_id = (
            profile.supporting_human_id
            if capability is CapabilitySlice.HUMAN_COORDINATION
            else profile.source_human_id
        )
        event = by_id.get(event_id)
        if (
            event is None
            or event.kind != "event"
            or event.outcome.evaluator != "information_received"
            or event.outcome.ground_truth["human_id"] != human_id
        ):
            raise ValueError(
                "Required interaction must record the actual unscored answer receipt"
            )
        affected = profile.affected_outcome_id
        if not _is_ancestor(event_id, affected, nodes):
            raise ValueError(
                "Required information must causally gate its affected task outcome"
            )
        intent_id = event.outcome.ground_truth["intent_id"]
        intent = next(
            (
                item
                for item in episode.interaction.intents
                if item.intent_id == intent_id
            ),
            None,
        )
        if intent is None or human_id not in intent.eligible_human_ids:
            raise ValueError(
                "Required private answer must be authored for the contributing Human"
            )
        if capability is CapabilitySlice.HUMAN_COORDINATION:
            milestone_parents = {
                ref.milestone_ref
                for ref in event.preconditions
                if isinstance(ref, MilestoneReference)
            }
            expected_parent = {
                "enroute_information": "task_understanding",
                "onsite_information": "onsite_target_observed",
                "delivery_information": "onsite_task_completed",
            }[profile.coordination_pattern]
            if milestone_parents != {expected_parent}:
                raise ValueError(
                    "Information contribution occurs outside its declared causal stage"
                )
        return
    raise AssertionError(f"Unhandled capability: {capability}")


def _validate_disruption_adaptation(
    episode: Any, profile: Any, nodes: tuple[Any, ...], by_id: dict[str, Any]
) -> None:
    """Bind a real contingency without changing the five task outcomes."""

    if any(node.event.task_updates for node in nodes):
        raise ValueError("Disruption adaptation cannot revise task bindings")
    if profile.subtype != "spatial_blocking":
        if len(nodes) != 5:
            raise ValueError("Timed disruptions do not add scored or causal gates")
        if profile.subtype in {"component_outage", "temporary_unavailability"}:
            from tarsphere.contracts.task_profile import (
                COMPONENT_OUTAGE_DURATION_PHYSICS_STEPS,
                TEMPORARY_UNAVAILABILITY_DURATION_PHYSICS_STEPS,
            )

            duration = (
                COMPONENT_OUTAGE_DURATION_PHYSICS_STEPS
                if profile.subtype == "component_outage"
                else TEMPORARY_UNAVAILABILITY_DURATION_PHYSICS_STEPS
            )
            if (
                episode.runtime.max_physics_steps is not None
                and duration >= episode.runtime.max_physics_steps
            ):
                raise ValueError(
                    "Temporary disruption duration must fit inside the episode physics budget"
                )
        if profile.subtype == "temporary_unavailability":
            if profile.human_id not in {human.id for human in episode.humans}:
                raise ValueError("Unavailable Human must belong to the episode")
            affected = by_id[profile.affected_outcome_id]
            if affected.outcome.evaluator not in {
                "information_received",
                "information_report_score",
                "handover_accepted",
            } or (affected.outcome.ground_truth.get("human_id") != profile.human_id):
                raise ValueError("Unavailable Human must own the affected interaction")
            return
        from tarsphere.bench.compilation.episode import (
            DEFAULT_NAVIGATION_ACTIONS,
            agent_action_space,
        )

        surface = agent_action_space()
        available = {
            *surface.cognitive,
            *surface.perceptual,
            *surface.communicative,
            *surface.physical,
            *DEFAULT_NAVIGATION_ACTIONS,
        }
        alternatives = COMPONENT_OUTAGE_TOOL_ALTERNATIVES.get(
            profile.component_tool_name
        )
        if alternatives is None:
            raise ValueError(
                "Component outage must name one of the three supported tools"
            )
        if (
            profile.component_tool_name not in available
            or not alternatives <= available
        ):
            raise ValueError(
                "Component outage target and every declared recovery "
                "alternative must be on the default evaluated Agent surface"
            )
        return

    human_id = profile.blocking_human_id
    human = next((item for item in episode.humans if item.id == human_id), None)
    if human is None or human.static:
        raise ValueError("Spatial blocker must be a movable episode Human")
    block = by_id.get(profile.blocking_milestone_id)
    request = by_id.get(profile.clearance_request_milestone_id)
    clear = by_id.get(profile.clear_milestone_id)
    if any(node is None or node.kind != "event" for node in (block, request, clear)):
        raise ValueError("Spatial blocking requires three non-scoring side-event nodes")
    if (
        block.event.type.value != "passage_blocked_by_human"
        or request.event.type.value != "interaction_opportunity_opened"
        or clear.event.type.value != "passage_cleared_by_human"
    ):
        raise ValueError(
            "Spatial blocking must use typed block, request and clear events"
        )
    if (
        block.outcome.evaluator != "human_event_completed"
        or request.outcome.evaluator != "information_received"
        or clear.outcome.evaluator != "human_event_completed"
    ):
        raise ValueError(
            "Spatial disruption requires actual motion and answer receipts"
        )
    if (
        request.outcome.ground_truth.get("human_id") != human_id
        or request.outcome.ground_truth.get("intent_id")
        != profile.passage_request_intent_id
    ):
        raise ValueError("Passage request must be delivered to the blocker")
    intent = next(
        (
            item
            for item in episode.interaction.intents
            if item.intent_id == profile.passage_request_intent_id
        ),
        None,
    )
    if (
        intent is None
        or not intent.public
        or human_id not in intent.eligible_human_ids
        or not any(
            ref.milestone_ref == block.milestone_id for ref in intent.preconditions
        )
    ):
        raise ValueError(
            "Clearance intent must be public, local to the blocker, and open "
            "after blockage"
        )
    poses = []
    for node in (block, clear):
        motion = [
            action
            for action in node.event.human_actions
            if action.get("command") == "goto" and action.get("human") == human_id
        ]
        if len(motion) != 1 or not isinstance(motion[0].get("target"), (list, tuple)):
            raise ValueError("Blocking and clearing require one real Human goto each")
        poses.append(tuple(float(value) for value in motion[0]["target"][:2]))
    clear_motion = next(
        action
        for action in clear.event.human_actions
        if action.get("command") == "goto" and action.get("human") == human_id
    )
    if clear_motion.get("route_planner") != "live_robot_aware_astar":
        raise ValueError("Spatial clearing must use the live Robot-aware A* planner")
    if sum((a - b) ** 2 for a, b in zip(*poses)) < 0.25:
        raise ValueError("Clear pose must move the Human away from the blocking pose")
    if not _is_ancestor(
        block.milestone_id, request.milestone_id, nodes
    ) or not _is_ancestor(request.milestone_id, clear.milestone_id, nodes):
        raise ValueError("Passage may clear only after an accepted local request")
    side_ids = (block.milestone_id, request.milestone_id, clear.milestone_id)
    gated_scores = {
        scored_id
        for scored_id in SCORED_OUTCOME_IDS
        if any(_is_ancestor(side_id, scored_id, nodes) for side_id in side_ids)
    }
    if gated_scores:
        raise ValueError(
            "Spatial-blocking response milestones are a non-gating DAG side "
            f"branch; they cannot gate scored outcomes {sorted(gated_scores)!r}"
        )


def _validate_task_collaboration(
    episode: Any, profile: Any, nodes: tuple[Any, ...]
) -> None:
    by_id = {node.milestone_id: node for node in nodes}
    result_id = profile.coordination_milestone_id
    result = by_id.get(result_id)
    if (
        result is None
        or result.kind != "event"
        or result.event.type != "information_disclosed"
        or result.outcome.evaluator != "human_event_completed"
        or result.outcome.ground_truth != {"milestone_id": result_id}
    ):
        raise ValueError(
            "Task collaboration requires an actual locally delivered Human result"
        )
    speech = [
        action
        for action in result.event.human_actions
        if action.get("command") == "speak"
        and action.get("human") == profile.supporting_human_id
        and action.get("message")
    ]
    proximity = [
        condition
        for condition in result.preconditions
        if isinstance(condition, MilestoneCondition)
        and condition.detector == "robot_near_human"
        and condition.arguments.get("human_id") == profile.supporting_human_id
    ]
    if len(speech) != 1 or len(proximity) != 1:
        raise ValueError(
            "Task collaboration result must be spoken locally to the Robot"
        )

    work_events = [
        node
        for node in nodes
        if node.kind == "event"
        and node.event.type == "contribution_completed"
        and node.outcome.evaluator == "human_event_completed"
        and _is_ancestor(node.milestone_id, result_id, nodes)
    ]
    if len(work_events) != 1:
        raise ValueError("Task collaboration requires one actual Human work completion")
    event = work_events[0]
    event_id = event.milestone_id
    if event.outcome.ground_truth != {"milestone_id": event_id}:
        raise ValueError("Task collaboration work completion must bind its own event")
    actions = event.event.human_actions
    work = [action for action in actions if action.get("command") == "perform_task"]
    if len(work) != 1 or work[0].get("human") != profile.supporting_human_id:
        raise ValueError(
            "Task collaboration must bind one auxiliary task to its contributing Human"
        )
    if actions[-1] != work[0] or any(
        action.get("command") not in {"goto", "perform_task"}
        or action.get("human") != profile.supporting_human_id
        for action in actions
    ):
        raise ValueError(
            "Work completion must not wait for return motion or result speech"
        )
    if not _is_ancestor(result_id, profile.affected_outcome_id, nodes):
        raise ValueError(
            "Delivered collaboration result must gate its affected task outcome"
        )
    accepted_requests = [
        node
        for node in nodes
        if node.kind == "event"
        and node.event.type == "contribution_requested"
        and node.outcome.evaluator == "information_received"
        and node.outcome.ground_truth["human_id"] == profile.supporting_human_id
        and _is_ancestor(node.milestone_id, event_id, nodes)
    ]
    if not accepted_requests:
        raise ValueError(
            "Auxiliary work requires an actually delivered request acceptance"
        )
    accepted = accepted_requests[0]
    accepted_intent_id = str(accepted.outcome.ground_truth.get("intent_id") or "")
    accepted_intent = next(
        (
            intent
            for intent in episode.interaction.intents
            if intent.intent_id == accepted_intent_id
        ),
        None,
    )
    if (
        accepted_intent is None
        or profile.supporting_human_id not in accepted_intent.eligible_human_ids
    ):
        raise ValueError(
            "Task collaboration acceptance must route through an authored "
            "intent owned by the contributing Human"
        )
    milestone_parents = {
        ref.milestone_ref
        for ref in accepted.preconditions
        if isinstance(ref, MilestoneReference)
    }
    expected_parents = {
        "preparatory_collaboration": {"task_understanding"},
        "parallel_collaboration": {"task_understanding"},
        "completion_collaboration": {"onsite_task_completed", "recipient_observed"},
    }[profile.coordination_pattern]
    if len(milestone_parents) != 1 or not milestone_parents <= expected_parents:
        raise ValueError(
            "Task collaboration request occurs outside its declared causal stage"
        )


def _episode_references(episode: Any) -> set[str]:
    world = episode.world_knowledge or {}
    objects = {
        str(item["object_id"])
        for item in world.get("objects", ())
        if isinstance(item, dict) and item.get("object_id")
    }
    anchors = world.get("spatial_anchors", {})
    anchor_ids = (
        set(anchors)
        if isinstance(anchors, dict)
        else {
            str(item.get("anchor_id") or item.get("id"))
            for item in anchors
            if isinstance(item, dict) and (item.get("anchor_id") or item.get("id"))
        }
    )
    return objects | anchor_ids | {human.id for human in episode.humans}


def _validate_intent_resolution(
    episode: Any, profile: Any, nodes: tuple[Any, ...]
) -> None:
    references = {
        option.entity_ref for option in profile.resolution_options if option.entity_ref
    }
    unresolved = references - _episode_references(episode)
    if unresolved:
        raise ValueError(f"IU options reference unknown entities: {sorted(unresolved)}")
    if profile.resolution_mode == "clarification_required":
        return
    if any(
        node.kind == "event" and node.outcome.evaluator == "information_received"
        for node in nodes
    ):
        raise ValueError(
            "context_inferable cannot require an auxiliary information receipt"
        )
    humans = {human.id: human for human in episode.humans}
    by_id = {node.milestone_id: node for node in nodes}
    for evidence in profile.public_evidence:
        if evidence.source == "public_profile":
            human = humans.get(evidence.owner_id)
            if human is None or evidence.fact_id not in {
                fact.fact_id for fact in human.public_profile_facts
            }:
                raise ValueError(
                    "IU profile evidence must resolve to an authored public Human fact"
                )
        else:
            node = by_id.get(evidence.owner_id)
            if node is None or not any(
                action.get("command") == "speak" and action.get("message")
                for action in node.event.human_actions
            ):
                raise ValueError(
                    "IU dialogue evidence must reference an actual authored utterance"
                )
            if node.milestone_id != "task_understanding" and not _is_ancestor(
                node.milestone_id, profile.affected_outcome_id, nodes
            ):
                raise ValueError(
                    "IU dialogue evidence must precede its affected outcome"
                )


def _validate_mainline(episode: Any, mainline: TaskMainline) -> None:
    milestones = tuple(episode.evaluation.milestones)
    held_nodes = [
        milestone
        for milestone in milestones
        if milestone.outcome.evaluator == "robot_holding_object"
    ]
    handover_nodes = [
        milestone
        for milestone in milestones
        if milestone.outcome.evaluator == "handover_accepted"
    ]
    physical = dict((episode.world_knowledge or {}).get("physical_task") or {})

    if mainline is TaskMainline.FETCH_AND_DELIVER:
        target = str(physical.get("target_object_id") or "")
        if not target:
            raise ValueError(
                "fetch_and_deliver requires physical_task.target_object_id"
            )
        if len(held_nodes) != 1 or len(handover_nodes) != 1:
            raise ValueError(
                "fetch_and_deliver requires exactly one "
                "robot_holding_object node and exactly one "
                "handover_accepted node; "
                f"found held={len(held_nodes)}, "
                f"handover={len(handover_nodes)}"
            )
        held_node = held_nodes[0]
        handover_node = handover_nodes[0]
        held_object = str(held_node.outcome.ground_truth.get("object_id") or "")
        handover_object = str(handover_node.outcome.ground_truth.get("object_id") or "")
        profile = episode.evaluation.task_profile
        if (
            profile.capability_slice is CapabilitySlice.TASK_UPDATE
            and profile.subtype == "task_content_update"
        ):
            held_object = str(
                final_outcome_ground_truth(
                    episode.evaluation,
                    held_node.milestone_id,
                ).get("object_id")
                or ""
            )
            handover_object = str(
                final_outcome_ground_truth(
                    episode.evaluation,
                    handover_node.milestone_id,
                ).get("object_id")
                or ""
            )
        if held_object != target or handover_object != target:
            raise ValueError(
                "fetch_and_deliver physical nodes must bind the exact "
                "physical_task target; "
                f"target={target!r}, held={held_object!r}, "
                f"handover={handover_object!r}"
            )
        if not _is_ancestor(
            str(held_node.milestone_id),
            str(handover_node.milestone_id),
            milestones,
        ):
            raise ValueError(
                "handover_accepted must be causally downstream of robot_holding_object"
            )
        return

    forbidden = [
        milestone.milestone_id
        for milestone in milestones
        if milestone.outcome.evaluator in _PHYSICAL_FETCH_EVALUATORS
    ]
    if forbidden:
        raise ValueError(
            f"navigation mainline cannot contain fetch/delivery closures: {forbidden!r}"
        )
    terminal_evaluators = {
        "inspection_submitted",
        "information_report_score",
        "visual_targets_observed",
        "human_visible",
        "robot_in_place",
    }
    if not any(
        milestone.outcome.evaluator in terminal_evaluators for milestone in milestones
    ):
        raise ValueError(
            "navigation requires a scored spatial, observation, inspection, "
            "or report result"
        )


def _validate_progress_structure(
    mainline: TaskMainline, capability: CapabilitySlice, milestones: tuple[Any, ...]
) -> None:
    """Fixed IDs identify task roles, not positions of changing event nodes."""
    from tarsphere.contracts.task_profile import SCORED_OUTCOME_IDS

    scored = tuple(node for node in milestones if node.kind == "scored")
    if tuple(node.milestone_id for node in scored) != SCORED_OUTCOME_IDS:
        raise ValueError(
            "Official episodes require exactly the five canonical scored outcome IDs"
        )
    evaluators = [node.outcome.evaluator for node in scored]
    if mainline is TaskMainline.FETCH_AND_DELIVER:
        expected = [
            "structured_plan_submitted",
            "visual_targets_observed",
            "robot_holding_object",
            "human_visible",
            "handover_accepted",
        ]
    elif evaluators[2] == "information_received":
        expected = [
            "structured_plan_submitted",
            "human_visible",
            "information_received",
            "human_visible",
            "information_report_score",
        ]
        observed_source = scored[1].outcome.ground_truth["human_id"]
        requested_source = scored[2].outcome.ground_truth["human_id"]
        if observed_source != requested_source:
            revised = next(
                (
                    node
                    for node in milestones
                    if node.milestone_id == "revised_target_observed"
                ),
                None,
            )
            if (
                capability is not CapabilitySlice.TASK_UPDATE
                or revised is None
                or revised.kind != "event"
                or revised.outcome.evaluator != "human_visible"
                or revised.outcome.ground_truth["human_id"] != requested_source
            ):
                raise ValueError(
                    "Inquiry discovery human_visible must bind the information source"
                )
    else:
        expected = [
            "structured_plan_submitted",
            "visual_targets_observed",
            "inspection_submitted",
            "human_visible",
            "information_report_score",
        ]
    if evaluators != expected:
        raise ValueError(
            f"Invalid five-stage progress structure: expected={expected}, got={evaluators}"
        )
    if (
        scored[3].outcome.ground_truth["human_id"]
        != scored[4].outcome.ground_truth["human_id"]
    ):
        raise ValueError("return human_visible must bind the final recipient")
    for previous, current in pairwise(scored):
        parents = {
            reference.milestone_ref
            for reference in current.preconditions
            if isinstance(reference, MilestoneReference)
        }
        if previous.milestone_id not in parents:
            raise ValueError(
                "Each canonical scored outcome must directly require the "
                "preceding canonical scored outcome"
            )


def _validate_base(
    mainline: TaskMainline,
    milestones: tuple[Any, ...],
) -> None:
    inquiry = (
        mainline is TaskMainline.NAVIGATE_AND_REPORT
        and milestones[2].outcome.evaluator == "information_received"
    )
    challenges = [
        milestone.milestone_id
        for milestone in milestones
        if (
            EVENT_TYPE_CATALOG[milestone.event.type].role in _CHALLENGE_CHANGE_ROLES
            or _is_agent_ask(milestone)
        )
    ]
    if challenges:
        raise ValueError(
            "base slice cannot require adaptation or a separate "
            f"Agent-initiated information-gap challenge: {challenges!r}"
        )
    influencing = _influencing_humans(milestones)
    maximum_influencing_humans = 3 if inquiry else 2
    if len(influencing) > maximum_influencing_humans:
        raise ValueError(
            "base slice has too many task-influencing Humans; each mainline "
            "permits one requester and one result recipient, plus an onsite "
            "information source for navigation inquiry, while "
            "unrelated bystanders stay "
            f"outside the DAG; got {sorted(influencing)!r}"
        )
    if len(influencing) < 2 and not inquiry:
        return
    requesters = {
        str(action["human"])
        for milestone in milestones
        if milestone.event.type.value == "task_requested"
        for action in milestone.event.human_actions
        if action.get("human")
    }
    terminal_evaluator = (
        "handover_accepted"
        if mainline is TaskMainline.FETCH_AND_DELIVER
        else "information_report_score"
    )
    recipients = {
        str(milestone.outcome.ground_truth["human_id"])
        for milestone in milestones
        if milestone.outcome.evaluator == terminal_evaluator
        and milestone.outcome.ground_truth.get("human_id")
    }
    if inquiry:
        discovery, receipt, return_visibility = milestones[1:4]
        source = str(receipt.outcome.ground_truth.get("human_id") or "")
        if not source or discovery.outcome.ground_truth.get("human_id") != source:
            raise ValueError(
                "navigation/base inquiry discovery human_visible must bind "
                "the information_received source Human"
            )
        if (
            len(recipients) != 1
            or return_visibility.outcome.ground_truth.get("human_id") not in recipients
        ):
            raise ValueError(
                "navigation/base inquiry return human_visible must bind "
                "the final information_report_score recipient"
            )
        if len(requesters) != 1:
            raise ValueError(
                "navigation/base inquiry requires exactly one task requester"
            )
        if influencing != requesters | recipients | {source}:
            raise ValueError(
                "navigation/base inquiry permits no task-influencing Human "
                "beyond the requester, onsite source, and result recipient"
            )
        return
    if len(requesters) != 1 or len(recipients) != 1:
        raise ValueError(
            f"{mainline.value}/base with two Humans requires exactly one "
            "task requester and one final result recipient"
        )
    if influencing != requesters | recipients:
        raise ValueError(
            f"{mainline.value}/base permits no task-influencing Human beyond "
            "the requester and final result recipient"
        )


def _is_agent_ask(milestone: Any) -> bool:
    return milestone.outcome.evaluator == "agent_interaction_selected"


def _has_downstream_result(
    milestone_id: str,
    milestones: tuple[Any, ...],
) -> bool:
    return any(
        milestone.milestone_id != milestone_id
        and _is_ancestor(milestone_id, milestone.milestone_id, milestones)
        for milestone in milestones
    )


def _milestone_parents(milestones: tuple[Any, ...]) -> dict[str, set[str]]:
    return {
        milestone.milestone_id: {
            item.milestone_ref
            for item in milestone.preconditions
            if isinstance(item, MilestoneReference)
        }
        for milestone in milestones
    }


def _is_ancestor(
    ancestor_id: str,
    descendant_id: str,
    milestones: tuple[Any, ...],
) -> bool:
    parents = _milestone_parents(milestones)
    frontier = list(parents.get(descendant_id, ()))
    visited: set[str] = set()
    while frontier:
        current = frontier.pop()
        if current == ancestor_id:
            return True
        if current in visited:
            continue
        visited.add(current)
        frontier.extend(parents.get(current, ()))
    return False


def _human_contribution_nodes(
    human_id: str,
    milestones: tuple[Any, ...],
) -> set[str]:
    return {
        milestone.milestone_id
        for milestone in milestones
        if (
            any(
                str(action.get("human") or "") == human_id
                for action in milestone.event.human_actions
            )
            or str(milestone.outcome.ground_truth.get("human_id") or "") == human_id
        )
    }


def _has_common_contributor_outcome(
    contributor_ids: tuple[str, ...],
    milestones: tuple[Any, ...],
) -> bool:
    contributions = {
        human_id: _human_contribution_nodes(human_id, milestones)
        for human_id in contributor_ids
    }
    for outcome in milestones:
        outcome_id = outcome.milestone_id
        if all(
            any(
                node_id == outcome_id or _is_ancestor(node_id, outcome_id, milestones)
                for node_id in node_ids
            )
            for node_ids in contributions.values()
        ):
            return True
    return False


def _influencing_humans(milestones: tuple[Any, ...]) -> set[str]:
    values: set[str] = set()
    for milestone in milestones:
        for condition in milestone.preconditions:
            if isinstance(condition, MilestoneCondition):
                human_id = condition.arguments.get("human_id")
                if human_id:
                    values.add(str(human_id))
        for action in milestone.event.human_actions:
            if action.get("human"):
                values.add(str(action["human"]))
        human_id = milestone.outcome.ground_truth.get("human_id")
        if human_id:
            values.add(str(human_id))
    return values


def _contributing_humans(milestones: tuple[Any, ...]) -> set[str]:
    values: set[str] = set()
    for milestone in milestones:
        values.update(
            str(action["human"])
            for action in milestone.event.human_actions
            if action.get("human")
        )
        human_id = milestone.outcome.ground_truth.get("human_id")
        if human_id:
            values.add(str(human_id))
    return values


__all__ = ["validate_episode_task_profile"]
