"""Compile authored intent interpretations over immutable paired Base episodes.

Semantics are authored in a reviewed design file. This module owns only binding,
receipt placement and paired-outcome invariants, not story generation.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.bench.schema import TARSphereEpisode

_CONTEXT_INFERENCE_REPLY = "You should know what I mean."


class IntentVariantDesign(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    subtype: Literal["preference_confirmation", "requirement_clarification"] = Field(
        default="requirement_clarification",
        description="The semantic distinction being resolved, independent of ASK/CONTEXT mode.",
    )
    topic: str = Field(
        default="Clarify the requester's intended task or recipient.",
        min_length=1,
        description="Private NPC routing description for a request-interpretation question; describe the question without labelling the correct answer.",
    )
    clarification_intent_id: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Episode-specific private routing ID, present only when clarification is required.",
    )
    mode: Literal["clarification_required", "context_inferable"] = Field(
        description="Whether the missing interpretation requires an answer or follows from public evidence."
    )
    locus: Literal["A", "B", "C"] = Field(
        description="Decision locus: dispatch, onsite execution, or final interaction; not a forced query time."
    )
    request: str = Field(
        min_length=1,
        description="Opening request without the preserved Base route cue.",
    )
    resolution: str = Field(
        min_length=1,
        description="Authorized clarification; never contains the inspection/inquiry answer.",
    )
    correct: str = Field(
        min_length=1, description="Interpretation preserving the Base task outcome."
    )
    alternative: str = Field(
        min_length=1,
        description="Plausible competing interpretation requiring a different action or report.",
    )
    clue: str | None = Field(
        default=None,
        min_length=1,
        description="Public evidence for context-inferable tasks, absent for must-ask tasks.",
    )
    evidence_source: Literal["dialogue_context", "public_profile"] | None = Field(
        default=None,
        description="Where the inferable clue is actually supplied to the Agent.",
    )
    clarification_human_id: str | None = Field(
        default=None,
        min_length=1,
        description="For a final-interaction decision, the authored Human who knows the missing requirement.",
    )
    deferred_route_cue: str | None = Field(
        default=None,
        min_length=1,
        description="Exact trailing Base route instruction revealing the final disposition. Omit from the opening; ASK supplies it with clarification, CONTEXT resolves the destination through its public clue.",
    )

    @model_validator(mode="after")
    def validate_evidence(self):
        inferable = self.mode == "context_inferable"
        if inferable != bool(self.clue):
            raise ValueError(
                "Only context-inferable variants have a public resolving clue"
            )
        if inferable != bool(self.evidence_source):
            raise ValueError(
                "Only context-inferable variants declare their public evidence channel"
            )
        if inferable and self.clarification_intent_id:
            raise ValueError(
                "Context-inferable variants do not declare a private clarification intent ID"
            )
        if self.clarification_intent_id == "request_meaning":
            raise ValueError(
                "Use a topic-specific private intent ID, not the batch-wide request_meaning shortcut"
            )
        if self.correct.strip() == self.alternative.strip():
            raise ValueError("Intent alternatives must change a task decision")
        if (self.locus == "C") != bool(self.clarification_human_id):
            raise ValueError(
                "A final-interaction decision must name its clarification source"
            )
        return self


def _ref(name: str) -> dict:
    return {"kind": "milestone", "milestone_ref": name}


def compile_intent_variant(base: dict, design: IntentVariantDesign) -> dict:
    """Preserve physical bindings and all five outcome standards; add no point."""
    parsed_base = TARSphereEpisode.model_validate(base)
    if parsed_base.evaluation.task_profile.capability_slice.value != "base":
        raise ValueError("IU compilation requires a Base episode")
    data = deepcopy(base)
    data["episode_id"] = (
        base["episode_id"].removesuffix("_base") + "_intent_understanding"
    )
    data["title"] = base["title"] + " / intent understanding"
    # Description is private metadata, not additional public instruction.
    data["description"] = design.correct
    nodes = data["evaluation"]["milestones"]
    by_id = {node["milestone_id"]: node for node in nodes}
    opening = by_id["task_understanding"]["event"]["human_actions"][0]
    requester = opening["human"]
    affected = "task_completed" if design.locus == "C" else "onsite_task_completed"
    source = design.clarification_human_id if design.locus == "C" else requester
    if source not in {human["id"] for human in data["humans"]}:
        raise ValueError(f"Unknown clarification Human: {source}")
    if (
        design.locus == "C"
        and parsed_base.evaluation.task_profile.mainline.value != "fetch_and_deliver"
    ):
        raise ValueError(
            "This batch only authors final-handover interpretation at C; no late inspection requirements"
        )
    # Preserve route guidance, not the Base question sometimes appended to it.
    # An inquiry topic reveals the intended interpretation just as an inspection
    # criterion does, even when its intent-directory entry is private.
    _, separator, route = opening["message"].partition("From your starting position")
    if not separator:
        raise ValueError(f"{base['episode_id']}: expected reviewed initial route cue")
    onsite = by_id["onsite_task_completed"]["reaction"]
    question = None
    if onsite["evaluator"] == "inspection_submitted":
        question = onsite["ground_truth"]["criterion"]
    elif onsite["evaluator"] == "information_received":
        question = next(
            intent["description"]
            for intent in data["interaction"]["intents"]
            if intent["intent_id"] == onsite["ground_truth"]["intent_id"]
        )
    if question and route.endswith(question):
        route = route[: -len(question)].rstrip()
    if design.deferred_route_cue:
        if not route.endswith(design.deferred_route_cue):
            raise ValueError(
                "Deferred route cue must be the exact trailing Base instruction"
            )
        route = route[: -len(design.deferred_route_cue)].rstrip()
        if route.endswith((",", ";")):
            route = route[:-1] + "."
    # Spoken egocentric guidance is relative to the speaker. Keep the reviewed
    # route itself while normalizing older Base prose that named the speaker.
    route = re.sub(r"^\s*facing [^,]+,", " facing me,", route)
    opening["message"] = f"{design.request} {separator}{route}"
    profile = {
        "mainline": parsed_base.evaluation.task_profile.mainline.value,
        "capability_slice": "intent_understanding",
        "subtype": design.subtype,
        "resolution_mode": design.mode,
        "affected_outcome_id": affected,
        "resolution_options": [
            {"option_id": "intended", "description": design.correct},
            {"option_id": "alternative", "description": design.alternative},
        ],
        "resolved_option_id": "intended",
        "gap_cues": [design.request],
    }
    intent_id = design.clarification_intent_id or (
        f"{base['episode_id'].removesuffix('_base')}_clarification"
        if design.mode == "clarification_required"
        else f"{base['episode_id'].removesuffix('_base')}_context_deflection"
    )
    intents = data.setdefault("interaction", {}).setdefault("intents", [])
    onsite_intent = None
    if onsite["evaluator"] == "information_received":
        onsite_intent_id = onsite["ground_truth"]["intent_id"]
        onsite_intent = next(
            (intent for intent in intents if intent["intent_id"] == onsite_intent_id),
            None,
        )
        if onsite_intent is None:
            raise ValueError(f"Missing onsite inquiry intent: {onsite_intent_id}")
        # An inquiry's exact requested topic is part of the IU decision and
        # therefore private. Other inherited public intents (including route
        # guidance) keep their Base visibility.
        onsite_intent["public"] = False
    if design.mode == "clarification_required":
        if any(intent["intent_id"] == intent_id for intent in intents):
            raise ValueError("Base already uses the variant clarification intent")
        # Keep unrelated inherited public guidance visible. The original scored
        # inquiry topic is private until its requested subject is clarified.
        intents.append(
            {
                "intent_id": intent_id,
                "public": False,
                "description": design.topic,
                "eligible_human_ids": [source],
                "answer": {
                    "kind": "authored_text",
                    "text": " ".join(
                        text
                        for text in (design.resolution, design.deferred_route_cue)
                        if text
                    ),
                },
            }
        )
        event_id = "intent_clarified"
        profile.update(
            fact_id=intent_id, source_human_id=source, interaction_milestone_id=event_id
        )
        event = {
            "milestone_id": event_id,
            "kind": "event",
            "preconditions": [_ref("task_understanding")],
            "event": {
                "type": "task_phase_opened",
                "description": f"{source} can clarify the request when addressed.",
                "human_actions": [],
                "task_updates": [],
            },
            "completion": {
                "description": f"The intended clarification is actually delivered by {source}.",
                "evaluator": "information_received",
                "ground_truth": {"human_id": source, "intent_id": intent_id},
            },
        }
        insertion = 4 if design.locus == "C" else 1
        # At C the receipt is independent of onsite work, but is displayed just
        # before the final outcome. Its sole task dependency remains the request.
        nodes.insert(insertion, event)
        by_id[affected]["preconditions"].append(_ref(event_id))
        if onsite["evaluator"] == "information_received":
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
            onsite_intent.setdefault("preconditions", []).append(_ref(event_id))
        route_point = (
            "completion_stance"
            if design.locus == "C" and source != requester
            else "task_start"
        )
        data["world_knowledge"]["episode_route"]["milestone_stops"].insert(
            insertion, {"milestone_id": event_id, "route_point_id": route_point}
        )
    else:
        if any(intent["intent_id"] == intent_id for intent in intents):
            raise ValueError("Base already uses the variant context-deflection intent")
        # Do not disclose or confirm the resolved option. The requester only
        # directs the Agent back to its own contextual inference; this response
        # creates no receipt and gates no outcome.
        intents.append(
            {
                "intent_id": intent_id,
                "public": False,
                "description": design.topic,
                "eligible_human_ids": [source],
                "answer": {"kind": "authored_text", "text": _CONTEXT_INFERENCE_REPLY},
            }
        )
        if design.evidence_source == "public_profile":
            human = next(h for h in data["humans"] if h["id"] == source)
            human.setdefault("public_profile_facts", []).append(
                {"fact_id": "request_context", "text": design.clue}
            )
            evidence = {
                "source": "public_profile",
                "owner_id": source,
                "fact_id": "request_context",
            }
        else:
            # Dialogue is an actual delivered utterance, not private author notes.
            opening["message"] = f"{design.request} {design.clue} {separator}{route}"
            evidence = {"source": "dialogue_context", "owner_id": "task_understanding"}
        profile.update(public_evidence=[evidence], discriminating_cues=[design.clue])
    data["evaluation"]["task_profile"] = profile
    plan = by_id["task_understanding"]["reaction"]["ground_truth"]
    plan["goal"] = (
        "Resolve the request from the available conversation or public context, then complete the errand."
    )
    plan["required_subgoals"] = [
        f"Resolve the request using {source}'s answer or the supplied public evidence. A conditional plan is correct before missing information is disclosed.",
        "Use the supplied route cues to reach the named onsite task location.",
        "Perform the onsite action appropriate to the resolved request; do not invent unseen inspection results or unheard answers.",
        "Reach the named completion person and report or offer the item as appropriate to the resolved request. Do not require advance knowledge of a private clarification.",
    ]
    if design.locus == "C":
        plan["goal"] = (
            "Collect the named item and resolve the intended final interaction before completing it. "
            "Clarification may happen early or after travel; either ordering is correct."
        )
        plan["required_subgoals"] = [
            "Resolve the item's final disposition now or after travel, but before completing that final interaction.",
            "Use the supplied route cues to reach and collect the named item; collection does not wait on a final-disposition clarification.",
            "Reach the person or grounded destination selected by the resolved request.",
            "Complete the resolved final interaction; a conditional plan is correct before the missing information is available.",
        ]
    data["authoring"]["achieved_gates"] = ["authored", "schema_valid"]
    data["authoring"]["external_evidence"] = {}
    data["authoring"]["difficulty_factors"] = []
    data["metadata"].update(
        paired_base_episode_id=base["episode_id"],
        intent_decision_locus=design.locus,
        task_source="paired_intent_variants",
    )
    data["tags"] = [tag for tag in data.get("tags", []) if tag != "base"] + [
        "intent_understanding"
    ]
    validate_intent_pair(base, data)
    return data


def validate_intent_pair(base: dict, variant: dict) -> None:
    """No geometric, final-goal, evaluator or denominator drift is permitted."""
    episode = TARSphereEpisode.model_validate(variant)
    for key in ("scene", "robot", "runtime", "asset_adaptation"):
        if base.get(key) != variant.get(key):
            raise ValueError(f"IU changed paired Base {key}")
    for key in base["world_knowledge"]:
        left, right = (
            deepcopy(base["world_knowledge"][key]),
            deepcopy(variant["world_knowledge"][key]),
        )
        if key == "episode_route":
            left.pop("milestone_stops", None)
            right.pop("milestone_stops", None)
        if left != right:
            raise ValueError(f"IU changed paired Base world field {key}")
    physical_humans = lambda humans: [
        {
            k: v
            for k, v in h.items()
            if k not in ("public_profile_facts", "answer_facts")
        }
        for h in humans
    ]
    if physical_humans(base["humans"]) != physical_humans(variant["humans"]):
        raise ValueError("IU changed paired Base actor placement or identity")
    original = base["evaluation"]["milestones"]
    scored = [
        n
        for n in variant["evaluation"]["milestones"]
        if n.get("kind", "scored") != "event"
    ]
    if [n["milestone_id"] for n in original] != [
        n["milestone_id"] for n in scored
    ] or len(scored) != 5:
        raise ValueError("IU must preserve the five canonical outcome IDs")
    for old, new in zip(original, scored):
        if old["reaction"]["evaluator"] != new["reaction"]["evaluator"]:
            raise ValueError("IU changed a Base evaluator")
        if (
            old["milestone_id"] != "task_understanding"
            and old["reaction"] != new["reaction"]
        ):
            raise ValueError("IU changed a paired task-outcome standard")
    profile = episode.evaluation.task_profile
    if profile.resolution_mode == "context_inferable":
        response_id = f"{base['episode_id'].removesuffix('_base')}_context_deflection"
        responses = [
            intent
            for intent in episode.interaction.intents
            if intent.intent_id == response_id
        ]
        if (
            len(responses) != 1
            or responses[0].public
            or len(responses[0].eligible_human_ids) != 1
            or responses[0].preconditions
        ):
            raise ValueError(
                "Context-inferable IU requires one immediately available "
                "private non-revealing response from the responsible Human"
            )
    if episode.authoring.difficulty_factors:
        raise ValueError("Variants inherit Base difficulty without authored factors")
