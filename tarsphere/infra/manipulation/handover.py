"""Deterministic handover claim evaluation.

The robot's ``handover_object`` call is a spoken claim, not a motion. This
module decides acceptance from live geometry and episode identity, and owns
the agent-facing reply templates. Measured distances stay out of those
replies: a number would let the agent solve for the human's coordinates.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Collection, Mapping, Sequence

from tarsphere.infra.control.physical_contracts import (
    HANDOVER_MAX_DISTANCE_M,
    HANDOVER_MIN_DISTANCE_M,
)


REASON_ACCEPTED = "accepted"
REASON_TOO_FAR = "too_far"
REASON_TOO_CLOSE = "too_close"
REASON_NOT_HOLDING = "not_holding"
REASON_NOT_TRANSPORT_READY = "not_transport_ready"
REASON_TARGET_UNRESOLVED = "target_unresolved"
REASON_TARGET_NOT_VISIBLE = "target_not_visible"
REASON_ALREADY_ACCEPTED = "already_accepted"

HANDOVER_FEEDBACK = {
    REASON_ACCEPTED: "Thank you. I accept that handover offer.",
    REASON_TOO_FAR: "You are out of reach. Please come closer.",
    REASON_TOO_CLOSE: "You are uncomfortably close. Please back up slightly.",
    REASON_NOT_HOLDING: "You have nothing to give.",
    REASON_NOT_TRANSPORT_READY: "The object has not cleared its support.",
    REASON_TARGET_UNRESOLVED: "There is nobody here by that description.",
    REASON_TARGET_NOT_VISIBLE: (
        "I am not in your current view. Face me before offering the object."
    ),
    REASON_ALREADY_ACCEPTED: "I already accepted that handover.",
}

_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "in",
        "of",
        "to",
        "for",
        "and",
        "with",
        "this",
        "that",
        "person",
        "human",
        "someone",
        "who",
    }
)
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class HandoverHuman:
    """Episode-local identity used to resolve a free-text handover target."""

    id: str
    name: str = ""
    character_name: str = ""
    asset: str = ""
    role: str = ""
    descriptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class HandoverVerdict:
    """Evaluator record for one handover claim."""

    accepted: bool
    reason: str
    feedback: str
    target: str
    resolved_human_id: str | None
    object_id: str | None
    measured_distance_m: float | None

    def as_private_record(self) -> dict[str, object]:
        """Evaluator-only payload. Never copy this onto an agent observation."""

        return {
            "target": self.target,
            "resolved_human_id": self.resolved_human_id,
            "accepted": self.accepted,
            "measured_distance_m": self.measured_distance_m,
            "refusal_reason": None if self.accepted else self.reason,
            "object_id": self.object_id,
        }


def agent_facing_handover_replies() -> tuple[str, ...]:
    """Every reply that may reach the agent after a handover claim."""

    return tuple(HANDOVER_FEEDBACK.values())


def handover_feedback(reason: str) -> str:
    """Return the deterministic spoken reply for one refusal or acceptance."""

    try:
        return HANDOVER_FEEDBACK[reason]
    except KeyError as exc:
        raise ValueError(f"Unknown handover reason: {reason!r}") from exc


def horizontal_distance_m(
    first_xy: Sequence[float],
    second_xy: Sequence[float],
) -> float:
    """Return the finite planar distance between two world positions."""

    if len(first_xy) < 2 or len(second_xy) < 2:
        raise ValueError("handover distance requires x and y for both poses")
    first = (float(first_xy[0]), float(first_xy[1]))
    second = (float(second_xy[0]), float(second_xy[1]))
    if not all(math.isfinite(value) for value in (*first, *second)):
        raise ValueError("handover distance requires finite x and y")
    return math.dist(first, second)


def resolve_handover_target(
    target: str,
    humans: Sequence[HandoverHuman],
) -> str | None:
    """Return the unique matching human id, or None if the target does not resolve."""

    needle = str(target).strip()
    if not needle:
        return None
    matches = [
        human.id
        for human in humans
        if human.id and _human_matches_target(human, needle)
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def evaluate_handover_claim(
    *,
    target: str,
    humans: Sequence[HandoverHuman],
    held_object_id: str | None,
    object_transport_ready: bool,
    robot_xy: Sequence[float],
    human_positions: Mapping[str, Sequence[float]],
    visible_human_ids: Collection[str],
    accepted_handovers: Mapping[str, str],
) -> HandoverVerdict:
    """Evaluate one spoken handover claim against live state.

    Distance uses the live human position supplied in ``human_positions``.
    Authored spawn coordinates must not be placed in that map.
    ``object_transport_ready`` is the same live physical hold/support-separation
    result consumed by acquisition scoring, never a milestone or judge score.
    """

    claimed = str(target)
    human_id = resolve_handover_target(claimed, humans)
    if human_id is None:
        return _verdict(
            reason=REASON_TARGET_UNRESOLVED,
            target=claimed,
            resolved_human_id=None,
            object_id=None,
            measured_distance_m=None,
        )

    object_id = str(held_object_id).strip() if held_object_id is not None else ""
    if not object_id:
        return _verdict(
            reason=REASON_NOT_HOLDING,
            target=claimed,
            resolved_human_id=human_id,
            object_id=None,
            measured_distance_m=None,
        )
    if not object_transport_ready:
        return _verdict(
            reason=REASON_NOT_TRANSPORT_READY,
            target=claimed,
            resolved_human_id=human_id,
            object_id=object_id,
            measured_distance_m=None,
        )
    if human_id not in {str(value) for value in visible_human_ids}:
        return _verdict(
            reason=REASON_TARGET_NOT_VISIBLE,
            target=claimed,
            resolved_human_id=human_id,
            object_id=object_id,
            measured_distance_m=None,
        )
    # One object can have only one accepted handover claim. The recorded
    # recipient speaks the refusal so another Human cannot accept it later.
    if object_id in accepted_handovers:
        prior_recipient = str(accepted_handovers[object_id]).strip()
        if not prior_recipient:
            raise ValueError(
                f"accepted_handovers[{object_id!r}] has no recipient id: "
                f"{accepted_handovers[object_id]!r}"
            )
        return _verdict(
            reason=REASON_ALREADY_ACCEPTED,
            target=claimed,
            resolved_human_id=prior_recipient,
            object_id=object_id,
            measured_distance_m=None,
        )

    if human_id not in human_positions:
        raise KeyError(
            f"No live position for human {human_id!r}; known ids="
            f"{sorted(human_positions)}"
        )
    distance_m = horizontal_distance_m(robot_xy, human_positions[human_id])
    if distance_m < HANDOVER_MIN_DISTANCE_M:
        reason = REASON_TOO_CLOSE
    elif distance_m > HANDOVER_MAX_DISTANCE_M:
        reason = REASON_TOO_FAR
    else:
        reason = REASON_ACCEPTED
    return _verdict(
        reason=reason,
        target=claimed,
        resolved_human_id=human_id,
        object_id=object_id,
        measured_distance_m=distance_m,
    )


def _verdict(
    *,
    reason: str,
    target: str,
    resolved_human_id: str | None,
    object_id: str | None,
    measured_distance_m: float | None,
) -> HandoverVerdict:
    return HandoverVerdict(
        accepted=reason == REASON_ACCEPTED,
        reason=reason,
        feedback=handover_feedback(reason),
        target=target,
        resolved_human_id=resolved_human_id,
        object_id=object_id,
        measured_distance_m=measured_distance_m,
    )


def _human_matches_target(human: HandoverHuman, target: str) -> bool:
    needle = _folded(target)
    identity_fields = (
        human.id,
        human.name,
        human.character_name,
        human.asset,
        human.role,
    )
    # Exact equality keeps short ids that the token path drops. Unanchored
    # substrings are not used: role "man" would match target "woman".
    for field in identity_fields:
        folded = _folded(field)
        if folded and folded == needle:
            return True
    description_text = " ".join(
        _folded(item) for item in human.descriptions if str(item).strip()
    )
    if needle and needle in description_text:
        return True
    searchable = " ".join((*identity_fields, description_text))
    return bool(_tokens(target) & _tokens(searchable))


def _folded(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _tokens(text: str) -> frozenset[str]:
    return frozenset(
        token
        for token in _TOKEN_SPLIT.split(_folded(text))
        if len(token) >= 3 and token not in _STOPWORDS
    )


__all__ = [
    "HANDOVER_FEEDBACK",
    "HandoverHuman",
    "HandoverVerdict",
    "REASON_ACCEPTED",
    "REASON_ALREADY_ACCEPTED",
    "REASON_NOT_HOLDING",
    "REASON_TARGET_UNRESOLVED",
    "REASON_TARGET_NOT_VISIBLE",
    "REASON_TOO_CLOSE",
    "REASON_TOO_FAR",
    "agent_facing_handover_replies",
    "evaluate_handover_claim",
    "handover_feedback",
    "horizontal_distance_m",
    "resolve_handover_target",
]
