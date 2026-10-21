"""Gripper profile registry: what one end effector can physically take.

A grasp plan needs facts about the hand that are neither episode data nor
solver settings: how wide the jaws open, where the closing centre sits relative
to the wrist the arm solver drives, and how far the joints must travel before
the closure counts. Those facts live here, one :class:`GripperProfile` per end
effector, so a canonical grasp can be checked against the hand actually mounted
instead of against a constant nobody re-derives when the hand changes.

``physical_contracts.TARGET_SCREENING_MAX_GRASP_SPAN_M`` already carried a note
that offline screening's span bound is embodiment-neutral and that "live
certification must apply the bound from the selected gripper profile". This
module is that profile, and the screening bound is now derived from it rather
than authored twice.

Only the Dex3-1 profile ships today, because it is the only end effector whose
grasp path is certified. The G1 grasp code names it explicitly; a parallel jaw
for the G1 needs a new USD, a new contact-link set, and full re-certification,
so wiring a robot-to-gripper selection is left to the change that adds one
rather than guessed at now.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

from tarsphere.embodiment.dex3 import (
    DEX3_GRASP_CENTER_FROM_PALM_M,
    DEX3_MAXIMUM_OPPOSING_NORMAL_DOT,
    DEX3_MINIMUM_CLOSING_AXIS_NORMAL_PROJECTION,
    DEX3_MINIMUM_CLOSURE_L2_RAD,
    DEX3_MINIMUM_NORMAL_IMPULSE_NS,
    DEX3_MINIMUM_OPPOSING_CONTACT_SPAN_M,
)


@dataclass(frozen=True)
class GripperProfile:
    """One end effector's physical grasp envelope."""

    # Canonical profile name, as named by the grasp path that mounts it.
    name: str
    # Widest object span the jaws can close around, in metres. A canonical
    # grasp asking for more than this cannot be executed by this hand.
    maximum_jaw_span_m: float
    # Closing-envelope centre in the frame of the link the arm solver drives.
    # The arm targets a grasp position, so the solver offsets by this to place
    # the wrist; it is a property of the hand, not of any one object.
    grasp_center_from_wrist_m: Tuple[float, float, float]
    # Joint travel away from the neutral pose below which a commanded closure
    # has not actually happened, so attachment must not be granted.
    minimum_closure_l2_rad: float
    # Substrings identifying the contact links that must oppose each other for
    # a closure to be a grasp: at least one from each group on the same target.
    # Naming the roles rather than the links keeps this readable when a hand
    # has several segments per finger.
    opposing_contact_roles: Tuple[Tuple[str, ...], ...]
    # Smallest measured span between the two contact roles along the closing
    # axis, in metres.
    minimum_opposing_contact_span_m: float
    # Largest dot product admitted between the two target-to-hand normals.
    maximum_opposing_normal_dot: float
    # Smallest magnitude each normal must retain along the closing axis.
    minimum_closing_axis_normal_projection: float
    # Smallest normal impulse measured for each role in one contact window.
    minimum_normal_impulse_ns: float
    # Human-readable provenance for docs and error messages.
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a gripper profile needs a name")
        if not math.isfinite(self.maximum_jaw_span_m) or self.maximum_jaw_span_m <= 0.0:
            raise ValueError(f"{self.name} needs a positive finite maximum jaw span")
        if len(self.grasp_center_from_wrist_m) != 3 or not all(
            math.isfinite(value) for value in self.grasp_center_from_wrist_m
        ):
            raise ValueError(f"{self.name} grasp centre must be three finite metres")
        if (
            not math.isfinite(self.minimum_closure_l2_rad)
            or self.minimum_closure_l2_rad <= 0.0
        ):
            raise ValueError(
                f"{self.name} needs a positive closure threshold; zero would "
                "let an open hand claim a grasp"
            )
        if len(self.opposing_contact_roles) < 2:
            raise ValueError(
                f"{self.name} needs at least two opposing contact groups; one "
                "group cannot pinch anything"
            )
        if (
            not math.isfinite(self.minimum_opposing_contact_span_m)
            or self.minimum_opposing_contact_span_m <= 0.0
        ):
            raise ValueError(f"{self.name} needs a positive opposing-contact span")
        if not -1.0 <= self.maximum_opposing_normal_dot < 0.0:
            raise ValueError(f"{self.name} needs a negative opposing-normal dot limit")
        if not 0.0 < self.minimum_closing_axis_normal_projection <= 1.0:
            raise ValueError(
                f"{self.name} needs a closing-axis normal projection in (0, 1]"
            )
        if (
            not math.isfinite(self.minimum_normal_impulse_ns)
            or self.minimum_normal_impulse_ns <= 0.0
        ):
            raise ValueError(f"{self.name} needs a positive normal-impulse threshold")

    def admits_span(self, object_span_m: float) -> bool:
        """Whether this hand can close around ``object_span_m`` metres."""
        span = float(object_span_m)
        if not math.isfinite(span) or span < 0.0:
            raise ValueError("an object span must be finite and non-negative")
        return span <= self.maximum_jaw_span_m


DEX3_1 = GripperProfile(
    name="dex3_1",
    # Carried forward from the certified offline screening envelope, which is
    # the widest span the promoted grasp evidence covers. It is a reviewed
    # authoring bound, not a URDF-derived kinematic maximum: the certified
    # golden targets are 0.07-0.08 m bodies, so anything approaching this
    # bound is admitted by screening and still has to pass live contact.
    maximum_jaw_span_m=0.18,
    grasp_center_from_wrist_m=DEX3_GRASP_CENTER_FROM_PALM_M,
    minimum_closure_l2_rad=DEX3_MINIMUM_CLOSURE_L2_RAD,
    opposing_contact_roles=(("_thumb_",), ("_index_", "_middle_")),
    minimum_opposing_contact_span_m=(DEX3_MINIMUM_OPPOSING_CONTACT_SPAN_M),
    maximum_opposing_normal_dot=DEX3_MAXIMUM_OPPOSING_NORMAL_DOT,
    minimum_closing_axis_normal_projection=(
        DEX3_MINIMUM_CLOSING_AXIS_NORMAL_PROJECTION
    ),
    minimum_normal_impulse_ns=DEX3_MINIMUM_NORMAL_IMPULSE_NS,
    description=(
        "Unitree Dex3-1 three-finger hand, seven joints. Fingers extend along "
        "palm +X and close across palm Y. A grasp requires thumb contact plus "
        "index-or-middle contact on the same target."
    ),
)

GRIPPER_PROFILES: Dict[str, GripperProfile] = {DEX3_1.name: DEX3_1}


def get_gripper_profile(name: str) -> GripperProfile:
    """Look up a gripper profile by name. Fail loud if unknown."""
    profile = GRIPPER_PROFILES.get(name)
    if profile is None:
        raise KeyError(
            f"Unknown gripper profile {name!r}. Known grippers: "
            f"{sorted(GRIPPER_PROFILES)}. Add a GripperProfile entry in "
            "tarsphere.embodiment.gripper_profiles to support a new end "
            "effector."
        )
    return profile


__all__ = [
    "DEX3_1",
    "GRIPPER_PROFILES",
    "GripperProfile",
    "get_gripper_profile",
]
