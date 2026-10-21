"""Unitree Dex3-1 kinematic and physical contract for the G1."""

from __future__ import annotations

import math
from typing import Literal, Sequence


Dex3Hand = Literal["left", "right"]

# TARSphere evaluates manipulation through one physically certified work hand.
# The left Dex3 remains part of the G1 body but never selects or executes a
# benchmark grasp, carry, or placement action.
G1_MANIPULATION_HAND: Dex3Hand = "right"

# Mounted links retain the donor Dex3's authored mass and inertia so its
# articulated fingers remain numerically well-conditioned. Gravity is disabled
# on those links by the composed G1 asset, removing their static wrist load
# without replacing real rigid-body dynamics with near-singular inertia.
DEX3_MINIMUM_LINK_MASS_KG = 1e-3
DEX3_MINIMUM_DIAGONAL_INERTIA_KG_M2 = 1e-8
# Finite rubber-like contact coefficients used by the friction-only transport
# setting. They alter no mass, joint, or locomotion-policy dimension.
DEX3_STATIC_FRICTION = 2.0
DEX3_DYNAMIC_FRICTION = 1.8

# The arm terminates at the palm origin, while a grasp is formed between the
# thumb and the two opposing fingers.  This palm-frame point follows directly
# from the Dex3 proximal finger bases (x≈0.078 m) and the first phalanx length
# (≈0.046 m); it is the centre of the closing envelope, not a per-object
# waypoint.
DEX3_GRASP_CENTER_FROM_PALM_M = (0.10, 0.0, -0.005)

# Per-hand order follows Unitree's Dex3-1 API and the Psi0 action contract.
DEX3_JOINT_NAMES: dict[Dex3Hand, tuple[str, ...]] = {
    hand: tuple(
        f"{hand}_hand_{finger}_{segment}_joint"
        for finger, segment_count in (
            ("thumb", 3),
            ("middle", 2),
            ("index", 2),
        )
        for segment in range(segment_count)
    )
    for hand in ("left", "right")
}

# The palm and all seven articulated finger links are the only links allowed
# to establish grasp contact for one hand.
DEX3_LINK_NAMES: dict[Dex3Hand, tuple[str, ...]] = {
    hand: (
        f"{hand}_hand_palm_link",
        *(
            f"{hand}_hand_{finger}_{segment}_link"
            for finger, segment_count in (
                ("thumb", 3),
                ("middle", 2),
                ("index", 2),
            )
            for segment in range(segment_count)
        ),
    )
    for hand in ("left", "right")
}

DEX3_PALM_LINK_NAMES: dict[Dex3Hand, str] = {
    hand: f"{hand}_hand_palm_link" for hand in ("left", "right")
}

# Mechanical stops from the Dex3 donor
# ``assets/robots/g1_with_hand/g1_29dof_with_hand_rev_1_0.xml``, in
# ``DEX3_JOINT_NAMES`` order (thumb 0-2, middle 0-1, index 0-1).
DEX3_JOINT_LOWER: dict[Dex3Hand, tuple[float, ...]] = {
    "left": (-1.0472, -0.724312, 0.0, -1.5708, -1.74533, -1.5708, -1.74533),
    "right": (-1.0472, -1.0472, -1.74533, 0.0, 0.0, 0.0, 0.0),
}
DEX3_JOINT_UPPER: dict[Dex3Hand, tuple[float, ...]] = {
    "left": (1.0472, 1.0472, 1.74533, 0.0, 0.0, 0.0, 0.0),
    "right": (1.0472, 0.724312, 0.0, 1.5708, 1.74533, 1.5708, 1.74533),
}

# These poses are the released GraspGenX ``unitree_g1`` open/close states,
# clamped only where the newer Dex3-1 donor has a slightly narrower URDF
# limit.  Using the same hand state for generation and execution keeps the
# model's swept finger volume physically meaningful. The left hand is the
# sagittal mirror of the released right-hand descriptor.
DEX3_OPEN_JOINT_POSITIONS: dict[Dex3Hand, tuple[float, ...]] = {
    "left": (0.0, -0.60, 0.0, -0.40, 0.0, -0.40, 0.0),
    "right": (0.0, 0.60, 0.0, 0.40, 0.0, 0.40, 0.0),
}
DEX3_GRASP_JOINT_POSITIONS: dict[Dex3Hand, tuple[float, ...]] = {
    "left": (0.0, 1.047, 1.50, -1.571, -1.745, -1.571, -1.745),
    "right": (0.0, -1.047, -1.50, 1.571, 1.745, 1.571, 1.745),
}

# Attach is legal only after the measured hand configuration has moved a
# meaningful distance away from the neutral pose.
DEX3_MINIMUM_CLOSURE_L2_RAD = 0.50

# A real pinch needs contacts on opposite sides of the generated closing
# centre, not merely two named finger links touching the same surface.  Five
# millimetres is a numerical lower bound on that measured contact span; it is
# independent of object class and far below every asset admitted by the hand's
# 0.18 m screening envelope.
DEX3_MINIMUM_OPPOSING_CONTACT_SPAN_M = 0.005
# The two object-to-finger surface normals must oppose by at least 120 degrees.
# This rejects a thumb and finger both pressing down on a rim while retaining
# the ordinary range of non-ideal mesh contacts around a curved object.
DEX3_MAXIMUM_OPPOSING_NORMAL_DOT = -0.50
# Each surface normal must carry at least half of its unit magnitude along the
# GraspGenX closing axis.  Contacts dominated by the approach or gravity axis
# are support contacts, not a pinch across the hand.
DEX3_MINIMUM_CLOSING_AXIS_NORMAL_PROJECTION = 0.50
# PhysX impulses below this scale are numerically zero for the grasp contract.
# The threshold does not score grasp strength; it only distinguishes a solver
# contact that exerted force during the authored closure window from a dormant
# persistent contact point.
DEX3_MINIMUM_NORMAL_IMPULSE_NS = 1e-8

# A transient first contact is not a physical grasp.  Contact-only trials must
# preserve thumb/opposing-finger contact for this many consecutive 240 Hz
# physics ticks while the position controller continues closing the hand.
DEX3_CONTACT_SETTLE_PHYSICS_STEPS = 24
# Closure ramps the seven commands from open to grasp across its window, so the
# fingers only receive the full command on its final tick and are still moving
# when it ends.  Retention must be measured on a hand that has stopped, not on
# a snapshot taken mid-close, so the closed command is held until the measured
# joints settle.  This is a physical settling criterion on real geometry; it
# loosens no contact, span, or closure threshold.
DEX3_CLOSURE_SETTLE_CONVERGENCE_RAD = 0.01
# Bound on that hold. The longest released Dex3 stroke is the thumb's
# 0.60 -> -1.047 rad travel; three seconds at 240 Hz is several times the
# free-space traversal and keeps a jammed hand from consuming the episode.
DEX3_CLOSURE_SETTLE_MAX_PHYSICS_STEPS = 720
# Closing must happen after the open hand reaches the generated pose.  The
# thumb is the slowest released Dex3 joint: its 1.047 rad stroke at 3.14 rad/s
# needs 81 ticks at 240 Hz.  Ninety-six ticks provide one controller interval
# of margin without changing the commanded joint target.
DEX3_CLOSURE_PHYSICS_STEPS = 96
# Opening traverses the same largest joint stroke and therefore uses the same
# 240 Hz transition duration.  Keeping it explicit prevents a symbolic release
# from being mistaken for an instantaneous loss of finger support.
DEX3_OPENING_PHYSICS_STEPS = 96
# A completed release is measured from the live articulation, not inferred
# from the command sent to the hand controller.  This joint-space tolerance
# admits ordinary position-controller tracking error while still requiring
# every finger to reach the released open configuration.
DEX3_RELEASE_OPENING_TOLERANCE_RAD = 0.12


def dex3_controller_name(hand: Dex3Hand) -> str:
    """Return the TARSphere joint controller owned by one Dex3 hand."""
    return f"{hand}_hand_joint_controller"


def waist_yaw_controller_name() -> str:
    """Return the TARSphere joint controller for VLA waist yaw.

    Ordinary G1 episodes command yaw through the 15-D locomotion upper
    target. Psi0 sends a position target here instead of that scaled slot.
    """
    return "waist_yaw_joint_controller"


def waist_pitch_controller_name() -> str:
    """Return the TARSphere joint controller for G1 waist pitch."""
    return "waist_pitch_joint_controller"


def waist_roll_controller_name() -> str:
    """Return the TARSphere joint controller for VLA waist roll.

    Ordinary G1 episodes do not register this controller. The certified
    composition keeps ``waist_roll_joint`` as a FixedJoint held at 0.
    """
    return "waist_roll_joint_controller"


def arm_controller_name(hand: Dex3Hand) -> str:
    """Return the TARSphere joint controller for one G1 arm.

    Grasp/carry sends positions through the reviewed 15-D locomotion upper
    target. The public local right-palm action and Psi0 use direct position
    control because neither should move the waist or contralateral arm.
    """
    return f"{hand}_arm_joint_controller"


def dex3_joint_positions(
    hand: Dex3Hand,
    preset: Literal["open", "grasp"],
) -> tuple[float, ...]:
    """Return the reviewed seven-joint command for one named hand preset."""
    if preset == "open":
        return DEX3_OPEN_JOINT_POSITIONS[hand]
    if preset == "grasp":
        return DEX3_GRASP_JOINT_POSITIONS[hand]
    raise ValueError(f"Unknown Dex3 hand preset: {preset!r}")


def dex3_continuous_joint_positions(
    hand: Dex3Hand,
    closure: float,
) -> tuple[float, ...]:
    """Interpolate the released Dex3 open/grasp targets continuously.

    ``closure=0`` is fully open and ``closure=1`` is the reviewed grasp
    target.  The interpolation stays inside every joint's mechanical limits
    because both endpoints are already certified and each coordinate follows
    their line segment.
    """
    value = float(closure)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(
            f"Dex3 closure must be finite and lie in [0, 1], got {closure!r}."
        )
    if value == 0.0:
        return DEX3_OPEN_JOINT_POSITIONS[hand]
    if value == 1.0:
        return DEX3_GRASP_JOINT_POSITIONS[hand]
    opened = DEX3_OPEN_JOINT_POSITIONS[hand]
    grasped = DEX3_GRASP_JOINT_POSITIONS[hand]
    return tuple(start + value * (end - start) for start, end in zip(opened, grasped))


def dex3_continuous_closure(
    hand: Dex3Hand,
    joint_positions: Sequence[float],
) -> float:
    """Project a measured hand state onto the public open-to-grasp coordinate."""

    measured = tuple(float(value) for value in joint_positions)
    opened = DEX3_OPEN_JOINT_POSITIONS[hand]
    closed = DEX3_GRASP_JOINT_POSITIONS[hand]
    if len(measured) != len(opened):
        raise ValueError(
            f"Dex3 {hand} state must contain {len(opened)} joints; got {len(measured)}."
        )
    if not all(math.isfinite(value) for value in measured):
        raise ValueError("Dex3 continuous-closure state must be finite.")
    stroke = tuple(end - start for start, end in zip(opened, closed))
    denominator = sum(value * value for value in stroke)
    if denominator <= 0.0:
        raise RuntimeError(f"Dex3 {hand} open/grasp targets are degenerate.")
    projected = (
        sum(
            (value - start) * direction
            for value, start, direction in zip(measured, opened, stroke)
        )
        / denominator
    )
    return min(1.0, max(0.0, float(projected)))


def dex3_closure_l2(joint_positions: Sequence[float]) -> float:
    """Measure hand closure as L2 distance from the neutral seven-joint pose."""
    positions = tuple(float(value) for value in joint_positions)
    if len(positions) != 7:
        raise ValueError(
            "Dex3 joint state must contain exactly seven positions; "
            f"got {len(positions)}."
        )
    if not all(math.isfinite(value) for value in positions):
        raise ValueError("Dex3 joint state must be finite.")
    return math.sqrt(sum(value * value for value in positions))


__all__ = [
    "DEX3_GRASP_JOINT_POSITIONS",
    "DEX3_GRASP_CENTER_FROM_PALM_M",
    "DEX3_JOINT_LOWER",
    "DEX3_JOINT_UPPER",
    "DEX3_CONTACT_SETTLE_PHYSICS_STEPS",
    "DEX3_CLOSURE_PHYSICS_STEPS",
    "DEX3_CLOSURE_SETTLE_CONVERGENCE_RAD",
    "DEX3_CLOSURE_SETTLE_MAX_PHYSICS_STEPS",
    "DEX3_OPENING_PHYSICS_STEPS",
    "DEX3_RELEASE_OPENING_TOLERANCE_RAD",
    "DEX3_JOINT_NAMES",
    "DEX3_MINIMUM_DIAGONAL_INERTIA_KG_M2",
    "DEX3_MINIMUM_LINK_MASS_KG",
    "DEX3_LINK_NAMES",
    "DEX3_MINIMUM_CLOSURE_L2_RAD",
    "DEX3_MINIMUM_CLOSING_AXIS_NORMAL_PROJECTION",
    "DEX3_MINIMUM_NORMAL_IMPULSE_NS",
    "DEX3_MINIMUM_OPPOSING_CONTACT_SPAN_M",
    "DEX3_MAXIMUM_OPPOSING_NORMAL_DOT",
    "DEX3_OPEN_JOINT_POSITIONS",
    "DEX3_PALM_LINK_NAMES",
    "DEX3_STATIC_FRICTION",
    "DEX3_DYNAMIC_FRICTION",
    "Dex3Hand",
    "G1_MANIPULATION_HAND",
    "dex3_closure_l2",
    "dex3_continuous_closure",
    "dex3_continuous_joint_positions",
    "arm_controller_name",
    "dex3_controller_name",
    "dex3_joint_positions",
    "waist_pitch_controller_name",
    "waist_roll_controller_name",
    "waist_yaw_controller_name",
]
