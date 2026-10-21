"""Single-source physical contracts shared by authoring, runtime, and eval.

These constants describe embodiment geometry or controller completion.  They
must not be reused for social-distance bands or milestone encounter radii;
those are episode semantics rather than physical safety contracts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from tarsphere.geometry import HUMAN_BODY_RADIUS_M
from tarsphere.embodiment.gripper_profiles import DEX3_1
from tarsphere.infra.action_errors import ActionRefusedError


# Isaac People uses 0.25 m as its final-target distance.  The scheduler and
# oracle must use the same authority or an injected Idle can stop a live GoTo.
HUMAN_GOTO_ARRIVAL_TOLERANCE_M = 0.25
# Released Human mesh facing differs from world +X-forward heading by +90°.
HUMAN_CHARACTER_YAW_OFFSET_RAD = math.pi / 2
# A newly spawned Human must still coincide with the authored world pose after
# the animation graph starts.  Larger disagreement means runtime-local motion
# has been mistaken for a world transform; collision and visibility evidence
# are invalid in that state.
HUMAN_INITIAL_POSE_TOLERANCE_M = 0.10
# Maximum displacement from the authored spawn during zero-command reset
# warmup. This validates initialization only; it is never an action veto.
G1_INITIAL_POSE_TOLERANCE_M = 0.10
# ``omni.anim.people`` plans every GoTo with this exact clearance radius.  A
# point-only NavMesh query without the radius can accept a route that the live
# character controller cannot execute.
HUMAN_NAVMESH_AGENT_RADIUS_M = 0.5

# Nominal lower-body policy target, in policy order. This is a kinematic
# reference pose, not a measured settled articulation or pelvis-height command.
G1_LOWER_POLICY_JOINT_NAMES = tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in (
        "hip_pitch",
        "hip_roll",
        "hip_yaw",
        "knee",
        "ankle_pitch",
        "ankle_roll",
    )
)
G1_DEFAULT_LOWER_DOF_POS_RAD = (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0) * 2

# Default-standing bounds measured from the released G1 USD. Coordinates are
# expressed in the robot frame: +X points forward and +Y points left. They are
# used for visualization and offline spatial certification, never as an online
# execution gate.
G1_FOOTPRINT_FRONT_M = 0.38
G1_FOOTPRINT_REAR_M = 0.10
G1_FOOTPRINT_HALF_WIDTH_M = 0.19
# Measured top-down G1 projection silhouette in the pelvis frame, metres,
# sampled around the released silhouette. LiDAR clearances, RGB-D robot-self
# filtering and occupancy planning share this exact embodiment geometry; it
# is never an online action veto.
G1_BODY_HULL_XY: tuple[tuple[float, float], ...] = (
    (0.376, -0.130),
    (0.376, -0.106),
    (0.353, -0.105),
    (0.339, -0.129),
    (0.321, -0.119),
    (0.301, -0.135),
    (0.249, -0.135),
    (0.235, -0.121),
    (0.173, -0.123),
    (0.151, -0.117),
    (0.125, -0.085),
    (0.074, -0.066),
    (0.084, -0.010),
    (0.074, 0.064),
    (0.123, 0.081),
    (0.151, 0.113),
    (0.195, 0.113),
    (0.251, 0.133),
    (0.301, 0.133),
    (0.319, 0.117),
    (0.337, 0.125),
    (0.353, 0.101),
    (0.376, 0.102),
    (0.376, 0.126),
    (0.357, 0.151),
    (0.287, 0.173),
    (0.175, 0.173),
    (0.115, 0.183),
    (0.103, 0.157),
    (0.083, 0.185),
    (-0.033, 0.183),
    (-0.066, 0.136),
    (-0.062, 0.094),
    (-0.078, 0.058),
    (-0.078, -0.060),
    (-0.062, -0.096),
    (-0.066, -0.140),
    (-0.046, -0.156),
    (-0.035, -0.185),
    (0.045, -0.189),
    (0.083, -0.187),
    (0.103, -0.161),
    (0.108, -0.180),
    (0.137, -0.175),
    (0.235, -0.185),
    (0.245, -0.175),
    (0.293, -0.175),
    (0.347, -0.159),
)
# Manufacturer standing stature, used only to bound sampled LiDAR elevation.
G1_STANDING_BODY_TOP_M = 1.32


def g1_body_surface_offset_m(bearing_deg: float) -> float:
    """Return pelvis-to-silhouette distance along one body-frame bearing."""

    angle = math.radians(float(bearing_deg))
    direction = (math.cos(angle), math.sin(angle))
    return max(
        vertex[0] * direction[0] + vertex[1] * direction[1]
        for vertex in G1_BODY_HULL_XY
    )


# Route authoring, Oracle certification, and the online local A* share this
# nominal circular centre-line clearance.  The locomotion controller remains
# physical/contact-driven; this planner radius neither force-tracks the body
# nor vetoes execution on contact.
G1_NAVIGATION_FOOTPRINT_RADIUS_M = 0.25
# RGB-D self filtering pads the measured G1 body hull so sparse renderer
# returns at the silhouette are not fused as external obstacles.  This is a
# sensor-filter parameter, not navigation clearance; route planning applies
# only ``G1_NAVIGATION_FOOTPRINT_RADIUS_M``.
G1_RGBD_SELF_FILTER_PADDING_M = 0.10
# The Mobile Franka pedestal is 0.42 × 0.32 m. Route clearance includes its
# 0.264 m half-diagonal and the path controller's 0.10 m tracking envelope.
# This is the centre-line clearance required by every scene, not an
# episode-specific obstacle margin.
MOBILE_FRANKA_NAVIGATION_CLEARANCE_RADIUS_M = math.hypot(0.42 / 2.0, 0.32 / 2.0) + 0.10
ROBOT_HUMAN_CENTER_CLEARANCE_M = G1_NAVIGATION_FOOTPRINT_RADIUS_M + HUMAN_BODY_RADIUS_M
# A Human GoTo authored at a face-to-face milestone must not sweep through
# the Robot's waiting pose. The additional margin absorbs occupancy-cell
# quantization and small differences between the certified path and the live
# AnimPeople trajectory.
ROBOT_HUMAN_MOTION_CLEARANCE_M = ROBOT_HUMAN_CENTER_CLEARANCE_M + 0.10

# Handover acceptance is an evaluation gate, not a locomotion safety cutoff.
# The closed band is centre-to-centre: the lower endpoint is the physical
# robot-footprint plus human-body clearance; the upper endpoint is the
# social-distance maximum.
HANDOVER_MIN_DISTANCE_M = ROBOT_HUMAN_CENTER_CLEARANCE_M
HANDOVER_MAX_DISTANCE_M = 1.5

# A FixedJoint is a post-contact simulator primitive, never evidence that a
# grasp occurred. Before creating it, the hand must be closed and native PhysX
# must report articulated-finger contact with the exact target at the generated
# grasp centre. Physical mode then proves retention by lifting and carrying the
# unconstrained object; Fixed mode only preserves that already realized pose.
# AABB overlap and IK residuals remain diagnostic only.
G1_GRASP_CONTACT_GAP_M = 0.005
# The reach ends when its trajectory runs out of waypoints, not when the arm
# arrives.  A finite-gain arm still carries steady-state error at that moment,
# and the joint-space tolerance that accepts it is far coarser than the gap
# above: a residual of 0.02 rad is ~0.02 m of palm error at manipulation reach,
# four times the contact gap.  Closing there puts the hand beside the target
# and shoves it.  The terminal reach waypoint is therefore held until the
# measured palm stops moving, so closure begins from an arrived arm.  This
# settles real geometry and loosens no acceptance threshold.
G1_REACH_SETTLE_PHYSICS_STEPS = 24
G1_REACH_SETTLE_CONVERGENCE_M = 0.0005
G1_REACH_SETTLE_MAX_PHYSICS_STEPS = 480
# After the reach settle loop, the measured palm still sits 2–4 cm from the
# planned goal because the analytic FK model is biased even when joint tracking
# is tight.  A measured-feedback pass cancels that bias by shifting the IK
# target by the live model offset.  Stop correcting once the residual is no
# larger than the finger closure gap: anything inside that envelope is where
# the fingers close through anyway.
G1_REACH_CORRECTION_TOLERANCE_M = G1_GRASP_CONTACT_GAP_M
# Residuals above this mean the palm is not near the planned goal at all — wrong
# object, blocked arm, or a stalled reach — not the repeatable 2–4 cm FK bias the
# offset shift cancels.  Fifteen centimetres is roughly half a forearm segment,
# well beyond stance-sweep bias yet still inside one arm workspace, so correcting
# beyond it would chase a mis-grasp rather than trim local model error.
G1_REACH_CORRECTION_MAX_M = 0.15
# Cap each pass to this much palm-space motion toward the shifted IK target.  Four
# centimetres comfortably exceeds the typical 2–3 cm stance-sweep bias so ordinary
# corrections finish in one step, while bounding how far a single mis-estimated step
# can throw the palm before the next re-measurement closes the loop.
G1_REACH_CORRECTION_MAX_STEP_M = 0.04
# Upper bound on correction passes; the insufficient-progress rule below ends the
# loop once the arm reaches its drive-coupling equilibrium.  Five passes remain
# as headroom for multi-step corrections while the settle budget guard stops the
# loop when ticks run out.
G1_REACH_CORRECTION_MAX_PASSES = 5
# After a pass completes, residual improvement smaller than this is
# indistinguishable from the 2–3 mm joint jitter measured in the stance sweep.
# The arm has reached actuation equilibrium; further passes only wind up PD
# command error without moving the palm.
G1_REACH_CORRECTION_MIN_PROGRESS_M = 0.002
# How far the palm may travel in one pass is bounded by the step clamp above;
# this is only a backstop against the redundant arm leaving the configuration
# family cuRobo certified, which a warm-started damped-least-squares solve does
# not do on its own.  The joint cost of a bounded palm step is set by local
# manipulability, so a deep crouch with a pitched waist honestly needs a third
# of a radian to move the palm two centimetres; measured solves that reached
# sub-millimetre accuracy spanned 0.10-0.33 rad, while the one genuine
# null-space jump asked for 1.44 rad and failed to converge anyway.  Half a
# radian separates the two with margin on both sides.
G1_REACH_CORRECTION_MAX_JOINT_DELTA_RAD = 0.5
# The IK rotation target is the model FK at the measured joints (zero rotation
# change).  This tolerance bounds how far the solve may drift from that model
# palm frame, keeping the Dex3 jaws square to the object while position error
# is taken out.
G1_REACH_CORRECTION_ORIENTATION_TOLERANCE_RAD = math.radians(3.0)
# Cosine ease-in-out over one settle chunk keeps the correction inside the same
# velocity envelope as the terminal reach hold.
G1_REACH_CORRECTION_RAMP_PHYSICS_STEPS = G1_REACH_SETTLE_PHYSICS_STEPS
# Minimum current surface gap between the grasped object's finite world AABB
# and its source support AABB. Sliding while still on the support has zero
# gap; carrying beside it does not require staying above the old table height.
G1_GRASP_SUPPORT_CLEARANCE_M = 0.005
# A grasp pose that lost its retention is excluded from later attempts by the
# world position of its closing-volume centre, never by its index in the
# proposal list: every decision re-runs GraspGenX on a fresh depth frame, so
# index k denotes a different pose each time.  Two proposals whose centres sit
# within this radius close on the same place and share one verdict.
G1_GRASP_POSE_EXCLUSION_RADIUS_M = 0.005
# After a pickable leaves its kinematic hold, let PhysX resolve contact with
# the authored support (for example a woven tray) before the arm enters.
G1_GRASP_SUPPORT_SETTLE_PHYSICS_STEPS = 120
# Fail loud when the support surface cannot hold the target near its authored
# pose once dynamics begin. Uneven supports may move slightly; larger shifts
# mean GraspGenX / GT grounding no longer matches the live object.
G1_GRASP_SUPPORT_SETTLE_MAX_HORIZONTAL_DISPLACEMENT_M = 0.015
G1_GRASP_SUPPORT_SETTLE_MAX_VERTICAL_DROP_M = 0.010
# Offline authoring admits the smaller horizontal AABB span up to the envelope
# of the gripper it will be executed with. The bound is deliberately not
# authored here: it belongs to the end effector, so it is read from the mounted
# Dex3 profile and moves when the hand does.
TARGET_SCREENING_MAX_GRASP_SPAN_M = DEX3_1.maximum_jaw_span_m
# A caption is only usable if the object actually looks like it in headless
# RTX. Constant-base-colour MDL shaders are scaled by (1 - metallic) and by
# opacity, so an otherwise bright asset can render near-black. Pickable
# configuration repairs that by forcing metallic off on the constant branch
# and, unless authored Opacity is at or below 0.05 (legitimate glass),
# forcing opacity to 1. Textured surfaces already survive and are counted as
# fully visible. Reject only when the MDL inputs cannot be read, or when the
# area-weighted authored Rec.709 luminance is below this cutoff: a small dark
# trim cannot sink a brighter body.
# The gate is authored_base_colour_luminance on the 448 size-passing GRScenes
# others/ graspables that have at least one constant-branch shader, with
# textured meshes counted as 1.0 (n=448, min 0.0103, p10 0.1113, p25 0.2932,
# 29 below 0.05). The cutoff is a captionability judgement on that continuous
# distribution, not an empty inter-mode gap. It rejects only the near-black
# regime that renders as a featureless blob. Merely-dark objects remain
# groundable: in a live rollout the evaluated model captioned the repaired
# beige bottle as a "small dark bottle" and SAM3 grounded it correctly.
TARGET_SCREENING_MIN_AUTHORED_BASE_COLOUR_LUMINANCE = 0.05
# These are policy commands, not hand-authored joint poses.  The official G1
# locomotion controller tracks the requested pelvis height while preserving
# balance. The interval is the continuous crouch range certified for the
# current locomotion policy. The upper endpoint remains below the separate
# standing command so the two public postures are unambiguous.
G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M = 0.46
G1_MAXIMUM_CROUCH_HEIGHT_COMMAND_M = 0.70
G1_STANDING_HEIGHT_COMMAND_M = 0.74
# The official G1 locomotion policy is standing-only. Crouch plants both feet
# for a low reach; walking or turning from that posture is unsupported and
# unstable. Public tools already hide locomotion while crouched; the runtime
# refuses the same actions, including oracle path/heading servos, until
# ``stand_up``.
G1_STANDING_LOCOMOTION_ACTIONS = frozenset(
    {
        "move_body",
        "navigate",
        "move_forward",
        "move_backward",
        "move_left",
        "move_right",
        "turn_left",
        "turn_right",
        "follow_path",
        "precise_approach",
        "precise_rotate",
        "rotate_to",
    }
)
# The actions that change the commanded pelvis height. Publishing that command
# to the controller is best-effort for every other action, but terminal for
# these: a lost write leaves the robot in the old posture while the runtime
# reports the requested one, which is exactly what the standing-only gait
# check above exists to prevent.
G1_BODY_POSTURE_ACTIONS = frozenset({"crouch", "stand_up"})
# The locomotion policy tracks a commanded body height, not the pelvis rigid
# body's world-Z coordinate exactly.  The certified minimum crouch settles a
# few centimetres below its 0.46 m command.  Fall detection therefore admits
# this bounded tracking difference while continuing to reject a collapsed
# pelvis below every supported standing/crouching posture.
G1_PELVIS_HEIGHT_TRACKING_TOLERANCE_M = 0.05
# Grasp planning uses locomotion-certified waist yaw and a separately
# controlled waist pitch on the manipulation stance. Waist roll remains
# FixedJoint at zero on the certified composition.
# The official locomotion policy can settle a visually neutral G1 just beyond
# 25 degrees at this joint (the live certified fixture measures 25.16
# degrees).  Thirty degrees admits that native state without exposing the
# asset's much wider +/-150 degree mechanical range to manipulation planning.
G1_MANIPULATION_WAIST_YAW_LIMIT_RAD = math.radians(30.0)
# Matches ``unitree_g1.yml`` and the donor URDF/USD mechanical stop (~29.8 deg).
G1_MANIPULATION_WAIST_PITCH_LIMIT_RAD = 0.5235999822616577

# A target whose live top surface is at or below this height is handled from
# the certified crouch posture. The grasp still has to pass right-hand cuRobo
# planning and
# native PhysX hand contact; this threshold selects posture, not success.  The
# 0.80 m boundary includes the independently certified 0.752 m table-bottle
# reference and avoids asking the standing arm to reach below its stable
# frontal workspace.
G1_CROUCH_GRASP_MAX_TARGET_TOP_M = 0.80
# Task-level reachability may admit a modest manipulation envelope, but the
# numerical solve must still refine the Dex3 closing centre to this accuracy
# before the pose is sent to the official controller.
G1_ARM_IK_SOLVER_TOLERANCE_M = 0.005
# One public Cartesian arm action is deliberately local.  Larger motions need
# a fresh camera observation and another decision instead of turning a delta
# controller into an open-loop arm trajectory.  Rotation is one body-frame
# rotation vector, so the norm (not each component independently) is bounded.
G1_END_EFFECTOR_DELTA_MAX_TRANSLATION_M = 0.10
G1_END_EFFECTOR_DELTA_MAX_ROTATION_RAD = math.radians(15.0)
G1_END_EFFECTOR_DELTA_POSITION_TOLERANCE_M = 0.015
# A Cartesian action is not finished merely because the arm arrived: its
# seventh coordinate is part of the same atomic contract.  This is expressed
# in the same normalized closure coordinate exposed to the model and report;
# mixing it with a per-joint radian threshold would make the early-stop gate
# disagree with its public residual.
G1_END_EFFECTOR_GRIPPER_TRACKING_TOLERANCE = 0.05
# A cuRobo trajectory is only complete when the live articulation reaches its
# terminal arm command.  This is a controller tracking contract, independent
# of the Cartesian IK residual above.  Five degrees admits actuator settling
# under contact while rejecting a visibly unretracted arm that would corrupt
# the locomotion policy observation.
G1_ARM_TRAJECTORY_TRACKING_TOLERANCE_RAD = math.radians(5.0)
# The closing-pose reach terminates adjacent to the support surface, where
# measured executions settle with more dispersion than the walking-posture
# returns above: observed clean reaches land around 4.4-5.2 degrees while a
# genuinely blocked arm measures over 70 degrees.  Ten degrees separates those
# two regimes.  This admission gate only decides whether finger closure may
# start; whether a grasp exists is still decided afterwards by measured force
# closure, the closing-volume check, and physical retention.
G1_GRASP_REACH_TRACKING_TOLERANCE_RAD = math.radians(10.0)
# The manipulation envelope a grasp reach is admitted at, and how far off the
# body's heading may be when it enters. These are the numbers live certification
# executed the promoted golden grasp with, so an evaluated agent's grasp reach
# uses them rather than a second envelope of its own: a grasp accepted for the
# oracle and refused for the agent would measure the runtime, not the agent.
# Acceptance is still the physical contact gate below, never this residual.
G1_GRASP_REACH_TOLERANCE_M = 0.08
G1_GRASP_ENTRY_MAX_FACING_ERROR_DEG = 20.0

# Legacy Cartesian IK uses this torso-forward retraction distance. The
# GraspGenX--cuRobo skill instead plans to its embodiment-level compact carry
# joint posture, so carry feasibility does not depend on the grasp direction.
G1_GRASP_CARRY_RETRACTION_M = 0.10
# Certified bottle grasp restore droop is 71.6 mm peak-to-trough; the carry
# lift must exceed that droop plus G1_GRASP_SUPPORT_CLEARANCE_M.
G1_GRASP_CARRY_WORLD_LIFT_M = 0.085
# After the post-contact lift leaves the source support, restore the official
# locomotion upper body by remaining fractions of the live grasp/retract
# command. Pelvis height is held during this interpolation; stand_up restores
# walking height afterwards. This is the historical Golden Oracle prepare_carry
# schedule, not a second cuRobo solve.
# Keep the final loaded shoulder motion as small as the earlier segments.  A
# direct 0.25 -> 0.0 transition can exceed the same controller tracking gate
# even after a valid force-closure lift, especially for broad objects such as
# books.  These extra waypoints preserve the required all-zero locomotion pose;
# they only make the measured transition to it gradual.
G1_CARRY_LOCOMOTION_UPPER_BODY_INTERMEDIATE_FRACTIONS = (
    0.75,
    0.50,
    0.25,
    0.125,
    0.0625,
)
# Do not carry a merely sub-terminal arm tracking bias into the next waist
# segment.  Two adjacent sub-terminal misses can otherwise accumulate into a
# terminal miss even though a short live-state correction at the first segment
# is sufficient.  The final acceptance gate remains the unchanged trajectory
# tracking tolerance below; this value only decides when to settle early.
G1_CARRY_WAIST_RESTORE_SETTLE_TOLERANCE_RAD = (
    0.5 * G1_ARM_TRAJECTORY_TRACKING_TOLERANCE_RAD
)


def g1_locomotion_upper_body_remaining_waypoint(
    start_waypoint: Sequence[float],
    remaining_fraction: float,
    *,
    expected_length: int,
) -> tuple[float, ...]:
    """Keep pelvis height and scale waist plus arm joints toward walking zeros."""

    fraction = float(remaining_fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError(
            "G1 locomotion upper-body remaining fraction must lie in [0, 1]."
        )
    values = tuple(float(value) for value in start_waypoint)
    if len(values) != expected_length or not all(
        math.isfinite(value) for value in values
    ):
        raise ValueError(
            "G1 locomotion interpolation requires a finite "
            f"{expected_length}-D waypoint."
        )
    return (
        values[0],
        *(component * fraction for component in values[1:]),
    )


def g1_control_ramp_samples(
    start_waypoint: Sequence[float],
    end_waypoint: Sequence[float],
    *,
    ticks: int,
) -> tuple[tuple[float, ...], ...]:
    """Spread one commanded transition over its ticks instead of stepping it.

    Repeating a terminal waypoint presents the drive with the whole remaining
    travel as instantaneous error. At the manipulation stiffness that demand is
    several times the joint force ceiling, so the arm is thrown to the target
    rather than servoed to it and a carried object slides out of the fingers.
    """

    steps = int(ticks)
    if steps < 1:
        raise ValueError("G1 control ramps need at least one tick.")
    start = tuple(float(value) for value in start_waypoint)
    end = tuple(float(value) for value in end_waypoint)
    if len(start) != len(end):
        raise ValueError(
            "G1 control ramp endpoints must share one waypoint length: "
            f"{len(start)} != {len(end)}."
        )
    if not all(math.isfinite(value) for value in (*start, *end)):
        raise ValueError("G1 control ramp endpoints must be finite.")
    return tuple(
        tuple(
            first + (last - first) * ((index + 1) / steps)
            for first, last in zip(start, end, strict=True)
        )
        for index in range(steps)
    )


def g1_control_smooth_ramp_samples(
    start_waypoint: Sequence[float],
    end_waypoint: Sequence[float],
    *,
    ticks: int,
) -> tuple[tuple[float, ...], ...]:
    """Spread one restore/carry transition with cosine ease-in-out.

    Velocity is zero at both ends of every segment, so chained restore
    stages do not kink at the waypoint boundaries. Reach execution and
    every cuRobo-planned trajectory keep ``g1_control_ramp_samples``
    (linear) and ``resample_control_trajectory`` unchanged.
    """

    steps = int(ticks)
    if steps < 1:
        raise ValueError("G1 control ramps need at least one tick.")
    start = tuple(float(value) for value in start_waypoint)
    end = tuple(float(value) for value in end_waypoint)
    if len(start) != len(end):
        raise ValueError(
            "G1 control ramp endpoints must share one waypoint length: "
            f"{len(start)} != {len(end)}."
        )
    if not all(math.isfinite(value) for value in (*start, *end)):
        raise ValueError("G1 control ramp endpoints must be finite.")
    return tuple(
        tuple(
            first
            + (last - first) * (0.5 * (1.0 - math.cos(math.pi * ((index + 1) / steps))))
            for first, last in zip(start, end, strict=True)
        )
        for index in range(steps)
    )


def g1_pickable_over_support_xy(
    object_bounds_min_xyz: Sequence[float],
    object_bounds_max_xyz: Sequence[float],
    support_bounds_min_xyz: Sequence[float],
    support_bounds_max_xyz: Sequence[float],
) -> bool:
    """Return True when a pickable AABB overlaps the support footprint in XY."""

    o_min = tuple(float(value) for value in object_bounds_min_xyz)
    o_max = tuple(float(value) for value in object_bounds_max_xyz)
    s_min = tuple(float(value) for value in support_bounds_min_xyz)
    s_max = tuple(float(value) for value in support_bounds_max_xyz)
    if not all(math.isfinite(value) for value in (*o_min, *o_max, *s_min, *s_max)):
        raise ValueError(
            "G1 support-overlap check requires finite object and support bounds."
        )
    return (
        o_min[0] < s_max[0]
        and o_max[0] > s_min[0]
        and o_min[1] < s_max[1]
        and o_max[1] > s_min[1]
    )


# Conservative absolute horizontal reach bound for the mounted G1 right arm.
# It is intentionally wider than the empirically reliable tabletop envelope:
# only a target whose nearest observed surface lies beyond this bound may be
# rejected without GraspGenX or cuRobo.  Targets inside it still require exact
# pose generation, collision-aware IK, motion generation, and physical contact.
G1_GRASP_ABSOLUTE_MAX_HORIZONTAL_REACH_M = 0.90

# Episode authoring samples no farther than the same physical outer bound. It
# is an alias, rather than a second independently tunable reach fact.
G1_GRASP_ENTRY_MAX_HORIZONTAL_DISTANCE_M = G1_GRASP_ABSOLUTE_MAX_HORIZONTAL_REACH_M

# Placement scoring is measured from the released object's live AABB, not from
# an IK residual or an arm-joint tracking predicate.  This is the semantic
# object--destination relation envelope; it is deliberately not the physical
# hand-opening gate below.
G1_PLACEMENT_RELATION_TOLERANCE_M = 0.10
# Release certification samples the object at this fixed number of consecutive
# physics intervals. This is a physical stability contract, not a task-progress
# weight or an episode-configurable evaluator parameter.
G1_PLACEMENT_SETTLE_SAMPLE_COUNT = 3
# A place skill aims the held object's horizontal centre at the observed
# support point before opening the hand.  This is a controller-completion
# tolerance, independent of the wider post-release relation score.
G1_PRE_RELEASE_HORIZONTAL_TOLERANCE_M = 0.05
# The target lower surface is commanded this far above the observed support.
# This is the maximum release-time surface separation: it avoids commanding
# mesh interpenetration while still requiring physical set-down proximity.
G1_PRE_RELEASE_CLEARANCE_M = 0.0
# Maximum absolute separation between the carried object's lower surface and
# the observed support plane before the fingers may open. The support point
# must also lie beneath the object's horizontal footprint. This is a physical
# contact-scale gate, not the wider object--destination scoring envelope.
G1_PRE_RELEASE_SUPPORT_CONTACT_TOLERANCE_M = 0.005
# Across the fixed post-release sample window, a dynamically released
# object's world-AABB centre may move by at most this amount between
# consecutive samples. Relation-only checks would accept an object sliding
# across or falling through a destination.
G1_PLACEMENT_SETTLE_MOTION_M = 0.02

# A grasp certificate must carry the target this far from its initial support.
# Runtime grasp admission uses the much smaller contact/source-clearance gate;
# this distance only proves a completed lift-and-carry transition.
G1_GRASP_CERTIFICATION_DISPLACEMENT_M = 0.05

# Navigation layers have distinct, explicitly named tolerances. The semantic
# goal is an episode-level region; controller reach is a lower-level physical
# completion rule and must not be reused as an evaluation threshold.
NAVIGATION_SEMANTIC_GOAL_TOLERANCE_M = 0.45
NAVIGATION_CONTROLLER_SPACING_M = 0.75
NAVIGATION_SEGMENT_HORIZON_M = 2.0
# A single controller decision must not carry stale waypoint state through a
# room-scale corner.  End the stable path prefix before such a turn, observe
# the live body again, then replan with the same path backend.
NAVIGATION_SEGMENT_MAX_TURN_RAD = math.radians(55.0)
# One physical reach envelope governs every waypoint transition, terminal
# segment check, and executor early-stop. Keeping these consumers on one value
# prevents an inner controller from orbiting a waypoint which the outer
# runtime already considers reached.
NAVIGATION_CONTROLLER_REACH_TOLERANCE_M = 0.20
# TARSphere's MoveToPoint uses one parameter for both its stop threshold and
# the half-width of a cubic speed taper. The TARSphere path adapter owns the 20 cm
# waypoint completion above and preserves the official 10 cm inner threshold
# solely to keep useful policy authority until that completion boundary.
NAVIGATION_CONTROLLER_SPEED_TAPER_THRESHOLD_M = 0.10
NAVIGATION_PATH_ENTRY_YAW_TOLERANCE_RAD = 0.12

# One fail-loud progress authority is owned by the rollout watchdog.  The
# Golden oracle may replan when a controller explicitly reports completion,
# but it does not run a second displacement-based stall detector.
# A model-controlled route is stopped only after eight consecutive locomotion
# decisions fail to create meaningful translation or rotation. Shorter windows
# confuse recovery attempts and learned-gait acceleration with policy collapse.
NAVIGATION_STUCK_INCIDENT_DECISIONS = 5
NAVIGATION_STUCK_TERMINATION_DECISIONS = 10
NAVIGATION_MIN_XY_PROGRESS_M = 0.03
NAVIGATION_MIN_HEADING_PROGRESS_RAD = 0.05


# ---------------------------------------------------------------------------
# Locomotion action-space magnitudes
# ---------------------------------------------------------------------------
# PhysX and the live safety monitor advance at this shared rate. The action
# router uses it to convert a public magnitude into the duration of the fixed
# velocity command; achieved body motion is measured separately after execution.
SIMULATION_PHYSICS_HZ = 240.0
# ---------------------------------------------------------------------------
# Human animation clock
# ---------------------------------------------------------------------------
# ``omni.anim.people`` advances on Kit app updates. Physics stays at 240 Hz;
# one explicit animation update every four physics ticks supplies 60 Hz motion,
# including ambient LookAround when no authored Human action is pending.
# The cadence and Kit manual step size must agree:
#
#     manual_step_size = HUMAN_ANIMATION_PUMP_EVERY_TICKS / SIMULATION_PHYSICS_HZ
#
# Native video renders use zero Kit delta so they cannot advance this clock a
# second time. Camera/observation renders likewise display the current pose.
HUMAN_ANIMATION_PUMP_EVERY_TICKS = 4
HUMAN_ANIMATION_KIT_STEP_SIZE_S = (
    HUMAN_ANIMATION_PUMP_EVERY_TICKS / SIMULATION_PHYSICS_HZ
)
# A locomotion decision ends at a stationary observation boundary.  The
# velocity command runs for the duration selected by the Agent, then the
# released G1 policy receives zero velocity for this fixed interval before the
# next observation is captured.  This preserves one contiguous command per
# decision while allowing the learned gait to settle naturally.
G1_POST_MOTION_SETTLE_S = 0.25
# ``move_body`` is a relative-pose command, not a promise that the released
# gait tracks a velocity for an exact amount of time.  The open-loop response
# below remains the feed-forward term, while the runtime closes the final pose
# error using only the live robot root pose.  These limits bound that feedback:
# they neither inspect scene geometry nor override native PhysX contact.
G1_RELATIVE_POSE_TRANSLATION_TOLERANCE_M = 0.04
G1_RELATIVE_POSE_YAW_TOLERANCE_RAD = math.radians(2.0)
# Pure lateral gait and in-place turns have larger stationary pelvis sway than
# forward/backward and coupled SE(2) motion.  Repeated empty.usd trials on
# 2026-08-31 measured about 4.7--4.9 cm for lateral motion and 6.1--8.1 cm
# after a closed-loop 90 deg turn, while their requested primary axes were
# correct and both repeats completed a 60-tick zero-speed hold.
# Keep the tighter default above for metric translation/coupled SE(2).
G1_RELATIVE_POSE_LATERAL_TRANSLATION_TOLERANCE_M = 0.05
G1_RELATIVE_POSE_LATERAL_YAW_TOLERANCE_RAD = math.radians(2.5)
G1_RELATIVE_POSE_ROTATION_TRANSLATION_TOLERANCE_M = 0.085
G1_RELATIVE_POSE_CORRECTION_ALLOWANCE_S = 1.50
G1_RELATIVE_POSE_TRANSLATION_GAIN_S_INV = 1.25
G1_RELATIVE_POSE_YAW_GAIN_S_INV = 2.0
# The released policy did not measurably respond to a 0.15 m/s residual
# command in the 2026-08-31 empty.usd sweep. At 0.3 m/s metric translation and
# coupled SE(2) converge without the overshoot produced by a 0.5 m/s endpoint
# correction. Pure turns intentionally do not append a translation command:
# doing so increased their settled 7--8 cm pelvis drift to 9--13 cm and added
# yaw error, changing the requested in-place turn into a second locomotion leg.
G1_RELATIVE_POSE_MIN_TRANSLATION_COMMAND_MPS = 0.3
# The released G1 locomotion controller accepts a yaw-rate command. Sub-unit
# inputs are absorbed around the learned policy's standing deadband, so both
# evaluated-agent and oracle turns use this certified 1 rad/s command when
# deriving and executing the requested turn duration. Live yaw remains the
# authority for reporting achieved rotation.
G1_ROTATION_POLICY_COMMAND = 1.0
# Below this a forward call cannot change the situation it was issued to change;
# it only spends a decision step.
LOCOMOTION_MIN_STEP_DISTANCE_M = 0.1
# One forward leg is capped at the longest window the agent may act on a single
# ego frame for: 1.5 m is 3 s at the commanded 0.5 m/s, which is the authored
# golden decision horizon. A longer leg would sweep ground the agent last looked
# at seconds earlier, and a person can enter that ground in the meantime.
LOCOMOTION_MAX_STEP_DISTANCE_M = 1.5
# Backing up is capped far shorter because the ego camera cannot see behind at
# all: the robot may give back ground it just occupied, never enter new ground
# blind. There is no rear clearance measurement to clamp this against.
LOCOMOTION_MAX_BACKWARD_DISTANCE_M = 0.5
# Sideways strafe uses the same short cap: the ego FOV only partly covers the
# leading flank, and the released gait tracks lateral velocity less cleanly
# than forward walking.
LOCOMOTION_MAX_LATERAL_DISTANCE_M = 0.5
# A body-frame vector uses the same certified component limits as the named
# translation primitives. Forward is asymmetric because the ego camera sees
# ahead but not behind; left is signed and symmetric.
LOCOMOTION_MIN_BODY_FORWARD_M = -LOCOMOTION_MAX_BACKWARD_DISTANCE_M
LOCOMOTION_MAX_BODY_FORWARD_M = LOCOMOTION_MAX_STEP_DISTANCE_M
LOCOMOTION_MAX_BODY_LEFT_M = LOCOMOTION_MAX_LATERAL_DISTANCE_M
# Below this a turn does not meaningfully change what the ego camera frames.
LOCOMOTION_MIN_TURN_DEG = 5.0
# A turn keeps the root near one point but rotates the asymmetric footprint;
# live sweep monitoring remains active. A half-turn stays one call because
# refreshing the view is the purpose of turning.
LOCOMOTION_MAX_TURN_DEG = 180.0
# A yield has to last long enough for a person to actually cross in front of the
# robot; below half a second nothing has moved, and beyond five seconds the
# agent is holding on a stale frame rather than yielding.
LOCOMOTION_MIN_YIELD_SECONDS = 0.5
LOCOMOTION_MAX_YIELD_SECONDS = 5.0


@dataclass(frozen=True)
class G1LocomotionAxisResponse:
    """Measured open-loop response of one signed gait-command axis.

    For a constant controller command ``u`` held for ``T`` seconds, followed
    by the standard decision-boundary settle, the achieved signed displacement
    is modelled as ``gain * u * (T - startup_loss_s)``. This is feed-forward
    command tracking only: collision response is never compensated or retried.
    """

    gain: float
    startup_loss_s: float


# Direction-specific feed-forward response for the g1_15000.onnx gait at
# 240 Hz. Each signed axis uses its own gain and startup allowance.
G1_LOCOMOTION_FORWARD_RESPONSE = G1LocomotionAxisResponse(
    gain=1.072425,
    startup_loss_s=0.09795953096953167,
)
G1_LOCOMOTION_BACKWARD_RESPONSE = G1LocomotionAxisResponse(
    gain=0.827003,
    startup_loss_s=0.2425756012976978,
)
G1_LOCOMOTION_LEFT_RESPONSE = G1LocomotionAxisResponse(
    gain=0.926605,
    startup_loss_s=0.0802642442033013,
)
G1_LOCOMOTION_RIGHT_RESPONSE = G1LocomotionAxisResponse(
    gain=0.92681,
    startup_loss_s=0.0904376301507321,
)
G1_LOCOMOTION_TURN_LEFT_RESPONSE = G1LocomotionAxisResponse(
    gain=1.006518,
    startup_loss_s=0.07203397886871213,
)
G1_LOCOMOTION_TURN_RIGHT_RESPONSE = G1LocomotionAxisResponse(
    gain=0.938099,
    startup_loss_s=0.08564776327899865,
)

# Horizontal target-to-stance distance within which the right-hand grasp is
# reliable. Measured by the archived grasp-stance calibration over a polar
# grid on the bare table, one cup, executor and scene held fixed and only the
# body moved: 0.400 m 5/5, 0.475 m 5/5, 0.550 m 4/5, then 0.625 m 0/5 and
# 0.700 m 0/5.  Lateral offset was immaterial across the full +/-0.20 m span at
# every distance, so distance alone bounds the reliable set.  The two failing
# bands differ in kind and that difference is why this constant exists: beyond
# 0.700 m cuRobo refuses with ``reach_envelope``, but at 0.625 m it plans and
# executes a reach whose accumulated palm error leaves the Dex3 closure just
# short of enclosing the object, reported as ``contact_miss``.  The planner's
# reachable set is therefore wider than the graspable one, and only the stance
# distance separates them.  Re-measure with the sweep when the arm model,
# palm-frame calibration, or body-height tracking changes.
GRASP_RELIABLE_STANCE_DISTANCE_M = 0.55

# Current-frame virtual LiDAR. It samples live PhysX geometry only and contains
# no task identity, route, target coordinate, or offline occupancy evidence.
# The Agent sees only the current scan. The private VoxelLayer integrates that
# scan with a replace-on-update inverse-sensor model: adjacent bearings fill
# the angular sector they jointly observe, measured endpoints and similar-range
# chords are occupied, and a max-range miss never writes a 1-cell-wide free
# tunnel. Sector free-range stops where the inter-beam gap would exceed the G1
# body width, so coarse rear beams do not invent a 5 m corridor. The map
# remains sensor-derived and is never written into textual memory. Forward
# bearings remain denser for longer forward moves.
LIDAR_ORIGIN_HEIGHT_M = 0.35
LIDAR_MAX_RANGE_M = 5.0
LIDAR_NO_RETURN = None
LIDAR_FORWARD_SECTOR_DEG = 45.0
LIDAR_FINE_AZIMUTH_STEP_DEG = 3.75
LIDAR_COARSE_AZIMUTH_STEP_DEG = 22.5
LIDAR_FULL_HEIGHT_REFERENCE_M = 1.0
LIDAR_VERTICAL_GAP_AT_MAX_LEG_M = 0.1
LIDAR_FLOOR_TOLERANCE_M = 0.05

# Ego-camera gaze tilt: the agent's commanded pitch offset from the robot
# profile's fixed mount calibration, in degrees, negative looking down (the
# project-wide sign convention — see geometry.assemble_camera_pose). It is
# absolute, not incremental: 0 is level with the mount, so the agent can always
# return to level in one call without tracking what it accumulated.
#
# The head view serves both navigation and close manipulation.  Keeping the
# horizon in every frame would cap the command below half the vertical FOV and
# make a certified arm-reach tabletop fall outside the image.  The command is
# absolute and reversible, so manipulation may deliberately leave the horizon
# out of frame and return to 0 afterwards.  Thirty degrees, combined with G1's
# fixed -15 degree mount and approximately 83.6 degree vertical FOV at the
# native 640x480 render size, covers the near-hand workspace while keeping the
# lower image ray above the camera's singular straight-down direction.
CAMERA_GAZE_TILT_LIMIT_DEG = 30.0


def g1_body_posture_from_height_command(
    height_command_m: float,
    *,
    maximum_crouch_height_m: float = G1_MAXIMUM_CROUCH_HEIGHT_COMMAND_M,
) -> str:
    """Map a locomotion height command to the public standing/crouched posture."""

    if float(height_command_m) <= float(maximum_crouch_height_m) + 1e-6:
        return "crouched"
    return "standing"


def require_standing_g1_locomotion(
    action: str,
    *,
    body_posture: str,
) -> None:
    """Refuse walking or turning while the G1 pelvis is commanded into a crouch."""

    if body_posture != "crouched":
        return
    if str(action) not in G1_STANDING_LOCOMOTION_ACTIONS:
        return
    raise ActionRefusedError(
        f"G1 locomotion action {action!r} is not supported from a crouch; "
        "the official gait is standing-only and becomes unstable when the "
        "pelvis is lowered. Call stand_up before walking or turning.",
        code="locomotion_requires_standing",
        details={
            "requested_action": str(action),
            "body_posture": str(body_posture),
            "required_action": "stand_up",
        },
    )


def resolve_persisted_g1_body_height(
    height: float,
    cached: float | None,
    *,
    standing_height_m: float = G1_STANDING_HEIGHT_COMMAND_M,
) -> tuple[float, float]:
    """Reuse a crouch command when a nested path wrapper omits height.

    Official path controllers call ``forward(height=0.74)`` because that is
    the parent default, not because the Agent stood up. An explicit standing
    command first writes the standing cache; an omitted default then matches
    that cache and stays standing.
    """

    command = float(height)
    stored = float(standing_height_m) if cached is None else float(cached)
    if not math.isfinite(command) or not math.isfinite(stored):
        raise ValueError("G1 body-height commands must be finite.")
    omitted_standing = math.isclose(
        command,
        standing_height_m,
        abs_tol=1e-9,
    )
    cached_crouch = not math.isclose(
        stored,
        standing_height_m,
        abs_tol=1e-9,
    )
    if omitted_standing and cached_crouch:
        return stored, stored
    return command, command


__all__ = [
    "CAMERA_GAZE_TILT_LIMIT_DEG",
    "G1_ARM_IK_SOLVER_TOLERANCE_M",
    "G1_END_EFFECTOR_DELTA_MAX_TRANSLATION_M",
    "G1_END_EFFECTOR_DELTA_MAX_ROTATION_RAD",
    "G1_END_EFFECTOR_DELTA_POSITION_TOLERANCE_M",
    "G1_END_EFFECTOR_GRIPPER_TRACKING_TOLERANCE",
    "G1_ARM_TRAJECTORY_TRACKING_TOLERANCE_RAD",
    "G1_GRASP_REACH_TRACKING_TOLERANCE_RAD",
    "G1_BODY_HULL_XY",
    "G1_FOOTPRINT_FRONT_M",
    "G1_FOOTPRINT_HALF_WIDTH_M",
    "G1_FOOTPRINT_REAR_M",
    "g1_body_surface_offset_m",
    "g1_body_posture_from_height_command",
    "require_standing_g1_locomotion",
    "resolve_persisted_g1_body_height",
    "G1_CROUCH_GRASP_MAX_TARGET_TOP_M",
    "G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M",
    "G1_MAXIMUM_CROUCH_HEIGHT_COMMAND_M",
    "G1_MANIPULATION_WAIST_YAW_LIMIT_RAD",
    "G1_MANIPULATION_WAIST_PITCH_LIMIT_RAD",
    "G1_PELVIS_HEIGHT_TRACKING_TOLERANCE_M",
    "G1_GRASP_CONTACT_GAP_M",
    "G1_REACH_SETTLE_PHYSICS_STEPS",
    "G1_REACH_SETTLE_CONVERGENCE_M",
    "G1_REACH_SETTLE_MAX_PHYSICS_STEPS",
    "G1_REACH_CORRECTION_TOLERANCE_M",
    "G1_REACH_CORRECTION_MAX_M",
    "G1_REACH_CORRECTION_MAX_STEP_M",
    "G1_REACH_CORRECTION_MAX_PASSES",
    "G1_REACH_CORRECTION_MIN_PROGRESS_M",
    "G1_REACH_CORRECTION_MAX_JOINT_DELTA_RAD",
    "G1_REACH_CORRECTION_ORIENTATION_TOLERANCE_RAD",
    "G1_REACH_CORRECTION_RAMP_PHYSICS_STEPS",
    "G1_GRASP_SUPPORT_CLEARANCE_M",
    "G1_GRASP_SUPPORT_SETTLE_PHYSICS_STEPS",
    "G1_GRASP_SUPPORT_SETTLE_MAX_HORIZONTAL_DISPLACEMENT_M",
    "G1_GRASP_SUPPORT_SETTLE_MAX_VERTICAL_DROP_M",
    "G1_GRASP_CARRY_RETRACTION_M",
    "G1_GRASP_CARRY_WORLD_LIFT_M",
    "G1_CARRY_LOCOMOTION_UPPER_BODY_INTERMEDIATE_FRACTIONS",
    "G1_CARRY_WAIST_RESTORE_SETTLE_TOLERANCE_RAD",
    "g1_control_ramp_samples",
    "g1_control_smooth_ramp_samples",
    "g1_locomotion_upper_body_remaining_waypoint",
    "g1_pickable_over_support_xy",
    "G1_GRASP_CERTIFICATION_DISPLACEMENT_M",
    "G1_GRASP_ENTRY_MAX_FACING_ERROR_DEG",
    "G1_GRASP_ENTRY_MAX_HORIZONTAL_DISTANCE_M",
    "G1_GRASP_ABSOLUTE_MAX_HORIZONTAL_REACH_M",
    "G1_GRASP_REACH_TOLERANCE_M",
    "G1_NAVIGATION_FOOTPRINT_RADIUS_M",
    "G1_RGBD_SELF_FILTER_PADDING_M",
    "G1_ROTATION_POLICY_COMMAND",
    "G1_RELATIVE_POSE_TRANSLATION_TOLERANCE_M",
    "G1_RELATIVE_POSE_YAW_TOLERANCE_RAD",
    "G1_RELATIVE_POSE_LATERAL_TRANSLATION_TOLERANCE_M",
    "G1_RELATIVE_POSE_LATERAL_YAW_TOLERANCE_RAD",
    "G1_RELATIVE_POSE_ROTATION_TRANSLATION_TOLERANCE_M",
    "G1_RELATIVE_POSE_CORRECTION_ALLOWANCE_S",
    "G1_RELATIVE_POSE_TRANSLATION_GAIN_S_INV",
    "G1_RELATIVE_POSE_YAW_GAIN_S_INV",
    "G1_RELATIVE_POSE_MIN_TRANSLATION_COMMAND_MPS",
    "G1LocomotionAxisResponse",
    "G1_LOCOMOTION_FORWARD_RESPONSE",
    "G1_LOCOMOTION_BACKWARD_RESPONSE",
    "G1_LOCOMOTION_LEFT_RESPONSE",
    "G1_LOCOMOTION_RIGHT_RESPONSE",
    "G1_LOCOMOTION_TURN_LEFT_RESPONSE",
    "G1_LOCOMOTION_TURN_RIGHT_RESPONSE",
    "G1_PLACEMENT_RELATION_TOLERANCE_M",
    "G1_PLACEMENT_SETTLE_SAMPLE_COUNT",
    "G1_PLACEMENT_SETTLE_MOTION_M",
    "G1_PRE_RELEASE_CLEARANCE_M",
    "G1_PRE_RELEASE_HORIZONTAL_TOLERANCE_M",
    "G1_PRE_RELEASE_SUPPORT_CONTACT_TOLERANCE_M",
    "G1_POST_MOTION_SETTLE_S",
    "G1_STANDING_HEIGHT_COMMAND_M",
    "G1_BODY_POSTURE_ACTIONS",
    "G1_STANDING_LOCOMOTION_ACTIONS",
    "G1_STANDING_BODY_TOP_M",
    "HANDOVER_MAX_DISTANCE_M",
    "HANDOVER_MIN_DISTANCE_M",
    "HUMAN_GOTO_ARRIVAL_TOLERANCE_M",
    "HUMAN_INITIAL_POSE_TOLERANCE_M",
    "HUMAN_NAVMESH_AGENT_RADIUS_M",
    "LOCOMOTION_MAX_BACKWARD_DISTANCE_M",
    "LOCOMOTION_MAX_BODY_FORWARD_M",
    "LOCOMOTION_MAX_BODY_LEFT_M",
    "LOCOMOTION_MAX_LATERAL_DISTANCE_M",
    "LOCOMOTION_MAX_STEP_DISTANCE_M",
    "LOCOMOTION_MAX_TURN_DEG",
    "LOCOMOTION_MAX_YIELD_SECONDS",
    "LOCOMOTION_MIN_STEP_DISTANCE_M",
    "LOCOMOTION_MIN_BODY_FORWARD_M",
    "LOCOMOTION_MIN_TURN_DEG",
    "LOCOMOTION_MIN_YIELD_SECONDS",
    "GRASP_RELIABLE_STANCE_DISTANCE_M",
    "LIDAR_COARSE_AZIMUTH_STEP_DEG",
    "LIDAR_FINE_AZIMUTH_STEP_DEG",
    "LIDAR_FLOOR_TOLERANCE_M",
    "LIDAR_FORWARD_SECTOR_DEG",
    "LIDAR_FULL_HEIGHT_REFERENCE_M",
    "LIDAR_MAX_RANGE_M",
    "LIDAR_NO_RETURN",
    "LIDAR_ORIGIN_HEIGHT_M",
    "LIDAR_VERTICAL_GAP_AT_MAX_LEG_M",
    "MOBILE_FRANKA_NAVIGATION_CLEARANCE_RADIUS_M",
    "NAVIGATION_CONTROLLER_SPACING_M",
    "NAVIGATION_CONTROLLER_REACH_TOLERANCE_M",
    "NAVIGATION_CONTROLLER_SPEED_TAPER_THRESHOLD_M",
    "NAVIGATION_MIN_HEADING_PROGRESS_RAD",
    "NAVIGATION_MIN_XY_PROGRESS_M",
    "NAVIGATION_PATH_ENTRY_YAW_TOLERANCE_RAD",
    "NAVIGATION_STUCK_INCIDENT_DECISIONS",
    "NAVIGATION_STUCK_TERMINATION_DECISIONS",
    "NAVIGATION_SEGMENT_HORIZON_M",
    "NAVIGATION_SEGMENT_MAX_TURN_RAD",
    "NAVIGATION_SEMANTIC_GOAL_TOLERANCE_M",
    "ROBOT_HUMAN_CENTER_CLEARANCE_M",
    "ROBOT_HUMAN_MOTION_CLEARANCE_M",
    "SIMULATION_PHYSICS_HZ",
    "TARGET_SCREENING_MIN_AUTHORED_BASE_COLOUR_LUMINANCE",
    "TARGET_SCREENING_MAX_GRASP_SPAN_M",
]
