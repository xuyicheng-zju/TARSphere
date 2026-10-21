"""Model-visible tool specifications.

Each ToolSpec is a stable, VLM-visible capability declaration. The VLM sees
these as its available actions and calls them via DecisionEnvelope
``tool_calls``; the registry dispatches each call to the paired handler.

Every spec provides one concise summary and documents every parameter inside
``input_schema`` because those are exactly the fields the prompt renders. See
:class:`~tarsphere.harness.core.types.ToolSpec` for what each field owns.
"""

from __future__ import annotations

from tarsphere.harness.core.types import ToolSpec
from tarsphere.contracts.human_interaction import (
    HUMAN_COMMUNICATION_RADIUS_M,
)
from tarsphere.infra.control.physical_contracts import (
    CAMERA_GAZE_TILT_LIMIT_DEG,
    G1_MAXIMUM_CROUCH_HEIGHT_COMMAND_M,
    G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M,
    G1_END_EFFECTOR_DELTA_MAX_ROTATION_RAD,
    G1_END_EFFECTOR_DELTA_MAX_TRANSLATION_M,
    LOCOMOTION_MAX_BACKWARD_DISTANCE_M,
    LOCOMOTION_MAX_BODY_FORWARD_M,
    LOCOMOTION_MAX_BODY_LEFT_M,
    LOCOMOTION_MAX_LATERAL_DISTANCE_M,
    LOCOMOTION_MAX_STEP_DISTANCE_M,
    LOCOMOTION_MAX_TURN_DEG,
    LOCOMOTION_MAX_YIELD_SECONDS,
    LOCOMOTION_MIN_STEP_DISTANCE_M,
    LOCOMOTION_MIN_BODY_FORWARD_M,
    LOCOMOTION_MIN_TURN_DEG,
    LOCOMOTION_MIN_YIELD_SECONDS,
)


def _no_args() -> dict:
    """The schema of a tool that takes no arguments.

    A function rather than a shared constant so no two specs alias the same
    mutable schema object.
    """
    return {"type": "object", "properties": {}, "additionalProperties": False}


# ---------------------------------------------------------------------------
# Locomotion tools
# ---------------------------------------------------------------------------

MOVE_BODY = ToolSpec(
    name="move_body",
    summary=(
        "Move by one continuous body-frame vector that may combine translation "
        "and turning."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "delta": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
                "description": (
                    "Body-frame `[x, y, theta]`: x is metres "
                    f"forward in [{LOCOMOTION_MIN_BODY_FORWARD_M}, "
                    f"{LOCOMOTION_MAX_BODY_FORWARD_M}]; y is metres left in "
                    f"[-{LOCOMOTION_MAX_BODY_LEFT_M}, "
                    f"{LOCOMOTION_MAX_BODY_LEFT_M}]; theta is yaw degrees in "
                    f"[-{LOCOMOTION_MAX_TURN_DEG}, {LOCOMOTION_MAX_TURN_DEG}]; "
                    "positive theta turns left."
                ),
            },
        },
        "required": ["delta"],
        "additionalProperties": False,
    },
    risk_level="medium",
    tags=frozenset({"locomotion"}),
)

MOVE_END_EFFECTOR = ToolSpec(
    name="move_end_effector",
    summary=(
        "Move the certified right palm by one local 6D pose delta while "
        "setting its gripper closure."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "delta": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 7,
                "maxItems": 7,
                "description": (
                    "`[dx, dy, dz, drx, dry, drz, gripper]`. Translation is "
                    "metres along current robot body axes (x forward, y left, "
                    "z up), each in "
                    f"[-{G1_END_EFFECTOR_DELTA_MAX_TRANSLATION_M}, "
                    f"{G1_END_EFFECTOR_DELTA_MAX_TRANSLATION_M}]. Rotation is "
                    "a body-axis rotation vector in radians whose norm is at "
                    f"most {G1_END_EFFECTOR_DELTA_MAX_ROTATION_RAD:.4f}. "
                    "Gripper is continuous: 0 fully open, 1 fully closed."
                ),
            },
        },
        "required": ["delta"],
        "additionalProperties": False,
    },
    risk_level="high",
    tags=frozenset({"manipulation"}),
)

MOVE_FORWARD = ToolSpec(
    name="move_forward",
    summary="Walk forward along the current heading by a chosen distance.",
    input_schema={
        "type": "object",
        "properties": {
            "distance_m": {
                "type": "number",
                "minimum": LOCOMOTION_MIN_STEP_DISTANCE_M,
                "maximum": LOCOMOTION_MAX_STEP_DISTANCE_M,
                "description": "Requested forward distance in metres.",
            },
        },
        "required": ["distance_m"],
        "additionalProperties": False,
    },
    risk_level="low",
    tags=frozenset({"locomotion"}),
)

MOVE_BACKWARD = ToolSpec(
    name="move_backward",
    summary="Translate backward without turning by a chosen distance.",
    input_schema={
        "type": "object",
        "properties": {
            "distance_m": {
                "type": "number",
                "minimum": LOCOMOTION_MIN_STEP_DISTANCE_M,
                "maximum": LOCOMOTION_MAX_BACKWARD_DISTANCE_M,
                "description": (
                    "Requested backward distance in metres; behind is unseen."
                ),
            },
        },
        "required": ["distance_m"],
        "additionalProperties": False,
    },
    risk_level="medium",
    tags=frozenset({"locomotion"}),
)

MOVE_LEFT = ToolSpec(
    name="move_left",
    summary="Translate left without turning by a chosen distance.",
    input_schema={
        "type": "object",
        "properties": {
            "distance_m": {
                "type": "number",
                "minimum": LOCOMOTION_MIN_STEP_DISTANCE_M,
                "maximum": LOCOMOTION_MAX_LATERAL_DISTANCE_M,
                "description": (
                    "Requested leftward distance in metres; flank view is partial."
                ),
            },
        },
        "required": ["distance_m"],
        "additionalProperties": False,
    },
    risk_level="medium",
    tags=frozenset({"locomotion"}),
)

MOVE_RIGHT = ToolSpec(
    name="move_right",
    summary="Translate right without turning by a chosen distance.",
    input_schema={
        "type": "object",
        "properties": {
            "distance_m": {
                "type": "number",
                "minimum": LOCOMOTION_MIN_STEP_DISTANCE_M,
                "maximum": LOCOMOTION_MAX_LATERAL_DISTANCE_M,
                "description": (
                    "Requested rightward distance in metres; flank view is partial."
                ),
            },
        },
        "required": ["distance_m"],
        "additionalProperties": False,
    },
    risk_level="medium",
    tags=frozenset({"locomotion"}),
)

TURN_LEFT = ToolSpec(
    name="turn_left",
    summary="Rotate left (counter-clockwise) in place by a chosen angle.",
    input_schema={
        "type": "object",
        "properties": {
            "degrees": {
                "type": "number",
                "minimum": LOCOMOTION_MIN_TURN_DEG,
                "maximum": LOCOMOTION_MAX_TURN_DEG,
                "description": "Counter-clockwise rotation in degrees.",
            },
        },
        "required": ["degrees"],
        "additionalProperties": False,
    },
    risk_level="low",
    tags=frozenset({"locomotion"}),
)

TURN_RIGHT = ToolSpec(
    name="turn_right",
    summary="Rotate right (clockwise) in place by a chosen angle.",
    input_schema={
        "type": "object",
        "properties": {
            "degrees": {
                "type": "number",
                "minimum": LOCOMOTION_MIN_TURN_DEG,
                "maximum": LOCOMOTION_MAX_TURN_DEG,
                "description": "Clockwise rotation in degrees.",
            },
        },
        "required": ["degrees"],
        "additionalProperties": False,
    },
    risk_level="low",
    tags=frozenset({"locomotion"}),
)

WAIT = ToolSpec(
    name="wait",
    summary="Stand still while the world continues moving.",
    input_schema={
        "type": "object",
        "properties": {
            "seconds": {
                "type": "number",
                "minimum": LOCOMOTION_MIN_YIELD_SECONDS,
                "maximum": LOCOMOTION_MAX_YIELD_SECONDS,
                "description": "Hold duration in seconds.",
            },
        },
        "required": ["seconds"],
        "additionalProperties": False,
    },
    risk_level="low",
    tags=frozenset({"locomotion", "interaction"}),
)

CROUCH = ToolSpec(
    name="crouch",
    summary=(
        "Lower the humanoid pelvis to a chosen stable height while keeping "
        "both feet planted. The official G1 gait is standing-only; walk or "
        "turn only after stand_up."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "pelvis_height_m": {
                "type": "number",
                "minimum": G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M,
                "maximum": G1_MAXIMUM_CROUCH_HEIGHT_COMMAND_M,
                "description": "Desired pelvis height above floor in metres.",
            },
        },
        "required": ["pelvis_height_m"],
        "additionalProperties": False,
    },
    risk_level="medium",
    tags=frozenset({"locomotion", "manipulation", "posture"}),
)

STAND_UP = ToolSpec(
    name="stand_up",
    summary="Return the humanoid from a crouch to stable standing height.",
    input_schema=_no_args(),
    risk_level="medium",
    tags=frozenset({"locomotion", "manipulation", "posture"}),
)

TILT_CAMERA = ToolSpec(
    name="tilt_camera",
    summary="Aim the head camera relative to its navigation view.",
    input_schema={
        "type": "object",
        "properties": {
            "pitch_deg": {
                "type": "number",
                "minimum": -CAMERA_GAZE_TILT_LIMIT_DEG,
                "maximum": CAMERA_GAZE_TILT_LIMIT_DEG,
                "description": (
                    "Absolute gaze-offset setpoint from the mounted default "
                    "navigation view: negative aims farther down, positive "
                    "farther up, and 0 restores that default (G1 camera axis "
                    "is already pitched -15 degrees down at offset 0). This "
                    "offset is not how far down you can see; it aims the "
                    "camera, not walking."
                ),
            },
        },
        "required": ["pitch_deg"],
        "additionalProperties": False,
    },
    risk_level="low",
    tags=frozenset({"perception", "posture"}),
)


# ---------------------------------------------------------------------------
# Communication tools
# ---------------------------------------------------------------------------

SPEAK = ToolSpec(
    name="speak",
    summary=(
        f"Say something to one visible human within the "
        f"{HUMAN_COMMUNICATION_RADIUS_M:g} m communication radius without "
        "requesting an answer."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "recipient": {
                "type": "string",
                "description": "Public name of the person being addressed.",
            },
            "message": {
                "type": "string",
                "description": "Verbatim text to say.",
            },
        },
        "required": ["recipient", "message"],
        "additionalProperties": False,
    },
    risk_level="low",
    tags=frozenset({"communication"}),
)

ASK = ToolSpec(
    name="ask",
    summary=(
        f"Ask a visible human within the {HUMAN_COMMUNICATION_RADIUS_M:g} m "
        "communication radius a question and expect an answer."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "recipient": {
                "type": "string",
                "description": "Public name of the person being asked.",
            },
            "intent_id": {
                "type": "string",
                "description": (
                    "Optional exact id from the available interaction intent list. "
                    "Omit it to ask a natural-language question."
                ),
            },
            "message": {
                "type": "string",
                "description": "Verbatim question text.",
            },
        },
        "required": ["recipient", "message"],
        "additionalProperties": False,
    },
    risk_level="low",
    tags=frozenset({"communication"}),
)

REPORT_DONE = ToolSpec(
    name="report_done",
    summary=(
        "Declare the requested task complete and end the episode. If the "
        "task asked you to report a result to a named person, deliver that "
        "report with `speak` to them first, while standing near them; "
        "calling `report_done` before speaking the result leaves the report "
        "milestone unscored."
    ),
    input_schema=_no_args(),
    risk_level="medium",
    tags=frozenset({"communication", "terminal"}),
)

HANDOVER_OBJECT = ToolSpec(
    name="handover_object",
    summary="Offer the currently held object to a person and hear the reply.",
    input_schema={
        "type": "object",
        "properties": {
            "recipient": {
                "type": "string",
                "description": "Public name of the receiving person.",
            },
        },
        "required": ["recipient"],
        "additionalProperties": False,
    },
    risk_level="medium",
    tags=frozenset({"communication"}),
)


# ---------------------------------------------------------------------------
# Perception tools
# ---------------------------------------------------------------------------

LOCATE_OBJECT = ToolSpec(
    name="locate_object",
    summary=(
        "Ground one named object in the current ego frame: the best "
        "SAM3 candidate is measured and saved to spatial memory, and the "
        "annotated overlay is returned. The saved landmark is shown on the "
        "online map."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "Short visual noun phrase for the object.",
            },
        },
        "required": ["target"],
        "additionalProperties": False,
    },
    risk_level="low",
    tags=frozenset({"perception"}),
    execution_kind="information",
)


REMEMBER_ENTITY = ToolSpec(
    name="remember_entity",
    summary=(
        "Save one currently observed Human, object, or place as a semantic "
        "landmark with its current ego keyframe and measured location, also "
        "marked by name on the exploration BEV. Reusing its id updates it."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "entity_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 64,
                "description": (
                    "Stable snake_case id to create or replace, such as "
                    "`nora` or `study_doorway`."
                ),
            },
            "entity_type": {
                "type": "string",
                "enum": ["human", "object", "place"],
                "description": "Kind of visible entity being retained.",
            },
            "label": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Visible Human name, concise object phrase for grounding, "
                    "or concise place/landmark name."
                ),
            },
            "description": {
                "type": "string",
                "description": (
                    "Concise task-relevant observation, preserving uncertainty."
                ),
            },
            "image_point": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": (
                    "Required only for `place`: normalized [x,y] in the "
                    "CURRENT ego RGB, each from 0 at left/top to 1 at right/bottom."
                ),
            },
        },
        "required": ["entity_id", "entity_type", "label"],
        "additionalProperties": False,
    },
    risk_level="low",
    tags=frozenset({"memory", "perception"}),
    execution_kind="bookkeeping",
)


INSPECT_ENTITY = ToolSpec(
    name="inspect_entity",
    summary=(
        "Show the saved keyframe for one semantic landmark; this is historical "
        "evidence and does not refresh its location."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "entity_id": {
                "type": "string",
                "minLength": 1,
                "description": "Existing semantic-map entity id.",
            },
        },
        "required": ["entity_id"],
        "additionalProperties": False,
    },
    risk_level="low",
    tags=frozenset({"memory", "perception"}),
    execution_kind="information",
)


# ---------------------------------------------------------------------------
# Grasping primitives (atomic manipulation tools)
# ---------------------------------------------------------------------------

GRASP_OBJECT = ToolSpec(
    name="grasp_object",
    summary=(
        "Pick up one named object in the current ego view: SAM3 grounds the "
        "best candidate on the live frame, the overlay is returned, and the "
        "right-hand grasp executes in the same call."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": (
                    "A concise visual phrase for the object that SAM3 must "
                    "segment in the current ego RGB frame."
                ),
            },
        },
        "required": ["target"],
        "additionalProperties": False,
    },
    risk_level="high",
    tags=frozenset({"manipulation"}),
)


POINTGOAL_USAGE = (
    "Delegate obstacle-aware navigation toward a known relative "
    "PointGoal. Use it when the destination is spatially grounded and one "
    "reliable move_body is insufficient: for example, a visible destination "
    "farther than one move, a known doorway or room entry beyond a nearby "
    "turn, or a known location supported by retained observation. The "
    "PointGoal may be outside the current RGB view, but its "
    "direction and distance must not be guessed. Give it in the CURRENT body "
    "frame (metres forward/left). The result reports measured motion and "
    "the remaining offset in the updated body frame. Do not use it to search for an unknown target, for "
    "a trivial clear one-move displacement, or as a global planner through "
    "an uncertain multi-room route. "
)

NAVIGATE_TO_POINTGOAL = ToolSpec(
    name="navigate_to_pointgoal",
    summary=POINTGOAL_USAGE
    + (
        "In online-map modes the harness first "
        "tries A* to reachable ground within 0.50 m of the goal, constrained "
        "to the robot-to-goal segment with ±0.25 m lateral slack. If no route "
        "exists, X-NavDP tracks the original PointGoal with fresh RGB-D internally "
        "and returns on arrival, a position/coverage window, or its physical-step "
        "limit. A* returns after one physical prefix."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "forward_m": {
                "type": "number",
                "description": (
                    "Goal displacement in metres along the CURRENT robot "
                    "heading; positive is forward and negative is backward."
                ),
            },
            "left_m": {
                "type": "number",
                "description": (
                    "Goal displacement in metres across the CURRENT robot "
                    "body; positive is left and negative is right."
                ),
            },
            "target_description": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Concise semantic description of the grounded destination "
                    "such as 'doorway into R08 study'; recorded for diagnosis "
                    "only and never used to change the numeric PointGoal."
                ),
            },
        },
        "required": ["forward_m", "left_m", "target_description"],
        "additionalProperties": False,
    },
    risk_level="medium",
    tags=frozenset({"locomotion", "navigation"}),
)


NAVIGATE_TO_FRONTIER = ToolSpec(
    name="navigate_to_frontier",
    summary=(
        "Navigate toward one optional A/B/C visual frontier from the current "
        "ego RGB. FrontierNet ranks these candidates by predicted information "
        "gain without testing map reachability; after you choose one, the "
        "harness tries A* to ground within 0.50 m of the candidate, inside "
        "the robot-to-candidate segment with ±0.25 m lateral slack. If that "
        "partial map has no route, X-NavDP uses the original candidate "
        "PointGoal in an internal RGB-D loop, returning on arrival, a "
        "position/coverage window, or its physical-step limit. A* returns "
        "after one physical prefix. Labels refresh at the next model decision."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "enum": ["A", "B", "C"],
                "description": "One visible label from the latest frontier overlay.",
            },
            "target_description": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "Concise reason this candidate is useful for the current "
                    "task; retained for diagnosis and never changes the route."
                ),
            },
        },
        "required": ["label", "target_description"],
        "additionalProperties": False,
    },
    risk_level="medium",
    tags=frozenset({"locomotion", "navigation"}),
)


NAVIGATE_TO_ENTITY = ToolSpec(
    name="navigate_to_entity",
    summary=(
        "Follow one obstacle-aware navigation prefix toward a previously saved "
        "semantic landmark. A* searches the observed map for a route to ground "
        "within 0.50 m of the saved location, with no approach-direction "
        "restriction. If no route exists, no movement is executed and no "
        "X-NavDP fallback runs; choose an intermediate goal from current "
        "evidence. Its location is historical; after each prefix, "
        "reassess current evidence, especially for a movable Human or object."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "entity_id": {
                "type": "string",
                "minLength": 1,
                "description": "Existing semantic-map entity id.",
            },
        },
        "required": ["entity_id"],
        "additionalProperties": False,
    },
    risk_level="medium",
    tags=frozenset({"memory", "locomotion", "navigation"}),
)


# The whole callable surface. Run-level configuration selects registered
# backends; task family, holding state, and posture do not hide tools.
ALL_ATOMIC_TOOLS: tuple[ToolSpec, ...] = (
    MOVE_BODY,
    MOVE_END_EFFECTOR,
    MOVE_FORWARD,
    MOVE_BACKWARD,
    MOVE_LEFT,
    MOVE_RIGHT,
    TURN_LEFT,
    TURN_RIGHT,
    WAIT,
    CROUCH,
    STAND_UP,
    TILT_CAMERA,
    SPEAK,
    ASK,
    REPORT_DONE,
    HANDOVER_OBJECT,
    LOCATE_OBJECT,
    REMEMBER_ENTITY,
    INSPECT_ENTITY,
    GRASP_OBJECT,
    NAVIGATE_TO_POINTGOAL,
    NAVIGATE_TO_FRONTIER,
    NAVIGATE_TO_ENTITY,
)


__all__ = [
    "MOVE_BODY",
    "MOVE_END_EFFECTOR",
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "WAIT",
    "CROUCH",
    "STAND_UP",
    "TILT_CAMERA",
    "SPEAK",
    "ASK",
    "REPORT_DONE",
    "HANDOVER_OBJECT",
    "LOCATE_OBJECT",
    "REMEMBER_ENTITY",
    "INSPECT_ENTITY",
    "GRASP_OBJECT",
    "NAVIGATE_TO_POINTGOAL",
    "NAVIGATE_TO_FRONTIER",
    "NAVIGATE_TO_ENTITY",
    "ALL_ATOMIC_TOOLS",
]
