"""Object-class grasp templates and model overrides for certification.

``resolve_grasp_pose`` combines a reviewed template with live target bounds.
Classes without an entry raise an authoring error. Instance suffixes are removed
for model-override lookup, so instances of one asset share its template.
The evaluated semantic grasp tool generates poses with GraspGenX."""

from __future__ import annotations

import re
from typing import Dict

from tarsphere.grasp_poses import CanonicalGrasp


# An object id is ``<class>/model_<hash>_<instance>``, and its first segment is
# the same word the episode's ``category`` field carries. The instance suffix is
# stripped for override lookup so every instance of one model shares a grasp.
_OBJECT_ID = re.compile(
    r"^(?P<object_class>[^/]+)/(?P<model>.+?)(?:_(?P<instance>\d+))?$"
)


# Upright bodies the hand can span: approached from whichever side the robot
# stands on, closing at 0.60 of the object's own height.
#
# This is the grasp live certification executed on the golden bottle
# (7.1 x 7.1 x 10.4 cm on a tray, and a 7.7 x 7.7 x 24.5 cm variant): the same
# 0.60 fraction of the same live AABB the promoted oracle reach used, so a
# `grasp_object` on a bottle reproduces a physically verified grasp rather than a
# new one. The 0.10 m opening covers both shipped models with margin for a
# scaled variant; anything wider is refused at resolve time instead of closing
# on air.
_BOTTLE = CanonicalGrasp(
    approach="frontal_horizontal",
    closing_center="body_center",
    height_fraction=0.60,
    standoff_m=0.12,
    jaw_span_m=0.10,
)

# Flat open vessels: too wide to span across the body, so they are taken from
# above by the robot-facing rim, with a quarter turn of wrist roll putting the
# jaws across the rim wall so one finger lands inside the vessel and the other
# outside it. The 2 cm inset is the reach past the rim a woven or moulded edge
# needs to end up between the fingers.
#
# Authored from the shipped tray's geometry (19 x 19 x 2 cm), not certified:
# the golden episodes list the tray in the pickable pool but grasp the bottle
# standing on it, so no live rollout has closed a hand on a tray. An episode
# that makes a tray the grasp target needs a certification rollout first.
_FLAT_VESSEL = CanonicalGrasp(
    approach="top_down",
    closing_center="facing_rim",
    rim_inset_m=0.02,
    height_fraction=0.50,
    standoff_m=0.12,
    jaw_span_m=0.06,
    wrist_roll_rad=1.5707963267948966,
)


CANONICAL_GRASPS_BY_CLASS: Dict[str, CanonicalGrasp] = {
    "bottle": _BOTTLE,
    "tray": _FLAT_VESSEL,
}

# Overrides keyed ``<class>/<model>`` for one model that its class grasp gets
# wrong — a class-atypical shape that still belongs to the class. Empty today:
# both shipped bottle models are plain upright bodies. An entry here wins over
# the class, and nothing else does.
CANONICAL_GRASPS_BY_MODEL: Dict[str, CanonicalGrasp] = {}


def object_class_of(object_id: str) -> str:
    """Return the class segment of one object id."""
    match = _OBJECT_ID.match(str(object_id).strip())
    if match is None:
        raise ValueError(
            f"{object_id!r} is not an object id of the form "
            "<class>/model_<hash>_<instance>"
        )
    return match.group("object_class")


def lookup_canonical_grasp(object_id: str) -> CanonicalGrasp:
    """Return the reviewed grasp for one object id.

    Raises:
        ValueError: when the id is malformed, or when neither the model nor its
            class has an authored grasp. The caller must report that as a
            refused call — this benchmark would rather say it cannot grasp an
            object than guess how.
    """
    match = _OBJECT_ID.match(str(object_id).strip())
    if match is None:
        raise ValueError(
            f"{object_id!r} is not an object id of the form "
            "<class>/model_<hash>_<instance>"
        )
    object_class = match.group("object_class")
    model_key = f"{object_class}/{match.group('model')}"
    override = CANONICAL_GRASPS_BY_MODEL.get(model_key)
    if override is not None:
        return override
    canonical = CANONICAL_GRASPS_BY_CLASS.get(object_class)
    if canonical is None:
        raise ValueError(
            f"no canonical grasp is authored for object class "
            f"{object_class!r}. Grasped classes are "
            f"{sorted(CANONICAL_GRASPS_BY_CLASS)}. Add the class to "
            "tarsphere.infra.manipulation.grasp_catalog before an episode makes it a "
            "grasp target."
        )
    return canonical


__all__ = [
    "CANONICAL_GRASPS_BY_CLASS",
    "CANONICAL_GRASPS_BY_MODEL",
    "lookup_canonical_grasp",
    "object_class_of",
]
