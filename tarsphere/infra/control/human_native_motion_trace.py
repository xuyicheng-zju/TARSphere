"""Optional read-only receipts of installed People command/navigation state.

Native vectors are retained in stage units. This module never imports Kit,
advances a clock, writes an animation variable, or substitutes unavailable data.
"""

from __future__ import annotations

import math


def json_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite native motion measurement.")
        return value
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if all(hasattr(value, axis) for axis in ("x", "y", "z")):
        axes = ("x", "y", "z", "w") if hasattr(value, "w") else ("x", "y", "z")
        return json_value([float(getattr(value, axis)) for axis in axes])
    raise TypeError(f"Unsupported native measurement: {type(value).__name__}")


def measured(read):
    try:
        return {"status": "available", "value": json_value(read())}
    except Exception as exc:
        return {
            "status": "unavailable",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def read_character_transform(character, joint_name=None, *, vector_factory=None):
    """Read the installed output-argument API; untouched outputs cannot pass.

    Joint queries use the real catalog joint leaf names, not a guessed pose.
    Factories keep the serializer importable without the native runtime.
    """
    if vector_factory is None:
        import carb

        vector_factory = lambda size: (carb.Float3 if size == 3 else carb.Float4)(
            *([float("nan")] * size)
        )
    position, rotation = vector_factory(3), vector_factory(4)
    if joint_name is None:
        result = character.get_world_transform(position, rotation)
    else:
        result = character.get_joint_transform(joint_name, position, rotation)
    if result is False:
        raise ValueError(f"Native transform query returned false for {joint_name!r}.")
    position, rotation = json_value(position), json_value(rotation)
    if sum(value * value for value in rotation) <= 0:
        raise ValueError("Native transform quaternion is unavailable or zero.")
    return {
        "position_stage_units": position,
        "rotation_xyzw": rotation,
        "interface_return": json_value(result),
    }


def read_animation_clock():
    """Observe the installed timeline without advancing it."""
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    return {
        "time_s": float(timeline.get_current_time()),
        "time_codes_per_second": float(timeline.get_time_codes_per_seconds()),
        "playing": bool(timeline.is_playing()),
    }


def read_native_motion(instance):
    """Read each optional installed interface independently; no control calls."""
    manager = getattr(instance, "navigation_manager", None)
    character = getattr(instance, "character", None)

    def command():
        current = instance.current_command
        if current is None:
            return None
        return {
            "name": current.get_command_name(),
            **{
                name: measured(lambda name=name: getattr(current, name))
                for name in (
                    "command_id",
                    "command",
                    "time_elapsed",
                    "is_setup",
                    "finished",
                    "actual_walk_speed",
                    "desired_walk_speed",
                    "current_action",
                    "sit_time",
                )
            },
        }

    def registry():
        registry = manager.character_manager
        return {
            str(name): {
                field: measured(
                    lambda method=method, name=name: getattr(registry, method)(name)
                )
                for field, method in (
                    ("current_position", "get_character_current_pos"),
                    ("future_position", "get_character_future_pos"),
                    ("radius", "get_character_radius"),
                )
            }
            for name in registry.get_all_managed_characters()
        }

    return {
        "native_position_stage_units": measured(
            lambda: instance.get_current_position()
        ),
        "character_world_transform": measured(
            lambda: read_character_transform(character)
        ),
        "joint_world_transforms": {
            name: measured(lambda name=name: read_character_transform(character, name))
            for name in ("Hip", "Pelvis", "L_Thigh", "R_Thigh", "L_Foot", "R_Foot")
        },
        "animation_clock": measured(read_animation_clock),
        "current_command": measured(command),
        "command_queue": measured(lambda: instance.commands),
        "navigation_manager": {
            name: measured(lambda name=name: getattr(manager, name))
            for name in (
                "character_name",
                "navmesh_enabled",
                "dynamic_avoidance_enabled",
                "path_points",
                "path_targets",
                "path_final_target_rot",
                "collision_list",
                "velocity_vec",
                "delta_time_list",
            )
        },
        "animation_variables": {
            name: measured(lambda name=name: character.get_variable(name))
            for name in ("Action", "Walk", "PathPoints", "SitWeight")
        },
        "crowd_registry": measured(registry),
        "measurement_scope": "Read-only native fields in stage units after the named boundary; no motion success, timeline advance, or seat-contact claim.",
    }
