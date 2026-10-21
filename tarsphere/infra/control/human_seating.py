"""Measured standing prerequisites for a native, stationary-root Sit clip."""

from __future__ import annotations

import math
from collections import deque

# Maximum residual root travel permitted when the native Sit blend starts.
SEAT_STANCE_TOLERANCE_M = 0.03
# Maximum residual orientation change permitted at the Walk-to-Sit handoff.
SEAT_YAW_TOLERANCE_RAD = math.radians(3.0)
# Native approach must turn over time; a one-frame final orientation snap fails.
SEAT_APPROACH_MAX_YAW_RATE_RAD_S = math.radians(90.0)
# Native nlerp can peak above its mean angular speed. 60 deg/s mean keeps
# its worst 180-degree peak (4/pi times mean) below the 90 deg/s guard.
SEAT_NATIVE_MEAN_YAW_RATE_RAD_S = math.radians(60.0)
# Start native stopping inside this radius; measured final stance still must
# satisfy the independent 3 cm gate after native deceleration completes.
SEAT_NATIVE_FINAL_TARGET_DISTANCE_M = 0.01


def yaw_error_rad(actual: float, desired: float) -> float:
    if not all(math.isfinite(v) for v in (actual, desired)):
        raise ValueError("Seat orientation must be finite.")
    return abs((actual - desired + math.pi) % (2 * math.pi) - math.pi)


def seat_stance_error(position, yaw_rad: float, anchor, desired_yaw_rad: float) -> str:
    if (
        len(position) != 3
        or len(anchor) != 3
        or not all(math.isfinite(v) for v in (*position, *anchor))
    ):
        raise ValueError("Seat stance requires finite world XYZ coordinates.")
    distance = math.dist(position, anchor)
    angle = yaw_error_rad(yaw_rad, desired_yaw_rad)
    if distance > SEAT_STANCE_TOLERANCE_M:
        return f"Seat stance is {distance:.4f} m from its standing anchor."
    if angle > SEAT_YAW_TOLERANCE_RAD:
        return f"Seat stance yaw differs by {math.degrees(angle):.3f} degrees."
    return ""


def native_sit_stance_error(position, yaw_rad, binding, *, phase, position_blend):
    """Gate original Sit's short position blend; final stance remains 3cm/3deg."""
    walk, interact = binding["walk_to_world_m"], binding["interact_world_m"]
    desired = math.radians(binding["character_yaw_deg"])
    if math.dist(walk, interact) < 1e-5 or phase not in {"sit", "stand"}:
        return seat_stance_error(position, yaw_rad, interact, desired)
    if not math.isfinite(position_blend) or not 0 <= position_blend <= 1:
        raise ValueError("Native Sit position blend must be finite inside0..1.")
    if position_blend >= 1:
        return seat_stance_error(
            position, yaw_rad, walk if phase == "stand" else interact, desired
        )
    delta = [b - a for a, b in zip(walk, interact)]
    length_squared = sum(v * v for v in delta)
    fraction = max(
        0.0,
        min(
            1.0,
            sum((p - a) * d for p, a, d in zip(position, walk, delta)) / length_squared,
        ),
    )
    corridor = [a + fraction * d for a, d in zip(walk, delta)]
    # Permit only the actual native short approach corridor. This does not
    # set/interpolate any root pose or allow a wider yaw gate.
    return seat_stance_error(position, yaw_rad, corridor, desired)


def native_sit_weight_complete(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return (
        isinstance(value, (int, float))
        and math.isfinite(value)
        and abs(value - 1.0) <= 1e-3
    )


def seat_turn_error(previous_yaw: float, yaw: float, elapsed_s: float) -> str:
    if not math.isfinite(elapsed_s) or elapsed_s <= 0:
        raise ValueError(
            "Seat approach samples must have strictly increasing timestamps."
        )
    rate = yaw_error_rad(yaw, previous_yaw) / elapsed_s
    if rate > SEAT_APPROACH_MAX_YAW_RATE_RAD_S:
        return f"Native seat approach yaw rate {math.degrees(rate):.3f} deg/s exceeds 90 deg/s."
    return ""


class SeatTurnMonitor:
    """Measure wrapped angular travel over native updates spanning 50 ms.

    Poll at every physics tick to reject a snap immediately. During startup,
    reserve a full window for the first quantized native update as well.
    """

    window_s = 0.05

    def __init__(self):
        self.reset()

    def reset(self):
        self._samples = deque()
        self._travel = 0.0
        self._last_poll = None

    def sample(self, yaw: float, timestamp_s: float) -> str:
        if not math.isfinite(timestamp_s):
            raise ValueError("Seat approach timestamps must be finite.")
        yaw_error_rad(yaw, yaw)
        if self._last_poll is not None and timestamp_s <= self._last_poll:
            raise ValueError(
                "Seat approach samples must have strictly increasing timestamps."
            )
        self._last_poll = timestamp_s
        if self._samples:
            previous_time, previous_yaw, _ = self._samples[-1]
            change = yaw_error_rad(yaw, previous_yaw)
            if change == 0:
                if timestamp_s - previous_time >= self.window_s:
                    # A long stationary interval must not dilute a later snap.
                    self._samples.clear()
                    self._samples.append((timestamp_s, yaw, self._travel))
                return ""
            self._travel += change
        self._samples.append((timestamp_s, yaw, self._travel))
        cutoff = timestamp_s - self.window_s
        # Store actual yaw updates, not held physics polls. Retain the update
        # at or before the window boundary to measure a complete native span.
        while len(self._samples) > 1 and self._samples[1][0] <= cutoff:
            self._samples.popleft()
        start, _, travel = self._samples[0]
        elapsed = max(self.window_s, timestamp_s - start)
        rate = (self._travel - travel) / elapsed
        if rate > SEAT_APPROACH_MAX_YAW_RATE_RAD_S:
            return f"Native seat approach yaw rate {math.degrees(rate):.3f} deg/s exceeds 90 deg/s."
        return ""


def native_seat_turn_duration_s(actual_yaw: float, desired_yaw: float) -> float:
    angle = yaw_error_rad(actual_yaw, desired_yaw)
    return max(0.25, angle / SEAT_NATIVE_MEAN_YAW_RATE_RAD_S)


def validate_terminal_approach(config):
    if config is None:
        return None
    if not {"anchors_world_m", "max_gap_m"} <= set(config) or set(config) - {
        "anchors_world_m",
        "max_gap_m",
        "via_points_world_m",
        "max_chain_m",
        "max_segment_m",
    }:
        raise ValueError(
            "Terminal approach requires an explicit anchor whitelist and gap bound."
        )
    gap = config["max_gap_m"]
    anchors = config["anchors_world_m"]
    if (
        not isinstance(gap, (float, int))
        or not math.isfinite(gap)
        or not 0 < gap <= 0.30
    ):
        raise ValueError(
            "Terminal approach max_gap_m must be positive and at most0.30m."
        )
    if not anchors or any(
        len(a) != 3 or not all(math.isfinite(v) for v in a) for a in anchors
    ):
        raise ValueError("Terminal approach requires finite world XYZ anchors.")
    via = config.get("via_points_world_m", [])
    if len(via) > 1 or any(
        len(p) != 3 or not all(math.isfinite(v) for v in p) for p in via
    ):
        raise ValueError("Terminal approach permits at most one finite via point.")
    if via:
        for key, maximum in [("max_segment_m", 0.55), ("max_chain_m", 0.60)]:
            value = config.get(
                key, config["max_gap_m"] if key == "max_segment_m" else None
            )
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 < value <= maximum
            ):
                raise ValueError(
                    f"Terminal approach {key} must be finite positive at most{maximum}m."
                )
    return config


def terminal_approach_anchor(point, terminal_approach, *, meters_per_unit=1.0):
    if not terminal_approach:
        return False
    return any(
        math.dist(point, [v / meters_per_unit for v in anchor]) * meters_per_unit < 1e-5
        for anchor in terminal_approach["anchors_world_m"]
    )


def validate_terminal_projection(
    anchor, projection, terminal_approach, *, meters_per_unit=1.0
):
    if not terminal_approach_anchor(
        anchor, terminal_approach, meters_per_unit=meters_per_unit
    ):
        raise RuntimeError("Terminal approach endpoint is not explicitly whitelisted.")
    gap = math.dist(anchor, projection) * meters_per_unit
    if gap > terminal_approach["max_gap_m"]:
        raise RuntimeError(
            "Whitelisted terminal approach gap exceeds its explicit bound."
        )
    floor = min(a[2] for a in terminal_approach["anchors_world_m"])
    if any(
        abs(p[2] * meters_per_unit - floor) > SEAT_STANCE_TOLERANCE_M
        for p in (anchor, projection)
    ):
        raise RuntimeError(
            "Whitelisted terminal approach is not on the floor within3cm."
        )
    return gap


def terminal_chain_points(anchor, projection, config, *, meters_per_unit=1.0):
    """Native prefix ends at the via projection; bound only added final steps."""
    via = [
        [v / meters_per_unit for v in p] for p in config.get("via_points_world_m", [])
    ]
    chain = [projection, *via, anchor]
    lengths = [math.dist(a, b) * meters_per_unit for a, b in zip(chain, chain[1:])]
    if any(
        d > config.get("max_segment_m", config["max_gap_m"]) + 1e-9 for d in lengths
    ):
        raise RuntimeError("Explicit terminal chain segment exceeds its bound.")
    if sum(lengths) > config.get("max_chain_m", config["max_gap_m"]) + 1e-9:
        raise RuntimeError("Explicit terminal chain total exceeds its bound.")
    floor = min(a[2] for a in config["anchors_world_m"])
    if any(
        abs(p[2] * meters_per_unit - floor) > SEAT_STANCE_TOLERANCE_M for p in chain
    ):
        raise RuntimeError("Explicit terminal chain crosses a non-floor surface.")
    return chain


def _point_like(point, values):
    if isinstance(point, list):
        return list(values)
    if isinstance(point, tuple):
        return tuple(values)
    return type(point)(*values)


def _install_owned_navmesh_radius(
    navigation,
    radius_m,
    *,
    terminal_approach=None,
    meters_per_unit=1.0,
    event=lambda *args, **kwargs: None,
):
    """Clone only this NPC's SDK namespaces; never mutate SDK modules/classes."""
    from types import FunctionType, MethodType

    if not math.isfinite(radius_m) or not 0.2 <= radius_m <= 0.5:
        raise ValueError("Native NPC radius must be inside the baked 0.2..0.5 m range.")
    validate_terminal_approach(terminal_approach)
    original_mesh = navigation.navmesh
    if original_mesh is None:
        raise ValueError("Native NPC radius requires an existing NavMesh.")
    restorations = []

    class PathProxy:
        def __init__(self, original, points):
            self.original, self.points = original, points

        def get_points(self):
            return self.points

        def __getattr__(self, name):
            return getattr(self.original, name)

    class MeshProxy:
        def __getattr__(self, name):
            return getattr(original_mesh, name)

        def projection(self, point):
            closest, island = original_mesh.query_closest_point(
                point, agent_radius=radius_m
            )
            if closest is None:
                raise RuntimeError(
                    "Whitelisted terminal approach has no native floor projection."
                )
            gap = validate_terminal_projection(
                point, closest, terminal_approach, meters_per_unit=meters_per_unit
            )
            event(
                "native_terminal_approach_projection",
                anchor=[float(v) * meters_per_unit for v in point],
                native_projection=[float(v) * meters_per_unit for v in closest],
                gap_m=gap,
                max_gap_m=terminal_approach["max_gap_m"],
            )
            return closest, island

        def prefix(self, point):
            projection, _ = self.projection(point)
            via = terminal_approach.get("via_points_world_m", [])
            if via:
                entry = _point_like(point, [v / meters_per_unit for v in via[0]])
                entry_projection, _ = original_mesh.query_closest_point(
                    entry, agent_radius=radius_m
                )
                if (
                    entry_projection is None
                    or math.dist(entry, entry_projection) * meters_per_unit
                    > SEAT_STANCE_TOLERANCE_M
                ):
                    raise RuntimeError(
                        "Explicit terminal via point does not project to the native floor within3cm."
                    )
                chain = terminal_chain_points(
                    point,
                    entry_projection,
                    terminal_approach,
                    meters_per_unit=meters_per_unit,
                )
                event(
                    "native_terminal_via_projection",
                    entry_world_m=via[0],
                    native_projection_world_m=[
                        float(v) * meters_per_unit for v in entry_projection
                    ],
                    seat_raw_projection_world_m=[
                        float(v) * meters_per_unit for v in projection
                    ],
                    terminal_chain_world_m=[
                        [float(v) * meters_per_unit for v in p] for p in chain
                    ],
                )
                return entry_projection, [_point_like(point, p) for p in chain[1:]]
            return projection, [point]

        def query_closest_point(self, *args, **kwargs):
            kwargs["agent_radius"] = radius_m
            point = args[0] if args else kwargs.get("target")
            if point is not None and terminal_approach_anchor(
                point, terminal_approach, meters_per_unit=meters_per_unit
            ):
                _, island = self.projection(point)
                # SDK Utils must preserve the explicit terminal destination;
                # this is a route endpoint, never a pose/transform substitute.
                return point, island
            return original_mesh.query_closest_point(*args, **kwargs)

        def query_shortest_path(self, *args, **kwargs):
            kwargs["agent_radius"] = radius_m
            endpoints = (
                list(args[:2])
                if len(args) >= 2
                else [kwargs.get("start_pos"), kwargs.get("end_pos")]
            )
            if any(p is None for p in endpoints):
                return original_mesh.query_shortest_path(*args, **kwargs)
            bridge = [
                terminal_approach_anchor(
                    p, terminal_approach, meters_per_unit=meters_per_unit
                )
                for p in endpoints
            ]
            if not any(bridge):
                return original_mesh.query_shortest_path(*args, **kwargs)
            plans = [
                self.prefix(p) if use else (p, []) for p, use in zip(endpoints, bridge)
            ]
            projected = [plan[0] for plan in plans]
            if len(args) >= 2:
                path = original_mesh.query_shortest_path(
                    *projected, *args[2:], **kwargs
                )
            else:
                kwargs.update(start_pos=projected[0], end_pos=projected[1])
                path = original_mesh.query_shortest_path(**kwargs)
            points = list(path.get_points()) if path is not None else []
            if not points:
                raise RuntimeError(
                    "Native projected terminal approach route is unreachable; no fallback."
                )
            for index, use in enumerate(bridge):
                if (
                    use
                    and math.dist(points[0 if index == 0 else -1], projected[index])
                    * meters_per_unit
                    > SEAT_STANCE_TOLERANCE_M
                ):
                    raise RuntimeError(
                        "Native path misses its validated terminal projection."
                    )
            if terminal_approach and any(
                abs(
                    p[2] * meters_per_unit
                    - min(a[2] for a in terminal_approach["anchors_world_m"])
                )
                > SEAT_STANCE_TOLERANCE_M
                for p in points
            ):
                raise RuntimeError(
                    "Native projected terminal route crosses a non-floor surface."
                )
            if bridge[0]:
                extra = list(reversed(plans[0][1]))
                if extra and math.dist(extra[-1], points[0]) * meters_per_unit < 1e-6:
                    extra.pop()
                points = extra + points
            if bridge[1]:
                extra = list(plans[1][1])
                if extra and math.dist(points[-1], extra[0]) * meters_per_unit < 1e-6:
                    extra.pop(0)
                points.extend(extra)
            event(
                "native_terminal_approach_route",
                points_world_m=[
                    [float(v) * meters_per_unit for v in p] for p in points
                ],
                prepended_start=bridge[0],
                appended_target=bridge[1],
            )
            return PathProxy(path, points)

    mesh = MeshProxy()

    class InterfaceProxy:
        def __init__(self, original):
            self.original = original

        def get_navmesh(self):
            return mesh

        def __getattr__(self, name):
            return getattr(self.original, name)

    class NavigationProxy:
        def __init__(self, original):
            self.original = original

        def acquire_interface(self):
            return InterfaceProxy(self.original.acquire_interface())

        def __getattr__(self, name):
            return getattr(self.original, name)

    def clone(function):
        namespace = dict(function.__globals__)
        for name in ("nav", "Utils", "SimulationUtil"):
            if name in namespace:
                original = namespace[name]
                namespace[name] = (
                    NavigationProxy(original)
                    if name == "nav"
                    else UtilityProxy(original)
                )
        result = FunctionType(
            function.__code__,
            namespace,
            function.__name__,
            function.__defaults__,
            function.__closure__,
        )
        result.__kwdefaults__ = function.__kwdefaults__
        return result

    class UtilityProxy:
        def __init__(self, original):
            self.original = original

        def __getattr__(self, name):
            value = getattr(self.original, name)
            function = getattr(value, "__func__", value)
            if isinstance(function, FunctionType):
                function = clone(function)
                return (
                    MethodType(function, value.__self__)
                    if hasattr(value, "__self__")
                    else function
                )
            return value

    def adapt(instance):
        # The exact installed SDK methods retain their code/defaults/closures.
        # Only private bound method namespaces redirect utility NavMesh queries.
        for name in dir(type(instance)):
            if name.startswith("__"):
                continue
            value = getattr(instance, name, None)
            function = getattr(value, "__func__", None)
            if not isinstance(function, FunctionType) or not any(
                key in function.__code__.co_names
                for key in ("nav", "Utils", "SimulationUtil")
            ):
                continue
            own = name in vars(instance)
            original = vars(instance).get(name)
            restorations.append((instance, name, own, original))
            setattr(instance, name, MethodType(clone(function), value.__self__))

    adapt(navigation)
    navigation.navmesh = mesh

    def restore():
        navigation.navmesh = original_mesh
        for instance, name, own, original in reversed(restorations):
            if own:
                setattr(instance, name, original)
            else:
                delattr(instance, name)

    return adapt, restore


def install_native_seat_navigation(
    instance,
    *,
    meters_per_unit: float,
    enabled,
    event,
    native_radius_m: float | None = None,
    terminal_approach=None,
    final_target_distance_m: float = SEAT_NATIVE_FINAL_TARGET_DISTANCE_M,
):
    """Adapt this NPC's verified People 0.7.9 entry points; return restoration.

    CharacterBehavior.get_command runs before execute. Hooking this factory
    therefore fixes rotation timing before even the first native command tick.
    All transforms and skeletal playback remain in original SDK methods.
    No global SDK settings, classes or installed source files are changed.
    """
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError("Native seat navigation requires positive meters_per_unit.")
    if (
        not math.isfinite(final_target_distance_m)
        or not 0.01 <= final_target_distance_m <= 0.05
    ):
        raise ValueError(
            "Owned native stopping trigger must be within0.01..0.05m; final stance remains3cm."
        )
    navigation = instance.navigation_manager
    original_factory = instance.get_command
    original_proximity = navigation.check_proximity_to_point
    rotations = []
    adapt_radius, restore_radius = (lambda current: None), (lambda: None)
    if native_radius_m is not None:
        adapt_radius, restore_radius = _install_owned_navmesh_radius(
            navigation,
            native_radius_m,
            terminal_approach=terminal_approach,
            meters_per_unit=meters_per_unit,
            event=event,
        )

    def proximity(point, distance):
        if enabled() and len(navigation.path_targets) == 1:
            distance = min(distance, final_target_distance_m / meters_per_unit)
        return original_proximity(point, distance)

    def factory(command_pair):
        current = original_factory(command_pair)
        if current is None or current.get_command_name() not in {"GoTo", "Sit"}:
            return current
        adapt_radius(current)
        original_rotate = current.rotate

        def rotate(dt):
            if enabled() and current.set_rotation is not True:
                from omni.anim.people.scripts.utils import Utils

                actual = math.radians(
                    Utils.convert_to_angle(Utils.get_character_rot(current.character))
                )
                target = math.radians(
                    Utils.convert_to_angle(
                        current.navigation_manager.get_path_target_rot()
                    )
                )
                # The SDK's unwrapped angle subtraction can return a negative
                # duration (e.g. -340.77 to +180 degrees) and instantly jump.
                # Set its existing timing fields, then call original nlerp.
                current.rotation_time_threshold = native_seat_turn_duration_s(
                    actual, target
                )
                current.rotation_time = 0.0
                current.char_start_rot = None
                current.set_rotation = True
                event(
                    "native_seat_turn_started",
                    command=current.get_command_name(),
                    wrapped_angle_deg=math.degrees(yaw_error_rad(actual, target)),
                    duration_s=current.rotation_time_threshold,
                )
            return original_rotate(dt)

        current.rotate = rotate
        rotations.append((current, original_rotate))
        return current

    instance.get_command = factory
    navigation.check_proximity_to_point = proximity
    event(
        "native_seat_navigation_installed",
        scope="one CharacterBehavior instance",
        final_target_distance_m=final_target_distance_m,
        mean_turn_rate_deg_s=math.degrees(SEAT_NATIVE_MEAN_YAW_RATE_RAD_S),
        native_radius_m=native_radius_m,
        terminal_approach=terminal_approach,
    )

    def restore():
        instance.get_command = original_factory
        navigation.check_proximity_to_point = original_proximity
        for current, original_rotate in rotations:
            current.rotate = original_rotate
        restore_radius()
        event("native_seat_navigation_restored")

    return restore
