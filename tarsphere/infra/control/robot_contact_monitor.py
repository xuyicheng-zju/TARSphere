"""Native PhysX contact evidence for robot navigation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

from tarsphere.infra.control.physical_contracts import SIMULATION_PHYSICS_HZ


# Ordinary floor support is not a navigation collision.  A contact whose
# normal is this close to vertical is therefore excluded from collision
# evidence; furniture and wall contacts have predominantly horizontal normals.
_FLOOR_CONTACT_NORMAL_Z = 0.80
# Benchmark safety policy. A quarter-second continuous body contact is a
# sustained press rather than an incidental brush. A 1 N*s normal impulse is
# separately treated as a clear impact. Both are evaluator/runtime evidence,
# not collision-avoidance padding around the Human.
HUMAN_CONTACT_SUSTAINED_SECONDS = 0.25
HUMAN_CONTACT_SUSTAINED_REPORTS = round(
    HUMAN_CONTACT_SUSTAINED_SECONDS * SIMULATION_PHYSICS_HZ
)
HUMAN_IMPACT_NORMAL_IMPULSE_NS = 1.0


@dataclass(frozen=True)
class RobotContact:
    """One native PhysX contact between the robot and the episode world.

    Attributes:
        robot_path: Robot actor or collider path reported by PhysX.
        other_path: World actor or collider path touching the robot.
        position_xyz: World-space contact point in metres.
        normal_xyz: PhysX contact normal in world coordinates.
        separation_m: Signed collider separation reported by PhysX, or unknown.
        human_id: Human identity when the other collider is a Human proxy.
    """

    robot_path: str
    other_path: str
    position_xyz: tuple[float, float, float]
    normal_xyz: tuple[float, float, float]
    separation_m: float | None
    normal_impulse_ns: float
    human_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible evidence for traces and evaluators."""

        return asdict(self)


@dataclass(frozen=True)
class HumanSafetyViolation:
    """One unsafe robot--Human contact incident recorded for evaluation."""

    human_id: str
    category: str
    contact: RobotContact
    contact_onset_count: int
    consecutive_contact_reports: int

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["contact"] = self.contact.as_dict()
        return payload


class RobotContactMonitor:
    """Record real robot-world contacts without predicting or blocking motion.

    The monitor has no occupancy map, safety radius, route, goal, or future
    action segment.  PhysX remains responsible for collision response.  This
    class only records contact onsets and exposes them as Agent feedback and
    navigation-safety evidence.
    """

    def __init__(
        self,
        *,
        robot_root_path: str,
        human_proxy_paths: Mapping[str, str] | None = None,
        is_retention_contact: Callable[[str, str], bool] | None = None,
    ) -> None:
        self.robot_root_path = _normalized_path(robot_root_path)
        self._is_retention_contact = is_retention_contact
        self._retention_contacts: dict[tuple[str, str], RobotContact] = {}
        self.human_proxy_paths = {
            str(human_id): _normalized_path(path)
            for human_id, path in (human_proxy_paths or {}).items()
        }
        self.collision_count = 0
        self.human_contact_count = 0
        self.human_safety_violation_count = 0
        # Episode-cumulative native contact evidence, split before any future
        # release-defined Human/severity weighting. Pair durations add across
        # simultaneous colliders; they are not wall-clock contact time.
        self._contact_totals = {
            target: {
                "collider_pair_events": 0,
                "contact_reports": 0,
                "normal_impulse_ns": 0.0,
                "touch_pair_events": 0,
                "touch_reports": 0,
                "unknown_touch_reports": 0,
            }
            for target in ("scene", "human")
        }
        # A retained incident ledger permits later severity calibration from
        # complete impulse/duration evidence rather than only group averages.
        self._contact_incidents: list[dict[str, Any]] = []
        self._active_contact_incidents: dict[tuple[str, str], int] = {}
        self._subscription: Any = None
        self._contacts: dict[tuple[str, str], RobotContact] = {}
        self._contact_report_counts: dict[tuple[str, str], int] = {}
        self._human_contact_onsets: dict[str, int] = {}
        self._active_human_contact_keys: dict[str, set[tuple[str, str]]] = {}
        self._unsafe_human_contacts: set[str] = set()
        # Latest report per collider pair during the observation window;
        # includes sustained contacts, even if they end before the drain.
        self._window_contacts: dict[tuple[str, str], RobotContact] = {}
        self._new_human_safety_violations: list[HumanSafetyViolation] = []
        self._human_safety_violations: list[HumanSafetyViolation] = []
        self._callback_error: Exception | None = None

    def start(self, stage: Any) -> None:
        """Subscribe after reset without mutating the live articulation USD."""

        if self._subscription is not None:
            raise RuntimeError("Robot contact monitor is already running.")
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysxSchema, Usd, UsdPhysics

        robot_root = stage.GetPrimAtPath(self.robot_root_path)
        if not robot_root or not robot_root.IsValid():
            raise RuntimeError(
                f"Cannot monitor missing robot root {self.robot_root_path!r}."
            )

        articulation_roots = [
            prim
            for prim in Usd.PrimRange.AllPrims(robot_root)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        if len(articulation_roots) != 1:
            raise RuntimeError(
                "Robot contact reporting requires exactly one articulation "
                f"under {self.robot_root_path!r}; found "
                f"{[str(prim.GetPath()) for prim in articulation_roots]}."
            )
        articulation_root = articulation_roots[0]
        if not articulation_root.HasAPI(PhysxSchema.PhysxContactReportAPI):
            raise RuntimeError(
                "Robot contact reporting was not authored on the articulation "
                f"before it became live: {articulation_root.GetPath()}."
            )
        rigid_body_count = 0
        for prim in Usd.PrimRange.AllPrims(robot_root):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            rigid_body_count += 1
        if rigid_body_count == 0:
            raise RuntimeError(
                f"Robot root {self.robot_root_path!r} has no rigid bodies."
            )

        interface = get_physx_simulation_interface()
        if interface is None:
            raise RuntimeError(
                "PhysX simulation interface is unavailable for robot contacts."
            )
        self._subscription = interface.subscribe_contact_report_events(
            self._on_contact_report
        )
        if self._subscription is None:
            raise RuntimeError("PhysX robot contact subscription could not be created.")

    def close(self) -> None:
        """Release the subscription and transient contact state."""

        self._subscription = None
        self._contacts.clear()
        self._retention_contacts.clear()
        self._contact_report_counts.clear()
        self._active_contact_incidents.clear()
        self._human_contact_onsets.clear()
        self._active_human_contact_keys.clear()
        self._unsafe_human_contacts.clear()
        self._window_contacts.clear()
        self._new_human_safety_violations.clear()
        self._human_safety_violations.clear()
        self._callback_error = None

    def drain_contacts(self) -> tuple[RobotContact, ...]:
        """Return measured contacts since the preceding drain, including persists."""

        self._raise_callback_error()
        contacts = tuple(self._window_contacts.values())
        self._window_contacts.clear()
        return contacts

    def drain_human_safety_violations(
        self,
    ) -> tuple[HumanSafetyViolation, ...]:
        """Return newly classified unsafe Human contacts exactly once."""

        self._raise_callback_error()
        violations = tuple(self._new_human_safety_violations)
        self._new_human_safety_violations.clear()
        return violations

    def feedback(
        self,
        contacts: Sequence[RobotContact],
        *,
        robot_position: Sequence[float],
        robot_yaw_rad: float,
    ) -> str:
        """Describe measured contact locations in the robot body frame."""

        if not contacts:
            return ""
        descriptions: list[str] = []
        for details in self.public_details(
            contacts,
            robot_position=robot_position,
            robot_yaw_rad=robot_yaw_rad,
        ):
            source = (
                f"Human {details['human_id']}"
                if details.get("human_id") is not None
                else "scene geometry"
            )
            descriptions.append(
                f"{source} at {details['direction']} "
                f"({float(details['relative_bearing_deg']):+.1f} deg; "
                "positive is left)"
            )
        return (
            "Physical contact occurred during the preceding body motion: "
            + "; ".join(descriptions)
            + "."
        )

    @staticmethod
    def public_details(
        contacts: Sequence[RobotContact],
        *,
        robot_position: Sequence[float],
        robot_yaw_rad: float,
    ) -> list[dict[str, Any]]:
        """Project contact points to non-oracle body-relative feedback."""

        robot_xy = _xy(robot_position, name="robot_position")
        yaw = float(robot_yaw_rad)
        if not math.isfinite(yaw):
            raise RuntimeError("Robot contact feedback requires finite yaw.")
        nearest_by_region: dict[tuple[str, str | None, str], dict[str, Any]] = {}
        for contact in contacts:
            dx = contact.position_xyz[0] - robot_xy[0]
            dy = contact.position_xyz[1] - robot_xy[1]
            local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
            local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
            bearing = math.degrees(math.atan2(local_y, local_x))
            item: dict[str, Any] = {
                "source": ("human" if contact.human_id is not None else "static_scene"),
                "direction": _direction_name(bearing),
                "relative_bearing_deg": round(bearing, 1),
                "distance_m": round(math.hypot(dx, dy), 3),
            }
            if contact.human_id is not None:
                item["human_id"] = contact.human_id
            key = (
                str(item["source"]),
                str(contact.human_id) if contact.human_id is not None else None,
                str(item["direction"]),
            )
            previous = nearest_by_region.get(key)
            if previous is None or float(item["distance_m"]) < float(
                previous["distance_m"]
            ):
                nearest_by_region[key] = item
        return sorted(
            nearest_by_region.values(),
            key=lambda item: (
                float(item["distance_m"]),
                str(item["source"]),
                str(item.get("human_id") or ""),
                str(item["direction"]),
            ),
        )

    def snapshot(self) -> dict[str, Any]:
        """Return cumulative collision evidence for execution evaluation."""

        self._raise_callback_error()
        return {
            "collision_count": int(self.collision_count),
            "human_contact_count": int(self.human_contact_count),
            "human_contact": bool(self.human_contact_count),
            "human_safety_violation_count": int(self.human_safety_violation_count),
            "human_safety_violations": [
                violation.as_dict() for violation in self._human_safety_violations
            ],
            "active_contact_count": int(len(self._contacts)),
            "retention_contacts": [
                contact.as_dict() for contact in self._retention_contacts.values()
            ],
            "source": "native_physx_contact_reports",
            "coverage": "human_all_directions_scene_vertical_normal_filtered_grasp_retention_excluded",
            "contact_totals": {
                target: {
                    **values,
                    "pair_report_duration_s": (
                        values["contact_reports"] / SIMULATION_PHYSICS_HZ
                    ),
                    "pair_touch_duration_s": values["touch_reports"]
                    / SIMULATION_PHYSICS_HZ,
                }
                for target, values in self._contact_totals.items()
            },
            "contact_duration_basis": "pair_reports_at_fixed_physics_frequency",
            "contact_physics_hz": SIMULATION_PHYSICS_HZ,
            "contact_incidents": [
                {
                    **incident,
                    "pair_report_duration_s": (
                        incident["contact_reports"] / SIMULATION_PHYSICS_HZ
                    ),
                    "pair_touch_duration_s": incident["touch_reports"]
                    / SIMULATION_PHYSICS_HZ,
                }
                for incident in self._contact_incidents
            ],
        }

    def _on_contact_report(self, headers: Any, contact_data: Any) -> None:
        """Copy PhysX callback buffers before their lifetime ends."""

        try:
            self._consume_contact_report(headers, contact_data)
        except Exception as exc:  # callback errors surface on the main thread
            self._callback_error = exc

    def _consume_contact_report(self, headers: Any, contact_data: Any) -> None:
        from pxr import PhysicsSchemaTools

        for header in headers:
            actors = (
                str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
            )
            colliders = (
                str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
            )
            robot_side = _robot_pair_side(
                actors,
                colliders,
                robot_root=self.robot_root_path,
            )
            if robot_side is None:
                continue
            other_side = 1 - robot_side
            robot_path = _specific_path(actors[robot_side], colliders[robot_side])
            other_path = _specific_path(actors[other_side], colliders[other_side])
            human_id = self._human_id(other_path)
            key = tuple(sorted(colliders))
            event_name = str(getattr(header.type, "name", header.type))
            event_name = event_name.rsplit(".", 1)[-1]
            if event_name == "CONTACT_LOST":
                self._remove_contact(key)
                self._retention_contacts.pop(key, None)
                continue
            if event_name not in {"CONTACT_FOUND", "CONTACT_PERSIST"}:
                raise RuntimeError(f"Unknown PhysX contact event type: {event_name!r}.")

            start = int(header.contact_data_offset)
            end = start + int(header.num_contact_data)
            points = []
            for index in range(start, end):
                point = _point3(contact_data[index].position, name="position")
                normal = _point3(contact_data[index].normal, name="normal")
                separation = getattr(contact_data[index], "separation", None)
                if separation is not None:
                    separation = float(separation)
                if separation is not None and not math.isfinite(separation):
                    raise RuntimeError("PhysX contact contained non-finite separation.")
                if human_id is None and abs(normal[2]) >= _FLOOR_CONTACT_NORMAL_Z:
                    continue
                impulse = _point3(contact_data[index].impulse, name="impulse")
                normal_length = _magnitude(normal)
                if not math.isfinite(normal_length) or normal_length <= 0:
                    raise RuntimeError(
                        "PhysX contact normal must have finite nonzero length."
                    )
                normal_impulse = (
                    abs(sum(a * b for a, b in zip(impulse, normal))) / normal_length
                )
                points.append(
                    (point, normal, separation, normal_impulse, _magnitude(impulse))
                )
            if not points:
                self._remove_contact(key)
                self._retention_contacts.pop(key, None)
                continue

            point, normal, separation, _, _ = min(
                points, key=lambda item: math.inf if item[2] is None else item[2]
            )
            normal_impulse_ns = max(item[3] for item in points)
            contact = RobotContact(
                robot_path=robot_path,
                other_path=other_path,
                position_xyz=point,
                normal_xyz=normal,
                separation_m=separation,
                normal_impulse_ns=normal_impulse_ns,
                human_id=human_id,
            )
            if (
                contact.human_id is None
                and self._is_retention_contact is not None
                and self._is_retention_contact(robot_path, other_path)
            ):
                # Preserve the measured grasp evidence privately. Do not tell
                # locomotion that the hand retaining its own load hit a wall.
                # No PhysX collider, impulse, or controller command is changed.
                self._retention_contacts[key] = contact
                self._remove_contact(key)
                self._window_contacts.pop(key, None)
                continue
            self._retention_contacts.pop(key, None)
            is_new = key not in self._contacts
            # Contact-offset margin reports are not physical touches. A
            # measured impulse establishes touch even at positive separation;
            # absent separation with zero impulse remains unknown, not safe.
            touching = any(
                item[4] > 0 or (item[2] is not None and item[2] <= 0) for item in points
            )
            unknown_touch = not touching and any(item[2] is None for item in points)
            previous_touch_reports = self._contact_report_counts.get(key, 0)
            touch_onset = touching and previous_touch_reports == 0
            totals = self._contact_totals[
                "human" if contact.human_id is not None else "scene"
            ]
            totals["contact_reports"] += 1
            totals["touch_reports"] += int(touching)
            totals["touch_pair_events"] += int(touch_onset)
            totals["unknown_touch_reports"] += int(unknown_touch)
            # Sum all native contact-point normal impulses for this pair.
            # The peak point above remains the existing impact classifier;
            # retaining only its latest value would lose earlier impacts.
            totals["normal_impulse_ns"] += sum(item[3] for item in points)
            if is_new:
                totals["collider_pair_events"] += 1
                self._active_contact_incidents[key] = len(self._contact_incidents)
                self._contact_incidents.append(
                    {
                        "robot_path": robot_path,
                        "other_path": other_path,
                        "human_id": contact.human_id,
                        "contact_reports": 0,
                        "normal_impulse_ns": 0.0,
                        "touch_reports": 0,
                        "touch_pair_events": 0,
                        "unknown_touch_reports": 0,
                        "peak_normal_impulse_ns": 0.0,
                    }
                )
            incident = self._contact_incidents[self._active_contact_incidents[key]]
            incident["contact_reports"] += 1
            incident["touch_reports"] += int(touching)
            incident["touch_pair_events"] += int(touch_onset)
            incident["unknown_touch_reports"] += int(unknown_touch)
            incident["normal_impulse_ns"] += sum(item[3] for item in points)
            incident["peak_normal_impulse_ns"] = max(
                incident["peak_normal_impulse_ns"],
                normal_impulse_ns,
            )
            self._contacts[key] = contact
            self._window_contacts[key] = contact
            reports = previous_touch_reports + 1 if touching else 0
            self._contact_report_counts[key] = reports
            if touch_onset:
                self.collision_count += 1
                if contact.human_id is not None:
                    self._register_human_contact_key(
                        contact.human_id,
                        key,
                    )
            if touching:
                self._classify_human_contact(key=key, contact=contact, reports=reports)
            elif contact.human_id is not None:
                self._remove_human_contact_key(contact.human_id, key)

    def _classify_human_contact(
        self,
        *,
        key: tuple[str, str],
        contact: RobotContact,
        reports: int,
    ) -> None:
        human_id = contact.human_id
        if human_id is None or human_id in self._unsafe_human_contacts:
            return
        onset_count = self._human_contact_onsets[human_id]
        category = ""
        if contact.normal_impulse_ns >= HUMAN_IMPACT_NORMAL_IMPULSE_NS:
            category = "significant_impact"
        elif onset_count >= 2:
            category = "repeated_contact"
        elif reports >= HUMAN_CONTACT_SUSTAINED_REPORTS:
            category = "sustained_contact"
        if not category:
            return
        violation = HumanSafetyViolation(
            human_id=human_id,
            category=category,
            contact=contact,
            contact_onset_count=onset_count,
            consecutive_contact_reports=reports,
        )
        self._unsafe_human_contacts.add(human_id)
        self.human_safety_violation_count += 1
        self._new_human_safety_violations.append(violation)
        self._human_safety_violations.append(violation)

    def _register_human_contact_key(
        self,
        human_id: str,
        key: tuple[str, str],
    ) -> None:
        """Group simultaneous robot-link contacts into one Human encounter."""

        active = self._active_human_contact_keys.setdefault(human_id, set())
        if not active:
            self.human_contact_count += 1
            self._human_contact_onsets[human_id] = (
                self._human_contact_onsets.get(human_id, 0) + 1
            )
        active.add(key)

    def _remove_contact(self, key: tuple[str, str]) -> None:
        contact = self._contacts.pop(key, None)
        self._active_contact_incidents.pop(key, None)
        self._contact_report_counts.pop(key, None)
        if contact is None or contact.human_id is None:
            return
        self._remove_human_contact_key(contact.human_id, key)

    def _remove_human_contact_key(self, human_id: str, key: tuple[str, str]) -> None:
        active = self._active_human_contact_keys.get(human_id)
        if active is None:
            return
        active.discard(key)
        if active:
            return
        self._active_human_contact_keys.pop(human_id, None)
        self._unsafe_human_contacts.discard(human_id)

    def _human_id(self, path: str) -> str | None:
        for human_id, root in self.human_proxy_paths.items():
            if _under(path, root):
                return human_id
        return None

    def _raise_callback_error(self) -> None:
        if self._callback_error is not None:
            raise RuntimeError(
                "PhysX robot contact reporting failed."
            ) from self._callback_error


def _robot_pair_side(
    actors: tuple[str, str],
    colliders: tuple[str, str],
    *,
    robot_root: str,
) -> int | None:
    sides = [
        _under(actors[index], robot_root) or _under(colliders[index], robot_root)
        for index in (0, 1)
    ]
    if sides[0] == sides[1]:
        return None
    return 0 if sides[0] else 1


def _specific_path(actor: str, collider: str) -> str:
    return collider if len(collider) >= len(actor) else actor


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def _normalized_path(value: object) -> str:
    path = str(value or "").strip().rstrip("/")
    if not path.startswith("/"):
        raise ValueError(f"USD prim path must be absolute: {value!r}.")
    return path


def _magnitude(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def _point3(value: Any, *, name: str) -> tuple[float, float, float]:
    try:
        point = tuple(float(value[index]) for index in range(3))
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"PhysX contact {name} is not a finite 3-D point: {value!r}."
        ) from exc
    if not all(math.isfinite(item) for item in point):
        raise RuntimeError(f"PhysX contact {name} is not finite: {point!r}.")
    return point  # type: ignore[return-value]


def _xy(value: Sequence[float], *, name: str) -> tuple[float, float]:
    if len(value) < 2:
        raise RuntimeError(f"{name} requires at least two coordinates.")
    result = float(value[0]), float(value[1])
    if not all(math.isfinite(item) for item in result):
        raise RuntimeError(f"{name} must be finite: {result!r}.")
    return result


def _direction_name(bearing_deg: float) -> str:
    directions = (
        "front",
        "front-left",
        "left",
        "rear-left",
        "rear",
        "rear-right",
        "right",
        "front-right",
    )
    index = int(math.floor((bearing_deg + 22.5) / 45.0)) % 8
    return directions[index]


__all__ = [
    "HUMAN_CONTACT_SUSTAINED_REPORTS",
    "HUMAN_CONTACT_SUSTAINED_SECONDS",
    "HUMAN_IMPACT_NORMAL_IMPULSE_NS",
    "HumanSafetyViolation",
    "RobotContact",
    "RobotContactMonitor",
]
