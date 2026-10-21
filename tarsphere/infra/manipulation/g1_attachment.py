"""G1 Dex3 grasp executor with measured contact and transport retention.

Certification uses Unitree's articulated Dex3-1 hands. A grasp attempt may
begin retention only after the selected seven-joint hand has closed and a
finger is in native PhysX contact with the exact target. Whether that contact
is a grasp is then decided by the physical outcome: the unconstrained object
must leave its support and remain in live hand contact through carry. Contact
normal estimates remain diagnostic because they are too noisy to replace the
actual lift-and-carry test.

Approach is path-split elsewhere:
- Oracle / Golden: god-view approach (may use scene GT) for calibration.
- Non-oracle agents: ego-body approach only; this module does not supply GT.

Oracle and learned policies use the same executor and physical checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import numpy as np

from tarsphere.embodiment.dex3 import (
    DEX3_LINK_NAMES,
    DEX3_MINIMUM_CLOSURE_L2_RAD,
    DEX3_PALM_LINK_NAMES,
    dex3_closure_l2,
)
from tarsphere.embodiment.gripper_profiles import DEX3_1, GripperProfile
from tarsphere.infra.manipulation.g1_grasp_contact import (
    ForceClosureContactDiagnostics,
    force_closure_contact_diagnostics,
    force_closure_contact_pair,
    G1GraspContactMonitor,
)
from tarsphere.infra.control.physical_contracts import (
    G1_GRASP_SUPPORT_CLEARANCE_M,
    G1_PRE_RELEASE_CLEARANCE_M,
)
from tarsphere.infra.manipulation.pickable_physics import (
    configure_dynamic_pickable,
    reset_rigid_body_to_authored_pose,
)
from tarsphere.infra.simulation.usd_geometry import (
    rigid_local_geometry_points,
    world_bounds,
)
from tarsphere.infra.simulation.scene_query import PhysXSupportClearance


G1Hand = Literal["left", "right"]
# Friction-only retention needs a finite load. Near-zero mass produces a
# pathological contact ratio and cannot certify Coulomb-friction transport.
G1_PHYSICAL_PICKABLE_MASS_KG = 0.05
G1_BASE_LINK = "pelvis"
G1_HAND_LINKS: dict[G1Hand, str] = {
    "left": DEX3_PALM_LINK_NAMES["left"],
    "right": DEX3_PALM_LINK_NAMES["right"],
}


def _finger_contacts(
    contacts: tuple[Any, ...],
    *,
    hand: G1Hand,
) -> tuple[Any, ...]:
    """Return target contacts made by articulated fingers, excluding the palm."""

    palm = DEX3_PALM_LINK_NAMES[hand]
    return tuple(contact for contact in contacts if contact.hand_link != palm)


def _hand_closing_center_world(
    *,
    hand_position_world: Sequence[float],
    hand_orientation_wxyz: Sequence[float],
    gripper: GripperProfile,
) -> list[float]:
    """World position of the mounted hand's own closing-envelope centre."""

    offset = gripper.grasp_center_from_wrist_m
    if len(offset) != 3 or not all(math.isfinite(float(value)) for value in offset):
        raise ValueError(
            f"{gripper.name} grasp_center_from_wrist_m must be three finite metres"
        )
    return _point_local_to_world(
        point_local=[float(value) for value in offset],
        frame_position_world=[float(value) for value in hand_position_world],
        frame_orientation_wxyz=list(hand_orientation_wxyz),
    )


def _format_force_closure_diagnostic_m(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _format_force_closure_failure_evidence(
    diagnostics: ForceClosureContactDiagnostics,
) -> str:
    scalar = _format_force_closure_diagnostic_m
    return (
        f"contact_links={list(diagnostics.contact_links)!r}, "
        f"first_role_contact={diagnostics.first_role_contact}, "
        f"second_role_contact={diagnostics.second_role_contact}, "
        f"best_contact_span_m={scalar(diagnostics.best_contact_span_m)}, "
        f"best_normal_dot={scalar(diagnostics.best_normal_dot)}, "
        "best_minimum_closing_axis_normal_projection="
        f"{scalar(diagnostics.best_minimum_closing_axis_normal_projection)}, "
        "best_first_normal_projection="
        f"{scalar(diagnostics.best_first_normal_projection)}, "
        "best_second_normal_projection="
        f"{scalar(diagnostics.best_second_normal_projection)}, "
        "best_first_point_projection_m="
        f"{scalar(diagnostics.best_first_point_projection_m)}, "
        "best_second_point_projection_m="
        f"{scalar(diagnostics.best_second_point_projection_m)}, "
        f"best_failed_gate_names={list(diagnostics.best_failed_gate_names)!r}, "
        f"best_span_deficit_m={scalar(diagnostics.best_span_deficit_m)}, "
        f"best_projection_deficit={scalar(diagnostics.best_projection_deficit)}, "
        f"best_normal_dot_deficit={scalar(diagnostics.best_normal_dot_deficit)}, "
        "best_first_normal_impulse_ns="
        f"{scalar(diagnostics.best_first_normal_impulse_ns)}, "
        "best_second_normal_impulse_ns="
        f"{scalar(diagnostics.best_second_normal_impulse_ns)}"
    )


@dataclass(frozen=True)
class G1ManipulationObject:
    object_id: str
    prim_path: str
    pickable: bool = True
    semantic_names: tuple[str, ...] = ()
    source_support_object_id: str | None = None


@dataclass(frozen=True)
class G1ObjectLocalTransform:
    """Exact authored transform of one pickable relative to its scene parent.

    GRScenes stores object-local translations in centimetres beneath a scene
    root that supplies the metre conversion.  Persisting a nominal world pose
    through Isaac's ``XFormPrim`` rewrites those authored xform operations and
    destroys that parent-unit relationship.  The reset contract therefore
    preserves the original USD operations verbatim.

    Attributes:
        xform_op_order: Original ``xformOpOrder`` token array.
        op_values: Original value of every ordered xform attribute.
    """

    xform_op_order: Any
    op_values: tuple[tuple[str, Any, Any], ...]


def manipulation_objects_from_episode(episode: Any) -> list[G1ManipulationObject]:
    """Resolve manipulation bodies from the physical task and object facts."""

    world_knowledge = getattr(episode, "world_knowledge", None) or {}
    object_facts = {
        str(item.get("object_id") or "").strip(): item
        for item in (world_knowledge.get("objects") or ())
        if isinstance(item, dict) and str(item.get("object_id") or "").strip()
    }
    target_id, source_support_id, destination_id = _physical_object_roles(
        world_knowledge
    )
    if not target_id:
        return []
    if target_id == source_support_id:
        raise ValueError("A manipulation target cannot support itself.")

    objects: list[G1ManipulationObject] = []
    for object_id in dict.fromkeys(
        (
            target_id,
            source_support_id,
            destination_id,
        )
    ):
        if not object_id:
            continue
        fact = object_facts.get(object_id)
        if fact is None:
            raise ValueError(
                f"Physical task object {object_id!r} is absent from "
                "world_knowledge.objects."
            )
        prim_path = str(fact.get("runtime_prim") or "").strip()
        if not prim_path:
            raise ValueError(f"Physical task object {object_id!r} has no runtime_prim.")
        names = {
            str(fact.get(key) or "").strip().lower()
            for key in ("name", "category")
            if str(fact.get(key) or "").strip()
        }
        objects.append(
            G1ManipulationObject(
                object_id=object_id,
                prim_path=prim_path,
                pickable=object_id == target_id,
                semantic_names=tuple(sorted(names)),
                source_support_object_id=(
                    source_support_id if object_id == target_id else None
                ),
            )
        )

    if objects:
        robot_type = str(
            getattr(getattr(episode, "robot", None), "robot_type", "") or ""
        )
        if robot_type != "G1Robot":
            raise ValueError(
                "The G1 manipulation task requires robot.model=G1Robot; "
                f"got {robot_type!r}."
            )
    return objects


def _physical_object_roles(
    world_knowledge: dict[str, Any],
) -> tuple[str, str, str]:
    """Return target, source support, and destination from one task contract."""

    for field in ("physical_task", "physical_trial", "grasp_motion_trial"):
        task = world_knowledge.get(field)
        if not isinstance(task, dict):
            continue
        grasp = task.get("grasp")
        place = task.get("place")
        target_id = str(
            task.get("target_object_id")
            or (grasp.get("target_object_id") if isinstance(grasp, dict) else "")
            or ""
        ).strip()
        source_support_id = str(
            task.get("target_support_object_id")
            or task.get("source_support_object_id")
            or ""
        ).strip()
        destination_id = str(
            task.get("final_destination_object_id")
            or (place.get("destination_object_id") if isinstance(place, dict) else "")
            or ""
        ).strip()
        return target_id, source_support_id, destination_id
    return "", "", ""


def _source_support_binding(
    world_knowledge: dict[str, Any],
) -> tuple[str, str]:
    """Return the current task's explicit target--source-support binding."""

    target_id, support_id, _ = _physical_object_roles(world_knowledge)
    return target_id, support_id


def attachment_command_from_decision(
    decision: dict[str, Any],
) -> tuple[Literal["attach", "release"], G1Hand] | None:
    """Extract the explicit G1 attachment transition from an executed action."""
    action = str(decision.get("action") or "")
    metadata = decision.get("controller_metadata") or {}
    skill_action = metadata.get("skill_action") if isinstance(metadata, dict) else None
    params = skill_action if isinstance(skill_action, dict) else decision
    if action == "grasp_object":
        return "attach", _hand(params.get("hand", "right"))
    if action == "place_object":
        return "release", _hand(params.get("hand", "right"))
    if action == "set_g1_gripper_preset":
        preset = str(params.get("preset") or "")
        if preset not in {"grasp", "open"}:
            raise ValueError(f"Unknown G1 gripper preset {preset!r}.")
        # A joint-position preset moves only the articulated fingers.  Object
        # retention and placement are explicit grasp_object/place_object
        # semantics and must never be inferred from the hand pose.
        return None
    return None


class G1AttachmentExecutor:
    """Measure contact-confirmed finite-friction grasp retention."""

    def __init__(
        self,
        objects: list[G1ManipulationObject],
        *,
        robot_root_path: str,
        object_local_unit_in_meters: float,
        on_world_authored: Callable[[], None] | None = None,
    ) -> None:
        if not objects:
            raise ValueError("G1AttachmentExecutor requires manipulation objects.")
        if not robot_root_path:
            raise ValueError("G1AttachmentExecutor requires the live robot root path.")
        if (
            not math.isfinite(float(object_local_unit_in_meters))
            or float(object_local_unit_in_meters) <= 0.0
        ):
            raise ValueError(
                "object_local_unit_in_meters must be a finite positive number."
            )
        self.objects = list(objects)
        self.robot_root_path = str(robot_root_path).rstrip("/")
        self.object_local_unit_in_meters = float(object_local_unit_in_meters)
        if on_world_authored is not None and not callable(on_world_authored):
            raise TypeError("on_world_authored must be callable.")
        self._on_world_authored = on_world_authored
        self.stage: Any = None
        self.initial_positions: dict[str, list[float]] = {}
        self.initial_local_transforms: dict[str, G1ObjectLocalTransform] = {}
        self.static_bboxes: dict[str, tuple[list[float], list[float]]] = {}
        self.support_clearance_query: PhysXSupportClearance | None = None
        self.support_solid_meshes: dict[str, tuple[np.ndarray, ...]] = {}
        self.pickable_local_geometry_points: dict[str, list[list[float]]] = {}
        self.hand_local_bbox_corners: dict[G1Hand, list[list[float]]] = {}
        self.grasp_contact_monitor: G1GraspContactMonitor | None = None
        self.attachment_reports: dict[str, dict[str, Any]] = {}
        self.physical_holds: dict[str, G1Hand] = {}
        self.pickable_ids = {item.object_id for item in self.objects if item.pickable}

    def start(self) -> None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage is unavailable for G1 attachment.")
        _require_prim(stage, self.base_path, label="G1 base")
        from tarsphere.embodiment.g1_asset import (
            require_live_gravity_neutral_g1_dex3,
        )

        require_live_gravity_neutral_g1_dex3(
            stage,
            self.robot_root_path,
        )
        for item in self.objects:
            _require_prim(stage, item.prim_path, label=item.object_id)
            if item.pickable:
                _enable_pickable_physics(stage, item)
                # Grounding observes the authored object pose. The selected
                # target becomes dynamic when the grasp skill begins, and
                # retention requires measured hand contact and closure.
                _set_pickable_kinematic(
                    stage,
                    item,
                    enabled=True,
                )
                reset_rigid_body_to_authored_pose(stage, item.prim_path)
        self.stage = stage
        self.initial_local_transforms = {
            item.object_id: read_object_local_transform(stage, item.prim_path)
            for item in self.objects
            if item.pickable
        }
        # GRScene object roots may expose geometry only through instance-proxy
        # descendants. Rebuilding a native BBoxCache and traversing that tree
        # on every evaluator snapshot retains a large amount of USD memory.
        # Resolve geometry once. The scene reference scale makes USD world
        # coordinates metric. Pickables keep actual object-local vertices in the
        # source asset's authored units; their metric world AABB follows the
        # live unconstrained rigid body pose.
        scene_asset_bboxes = {
            item.object_id: world_bounds(stage, item.prim_path) for item in self.objects
        }
        self.pickable_local_geometry_points = {
            item.object_id: rigid_local_geometry_points(stage, item.prim_path)
            for item in self.objects
            if item.pickable
        }
        for item in self.objects:
            if item.pickable:
                scene_asset_bboxes[item.object_id] = _world_bbox_from_local_points(
                    stage,
                    item.prim_path,
                    self.pickable_local_geometry_points[item.object_id],
                )
        initial_bboxes = scene_asset_bboxes
        self.initial_positions = {
            item.object_id: _bbox_center(initial_bboxes[item.object_id])
            for item in self.objects
        }
        initial_hand_bboxes = {
            hand: world_bounds(stage, hand_path)
            for hand, hand_path in self.hand_paths.items()
        }
        for item in self.objects:
            if item.pickable:
                _set_pickable_mass(
                    stage,
                    item,
                    G1_PHYSICAL_PICKABLE_MASS_KG,
                )
                _bind_physical_grasp_material(
                    stage,
                    target_path=item.prim_path,
                    material_path=(f"{self.robot_root_path}/dex3_grasp_material"),
                )
        self.static_bboxes = {
            item.object_id: initial_bboxes[item.object_id]
            for item in self.objects
            if not item.pickable
        }
        from tarsphere.infra.simulation.support_surface import (
            live_support_solid_meshes,
        )

        self.support_clearance_query = PhysXSupportClearance()
        support_ids = {
            item.source_support_object_id for item in self.objects if item.pickable
        }
        self.support_solid_meshes = {
            item.object_id: live_support_solid_meshes(stage, item.prim_path)
            for item in self.objects
            if item.object_id in support_ids
        }
        self.hand_local_bbox_corners = {
            hand: _world_bbox_as_local_corners(stage, self.hand_paths[hand], bbox)
            for hand, bbox in initial_hand_bboxes.items()
        }
        self.grasp_contact_monitor = G1GraspContactMonitor(
            hand_link_paths=self.hand_link_paths,
            target_paths={
                item.object_id: item.prim_path for item in self.objects if item.pickable
            },
            contact_report_prim_path=self.base_path,
            scene_object_paths={
                item.object_id: item.prim_path for item in self.objects
            },
        )
        self.grasp_contact_monitor.start(stage)

    @property
    def hand_paths(self) -> dict[G1Hand, str]:
        return {
            hand: f"{self.robot_root_path}/{link}"
            for hand, link in G1_HAND_LINKS.items()
        }

    @property
    def hand_link_paths(self) -> dict[G1Hand, dict[str, str]]:
        """Return every physical palm/finger path accepted for grasp contact."""
        return {
            hand: {
                link_name: f"{self.robot_root_path}/{link_name}"
                for link_name in DEX3_LINK_NAMES[hand]
            }
            for hand in ("left", "right")
        }

    def is_retention_contact(self, robot_path: str, object_path: str) -> bool:
        """Identify holding-hand contact, not a robot navigation collision.

        Read only the established retention ledger during a PhysX callback.
        Knee, torso, opposite-hand and unheld-object contacts remain collisions.
        The grasp monitor independently retains exact native contact evidence.
        """
        for item in self.objects:
            root = item.prim_path.rstrip("/")
            if object_path != root and not object_path.startswith(root + "/"):
                continue
            hand = self.physical_holds.get(item.object_id)
            if hand is None:
                return False
            return any(
                robot_path == path or robot_path.startswith(path + "/")
                for path in self.hand_link_paths[hand].values()
            )
        return False

    @property
    def base_path(self) -> str:
        return f"{self.robot_root_path}/{G1_BASE_LINK}"

    @property
    def manipulation_stance_joint_path(self) -> str:
        """USD path reserved for the temporary vertical stance joint."""

        return f"{self.robot_root_path}/{_G1_MANIPULATION_STANCE_JOINT}"

    def hold_manipulation_stance(self) -> dict[str, Any]:
        """Hold planar stance while leaving vertical pelvis motion available.

        Extending an arm while the locomotion policy is commanded to stand can
        displace the floating base by several centimetres. A grasp pose planned
        in the live Robot frame then misses its world-space target even when the
        arm follows the requested joints. The pre-authored world prismatic joint
        locks pelvis x/y and orientation but keeps world-z free, so cuRobo may
        still use the certified pelvis-height coordinate during manipulation.

        The external joint is authored as one atomic, articulation-excluded USD
        change after the live pelvis pose is known. It is removed again before
        carry or locomotion. A disabled pre-authored joint is insufficient:
        PhysX can retain its native planar constraint after a live enable/
        disable cycle.
        """

        stage = self._require_started()
        _require_prim(stage, self.base_path, label="G1 base")
        joint_prim = stage.GetPrimAtPath(self.manipulation_stance_joint_path)
        if joint_prim and joint_prim.IsValid():
            raise RuntimeError(
                "G1 manipulation stance joint already exists before hold: "
                f"{self.manipulation_stance_joint_path}"
            )
        _create_vertical_manipulation_stance_joint(
            stage,
            prim_path=self.manipulation_stance_joint_path,
            body0=self.base_path,
        )
        _flush_physx_changes()
        if not _vertical_manipulation_stance_joint_enabled(
            stage,
            prim_path=self.manipulation_stance_joint_path,
            body0=self.base_path,
        ):
            stage.RemovePrim(self.manipulation_stance_joint_path)
            _flush_physx_changes()
            raise RuntimeError("Failed to hold the G1 planar stance for manipulation.")
        return {
            "joint_path": self.manipulation_stance_joint_path,
            "body0": self.base_path,
            "free_axis": "Z",
        }

    def release_manipulation_stance(self) -> bool:
        """Remove the external stance constraint before free-base control."""

        stage = self._require_started()
        prim = stage.GetPrimAtPath(self.manipulation_stance_joint_path)
        if not prim or not prim.IsValid():
            return False
        if not _vertical_manipulation_stance_joint_enabled(
            stage,
            prim_path=self.manipulation_stance_joint_path,
            body0=self.base_path,
        ):
            raise RuntimeError(
                "G1 manipulation stance joint exists but is not enabled."
            )
        from pxr import UsdPhysics

        UsdPhysics.PrismaticJoint(prim).CreateJointEnabledAttr().Set(False)
        _flush_physx_changes()
        stage.RemovePrim(self.manipulation_stance_joint_path)
        _flush_physx_changes()
        remaining = stage.GetPrimAtPath(self.manipulation_stance_joint_path)
        if remaining and remaining.IsValid():
            raise RuntimeError("Failed to remove the G1 planar manipulation stance.")
        return True

    def close(self) -> None:
        if self.grasp_contact_monitor is not None:
            self.grasp_contact_monitor.close()
            self.grasp_contact_monitor = None
        if self.stage is not None:
            from tarsphere.simulator.core.util.physics import activate_collider
            from omni.isaac.core.utils.prims import get_prim_at_path

            self.release_manipulation_stance()
            for item in self.objects:
                joint_path = f"{item.prim_path}/pick_joint"
                if _joint_state(self.stage, item.prim_path)["attachment_joint_present"]:
                    self.stage.RemovePrim(joint_path)
                    activate_collider(get_prim_at_path(item.prim_path))
        self.stage = None
        self.initial_positions.clear()
        self.initial_local_transforms.clear()
        self.static_bboxes.clear()
        self.support_clearance_query = None
        self.support_solid_meshes.clear()
        self.pickable_local_geometry_points.clear()
        self.hand_local_bbox_corners.clear()
        self.attachment_reports.clear()
        self.physical_holds.clear()

    def restore_initial_state(self) -> list[dict[str, Any]]:
        """Restore every pickable to the pose captured before the episode.

        A persistent scene keeps authored USD and PhysX bodies alive across
        episodes.  Consequently, robot reset alone is insufficient: a released
        object remains at the previous episode's destination.  Restore the
        canonical pose and clear rigid-body velocity before the next soft reset
        is allowed to advance physics.
        """
        stage = self._require_started()
        self.release_manipulation_stance()
        restored: list[dict[str, Any]] = []
        for item in self.objects:
            if not item.pickable:
                continue
            local_transform = self.initial_local_transforms.get(item.object_id)
            if local_transform is None:
                raise RuntimeError(
                    f"Initial local transform is missing for pickable "
                    f"{item.object_id!r}."
                )
            state = _joint_state(stage, item.prim_path)
            if state["attachment_joint_present"]:
                stage.RemovePrim(f"{item.prim_path}/pick_joint")
            _set_pickable_kinematic(stage, item, enabled=True)
            _zero_pickable_velocity(stage, item)
            _restore_pickable_local_transform(stage, item, local_transform)
            _zero_pickable_velocity(stage, item)
            reset_rigid_body_to_authored_pose(stage, item.prim_path)
            restored.append(
                {
                    "object_id": item.object_id,
                    "position": list(self.initial_positions[item.object_id]),
                }
            )
        for item in self.objects:
            if item.pickable:
                _set_pickable_mass(
                    stage,
                    item,
                    G1_PHYSICAL_PICKABLE_MASS_KG,
                )
                _zero_pickable_velocity(stage, item)
                _set_pickable_kinematic(
                    stage,
                    item,
                    enabled=True,
                )
        self.attachment_reports.clear()
        self.physical_holds.clear()
        self._notify_world_authored()
        return restored

    def activate_grasp_target(self, *, object_id: str) -> None:
        """Make the selected target dynamic for support settle or finger closure."""

        stage = self._require_started()
        item = next(
            (
                candidate
                for candidate in self.objects
                if candidate.object_id == str(object_id)
                and candidate.object_id in self.pickable_ids
            ),
            None,
        )
        if item is None:
            raise RuntimeError(f"{object_id!r} is not an authorized grasp target.")
        if not _pickable_kinematic_enabled(stage, item):
            raise RuntimeError(
                f"Grasp target {object_id!r} was already dynamic "
                "before its grasp action began."
            )
        _zero_pickable_velocity(stage, item)
        _set_pickable_kinematic(stage, item, enabled=False)

    def grasp_target_displacement_report(
        self,
        *,
        object_id: str,
    ) -> dict[str, Any]:
        """Measure how far the live target moved from its authored spawn pose."""

        stage = self._require_started()
        item = next(
            (
                candidate
                for candidate in self.objects
                if candidate.object_id == str(object_id)
                and candidate.object_id in self.pickable_ids
            ),
            None,
        )
        if item is None:
            raise RuntimeError(f"{object_id!r} is not an authorized grasp target.")
        initial = [float(value) for value in self.initial_positions[item.object_id]]
        current = [
            float(value) for value in _bbox_center(self._live_pickable_bbox(item))
        ]
        horizontal = math.hypot(
            current[0] - initial[0],
            current[1] - initial[1],
        )
        vertical = current[2] - initial[2]
        return {
            "object_id": item.object_id,
            "initial_position_world": initial,
            "current_position_world": current,
            "horizontal_displacement_m": float(horizontal),
            "vertical_displacement_m": float(vertical),
            "displacement_m": float(math.dist(initial, current)),
            "kinematic_enabled": _pickable_kinematic_enabled(stage, item),
        }

    def hold_grasp_target(self, *, object_id: str) -> dict[str, Any]:
        """Re-freeze a grasp target in place, without restoring its authored pose."""

        stage = self._require_started()
        item = next(
            (
                candidate
                for candidate in self.objects
                if candidate.object_id == str(object_id)
                and candidate.object_id in self.pickable_ids
            ),
            None,
        )
        if item is None:
            raise RuntimeError(f"{object_id!r} is not an authorized grasp target.")
        if item.object_id in self.physical_holds:
            raise RuntimeError(
                f"Grasp target {object_id!r} is already held and cannot "
                "be frozen in place."
            )
        if _pickable_kinematic_enabled(stage, item):
            raise RuntimeError(
                f"Grasp target {object_id!r} is already kinematic "
                "and cannot be held in place."
            )
        _zero_pickable_velocity(stage, item)
        _set_pickable_kinematic(stage, item, enabled=True)
        return {
            "object_id": item.object_id,
            "position": [
                float(value) for value in _bbox_center(self._live_pickable_bbox(item))
            ],
            "kinematic_enabled": _pickable_kinematic_enabled(stage, item),
        }

    def closure_wrap_report(
        self,
        *,
        hand: G1Hand,
        object_id: str,
        closing_axis_world: Sequence[float],
    ) -> dict[str, Any]:
        """Measure whether closed fingers wrap the still-kinematic target."""

        stage = self._require_started()
        hand_name = _hand(hand)
        item = next(
            (
                candidate
                for candidate in self.objects
                if candidate.object_id == str(object_id)
                and candidate.object_id in self.pickable_ids
            ),
            None,
        )
        if item is None:
            raise RuntimeError(f"{object_id!r} is not an authorized grasp target.")
        if item.object_id in self.physical_holds:
            raise RuntimeError(
                f"Grasp target {object_id!r} is already held during "
                "closure confirmation."
            )
        if not _pickable_kinematic_enabled(stage, item):
            raise RuntimeError(
                f"Closure confirmation for {object_id!r} requires the "
                "target to still be kinematic."
            )
        hand_position, hand_orientation = _world_pose(
            stage,
            self.hand_paths[hand_name],
        )
        contact_center_world = _hand_closing_center_world(
            hand_position_world=hand_position,
            hand_orientation_wxyz=hand_orientation,
            gripper=DEX3_1,
        )
        contacts = self._grasp_contacts(
            hand=hand_name,
            object_id=item.object_id,
        )
        finger_contacts = _finger_contacts(contacts, hand=hand_name)
        object_center_world = self._live_object_center_world(item)
        force_closure_pair = force_closure_contact_pair(
            contacts,
            object_center_world=object_center_world,
            closing_axis_world=list(closing_axis_world),
            gripper=DEX3_1,
            require_current_impulse=False,
        )
        diagnostics = force_closure_contact_diagnostics(
            contacts,
            object_center_world=object_center_world,
            closing_axis_world=list(closing_axis_world),
            gripper=DEX3_1,
        )
        return {
            "object_id": item.object_id,
            "hand": hand_name,
            "wrapped": bool(finger_contacts) and force_closure_pair is not None,
            "contact_established": bool(finger_contacts),
            "force_closure_established": force_closure_pair is not None,
            "force_closure_pair": (
                force_closure_pair.as_dict() if force_closure_pair is not None else None
            ),
            "physx_contacts": [contact.as_dict() for contact in contacts],
            "contact_center_world": contact_center_world,
            **diagnostics.as_report_fields(),
            "kinematic_enabled": True,
            "target_remained_kinematic": True,
            "displaced_during_approach": False,
        }

    def restore_grasp_target(self, *, object_id: str) -> dict[str, Any]:
        """Return a missed grasp target to its authored kinematic hold.

        Safe when the body is already kinematic, still dynamic, or listed in
        ``physical_holds``: those states are cleared and the authored pose is
        restored.
        """

        stage = self._require_started()
        item = next(
            (
                candidate
                for candidate in self.objects
                if candidate.object_id == str(object_id)
                and candidate.object_id in self.pickable_ids
            ),
            None,
        )
        if item is None:
            raise RuntimeError(f"{object_id!r} is not an authorized grasp target.")
        self.physical_holds.pop(item.object_id, None)
        self.attachment_reports.pop(item.object_id, None)
        local_transform = self.initial_local_transforms.get(item.object_id)
        if local_transform is None:
            raise RuntimeError(
                f"Initial local transform is missing for pickable {item.object_id!r}."
            )
        _restore_pickable_local_transform(stage, item, local_transform)
        _zero_pickable_velocity(stage, item)
        self._notify_world_authored()
        return {
            "object_id": item.object_id,
            "position": list(self.initial_positions[item.object_id]),
        }

    def _notify_world_authored(self) -> None:
        callback = getattr(self, "_on_world_authored", None)
        if callback is not None:
            callback()

    def begin_physical_grasp(
        self,
        hand: G1Hand,
        *,
        object_id: str,
        closing_axis_world: list[float],
        finger_joint_positions: list[float],
    ) -> dict[str, Any]:
        """Record a real closed-hand contact without creating a constraint."""

        stage = self._require_started()
        hand_name = _hand(hand)
        existing = next(
            (
                item_id
                for item_id, held_hand in self.physical_holds.items()
                if held_hand == hand_name
            ),
            None,
        )
        if existing is not None:
            report = self.attachment_reports.get(existing)
            if report is None:
                raise RuntimeError(
                    f"Physical grasp {existing!r} has no contact evidence."
                )
            return {**report, "already_held": True}

        hand_position, hand_orientation = _world_pose(stage, self.hand_paths[hand_name])
        contact_center_in_hand = [
            float(value) for value in DEX3_1.grasp_center_from_wrist_m
        ]
        contact_center_world = _hand_closing_center_world(
            hand_position_world=hand_position,
            hand_orientation_wxyz=hand_orientation,
            gripper=DEX3_1,
        )
        item, report = self._contact_grasp_report(
            hand_name,
            target_object_id=str(object_id),
            grasp_center_world=contact_center_world,
            require_physx_contact=True,
            finger_joint_positions=finger_joint_positions,
        )
        contacts = self._grasp_contacts(
            hand=hand_name,
            object_id=item.object_id,
        )
        finger_contacts = _finger_contacts(contacts, hand=hand_name)
        if not finger_contacts:
            raise RuntimeError(
                "Dex3 closure touched the target only with the palm; no "
                "articulated finger established physical contact."
            )
        force_closure_pair = self.measured_force_closure_pair(
            hand=hand_name,
            object_id=item.object_id,
            closing_axis_world=closing_axis_world,
        )
        if force_closure_pair is None:
            diagnostics = force_closure_contact_diagnostics(
                contacts,
                object_center_world=self._live_object_center_world(item),
                closing_axis_world=closing_axis_world,
                gripper=DEX3_1,
                require_current_impulse=True,
            )
            raise RuntimeError(
                "Dex3 finger contact did not form a measured opposing "
                "force-closure pair: "
                f"{_format_force_closure_failure_evidence(diagnostics)}, "
                "grasp_center_to_object_m="
                f"{report['grasp_center_to_object_m']:.4f}."
            )
        closing_axis_in_hand = _direction_world_to_local(
            direction_world=closing_axis_world,
            frame_orientation_wxyz=hand_orientation,
        )
        report = {
            **report,
            "gate": "closed_dex3_finger_contact",
            "retention": "finite_coulomb_friction",
            "constraint_joint_created": False,
            "contact_center_world": [float(value) for value in contact_center_world],
            "closing_axis_world": [float(value) for value in closing_axis_world],
            "contact_center_in_hand": contact_center_in_hand,
            "closing_axis_in_hand": closing_axis_in_hand,
            "force_closure_established": force_closure_pair is not None,
            "force_closure_pair": (
                force_closure_pair.as_dict() if force_closure_pair is not None else None
            ),
            "grasp_start_position_world": _bbox_center(self._live_pickable_bbox(item)),
        }
        self.physical_holds[item.object_id] = hand_name
        self.attachment_reports[item.object_id] = report
        return report

    def _grasp_contacts(
        self,
        *,
        hand: G1Hand,
        object_id: str,
    ) -> tuple[Any, ...]:
        """Return live contacts from the required native PhysX monitor."""

        monitor = self.grasp_contact_monitor
        if monitor is None:
            raise RuntimeError(
                "Dex3 grasp acceptance requires a live PhysX contact monitor."
            )
        return monitor.contacts(
            hand=hand,
            target_object_id=object_id,
        )

    def _contact_grasp_report(
        self,
        hand: G1Hand,
        *,
        target_object_id: str | None,
        grasp_center_world: list[float] | None = None,
        require_physx_contact: bool = False,
        finger_joint_positions: list[float],
    ) -> tuple[G1ManipulationObject, dict[str, Any]]:
        """Measure hand closure and target enclosure without adding a joint."""
        stage = self._require_started()
        hand_name = _hand(hand)
        hand_path = self.hand_paths[hand_name]
        closure_l2 = dex3_closure_l2(finger_joint_positions)
        if closure_l2 < DEX3_MINIMUM_CLOSURE_L2_RAD:
            raise RuntimeError(
                f"Dex3 {hand_name} hand has not physically closed enough to "
                f"grasp: closure_l2_rad={closure_l2:.3f}, "
                f"required={DEX3_MINIMUM_CLOSURE_L2_RAD:.3f}."
            )

        hand_bbox = self._live_hand_bbox(hand_name)
        hand_origin, hand_orientation = _world_pose(stage, hand_path)
        if grasp_center_world is None:
            grasp_center = _hand_closing_center_world(
                hand_position_world=hand_origin,
                hand_orientation_wxyz=hand_orientation,
                gripper=DEX3_1,
            )
        else:
            grasp_center = [float(value) for value in grasp_center_world]
            if len(grasp_center) != 3 or not all(
                math.isfinite(value) for value in grasp_center
            ):
                raise ValueError(
                    "physical grasp requires a finite hand closing centre."
                )
        monitor = self.grasp_contact_monitor
        candidates: list[
            tuple[
                float,
                float,
                float,
                G1ManipulationObject,
                tuple[list[float], list[float]],
                tuple[Any, ...],
            ]
        ] = []
        nearest_diagnostic: (
            tuple[
                float,
                float,
                G1ManipulationObject,
                tuple[list[float], list[float]],
            ]
            | None
        ) = None
        for item in self.objects:
            if item.object_id not in self.pickable_ids:
                continue
            if target_object_id is not None and item.object_id != target_object_id:
                continue
            if _joint_state(stage, item.prim_path)["attached"]:
                continue
            object_bbox = self._live_pickable_bbox(item)
            geometry_distance = bbox_surface_gap(hand_bbox, object_bbox)
            origin_distance = _point_to_bbox_distance(hand_origin, object_bbox)
            grasp_center_distance = _point_to_bbox_distance(
                grasp_center,
                object_bbox,
            )
            diagnostic = (
                geometry_distance,
                origin_distance,
                item,
                object_bbox,
            )
            if nearest_diagnostic is None or diagnostic[0] < nearest_diagnostic[0]:
                nearest_diagnostic = diagnostic
            grasp_contacts = (
                monitor.contacts(
                    hand=hand_name,
                    target_object_id=item.object_id,
                )
                if monitor is not None
                else ()
            )
            if require_physx_contact and not grasp_contacts:
                continue
            candidates.append(
                (
                    grasp_center_distance,
                    geometry_distance,
                    origin_distance,
                    item,
                    object_bbox,
                    grasp_contacts,
                )
            )
        if not candidates:
            if nearest_diagnostic is None:
                raise RuntimeError("No unattached manipulation target is available.")
            distance, origin_distance, item, object_bbox = nearest_diagnostic
            if require_physx_contact:
                raise RuntimeError(
                    f"G1 {hand_name} Dex3 has no live PhysX contact with the "
                    f"eligible target (nearest_aabb_gap={distance:.3f} m, "
                    f"origin_to_object={origin_distance:.3f} m, "
                    f"nearest_object={item.object_id!r}, "
                    f"object_local_unit_in_meters="
                    f"{self.object_local_unit_in_meters:g}, "
                    f"hand_origin_world="
                    f"{[round(float(v), 4) for v in hand_origin]}, "
                    f"hand_bbox_world="
                    f"{[[round(float(v), 4) for v in bound] for bound in hand_bbox]}, "
                    f"object_bbox_world="
                    f"{[[round(float(v), 4) for v in bound] for bound in object_bbox]})."
                )
            raise RuntimeError("No unattached manipulation target is available.")
        (
            grasp_center_distance,
            distance,
            origin_distance,
            item,
            object_bbox,
            grasp_contacts,
        ) = min(candidates, key=lambda pair: pair[0])
        closest_contact = min(
            grasp_contacts,
            key=lambda contact: contact.separation_m,
            default=None,
        )
        report = {
            "object_id": item.object_id,
            "hand": hand_name,
            "gate": (
                "physx_opposing_finger_contact"
                if require_physx_contact
                else "generated_grasp_center"
            ),
            "distance_m": distance,
            "aabb_surface_gap_m": distance,
            "grasp_center_world": grasp_center,
            "grasp_center_to_object_m": grasp_center_distance,
            "physx_contact": (
                closest_contact.as_dict() if closest_contact is not None else None
            ),
            "physx_contacts": [contact.as_dict() for contact in grasp_contacts],
            "contact_separation_m": (
                min(contact.separation_m for contact in grasp_contacts)
                if grasp_contacts
                else None
            ),
            "hand_origin_distance_m": origin_distance,
            "finger_joint_positions": [
                float(value) for value in finger_joint_positions
            ],
            "finger_closure_l2_rad": closure_l2,
            "minimum_finger_closure_l2_rad": (DEX3_MINIMUM_CLOSURE_L2_RAD),
        }
        return item, report

    def _live_hand_bbox(
        self,
        hand: G1Hand,
    ) -> tuple[list[float], list[float]]:
        corners = self.hand_local_bbox_corners.get(hand)
        if corners is None:
            raise RuntimeError(
                f"G1 {hand} hand local bbox was not cached at scene start."
            )
        return _world_bbox_from_local_points(
            self._require_started(), self.hand_paths[hand], corners
        )

    def measured_force_closure_pair(
        self, *, hand: G1Hand, object_id: str, closing_axis_world: list[float]
    ):
        """Read current force closure without creating or updating a hold."""
        item = self._pickable(object_id)
        return force_closure_contact_pair(
            self._grasp_contacts(hand=_hand(hand), object_id=item.object_id),
            object_center_world=self._live_object_center_world(item),
            closing_axis_world=closing_axis_world,
            gripper=DEX3_1,
            require_current_impulse=True,
        )

    def _live_object_center_world(
        self,
        item: G1ManipulationObject,
    ) -> list[float]:
        """Return the live ground-truth world centre of one pickable object."""

        return _bbox_center(self._live_pickable_bbox(item))

    def _live_pickable_bbox(
        self,
        item: G1ManipulationObject,
    ) -> tuple[list[float], list[float]]:
        stage = self._require_started()
        if _joint_state(stage, item.prim_path)["attachment_joint_present"]:
            raise RuntimeError(
                "G1 finite-friction retention forbids object attachment joints."
            )
        points = self.pickable_local_geometry_points.get(item.object_id)
        if points is None:
            raise RuntimeError(
                f"Object-local geometry was not cached for pickable {item.object_id!r}."
            )
        return _world_bbox_from_local_points(
            stage,
            item.prim_path,
            points,
        )

    def physical_hold_state(self, *, object_id: str) -> dict[str, Any]:
        """Measure friction retention from live geometry and constraint state."""

        stage = self._require_started()
        item = next(
            (
                candidate
                for candidate in self.objects
                if candidate.object_id == object_id
            ),
            None,
        )
        hand = self.physical_holds.get(str(object_id))
        report = self.attachment_reports.get(str(object_id))
        if item is None or hand not in {"left", "right"} or report is None:
            return {"held": False, "object_id": str(object_id)}
        joint = _joint_state(stage, item.prim_path)
        object_bbox = self._live_pickable_bbox(item)
        hand_bbox = self._live_hand_bbox(hand)
        surface_gap = bbox_surface_gap(hand_bbox, object_bbox)
        start_position = report.get("grasp_start_position_world")
        current_position = _bbox_center(object_bbox)
        displacement = (
            math.dist(current_position, [float(value) for value in start_position])
            if isinstance(start_position, (list, tuple)) and len(start_position) == 3
            else 0.0
        )
        support_bbox = self.static_bboxes.get(item.source_support_object_id)
        support = next(
            (
                candidate
                for candidate in self.objects
                if candidate.object_id == item.source_support_object_id
            ),
            None,
        )
        if (
            support_bbox is None
            or support is None
            or self.support_clearance_query is None
            or item.source_support_object_id not in self.support_solid_meshes
        ):
            raise RuntimeError(
                f"Physical grasp {object_id!r} requires measured source support "
                f"geometry for {item.source_support_object_id!r}."
            )
        # Query the real finite collider, including high side panels without
        # filling the air over a lower tier. Recompute each live sample.
        support_clearance = self.support_clearance_query(
            target_bounds_xyz=object_bbox,
            support_bounds_xyz=support_bbox,
            support_prim_path=support.prim_path,
            solid_meshes=self.support_solid_meshes[item.source_support_object_id],
        )
        contacts = self._grasp_contacts(hand=hand, object_id=item.object_id)
        hand_position, hand_orientation = _world_pose(stage, self.hand_paths[hand])
        grasp_center_world = _point_local_to_world(
            point_local=report["contact_center_in_hand"],
            frame_position_world=hand_position,
            frame_orientation_wxyz=hand_orientation,
        )
        closing_axis_world = _direction_local_to_world(
            direction_local=report["closing_axis_in_hand"],
            frame_orientation_wxyz=hand_orientation,
        )
        finger_contacts = _finger_contacts(contacts, hand=hand)
        force_closure_pair = force_closure_contact_pair(
            contacts,
            object_center_world=self._live_object_center_world(item),
            closing_axis_world=closing_axis_world,
            gripper=DEX3_1,
            require_current_impulse=False,
        )
        held = bool(
            not joint["attachment_joint_present"]
            and not _pickable_kinematic_enabled(stage, item)
            and finger_contacts
        )
        return {
            "held": held,
            "contact_established": bool(finger_contacts),
            "transport_ready": bool(
                held and support_clearance >= G1_GRASP_SUPPORT_CLEARANCE_M
            ),
            "object_id": item.object_id,
            "hand": hand,
            "retention": "finite_coulomb_friction",
            "constraint_joint_present": bool(joint["attachment_joint_present"]),
            "aabb_surface_gap_m": float(surface_gap),
            "displacement_since_grasp_m": float(displacement),
            # Conservative 3-D gap to actual cooked source colliders, measured
            # by collision-free AABB expansion, not original tabletop height.
            "support_clearance_m": float(support_clearance),
            "source_support_object_id": item.source_support_object_id,
            "position_world": current_position,
            "grasp_center_world": grasp_center_world,
            "closing_axis_world": closing_axis_world,
            "physx_contacts": [contact.as_dict() for contact in contacts],
            "force_closure_established": force_closure_pair is not None,
            "force_closure_pair": (
                force_closure_pair.as_dict() if force_closure_pair is not None else None
            ),
        }

    def require_contact_physical_grasp(
        self,
        *,
        object_id: str,
    ) -> dict[str, Any]:
        """Fail unless friction retains live finger contact on the target."""

        state = self.physical_hold_state(object_id=object_id)
        if not state.get("held"):
            raise RuntimeError(
                "Dex3 friction retention lost live finger contact on the "
                "target: "
                f"object={object_id!r}, state={state!r}."
            )
        return state

    def detect_dropped_objects(self) -> list[dict[str, Any]]:
        """Retire friction holds whose live hand contact has been lost.

        This is an observer, not a stabilizer: it neither constrains the object
        nor changes robot motion. Calling it after each public action makes a
        transport failure persistent and observable instead of allowing stale
        bookkeeping to claim that a fallen object is still held.
        """
        dropped: list[dict[str, Any]] = []
        for object_id in tuple(self.physical_holds):
            state = self.physical_hold_state(object_id=object_id)
            if state.get("held"):
                continue
            dropped.append(state)
            self.physical_holds.pop(object_id, None)
            self.attachment_reports.pop(object_id, None)
        return dropped

    def release(self, hand: G1Hand) -> dict[str, Any]:
        """Release the physical retention ledger without moving the object."""
        stage = self._require_started()
        hand_name = _hand(hand)
        item = next(
            (
                candidate
                for candidate in self.objects
                if self.physical_holds.get(candidate.object_id) == hand_name
            ),
            None,
        )
        if item is None:
            return {"object_id": None, "hand": hand_name, "already_released": True}
        if _joint_state(stage, item.prim_path)["attachment_joint_present"]:
            raise RuntimeError(
                "G1 finite-friction release forbids object attachment joints."
            )
        self.physical_holds.pop(item.object_id)
        self.attachment_reports.pop(item.object_id, None)
        _activate_pickable_collider(item)
        _set_pickable_kinematic(stage, item, enabled=False)
        return {
            "object_id": item.object_id,
            "hand": hand_name,
            "dynamic_release": True,
            "retention": "finite_coulomb_friction",
        }

    def placement_hand_target(
        self,
        *,
        hand: G1Hand | str,
        object_id: str,
        support_surface_world: list[float],
        clearance_m: float = G1_PRE_RELEASE_CLEARANCE_M,
        correction_gain: float = 1.0,
    ) -> dict[str, Any]:
        """Translate a held object's live geometry onto a support surface.

        The target is compiled from the live hand and object geometry, so the
        same correction remains valid for a friction hold that has slipped.
        No episode-specific arm pose
        or joint angle enters this calculation.
        """
        stage = self._require_started()
        item = next(
            (
                candidate
                for candidate in self.objects
                if candidate.object_id == object_id
            ),
            None,
        )
        if item is None or item.object_id not in self.pickable_ids:
            raise RuntimeError(
                f"Placement target {object_id!r} is not an authorized pickable object."
            )
        joint = _joint_state(stage, item.prim_path)
        physical_hand = self.physical_holds.get(item.object_id)
        requested_hand = str(hand or "auto").strip().lower()
        if requested_hand == "auto":
            live_hand = physical_hand
            if live_hand not in G1_HAND_LINKS:
                raise RuntimeError(
                    f"Placement cannot auto-select a hand before {object_id!r} "
                    f"is held; live_joint={joint!r}."
                )
            hand_name = _hand(live_hand)
        else:
            hand_name = _hand(requested_hand)
        friction_held = physical_hand == hand_name
        if not friction_held:
            raise RuntimeError(
                f"Placement requires {object_id!r} to be held by the G1 "
                f"{hand_name} hand; live_joint={joint!r}, "
                f"physical_hand={physical_hand!r}."
            )
        local_points = self.pickable_local_geometry_points.get(item.object_id)
        if local_points is None:
            raise RuntimeError(
                f"Pickable local geometry was not cached for {item.object_id!r}."
            )
        object_bbox = self._live_pickable_bbox(item)
        hand_world = _world_position(stage, self.hand_paths[hand_name])
        target_world = placement_hand_target_from_geometry(
            hand_world=hand_world,
            object_bbox_world=object_bbox,
            support_surface_world=support_surface_world,
            clearance_m=clearance_m,
            correction_gain=correction_gain,
        )
        return {
            "hand": hand_name,
            "object_id": item.object_id,
            "hand_position_world": hand_world,
            "object_bbox_world": [list(object_bbox[0]), list(object_bbox[1])],
            "support_surface_world": [float(value) for value in support_surface_world],
            "clearance_m": float(clearance_m),
            "correction_gain": float(correction_gain),
            "target_position_world": target_world,
        }

    def snapshot(self) -> dict[str, Any]:
        """Read actual USD joint and geometry state for the private backend.

        ``hands`` exposes both Dex3 palm poses and live world AABBs in
        metres before grasping. Dataset certification can therefore measure
        an executed arm reach before checking native finger contact.
        ``objects`` exposes the live physical geometry and retention evidence.
        """
        stage = self._require_started()
        objects: dict[str, dict[str, Any]] = {}
        for item in self.objects:
            joint = _joint_state(stage, item.prim_path)
            if item.pickable:
                local_points = self.pickable_local_geometry_points.get(item.object_id)
                if local_points is None:
                    raise RuntimeError(
                        f"Pickable local geometry was not cached for {item.object_id!r}."
                    )
                bbox_min, bbox_max = self._live_pickable_bbox(item)
            else:
                cached = self.static_bboxes.get(item.object_id)
                if cached is None:
                    raise RuntimeError(
                        f"Static anchor bbox was not cached for {item.object_id!r}."
                    )
                bbox_min, bbox_max = cached
            object_state = {
                "prim_path": item.prim_path,
                "position": _bbox_center((bbox_min, bbox_max)),
                "initial_position": list(self.initial_positions[item.object_id]),
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                **joint,
            }
            if item.pickable:
                # Save the live rigid orientation beside true-vertex bounds
                # so target/support measurements can be independently replayed.
                object_state["orientation_world_wxyz"] = _world_pose(
                    stage, item.prim_path
                )[1]
                object_state["kinematic_enabled"] = _pickable_kinematic_enabled(
                    stage, item
                )
                attachment_evidence = self.attachment_reports.get(item.object_id)
                if attachment_evidence is not None:
                    object_state["grasp_attachment_evidence"] = dict(
                        attachment_evidence
                    )
                physical_hold = self.physical_hold_state(object_id=item.object_id)
                object_state["physically_held"] = bool(physical_hold.get("held"))
                if physical_hold.get("hand") in {"left", "right"}:
                    object_state["held_by_hand"] = physical_hold["hand"]
                object_state["physical_hold"] = physical_hold
            live_hand = self.physical_holds.get(item.object_id)
            if live_hand in self.hand_paths:
                # Private oracle evidence only: expose the exact live hand
                # point used by placement_hand_target().  This prevents the
                # planner from reconstructing it from a nominal arm posture.
                object_state["hand_position_world"] = _world_position(
                    stage, self.hand_paths[live_hand]
                )
            objects[item.object_id] = object_state
        hands: dict[str, dict[str, Any]] = {}
        for hand, prim_path in self.hand_paths.items():
            bbox_min, bbox_max = self._live_hand_bbox(hand)
            hand_position, hand_orientation = _world_pose(stage, prim_path)
            hands[hand] = {
                "prim_path": prim_path,
                "position_world": hand_position,
                "grasp_center_world": _hand_closing_center_world(
                    hand_position_world=hand_position,
                    hand_orientation_wxyz=hand_orientation,
                    gripper=DEX3_1,
                ),
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
            }
        return {
            "robot_type": "G1Robot",
            "hands": hands,
            "objects": objects,
            "grasp_contacts": [
                contact.as_dict()
                for contact in (
                    self.grasp_contact_monitor.current_contacts()
                    if self.grasp_contact_monitor is not None
                    else ()
                )
            ],
        }

    def _pickable(self, object_id: str) -> G1ManipulationObject:
        for item in self.objects:
            if item.object_id == str(object_id) and item.object_id in self.pickable_ids:
                return item
        raise ValueError(f"Unknown pickable object: {object_id!r}.")

    def _require_started(self) -> Any:
        if self.stage is None:
            raise RuntimeError("G1AttachmentExecutor.start() has not been called.")
        return self.stage


def _hand(value: Any) -> G1Hand:
    hand = str(value or "").strip().lower()
    if hand not in G1_HAND_LINKS:
        raise ValueError(f"G1 hand must be left or right, got {value!r}.")
    return hand  # type: ignore[return-value]


def _require_prim(stage: Any, prim_path: str, *, label: str) -> Any:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"{label} prim is missing: {prim_path}")
    return prim


def _world_position(stage: Any, prim_path: str) -> list[float]:
    return _world_pose(stage, prim_path)[0]


def _world_pose(
    stage: Any,
    prim_path: str,
) -> tuple[list[float], list[float]]:
    """Return one live USD pose as position and scalar-first quaternion."""
    from pxr import Gf, Usd, UsdGeom

    prim = _require_prim(stage, prim_path, label="world-position")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    point = matrix.ExtractTranslation()
    # GRScene object matrices inherit a 0.01 scene scale.  Matrix4d's direct
    # ExtractRotationQuat treats that scale as part of the rotation and turns,
    # for example, a 90-degree rotation into roughly 2.3 degrees.  Decompose
    # scale first so object and Robot orientations share one metric contract.
    rotation = Gf.Transform(matrix).GetRotation().GetQuat().GetNormalized()
    imaginary = rotation.GetImaginary()
    return (
        [float(point[0]), float(point[1]), float(point[2])],
        [
            float(rotation.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ],
    )


def read_object_local_transform(
    stage: Any,
    prim_path: str,
) -> G1ObjectLocalTransform:
    """Capture the authored xform operations without normalizing their units."""
    from pxr import UsdGeom

    prim = _require_prim(stage, prim_path, label="local-transform")
    xformable = UsdGeom.Xformable(prim)
    order_attr = prim.GetAttribute("xformOpOrder")
    order = order_attr.Get() if order_attr and order_attr.IsValid() else None
    operations = xformable.GetOrderedXformOps()
    if order is None or not operations:
        raise RuntimeError(
            f"Pickable object requires an authored local transform: {prim_path}"
        )
    values: list[tuple[str, Any, Any]] = []
    for operation in operations:
        attribute = operation.GetAttr()
        value = attribute.Get()
        if value is None:
            raise RuntimeError(
                f"Pickable xform attribute has no authored value: "
                f"{prim_path}.{attribute.GetName()}"
            )
        values.append(
            (
                str(attribute.GetName()),
                attribute.GetTypeName(),
                value,
            )
        )
    return G1ObjectLocalTransform(
        xform_op_order=order,
        op_values=tuple(values),
    )


def _restore_pickable_local_transform(
    stage: Any,
    item: G1ManipulationObject,
    local_transform: G1ObjectLocalTransform,
) -> None:
    """Restore authored local xform operations and remove carried momentum."""
    from pxr import Gf, UsdPhysics
    from tarsphere.simulator.core.util.physics import activate_collider
    from omni.isaac.core.utils.prims import get_prim_at_path

    prim = _require_prim(stage, item.prim_path, label=item.object_id)
    activate_collider(get_prim_at_path(item.prim_path))
    restore_authored_xform_ops(prim, local_transform)
    rigid_body = UsdPhysics.RigidBodyAPI(prim)
    if rigid_body:
        rigid_body.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        rigid_body.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    _set_pickable_kinematic(stage, item, enabled=True)


def restore_authored_xform_ops(
    prim: Any,
    local_transform: G1ObjectLocalTransform,
) -> None:
    """Recreate the exact authored xform stack after PhysX rewrites it."""
    from pxr import Sdf, UsdGeom

    xformable = UsdGeom.Xformable(prim)
    expected_names = {
        attribute_name for attribute_name, _, _ in local_transform.op_values
    }
    current_names = tuple(
        str(operation.GetAttr().GetName())
        for operation in xformable.GetOrderedXformOps()
    )
    xformable.ClearXformOpOrder()
    for attribute_name in current_names:
        if attribute_name not in expected_names:
            prim.RemoveProperty(attribute_name)
    for attribute_name, type_name, value in local_transform.op_values:
        attribute = prim.GetAttribute(attribute_name)
        if not attribute or not attribute.IsValid():
            attribute = prim.CreateAttribute(
                attribute_name,
                type_name,
                custom=False,
            )
        if attribute.GetTypeName() != type_name:
            raise RuntimeError(
                "Pickable xform attribute changed type before reset: "
                f"{prim.GetPath()}.{attribute_name}"
            )
        attribute.Set(value)
    order_attr = prim.GetAttribute("xformOpOrder")
    if not order_attr or not order_attr.IsValid():
        order_attr = prim.CreateAttribute(
            "xformOpOrder",
            Sdf.ValueTypeNames.TokenArray,
            custom=False,
        )
    order_attr.Set(local_transform.xform_op_order)


_read_object_local_transform = read_object_local_transform
_restore_authored_xform_ops = restore_authored_xform_ops


def _bbox_center(
    bbox: tuple[list[float], list[float]],
) -> list[float]:
    """Return the metric centre of a finite world-space bounding box."""
    lower, upper = bbox
    if (
        len(lower) != 3
        or len(upper) != 3
        or any(not math.isfinite(float(value)) for value in (*lower, *upper))
        or any(float(lower[axis]) > float(upper[axis]) for axis in range(3))
    ):
        raise ValueError("World bbox must contain finite ordered 3-D bounds.")
    return [0.5 * (float(lower[axis]) + float(upper[axis])) for axis in range(3)]


def _normalized_quaternion(values: list[float]) -> list[float]:
    quaternion = [float(value) for value in values]
    if len(quaternion) != 4 or not all(math.isfinite(value) for value in quaternion):
        raise ValueError("Quaternion must contain four finite values.")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-12:
        raise ValueError("Quaternion norm must be positive.")
    return [value / norm for value in quaternion]


def _multiply_quaternions(first: list[float], second: list[float]) -> list[float]:
    return _normalized_quaternion(_multiply_quaternions_raw(first, second))


def _multiply_quaternions_raw(
    first: list[float],
    second: list[float],
) -> list[float]:
    first_w, first_x, first_y, first_z = first
    second_w, second_x, second_y, second_z = second
    return [
        first_w * second_w
        - first_x * second_x
        - first_y * second_y
        - first_z * second_z,
        first_w * second_x
        + first_x * second_w
        + first_y * second_z
        - first_z * second_y,
        first_w * second_y
        - first_x * second_z
        + first_y * second_w
        + first_z * second_x,
        first_w * second_z
        + first_x * second_y
        - first_y * second_x
        + first_z * second_w,
    ]


def _inverse_quaternion(quaternion: list[float]) -> list[float]:
    normalized = _normalized_quaternion(quaternion)
    return [normalized[0], -normalized[1], -normalized[2], -normalized[3]]


def _rotate_vector(quaternion: list[float], vector: list[float]) -> list[float]:
    normalized = _normalized_quaternion(quaternion)
    vector_quaternion = [0.0, *[float(value) for value in vector]]
    rotated = _multiply_quaternions_raw(
        _multiply_quaternions_raw(normalized, vector_quaternion),
        _inverse_quaternion(normalized),
    )
    return rotated[1:]


def _point_world_to_local(
    *,
    point_world: list[float],
    frame_position_world: list[float],
    frame_orientation_wxyz: list[float],
) -> list[float]:
    """Express a metric world point in one moving rigid frame."""

    if len(point_world) != 3 or len(frame_position_world) != 3:
        raise ValueError("Rigid-frame point transforms require two 3-D points.")
    return _rotate_vector(
        _inverse_quaternion(frame_orientation_wxyz),
        _subtract_vectors(point_world, frame_position_world),
    )


def _point_local_to_world(
    *,
    point_local: list[float],
    frame_position_world: list[float],
    frame_orientation_wxyz: list[float],
) -> list[float]:
    """Transform a metric rigid-frame point into world coordinates."""

    if len(point_local) != 3 or len(frame_position_world) != 3:
        raise ValueError("Rigid-frame point transforms require two 3-D points.")
    return _add_vectors(
        frame_position_world,
        _rotate_vector(frame_orientation_wxyz, point_local),
    )


def _direction_world_to_local(
    *,
    direction_world: list[float],
    frame_orientation_wxyz: list[float],
) -> list[float]:
    """Express a unit world direction in one moving rigid frame."""

    return _unit_vector(
        _rotate_vector(
            _inverse_quaternion(frame_orientation_wxyz),
            direction_world,
        )
    )


def _direction_local_to_world(
    *,
    direction_local: list[float],
    frame_orientation_wxyz: list[float],
) -> list[float]:
    """Transform a unit rigid-frame direction into world coordinates."""

    return _unit_vector(
        _rotate_vector(
            frame_orientation_wxyz,
            direction_local,
        )
    )


def _unit_vector(values: list[float]) -> list[float]:
    vector = [float(value) for value in values]
    if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
        raise ValueError("Direction vectors must contain three finite values.")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-12:
        raise ValueError("Direction vectors must have positive length.")
    return [value / norm for value in vector]


def _add_vectors(first: list[float], second: list[float]) -> list[float]:
    return [first[axis] + second[axis] for axis in range(3)]


def _subtract_vectors(first: list[float], second: list[float]) -> list[float]:
    return [first[axis] - second[axis] for axis in range(3)]


def _activate_pickable_collider(item: G1ManipulationObject) -> None:
    """Activate a pickable collider without leaking Isaac imports to tests."""
    from tarsphere.simulator.core.util.physics import activate_collider
    from omni.isaac.core.utils.prims import get_prim_at_path

    activate_collider(get_prim_at_path(item.prim_path))


def _set_pickable_metric_world_pose(
    stage: Any,
    item: G1ManipulationObject,
    *,
    position_world: list[float],
    orientation_wxyz: list[float],
    object_local_unit_in_meters: float,
) -> None:
    """Author a metric world pose without erasing the scene's unit scale."""
    from pxr import Gf, Usd, UsdGeom

    position = [float(value) for value in position_world]
    orientation = _normalized_quaternion(orientation_wxyz)
    scale = float(object_local_unit_in_meters)
    if (
        len(position) != 3
        or not all(math.isfinite(value) for value in position)
        or not math.isfinite(scale)
        or scale <= 0.0
    ):
        raise ValueError(
            "Pickable world pose requires a finite position and positive "
            "object-local unit."
        )
    prim = _require_prim(stage, item.prim_path, label=item.object_id)
    parent = prim.GetParent()
    if not parent or not parent.IsValid():
        raise RuntimeError(f"Pickable {item.object_id!r} has no valid scene parent.")
    parent_world = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    metric_world = Gf.Transform()
    metric_world.SetScale(Gf.Vec3d(scale, scale, scale))
    metric_world.SetRotation(
        Gf.Rotation(
            Gf.Quatd(
                orientation[0],
                Gf.Vec3d(orientation[1], orientation[2], orientation[3]),
            )
        )
    )
    metric_world.SetTranslation(Gf.Vec3d(*position))
    local_matrix = metric_world.GetMatrix() * parent_world.GetInverse()
    transform_op = UsdGeom.Xformable(prim).MakeMatrixXform()
    transform_op.Set(local_matrix)


def _set_pickable_kinematic(
    stage: Any,
    item: G1ManipulationObject,
    *,
    enabled: bool,
) -> None:
    """Hold authored pose before grasp; release dynamics only when handled."""
    from pxr import UsdPhysics

    prim = _require_prim(stage, item.prim_path, label=item.object_id)
    rigid_body = UsdPhysics.RigidBodyAPI(prim)
    if not rigid_body:
        raise RuntimeError(
            f"Pickable rigid body API is missing for {item.object_id!r}."
        )
    rigid_body.CreateKinematicEnabledAttr().Set(bool(enabled))


def _pickable_kinematic_enabled(
    stage: Any,
    item: G1ManipulationObject,
) -> bool:
    """Read whether one pickable is currently excluded from dynamics."""
    from pxr import UsdPhysics

    prim = _require_prim(stage, item.prim_path, label=item.object_id)
    rigid_body = UsdPhysics.RigidBodyAPI(prim)
    if not rigid_body:
        raise RuntimeError(
            f"Pickable rigid body API is missing for {item.object_id!r}."
        )
    attribute = rigid_body.GetKinematicEnabledAttr()
    value = attribute.Get() if attribute and attribute.IsValid() else None
    if value is None:
        raise RuntimeError(
            f"Pickable kinematic state is unavailable for {item.object_id!r}."
        )
    return bool(value)


def _zero_pickable_velocity(stage: Any, item: G1ManipulationObject) -> None:
    """Remove constraint-carried momentum without changing object pose."""
    from pxr import Gf, UsdPhysics

    prim = _require_prim(stage, item.prim_path, label=item.object_id)
    rigid_body = UsdPhysics.RigidBodyAPI(prim)
    if not rigid_body:
        raise RuntimeError(
            f"Pickable rigid body API is missing for {item.object_id!r}."
        )
    rigid_body.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    rigid_body.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))


def _set_pickable_mass(
    stage: Any,
    item: G1ManipulationObject,
    mass_kg: float,
) -> None:
    """Override unreliable asset-inferred mass on the semantic rigid root."""
    from pxr import UsdPhysics

    prim = _require_prim(stage, item.prim_path, label=item.object_id)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    if not mass_api:
        raise RuntimeError(f"Cannot apply mass to pickable {item.object_id!r}.")
    mass_api.CreateMassAttr().Set(float(mass_kg))


def _enable_pickable_physics(stage: Any, item: G1ManipulationObject) -> None:
    prim = _require_prim(stage, item.prim_path, label=item.object_id)
    configure_dynamic_pickable(prim, kinematic=False)


def _bind_physical_grasp_material(
    stage: Any,
    *,
    target_path: str,
    material_path: str,
) -> None:
    """Bind the composed Dex3 friction material to one dynamic target."""

    from pxr import UsdShade

    target = _require_prim(stage, target_path, label="physical grasp target")
    material_prim = _require_prim(
        stage, material_path, label="Dex3 physical grasp material"
    )
    material = UsdShade.Material(material_prim)
    if not material:
        raise RuntimeError(
            f"Dex3 grasp material is not a UsdShade material: {material_path}"
        )
    UsdShade.MaterialBindingAPI.Apply(target).Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )


def _create_attachment_joint(
    stage: Any,
    *,
    prim_path: str,
    body0: str,
    body1: str,
) -> Any:
    """Create a FixedJoint whose two frames coincide at the live object pose."""
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    object_prim = _require_prim(stage, body0, label="attachment body0")
    hand_prim = _require_prim(stage, body1, label="attachment body1")
    world_from_object = UsdGeom.Xformable(object_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    world_from_hand = UsdGeom.Xformable(hand_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    object_world_position = world_from_object.ExtractTranslation()
    object_world_rotation = world_from_object.ExtractRotationQuat().GetNormalized()
    hand_world_rotation = world_from_hand.ExtractRotationQuat().GetNormalized()
    object_position_in_hand = world_from_hand.GetInverse().Transform(
        object_world_position
    )
    object_rotation_in_hand = (
        hand_world_rotation.GetInverse() * object_world_rotation
    ).GetNormalized()

    joint = UsdPhysics.FixedJoint.Define(stage, prim_path)
    joint.GetBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.GetBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*object_position_in_hand))
    joint.CreateLocalRot1Attr().Set(_quatd_to_quatf(Gf, object_rotation_in_hand))
    joint.CreateJointEnabledAttr().Set(True)
    joint.CreateExcludeFromArticulationAttr().Set(True)
    PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
    return joint.GetPrim()


def _create_vertical_manipulation_stance_joint(
    stage: Any,
    *,
    prim_path: str,
    body0: str,
) -> Any:
    """Atomically create one articulation-excluded vertical stance joint."""

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    body0_prim = _require_prim(stage, body0, label="world FixedJoint body0")
    world_from_body0 = UsdGeom.Xformable(body0_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    world_position = world_from_body0.ExtractTranslation()
    world_rotation = world_from_body0.ExtractRotationQuat().GetNormalized()

    existing = stage.GetPrimAtPath(prim_path)
    if existing and existing.IsValid():
        raise RuntimeError(
            f"G1 manipulation stance path is already occupied: {prim_path}"
        )
    joint = UsdPhysics.PrismaticJoint.Define(stage, prim_path)
    # Define begins without bodies. Disable it first, then exclude it from the
    # articulation before either relationship is authored; PhysX therefore
    # never observes a live external joint as part of G1's articulation tree.
    joint.CreateJointEnabledAttr().Set(False)
    joint.CreateExcludeFromArticulationAttr().Set(True)
    joint.GetBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.GetBody1Rel().SetTargets([])
    joint.CreateAxisAttr().Set(UsdPhysics.Tokens.z)
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    # Express both frames in the world basis. The free Z axis must not inherit
    # the pelvis tilt at the instant the stance is established.
    joint.CreateLocalRot0Attr().Set(_quatd_to_quatf(Gf, world_rotation.GetInverse()))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*world_position))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateJointEnabledAttr().Set(True)
    return joint.GetPrim()


def _flush_physx_changes() -> None:
    """Apply a live joint toggle before the next controller takes ownership."""

    from omni.physx import get_physx_simulation_interface

    interface = get_physx_simulation_interface()
    if interface is None:
        raise RuntimeError("PhysX simulation interface is unavailable.")
    interface.flush_changes()


def _require_vertical_manipulation_stance_joint(
    stage: Any,
    *,
    prim_path: str,
    body0: str,
) -> Any:
    """Return the frozen external joint or fail on any topology drift."""

    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsA(UsdPhysics.PrismaticJoint):
        raise RuntimeError(
            f"Pre-authored G1 manipulation stance joint is missing: {prim_path}"
        )
    joint = UsdPhysics.PrismaticJoint(prim)
    body0_targets = [str(path) for path in joint.GetBody0Rel().GetTargets()]
    body1_targets = [str(path) for path in joint.GetBody1Rel().GetTargets()]
    if body0_targets != [body0]:
        raise RuntimeError(
            "G1 manipulation stance joint must bind exactly one live pelvis: "
            f"expected={[body0]!r}, actual={body0_targets!r}."
        )
    if body1_targets:
        raise RuntimeError(
            "G1 manipulation stance joint must bind its second frame to the "
            f"world, not {body1_targets!r}."
        )
    if joint.GetAxisAttr().Get() != UsdPhysics.Tokens.z:
        raise RuntimeError(
            "G1 manipulation stance joint must leave the world Z axis free."
        )
    if joint.GetExcludeFromArticulationAttr().Get() is not True:
        raise RuntimeError(
            "G1 manipulation stance joint must be excluded from the live "
            "articulation topology."
        )
    return joint


def _vertical_manipulation_stance_joint_enabled(
    stage: Any,
    *,
    prim_path: str,
    body0: str,
) -> bool:
    joint = _require_vertical_manipulation_stance_joint(
        stage,
        prim_path=prim_path,
        body0=body0,
    )
    enabled_attr = joint.GetJointEnabledAttr()
    enabled = (
        bool(enabled_attr.Get()) if enabled_attr and enabled_attr.IsValid() else True
    )
    return enabled


def _quatd_to_quatf(gf: Any, value: Any) -> Any:
    imaginary = value.GetImaginary()
    return gf.Quatf(
        float(value.GetReal()),
        gf.Vec3f(*(float(imaginary[index]) for index in range(3))),
    )


def _world_bbox_as_local_corners(
    stage: Any,
    prim_path: str,
    world_bbox: tuple[list[float], list[float]],
) -> list[list[float]]:
    """Express a one-time world AABB as rigid object-local corner points."""
    from pxr import Gf, Usd, UsdGeom

    prim = _require_prim(stage, prim_path, label="local-bbox")
    world_from_local = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    local_from_world = world_from_local.GetInverse()
    lower, upper = world_bbox
    return [
        _gf_point(local_from_world.Transform(Gf.Vec3d(x, y, z)))
        for x in (lower[0], upper[0])
        for y in (lower[1], upper[1])
        for z in (lower[2], upper[2])
    ]


def _world_bbox_from_local_points(
    stage: Any,
    prim_path: str,
    local_points: list[list[float]],
) -> tuple[list[float], list[float]]:
    """Transform cached rigid geometry by its current live USD pose."""
    from pxr import Usd, UsdGeom

    points = np.asarray(local_points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[0] == 0
        or points.shape[1] != 3
        or not np.all(np.isfinite(points))
    ):
        raise RuntimeError(
            f"Cached geometry for {prim_path!r} must contain finite 3D points."
        )
    prim = _require_prim(stage, prim_path, label="live-bbox")
    world_from_local = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    matrix = np.asarray(world_from_local, dtype=float)
    world_points = points @ matrix[:3, :3] + matrix[3, :3]
    if not np.all(np.isfinite(world_points)):
        raise RuntimeError(f"Non-finite live geometry transform: {prim_path}")
    return (
        world_points.min(axis=0).tolist(),
        world_points.max(axis=0).tolist(),
    )


def _gf_point(point: Any) -> list[float]:
    values = [float(point[axis]) for axis in range(3)]
    if not all(math.isfinite(value) and abs(value) < 1e6 for value in values):
        raise RuntimeError(
            f"USD transform produced a non-finite bbox point: {values!r}."
        )
    return values


def _point_to_bbox_distance(
    point: list[float],
    bbox: tuple[list[float], list[float]],
) -> float:
    lower, upper = bbox
    squared = 0.0
    for axis in range(3):
        delta = max(lower[axis] - point[axis], 0.0, point[axis] - upper[axis])
        squared += delta * delta
    return math.sqrt(squared)


def bbox_surface_gap(
    first: tuple[list[float], list[float]],
    second: tuple[list[float], list[float]],
) -> float:
    """Return the Euclidean surface gap between two finite world AABBs."""
    first_lower, first_upper = first
    second_lower, second_upper = second
    values = [*first_lower, *first_upper, *second_lower, *second_upper]
    if (
        len(first_lower) != 3
        or len(first_upper) != 3
        or len(second_lower) != 3
        or len(second_upper) != 3
        or not all(math.isfinite(float(value)) for value in values)
        or any(first_lower[axis] > first_upper[axis] for axis in range(3))
        or any(second_lower[axis] > second_upper[axis] for axis in range(3))
    ):
        raise ValueError("World AABBs must contain finite ordered 3-D bounds.")
    squared = 0.0
    for axis in range(3):
        gap = max(
            float(first_lower[axis]) - float(second_upper[axis]),
            float(second_lower[axis]) - float(first_upper[axis]),
            0.0,
        )
        squared += gap * gap
    return math.sqrt(squared)


def placement_hand_target_from_geometry(
    *,
    hand_world: list[float],
    object_bbox_world: tuple[list[float], list[float]],
    support_surface_world: list[float],
    clearance_m: float = G1_PRE_RELEASE_CLEARANCE_M,
    correction_gain: float = 1.0,
) -> list[float]:
    """Return a feedback hand target that moves a held AABB to a surface."""
    hand = [float(value) for value in hand_world]
    lower = [float(value) for value in object_bbox_world[0]]
    upper = [float(value) for value in object_bbox_world[1]]
    surface = [float(value) for value in support_surface_world]
    clearance = float(clearance_m)
    gain = float(correction_gain)
    if (
        len(hand) != 3
        or len(lower) != 3
        or len(upper) != 3
        or len(surface) != 3
        or not all(
            math.isfinite(value)
            for value in [
                *hand,
                *lower,
                *upper,
                *surface,
                clearance,
                gain,
            ]
        )
        or any(lower[axis] > upper[axis] for axis in range(3))
        or clearance < 0.0
        or not 0.0 < gain <= 1.0
    ):
        raise ValueError("Placement geometry must contain finite, ordered 3-D values.")
    object_center_xy = [0.5 * (lower[axis] + upper[axis]) for axis in range(2)]
    translation = [
        surface[0] - object_center_xy[0],
        surface[1] - object_center_xy[1],
        surface[2] + clearance - lower[2],
    ]
    return [hand[axis] + gain * translation[axis] for axis in range(3)]


def _joint_state(stage: Any, object_path: str) -> dict[str, Any]:
    from pxr import UsdPhysics

    joint_path = f"{object_path}/pick_joint"
    prim = stage.GetPrimAtPath(joint_path)
    if not prim or not prim.IsValid() or not prim.IsA(UsdPhysics.FixedJoint):
        return {
            "attached": False,
            "attachment_joint_present": bool(prim and prim.IsValid()),
            "attachment_joint_valid": False,
            "attached_to": None,
            "hand": None,
        }
    joint = UsdPhysics.FixedJoint(prim)
    body0 = [str(path) for path in joint.GetBody0Rel().GetTargets()]
    body1 = [str(path) for path in joint.GetBody1Rel().GetTargets()]
    enabled_attr = prim.GetAttribute("physics:jointEnabled")
    enabled = (
        bool(enabled_attr.Get()) if enabled_attr and enabled_attr.IsValid() else True
    )
    attached_to = body1[0] if len(body1) == 1 else None
    hand = None
    for candidate, link in G1_HAND_LINKS.items():
        if attached_to and attached_to.endswith("/" + link):
            hand = candidate
            break
    valid = (
        enabled
        and body0 == [object_path]
        and attached_to is not None
        and hand is not None
    )
    return {
        "attached": bool(valid),
        "attachment_joint_present": True,
        "attachment_joint_valid": bool(valid),
        "attached_to": attached_to,
        "hand": hand,
    }


__all__ = [
    "G1AttachmentExecutor",
    "G1ManipulationObject",
    "bbox_surface_gap",
    "attachment_command_from_decision",
    "placement_hand_target_from_geometry",
    "manipulation_objects_from_episode",
    "read_object_local_transform",
    "restore_authored_xform_ops",
]
_G1_MANIPULATION_STANCE_JOINT = "manipulation_stance_joint"
