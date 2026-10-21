"""Episode-scoped text, visual, spatial, key, human, and grounding memory."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from tarsphere.harness.memory.contracts import MemoryState
from tarsphere.harness.memory.human import HumanMemory
from tarsphere.harness.memory.key import KeyMemoryLedger
from tarsphere.harness.memory.semantic import SemanticMap
from tarsphere.harness.memory.text import TextMemory
from tarsphere.harness.memory.visual import VisualMemory
from tarsphere.harness.harness import ProtocolConfig
from tarsphere.infra.manipulation.agent_grasp import GraspFailureCode
from tarsphere.infra.control.physical_contracts import (
    G1_GRASP_SUPPORT_SETTLE_MAX_HORIZONTAL_DISPLACEMENT_M,
)


@dataclass
class GroundingEntry:
    """One agent-established grounding result for a semantic label."""

    label: str
    world_xy: list[float]
    world_z_m: float
    grounding_depth_m: float
    bbox_2d: list[float]
    score: float
    action_step: int
    geometry_valid: bool = True

    @property
    def world_xyz(self) -> list[float]:
        return [
            float(self.world_xy[0]),
            float(self.world_xy[1]),
            float(self.world_z_m),
        ]

    def describe(self) -> str:
        validity = (
            "geometry valid"
            if self.geometry_valid
            else "geometry invalidated; reacquire required"
        )
        return (
            f'"{self.label}" grounded in ego RGB at action step '
            f"{self.action_step} (score {self.score:.2f}; {validity})"
        )


class GroundingCache:
    """Last agent-established grounding per label for the current episode."""

    def __init__(self) -> None:
        self._entries: dict[str, GroundingEntry] = {}

    def put(self, entry: GroundingEntry) -> None:
        self._entries[entry.label] = entry

    def get(self, label: str) -> GroundingEntry | None:
        return self._entries.get(label)

    def invalidate_geometry(self, label: str) -> None:
        """Drop remembered metric state while keeping the target identity."""

        entry = self._entries.get(label)
        if entry is None:
            return
        self._entries[label] = GroundingEntry(
            label=entry.label,
            world_xy=list(entry.world_xy),
            world_z_m=float(entry.world_z_m),
            grounding_depth_m=float(entry.grounding_depth_m),
            bbox_2d=list(entry.bbox_2d),
            score=float(entry.score),
            action_step=int(entry.action_step),
            geometry_valid=False,
        )

    @property
    def entries(self) -> dict[str, GroundingEntry]:
        return dict(self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def remove(self, label: str) -> None:
        self._entries.pop(str(label), None)


class ReplaceGroundingEffect:
    """Reversible current-view grounding update for an executable block."""

    def __init__(
        self,
        *,
        memory: "EpisodicMemory",
        label: str,
        grounding: Any,
        bbox: list[float],
        action_step: int,
        capture_id: int,
    ) -> None:
        self._memory = memory
        self._label = str(label)
        self._grounding = grounding
        self._bbox = list(bbox)
        self._action_step = int(action_step)
        self._capture_id = int(capture_id)
        self._previous_entry = memory.grounding.get(self._label)
        self._previous_current = memory._current_view_groundings.get(self._label)
        self._commit_started = False
        self._committed = False

    def commit(self) -> None:
        if self._committed:
            raise RuntimeError("Grounding effect was committed twice.")
        self._commit_started = True
        detection = self._grounding.detection
        try:
            self._memory.remember_grounding(
                label=self._label,
                world_xy=list(self._grounding.target_center_world_m[:2]),
                world_z_m=float(self._grounding.target_center_world_m[2]),
                grounding_depth_m=float(self._grounding.target_center_camera_m[2]),
                bbox_2d=self._bbox,
                score=float(detection.score),
                action_step=self._action_step,
            )
            self._memory.remember_current_view_grounding(
                label=self._label,
                capture_id=self._capture_id,
                grounding=self._grounding,
            )
        except Exception:
            self.rollback()
            raise
        self._committed = True

    def rollback(self) -> None:
        if not self._commit_started:
            return
        if self._previous_entry is None:
            self._memory.grounding.remove(self._label)
        else:
            self._memory.grounding.put(self._previous_entry)
        if self._previous_current is None:
            self._memory._current_view_groundings.pop(self._label, None)
        else:
            self._memory._current_view_groundings[self._label] = self._previous_current
        self._commit_started = False
        self._committed = False


class EpisodicMemory:
    """One authority for all public state accumulated during an episode."""

    def __init__(
        self,
        *,
        protocol: ProtocolConfig,
        actor_id: str = "robot",
    ) -> None:
        self.protocol = protocol
        self.text = TextMemory(actor_id=actor_id)
        self.human = HumanMemory(actor_id=actor_id)
        self.key = KeyMemoryLedger()
        # The protocol value counts history only. VisualMemory retains the
        # current observation as the final record in addition to that history.
        self.visual = VisualMemory(
            horizon=protocol.visual_history_horizon + 1,
        )
        self.grounding = GroundingCache()
        self._current_view_groundings: dict[str, tuple[int, Any]] = {}
        self.semantic = SemanticMap()
        self.actor_id = actor_id
        self._latest_physical_contact: dict[str, float | str] | None = None

    @property
    def strategy_planner(self) -> Any:
        return self.text.strategy_planner

    def begin_action_step(
        self,
        *,
        observation: Any,
        perception: Any,
        action_step: int,
        record_visual_history: bool = True,
    ) -> MemoryState:
        """Integrate the pre-action observation and retrieve policy context."""

        pose = getattr(observation, "pose", None)
        self.text.begin_action_step(
            action_step=action_step,
            observation=observation,
            perception=perception,
        )
        self.human.begin_action_step(
            action_step=action_step,
            observation=observation,
        )
        visible_humans = dict(getattr(observation, "visible_humans", None) or {})
        for human_id, human_xy in visible_humans.items():
            self.human.record_visible_sighting(
                human_id,
                human_xy,
                action_step=int(action_step),
                place="",
            )
        if record_visual_history:
            self.visual.record(
                action_step=action_step,
                views=getattr(observation, "views", None) or {},
            )
        # Public manipulation/motion events belong to this observation. Apply
        # their invalidations before materializing model-visible memory so the
        # Agent never receives one stale action-step of metric geometry.
        self._invalidate_grounding_from_public_events(
            getattr(observation, "public_events", None) or [],
        )
        state = MemoryState(
            actor_id=self.actor_id,
            action_step=int(action_step),
            text_history=self.text.retrieved_records(
                self.protocol.text_history_horizon
            ),
            current_text=self.text.current_record,
            visual_history=self.visual.retrieved_frames(),
            camera_gaze_tilt_deg=self._camera_gaze_tilt_deg(pose),
            body_posture=self._body_posture(pose),
            known_objects=(list(self.grounding.entries.values())),
            key_memories=(self.key.records),
            grasped_hands=self.text.grasped_hands,
            humans=self.human.retrieved_humans(),
            semantic_entities=(self.semantic.summaries),
        )
        return state

    def record_policy_decision(
        self,
        *,
        policy_step: int,
        envelope: Any,
        actions: list[dict[str, Any]],
        accepted_memory_update_batches: list[list[dict[str, Any]]],
        accepted_event_output_batches: list[dict[str, Any]],
        plan_updates_accepted: bool,
    ) -> None:
        plan_update = None
        if plan_updates_accepted:
            for batch in accepted_event_output_batches:
                for kind in ("plan",):
                    if kind in batch:
                        plan_update = {"kind": kind, **dict(batch[kind])}
        self.text.record_policy_decision(
            policy_step=policy_step,
            plan_update=plan_update,
            proposed_actions=actions,
        )
        self.key.apply_batches(
            accepted_memory_update_batches,
            action_step=int(self.text.current_record.action_step),
        )

    def record_executed_action(
        self,
        *,
        policy_step: int,
        action: dict[str, Any],
    ) -> None:
        self.text.record_executed_action(
            policy_step=policy_step,
            action=action,
        )

    def complete_action_step(self, feedback: dict[str, Any]) -> None:
        self.text.complete_action_step(feedback)
        self._latest_physical_contact = self._public_physical_contact(feedback)
        self._invalidate_grounding_from_feedback(feedback)

    @staticmethod
    def _public_physical_contact(
        feedback: dict[str, Any],
    ) -> dict[str, float | str] | None:
        """Keep one body-relative marker from the latest physical contact."""

        if not bool(feedback.get("physical_contact", False)):
            return None
        contacts = feedback.get("physical_contact_details")
        if not isinstance(contacts, list) or not contacts:
            raise ValueError(
                "Physical contact feedback requires physical_contact_details."
            )
        details = contacts[0]
        if not isinstance(details, dict):
            raise ValueError("Physical contact detail must be a mapping.")
        try:
            bearing_deg = float(details["relative_bearing_deg"])
            distance_m = float(details["distance_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Physical contact feedback requires numeric "
                "relative_bearing_deg and distance_m."
            ) from exc
        if not math.isfinite(bearing_deg):
            raise ValueError("Physical contact bearing must be finite.")
        if not math.isfinite(distance_m) or distance_m < 0.0:
            raise ValueError(
                "Physical contact distance must be finite and non-negative."
            )
        return {
            "relative_bearing_deg": round(bearing_deg, 1),
            "distance_m": round(distance_m, 2),
            "direction": str(details.get("direction") or "obstacle"),
        }

    def remember_grounding(
        self,
        *,
        label: str,
        world_xy: list[float],
        world_z_m: float,
        grounding_depth_m: float,
        bbox_2d: list[float],
        score: float,
        action_step: int,
    ) -> None:
        self.grounding.put(
            GroundingEntry(
                label=label,
                world_xy=list(world_xy),
                world_z_m=float(world_z_m),
                grounding_depth_m=float(grounding_depth_m),
                bbox_2d=list(bbox_2d),
                score=float(score),
                action_step=int(action_step),
                geometry_valid=True,
            )
        )

    def remember_current_view_grounding(
        self,
        *,
        label: str,
        capture_id: int,
        grounding: Any,
    ) -> None:
        self._current_view_groundings[str(label)] = (
            int(capture_id),
            grounding,
        )

    def recalled_current_view_grounding(
        self,
        label: str,
        *,
        capture_id: int,
    ) -> Any | None:
        cached = self._current_view_groundings.get(str(label))
        if cached is None or int(cached[0]) != int(capture_id):
            return None
        return cached[1]

    def recalled_grounding(self, label: str) -> GroundingEntry | None:
        return self.grounding.get(label)

    def invalidate_grounding_geometry(self, label: str) -> None:
        self.grounding.invalidate_geometry(label)

    def _invalidate_grounding_from_public_events(
        self,
        events: list[Any],
    ) -> None:
        """Drop remembered geometry after observed human object manipulation."""

        for event in events:
            if not isinstance(event, dict):
                continue
            for key in ("target", "object"):
                label = str(event.get(key) or "").strip()
                if label and label in self.grounding.entries:
                    self.grounding.invalidate_geometry(label)

    @staticmethod
    def _grasp_target_object_id(feedback: dict[str, Any]) -> str:
        """Return the episode object id for one grasp feedback payload."""

        object_id = str(feedback.get("object_id") or "").strip()
        if object_id:
            return object_id
        grasp_skill = feedback.get("grasp_skill") or {}
        if isinstance(grasp_skill, dict):
            return str(grasp_skill.get("object_id") or "").strip()
        return ""

    @classmethod
    def _feedback_reports_grasp_target_contact(
        cls,
        feedback: dict[str, Any],
    ) -> bool:
        """True when Dex3 PhysX contacts name the grasp target object."""

        details = feedback.get("grasp_contact_details")
        if not isinstance(details, list) or not details:
            return False
        object_id = cls._grasp_target_object_id(feedback)
        if not object_id:
            return False
        for item in details:
            if not isinstance(item, dict):
                continue
            if str(item.get("record_type") or "").strip() == "grasp_grounding":
                continue
            contact_object_id = str(
                item.get("target_object_id") or item.get("object_id") or ""
            ).strip()
            if contact_object_id == object_id:
                return True
        return False

    def _invalidate_grounding_from_feedback(self, feedback: dict[str, Any]) -> None:
        """Drop remembered geometry after a displacing physical event."""

        target = str(feedback.get("target") or "").strip()
        action = str(feedback.get("action") or "").strip()
        if not target:
            return
        dropped_objects = feedback.get("dropped_objects") or []
        if isinstance(dropped_objects, list) and dropped_objects:
            self.grounding.invalidate_geometry(target)
            return
        grasp_failure = feedback.get("grasp_failure") or {}
        failure_code = str(grasp_failure.get("code") or "").strip()
        if action == "grasp_object":
            if feedback.get("grasp_skill"):
                self.grounding.invalidate_geometry(target)
                return
            if self._feedback_reports_grasp_target_contact(feedback):
                self.grounding.invalidate_geometry(target)
                return
            if failure_code in {
                GraspFailureCode.CONTACT_MISS,
                GraspFailureCode.RETENTION_LOST,
                GraspFailureCode.SUPPORT_UNSTABLE,
            }:
                self.grounding.invalidate_geometry(target)
                return
        displacement_m = feedback.get("grounding_displacement_m")
        if displacement_m is None:
            return
        try:
            measured = float(displacement_m)
        except (TypeError, ValueError):
            raise ValueError(
                "grounding_displacement_m must be numeric when present."
            ) from None
        if (
            math.isfinite(measured)
            and measured >= G1_GRASP_SUPPORT_SETTLE_MAX_HORIZONTAL_DISPLACEMENT_M
        ):
            self.grounding.invalidate_geometry(target)

    def configure_human_profiles(
        self,
        profiles: list[dict[str, Any]],
    ) -> None:
        """Bind the episode's public character directory to Human memory."""

        self.human.configure_episode_profiles(profiles)

    @staticmethod
    def _camera_gaze_tilt_deg(pose: Any) -> float:
        if not isinstance(pose, dict):
            return 0.0
        return float(pose.get("camera_gaze_tilt_deg") or 0.0)

    @staticmethod
    def _body_posture(pose: Any) -> str:
        if not isinstance(pose, dict):
            raise ValueError("Robot memory requires pose.body_posture.")
        posture = str(pose.get("body_posture") or "").strip()
        if posture not in {"standing", "crouched"}:
            raise ValueError(
                'Robot pose.body_posture must be "standing" or "crouched"; '
                f"got {posture!r}."
            )
        return posture

    def reset(self) -> None:
        self.text.reset()
        self.human.reset()
        self.key.reset()
        self.visual.reset()
        self.grounding.clear()
        self._current_view_groundings.clear()
        self.semantic.reset()
        self._latest_physical_contact = None


__all__ = [
    "EpisodicMemory",
    "GroundingCache",
    "GroundingEntry",
    "ReplaceGroundingEffect",
]
