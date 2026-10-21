"""Step-aligned text memory and persistent high-level working plan."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from tarsphere.harness.memory.contracts import GraspedHand, TextStepRecord
from tarsphere.harness.planner.strategy import StrategyPlanner
from tarsphere.harness.harness import TEXT_HISTORY_ALL


class TextMemory:
    """Own all textual episode state, indexed by executed action step."""

    def __init__(self, *, actor_id: str = "robot") -> None:
        self.actor_id = actor_id
        self.strategy_planner = StrategyPlanner()
        self._records: list[TextStepRecord] = []
        self._current: TextStepRecord | None = None
        self._grasped_hands: dict[str, GraspedHand] = {}

    def begin_action_step(
        self,
        *,
        action_step: int,
        observation: Any,
        perception: Any,
    ) -> TextStepRecord:
        """Open the unique text record for the current action step."""

        if self._current is not None:
            raise RuntimeError(
                f"action step {self._current.action_step} has no execution "
                "feedback; cannot open another text record"
            )
        self._current = TextStepRecord(
            action_step=int(action_step),
            observation_note=self._observation_note(
                observation=observation,
                perception=perception,
            ),
        )
        return deepcopy(self._current)

    def record_policy_decision(
        self,
        *,
        policy_step: int,
        plan_update: dict[str, Any] | None,
        proposed_actions: list[dict[str, Any]],
    ) -> None:
        current = self._require_current()
        current.policy_step = int(policy_step)
        current.plan_update = deepcopy(plan_update)
        current.proposed_actions = deepcopy(proposed_actions)

    def record_executed_action(
        self,
        *,
        policy_step: int,
        action: dict[str, Any],
    ) -> None:
        current = self._require_current()
        if current.executed_action is not None:
            raise RuntimeError(
                f"action step {current.action_step} already has an action"
            )
        current.policy_step = int(policy_step)
        current.executed_action = deepcopy(action)

    def complete_action_step(self, feedback: dict[str, Any]) -> None:
        current = self._require_current()
        if current.executed_action is None:
            raise RuntimeError(
                f"action step {current.action_step} has feedback but no action"
            )
        clean = {
            key: value
            for key, value in dict(feedback).items()
            if value not in (None, "")
        }
        current.execution_feedback = clean
        self._track_grasp(clean)
        self._records.append(current)
        self._current = None

    def retrieved_records(self, horizon: int | str) -> list[TextStepRecord]:
        """Return completed records selected by Text-History Horizon."""

        if horizon == TEXT_HISTORY_ALL:
            records = self._records
        else:
            count = int(horizon)
            records = self._records[-count:] if count else []
        return deepcopy(records)

    @property
    def current_record(self) -> TextStepRecord | None:
        return deepcopy(self._current)

    @property
    def grasped_hands(self) -> list[GraspedHand]:
        return list(self._grasped_hands.values())

    @staticmethod
    def _observation_note(*, observation: Any, perception: Any) -> str:
        """Collect only runtime-authored public facts for this observation.

        This note is not an RGB caption. It prefers perception-owned facts,
        then the transient public runtime note, and finally labels already
        detected in the same view. The policy's image description is stored
        separately in ``thinking`` after model inference.
        """

        parts: list[str] = []
        facts = [
            str(fact).strip()
            for fact in (getattr(perception, "semantic_facts", None) or [])
            if str(fact).strip()
        ]
        if facts:
            parts.extend(facts)
        else:
            note = str(getattr(observation, "text", "") or "").strip()
            if note:
                parts.append(note)
        labels = sorted(
            {
                str(getattr(item, "label", "") or "").strip()
                for item in (getattr(perception, "detections", None) or [])
                if str(getattr(item, "label", "") or "").strip()
            }
        )
        if labels:
            parts.append("saw " + ", ".join(labels))
        return "; ".join(parts) or "nothing to report"

    def _track_grasp(self, feedback: dict[str, Any]) -> None:
        if not feedback.get("executed"):
            return
        action = str(feedback.get("action") or "")
        if action not in {"grasp_object", "place_object", "move_end_effector"}:
            return
        hand = str(feedback.get("hand") or "right")
        if hand != "right":
            raise ValueError(
                "Manipulation feedback must name the certified right hand."
            )
        if action == "place_object":
            placed = feedback.get("placed")
            if not isinstance(placed, bool):
                raise TypeError(
                    "Successful place_object feedback must report Boolean placed state."
                )
            if placed:
                self._grasped_hands.pop(hand, None)
            return
        if action == "move_end_effector":
            return
        action_step = int(feedback.get("action_step", feedback.get("step", 0)))
        target_label = str(
            feedback.get("target") or feedback.get("object_id") or ""
        ).strip()
        controller_owned = bool(
            feedback.get("preset")
            or feedback.get("reach")
            or feedback.get("grasp_skill")
        )
        if not target_label:
            if controller_owned:
                raise ValueError(
                    "A successful grasp_object feedback record requires "
                    "target or object_id."
                )
            # Handler-completed grasp identity (out of reach, miss, overlay)
            # still names the tool. It is not an attachment.
            return
        self._grasped_hands[hand] = GraspedHand(
            hand=hand,
            target_label=target_label,
            since_action_step=action_step,
            object_id=str(feedback.get("object_id") or "").strip(),
        )

    def _require_current(self) -> TextStepRecord:
        if self._current is None:
            raise RuntimeError("no current action-step text record")
        return self._current

    def reset(self) -> None:
        self.strategy_planner.reset()
        self._records.clear()
        self._current = None
        self._grasped_hands.clear()


__all__ = ["TextMemory"]
