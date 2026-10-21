"""High-level plan state: ingest, persist, expose the VLM's plan.

The model submits a complete plan (``{goal, steps[]}``) at the task start and
whenever task facts change; submitting a new plan replaces the active one and
archives it. Progress is declarative: the model states ``subgoal: "checked"``
when it considers the active step complete, and this planner advances the
internal cursor by exactly one step per declaration. The model never tracks
an index.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlanRecord:
    """One Agent-authored goal and ordered step sequence."""

    policy_step: int
    plan: dict[str, Any]
    goal: str = ""
    steps: list[str] = field(default_factory=list)
    # Harness-owned cursor: number of leading steps the Agent declared
    # complete. Never authored or inferred by the model.
    completed_steps: int = 0
    terminal_confirmation_pending: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_step": self.policy_step,
            "goal": self.goal,
            "steps": list(self.steps),
            "completed_steps": self.completed_steps,
            "terminal_confirmation_pending": (self.terminal_confirmation_pending),
            "raw": dict(self.plan),
        }


class StrategyPlanner:
    """High-level plan state: ingest, persist, expose the VLM's plan."""

    def __init__(self) -> None:
        self._history: list[PlanRecord] = []
        self._active: PlanRecord | None = None

    @property
    def active_plan(self) -> PlanRecord | None:
        return self._active

    @property
    def history(self) -> list[PlanRecord]:
        return list(self._history)

    def ingest(
        self,
        plan_dict: dict[str, Any],
        *,
        policy_step: int,
    ) -> PlanRecord:
        """Accept a ``plan`` {goal, steps[]} submission.

        A first submission activates step 0 automatically. A later
        submission replaces the active plan (cursor resets to 0) and the
        superseded record stays in history so Adaptation scoring can read
        the change against what the Agent was doing before.
        """

        if not isinstance(plan_dict, dict):
            raise TypeError(f"plan must be a dict, got {type(plan_dict).__name__}")

        goal = str(plan_dict.get("goal") or "").strip()
        steps: list[str] = []
        raw_steps = plan_dict.get("steps")
        if isinstance(raw_steps, list):
            steps = [s.strip() for s in raw_steps if isinstance(s, str) and s.strip()]
        if not steps:
            raise ValueError("plan must contain at least one step")

        record = PlanRecord(
            policy_step=policy_step,
            plan=dict(plan_dict),
            goal=goal,
            steps=steps,
            completed_steps=0,
        )
        self._history.append(record)
        self._active = record
        return record

    def declare_subgoal_checked(self, *, policy_step: int) -> PlanRecord:
        """Advance one step, with two-phase confirmation for the final step."""

        if self._active is None:
            raise ValueError(
                "subgoal checked requires an active plan; submit a plan first"
            )
        if self._active.completed_steps >= len(self._active.steps):
            raise ValueError(
                "every step of the active plan is already complete; submit a "
                "revised plan if the task needs more steps"
            )
        final_step_index = len(self._active.steps) - 1
        if self._active.completed_steps < final_step_index:
            self._active.completed_steps += 1
            self._active.terminal_confirmation_pending = False
        elif self._active.terminal_confirmation_pending:
            self._active.completed_steps += 1
            self._active.terminal_confirmation_pending = False
        else:
            # The first declaration on the final step is a reversible request
            # for confirmation. A second declaration in the next accepted
            # policy decision closes the plan; omission cancels the request.
            self._active.terminal_confirmation_pending = True
        return self._active

    def cancel_terminal_confirmation(self) -> None:
        """Keep the final step active when the next decision omits confirmation."""

        if self._active is not None:
            self._active.terminal_confirmation_pending = False

    def active_plan_summary(self) -> str:
        """Compact text summary for the prompt.

        When the plan has been revised, the superseded goal is named alongside
        it. A plan the agent silently replaced reads, one step later, exactly
        like a plan it never had — and Adaptation is scored on whether the
        agent revised in response to a change, which is only legible against
        what it was doing before.
        """
        if self._active is None:
            return ""
        lines = [f"Active plan (emitted at policy step {self._active.policy_step}):"]
        if self._active.goal:
            lines.append(f"- goal: {self._active.goal}")
        cursor = self._active.completed_steps
        for index, step in enumerate(self._active.steps):
            marker = "✓" if index < cursor else ("▶" if index == cursor else "○")
            lines.append(f"- {marker} [{index}] {step}")
        if cursor == len(self._active.steps):
            lines.append("- ✓ all steps complete")
        elif self._active.terminal_confirmation_pending:
            lines.extend(
                (
                    "- ? final-step completion is awaiting your confirmation",
                    "- Choose exactly one response in this decision:",
                    '  1. Confirm completion: output `subgoal`: `"checked"` '
                    "again. The plan becomes complete after this decision's "
                    "action block finishes.",
                    "  2. Cancel the first check: omit `subgoal`. The final step "
                    "remains active and incomplete.",
                    "  3. Replace the plan: submit a new complete `plan`. The "
                    "first check is canceled and progress restarts at the new "
                    "plan's first step.",
                )
            )
        if len(self._history) > 1:
            previous = self._history[-2]
            lines.append(
                f"- revised at policy step {self._active.policy_step}; it "
                f"replaced the plan from policy step {previous.policy_step}"
                + (f" (goal: {previous.goal})" if previous.goal else "")
            )
        return "\n".join(lines)

    def commit_from(self, validated: "StrategyPlanner") -> None:
        """Atomically replace state with an independently validated snapshot."""

        if not isinstance(validated, StrategyPlanner):
            raise TypeError("validated plan state must be a StrategyPlanner.")
        snapshot = deepcopy(validated)
        self._history = snapshot._history
        self._active = snapshot._active

    def reset(self) -> None:
        self._history.clear()
        self._active = None


__all__ = ["PlanRecord", "StrategyPlanner"]
