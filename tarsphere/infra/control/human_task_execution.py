"""Apply symbolic NPC work only at its authored physical operation stance."""

from __future__ import annotations

import math
from typing import Callable

from tarsphere.contracts.human_interaction import HUMAN_COMMUNICATION_RADIUS_M
from tarsphere.contracts.human_task import HumanTaskAction, human_task_stance_error
from tarsphere.infra.control.human_body import HumanBodyResult, HumanBodyState


def execute_human_task(
    episode,
    human_id: str,
    action: dict,
    get_state: Callable[[str], HumanBodyState | None],
) -> HumanBodyResult:
    task = HumanTaskAction.model_validate(
        {
            "human": human_id,
            **{key: value for key, value in action.items() if key != "milestone_id"},
        }
    )
    state = get_state(human_id)
    if state is None:
        return HumanBodyResult(
            human_id, task.command, False, "Human has no current body pose"
        )
    anchor = episode.world_knowledge["spatial_anchors"][task.target_anchor]
    error = human_task_stance_error(
        state.position,
        anchor,
        # Animation keeps the last command name after success; status owns
        # completion for both motion backends, not the nonempty command label.
        moving=state.action_status not in {"idle", "succeed"},
        stale=state.pose_stale,
    )
    if not error and task.partner_human_id:
        partner = get_state(task.partner_human_id)
        if partner is None or partner.pose_stale:
            error = "Partner Human has no current body pose"
        elif human_task_stance_error(
            partner.position, partner.position, moving=False, stale=False
        ):
            error = "Partner Human has no finite body pose"
        elif (
            math.dist(state.position[:2], partner.position[:2])
            > HUMAN_COMMUNICATION_RADIUS_M
        ):
            error = "Partner Human is outside interaction distance"
    return HumanBodyResult(
        human_id,
        task.command,
        not error,
        error or "Auxiliary Human task executed at its operation stance",
        state,
    )
