# SPDX-License-Identifier: Apache-2.0
"""Compatibility shims across cuRobo versions for the end2end pipeline.

The motion-planning *goal* API differs between the lab cuRobo fork and public
NVlabs cuRobo (v0.8+):

* lab fork — ``MotionPlanner.plan_grasp`` / ``plan_pose`` accept goals as a
  ``{link_name: Pose}`` dict (the ``Pose`` carries the goalset batch).
* public cuRobo — they require a 5D :class:`GoalToolPose`
  ``[batch, horizon, num_links, num_goalset, 3/4]``.

The result objects (``GraspPlanResult`` / pose-plan result) expose the same
fields in both, so only the *goal input* needs adapting. :func:`grasp_goals`
builds whichever the installed cuRobo expects, detected once via the
``plan_grasp`` signature.
"""

from __future__ import annotations

import inspect
from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def wants_goal_tool_pose() -> bool:
    """True if the installed cuRobo's plan_grasp wants a GoalToolPose."""
    from curobo.motion_planner import MotionPlanner

    ann = str(
        inspect.signature(MotionPlanner.plan_grasp).parameters["grasp_poses"].annotation
    )
    return "GoalToolPose" in ann


def grasp_goals(target_link: str, pos_t: torch.Tensor, quat_t: torch.Tensor):
    """Build the goal object ``plan_grasp`` / ``plan_pose`` expects.

    Args:
        target_link: tool/grasp frame name.
        pos_t: ``[1, G, 3]`` positions for ``G`` grasps/goals on that link.
        quat_t: ``[1, G, 4]`` quaternions (wxyz).

    Returns a 5D ``GoalToolPose`` (public cuRobo) or a ``{link: Pose}`` dict
    (lab fork) — both accepted positionally as the first plan_grasp/plan_pose
    argument.
    """
    if wants_goal_tool_pose():
        from curobo._src.types.tool_pose import GoalToolPose

        g = pos_t.shape[1]
        return GoalToolPose(
            tool_frames=[target_link],
            position=pos_t.reshape(1, 1, 1, g, 3),
            quaternion=quat_t.reshape(1, 1, 1, g, 4),
        )
    from curobo.types import Pose

    return {target_link: Pose(position=pos_t, quaternion=quat_t)}
