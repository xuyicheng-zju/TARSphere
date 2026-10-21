#!/usr/bin/env python
"""Plan one native-Dex3 G1 pregrasp and closing approach in isolated cuRoboV2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from tarsphere.infra.manipulation.g1_grasp import (
    G1_GRASP_WORKER_STAGES_FILENAME,
    G1CarryPlanningInfeasible,
    G1GraspPlanner,
    G1GraspPlanningInfeasible,
    carry_planning_infeasible_payload,
    carry_planning_success_payload,
    forbid_g1_scene_aabb_obstacles,
    grasp_planning_infeasible_payload,
    grasp_planning_success_payload,
    support_surface_obstacle_from_payload,
    target_object_obstacle_from_payload,
)
from tarsphere.infra.manipulation.graspgenx import grasp_pose_from_payload


def _stage_recorder(stages_path: Path):
    """Append one flushed JSON line per completed planning stage.

    Progressive flushing is the point: if this process is killed by the
    client's wall-clock timeout, the surviving lines identify exactly which
    boundary (import, construction, warm-up, pregrasp or closing approach) consumed the
    budget.
    """

    def record(stage: str, duration_s: float) -> None:
        line = json.dumps(
            {
                "stage": stage,
                "duration_s": round(float(duration_s), 3),
                "completed_unix_s": round(time.time(), 3),
            },
            ensure_ascii=False,
        )
        with stages_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()

    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    forbid_g1_scene_aabb_obstacles(request.get("collision_cuboids"))
    record_stage = _stage_recorder(args.result.parent / G1_GRASP_WORKER_STAGES_FILENAME)
    planner = G1GraspPlanner(planning_stage_recorder=record_stage)
    operation = request.get("operation")
    if operation == "carry":
        try:
            plan = planner.plan_carry(
                hand=str(request["hand"]),
                current_joint_positions=request["current_joint_positions"],
                current_body_height_command_m=float(
                    request["current_body_height_command_m"]
                ),
                closing_palm_pose_robot=np.asarray(
                    request["closing_palm_pose_robot"],
                    dtype=np.float64,
                ),
            )
        except G1CarryPlanningInfeasible as exc:
            result = carry_planning_infeasible_payload(exc)
        else:
            result = carry_planning_success_payload(plan)
    elif operation == "grasp":
        try:
            plan = planner.plan(
                candidates=tuple(
                    grasp_pose_from_payload(item) for item in request["candidates"]
                ),
                robot_root_from_camera=np.asarray(
                    request["robot_root_from_camera"],
                    dtype=np.float64,
                ),
                current_joint_positions=request["current_joint_positions"],
                current_body_height_command_m=float(
                    request["current_body_height_command_m"]
                ),
                hand=str(request["hand"]),
                grasp_target_object_id=str(request["grasp_target_object_id"]),
                source_support_object_id=str(request["source_support_object_id"]),
                target_obstacle=target_object_obstacle_from_payload(
                    request["target_obstacle"]
                ),
                support_obstacle=support_surface_obstacle_from_payload(
                    request["support_surface_obstacle"]
                ),
                exclude_pose_indices=tuple(
                    int(index) for index in (request.get("exclude_pose_indices") or ())
                ),
            )
        except G1GraspPlanningInfeasible as exc:
            result = grasp_planning_infeasible_payload(
                exc.failures,
                reach_correction=exc.reach_correction,
            )
        else:
            result = grasp_planning_success_payload(plan)
    else:
        raise ValueError(f"Unknown G1 planning operation: {operation!r}.")
    args.result.write_text(
        json.dumps(result, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
