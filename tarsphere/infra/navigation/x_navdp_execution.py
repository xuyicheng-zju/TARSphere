"""One MLLM navigation call, closed-loop RGB-D execution, one measured result."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from tarsphere.harness.agent import agent_observation_from_runtime
from tarsphere.infra.control.physical_contracts import (
    NAVIGATION_CONTROLLER_REACH_TOLERANCE_M,
    SIMULATION_PHYSICS_HZ,
)
from tarsphere.infra.navigation.x_navdp import XNavDPNoMotionError, prefix_body_delta
from tarsphere.infra.control.robot_contact_monitor import (
    RobotContact,
    RobotContactMonitor,
)


@dataclass(frozen=True)
class XNavDPExecutionSettings:
    """Runtime yield conditions, not task success or collision restrictions."""

    # Counts actual physics ticks, including braking; excludes network/render waits.
    # Bound one policy-visible navigation call to 30 seconds of simulated
    # motion. A longer call can monopolize an episode after a bad PointGoal,
    # especially when the local policy keeps circling in a confined aisle.
    max_physics_steps: int = 7_200
    # Uses the existing controller's metric arrival tolerance, not evaluator GT.
    goal_tolerance_m: float = NAVIGATION_CONTROLLER_REACH_TOLERANCE_M
    # Provisional stall calibration: confined position AND heading over this window.
    stationary_physics_steps: int = 2_400
    stationary_radius_m: float = 0.10
    stationary_yaw_rad: float = 0.10
    # Revisiting already traversed cells without a new best goal distance.
    revisit_physics_steps: int = 4_800
    coverage_cell_m: float = 0.25

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"X-NavDP {name} must be positive and finite.")
            if name.endswith("physics_steps") and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise ValueError(f"X-NavDP {name} must be an integer.")


def _pose(raw: dict[str, Any]) -> tuple[np.ndarray, float]:
    pose = raw["pose"]
    xy = np.asarray(pose["position"][:2], dtype=float)
    yaw = float(pose["yaw_rad"])
    if xy.shape != (2,) or not np.isfinite(xy).all() or not math.isfinite(yaw):
        raise ValueError("Closed-loop navigation requires a finite measured pose.")
    return xy, yaw


def _rotation(yaw: float) -> np.ndarray:
    return np.asarray([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])


def execute_x_navdp_navigation(
    *,
    decision: dict[str, Any],
    observation: dict[str, Any],
    backend: Any,
    camera_views: Any,
    execute_motion: Callable[
        [dict[str, Any]], tuple[dict[str, Any], bool, dict[str, Any]]
    ],
    remaining_episode_physics_steps: int | None = None,
    settings: XNavDPExecutionSettings = XNavDPExecutionSettings(),
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    """Track the original world point; never call an MLLM or consult task GT.

    Each native motion retains the simulator's collision, drop/fall, braking,
    moving-Human and mapping callbacks. Only outer execution is a policy step.
    Inner contacts and controller outcomes remain in the private execution trace.
    """

    command = decision["navigation_command"]
    if command["backend"] not in {"x_navdp", "x_navdp_fallback"}:
        raise ValueError("The X-NavDP executor cannot execute an A* request.")
    start_xy, start_yaw = _pose(observation)
    goal = np.asarray(command["pointgoal_body_xy"], dtype=float)
    if goal.shape != (2,) or not np.isfinite(goal).all():
        raise ValueError("PointGoal must contain two finite body-frame metres.")
    world_goal = start_xy + _rotation(start_yaw) @ goal
    maximum_ticks = settings.max_physics_steps
    if remaining_episode_physics_steps is not None:
        if remaining_episode_physics_steps < 0:
            raise ValueError("Remaining episode physics steps cannot be negative.")
        maximum_ticks = min(maximum_ticks, remaining_episode_physics_steps)
    ticks = 0
    travelled_m = 0.0
    distance_available = True
    history = [(0, start_xy, start_yaw)]
    visited = {tuple(np.floor(start_xy / settings.coverage_cell_m).astype(int))}
    last_new_space_tick = 0
    best_distance_m = float(np.linalg.norm(goal))
    trace: list[dict[str, Any]] = []
    contact_seen = False
    contacts: list[RobotContact] = []
    safety_events: list[dict[str, Any]] = []
    info: dict[str, Any] = {"executed": True, "physical_contact": False}
    done = False
    reason = "physics_step_limit"
    backend_error = ""
    inference_seconds = 0.0
    while ticks < maximum_ticks:
        xy, yaw = _pose(observation)
        remaining = _rotation(yaw).T @ (world_goal - xy)
        if float(np.linalg.norm(remaining)) <= settings.goal_tolerance_m:
            reason = "goal_distance_tolerance"
            break
        try:
            prediction = backend.infer(
                observation=agent_observation_from_runtime(
                    observation,
                    camera_views=camera_views,
                ),
                pointgoal_body_xy=remaining,
            )
            delta = prefix_body_delta(prediction.command_body_xy)
            inference_seconds += float(prediction.inference_seconds)
        except XNavDPNoMotionError as exc:
            reason, backend_error = "trajectory_displacement_below_minimum", str(exc)
            break
        except (OSError, RuntimeError, ValueError) as exc:
            # No silent retry, default motion, or fabricated arrival.
            reason, backend_error = "backend_error", str(exc)
            break
        # Do not replay envelope, event outputs, model_calls or memory internally.
        inner = {
            "action": "move_body",
            "delta": delta,
            "navigation_physics_step_limit": maximum_ticks - ticks,
        }
        observation, done, info = execute_motion(inner)
        elapsed_ticks = int(info["action_physics_steps"])
        if (
            elapsed_ticks < 0
            or (elapsed_ticks == 0 and info.get("executed"))
            or elapsed_ticks > maximum_ticks - ticks
        ):
            raise ValueError("Native NavDP motion violated its physics tick budget.")
        ticks += elapsed_ticks
        xy, yaw = _pose(observation)
        measured_distance = info.get("travelled_distance_m")
        if measured_distance is None:
            distance_available = False
        else:
            travelled_m += float(measured_distance)
        contact_seen |= bool(info.get("physical_contact"))
        contacts.extend(
            RobotContact(**item) for item in info.get("native_contacts", [])
        )
        violation = info.get("safety_violation")
        if violation:
            safety_events.extend(violation["events"])
        trace.append(
            {
                "delta": list(delta),
                "pose_after": dict(observation["pose"]),
                "prediction_body_xy": [
                    list(point) for point in prediction.selected_trajectory_body_xy
                ],
                "remaining_pointgoal_body_xy": (
                    _rotation(yaw).T @ (world_goal - xy)
                ).tolist(),
                "execution": dict(info),
            }
        )
        if done or info.get("dropped_objects"):
            reason = "environment_terminated" if done else "object_dropped"
            break
        if not info.get("executed"):
            reason = (
                "component_outage"
                if info.get("disruption") == "component_outage"
                else "motion_not_executed"
            )
            break
        distance = float(np.linalg.norm(world_goal - xy))
        if distance <= settings.goal_tolerance_m:
            reason = "goal_distance_tolerance"
            break
        samples = np.asarray(info["navigation_motion_samples"], dtype=float)
        if (
            samples.ndim != 2
            or samples.shape[1] != 4
            or len(samples) < 2
            or not np.isfinite(samples).all()
            or samples[0, 0] != 0
            or samples[-1, 0] != elapsed_ticks
            or np.any(np.diff(samples[:, 0]) <= 0)
        ):
            raise ValueError("NavDP requires ordered physical-tick pose samples.")
        for tick, px, py, angle in samples[1:]:
            sample_tick = ticks - elapsed_ticks + int(tick)
            point = np.asarray([px, py])
            history.append((sample_tick, point, angle))
            cell = tuple(np.floor(point / settings.coverage_cell_m).astype(int))
            sample_distance = float(np.linalg.norm(world_goal - point))
            if (
                cell not in visited
                or sample_distance < best_distance_m - settings.stationary_radius_m
            ):
                last_new_space_tick = sample_tick
                best_distance_m = min(best_distance_m, sample_distance)
            visited.add(cell)
        while (
            len(history) > 2
            and history[1][0] <= ticks - settings.stationary_physics_steps
        ):
            history.pop(0)
        first_time, first_xy, first_yaw = history[0]
        confined = all(
            float(np.linalg.norm(point - first_xy)) <= settings.stationary_radius_m
            and abs(
                math.atan2(math.sin(angle - first_yaw), math.cos(angle - first_yaw))
            )
            <= settings.stationary_yaw_rad
            for _, point, angle in history
        )
        if ticks - first_time >= settings.stationary_physics_steps and confined:
            reason = "position_heading_window"
            break
        if ticks - last_new_space_tick >= settings.revisit_physics_steps:
            reason = "coverage_window"
            break

    # Preserve every measured contact window, but express its historical
    # contact positions in the final body frame. Never reuse stale bearings.
    final = dict(info)
    final["executed"] = bool(trace) and bool(info.get("executed"))
    final["physical_contact"] = contact_seen
    if contact_seen:
        if not contacts:
            raise ValueError("Navigation contact flag has no native contact evidence.")
        final["physical_contact_details"] = RobotContactMonitor.public_details(
            contacts,
            robot_position=observation["pose"]["position"],
            robot_yaw_rad=observation["pose"]["yaw_rad"],
        )
        final["native_contacts"] = [contact.as_dict() for contact in contacts]
        final["contact_reference"] = "navigation_contact_positions_in_final_body_frame"
    if safety_events:
        final["safety_violation"] = {
            "type": (info.get("safety_violation") or {}).get(
                "type", "unsafe_human_contact"
            ),
            "events": safety_events,
        }
    final["navigation_trace"] = trace
    final["navigation_skill"] = {
        "stop_reason": reason,
        "physics_steps": ticks,
        "physics_step_limit": maximum_ticks,
        "goal_tolerance_m": settings.goal_tolerance_m,
        "simulation_seconds": ticks / SIMULATION_PHYSICS_HZ,
        "travelled_distance_m": travelled_m if distance_available else None,
        "motion_count": len(trace),
        "inference_seconds": inference_seconds,
        "backend_error": backend_error,
    }
    if reason == "position_heading_window":
        final["navigation_skill"].update(
            {
                "position_heading_window_steps": ticks - history[0][0],
                "position_window_radius_m": max(
                    float(np.linalg.norm(p - history[0][1])) for _, p, _ in history
                ),
                "heading_window_span_rad": max(
                    abs(
                        math.atan2(
                            math.sin(a - history[0][2]), math.cos(a - history[0][2])
                        )
                    )
                    for _, _, a in history
                ),
            }
        )
    elif reason == "coverage_window":
        final["navigation_skill"]["steps_without_new_coverage_or_closer_distance"] = (
            ticks - last_new_space_tick
        )
    final["feedback"] = (
        str(info["feedback"])
        if reason == "component_outage"
        else f"X-NavDP returned control: {reason}."
    )
    return observation, done, final
