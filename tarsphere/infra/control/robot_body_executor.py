"""Robot body execution helpers for simulator-backed environments."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

from tarsphere.harness.actions.router import (
    ActionRouter,
    LocomotionControlProfile,
    commanded_linear_speed_mps,
    duration_physics_steps,
    locomotion_physics_steps,
    locomotion_control_profile,
    metric_body_delta,
    relative_pose_control_physics_steps,
)
from tarsphere.infra.control.g1_balance_control import (
    G1BodyDynamics,
)
from tarsphere.infra.control.physical_contracts import (
    G1_POST_MOTION_SETTLE_S,
    NAVIGATION_CONTROLLER_REACH_TOLERANCE_M,
    SIMULATION_PHYSICS_HZ,
)
from tarsphere.infra.control.relative_pose import (
    BodyPose2D,
    RelativePoseController,
)
from tarsphere.infra.episode.rollout_control import AbortChecker, EpisodeAbortError


_PRECISE_ROTATE_MEDIUM_ERROR_RAD = 0.30
_PRECISE_ROTATE_LARGE_ERROR_RAD = 1.20
_MAXIMUM_POST_MOTION_SETTLE_S = 0.50
_POST_MOTION_SETTLE_ENV = "TARSPHERE_G1_POST_MOTION_SETTLE_S"
# The released gait remains fast for ordinary forward transport.  A retained
# upper-body pose is empirically unstable only when the commanded body
# displacement is backward or lateral-dominant; cap that narrow regime rather
# than slowing every carrying segment.
_G1_CARRY_NONFORWARD_MAX_LINEAR_SPEED_MPS = 0.25


def _relative_pose_maximum_linear_speed_mps(
    env_action: Mapping[str, Any],
    target: tuple[float, float, float],
) -> float:
    maximum = commanded_linear_speed_mps()
    move_by_speed = env_action.get("move_by_speed")
    carrying_upper_body = bool(
        isinstance(move_by_speed, (list, tuple))
        and len(move_by_speed) > 4
        and isinstance(move_by_speed[4], (list, tuple))
        and len(move_by_speed[4]) == 15
        and any(abs(float(value)) > 1e-9 for value in move_by_speed[4])
    )
    forward_m, left_m, _ = target
    nonforward = forward_m <= abs(left_m)
    if carrying_upper_body and nonforward:
        return min(maximum, _G1_CARRY_NONFORWARD_MAX_LINEAR_SPEED_MPS)
    return maximum


def _post_motion_settle_seconds(explicit_seconds: float | None) -> float:
    raw_value: float | str = (
        os.environ.get(
            _POST_MOTION_SETTLE_ENV,
            str(G1_POST_MOTION_SETTLE_S),
        )
        if explicit_seconds is None
        else explicit_seconds
    )
    try:
        seconds = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{_POST_MOTION_SETTLE_ENV} must be a finite number of seconds; "
            f"got {raw_value!r}."
        ) from exc
    if not math.isfinite(seconds):
        raise ValueError(f"{_POST_MOTION_SETTLE_ENV} must be finite; got {seconds!r}.")
    return min(_MAXIMUM_POST_MOTION_SETTLE_S, max(0.0, seconds))


@dataclass
class RobotBodyCommand:
    """Controller-level command plus public execution feedback."""

    action_name: str
    env_action: Dict[str, Any]
    executed: bool
    feedback: str = ""
    # Public magnitudes receive one fixed, bounded feedback window.  The
    # nominal profile remains feed-forward; live root pose closes only the
    # requested local pose and never reads or overrides scene geometry.
    physics_step_budget: Optional[int] = None
    relative_pose_target: Optional[tuple[float, float, float]] = None
    relative_pose_profile: Optional[LocomotionControlProfile] = None


@dataclass
class RobotBodyStepResult:
    """Result of advancing the simulator for one robot decision."""

    observation: Dict[str, Any] = field(default_factory=dict)
    terminated: bool = False
    truncated: bool = False
    physics_steps: int = 0
    #: Kit app updates issued by the Human-activity gate during this window.
    #: Zero whenever no character held an unfinished command, which is the
    #: common case and the reason the gate is affordable.
    kit_updates: int = 0
    post_motion_settle_ticks: int = 0
    balance_control: Dict[str, Any] = field(
        default_factory=lambda: {"intervened": False}
    )
    balance_feedback: str = ""
    terminal_controller_action: Dict[str, Any] = field(default_factory=dict)
    #: Measured outcome of a metric relative-pose command. Empty for actions
    #: that do not request a relative base pose.
    relative_pose_control: Dict[str, Any] = field(default_factory=dict)
    #: Sum of measured XY motion at every physics tick, including braking.
    #: None means no pose sampler was supplied, not zero movement.
    travelled_distance_m: float | None = None
    #: Navigation-only proprioception: [local physics tick, world x, world y, yaw].
    navigation_motion_samples: list[list[float]] = field(default_factory=list)


class RobotBodyExecutor:
    """Translate robot decisions into simulator controller commands.

    This is deliberately simulator-adjacent runtime code, not robot cognition:
    the evaluated agent selects body movement or navigation tools. This
    executor realizes the request through the learned gait. Metric body poses
    use bounded proprioceptive feedback; Native collision response remains
    owned by PhysX and can prevent completion.
    """

    def __init__(
        self,
        *,
        router: Optional[ActionRouter] = None,
        stationary_physics_steps: int = 8,
        precise_approach_physics_steps: int = 60,
        precise_rotate_physics_steps: int = 60,
        precise_rotate_medium_physics_steps: int = 120,
        precise_rotate_large_physics_steps: int = 180,
        carry_transition_physics_steps: int = 180,
        body_height_transition_physics_steps: int = 180,
        post_motion_settle_seconds: float | None = None,
    ):
        self.router = router or ActionRouter()
        self.stationary_physics_steps = max(1, int(stationary_physics_steps))
        self.precise_approach_physics_steps = max(
            1, int(precise_approach_physics_steps)
        )
        self.precise_rotate_physics_steps = max(1, int(precise_rotate_physics_steps))
        self.precise_rotate_medium_physics_steps = max(
            self.precise_rotate_physics_steps,
            int(precise_rotate_medium_physics_steps),
        )
        self.precise_rotate_large_physics_steps = max(
            self.precise_rotate_medium_physics_steps,
            int(precise_rotate_large_physics_steps),
        )
        self.carry_transition_physics_steps = max(
            1, int(carry_transition_physics_steps)
        )
        self.body_height_transition_physics_steps = max(
            1, int(body_height_transition_physics_steps)
        )
        self.post_motion_settle_seconds = _post_motion_settle_seconds(
            post_motion_settle_seconds
        )
        self.post_motion_settle_ticks = max(
            0,
            int(round(self.post_motion_settle_seconds * SIMULATION_PHYSICS_HZ)),
        )

    def command_for_decision(
        self,
        decision: Dict[str, Any],
    ) -> RobotBodyCommand:
        """Convert a runner decision into a controller-level environment action.

        The returned command preserves the agent's requested magnitude.
        Native collision response occurs while physics advances; no privileged
        occupancy pre-scan rejects or shortens the leg. Egocentric depth remains
        Agent perception and never authorizes motion.
        """
        action = decision.get("action")
        if not action:
            raise ValueError(f"Robot decision missing action: {decision!r}")
        if str(action) == "navigate":
            raise ValueError(
                "navigate is one VLM skill; DualVLN inner steps must be "
                "applied as move_body or wait, not as a navigate "
                "controller_action."
            )
        step_params = {key: value for key, value in decision.items() if key != "action"}
        relative_pose_target = metric_body_delta(
            str(action),
            params=step_params,
        )
        relative_pose_profile = locomotion_control_profile(
            str(action),
            params=step_params,
        )
        if "controller_action" in decision:
            env_action = dict(decision.get("controller_action") or {})
        else:
            env_action, _ = self.router.route(action, params=step_params)
        executed = bool(env_action) or action in {
            "wait",
            "ask",
            "speak",
            "report_done",
            "handover_object",
        }
        feedback = ""

        if not env_action:
            env_action = self.zero_velocity_action()
        if not executed:
            feedback = feedback or (
                f'Action "{action}" is unavailable without a perception-grounded '
                "skill executor."
            )
        physics_step_budget = (
            relative_pose_control_physics_steps(
                str(action),
                params=step_params,
                post_motion_settle_seconds=self.post_motion_settle_seconds,
            )
            if executed and relative_pose_target is not None
            else locomotion_physics_steps(str(action), params=step_params)
            if executed
            else self.stationary_physics_steps
        )
        if physics_step_budget is None and str(action) == "precise_rotate":
            metadata = decision.get("controller_metadata") or {}
            angular_error = metadata.get("angular_error_rad")
            if angular_error is not None:
                error_magnitude = abs(float(angular_error))
                if error_magnitude > _PRECISE_ROTATE_LARGE_ERROR_RAD:
                    physics_step_budget = self.precise_rotate_large_physics_steps
                elif error_magnitude > _PRECISE_ROTATE_MEDIUM_ERROR_RAD:
                    physics_step_budget = self.precise_rotate_medium_physics_steps
                else:
                    physics_step_budget = self.precise_rotate_physics_steps
        if physics_step_budget is None and str(action) == "coordination_wait":
            physics_step_budget = duration_physics_steps(1.0)
        if physics_step_budget is None and str(action) == "follow_path":
            metadata = decision.get("controller_metadata") or {}
            skill_action = metadata.get("skill_action") or {}
            raw_points = skill_action.get("path_points") or []
            if not raw_points:
                routed_path = env_action.get("move_along_path") or []
                if isinstance(routed_path, (list, tuple)) and len(routed_path) == 1:
                    raw_points = routed_path[0]
            points = [
                (float(point[0]), float(point[1]))
                for point in raw_points
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ]
            if len(points) < 2:
                raise ValueError(
                    "follow_path requires at least two path points to derive "
                    "its physics window."
                )
            path_length_m = sum(
                math.dist(start, end) for start, end in zip(points, points[1:])
            )
            physics_step_budget = duration_physics_steps(
                path_length_m / commanded_linear_speed_mps() + 1.0
            )
        return RobotBodyCommand(
            action_name=str(action),
            env_action=env_action,
            executed=executed,
            feedback=feedback,
            physics_step_budget=physics_step_budget,
            relative_pose_target=relative_pose_target,
            relative_pose_profile=relative_pose_profile,
        )

    def step_simulation(
        self,
        *,
        env: Any,
        action_name: str,
        env_action: Dict[str, Any],
        physics_step_budget: int | None = None,
        step_human_bodies: Callable[[float], None],
        pump_kit_updates: Callable[..., None],
        pump_motion_window: Callable[..., int] | None = None,
        global_tick: int = 0,
        sample_body_dynamics: Callable[[], G1BodyDynamics] | None = None,
        begin_decision_window: Callable[[int], None] | None = None,
        prepare_final_frame: Callable[[], None] | None = None,
        end_decision_window: Callable[[], None] | None = None,
        diagnose_physics_tick: Callable[[str, int], None] | None = None,
        observe_motion_tick: Callable[[str, int], None] | None = None,
        finish_when: Callable[[], bool] | None = None,
        control_for_tick: Callable[[int], Dict[str, Any]] | None = None,
        relative_pose_target: tuple[float, float, float] | None = None,
        relative_pose_profile: LocomotionControlProfile | None = None,
        sample_body_pose: Callable[[], BodyPose2D] | None = None,
        physics_step_limit: int | None = None,
        abort_requested: AbortChecker | None = None,
        dt: float = 1.0 / SIMULATION_PHYSICS_HZ,
    ) -> RobotBodyStepResult:
        """Advance physics with the supplied controller action."""
        result = RobotBodyStepResult()
        previous_pose = sample_body_pose() if sample_body_pose is not None else None
        if previous_pose is not None:
            result.travelled_distance_m = 0.0
            if physics_step_limit is not None:
                result.navigation_motion_samples.append(
                    [
                        0,
                        previous_pose.x_m,
                        previous_pose.y_m,
                        previous_pose.yaw_rad,
                    ]
                )
        if physics_step_limit is not None and physics_step_limit < 1:
            raise ValueError("physics_step_limit must be positive.")
        relative_pose_controller: RelativePoseController | None = None
        finish_includes_post_motion_settle = False
        if relative_pose_target is not None:
            if relative_pose_profile is None or sample_body_pose is None:
                raise ValueError(
                    "Relative-pose execution requires its calibrated profile "
                    "and a live body-pose sampler."
                )
            if control_for_tick is not None or finish_when is not None:
                raise ValueError(
                    "Relative-pose execution owns its per-tick control and "
                    "physical completion predicate."
                )
            relative_pose_controller = RelativePoseController(
                target_delta=relative_pose_target,
                profile=relative_pose_profile,
                command_template=env_action,
                sample_pose=sample_body_pose,
                stable_zero_ticks=max(1, self.post_motion_settle_ticks),
                maximum_linear_speed_mps=(
                    _relative_pose_maximum_linear_speed_mps(
                        env_action,
                        relative_pose_target,
                    )
                ),
            )
            control_for_tick = relative_pose_controller.action_for_tick
            finish_when = relative_pose_controller.observe_after_tick
            # Completion already proves the pose survived the entire zero-speed
            # hold. A blocked/timed-out command still receives the regular
            # decision-boundary hold below.
            finish_includes_post_motion_settle = True
        # Locomotion controllers need the authored long horizon to traverse a
        # useful distance per model decision. Communication, waiting, and hand
        # tools must refresh cognition/control at the teleop-validated cadence;
        # holding G1's zero-speed RL command for the same 240-step
        # navigation horizon makes the added-hand morphology tip over before
        # the next decision arrives.
        locomotion_actions = {
            "move_body",
            "move_forward",
            "move_backward",
            "move_left",
            "move_right",
            "turn_left",
            "turn_right",
            "rotate_to",
            "follow_path",
            # Golden coordination waits advance a full physics second so a
            # routed Human can clear a shared narrow passage while G1 holds.
            "coordination_wait",
            # Arm-pose transitions need the authored physical horizon, unlike
            # language/wait/release decisions which only need an 8-step hold.
            "prepare_grasp",
            "prepare_place",
            "prepare_carry",
        }
        settled_motion_actions = {
            "move_body",
            "move_forward",
            "move_backward",
            "move_left",
            "move_right",
            "turn_left",
            "turn_right",
            "precise_rotate",
            "precise_approach",
            "rotate_to",
            "follow_path",
        }
        if control_for_tick is not None:
            if physics_step_budget is None or int(physics_step_budget) < 1:
                raise ValueError(
                    "A per-tick control trajectory requires a positive exact "
                    "physics_step_budget."
                )
            step_budget = int(physics_step_budget)
        elif str(action_name) == "precise_approach":
            # A manipulation approach is a local pose servo, not global
            # navigation.  The official G1 MoveToPoint controller can cross
            # the authored stance even when its controller observation never
            # reaches the stricter 5 cm finish threshold. Preserve the
            # teleop-validated 60-tick observation
            # cadence so the oracle can recompute the remaining XY delta.
            step_budget = (
                self.precise_approach_physics_steps
                if physics_step_budget is None
                else max(1, int(physics_step_budget))
            )
        elif str(action_name) == "precise_rotate":
            # Keep one rotation backend while adapting only its observation
            # horizon.  Large errors receive a longer bang-bang window; the
            # final <=0.30 rad returns to the validated 60-tick cadence,
            # preserving manipulation-facing angular granularity. A 0.30–0.50
            # rad turn needs the 120-tick horizon for the official gait to
            # leave its post-path settling state instead of oscillating.
            rotation_budget = (
                self.precise_rotate_physics_steps
                if physics_step_budget is None
                else max(1, int(physics_step_budget))
            )
            step_budget = rotation_budget
        elif str(action_name) == "prepare_carry":
            # Carry preparation is already split into four monotone upper-body
            # targets (75%, 50%, 25%, zero). A short hold lets the lower-body
            # policy fight a stale arm moment and walk the attached base away
            # from the support. Each target is sampled for
            # carry_transition_physics_steps (180 ticks = 0.75 s at 240 Hz);
            # the four stages together last 3.0 s.
            step_budget = self.carry_transition_physics_steps
        elif str(action_name) in {"crouch", "stand_up"}:
            # Body height is an official locomotion-policy command.  The
            # certified transition settles within 180 physics ticks; a longer
            # decision window adds latency without changing the target.
            step_budget = self.body_height_transition_physics_steps
        elif str(action_name) == "place_object":
            # Placement is a closed-loop arm transition while the pelvis and
            # held object remain fixed together.  A high worktop can require
            # more than the one-second grasp horizon. Its physical joint
            # predicate may end this action-specific window earlier.
            step_budget = (
                duration_physics_steps(3.0)
                if physics_step_budget is None
                else max(1, int(physics_step_budget))
            )
        elif str(action_name) in {
            "grasp_object",
            "move_end_effector",
            "set_g1_gripper_preset",
            "prepare_grasp",
            "prepare_place",
        }:
            # Arm transitions need physical settling, but the global long
            # locomotion horizon must not hold an upper-body pose for seconds.
            # Public grasp actions, internal hand-preset stages, and authored
            # prepare poses share this bounded articulation horizon. A
            # carrying move_body still uses the magnitude-derived locomotion
            # window below even when the same 15-D hold is attached. Release
            # and carry preparation are handled above because their physical
            # horizons differ.
            step_budget = (
                duration_physics_steps(1.0)
                if physics_step_budget is None
                else max(1, int(physics_step_budget))
            )
        elif str(action_name) == "wait" and physics_step_budget is not None:
            # Yielding is the one stationary decision whose entire content is
            # duration: hold zero velocity for exactly the seconds the agent
            # asked for, through the same channel ``coordination_wait`` already
            # uses for the authored horizon.
            step_budget = max(1, int(physics_step_budget))
        elif str(action_name) in locomotion_actions:
            # Public magnitudes and internal path geometry determine their
            # exact command duration. Never clip a valid motion against an
            # unrelated generic per-action window.
            if physics_step_budget is None:
                raise ValueError(
                    f"{action_name} is missing its derived physics_step_budget"
                )
            step_budget = max(1, int(physics_step_budget))
        else:
            step_budget = self.stationary_physics_steps
        post_motion_settle_ticks = (
            self.post_motion_settle_ticks
            if str(action_name) in settled_motion_actions
            else 0
        )
        total_step_budget = step_budget + post_motion_settle_ticks + 1
        if physics_step_limit is not None:
            # Reserve the end of this invocation for the native zero-speed
            # hold. This limits execution time, never sets or corrects pose.
            total_step_budget = min(total_step_budget, physics_step_limit)
            step_budget = min(
                step_budget,
                max(
                    0,
                    total_step_budget - max(1, post_motion_settle_ticks),
                ),
            )
        if begin_decision_window is not None:
            begin_decision_window(step_budget + post_motion_settle_ticks)
        command_finished = False
        final_zero_tick_pending = False
        last_controller_action: Dict[str, Any] | None = None
        command_ticks = 0
        settle_ticks = 0
        exhausted_loop = True
        try:
            for step_index in range(total_step_budget):
                if abort_requested is not None and abort_requested():
                    raise EpisodeAbortError(
                        "Episode rollout aborted during the active physics window."
                    )
                limit_hold = (
                    physics_step_limit is not None
                    and step_index
                    >= total_step_budget - max(1, post_motion_settle_ticks)
                )
                if limit_hold:
                    command_finished = True
                if command_finished:
                    controller_action = self._completion_hold_action(
                        last_controller_action or env_action
                    )
                elif command_ticks < step_budget:
                    controller_action = (
                        control_for_tick(command_ticks)
                        if control_for_tick is not None
                        else env_action
                    )
                    if not isinstance(controller_action, dict):
                        raise TypeError(
                            "control_for_tick must return one controller-action "
                            "mapping per physics tick."
                        )
                else:
                    command_finished = True
                    controller_action = self.zero_velocity_action()
                last_controller_action = controller_action
                final_settle_tick = bool(
                    command_finished
                    and post_motion_settle_ticks > 0
                    and settle_ticks == post_motion_settle_ticks - 1
                )
                terminal_zero_tick = bool(
                    command_finished
                    and post_motion_settle_ticks == 0
                    and final_zero_tick_pending
                )
                final_command_tick = bool(
                    not command_finished
                    and command_ticks == step_budget - 1
                    and post_motion_settle_ticks == 0
                )
                if prepare_final_frame is not None and (
                    final_settle_tick or final_command_tick or terminal_zero_tick
                ):
                    prepare_final_frame()
                if diagnose_physics_tick is not None:
                    diagnose_physics_tick("before_human", step_index)
                step_human_bodies(dt)
                # ``step_human_bodies`` only reads character state back; the
                # animation clock itself is driven by Kit app updates, which
                # ``env.step(render=False)`` never issues. Pump at the shared
                # animation cadence, including ambient LookAround. The tick
                # hooks keep native video renders at zero animation delta so
                # rendering does not count the same physical interval twice.
                if pump_motion_window is not None:
                    result.kit_updates += int(
                        pump_motion_window(physics_tick=global_tick + step_index)
                    )
                if diagnose_physics_tick is not None:
                    diagnose_physics_tick("after_human", step_index)
                if diagnose_physics_tick is not None:
                    diagnose_physics_tick("before", step_index)
                try:
                    observation, _, terminated, truncated, _ = env.step(
                        action=controller_action
                    )
                except Exception as step_exc:
                    if diagnose_physics_tick is not None:
                        diagnose_physics_tick("error", step_index)
                    raise RuntimeError(
                        "Isaac failed while advancing one commanded physics "
                        f"tick for {action_name!r}."
                    ) from step_exc
                # env.step returned: this native tick happened even if a
                # subsequent camera/mapping/readback hook raises. Preserve
                # that interval independently of action completion.
                result.observation = observation
                result.terminated = bool(terminated)
                result.truncated = bool(truncated)
                result.physics_steps += 1
                if diagnose_physics_tick is not None:
                    diagnose_physics_tick("after", step_index)
                if observe_motion_tick is not None:
                    observe_motion_tick(
                        str(action_name),
                        global_tick + step_index + 1,
                    )
                if previous_pose is not None:
                    measured_pose = sample_body_pose()
                    result.travelled_distance_m += math.hypot(
                        measured_pose.x_m - previous_pose.x_m,
                        measured_pose.y_m - previous_pose.y_m,
                    )
                    previous_pose = measured_pose
                    if physics_step_limit is not None:
                        result.navigation_motion_samples.append(
                            [
                                result.physics_steps,
                                measured_pose.x_m,
                                measured_pose.y_m,
                                measured_pose.yaw_rad,
                            ]
                        )
                if terminal_zero_tick:
                    final_zero_tick_pending = False
                    exhausted_loop = False
                    break
                if not command_finished:
                    command_ticks += 1
                else:
                    settle_ticks += 1
                if (
                    physics_step_limit is not None
                    and result.physics_steps >= total_step_budget
                ):
                    exhausted_loop = False
                    break
                if result.terminated or result.truncated:
                    exhausted_loop = False
                    break
                if (
                    not command_finished
                    and finish_when is not None
                    and bool(finish_when())
                ):
                    # The physical predicate (for example, a hand entering the
                    # attachment gate) is now authoritative.  Advance one
                    # terminal rendered tick, matching controller completion,
                    # instead of holding the command for the full horizon and
                    # allowing the body to drift back out of the envelope.
                    command_finished = True
                    if finish_includes_post_motion_settle:
                        settle_ticks = post_motion_settle_ticks
                        final_zero_tick_pending = False
                    else:
                        final_zero_tick_pending = post_motion_settle_ticks == 0
                if (
                    step_index < step_budget - 1
                    and not command_finished
                    and str(action_name)
                    in {
                        "follow_path",
                        "precise_approach",
                    }
                    and self._controller_finished(observation, env_action)
                ):
                    command_finished = True
                    final_zero_tick_pending = post_motion_settle_ticks == 0
                if command_ticks >= step_budget:
                    command_finished = True
                if (
                    command_finished
                    and not final_zero_tick_pending
                    and settle_ticks >= post_motion_settle_ticks
                ):
                    exhausted_loop = False
                    break
            if exhausted_loop:
                raise RuntimeError(
                    "Robot body physics loop exhausted its bounded action window: "
                    f"action={action_name!r}, command_ticks={command_ticks}, "
                    f"required_command_ticks={step_budget}."
                )
        except Exception as motion_error:
            if diagnose_physics_tick is not None:
                # Completed ticks, not the attempted next step index. This
                # does not infer whether a native env.step that raised had
                # partially advanced; only successful returns are counted.
                try:
                    diagnose_physics_tick("interrupted", result.physics_steps)
                except Exception as accounting_error:
                    motion_error.add_note(
                        "Interrupted physics-clock bookkeeping failed: "
                        f"{type(accounting_error).__name__}: {accounting_error}"
                    )
            raise
        finally:
            if end_decision_window is not None:
                end_decision_window()
        result.post_motion_settle_ticks = settle_ticks
        result.terminal_controller_action = dict(last_controller_action or {})
        if relative_pose_controller is not None:
            result.relative_pose_control = relative_pose_controller.report(
                timed_out=bool(
                    not result.terminated
                    and not result.truncated
                    and not relative_pose_controller.completed
                    and command_ticks >= step_budget
                ),
            )
        return result

    @staticmethod
    def observation_yaw(observation: Mapping[str, Any]) -> float:
        """Read a finite world yaw from a controller observation quaternion."""

        orientation = observation.get("orientation")
        if hasattr(orientation, "cpu"):
            orientation = orientation.cpu().numpy()
        if orientation is None or len(orientation) != 4:
            raise RuntimeError(
                "Measured turn completion requires live robot orientation in "
                "every controller observation."
            )
        w, x, y, z = (float(value) for value in orientation)
        if not all(math.isfinite(value) for value in (w, x, y, z)):
            raise RuntimeError(
                "Measured turn completion received an invalid robot "
                f"orientation: {(w, x, y, z)!r}."
            )
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    @staticmethod
    def _controller_finished(
        observation: Dict[str, Any],
        env_action: Dict[str, Any],
    ) -> bool:
        """Read completion for the controller driven by this decision."""
        if not isinstance(observation, dict) or not isinstance(env_action, dict):
            return False
        controllers = observation.get("controllers")
        if not isinstance(controllers, dict):
            return False
        for controller_name in env_action:
            state = controllers.get(str(controller_name))
            if isinstance(state, dict) and bool(state.get("finished", False)):
                return True
            if str(controller_name) != "move_along_path":
                continue
            action = env_action.get(controller_name)
            position = observation.get("position")
            try:
                terminal = action[0][-1]
                if hasattr(position, "cpu"):
                    position = position.cpu().numpy()
                distance = math.hypot(
                    float(position[0]) - float(terminal[0]),
                    float(position[1]) - float(terminal[1]),
                )
            except (IndexError, KeyError, TypeError, ValueError):
                continue
            if distance <= NAVIGATION_CONTROLLER_REACH_TOLERANCE_M:
                return True
        return False

    @staticmethod
    def zero_velocity_action() -> Dict[str, Any]:
        """Controller payload that holds the G1 base stationary.

        This is runtime control state used for stationary or unavailable
        commands. It is not an agent-callable action.
        """
        return {"move_by_speed": [0.0, 0.0, 0.0]}

    @staticmethod
    def _completion_hold_action(
        env_action: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Stop the base without discarding a reached upper-body pose.

        A physical finish predicate is followed by one rendered boundary tick.
        Replacing a 19-field G1 command with a three-field zero-speed command
        on that tick resets the arms toward locomotion neutral and invalidates
        the pose that just satisfied the predicate.  Preserve the body-height,
        arm, and hand targets while zeroing only base velocity.  Ordinary
        locomotion commands still receive the plain zero-speed action.
        """

        move_by_speed = env_action.get("move_by_speed")
        if not (
            isinstance(move_by_speed, (list, tuple))
            and len(move_by_speed) > 4
            and isinstance(move_by_speed[4], (list, tuple))
            and len(move_by_speed[4]) == 15
        ):
            # Direct joint control never activated locomotion. Preserve only
            # those arm/hand targets for the boundary tick; injecting a
            # move_by_speed command would violate its arm-only contract.
            if any(str(name).endswith("_joint_controller") for name in env_action):
                return dict(env_action)
            return RobotBodyExecutor.zero_velocity_action()
        held = dict(env_action)
        held_move = list(move_by_speed)
        held_move[:3] = [0.0, 0.0, 0.0]
        held["move_by_speed"] = held_move
        return held
