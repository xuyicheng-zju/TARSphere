"""Small deterministic runtime backend for testing the autonomous episode loop.

The mock backend keeps ground-truth positions internally but exposes only
local, egocentric observations. It must not resolve semantic action targets
to hidden world coordinates.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tarsphere import geometry as G
from tarsphere.infra.control.human_body import HumanBodyResult
from tarsphere.contracts.human_interaction import (
    HUMAN_COMMUNICATION_RADIUS_M,
)
from tarsphere.bench.schema import DiagnosticCheck
from tarsphere.infra.control.physical_contracts import (
    CAMERA_GAZE_TILT_LIMIT_DEG,
    G1_END_EFFECTOR_DELTA_MAX_ROTATION_RAD,
    G1_END_EFFECTOR_DELTA_MAX_TRANSLATION_M,
    require_standing_g1_locomotion,
)
from tarsphere.harness.harness import AgentHarnessConfig
from tarsphere.infra.control.internnav import (
    DUALVLN_MAX_INNER_STEPS,
    DUALVLN_RGB_HEIGHT,
    DUALVLN_RGB_WIDTH,
    DualVLNObservation,
    internav_backend_available,
    build_internnav_backend,
    official_norm_depth,
)


class MockObject:
    def __init__(self, name: str, position: List[float], category: str = "object"):
        self.name = name
        self.position = np.asarray(position, dtype=float)
        self.category = category
        self.held_by: Optional[str] = None


class MockEnv:
    """Egocentric test environment with low-level locomotion."""

    def __init__(
        self,
        scene_setup: Optional[Dict[str, Any]] = None,
        *,
        agent_harness_config: AgentHarnessConfig | None = None,
    ):
        setup = scene_setup or {}
        from tarsphere.embodiment.robot_profiles import (
            get_robot_profile,
            DEFAULT_ROBOT_TYPE,
        )

        default_z = get_robot_profile(DEFAULT_ROBOT_TYPE).standing_z
        self._robot_start = np.asarray(
            setup.get("robot_start", [0.0, -1.0, default_z]), dtype=float
        )
        self._human_start = np.asarray(
            setup.get("human_start", [2.0, 0.0, 0.9]), dtype=float
        )
        self._object_setup = setup.get("objects", {})
        self._human_specs = list(
            setup.get("humans")
            or [
                {
                    "id": "human",
                    "name": "human",
                }
            ]
        )
        self._spatial_clearance_plans = [
            dict(plan) for plan in setup.get("spatial_clearance_plans", ())
        ]
        self.spatial_clearance_plan_calls: list[dict[str, Any]] = []
        self.visibility_radius = float(setup.get("visibility_radius", 2.5))
        self.max_steps = int(setup.get("max_steps", 80))
        # Camera config for the synthetic RGBD frames observe() emits so the
        # private trace has real pose/depth/camera signals on the mock path.
        self._camera_fov_deg = float(setup.get("camera_fov_deg", 90.0))
        self._camera_width = int(setup.get("camera_width", 320))
        self._camera_height = int(setup.get("camera_height", 240))
        self._camera_height_m = float(setup.get("camera_height_m", 1.0))
        # Fixed mount calibration, mirroring ``RobotProfile.camera_mount_pitch_deg``.
        # The agent's own gaze tilt is separate; see ``camera_gaze_tilt_deg``.
        self._camera_mount_pitch_deg = float(setup.get("camera_pitch_deg", 0.0))
        self.agent_harness_config = agent_harness_config or AgentHarnessConfig()
        self.reset()

    def configure_agent_harness(self, config: AgentHarnessConfig) -> None:
        if not isinstance(config, AgentHarnessConfig):
            raise TypeError("MockEnv requires an AgentHarnessConfig")
        self.agent_harness_config = config

    def execution_measurements(self) -> Dict[str, Any]:
        """Mock decisions are not native physics or contact measurements."""
        return {
            "status": "unavailable",
            "source": "mock",
            "missing_evidence": ["mock_has_no_native_physics_measurements"],
            "physics_steps": None,
            "path_length_m": None,
            "execution_safety": None,
        }

    def reset(self) -> Dict[str, Any]:
        self.robot_pos = self._robot_start.copy()
        self._human_positions = self._positions_from_human_specs()
        self._primary_human_id = next(iter(self._human_positions))
        self.human_pos = self._human_positions[self._primary_human_id]
        self.robot_yaw = 0.0
        # Commanded ego-camera gaze tilt, degrees, negative down, absolute
        # against the mount calibration.  A new episode starts at the mounted
        # zero; see IsaacAutonomousEnv.set_camera_gaze_tilt_deg.
        self._camera_gaze_tilt_deg = 0.0
        self._body_posture = "standing"
        self._right_gripper_closure = 0.0
        self.step_count = 0
        self.last_action = ""
        self.last_utterance: Optional[Dict[str, Any]] = None
        self._interaction_feedback = ""
        self.robot_holding: Optional[str] = None
        self._accepted_handovers: Dict[str, str] = {}
        self._last_handover_claim: Optional[Dict[str, Any]] = None
        self.objects: Dict[str, MockObject] = {
            name: MockObject(name, spec["position"], spec.get("category", "object"))
            for name, spec in self._object_setup.items()
        }
        return self.observe()

    def observe(self) -> Dict[str, Any]:
        parts = [f"Decision step {self.step_count}."]
        human_relation = self._visible_relation(self.human_pos)
        if human_relation:
            parts.append(f"A human is visible {human_relation}.")

        for obj in self.objects.values():
            relation = self._visible_relation(obj.position)
            if relation:
                holder = f", held by {obj.held_by}" if obj.held_by else ""
                parts.append(
                    f"A {obj.category} identified as {obj.name} is visible {relation}{holder}."
                )

        if self.robot_holding:
            parts.append(f"You are holding {self.robot_holding}.")
        if self._interaction_feedback:
            parts.append(self._interaction_feedback)
            self._interaction_feedback = ""

        pose = self._build_pose_payload()
        camera_params = self._build_camera_params()
        depth = self._synthesise_depth()
        rgb = self._synthesise_rgb()
        return {
            "text": " ".join(parts),
            "step": self.step_count,
            # Pose / camera / depth support geometry and perception tests on
            # the mock path. RGB is a non-black solid frame; depth is a constant
            # field keyed off the nearest visible obstacle.
            "pose": pose,
            "camera_params": camera_params,
            "depth": depth,
            "rgb": rgb,
            # Normalized agent-facing cameras. The flat ego fields remain for
            # evaluator geometry, but RobotAgent reads only this mapping.
            "views": {
                "ego": {
                    "rgb": rgb,
                    "depth": depth,
                    "camera_params": camera_params,
                },
                **self._synthetic_wrist_views(depth),
            },
        }

    def camera_gaze_tilt_deg(self) -> float:
        """Return the agent's commanded ego-camera gaze tilt, negative down."""
        return float(self._camera_gaze_tilt_deg)

    def set_camera_gaze_tilt_deg(self, pitch_deg: float) -> None:
        """Aim the ego camera; see IsaacAutonomousEnv.set_camera_gaze_tilt_deg.

        The range is enforced here for the same reason: the ``tilt_camera`` tool
        rejects an out-of-range request, so an angle past the limit arriving at a
        backend means that boundary was bypassed.
        """
        requested = float(pitch_deg)
        limit = float(CAMERA_GAZE_TILT_LIMIT_DEG)
        if not math.isfinite(requested) or abs(requested) > limit:
            raise ValueError(
                f"Camera gaze tilt {pitch_deg!r} is outside the contracted "
                f"+/-{limit:.1f} deg range."
            )
        self._camera_gaze_tilt_deg = requested

    def _build_pose_payload(self) -> Dict[str, Any]:
        """Pose payload for private trace geometry.

        ``camera_pitch_deg`` is the camera's live total pitch — mount calibration
        plus commanded gaze tilt — because that is the angle a back-projection
        has to use.  ``camera_gaze_tilt_deg`` is the agent's own share of it.
        """
        return {
            "position": self.robot_pos.tolist(),
            "yaw_rad": float(math.radians(self.robot_yaw)),
            "camera_height_m": self._camera_height_m,
            "camera_pitch_deg": (
                self._camera_mount_pitch_deg + self.camera_gaze_tilt_deg()
            ),
            "camera_gaze_tilt_deg": self.camera_gaze_tilt_deg(),
            "body_posture": self._body_posture,
        }

    def _build_camera_params(self) -> Dict[str, Any]:
        pose = self._build_pose_payload()
        transform = G.assemble_camera_pose(
            G.RobotBasePose(
                position=tuple(pose["position"]),
                yaw_rad=float(pose["yaw_rad"]),
                camera_height_m=float(pose["camera_height_m"]),
                camera_pitch_deg=float(pose["camera_pitch_deg"]),
            )
        )
        capture_id = int(self.step_count) + 1
        return {
            "fov_deg": self._camera_fov_deg,
            "width": self._camera_width,
            "height": self._camera_height,
            "source": "mock_robot_ego",
            "world_T_cam": transform.tolist(),
            "capture_id": capture_id,
            "depth_capture_id": capture_id,
            "physics_step": int(self.step_count),
        }

    def _synthesise_depth(self) -> np.ndarray:
        """A synthetic depth frame keyed off the nearest visible object/human.

        The frame is a constant depth image whose value is the distance to the
        nearest obstacle straight ahead; good enough for the evaluator's
        geometry path to exercise back-projection end-to-end on the mock
        path. NOT a real rendered depth map.
        """
        h, w = self._camera_height, self._camera_width
        # Distance to the nearest object roughly in front; default 5 m (clear).
        depth_val = 5.0
        for obj in self.objects.values():
            d = float(np.linalg.norm(obj.position[:2] - self.robot_pos[:2]))
            if d <= self.visibility_radius and d < depth_val:
                depth_val = d
        d_human = float(np.linalg.norm(self.human_pos[:2] - self.robot_pos[:2]))
        if d_human <= self.visibility_radius and d_human < depth_val:
            depth_val = d_human
        return np.full((h, w), float(depth_val), dtype=np.float32)

    def _synthesise_rgb(self) -> np.ndarray:
        """A synthetic solid-colour RGB frame (non-black so visibility checks pass)."""
        h, w = self._camera_height, self._camera_width
        return np.full((h, w, 3), 128, dtype=np.uint8)

    def _synthetic_wrist_views(
        self,
        ego_depth: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """Return distinct mock wrist frames for protocol tests.

        These are deliberately not aliases of the ego arrays.  They test
        multi-view ordering and validation only; ``source`` makes their
        synthetic status explicit so they cannot be mistaken for calibrated
        Isaac wrist cameras.
        """

        height, width = self._camera_height, self._camera_width
        x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
        left_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        left_rgb[..., 1] = x
        left_rgb[..., 2] = 96
        right_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        right_rgb[..., 0] = x[:, ::-1]
        right_rgb[..., 1] = 96
        views: Dict[str, Dict[str, Any]] = {}
        for name, rgb, lateral in (
            ("left_wrist", left_rgb, 0.25),
            ("right_wrist", right_rgb, -0.25),
        ):
            params = dict(self._build_camera_params())
            params.update(
                {
                    "source": f"mock_robot_{name}_synthetic",
                    "synthetic_test_view": True,
                    "lateral_offset_m": lateral,
                }
            )
            views[name] = {
                "rgb": rgb,
                "depth": np.array(ego_depth, copy=True),
                "camera_params": params,
            }
        return views

    def get_human_observations(
        self,
        human_ids: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Return ego observations for human policies, never for the robot."""
        ids = human_ids or [self._primary_human_id]
        observations: Dict[str, Dict[str, Any]] = {}
        for human_id in ids:
            position = np.asarray(self.get_human_position(human_id), dtype=float)
            distance = self._distance(self.robot_pos, position)
            robot_visible = distance <= self.visibility_radius
            relation = "nearby" if distance <= 1.5 else "farther away"
            if not robot_visible:
                relation = "not visible from the current human position"
            observations[human_id] = {
                "text": f"From your position, the robot is {relation}.",
                "position": position.tolist(),
                "robot_visible": robot_visible,
                "robot_distance_m": round(distance, 2),
                "robot_can_communicate": self.can_communicate(human_id),
                "last_utterance": self.last_utterance,
                "ego_view": {
                    "status": "text_only_mock",
                    "rgb": None,
                    "depth": None,
                },
            }
        return observations

    def get_world_snapshot(
        self,
        human_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        ids = human_ids or [self._primary_human_id]
        humans: Dict[str, Dict[str, Any]] = {}
        for human_id in ids:
            position = np.asarray(self.get_human_position(human_id), dtype=float)
            humans[human_id] = {
                "position": position.tolist(),
                "robot_distance_m": round(self._distance(self.robot_pos, position), 2),
                "action_status": "idle",
                "animation_state": "idle",
            }
        return {
            "step": self.step_count,
            "robot": {
                "position": self.robot_pos.tolist(),
                "yaw_rad": float(math.radians(self.robot_yaw)),
                "last_action": self.last_action,
            },
            "humans": humans,
            "last_utterance": self.last_utterance,
        }

    def can_communicate(self, human_name: str) -> bool:
        position = np.asarray(
            self.get_human_position(self._require_human_id(human_name)),
            dtype=float,
        )
        return self._distance(
            self.robot_pos,
            position,
        ) <= HUMAN_COMMUNICATION_RADIUS_M and self.is_human_visible(human_name)

    def is_human_visible(self, human_name: str) -> bool:
        position = np.asarray(
            self.get_human_position(self._require_human_id(human_name)),
            dtype=float,
        )
        vector = position[:2] - self.robot_pos[:2]
        distance = float(np.linalg.norm(vector))
        if distance > self.visibility_radius:
            return False
        bearing = math.degrees(math.atan2(vector[1], vector[0]))
        delta = (bearing - self.robot_yaw + 180.0) % 360.0 - 180.0
        return abs(delta) <= self._camera_fov_deg * 0.5

    def evaluate_visual_object_visibility(
        self,
        *,
        object_ids: tuple[str, ...],
    ) -> Dict[str, Any]:
        """Evaluate exact episode objects in the current synthetic ego view."""

        required = tuple(dict.fromkeys(str(value) for value in object_ids))
        unknown = [value for value in required if value not in self.objects]
        if unknown:
            raise ValueError(
                f"visual milestone references unknown mock objects: {unknown}"
            )
        visible = {
            object_id: bool(self._visible_relation(self.objects[object_id].position))
            for object_id in required
        }
        return {
            "matched": all(visible.values()),
            "evidence_source": "mock_egocentric_geometry",
            "required_object_ids": list(required),
            "visible_object_ids": [key for key, value in visible.items() if value],
            "visibility": visible,
        }

    def set_interaction_feedback(self, message: str):
        self._interaction_feedback = message

    @staticmethod
    def _required_magnitude(decision: Dict[str, Any], key: str) -> float:
        """Read a commanded magnitude, refusing a decision that omits one.

        Every locomotion primitive and the camera tilt carry their own magnitude,
        and the tool handler has already validated the argument's presence and
        range by the time a decision reaches a backend. A decision arriving here
        without one therefore means a producer bypassed that contract, so it
        fails here rather than moving the mock robot by an invented default.
        """
        value = decision.get(key)
        if value is None:
            raise ValueError(
                f"MockEnv decision {decision.get('action')!r} missing "
                f"{key!r}: {decision!r}"
            )
        return float(value)

    def step(
        self, decision: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        self.step_count += 1
        action = decision.get("action")
        if not action:
            raise ValueError(f"MockEnv decision missing action: {decision!r}")
        require_standing_g1_locomotion(
            action,
            body_posture=self._body_posture,
        )
        self.last_action = action
        executed = True
        feedback = ""
        end_effector_motion = None
        navigate_skill = None

        if action == "move_body":
            self._apply_move_body_delta(decision.get("delta"))
        elif action in {
            "move_forward",
            "move_backward",
            "move_left",
            "move_right",
        }:
            # The mock has no gait and no physics, so the requested magnitude is
            # applied exactly: this backend's purpose is a reproducible pose, and
            # learned-gait tracking residual in Isaac is not something to
            # imitate with a fake error term.
            distance_m = self._required_magnitude(decision, "distance_m")
            radians = math.radians(self.robot_yaw)
            if action in {"move_forward", "move_backward"}:
                sign = 1.0 if action == "move_forward" else -1.0
                direction = np.array([math.cos(radians), math.sin(radians)])
            else:
                sign = 1.0 if action == "move_left" else -1.0
                direction = np.array([-math.sin(radians), math.cos(radians)])
            self.robot_pos[:2] += sign * distance_m * direction
        elif action == "turn_left":
            degrees = self._required_magnitude(decision, "degrees")
            self.robot_yaw = (self.robot_yaw + degrees) % 360.0
        elif action == "turn_right":
            degrees = self._required_magnitude(decision, "degrees")
            self.robot_yaw = (self.robot_yaw - degrees) % 360.0
        elif action == "tilt_camera":
            self.set_camera_gaze_tilt_deg(
                self._required_magnitude(decision, "pitch_deg")
            )
        elif action == "crouch":
            self._body_posture = "crouched"
        elif action == "stand_up":
            self._body_posture = "standing"
        elif action == "navigate":
            executed, feedback, navigate_skill = self._execute_navigate_skill(
                decision,
            )
        elif action == "move_end_effector":
            raw_delta = decision.get("delta")
            if not isinstance(raw_delta, (list, tuple)) or len(raw_delta) != 7:
                raise ValueError(
                    "MockEnv move_end_effector requires a seven-number delta."
                )
            delta = np.asarray(raw_delta, dtype=np.float64).reshape(-1)
            if (
                not np.all(np.isfinite(delta))
                or np.any(
                    np.abs(delta[:3]) > G1_END_EFFECTOR_DELTA_MAX_TRANSLATION_M + 1e-12
                )
                or float(np.linalg.norm(delta[3:6]))
                > G1_END_EFFECTOR_DELTA_MAX_ROTATION_RAD + 1e-12
                or not 0.0 <= float(delta[6]) <= 1.0
            ):
                raise ValueError(
                    "MockEnv move_end_effector delta is outside its contract."
                )
            self._right_gripper_closure = float(delta[6])
            end_effector_motion = {
                "executed": True,
                "hand": "right",
                "requested_delta": delta.tolist(),
                "achieved_delta": delta.tolist(),
                "position_residual_m": 0.0,
                "orientation_residual_rad": 0.0,
                "gripper_residual": 0.0,
            }
            feedback = (
                "Mock right-palm delta executed exactly; measured gripper "
                f"{self._right_gripper_closure:.2f}."
            )
        elif action == "pick":
            candidate = self._nearest_reachable_object()
            if candidate is None:
                executed = False
                feedback = "No reachable object is visible."
            else:
                candidate.held_by = "robot"
                self.robot_holding = candidate.name
        elif action == "handover_object":
            claim = self._apply_handover_object(decision)
            feedback = str(claim["feedback"])
            executed = True
        elif action in {"wait", "ask", "speak", "report_done"}:
            pass
        else:
            executed = False
            feedback = (
                f'Action "{action}" has no perception-grounded skill executor. '
                "Use low-level movement or a supported physical skill."
            )

        if feedback:
            self._interaction_feedback = feedback
        info = {
            "executed": executed,
            "feedback": feedback,
            "controller_mode": "standard",
        }
        if end_effector_motion is not None:
            info["end_effector_motion"] = end_effector_motion
        if navigate_skill is not None:
            info["navigate_skill"] = navigate_skill
        if self._last_handover_claim is not None and action == "handover_object":
            info["handover"] = dict(self._last_handover_claim["private"])
        return self.observe(), self.step_count >= self.max_steps, info

    def _execute_navigate_skill(
        self,
        decision: Dict[str, Any],
    ) -> tuple[bool, str, dict[str, Any]]:
        raw = decision.get("instruction")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("navigate requires `instruction`.")
        instruction = raw.strip()
        if not internav_backend_available():
            raise RuntimeError(
                "navigate reached MockEnv without a DualVLN backend. "
                "Configure the DualVLN agent-server."
            )
        backend = build_internnav_backend()
        backend.reset()
        inner: list[dict[str, Any]] = []
        stop_reason = "max_inner_steps"
        for index in range(DUALVLN_MAX_INNER_STEPS):
            command = backend.step(self._dualvln_observation(instruction))
            inner.append(
                {
                    "inner_step": index,
                    "status": command.status,
                    "raw_action": command.raw_action,
                }
            )
            if command.status == "stop":
                stop_reason = "dualvln_stop"
                break
            self._apply_move_body_delta(command.delta)
        telemetry = {
            "backend": backend.name,
            "instruction": instruction,
            "inner_steps": inner,
            "stop_reason": stop_reason,
        }
        return (
            True,
            (
                f"DualVLN finished {instruction!r} after {len(inner)} inner "
                f"steps ({stop_reason})."
            ),
            telemetry,
        )

    def _apply_move_body_delta(self, delta: Any) -> None:
        """Apply one body-frame ``[forward_m, left_m, turn_deg]`` pose update."""

        if not isinstance(delta, (list, tuple)) or len(delta) != 3:
            raise ValueError("MockEnv move_body requires delta=[x, y, theta]")
        forward_m, left_m, turn_deg = (float(value) for value in delta)
        radians = math.radians(self.robot_yaw)
        self.robot_pos[:2] += forward_m * np.array(
            [math.cos(radians), math.sin(radians)]
        ) + left_m * np.array([-math.sin(radians), math.cos(radians)])
        self.robot_yaw = (self.robot_yaw + turn_deg) % 360.0

    def _dualvln_observation(self, instruction: str) -> DualVLNObservation:
        rgb = np.zeros(
            (DUALVLN_RGB_HEIGHT, DUALVLN_RGB_WIDTH, 3),
            dtype=np.uint8,
        )
        depth = official_norm_depth(
            np.ones((DUALVLN_RGB_HEIGHT, DUALVLN_RGB_WIDTH), dtype=np.float32)
        )
        yaw = math.radians(self.robot_yaw)
        return DualVLNObservation(
            rgb=rgb,
            depth=depth,
            instruction=instruction,
            globalgps=(
                float(self.robot_pos[0]),
                float(self.robot_pos[1]),
                float(self.robot_pos[2]),
            ),
            globalrotation=(0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)),
        )

    def set_human_speech(self, message: str, *, speaker: str):
        """Record who said what; see IsaacAutonomousEnv.set_human_speech."""
        speaker = str(speaker).strip()
        if not speaker:
            raise ValueError(
                "set_human_speech requires a speaker; an unattributed "
                "utterance cannot be resolved in a multi-Human episode."
            )
        self.last_utterance = {
            "speaker": speaker,
            "text": str(message),
            "physics_steps": self.step_count,
        }

    def apply_human_action(self, human_name: str, action: Dict[str, Any]):
        command = str(action.get("command") or "")
        if command in {"speak", "ask", "report_done"}:
            self.set_human_speech(
                str(action.get("message", "")),
                speaker=human_name,
            )
        elif command == "goto":
            target = action.get("target")
            if target is not None and len(target) >= 2:
                human_id = self._require_human_id(human_name)
                position = self._human_positions[human_id]
                position[:2] = np.asarray(target[:2], dtype=float)
                if len(target) >= 3:
                    position[2] = float(target[2])
            else:
                return HumanBodyResult(
                    human_name, command, False, "goto requires a target"
                )
        elif command not in {"idle", "listen"}:
            return HumanBodyResult(
                human_name, command, False, "Unsupported mock Human command"
            )
        return HumanBodyResult(human_name, command, True)

    def plan_spatial_blocker_clearance(self, human_id: str) -> Dict[str, Any]:
        """Return the next explicitly configured Robot-aware test plan."""

        call = {
            "human_id": str(human_id),
            "robot_position": self.robot_pos.tolist(),
            "human_position": self.get_human_position(human_id),
        }
        self.spatial_clearance_plan_calls.append(call)
        if not self._spatial_clearance_plans:
            raise RuntimeError(
                "Mock spatial blocking requires an explicit "
                "scene_setup.spatial_clearance_plans sequence."
            )
        plan = dict(self._spatial_clearance_plans.pop(0))
        plan.setdefault("robot_position", call["robot_position"])
        return plan

    def evaluate_success(self, check: DiagnosticCheck) -> bool:
        """Evaluate hidden ground truth. This output never enters observations."""
        params = check.params
        if check.type == "robot_near_human":
            return self._distance(self.robot_pos, self.human_pos) <= float(
                params.get("distance", 1.5)
            )
        if check.type == "robot_near_position":
            target = np.asarray(
                params.get("position", [math.inf, math.inf, 0]), dtype=float
            )
            return self._distance(self.robot_pos, target) <= float(
                params.get("distance", 1.0)
            )
        if check.type == "robot_holding_object":
            object_id = str(params.get("object_id") or "").strip()
            if not object_id:
                raise ValueError("robot_holding_object requires params.object_id")
            return self.robot_holding == object_id
        if check.type == "object_supported_by":
            object_id = str(params.get("object_id") or "").strip()
            destination_id = str(params.get("destination_object_id") or "").strip()
            if not object_id or not destination_id:
                raise ValueError(
                    "object_supported_by requires params.object_id and "
                    "params.destination_object_id"
                )
            target = self.objects.get(object_id)
            destination = self.objects.get(destination_id)
            if target is None or destination is None or target.held_by is not None:
                return False
            tolerance = float(params.get("distance", 0.10))
            return self._distance(target.position, destination.position) <= tolerance
        if check.type == "handover_accepted":
            human_id = str(params.get("human_id") or "").strip()
            object_id = str(params.get("object_id") or "").strip()
            if not human_id:
                raise ValueError("handover_accepted requires params.human_id")
            if not object_id:
                raise ValueError("handover_accepted requires params.object_id")
            return self.handover_was_accepted(
                human_id=human_id,
                object_id=object_id,
            )
        return False

    def evaluate_object_localization(
        self,
        *,
        object_id: str,
        projected_world_position: List[float],
        tolerance_m: float,
    ) -> Dict[str, Any]:
        """Compare one sensor projection with live mock GT, evaluator-only."""

        point = np.asarray(projected_world_position, dtype=float).reshape(-1)
        tolerance = float(tolerance_m)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(
                "projected_world_position must contain three finite values"
            )
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance_m must be finite and positive")
        target = self.objects.get(str(object_id))
        if target is None:
            return {
                "object_id": str(object_id),
                "projected_world_position": point.tolist(),
                "tolerance_m": tolerance,
                "matched": False,
                "reason": "object_not_present",
            }
        gt_center = np.asarray(target.position, dtype=float)
        error = float(np.linalg.norm(point - gt_center))
        return {
            "object_id": str(object_id),
            "projected_world_position": point.tolist(),
            "gt_center_world_position": gt_center.tolist(),
            "center_error_m": error,
            "tolerance_m": tolerance,
            "matched": error <= tolerance,
        }

    def get_human_position(self, human_id: str) -> list[float]:
        key = str(human_id)
        if key not in self._human_positions:
            raise KeyError(
                f"No live position for human {key!r}; known ids="
                f"{sorted(self._human_positions)}"
            )
        return self._human_positions[key].tolist()

    def _require_human_id(self, human_ref: str) -> str:
        key = str(human_ref)
        if key in self._human_positions:
            return key
        matches = [
            str(spec.get("id") or "")
            for spec in self._human_specs
            if str(spec.get("id") or "") == key or str(spec.get("name") or "") == key
        ]
        matches = [item for item in matches if item in self._human_positions]
        if len(matches) != 1:
            raise KeyError(
                f"No live position for human {human_ref!r}; known ids="
                f"{sorted(self._human_positions)}"
            )
        return matches[0]

    def _positions_from_human_specs(self) -> Dict[str, np.ndarray]:
        positions: Dict[str, np.ndarray] = {}
        for index, spec in enumerate(self._human_specs):
            human_id = str(spec.get("id") or "").strip()
            if not human_id:
                raise ValueError(f"MockEnv human spec missing id: {spec!r}")
            if human_id in positions:
                raise ValueError(f"Duplicate MockEnv human id: {human_id!r}")
            raw = spec.get("start_position")
            if raw is None:
                if index != 0:
                    raise ValueError(
                        f"MockEnv human {human_id!r} is missing start_position"
                    )
                raw = self._human_start
            try:
                position = np.asarray(raw, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"MockEnv human {human_id!r} has invalid start_position: {raw!r}"
                ) from exc
            if (
                position.ndim != 1
                or position.shape[0] < 2
                or not np.all(np.isfinite(position))
            ):
                raise ValueError(
                    f"MockEnv human {human_id!r} has invalid start_position: {raw!r}"
                )
            if position.shape[0] < 3:
                position = np.array(
                    [float(position[0]), float(position[1]), 0.0],
                    dtype=float,
                )
            positions[human_id] = position
        return positions

    def handover_was_accepted(self, *, human_id: str, object_id: str) -> bool:
        """Return accepted spoken-handover state, never physical Human hold."""

        return self._accepted_handovers.get(str(object_id)) == str(human_id)

    def _apply_handover_object(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        from tarsphere.infra.manipulation.handover import (
            HandoverHuman,
            evaluate_handover_claim,
        )

        recipient = decision.get("recipient")
        if not isinstance(recipient, str):
            recipient = ""
        humans = [
            HandoverHuman(
                id=str(human.get("id") or ""),
                name=str(human.get("name") or ""),
                character_name=str(human.get("character_name") or ""),
                asset=str(human.get("asset") or ""),
                role=str(human.get("role") or ""),
                descriptions=tuple(human.get("descriptions") or ()),
            )
            for human in self._human_specs
        ]
        live_positions = {
            human.id: self.get_human_position(human.id) for human in humans if human.id
        }
        verdict = evaluate_handover_claim(
            target=recipient,
            humans=humans,
            held_object_id=self.robot_holding,
            object_transport_ready=self.robot_holding is not None,
            robot_xy=self.robot_pos[:2],
            human_positions=live_positions,
            visible_human_ids={
                human.id
                for human in humans
                if human.id and self.is_human_visible(human.id)
            },
            accepted_handovers=self._accepted_handovers,
        )
        if verdict.accepted:
            if not verdict.object_id or not verdict.resolved_human_id:
                raise RuntimeError(
                    "Accepted handover is missing the object or human id."
                )
            self._accepted_handovers[verdict.object_id] = verdict.resolved_human_id
        speaker = ""
        if verdict.resolved_human_id:
            speaker = next(
                (
                    str(human.get("name") or human.get("id") or "")
                    for human in self._human_specs
                    if str(human.get("id") or "") == verdict.resolved_human_id
                ),
                verdict.resolved_human_id,
            )
        if speaker:
            self.set_human_speech(verdict.feedback, speaker=speaker)
        claim = {
            "feedback": verdict.feedback,
            "private": verdict.as_private_record(),
        }
        self._last_handover_claim = claim
        return claim

    def _nearest_reachable_object(self) -> Optional[MockObject]:
        reachable = [
            obj
            for obj in self.objects.values()
            if obj.held_by is None
            and self._distance(self.robot_pos, obj.position) <= 0.8
        ]
        return (
            min(reachable, key=lambda obj: self._distance(self.robot_pos, obj.position))
            if reachable
            else None
        )

    def _visible_relation(
        self, target: np.ndarray, *, origin: Optional[np.ndarray] = None
    ) -> Optional[str]:
        origin = self.robot_pos if origin is None else origin
        vector = target[:2] - origin[:2]
        distance = float(np.linalg.norm(vector))
        if distance > self.visibility_radius:
            return None
        if origin is not self.robot_pos:
            return "nearby"
        bearing = math.degrees(math.atan2(vector[1], vector[0]))
        delta = (bearing - self.robot_yaw + 180.0) % 360.0 - 180.0
        if abs(delta) <= 35:
            direction = "ahead"
        elif delta > 0:
            direction = "to the left"
        else:
            direction = "to the right"
        proximity = "nearby" if distance < 1.0 else "a short distance away"
        return f"{direction}, {proximity}"

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a[:2] - b[:2]))

    # Compatibility helpers used by older callers.
    def get_distance_to_human(self) -> float:
        return self._distance(self.robot_pos, self.human_pos)

    def get_distance_to_object(self, name: str) -> float:
        obj = self.objects.get(name)
        return self._distance(self.robot_pos, obj.position) if obj else float("inf")
