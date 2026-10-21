"""Own G1 locomotion policy output and persistent upper-body controller state.

The lower-body policy and articulation mapping remain the official
``G1MoveBySpeedController``. TARSphere's path wrappers omit the optional
15-D upper-body argument and the pelvis-height argument when they call that
controller as a sub-controller. This adapter changes only that omitted-argument
behavior: an explicit target updates the cache, while an omitted target reuses
it. After grasp attachment, the oracle explicitly restores the official
all-zero locomotion target, so the cache does not preserve the transient
reaching IK during transport. The same rule keeps a crouch height active
until ``stand_up`` writes the standing command.

TARSphere's loader copies the CPU ONNX result to the literal device
``cuda:0`` and immediately copies it back to CPU. That unrelated GPU round trip
breaks whenever Kit is assigned another physical GPU and GPU 0 hosts a model
service. This adapter replaces only that loader result with an equivalent CPU
tensor; locomotion inference and downstream joint mapping are unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tarsphere.simulator.core.robot.controller import BaseController
from tarsphere.simulator.extensions.controllers.g1_move_by_speed_controller import (
    G1MoveBySpeedController,
)

from tarsphere.infra.control.physical_contracts import (
    G1_STANDING_HEIGHT_COMMAND_M,
    G1_DEFAULT_LOWER_DOF_POS_RAD,
    resolve_persisted_g1_body_height,
)

_G1_LOWER_POLICY_TARGET_SIZE = 12
_G1_UPPER_POLICY_TARGET_SIZE = 15
_G1_DEFAULT_LOWER_DOF_POS = np.array(
    G1_DEFAULT_LOWER_DOF_POS_RAD,
    dtype=np.float64,
)
_G1_DEFAULT_UPPER_DOF_POS = np.zeros(_G1_UPPER_POLICY_TARGET_SIZE, dtype=np.float64)
_G1_RIGHT_WRIST_YAW_UPPER_INDEX = 14


def _load_cpu_onnx_policy(path: str):
    """Return the official ONNX inference shape without a CUDA round trip."""
    import onnxruntime as ort
    import torch

    model = ort.InferenceSession(path)
    input_name = model.get_inputs()[0].name

    def run_inference(input_tensor):
        ort_outputs = model.run(
            None,
            {input_name: input_tensor.detach().cpu().numpy()},
        )
        return torch.from_numpy(np.asarray(ort_outputs[0]))

    return run_inference


@BaseController.register("TARSphereG1MoveBySpeedController")
class TARSphereG1MoveBySpeedController(G1MoveBySpeedController):
    """Official G1 locomotion with persistent omitted upper-body targets."""

    def __init__(self, config: Any, robot: Any, scene: Any) -> None:
        super().__init__(config=config, robot=robot, scene=scene)
        self._policy = _load_cpu_onnx_policy(config.policy_weights_path)
        if not hasattr(self.robot, "_tarsphere_upper_body_target"):
            self.robot._tarsphere_upper_body_target = np.zeros(15, dtype=np.float64)
        if not hasattr(self.robot, "_tarsphere_body_height"):
            self.robot._tarsphere_body_height = float(G1_STANDING_HEIGHT_COMMAND_M)

    @staticmethod
    def _default_gym_dof_pos() -> np.ndarray:
        return np.concatenate([_G1_DEFAULT_LOWER_DOF_POS, _G1_DEFAULT_UPPER_DOF_POS])

    def _refresh_cached_upper_body_command(
        self,
        target_upper_joint_pos: np.ndarray,
    ) -> None:
        """Recompute the cached articulation target without rerunning the policy.

        The official controller reuses ``applied_joint_positions`` for three
        physics ticks between policy inferences. That is fine for locomotion,
        but manipulation trajectories supply a fresh 15-D upper-body vector
        every tick; returning the stale cache drops those commands entirely.
        """
        upper = np.asarray(
            target_upper_joint_pos,
            dtype=np.float64,
        ).reshape(_G1_UPPER_POLICY_TARGET_SIZE)
        lower = np.asarray(
            self._last_target_joint_positions[:_G1_LOWER_POLICY_TARGET_SIZE],
            dtype=np.float64,
        )
        self._last_target_joint_positions = np.clip(
            np.concatenate([lower, upper]),
            -100.0,
            100.0,
        )
        gym_applied = (
            self._last_target_joint_positions * 0.25 + self._default_gym_dof_pos()
        )
        self.applied_joint_positions = self.gym_adapter.gym2sim(gym_applied)

    def forward(
        self,
        forward_speed: float = 0.0,
        rotation_speed: float = 0.0,
        lateral_speed: float = 0.0,
        height: float = 0.74,
        target_upper_joint_pos: Any = None,
    ):
        if target_upper_joint_pos is None:
            target = np.asarray(
                self.robot._tarsphere_upper_body_target, dtype=np.float64
            ).reshape(15)
        else:
            target = np.asarray(target_upper_joint_pos, dtype=np.float64).reshape(-1)
            if target.size != 15 or not np.all(np.isfinite(target)):
                raise ValueError("G1 upper-body target must contain 15 finite values.")
            # Top-level and nested path controllers are distinct instances but
            # share one robot. Store the explicit pose on that robot so every
            # official locomotion wrapper sees the same upper-body target.
            self.robot._tarsphere_upper_body_target = target.copy()
        height, self.robot._tarsphere_body_height = resolve_persisted_g1_body_height(
            height,
            getattr(self.robot, "_tarsphere_body_height", None),
        )
        if int(getattr(self, "_apply_times_left", 0)) > 0:
            self._apply_times_left -= 1
            self._refresh_cached_upper_body_command(target)
            if self.joint_subset is None:
                from tarsphere.simulator.core.robot.articulation_action import (
                    ArticulationAction,
                )

                return ArticulationAction(
                    joint_positions=self.applied_joint_positions,
                )
            return self.joint_subset.make_articulation_action(
                joint_positions=self.applied_joint_positions,
                joint_velocities=None,
            )
        return super().forward(
            forward_speed=forward_speed,
            rotation_speed=rotation_speed,
            lateral_speed=lateral_speed,
            height=height,
            target_upper_joint_pos=target,
        )
