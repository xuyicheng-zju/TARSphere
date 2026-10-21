"""Isolated Psi0 HTTP client. Not a harness tool.

Isaac never loads Psi0 weights. Official serve is ``POST /act``.
This optional package is imported only by explicit Psi0 integrations; it is
not selected by the formal eight-model evaluation suite.
"""

from tarsphere.policies.psi0.apply import (
    g1_fixed_base_controller_action,
    remap_right_hand_to_dex3,
)
from tarsphere.policies.psi0.contract import (
    TARSPHERE_PSI0_ENDPOINT_ENV,
    PSI0_ACTION_DIM,
    PSI0_ACTION_EXEC_HORIZON,
    PSI0_DEFAULT_INSTRUCTION,
    PSI0_FROZEN_LOCO_DIM,
    PSI0_IMAGE_KEY,
    PSI0_TABLETOP_EVAL_INSTRUCTION,
    PSI0_TABLETOP_REMOTE_DIR,
    PSI0_TABLETOP_TASK,
    PSI0_WAIST_JOINT_NAMES,
    Psi0ContractError,
    require_not_sonic_checkpoint,
    split_psi0_action,
)
from tarsphere.policies.psi0.observation import (
    named_joint_positions,
    observation_from_g1,
    proprio_from_named_joints,
    require_head_rgb,
    torso_rpyh_from_named_joints,
)
from tarsphere.policies.psi0.server import (
    Psi0ActSession,
    Psi0ServerEndpoint,
    Psi0ServerError,
    require_psi0_endpoint,
)

__all__ = [
    "TARSPHERE_PSI0_ENDPOINT_ENV",
    "PSI0_ACTION_DIM",
    "PSI0_ACTION_EXEC_HORIZON",
    "PSI0_DEFAULT_INSTRUCTION",
    "PSI0_FROZEN_LOCO_DIM",
    "PSI0_IMAGE_KEY",
    "PSI0_TABLETOP_EVAL_INSTRUCTION",
    "PSI0_TABLETOP_REMOTE_DIR",
    "PSI0_TABLETOP_TASK",
    "PSI0_WAIST_JOINT_NAMES",
    "Psi0ActSession",
    "Psi0ContractError",
    "Psi0ServerEndpoint",
    "Psi0ServerError",
    "g1_fixed_base_controller_action",
    "named_joint_positions",
    "observation_from_g1",
    "proprio_from_named_joints",
    "remap_right_hand_to_dex3",
    "require_head_rgb",
    "require_not_sonic_checkpoint",
    "require_psi0_endpoint",
    "split_psi0_action",
    "torso_rpyh_from_named_joints",
]
