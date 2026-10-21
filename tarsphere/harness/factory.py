"""Construct the robot agent for episode runners.

The only supported robot agent is the envelope-protocol pipeline
(tools + StrategyPlanner; EVALUATION.md):

    VLM (one strict JSON decision) -> EnvelopeKernel -> tools -> simulator

The factory returns an object implementing
``act(observation, instruction, action_space, *, decision_requirements=None) -> dict``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Optional

from tarsphere.configs.secrets import DEFAULT_AGENT_MAX_TOKENS, agent_llm_defaults
from tarsphere.harness.harness import AgentHarnessConfig


def build_robot_agent_envelope(
    *,
    api_key: Optional[str] = None,
    fallback_api_keys: tuple[str, ...] = (),
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    max_tokens: int = DEFAULT_AGENT_MAX_TOKENS,
    agent_harness_config: AgentHarnessConfig | None = None,
    episode_actions: Iterable[str] | None = None,
) -> object:
    """Build the new-protocol agent: atomic tools + EnvelopeKernel.

    Wires the active architecture (tools + StrategyPlanner):
    - every capability registered as one tool via register_atomic_tools
    - EnvelopeKernel parses one JSON object and resolves its tool_calls proposal
    - StrategyPlanner persists planning for the evaluator

    Returns a RobotAgent whose kernel is EnvelopeKernel and whose ``act()``
    decision dict carries the full envelope.
    """
    from tarsphere.harness.agent import RobotAgent
    from tarsphere.harness.kernel.envelope_kernel import EnvelopeKernel
    from tarsphere.harness.kernel.vlm_client import VLMClient
    from tarsphere.harness.perception.module import RobotPerceptionModule

    defaults = agent_llm_defaults(api_key=api_key, model=model, api_base=api_base)
    resolved_config = agent_harness_config or AgentHarnessConfig()

    tools, memory = build_robot_tool_registry(
        agent_harness_config=resolved_config,
        episode_actions=episode_actions,
    )

    perception = RobotPerceptionModule()
    vlm = VLMClient(
        api_key=defaults.api_key,
        fallback_api_keys=fallback_api_keys,
        api_base=defaults.api_base,
        model=defaults.model,
        max_tokens=max_tokens,
    )
    kernel = EnvelopeKernel(
        vlm=vlm,
        tools=tools,
        protocol=resolved_config.protocol,
        strategy_planner=memory.strategy_planner,
    )

    return RobotAgent(
        agent_harness_config=resolved_config,
        perception=perception,
        memory=memory,
        kernel=kernel,
        tools=tools,
    )


def build_robot_tool_registry(
    *,
    agent_harness_config: AgentHarnessConfig | None = None,
    grasp_grounding: str = "semantic",
    episode_actions: Iterable[str] | None = None,
) -> tuple[object, object]:
    """Build the physical/information tools without constructing a VLM.

    Deterministic simulator certification uses this entry point to exercise
    the exact same SAM3, GraspGenX, and action handlers as an evaluated Agent
    while proving that no language-model policy participates in the rollout.

    ``grasp_grounding`` names who supplies the current target mask every
    GraspGenX pose is conditioned on. ``'semantic'`` (default) requires the
    SAM3 backend. ``'oracle_gt'`` means the caller supplies that mask from
    the frozen episode AABB, so this factory must not construct SAM3.
    """

    from tarsphere.harness.memory.episodic import EpisodicMemory
    from tarsphere.harness.tools.atomic import register_atomic_tools
    from tarsphere.harness.tools.registry import ToolRegistry

    if grasp_grounding not in {"semantic", "oracle_gt"}:
        raise ValueError(
            "grasp_grounding must be 'semantic' or 'oracle_gt'; "
            f"got {grasp_grounding!r}."
        )

    resolved_config = agent_harness_config or AgentHarnessConfig()
    caller_supplies_grasp_mask = grasp_grounding == "oracle_gt"
    admitted_actions = (
        None
        if episode_actions is None
        else frozenset(str(action) for action in episode_actions)
    )

    def admits_any(*names: str) -> bool:
        return admitted_actions is None or bool(admitted_actions.intersection(names))

    # Perception backends (SAM3): semantic 2D
    # grounding backing the public `locate_object` and `grasp_object` tools.
    # Exact object-id visibility uses RGB-D + episode AABBs and does not need
    # SAM3. Oracle GT grasp grounding
    # supplies its own mask, so SAM3 is not constructed on that path.
    use_sam3 = (not caller_supplies_grasp_mask) and admits_any(
        "locate_object", "grasp_object"
    )
    sam3_backend = None
    if use_sam3:
        from tarsphere.harness.perception.remote_backends import (
            RemoteSAM3PerceptionBackend,
        )

        # Model weights are never committed, so the checkpoint location is a
        # deployment fact the operator must state. Defaulting to one machine's
        # path silently binds the runtime to that checkout.
        sam3_ckpt = (os.environ.get("TARSPHERE_SAM3_CHECKPOINT") or "").strip()
        if not sam3_ckpt:
            raise RuntimeError(
                "SAM3 requires TARSPHERE_SAM3_CHECKPOINT to name the "
                "SAM3.1 multiplex checkpoint. Weights are not tracked in git; see "
                "third_party/README.md for the download step."
            )
        if not os.path.isfile(sam3_ckpt) or not os.access(sam3_ckpt, os.R_OK):
            raise RuntimeError(
                f"TARSPHERE_SAM3_CHECKPOINT does not name a readable file: {sam3_ckpt}"
            )
        sam3_backend = RemoteSAM3PerceptionBackend()
        # Tool registration must not advertise an unloaded capability and then
        # spend policy calls discovering missing GPU memory at the first grasp.
        # The worker retains these weights across episode resets/rebuilds.
        sam3_backend.require_ready()

    grasp_model = None
    if admits_any("grasp_object"):
        if sam3_backend is None and not caller_supplies_grasp_mask:
            raise RuntimeError(
                "GraspGenX requires SAM3 because every "
                "grasp pose is conditioned on a current SAM3 target mask."
            )
        grasp_model = build_graspgenx_backend()

    memory = EpisodicMemory(protocol=resolved_config.protocol)

    navigation_backend = None
    navigation_fallback_backend = None
    if admits_any("navigate_to_frontier"):
        from tarsphere.infra.navigation.open_frontier.harness import (
            OpenFrontierHarnessBackend,
            OpenFrontierHarnessSettings,
        )

        navigation_backend = OpenFrontierHarnessBackend(
            OpenFrontierHarnessSettings.from_environment()
        )
    from tarsphere.infra.navigation.x_navdp import (
        XNavDPClient,
        XNavDPSettings,
    )

    navigation_fallback_backend = XNavDPClient(XNavDPSettings.from_environment())

    tools = ToolRegistry()
    register_atomic_tools(
        tools,
        semantic_backend=sam3_backend,
        grasp_backend=grasp_model,
        memory=memory,
        navigation_backend=navigation_backend,
        navigation_fallback_backend=navigation_fallback_backend,
    )
    if caller_supplies_grasp_mask and grasp_model is not None:
        from tarsphere.harness.tools.handlers import GraspObjectHandler
        from tarsphere.harness.tools.specs import GRASP_OBJECT

        # Atomic registration withholds grasp_object unless SAM3 is present.
        # Oracle GT grounding still needs the GraspGenX client on that handler
        # so the caller can condition poses on its own current target mask.
        tools.register(
            GRASP_OBJECT,
            GraspObjectHandler(
                semantic_backend=None,
                grasp_backend=grasp_model,
            ),
        )

    return tools, memory


def build_graspgenx_backend() -> object:
    """Connect to the configured GraspGenX replica for this worker lane."""

    import json
    from pathlib import Path

    from tarsphere.infra.manipulation.graspgenx import (
        GraspGenXClient,
        GraspGenXSettings,
    )

    port = int(os.environ.get("TARSPHERE_GRASPGENX_PORT", "5556"))
    timeout_ms = 180_000
    cache_root = os.environ.get("TARSPHERE_EVALUATION_CACHE_ROOT")
    pool_path = Path(cache_root) / "graspgenx_pool.json" if cache_root else None
    if pool_path is not None and pool_path.is_file():
        pool = json.loads(pool_path.read_text(encoding="utf-8"))
        ports = tuple(int(value) for value in pool["ports"])
        if (
            not ports
            or len(set(ports)) != len(ports)
            or port not in ports
            or any(not 1 <= value <= 65535 for value in ports)
        ):
            raise ValueError(f"Invalid GraspGenX replica pool: {pool_path}")
        gpu = int(os.environ["TARSPHERE_KIT_ACTIVE_GPU"])
        slot = int(os.environ["TARSPHERE_ISAAC_RUNTIME_SLOT"])
        port = ports[(gpu * 3 + slot) % len(ports)]
        timeout_ms = int(pool.get("request_timeout_ms", 600_000))
        if not 180_000 <= timeout_ms <= 900_000:
            raise ValueError(f"Invalid GraspGenX pool timeout: {pool_path}")
        print(
            f"[GRASPGENX] GPU {gpu} slot {slot} uses port {port} "
            f"with {timeout_ms} ms timeout.",
            flush=True,
        )
    backend = GraspGenXClient(
        settings=GraspGenXSettings(
            host=os.environ.get("TARSPHERE_GRASPGENX_HOST", "127.0.0.1"),
            port=port,
            request_timeout_ms=timeout_ms,
        ),
    )
    backend.require_ready()
    return backend


__all__ = [
    "build_graspgenx_backend",
    "build_robot_agent_envelope",
    "build_robot_tool_registry",
]
