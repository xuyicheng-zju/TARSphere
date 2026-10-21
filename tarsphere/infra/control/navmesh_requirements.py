"""Detect whether a runtime episode needs AnimPeople NavMesh locomotion.

NavMesh bake exists for Human ``goto`` (omni.anim.people GoTo).
Robot locomotion uses ``move_by_speed`` and does not require a baked NavMesh.
Stationary humans (``speak`` / ``idle`` only) also do not.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator

# Matches HumanActorRuntime locomotion commands that plan on NavMesh.
_NAVMESH_HUMAN_COMMANDS = frozenset({"goto"})


def _normalize_command(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def iter_milestone_human_actions(episode: Any) -> Iterator[dict]:
    """Yield Human actions from the episode's sole milestone authority."""

    evaluation = getattr(episode, "evaluation", None)
    if evaluation is None and isinstance(episode, dict):
        evaluation = episode.get("evaluation")
    if isinstance(evaluation, dict):
        milestones = evaluation.get("milestones") or []
    else:
        milestones = getattr(evaluation, "milestones", None) or []
    for milestone in milestones:
        if isinstance(milestone, dict):
            event = milestone.get("event") or {}
        else:
            event = getattr(milestone, "event", None)
        if isinstance(event, dict):
            actions = event.get("human_actions") or []
        else:
            actions = getattr(event, "human_actions", None) or []
        for action in actions:
            if isinstance(action, dict):
                yield action


def _iter_named_action_lists(node: Any, *, keys: Iterable[str]) -> Iterator[dict]:
    """Yield action dicts from known list keys (human_actions / actor scripts)."""
    key_set = set(keys)
    stack: list[Any] = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            for key, value in cur.items():
                if key in key_set and isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            yield item
                elif isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(cur, list):
            stack.extend(cur)


def _iter_world_knowledge_actions(episode: Any) -> Iterator[dict]:
    """Scan authored actor scripts nested under world_knowledge (known list keys)."""
    wk = getattr(episode, "world_knowledge", None)
    if wk is None and isinstance(episode, dict):
        wk = episode.get("world_knowledge")
    if not isinstance(wk, dict):
        return
    yield from _iter_named_action_lists(
        wk,
        keys=("human_actions", "actor_actions", "actor_script", "human_script"),
    )


def episode_navmesh_reasons(episode: Any) -> list[str]:
    """Return human-readable reasons the episode needs a baked NavMesh."""
    reasons: list[str] = []
    for action in iter_milestone_human_actions(episode):
        command = _normalize_command(action.get("command"))
        if command in _NAVMESH_HUMAN_COMMANDS:
            human = action.get("human", action.get("human_id", "?"))
            reasons.append(f"milestone human_actions: human={human} command={command}")
    for action in _iter_world_knowledge_actions(episode):
        command = _normalize_command(action.get("command"))
        if command in _NAVMESH_HUMAN_COMMANDS:
            human = action.get("human", action.get("human_id", "?"))
            reasons.append(
                f"world_knowledge actor script: human={human} command={command}"
            )
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for reason in reasons:
        if reason in seen:
            continue
        seen.add(reason)
        unique.append(reason)
    return unique


def episode_needs_navmesh(episode: Any) -> bool:
    """True if any authored human action requires AnimPeople NavMesh GoTo."""
    return bool(episode_navmesh_reasons(episode))


def assert_navmesh_policy(
    episode: Any,
    *,
    skip_navmesh: bool,
    human_motion_backend: str = "animation",
) -> None:
    """Fail loud when ``--no-navmesh`` conflicts with episode locomotion needs.

    When ``skip_navmesh`` is False, this is a no-op (bake proceeds as usual).
    """
    if human_motion_backend not in {"animation", "kinematic"}:
        raise ValueError("human_motion_backend must be kinematic or animation")
    if human_motion_backend == "kinematic" or not skip_navmesh:
        return
    reasons = episode_navmesh_reasons(episode)
    if not reasons:
        return
    episode_id = getattr(episode, "episode_id", None)
    if episode_id is None and isinstance(episode, dict):
        episode_id = episode.get("episode_id")
    joined = "; ".join(reasons[:8])
    more = "" if len(reasons) <= 8 else f" (+{len(reasons) - 8} more)"
    raise RuntimeError(
        f"--no-navmesh / skip_navmesh=True but episode {episode_id!r} requires "
        f"AnimPeople NavMesh locomotion ({joined}{more}). Restart the scene "
        "server without --no-navmesh (mid-process NavMesh bake is not supported)."
    )
