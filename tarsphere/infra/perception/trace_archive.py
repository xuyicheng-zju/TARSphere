"""Normalize saved decision traces for the presentation renderer."""

from __future__ import annotations

import copy


def normalize_archived_trace(trace):
    """Adapt the September archive's field names for display, leaving raw logs intact.

    Older tool_calls include information tools before the physical call; use
    the executed call_name rather than mislabeling the first queued tool active.
    Older plans do not carry declared progress, so no progress is synthesized.
    """
    normalized = copy.deepcopy(trace)
    for item in normalized:
        envelope = item.get("envelope", {})
        if "actions" in envelope or "tool_calls" not in envelope:
            continue
        item["presentation_legacy_schema"] = True
        envelope["thinking"] = envelope.get("reasoning", "")
        envelope["actions"] = [
            {"tool": c.get("name", "unknown"), "args": c.get("args", {})}
            for c in envelope["tool_calls"]
        ]
        matches = [
            i
            for i, c in enumerate(envelope["actions"])
            if c["tool"] == item.get("call_name")
        ]
        if matches:
            item["action_index"] = matches[-1]
        item["action_sequence_length"] = len(envelope["actions"])
        plan = envelope.get("planning")
        if plan:
            envelope["plan"] = {
                "goal": plan.get("goal", ""),
                "steps": plan.get("subgoals", []),
            }
        envelope["memory"] = [
            {"op": m.get("operation"), "content": m.get("content", "")}
            for m in envelope.get("key_memory_updates", [])
        ]
    return normalized
