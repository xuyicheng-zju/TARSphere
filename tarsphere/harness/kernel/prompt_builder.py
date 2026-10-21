"""Build the Agent's strict JSON interface for the current task event.

Every turn requires a tool proposal. A public episode event may additionally
require structured planning or task-update output so its event-local evaluator
has contemporaneous evidence. ``DecisionRequirements``
contains only output shape and public scope; event ground truth stays private.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from tarsphere.harness.core.failures import (
    ARGUMENT_FAILURES,
    BACKEND_FAILURES,
    WORLD_FAILURES,
    failure_vocabulary_line,
)
from tarsphere.harness.core.types import ToolSpec
from tarsphere.harness.memory.key import (
    MAX_KEY_MEMORIES,
    MAX_KEY_MEMORY_CONTENT_CHARS,
)
from tarsphere.contracts import DecisionRequirements
from tarsphere.harness.harness import (
    DEFAULT_VISUAL_HISTORY_FRAMES,
    DYNAMIC_ACTION_HORIZON,
    MAX_ACTION_HORIZON,
)


# ---------------------------------------------------------------------------
# Shared observation frames and identifiers
# ---------------------------------------------------------------------------
# Call-argument units and frames come from each tool's input schema. This
# section contains only conventions shared by observation channels or public
# identifiers, so the prompt has one authority for each fact.
def _interface_semantics(
    tools: Iterable[ToolSpec],
    *,
    lidar_access: bool = False,
) -> str:
    """Describe observation conventions not owned by a tool input schema."""

    tool_list = list(tools)
    tool_names = {tool.name for tool in tool_list}
    lines: list[str] = []
    if "locate_object" in tool_names:
        lines.append(
            "- Image boxes: `[x1,y1,x2,y2]` pixels in ego RGB, origin top-left."
        )
    if tool_names & {
        "move_body",
        "navigate_to_pointgoal",
        "navigate_to_frontier",
        "navigate_to_entity",
        "move_forward",
        "move_backward",
        "move_left",
        "move_right",
        "turn_left",
        "turn_right",
    }:
        lines.append(
            "- **Body-direction signs: +x = forward; +y = LEFT; -y = RIGHT; "
            "positive yaw = LEFT/counter-clockwise; negative yaw = "
            "RIGHT/clockwise.** +z is up. This right-handed metric frame is "
            "independent of image pixels."
        )
    if lidar_access:
        lines.append(
            "- **LiDAR bearing signs: 0° = ahead; positive = LEFT; negative "
            "= RIGHT.** Distances are metres from the body surface. Returns "
            "are unlabelled sampled surfaces, "
            "not identities, routes, arrival evidence, persistent maps, or "
            "proof of body-width clearance. Interpret them with current RGB "
            "and the intended motion; proximity alone is not a movement veto."
        )
    if tool_names & {"ask", "speak"}:
        lines.append(
            "- People: Human memory lists the episode cast by public display "
            "name with catalog appearance and the episode-specific public "
            "role when supplied. Bind current RGB appearance to those descriptions. A "
            "Human visibility check runs once before every action step, not "
            "at every physics tick. Language updates dialogue and its latest "
            "step but creates no spatial marker because no spatialized audio "
            "is exposed. Human memory retains the latest visually grounded "
            "step but exposes no position."
            + (" LiDAR returns never identify a person." if lidar_access else "")
        )
    if "navigate_to_frontier" in tool_names:
        lines.append(
            "- OpenFrontier overlay: the CURRENT ego RGB may show up to three "
            "FrontierNet candidates A/B/C, ranked by predicted information "
            "gain after visited-view discounting. Only candidates above the "
            "gain threshold are shown; labels may be absent. Candidate "
            "generation is independent of the occupancy map, so a mark is "
            "not proof of reachability. They are optional exploration "
            "suggestions, not task targets, success evidence, or mandatory "
            "actions. Choose by visual relevance to the current objective; "
            "execution will try A* and may use X-NavDP when the selected map "
            "cannot route the chosen point."
        )
        lines.append(
            "- Exploration BEV: the second image shows persistent occupancy, "
            "not task success evidence. World +X is up and +Y is left; the red "
            "arrow is your current pose/heading. The faint amber wedge is the "
            "current RGB camera horizontal bearing only (its drawn length does "
            "not claim visibility, range, free space, or reachability). Pale "
            "space is mapped free space, dark space is mapped obstacle, and "
            "grey space, if present, remains unknown. A green boundary and "
            "dashed blue travel trail may appear where the map has accumulated "
            "online observations. A/B/C mark the same current-view FrontierNet "
            "candidates as the ego RGB, without asserting that A* can already "
            "reach them."
            + (
                " Purple named markers are positions you explicitly saved "
                "with `remember_entity`: person icons for Humans, squares for "
                "objects, diamonds for places. These are historical anchors, "
                "not live tracking or verified identities; re-save to update them."
                if "remember_entity" in tool_names
                else ""
            )
        )
    return "\n".join(lines) or "- None beyond the current call schemas."


# ---------------------------------------------------------------------------
# Output schema declaration (tells the model the JSON shape it must emit)
# ---------------------------------------------------------------------------


def _format_parameters(schema: dict[str, Any]) -> list[str]:
    """Render one line per parameter, from the spec's own ``input_schema``.

    The schema is the single authority for a call's arguments, so the prompt
    derives the parameter list from it rather than restating it in prose that
    could drift. Each line carries everything a caller needs — name, type,
    whether it is mandatory, the values it is allowed to take, and what it
    means — which is the table from the design figure written as text a model
    reads in one pass.
    """
    properties = dict(schema.get("properties") or {})
    if not properties:
        return ["  params: (none)"]
    required = set(schema.get("required") or ())
    lines = ["  params:"]
    for key, meta in properties.items():
        meta = meta if isinstance(meta, dict) else {}
        kind = str(meta.get("type") or "any")
        if kind == "array":
            item_kind = (meta.get("items") or {}).get("type")
            kind = f"array of {item_kind}" if item_kind else "array"
        facets = [kind, "required" if key in required else "optional"]
        if meta.get("enum"):
            facets.append("one of " + "|".join(str(v) for v in meta["enum"]))
        # A declared numeric range is part of the contract the caller must
        # satisfy — an argument outside it is rejected — so it belongs on the
        # rendered line rather than only in the schema the model never sees.
        minimum, maximum = meta.get("minimum"), meta.get("maximum")
        if minimum is not None and maximum is not None:
            facets.append(f"{minimum} to {maximum}")
        elif minimum is not None:
            facets.append(f"at least {minimum}")
        elif maximum is not None:
            facets.append(f"at most {maximum}")
        if meta.get("default") is not None:
            facets.append(f"default {meta['default']}")
        line = f"  - `{key}` ({', '.join(facets)})"
        description = str(meta.get("description") or "").strip()
        if description:
            line += f" — {description}"
        lines.append(line)
    return lines


def _format_spec_list(specs: Iterable[ToolSpec]) -> str:
    """Render a compact catalog from each tool's schema authority."""
    blocks: list[str] = []
    for spec in specs:
        lines = [f"- `{spec.name}` — {spec.summary}"]
        lines.extend(_format_parameters(spec.input_schema))
        blocks.append("\n".join(lines))
    return "\n".join(blocks) if blocks else "- (none registered)"


def _evidence_fusion_example(
    *,
    tool_names: set[str],
    action_horizon: int | str,
    lidar_access: bool,
) -> str:
    """Return one contract-valid example of cross-channel route reasoning."""

    if not {"move_body", "wait"} <= tool_names:
        return ""
    dynamic_horizon = action_horizon == DYNAMIC_ACTION_HORIZON
    call_count = 1 if dynamic_horizon else int(action_horizon)
    calls = [
        {
            "tool": "move_body",
            "args": {"delta": [0.0, 0.0, 45.0]},
        }
    ]
    calls.extend(
        {"tool": "wait", "args": {"seconds": 0.5}} for _ in range(call_count - 1)
    )
    evidence = [
        "- Active plan: reach the reception desk, then ask the receptionist.",
        "- Current ego RGB: a chair blocks the space directly ahead; a broad "
        "opening is visible toward the front-left.",
    ]
    reasoning_parts = [
        "Plan: the next objective is reaching reception; asking is premature "
        "until the receptionist is visible.",
        "RGB: the chair makes the space directly ahead unusable, while the "
        "front-left opening is visible.",
    ]
    if "navigate_to_frontier" in tool_names:
        evidence.append(
            "- Exploration BEV: mapped free space appears toward the "
            "front-left opening, while an obstacle appears directly ahead."
        )
        reasoning_parts.append(
            "BEV: the mapped front-left opening supports turning to inspect "
            "it, but does not locate reception or prove the whole route."
        )
    if lidar_access:
        evidence.append("- LiDAR: 0.7 m ahead and 2.4 m at +45° left.")
        reasoning_parts.append(
            "LiDAR: the +45° return confirms substantially more clearance "
            "than straight ahead."
        )
    evidence.append(
        "- Latest feedback: the previous forward command stopped early after "
        "front contact."
    )
    evidence.append(
        "- Episode Memory: the memory ledger currently says the aisle "
        "straight ahead is open."
    )
    reasoning_parts.extend(
        (
            "Feedback: the short motion and front contact independently show that "
            "repeating forward would fail again.",
            "Tools: move_body can align the Robot with the supported opening; ask "
            "is not grounded because no Human is visible, and the current channels "
            "already support one next action.",
            "Execute: turn 45° left only, then use the fresh ego view before any "
            "translation."
            if dynamic_horizon
            else "Execute: turn 45° left only; the remaining fixed-horizon slots wait "
            "because translation requires the fresh post-turn ego view.",
        )
    )
    reasoning_parts.append(
        "Memory: current RGB and feedback contradict the stored route fact, "
        "so revise that fact instead of retaining the stale route."
    )
    decision: dict[str, Any] = {
        "thinking": " ".join(reasoning_parts),
        "actions": calls,
    }
    decision["memory"] = [
        {
            "op": "revise",
            "note": (
                "The aisle straight ahead is blocked by a chair; the "
                "front-left corridor is the currently supported route toward "
                "reception."
            ),
        }
    ]
    return "\n".join(
        (
            "### ICL example",
            "Ordinary-turn example. It illustrates the reasoning pattern only; "
            "never copy its scene facts into the current episode.",
            *evidence,
            "Example output:",
            json.dumps(decision, ensure_ascii=False, separators=(",", ":")),
        )
    )


def envelope_schema_section(
    *,
    tools: list[ToolSpec],
    requirements: DecisionRequirements,
    action_horizon: int | str = 1,
    visual_history_horizon: int = DEFAULT_VISUAL_HISTORY_FRAMES,
    lidar_access: bool = False,
    interaction_intents: list[dict[str, Any]] | None = None,
) -> str:
    """The section declaring the DecisionEnvelope JSON shape + available calls.

    Always declares ``tool_calls``; event-specific fields are added only when
    the current public task event requires them.
    """
    tool_names = {tool.name for tool in tools}
    del visual_history_horizon
    has_locomotion = bool(
        tool_names
        & {
            "move_body",
            "navigate_to_pointgoal",
            "navigate_to_frontier",
            "move_forward",
            "move_backward",
            "move_left",
            "move_right",
            "turn_left",
            "turn_right",
        }
    )
    has_object_grounding = "locate_object" in tool_names
    has_grasp = "grasp_object" in tool_names
    has_information_tool = any(tool.execution_kind == "information" for tool in tools)
    visual_label = "ego RGB"

    parts: list[str] = ["## Output format"]
    dynamic_horizon = action_horizon == DYNAMIC_ACTION_HORIZON
    call_count = MAX_ACTION_HORIZON if dynamic_horizon else int(action_horizon)
    call_word = "call" if call_count == 1 else "calls"
    output_contract = (
        "Emit exactly one JSON object, with no Markdown or prose outside it:\n"
        '{"thinking": "...", "actions": [{"tool": "...", "args": {...}}]}\n'
        "Required:\n"
        "- `thinking`: your evidence-grounded decision logic. Never keep "
        "thinking indefinitely: stop within 512 characters, then emit `actions` and "
        "close the JSON object.\n"
        "- `actions`: one or more tool calls (see the calls list below).\n"
        "Optional (omit when not applicable):\n"
        '- `subgoal`: `"checked"` when the active step of your plan is '
        "complete (one step per decision at most). The final plan step uses "
        "two-phase confirmation: after its first `checked`, follow the next "
        "Active plan prompt and choose one of its three explicit branches: "
        "emit `checked` again to confirm completion; omit `subgoal` to "
        "automatically cancel the first check and keep the final step active; "
        "or submit a new complete `plan` to cancel the first check and replace "
        "the plan.\n"
        "- `inspection`: your final inspection conclusion string, submitted "
        "once when the requested inspection is complete.\n"
    )
    output_contract += (
        '- `memory`: list of `{"op": "remember|revise|forget", '
        '"note": "..."}` — durable facts worth keeping after recent '
        "history expires. Omit when nothing needs recording.\n"
    )
    if dynamic_horizon:
        output_contract += (
            f"- Ordinary decisions contain between 1 and {call_count} "
            "ordered calls. Use fewer when the next action depends "
            "on fresh evidence; use more only when one current observation "
            "supports the whole sequence (for example, clear open space).\n"
        )
    else:
        output_contract += (
            f"- Ordinary decisions contain exactly {call_count} executable "
            f"{call_word} in order. Put every consecutive action whose effect "
            "is supported by current evidence into this one block, rather than "
            "requesting a new policy inference after each certain primitive. "
        )
        if "wait" in tool_names:
            output_contract += (
                "If fewer consequential actions are currently safe and "
                "certain, fill the remaining slots with `wait` instead of "
                "guessing. Use `wait` only for genuine uncertainty or "
                "deliberate timing: when an open route and successive motion "
                "effects are clearly supported, use those motion slots rather "
                "than defaulting to one move followed by padding waits. Near "
                "people or furniture, after contact, or whenever the next "
                "motion depends on the measured result of the first, queue "
                "only that corrective motion and use `wait` for the remaining "
                "slots. "
            )
    output_contract += (
        "- Every tool result is an ordinary turn outcome: motion, "
        "observation, semantic-state write, or failure becomes the next "
        "Current observation. Use one call when the next choice depends on "
        "that result.\n"
        "- Calls in one multi-call decision execute serially without another "
        "model inference. Each call is invoked only after the previous call "
        "finishes, using the then-current observation, pose, memory, and "
        "sensor timestamp.\n"
        "- Execution stops immediately after the first sole-call tool below: "
        "an ordinary prefix executes before it, that tool executes and returns "
        "feedback, and the remaining suffix is discarded before replanning.\n"
    )
    if has_information_tool:
        output_contract += (
            "- An information tool ends execution of the current block. An "
            "ordinary prefix may run before it; read its returned evidence in "
            "the next Current observation before choosing what follows.\n"
        )
    if has_grasp:
        output_contract += (
            "- `grasp_object` ends execution of the current block. An ordinary "
            "prefix may run before it; read its grounding and physical outcome "
            "before choosing what follows.\n"
        )
    if "report_done" in tool_names:
        output_contract += (
            "- `report_done` ends execution of the current block and the "
            "episode; if the task asked you to "
            "report a result to a named person, speak that report first.\n"
        )
    if "navigate_to_pointgoal" in tool_names:
        output_contract += (
            "- `navigate_to_pointgoal` ends execution of the current block. It "
            "returns after one physical prefix, and the next route decision "
            "uses the post-execution observation.\n"
        )
    if "navigate_to_frontier" in tool_names:
        output_contract += (
            "- `navigate_to_frontier` ends execution of the current block. Its "
            "label must appear on the CURRENT ego RGB, and labels refresh "
            "after its physical prefix returns.\n"
        )
    if "navigate_to_entity" in tool_names:
        output_contract += (
            "- `navigate_to_entity` ends execution of the current block. It "
            "follows a saved historical landmark and returns a fresh physical "
            "result after one prefix.\n"
        )
    output_contract += (
        "- Emit the event-required `plan` field below when this event asks for one."
    )
    parts.extend((output_contract, "", "### Decision checklist"))
    evidence_steps = [
        "Plan: confirm the current objective and the next unfinished plan "
        "step against public task state.",
        f"RGB: describe what the current {visual_label} actually shows, "
        "including relevant entities, openings, obstacles, and uncertainty.",
    ]
    if "navigate_to_frontier" in tool_names:
        evidence_steps.append(
            "BEV: orient using the Robot heading, compare the travelled trail "
            "and explored/unknown space, and distinguish purposeful return "
            "from repeating an unproductive search."
        )
    if has_locomotion and lidar_access:
        evidence_steps.append(
            "LiDAR: check measured clearance along the body-relative bearing "
            "of any intended translation."
        )
    evidence_steps.append(
        "Feedback: inspect the immediately preceding action result, including "
        "measured motion, contact, stability, or task-state change."
    )
    evidence_steps.extend(
        (
            "Tools: review the currently available call schemas and choose only "
            "calls whose requirements are supported by the evidence above.",
            "Execute: emit the shortest evidence-supported action sequence; stop "
            "before any action whose choice depends on a fresh observation.",
            "Memory: after choosing the calls, retain only important episode facts "
            "that must survive the bounded recent history; otherwise omit the "
            "memory field.",
        )
    )
    parts.extend(
        (
            "\n".join(
                f"{index}. {step}" for index, step in enumerate(evidence_steps, start=1)
            ),
            "",
            "### Rules",
        )
    )
    rules = [
        "- Never invent coordinates; ids and labels in task text are not poses.",
    ]
    if "report_done" in tool_names:
        rules.append(
            "- Use `report_done` only with positive evidence that every task "
            "clause is complete."
        )
    if has_locomotion:
        rules.extend(
            (
                "- Use measured motion rather than assuming the requested "
                "displacement; contact and balance can alter the trajectory.",
                "- Contact bearing is positive left and negative right; it does "
                "not prescribe the next action. Contact is evidence, not a motion "
                "veto, and never proves that a target or destination was reached; "
                "repeated contact or negligible displacement requires a materially "
                "different choice.",
                "- A crossing person must be yielded to; choose the timing or "
                "motion response from current evidence.",
            )
        )
    rules.append(
        failure_vocabulary_line(
            argument_failures=ARGUMENT_FAILURES,
            world_failures=(WORLD_FAILURES if has_object_grounding else ()),
            backend_failures=(
                BACKEND_FAILURES
                if has_object_grounding or has_grasp
                else tuple(
                    reason for reason in BACKEND_FAILURES if reason == "internal_error"
                )
            ),
        )
    )
    parts.extend(("\n".join(rules), "", "### Observation conventions"))
    parts.append(
        _interface_semantics(
            tools,
            lidar_access=lidar_access and has_locomotion,
        )
    )
    parts.extend(("", "### Calls"))
    parts.append(_format_spec_list(tools))
    if "ask" in tool_names:
        parts.extend(("", "### Available interaction intents"))
        intents = list(interaction_intents or ())
        if not intents:
            parts.append(
                "- Ask a natural-language question with `recipient` and `message`. "
                "A Human can answer only from their authorized knowledge."
            )
        else:
            parts.append(
                "When a listed topic matches, use its exact `intent_id`. "
                "For other questions, omit `intent_id` and ask in `message`. "
                "Humans may not know the answer."
            )
            for intent in intents:
                eligible = tuple(intent.get("eligible_human_ids") or ())
                recipient_text = (
                    ", ".join(str(item) for item in eligible)
                    if eligible
                    else "any currently addressable Human"
                )
                parts.append(
                    f"- `{intent['intent_id']}` — "
                    f"{intent['description']} "
                    f"Eligible recipient: {recipient_text}."
                )

    if requirements.planning is not None:
        parts.append(_planning_section(requirements.planning))
    parts.extend(
        (
            "",
            "### Optional inspection conclusion",
            'For an inspection task, add `inspection: "<your observed findings>"` '
            "once for that inspection. State the requested facts you observed and "
            "anything that remains unclear. Omit it while still investigating. "
            "The first eligible submission is final; later submissions do not "
            "replace it. This records your inspection conclusion, not a report "
            "delivered to a Human.",
        )
    )

    parts.append("")
    parts.append("### Required")
    parts.append("- JSON fields: `thinking`, `actions`.")
    parts.extend(
        (
            "- `memory` retains only concise facts needed after recent history "
            "expires. Never duplicate the task, plan, Human directory or dialogue, "
            "hands, grounding ledger, semantic landmark directory, or routine "
            "actions. Spatial facts must name a stable place/landmark/viewpoint "
            'rather than only "here"/"ahead"; changing facts must name their '
            "subject; preserve uncertainty. Omit the field when nothing needs "
            "recording. The ledger holds up to "
            f"{MAX_KEY_MEMORIES} facts, each up to "
            f"{MAX_KEY_MEMORY_CONTENT_CHARS} characters.",
            '- Memory operations write `{"op":"remember|revise|forget",'
            '"note":"..."}`. You never manage ids: `revise`/`forget` name the '
            "existing fact in the note and the harness resolves the target. Do "
            "not act merely to maintain memory.",
        )
    )

    example = _evidence_fusion_example(
        tool_names=tool_names,
        action_horizon=action_horizon,
        lidar_access=lidar_access and has_locomotion,
    )
    if example:
        parts.extend(("", example))

    final_output: dict[str, Any] = {
        "thinking": "<evidence-grounded decision>",
        "actions": [{"tool": "<tool>", "args": {}}],
    }
    if requirements.planning is not None:
        final_output["plan"] = {
            "goal": "<final outcome>",
            "steps": ["<concrete step>"],
        }
    parts.extend(
        (
            "",
            "### Final JSON output",
            "Return exactly one valid json (JSON) object in this shape, adding the optional "
            "inspection conclusion when submitting it, with no Markdown or text outside it:",
            json.dumps(final_output, ensure_ascii=False, separators=(",", ":")),
        )
    )

    return "\n".join(parts)


def _planning_section(scope: dict[str, Any]) -> str:
    """Prompt section requiring structured ``plan`` (goal + steps).

    Does NOT leak GT required_subgoals (EVALUATION.md). Only forces the output
    schema at the task event that requires an initial plan.
    """
    del scope  # GT never leaks into the planning prompt
    lines = [
        "",
        "### Required: `plan`",
        (
            "This public event requires an initial plan. You MUST include a "
            "`plan` field with your current goal and an ordered list of "
            "concrete steps (physical actions the robot can execute)."
        ),
        '- Shape: `{"goal": "<final outcome>", "steps": ["<step1>", "<step2>", ...]}`',
        "- Steps must be concrete (find / navigate / pick / handover / …), "
        'not vague phrases like "finish the task" or "somehow give …".',
        "- Do NOT invent simulator-illegal actions (teleport, mind-read, "
        "emotion-as-action).",
        "- When task facts change (for example a revised recipient or "
        "delivery place), submit a new complete `plan`; it replaces the "
        "active one and progress restarts from its first step.",
    ]
    return "\n".join(lines)


__all__ = ["envelope_schema_section"]
