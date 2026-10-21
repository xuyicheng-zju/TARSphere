"""The closed vocabulary of reasons a tool call can fail.

Every failed call tells the agent two things: prose in ``content`` it can read,
and one word in ``metadata['error_type']`` it can branch on. This module owns
that word.

It exists because the word used to be ``type(exc).__name__``. That made the
vocabulary open — the same "I could not resolve your target" outcome arrived as
``TargetNotGrounded`` from one handler, ``ValueError`` from the next, and
``KeyError`` from a third, depending on which layer happened to raise. An open
vocabulary is a measurement problem, not just an untidy one: Adaptation asks
whether the agent changed course *because of* what execution told it, and that
question is only answerable if "what execution told it" is drawn from a fixed
set. It is also a prompt problem, since the set is shown to the model
(``failure_vocabulary_line``) so a failure reason it reads is one it was told to
expect.

The words are grouped by what the agent should do about them, because that is
the only distinction it can act on:

- **Argument-level** — the call could not be attempted as written. Nothing in
  the world needs to change; the agent must fix the call or gather the missing
  fact first.
- **World-level** — the call was attempted and the world refused it. Repeating
  it unchanged will refuse again; something about the situation has to move.
- **Backend-level** — the capability is not available in this run. The runner
  invalidates the attempt; this must not be scored as an agent decision or
  disguised as world evidence.

Every word here has a producer. A reason no handler can emit would still be
advertised to the agent by :func:`failure_vocabulary_line`, which teaches it to
expect something that never arrives; ``tests/test_failure_vocabulary.py`` holds
that line.
"""

from __future__ import annotations

from typing import Any, Final

from tarsphere.harness.core.types import ToolCall, ToolResult, tool_turn_action


class FailureReason:
    """The word a failed call reports in ``metadata['error_type']``."""

    # -- Argument-level: fix the call, or gather the missing fact first. -----
    # A required argument was absent or empty.
    MISSING_ARGUMENT: Final = "missing_argument"
    # An argument was present but the wrong type or shape.
    MALFORMED_ARGUMENT: Final = "malformed_argument"
    # A well-formed argument named something this episode does not contain: a
    # command outside the enum, a room absent from the floor plan, a person
    # outside the cast.
    UNKNOWN_VALUE: Final = "unknown_value"

    # -- World-level: the world refused; the situation has to move. ---------
    # The grounder ran and returned no instance for the phrase in the current
    # ego view — a missed detection is possible, not proof of absence.
    NOT_IN_VIEW: Final = "not_in_view"
    # A box was detected but depth could not lift it into the world frame.
    DEPTH_UNAVAILABLE: Final = "depth_unavailable"
    # Current target geometry has no collision-free, gripper-compatible grasp.
    NO_GRASP_CANDIDATE: Final = "no_grasp_candidate"
    # Same-capture target geometry is beyond the embodiment's absolute reach;
    # the robot must move before pose generation can be useful.
    OUT_OF_REACH: Final = "out_of_reach"
    # Confirmation refers to visual evidence from a different camera/world
    # capture. The Agent must request and inspect a new preview.
    STALE_EVIDENCE: Final = "stale_evidence"
    # A previously working episode component is unavailable only for a
    # bounded simulation-time window. Unlike BACKEND_UNAVAILABLE it recovers.
    TEMPORARILY_UNAVAILABLE: Final = "temporarily_unavailable"
    # -- Backend-level: route around the missing capability. -----------------
    # The capability's sensor or backend is not available in this run.
    BACKEND_UNAVAILABLE: Final = "backend_unavailable"
    # A defect, not a decision: an unexpected exception crossed the handler.
    # Kept distinct so a bug is never reported to the agent as a world state it
    # could have avoided.
    INTERNAL_ERROR: Final = "internal_error"


ARGUMENT_FAILURES: Final[tuple[str, ...]] = (
    FailureReason.MISSING_ARGUMENT,
    FailureReason.MALFORMED_ARGUMENT,
    FailureReason.UNKNOWN_VALUE,
)

WORLD_FAILURES: Final[tuple[str, ...]] = (
    FailureReason.NOT_IN_VIEW,
    FailureReason.DEPTH_UNAVAILABLE,
    FailureReason.NO_GRASP_CANDIDATE,
    FailureReason.OUT_OF_REACH,
    FailureReason.STALE_EVIDENCE,
)

BACKEND_FAILURES: Final[tuple[str, ...]] = (
    FailureReason.BACKEND_UNAVAILABLE,
    FailureReason.INTERNAL_ERROR,
)

TRANSIENT_FAILURES: Final[tuple[str, ...]] = (FailureReason.TEMPORARILY_UNAVAILABLE,)

ALL_FAILURE_REASONS: Final[tuple[str, ...]] = (
    ARGUMENT_FAILURES + WORLD_FAILURES + BACKEND_FAILURES + TRANSIENT_FAILURES
)


def require_failure_reason(reason: str) -> str:
    """Return ``reason`` if it is in the vocabulary, else raise.

    Handlers call this on the way out so a reason invented at the call site
    fails here — inside the process that produced it — rather than reaching the
    agent as a word it was never told to expect, or reaching the trace as an
    unscoreable one.
    """
    if reason not in ALL_FAILURE_REASONS:
        raise ValueError(
            f"{reason!r} is not a failure reason. The vocabulary is closed; "
            f"use one of {list(ALL_FAILURE_REASONS)} or add the new reason to "
            f"FailureReason and to the prompt line that shows it to the agent."
        )
    return reason


def failure_vocabulary_line(
    *,
    argument_failures: tuple[str, ...] = ARGUMENT_FAILURES,
    world_failures: tuple[str, ...] = WORLD_FAILURES,
    backend_failures: tuple[str, ...] = BACKEND_FAILURES,
    transient_failures: tuple[str, ...] = TRANSIENT_FAILURES,
) -> str:
    """The prompt sentence naming every failure reason the agent may receive.

    Rendered into the general rules so the set the agent reads is the set the
    handlers can produce. Derived from the tuples above rather than restated, so
    a new reason cannot be added without the prompt gaining it too.
    """
    clauses: list[str] = []
    if argument_failures:
        clauses.append(
            "fix the call for "
            + ", ".join(f"`{reason}`" for reason in argument_failures)
        )
    if world_failures:
        clauses.append(
            "change the situation for "
            + ", ".join(f"`{reason}`" for reason in world_failures)
        )
    if backend_failures:
        clauses.append(
            "do not infer world state from "
            + ", ".join(f"`{reason}`" for reason in backend_failures)
        )
    if transient_failures:
        clauses.append(
            "a temporary component condition for "
            + ", ".join(f"`{reason}`" for reason in transient_failures)
        )
    guidance = "; ".join(clauses)
    guidance = guidance[:1].upper() + guidance[1:]
    return "- When a call fails it names one reason. " + guidance + "."


# ---------------------------------------------------------------------------
# The only way to build a failed result
# ---------------------------------------------------------------------------
# Both builders live here rather than beside their handlers so a failed call
# cannot be constructed without passing through require_failure_reason. When
# each handler formatted its own metadata dict, several failure paths shipped
# prose with no error_type at all.


def tool_failure(
    call: ToolCall,
    reason: str,
    content: str,
    *,
    images: tuple[str, ...] = (),
    **metadata: Any,
) -> ToolResult:
    """A failed tool call: identity payload, prose to read, one reason to branch on."""
    return ToolResult(
        call_id=call.call_id,
        name=call.tool,
        action=tool_turn_action(call.tool),
        content=content,
        is_error=True,
        metadata={"error_type": require_failure_reason(reason), **metadata},
        images=images,
    )


def unexpected_tool_failure(call: ToolCall, exc: Exception) -> ToolResult:
    """Report an exception the tool handler did not anticipate.

    Reported as ``internal_error`` rather than mapped onto a world-state reason:
    a defect in our code must not reach the agent disguised as a situation it
    could have avoided, or the Adaptation score would credit it for working
    around our bug. The exception type stays in the metadata for the trace.
    """
    return tool_failure(
        call,
        FailureReason.INTERNAL_ERROR,
        f"{call.tool} failed: {type(exc).__name__}: {exc}",
        exception_type=type(exc).__name__,
    )


__all__ = [
    "FailureReason",
    "ARGUMENT_FAILURES",
    "WORLD_FAILURES",
    "BACKEND_FAILURES",
    "TRANSIENT_FAILURES",
    "ALL_FAILURE_REASONS",
    "require_failure_reason",
    "failure_vocabulary_line",
    "tool_failure",
    "unexpected_tool_failure",
]
