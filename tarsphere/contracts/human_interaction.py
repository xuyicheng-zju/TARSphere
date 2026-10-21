"""Public Human-interaction geometry shared by compiler and runtimes."""

from __future__ import annotations


# Public speech is audible inside one Euclidean XY radius.  Human-to-Agent
# delivery uses this distance only: a Human does not disappear merely because
# the Robot camera is facing elsewhere.  Agent-to-Human interaction additionally
# requires current named-Human RGB-D visibility at the runtime boundary.
# Communication must never depend on a room id, route region, or which side of
# an annotation boundary owns either actor.
HUMAN_COMMUNICATION_RADIUS_M = 2.5


__all__ = ["HUMAN_COMMUNICATION_RADIUS_M"]
