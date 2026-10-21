"""Evaluated-agent tool surface: specs, handlers, registry, and availability.

``register_atomic_tools`` (see ``atomic.py``) builds the whole callable
surface the agent sees, from the individual ``ToolSpec`` constants in
``specs.py``. There is no second call kind and no aggregate spec bundle: one
registry holds every call, and a capability whose execution takes several
motions still costs the agent one call.
"""

from tarsphere.harness.tools.registry import ToolRegistry

__all__ = [
    "ToolRegistry",
]
