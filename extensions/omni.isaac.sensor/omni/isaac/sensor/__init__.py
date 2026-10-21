"""No-op replacement for Isaac's unused deprecated sensor compatibility API.

TARSphere uses Replicator RGB-D and PhysX contact reports directly. Importing classes
from this deprecated namespace is therefore an integration error rather than a
supported fallback.
"""

__all__: list[str] = []
