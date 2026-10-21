"""No-op Kit extension: satisfies dependency edges only."""

from __future__ import annotations

import omni


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        return

    def on_shutdown(self) -> None:
        return
