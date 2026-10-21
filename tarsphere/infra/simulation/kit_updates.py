"""Refresh Kit/animation without advancing the robot's physics clock."""

from __future__ import annotations


def pump_kit_updates(*, count: int = 1) -> None:
    """Use Isaac's render-only update contract, preserving the caller setting.

    The timeline stays playing so AnimGraph and Replicator remain attached.
    Only the environment's explicit physics steps may advance PhysX.
    """
    import carb.settings
    import omni.kit.app

    settings = carb.settings.get_settings()
    path = "/app/player/playSimulations"
    previous = settings.get(path)
    app = omni.kit.app.get_app()
    settings.set_bool(path, False)
    try:
        for _ in range(count):
            app.update()
    finally:
        settings.set(path, previous)
