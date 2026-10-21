"""Runtime integration boundaries.

``episode`` owns orchestration and Human event release, ``simulation`` owns
scene/server integration, ``control`` owns navigation and body execution,
``perception`` owns runtime sensor processing, and ``manipulation`` owns grasp,
handover, and articulated-object execution.
"""
