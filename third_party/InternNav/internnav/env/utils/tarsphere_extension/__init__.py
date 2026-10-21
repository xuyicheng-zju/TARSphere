from tarsphere.simulator.extensions import import_extensions as import_tarsphere_extension


def import_extensions():
    import_tarsphere_extension()
    from . import controllers, robots, sensors, tasks  # noqa: F401
    from .metrics.vln_pe_metrics import VLNPEMetrics
    from .tasks.vln_eval_task import VLNEvalTask  # noqa: F401
