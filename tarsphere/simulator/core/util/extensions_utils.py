def dump_extensions():

    extensions = {}
    from tarsphere.simulator.core.robot.controller import BaseController

    extensions["controllers"] = BaseController.controllers
    from tarsphere.simulator.core.util.interaction import BaseInteraction

    extensions["interactions"] = BaseInteraction.interactions
    from tarsphere.simulator.core.task.metric import BaseMetric

    extensions["metrics"] = BaseMetric.metrics
    from tarsphere.simulator.core.object.object import BaseObject

    extensions["objs"] = BaseObject.objs
    from tarsphere.simulator.core.robot.robot import BaseRobot

    extensions["robots"] = BaseRobot.robots
    from tarsphere.simulator.core.sensor.sensor import BaseSensor

    extensions["sensors"] = BaseSensor.sensors
    from tarsphere.simulator.core.task import BaseTask

    extensions["tasks"] = BaseTask.tasks
    return extensions


def reload_extensions(extensions):

    from tarsphere.simulator.core.robot.controller import BaseController

    BaseController.controllers = extensions["controllers"]
    from tarsphere.simulator.core.util.interaction import BaseInteraction

    BaseInteraction.interactions = extensions["interactions"]
    from tarsphere.simulator.core.task.metric import BaseMetric

    BaseMetric.metrics = extensions["metrics"]
    from tarsphere.simulator.core.object.object import BaseObject

    BaseObject.objs = extensions["objs"]
    from tarsphere.simulator.core.robot.robot import BaseRobot

    BaseRobot.robots = extensions["robots"]
    from tarsphere.simulator.core.sensor.sensor import BaseSensor

    BaseSensor.sensors = extensions["sensors"]
    from tarsphere.simulator.core.task import BaseTask

    BaseTask.tasks = extensions["tasks"]
