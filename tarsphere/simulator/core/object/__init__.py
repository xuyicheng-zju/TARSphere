from collections import OrderedDict

from tarsphere.simulator.core.config import TaskCfg
from tarsphere.simulator.core.object.object import BaseObject, create_objects
from tarsphere.simulator.core.scene.scene import IScene


def init_objects(task_config: TaskCfg, scene: IScene) -> OrderedDict[str, BaseObject]:
    return create_objects(task_config, scene)
