from tarsphere.simulator.core.config import TaskCfg
from tarsphere.simulator.core.scene.scene import IScene
from tarsphere.simulator.core.task import BaseTask


@BaseTask.register("SingleInferenceTask")
class SimpleInferenceTask(BaseTask):
    def __init__(self, config: TaskCfg, scene: IScene):
        super().__init__(config, scene)

    def calculate_metrics(self) -> dict:
        pass

    def is_done(self) -> bool:
        return False
