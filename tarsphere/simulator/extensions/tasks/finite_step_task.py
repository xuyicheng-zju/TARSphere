from tarsphere.simulator.core.scene.scene import IScene
from tarsphere.simulator.core.task import BaseTask
from tarsphere.simulator.extensions.configs.tasks.finite_step_task import (
    FiniteStepTaskCfg,
)


@BaseTask.register("FiniteStepTask")
class FiniteStepTask(BaseTask):
    def __init__(self, config: FiniteStepTaskCfg, scene: IScene):
        super().__init__(config, scene)
        self.stop_count = 1
        self.max_steps = config.max_steps

    def is_done(self) -> bool:
        self.stop_count += 1
        return self.stop_count > self.max_steps
