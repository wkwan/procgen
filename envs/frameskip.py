from ray.tune import registry

from envs.procgen_env_wrapper import ProcgenEnvWrapper

import numpy as np
from gym import ActionWrapper

class FrameSkip(ActionWrapper):
    def step(self, action):
        self.env.step(self.action(action))
        return self.env.step(self.action(action))

    def action(self, action):
        return action

# Register Env in Ray
registry.register_env(
    "frameskip_procgen_env",  # This should be different from procgen_env_wrapper
    lambda config: FrameSkip(ProcgenEnvWrapper(config)),
)
