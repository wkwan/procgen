from ray.tune import registry
from gym.wrappers import FrameStack

from envs.procgen_env_wrapper import ProcgenEnvWrapper

import numpy as np
from gym import ActionWrapper

class FrameSkip(ActionWrapper, FrameStack):

    def step(self, action):
        self.env.step(self.action(action))
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self._get_observation(), reward, done, info

    def action(self, action):
        return action

# Register Env in Ray
registry.register_env(
    "framestack_frameskip_procgen_env",  # This should be different from procgen_env_wrapper
    lambda config: FrameStack(ProcgenEnvWrapper(config), 4),
)
