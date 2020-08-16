from ray.tune import registry

from envs.procgen_env_wrapper import ProcgenEnvWrapper

import numpy as np
from gym import ObservationWrapper

class Flip(ObservationWrapper):
    def step(self, action):
        obs, reward, done, obs = self.env.step(action)
        return observation(observation), reward, done, info

    def observation(self, obs):
        return np.flip(obs, 2)

# Register Env in Ray
registry.register_env(
    "flip_procgen_env",  # This should be different from procgen_env_wrapper
    lambda config: Flip(ProcgenEnvWrapper(config)),
)
