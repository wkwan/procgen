from ray.tune import registry

from envs.procgen_env_wrapper import ProcgenEnvWrapper

import numpy as np
from gym import ObservationWrapper

class Flip(ObservationWrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def observation(self, obs):
        # print("do the flip")
        print("before flip", obs)
        flipped = np.flipud(obs)
        print("after flip", flipped)
        return flipped

# Register Env in Ray
registry.register_env(
    "flip_procgen_env",  # This should be different from procgen_env_wrapper
    lambda config: Flip(ProcgenEnvWrapper(config)),
)
