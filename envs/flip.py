from ray.tune import registry

from envs.procgen_env_wrapper import ProcgenEnvWrapper

import numpy as np
from gym import ObservationWrapper

class Flip(ObservationWrapper):
    def __init__(self, env_wrapper):
        super().__init__(env_wrapper)
        self.prev_obs = None
        self.prev_reward = None
        self.prev_done = None
        self.prev_info = None

    def step(self, action):
        if self.prev_obs is not None:
            print("do flip")
            obs = np.flipud(self.prev_obs)
            reward = self.prev_reward
            done = self.prev_done
            info = self.prev_info
            self.prev_obs = None
            self.prev_reward = None
            self.prev_done = None
            self.prev_info = None
            return obs, reward, done, info
        obs, reward, done, info = self.env.step(action)
        print("don't do flip")
        self.prev_obs = obs
        self.prev_reward = reward
        self.prev_done = done
        self.prev_info = info
        return obs, reward, done, info

    def observation(self, obs):
        return obs

# Register Env in Ray
registry.register_env(
    "flip_procgen_env",  # This should be different from procgen_env_wrapper
    lambda config: Flip(ProcgenEnvWrapper(config)),
)
