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
            
            # obs = np.flipud(self.prev_obs)
            box_min = 7
            box_max = 22
            pivot_h = 12
            pivot_w = 24
            w1 = np.random.randint(box_min, box_max)
            h1 = np.random.randint(box_min, box_max)

            self.prev_obs[pivot_h+h1:pivot_h+h1+h1, 
                    pivot_w+w1:pivot_w+w1+w1, :] = 0
            obs = self.prev_obs
            reward = self.prev_reward
            done = self.prev_done
            info = self.prev_info
            self.prev_obs = None
            self.prev_reward = None
            self.prev_done = None
            self.prev_info = None
            return obs, reward, done, info
        obs, reward, done, info = self.env.step(action)
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
