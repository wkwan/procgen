from ray.tune import registry

from envs.procgen_env_wrapper import ProcgenEnvWrapper

import numpy as np
from gym import ObservationWrapper

class Flip(ObservationWrapper):

    prev_obs = None
    prev_reward = None
    prev_done = None
    prev_info = None

    def step(self, action):
        if prev_obs is not None:
            obs = np.flipud(obs)
            reward = prev_reward
            done = prev_done
            info = prev_info
            prev_obs = None
            prev_reward = None
            prev_done = None
            prev_info = None
            return obs, reward, done, info
        obs, reward, done, info = self.env.step(action)
        prev_obs = obs
        prev_reward = reward
        prev_done = done
        prev_info = info
        return obs, reward, done, info

    def observation(self, obs):
        return obs

# Register Env in Ray
registry.register_env(
    "flip_procgen_env",  # This should be different from procgen_env_wrapper
    lambda config: Flip(ProcgenEnvWrapper(config)),
)
