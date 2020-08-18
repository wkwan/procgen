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
            
            #FLIP
            # obs = np.flipud(self.prev_obs)

            #CUTOUT
            # box_min = 7
            # box_max = 22
            # pivot_h = 12
            # pivot_w = 24
            # w1 = np.random.randint(box_min, box_max)
            # h1 = np.random.randint(box_min, box_max)

            # self.prev_obs[pivot_h+h1:pivot_h+h1+h1, 
            #         pivot_w+w1:pivot_w+w1+w1, :] = 0


            #RAND CROP

            h_start = np.random.randint(0, 33)
            w_start = np.random.randint(0, 33)

            cropped_img = np.zeros((64, 64, 3))
            cropped_img[h_start:h_start + 32, w_start:w_start + 32, :] = self.prev_obs[h_start:h_start + 32, w_start:w_start + 32, :]

            # self.prev_obs = np.transpose(self.prev_obs, (1, 0, 2))

            # crop_size = 64
            # crop_max = 75 - crop_size
            # w1 = np.random.randint(0, crop_max)
            # h1 = np.random.randint(0, crop_max)

            # print("the crop vars", w1, h1, self.prev_obs.shape)

            # # creates all sliding windows combinations of size (output_size)
            # windows = view_as_windows(
            #     self.prev_obs, (crop_size, crop_size, 1))
            # # selects a random window for each batch element
            # print("windows shape", windows.shape)
            # cropped_img = windows[...,:, w1, h1, 0]
            # cropped_img = np.swapaxes(cropped_img,0,2)

            obs = cropped_img
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
