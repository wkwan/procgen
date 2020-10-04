from ray.tune import registry

from envs.procgen_env_wrapper import ProcgenEnvWrapper

import numpy as np
from gym import ObservationWrapper

import torch
import torch.nn as nn
import kornia 

import imageio



class Crop(ObservationWrapper):
    def __init__(self, env_wrapper, is_rollout):
        super().__init__(env_wrapper)
        self.prev_obs = None
        self.prev_reward = None
        self.prev_done = None
        self.prev_info = None
        self.is_rollout = is_rollout

    def step(self, action):
        if not self.is_rollout and self.prev_obs is not None:
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
            imageio.imwrite('/home/ubuntu/procgen-competition/bcropstepbefore.png', self.prev_obs)

            self.prev_obs = torch.from_numpy(self.prev_obs).float()

            self.prev_obs = self.prev_obs.permute(2,0,1)
            aug_trans = nn.Sequential(nn.ReplicationPad2d((12,12)),
                                kornia.augmentation.RandomCrop((64, 64)))
            self.prev_obs = aug_trans(self.prev_obs)
            self.prev_obs = self.prev_obs[0].permute(1,2,0)

            self.prev_obs = self.prev_obs.detach().numpy().astype(np.uint8)

            imageio.imwrite('/home/ubuntu/procgen-competition/bcropstep.png', self.prev_obs)


            obs = self.prev_obs
            reward = self.prev_reward
            done = self.prev_done
            info = self.prev_info

            self.prev_obs = None 
            self.prev_reward = None
            self.prev_done = None
            self.prev_info = None

            # h_start = np.random.randint(0, 33)
            # w_start = np.random.randint(0, 33)

            # cropped_img = np.zeros((64, 64, 3))
            # cropped_img[h_start:h_start + 32, w_start:w_start + 32, :] = self.prev_obs[h_start:h_start + 32, w_start:w_start + 32, :]
            
            # obs = cropped_img
            # reward = self.prev_reward
            # done = self.prev_done
            # info = self.prev_info
            # self.prev_obs = None
            # self.prev_reward = None
            # self.prev_done = None
            # self.prev_info = None
            return obs, reward, done, info
        obs, reward, done, info = self.env.step(action)
        self.prev_obs = obs
        self.prev_reward = reward
        self.prev_done = done
        self.prev_info = info
        return obs, reward, done, info

    def observation(self, obs):
        return obs

def create_my_custom_env(config):
    # print("IS ROLLOUT", config["rollout"] if ("rollout" in config) else False)
    return Crop(ProcgenEnvWrapper(config), config["rollout"] if ("rollout" in config) else False)

# Register Env in Ray
registry.register_env(
    "crop_procgen_env",  # This should be different from procgen_env_wrapper
    create_my_custom_env
)
