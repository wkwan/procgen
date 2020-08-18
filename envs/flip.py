from ray.tune import registry

from envs.procgen_env_wrapper import ProcgenEnvWrapper

import numpy as np
from gym import ObservationWrapper

class Cutout(object):
    def __init__(self, 
                 box_min=7, 
                 box_max=22, 
                 pivot_h=12, 
                 pivot_w=24):
        
        self.box_min = box_min
        self.box_max = box_max
        self.pivot_h = pivot_h
        self.pivot_w = pivot_w
        self.w1 = np.random.randint(self.box_min, self.box_max)
        self.h1 = np.random.randint(self.box_min, self.box_max)
        
    def do_augmentation(self, img):
        cut_img = img.copy()
        for (w11, h11) in np.array(list(zip(self.w1,self.w2))):
            cut_img[self.pivot_h+h11:self.pivot_h+h11+h11, 
                    self.pivot_w+w11:self.pivot_w+w11+w11, :] = 0
        return cut_img
    
    def change_randomization_params(self, index_):
        self.w1[index_] = np.random.randint(self.box_min, self.box_max)
        self.h1[index_] = np.random.randint(self.box_min, self.box_max)

    def change_randomization_params_all(self):
        self.w1 = np.random.randint(self.box_min, self.box_max)
        self.h1 = np.random.randint(self.box_min, self.box_max)

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
            cutout_maker = Cutout()
            obs = cutout_maker.do_augmentation(self.prev_obs)
            print("did obs", obs)
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
