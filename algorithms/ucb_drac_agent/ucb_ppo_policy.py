import logging

import ray
from ray.rllib.agents.a3c.a3c_torch_policy import apply_grad_clipping
from ray.rllib.agents.ppo.ppo_tf_policy import postprocess_ppo_gae, \
    setup_config
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from .torch_policy import EntropyCoeffSchedule, LearningRateSchedule
from .torch_policy_template import build_torch_policy
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.torch_ops import sequence_mask
from ray.rllib.utils import try_import_torch
from torchvision.utils import save_image
import imageio

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn as nn
import numbers
import random
import time
import kornia 

from collections import deque


class Grayscale(object):
    """
    Grayscale Augmentation
    """
    def __init__(self,  
                 batch_size, 
                 *_args, 
                 **_kwargs):
        
        self.batch_size = batch_size
        self.transform = kornia.color.gray.RgbToGrayscale()
        
    def do_augmentation(self, x):
        self.batch_size = x.shape[0]
        x_copy = x.clone()
        x_copy = x_copy.permute(0,3,1,2)
        x_copy = self.transform(x_copy)
        x_copy = x_copy.repeat([1,3,1,1])
        x_copy = x_copy.permute(0,2,3,1)
        # imageio.imwrite('/home/ubuntu/procgen-competition/grayscale.png', x_copy[0].cpu().numpy())
        return x_copy

    def change_randomization_params(self, index_):
        pass

    def change_randomization_params_all(self):
        pass

    def print_parms(self):
        pass
        
        
class Cutout(object):
    """
    Cutout Augmentation
    """
    def __init__(self, 
                 batch_size, 
                 box_min=7, 
                 box_max=22, 
                 pivot_h=12, 
                 pivot_w=24,
                 *_args, 
                 **_kwargs):
        
        self.box_min = box_min
        self.box_max = box_max
        self.pivot_h = pivot_h
        self.pivot_w = pivot_w
        self.batch_size = batch_size
        self.w1 = np.random.randint(self.box_min, self.box_max, batch_size)
        self.h1 = np.random.randint(self.box_min, self.box_max, batch_size)
        
    def do_augmentation(self, imgs):
        # imgs = imgs.transpose(0,3,1,2)
        imgs = imgs.permute(0,3,1,2)
        n, c, h, w = imgs.shape
        # print("imgs shape", imgs.shape)
        self.batch_size = n
        self.change_randomization_params_all()
        # cutouts = torch.empty((n, c, h, w), dtype=imgs.dtype, device=imgs.device)
        cutouts = torch.empty((n, c, h, w))

        for i, (img, w11, h11) in enumerate(zip(imgs, self.w1, self.h1)):
            cut_img = img.clone()

            cut_img[:, 
                    self.pivot_h+h11:self.pivot_h+h11+h11, 
                    self.pivot_w+w11:self.pivot_w+w11+w11] = 0
            cutouts[i] = cut_img
        # cutouts = cutouts.transpose(0,2,3,1)
        cutouts = cutouts.permute(0,2,3,1)
        # imageio.imwrite('/home/ubuntu/procgen-competition/cutout.png', cutouts[0].cpu().numpy())
        return cutouts
    
    def change_randomization_params(self, index_):
        self.w1[index_] = np.random.randint(self.box_min, self.box_max)
        self.h1[index_] = np.random.randint(self.box_min, self.box_max)

    def change_randomization_params_all(self):
        self.w1 = np.random.randint(self.box_min, self.box_max, self.batch_size)
        self.h1 = np.random.randint(self.box_min, self.box_max, self.batch_size)
        
    def print_parms(self):
        print(self.w1)
        print(self.h1)
        
        
class CutoutColor(object):
    """
    Cutout-Color Augmentation
    """
    def __init__(self, 
                 batch_size, 
                 box_min=7, 
                 box_max=22, 
                 pivot_h=12, 
                 pivot_w=24, 
                 obs_dtype='uint8', 
                 *_args, 
                 **_kwargs):
        
        self.box_min = box_min
        self.box_max = box_max
        self.pivot_h = pivot_h
        self.pivot_w = pivot_w
        self.batch_size = batch_size
        self.w1 = np.random.randint(self.box_min, self.box_max, batch_size)
        self.h1 = np.random.randint(self.box_min, self.box_max, batch_size)
        self.rand_box = np.random.randint(0, 255, size=(batch_size, 1, 1, 3), dtype=obs_dtype) / 255.
        self.obs_dtype = obs_dtype
        
    def do_augmentation(self, imgs):
        device = imgs.device
        imgs = imgs.cpu().numpy()
        imgs = imgs.transpose(0,3,1,2)
        n, c, h, w = imgs.shape
        self.batch_size = n
        self.change_randomization_params_all()

        # cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
        cutouts = np.empty((n, c, h, w))

        for i, (img, w11, h11) in enumerate(zip(imgs, self.w1, self.h1)):
            cut_img = img.copy()
            cut_img[:, self.pivot_h+h11:self.pivot_h+h11+h11, 
                       self.pivot_w+w11:self.pivot_w+w11+w11] = np.tile(self.rand_box[i].reshape(-1, 1, 1), 
                (1,) + cut_img[:, self.pivot_h+h11:self.pivot_h+h11+h11, self.pivot_w+w11:self.pivot_w+w11+w11].shape[1:])
            cutouts[i] = cut_img
        cutouts = cutouts.transpose(0,2,3,1)
        cutouts = torch.tensor(cutouts, device=device)
        # imageio.imwrite('/home/ubuntu/procgen-competition/cutoutcolor.png', cutouts[0].cpu().numpy())
        return cutouts
        
    def change_randomization_params(self, index_):
        self.w1[index_] = np.random.randint(self.box_min, self.box_max)
        self.h1[index_] = np.random.randint(self.box_min, self.box_max)
        self.rand_box[index_] = np.random.randint(0, 255, size=(1, 1, 1, 3), dtype=self.obs_dtype)

    def change_randomization_params_all(self):
        self.w1 = np.random.randint(self.box_min, self.box_max, self.batch_size)
        self.h1 = np.random.randint(self.box_min, self.box_max, self.batch_size)
        self.rand_box = np.random.randint(0, 255, size=(self.batch_size, 1, 1, 3), dtype=self.obs_dtype)
        
    def print_parms(self):
        print(self.w1)
        print(self.h1)
        

class Flip(object):
    """
    Flip Augmentation
    """
    def __init__(self,  
                 batch_size, 
                 p_rand=0.5,
                 *_args, 
                 **_kwargs):
        
        self.p_flip = p_rand
        self.batch_size = batch_size
        self.random_inds = np.random.choice([True, False], 
                                            batch_size, 
                                            p=[self.p_flip, 1 - self.p_flip])
        
    def do_augmentation(self, images):
        device = images.device
        images = images.cpu().numpy()
        self.batch_size = images.shape[0]
        self.change_randomization_params_all()
        if self.random_inds.sum() > 0:
            images = images.transpose(0,3,1,2)
            images[self.random_inds] = np.flip(images[self.random_inds], 2)
            images = images.transpose(0,2,3,1)
        images = torch.tensor(images, device=device)
        # if self.random_inds.sum() > 0:
        #     imageio.imwrite('/home/ubuntu/procgen-competition/flip.png', images[self.random_inds][0].cpu().numpy())
        return images
    
    def change_randomization_params(self, index_):
        self.random_inds[index_] = np.random.choice([True, False], 1, 
                                                    p=[self.p_flip, 1 - self.p_flip])

    def change_randomization_params_all(self):
        self.random_inds = np.random.choice([True, False], 
                                            self.batch_size, 
                                            p=[self.p_flip, 1 - self.p_flip])
        
    def print_parms(self):
        print(self.random_inds)
        

class Rotate(object):
    """
    Rotate Augmentation
    """
    def __init__(self,  
                 batch_size, 
                 *_args, 
                 **_kwargs):
        
        self.batch_size = batch_size
        self.random_inds = np.random.randint(4, size=batch_size) * batch_size + np.arange(batch_size)
        
    def do_augmentation(self, imgs):
        self.batch_size = imgs.shape[0]
        # imgs = imgs.permute(0,3,1,2)
        self.change_randomization_params_all()
        tot_imgs = imgs
        for k in range(3):
            rot_imgs = np.ascontiguousarray(np.rot90(imgs,k=(k+1),axes=(1, 2)))
            tot_imgs = np.concatenate((tot_imgs, rot_imgs), 0)
        # imageio.imwrite('/home/ubuntu/procgen-competition/rotate.png', tot_imgs[self.random_inds][0].cpu().detach().numpy())
        images = torch.tensor(tot_imgs[self.random_inds])
        return images
    
    def change_randomization_params(self, index_):
        temp = np.random.randint(4)            
        self.random_inds[index_] = index_ + temp * self.batch_size
        
    def change_randomization_params_all(self):
        self.random_inds = np.random.randint(4, size=self.batch_size) * self.batch_size + np.arange(self.batch_size)
        
    def print_parms(self):
        print(self.random_inds)
        

class Crop(object):
    """
    Crop Augmentation
    """
    def __init__(self,  
                 batch_size, 
                 *_args, 
                 **_kwargs):
        self.batch_size = batch_size 

    def do_augmentation(self, x):
        x = x.clone()
        x = x.permute(0,3,1,2).to(device=x.device, dtype=torch.float32)
        aug_trans = nn.Sequential(nn.ReplicationPad2d(12),
                            kornia.augmentation.RandomCrop((64, 64)))
        x = aug_trans(x)
        x = x.permute(0,2,3,1)
        # imageio.imwrite('/home/ubuntu/procgen-competition/crop.png', x[0].cpu().numpy())
        return x

    def change_randomization_params(self, index_):
        pass

    def change_randomization_params_all(self):
        pass

    def print_parms(self):
        pass


class RandomConv(object):
    """
    Random-Conv Augmentation
    """
    def __init__(self,  
                batch_size, 
                *_args, 
                **_kwargs):
        self.batch_size = batch_size 
        
    def do_augmentation(self, x):
        _device = x.device
        x = x.clone()
        x = x.permute(0,3,1,2).to(_device, dtype=torch.float32)
        
        img_h, img_w = x.shape[2], x.shape[3]
        num_stack_channel = x.shape[1]
        num_batch = x.shape[0]
        num_trans = num_batch
        batch_size = int(num_batch / num_trans)
        
        # initialize random covolution
        rand_conv = nn.Conv2d(3, 3, kernel_size=3, bias=False, padding=1).to(_device)
        
        for trans_index in range(num_trans):
            torch.nn.init.xavier_normal_(rand_conv.weight.data)
            temp_x = x[trans_index*batch_size:(trans_index+1)*batch_size]
            temp_x = temp_x.reshape(-1, 3, img_h, img_w) # (batch x stack, channel, h, w)
            rand_out = rand_conv(temp_x)
            if trans_index == 0:
                total_out = rand_out
            else:
                total_out = torch.cat((total_out, rand_out), 0)
        total_out = total_out.reshape(-1, num_stack_channel, img_h, img_w)
        total_out = total_out.permute(0,2,3,1)
        # imageio.imwrite('/home/ubuntu/procgen-competition/randomconv.png', total_out[0].cpu().detach().numpy())
        return total_out

    def change_randomization_params(self, index_):
        pass

    def change_randomization_params_all(self):
        pass

    def print_parms(self):
        pass

        
class ColorJitter(nn.Module):
    """
    Color-Jitter Augmentation
    """
    def __init__(self, 
                 batch_size,
                 brightness=0.4,                              
                 contrast=0.4,
                 saturation=0.4, 
                 hue=0.5,
                 p_rand=1.0,
                 stack_size=1, 
                 *_args,
                 **_kwargs):
        super(ColorJitter, self).__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        
        self.prob = p_rand
        self.batch_size = batch_size
        self.stack_size = stack_size
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        # random paramters
        factor_contrast = torch.empty(self.batch_size, device=self._device).uniform_(*self.contrast)
        self.factor_contrast = factor_contrast.reshape(-1,1).repeat(1, self.stack_size).reshape(-1)
        
        factor_hue = torch.empty(self.batch_size, device=self._device).uniform_(*self.hue)
        self.factor_hue = factor_hue.reshape(-1,1).repeat(1, self.stack_size).reshape(-1)
        
        factor_brightness = torch.empty(self.batch_size, device=self._device).uniform_(*self.brightness)
        self.factor_brightness = factor_brightness.reshape(-1,1).repeat(1, self.stack_size).reshape(-1)
        
        factor_saturate = torch.empty(self.batch_size, device=self._device).uniform_(*self.saturation)
        self.factor_saturate = factor_saturate.reshape(-1,1).repeat(1, self.stack_size).reshape(-1)
        

        
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))
        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value
    
    def adjust_contrast(self, x):
        """
            Args:
                x: torch tensor img (rgb type)
            Factor: torch tensor with same length as x
                    0 gives gray solid image, 1 gives original image,
            Returns:
                torch tensor image: Brightness adjusted
        """
        means = torch.mean(x, dim=(2, 3), keepdim=True)
        return torch.clamp((x - means)
                           * self.factor_contrast.view(len(x), 1, 1, 1) + means, 0, 1)
    
    def adjust_hue(self, x):
        h = x[:, 0, :, :]
        h = h + (self.factor_hue.view(len(x), 1, 1) * 255. / 360.)
        h = (h % 1)
        x[:, 0, :, :] = h
        return x
    
    def adjust_brightness(self, x):
        """
            Args:
                x: torch tensor img (hsv type)
            Factor:
                torch tensor with same length as x
                0 gives black image, 1 gives original image,
                2 gives the brightness factor of 2.
            Returns:
                torch tensor image: Brightness adjusted
        """
        x[:, 2, :, :] = torch.clamp(x[:, 2, :, :]
                                     * self.factor_brightness.view(len(x), 1, 1), 0, 1)
        return torch.clamp(x, 0, 1)
    
    def adjust_saturate(self, x):
        """
            Args:
                x: torch tensor img (hsv type)
            Factor:
                torch tensor with same length as x
                0 gives black image and white, 1 gives original image,
                2 gives the brightness factor of 2.
            Returns:
                torch tensor image: Brightness adjusted
        """
        x[:, 1, :, :] = torch.clamp(x[:, 1, :, :]
                                    * self.factor_saturate.view(len(x), 1, 1), 0, 1)
        return torch.clamp(x, 0, 1)
    
    def transform(self, inputs):
        hsv_transform_list = [rgb2hsv, self.adjust_brightness,
                              self.adjust_hue, self.adjust_saturate,
                              hsv2rgb]
        rgb_transform_list = [self.adjust_contrast]
        
        # Shuffle transform
        if random.uniform(0,1) >= 0.5:
            transform_list = rgb_transform_list + hsv_transform_list
        else:
            transform_list = hsv_transform_list + rgb_transform_list
        for t in transform_list:
            inputs = t(inputs)
        return inputs
    
    def do_augmentation(self, imgs):
        # batch size
        imgs_copy = imgs.clone().to(self._device, dtype=torch.float32)
        imgs_copy = imgs_copy.permute(0,3,1,2)
        self.batch_size = imgs_copy.shape[0]
        self.change_randomization_params_all()
        outputs = self.forward(imgs_copy)
        outputs = outputs.permute(0,2,3,1)
        # imageio.imwrite('/home/ubuntu/procgen-competition/colorjitter.png', outputs[0].cpu().numpy())
        return outputs

    def change_randomization_params(self, index_):
        self.factor_contrast[index_] = torch.empty(1, device=self._device).uniform_(*self.contrast)
        self.factor_hue[index_] = torch.empty(1, device=self._device).uniform_(*self.hue)
        self.factor_brightness[index_] = torch.empty(1, device=self._device).uniform_(*self.brightness)
        self.factor_saturate[index_] = torch.empty(1, device=self._device).uniform_(*self.saturation)

    def change_randomization_params_all(self):
        factor_contrast = torch.empty(self.batch_size, device=self._device).uniform_(*self.contrast)
        self.factor_contrast = factor_contrast.reshape(-1,1).repeat(1, self.stack_size).reshape(-1)
        
        factor_hue = torch.empty(self.batch_size, device=self._device).uniform_(*self.hue)
        self.factor_hue = factor_hue.reshape(-1,1).repeat(1, self.stack_size).reshape(-1)
        
        factor_brightness = torch.empty(self.batch_size, device=self._device).uniform_(*self.brightness)
        self.factor_brightness = factor_brightness.reshape(-1,1).repeat(1, self.stack_size).reshape(-1)
        
        factor_saturate = torch.empty(self.batch_size, device=self._device).uniform_(*self.saturation)
        self.factor_saturate = factor_saturate.reshape(-1,1).repeat(1, self.stack_size).reshape(-1)
        
    def print_parms(self):
        print(self.factor_hue)
        
    def forward(self, inputs):
        # batch size
        random_inds = np.random.choice(
            [True, False], len(inputs), p=[self.prob, 1 - self.prob])
        inds = torch.tensor(random_inds).to(self._device)
        if random_inds.sum() > 0:
            inputs[inds] = self.transform(inputs[inds])
        return inputs

def rgb2hsv(rgb, eps=1e-8):
    # Reference: https://www.rapidtables.com/convert/color/rgb-to-hsv.html
    # Reference: https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L287

    _device = rgb.device
    r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]

    Cmax = rgb.max(1)[0]
    Cmin = rgb.min(1)[0]
    delta = Cmax - Cmin

    hue = torch.zeros((rgb.shape[0], rgb.shape[2], rgb.shape[3])).to(_device)
    hue[Cmax== r] = (((g - b)/(delta + eps)) % 6)[Cmax == r]
    hue[Cmax == g] = ((b - r)/(delta + eps) + 2)[Cmax == g]
    hue[Cmax == b] = ((r - g)/(delta + eps) + 4)[Cmax == b]
    hue[Cmax == 0] = 0.0
    hue = hue / 6. # making hue range as [0, 1.0)
    hue = hue.unsqueeze(dim=1)

    saturation = (delta) / (Cmax + eps)
    saturation[Cmax == 0.] = 0.
    saturation = saturation.to(_device)
    saturation = saturation.unsqueeze(dim=1)

    value = Cmax
    value = value.to(_device)
    value = value.unsqueeze(dim=1)

    return torch.cat((hue, saturation, value), dim=1)

def hsv2rgb(hsv):
    # Reference: https://www.rapidtables.com/convert/color/hsv-to-rgb.html
    # Reference: https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L287

    _device = hsv.device

    hsv = torch.clamp(hsv, 0, 1)
    hue = hsv[:, 0, :, :] * 360.
    saturation = hsv[:, 1, :, :]
    value = hsv[:, 2, :, :]

    c = value * saturation
    x = - c * (torch.abs((hue / 60.) % 2 - 1) - 1)
    m = (value - c).unsqueeze(dim=1)

    rgb_prime = torch.zeros_like(hsv).to(_device)

    inds = (hue < 60) * (hue >= 0)
    rgb_prime[:, 0, :, :][inds] = c[inds]
    rgb_prime[:, 1, :, :][inds] = x[inds]

    inds = (hue < 120) * (hue >= 60)
    rgb_prime[:, 0, :, :][inds] = x[inds]
    rgb_prime[:, 1, :, :][inds] = c[inds]

    inds = (hue < 180) * (hue >= 120)
    rgb_prime[:, 1, :, :][inds] = c[inds]
    rgb_prime[:, 2, :, :][inds] = x[inds]

    inds = (hue < 240) * (hue >= 180)
    rgb_prime[:, 1, :, :][inds] = x[inds]
    rgb_prime[:, 2, :, :][inds] = c[inds]

    inds = (hue < 300) * (hue >= 240)
    rgb_prime[:, 2, :, :][inds] = c[inds]
    rgb_prime[:, 0, :, :][inds] = x[inds]

    inds = (hue < 360) * (hue >= 300)
    rgb_prime[:, 2, :, :][inds] = x[inds]
    rgb_prime[:, 0, :, :][inds] = c[inds]

    rgb = rgb_prime + torch.cat((m, m, m), dim=1)
    rgb = rgb.to(_device)

    return torch.clamp(rgb, 0, 1)


def Identity(x):
    """
    No Augmentation
    """
    return x

aug_to_func = {    
        # 'crop': Crop, #works
        # 'random-conv': RandomConv,
        # 'grayscale': Grayscale, #works
        # 'flip': Flip, #works
        'rotate': Rotate, #works
        # 'cutout': Cutout, #works
        # 'cutout-color': CutoutColor, #works
        # 'color-jitter': ColorJitter #works
}

aug_list = [aug_to_func[t](batch_size=2048) 
            for t in list(aug_to_func.keys())]

num_aug_types = len(aug_list)
expl_action = [0.] * num_aug_types
ucb_action = [0.] * num_aug_types
total_num = 1
num_action = [1.] * num_aug_types
qval_action = [0.] * num_aug_types
ucb_exploration_coef = 0.5
ucb_window_length = 10
return_action = []
for i in range(num_aug_types):
    return_action.append(deque(maxlen=ucb_window_length))

ucb_aug_id = np.argmax(ucb_action)

prev_value_fn = None
prev_ppo_loss = None

class PPOLoss:
    def __init__(self,
                 dist_class,
                 model,
                 value_targets,
                 advantages,
                 actions,
                 prev_logits,
                 prev_actions_logp,
                 vf_preds,
                 cur_obs,
                 curr_action_dist,
                 value_fn,
                 cur_kl_coeff,
                 valid_mask,
                 entropy_coeff=0,
                 clip_param=0.1,
                 vf_clip_param=0.1,
                 vf_loss_coeff=1.0,
                 use_gae=True):
        """Constructs the loss for Proximal Policy Objective.
        Arguments:
            dist_class: action distribution class for logits.
            value_targets (Placeholder): Placeholder for target values; used
                for GAE.
            actions (Placeholder): Placeholder for actions taken
                from previous model evaluation.
            advantages (Placeholder): Placeholder for calculated advantages
                from previous model evaluation.
            prev_logits (Placeholder): Placeholder for logits output from
                previous model evaluation.
            prev_actions_logp (Placeholder): Placeholder for prob output from
                previous model evaluation.
            vf_preds (Placeholder): Placeholder for value function output
                from previous model evaluation.
            curr_action_dist (ActionDistribution): ActionDistribution
                of the current model.
            value_fn (Tensor): Current value function output Tensor.
            cur_kl_coeff (Variable): Variable holding the current PPO KL
                coefficient.
            valid_mask (Tensor): A bool mask of valid input elements (#2992).
            entropy_coeff (float): Coefficient of the entropy regularizer.
            clip_param (float): Clip parameter
            vf_clip_param (float): Clip parameter for the value function
            vf_loss_coeff (float): Coefficient of the value function loss
            use_gae (bool): If true, use the Generalized Advantage Estimator.
        """
        if valid_mask is not None:
            num_valid = torch.sum(valid_mask)

            def reduce_mean_valid(t):
                return torch.sum(t * valid_mask) / num_valid

        else:

            def reduce_mean_valid(t):
                return torch.mean(t)

        prev_dist = dist_class(prev_logits, model)
        # Make loss functions.
        logp_ratio = torch.exp(
            curr_action_dist.logp(actions) - prev_actions_logp)
        action_kl = prev_dist.kl(curr_action_dist)
        self.mean_kl = reduce_mean_valid(action_kl)

        curr_entropy = curr_action_dist.entropy()
        self.mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            advantages * logp_ratio,
            advantages * torch.clamp(logp_ratio, 1 - clip_param,
                                     1 + clip_param))
        self.mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        if use_gae:
            vf_loss1 = torch.pow(value_fn - value_targets, 2.0)
            vf_clipped = vf_preds + torch.clamp(value_fn - vf_preds,
                                                -vf_clip_param, vf_clip_param)
            vf_loss2 = torch.pow(vf_clipped - value_targets, 2.0)
            vf_loss = torch.max(vf_loss1, vf_loss2)
            self.mean_vf_loss = reduce_mean_valid(vf_loss)
            loss = reduce_mean_valid(
                -surrogate_loss + cur_kl_coeff * action_kl +
                vf_loss_coeff * vf_loss - entropy_coeff * curr_entropy)
        else:
            self.mean_vf_loss = 0.0
            loss = reduce_mean_valid(-surrogate_loss +
                                     cur_kl_coeff * action_kl -
                                     entropy_coeff * curr_entropy)
        self.loss = loss

        # cur_obs_aug = current_aug_func.do_augmentation(cur_obs)
        # print("we did the augmentation yo")

def ppo_surrogate_loss(policy, model, dist_class, train_batch):
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    mask = None
    if state:
        max_seq_len = torch.max(train_batch["seq_lens"])
        mask = sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = torch.reshape(mask, [-1])

    policy.loss_obj = PPOLoss(
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[SampleBatch.ACTION_DIST_INPUTS],
        train_batch[SampleBatch.ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        train_batch[SampleBatch.CUR_OBS],
        action_dist,
        model.value_function(),
        policy.kl_coeff,
        mask,
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        use_gae=policy.config["use_gae"]
    )

    aug_train_batch = train_batch

    current_aug_func = aug_list[ucb_aug_id]

    aug_train_batch["obs"] = current_aug_func.do_augmentation(aug_train_batch["obs"]).cuda()

    #should update the ucb vals at end of every "step" (is that an episode?)

    aug_logits, aug_state = model.from_batch(aug_train_batch)
    # aug_action_dist = dist_class(aug_logits, aug_state)

    action_loss_aug = - torch.mean(aug_logits)

    # print("ACTION LOSS AUG", action_loss_aug)

    global prev_ppo_loss
    global prev_value_fn
    # print("prev ppos loss", prev_ppo_loss)
    # print("prev value fun", prev_value_fn)

    if prev_ppo_loss is None:
        prev_ppo_loss = policy.loss_obj.loss
        prev_value_fn = model.value_function()
        return None

    # print("PREV VALUE FN", prev_value_fn)
    value_loss_aug = 0.5 * (prev_value_fn - model.value_function()).pow(2).mean()
    
    # value_loss_aug = 0.5 * (prev_value_fn - model.value_function()).pow(2).mean()

    regularized_loss = prev_ppo_loss + 0.1 * (value_loss_aug + action_loss_aug) 

    # print("prev ppo loss", policy.loss_obj.loss)
    # print("value loss aug", value_loss_aug)
    # print("action loss aug", action_loss_aug)
    # print("regularized_loss", regularized_loss)
    
    # print()
    # print("REGULARIZED", regularized_loss)
    # print("JUST PPO", prev_ppo_loss)
    # print("VALUE LOSS AUG", value_loss_aug)
    # print("ACTION LOSS AUG", action_loss_aug)
    prev_ppo_loss = policy.loss_obj.loss
    prev_value_fn = model.value_function()
    return regularized_loss

def update_ucb_values(rollout_reward_mean):
    global num_aug_types
    global expl_action
    global ucb_action
    global total_num
    global num_action
    global qval_action
    global ucb_exploration_coef
    global ucb_window_length
    global return_action
    global total_num
    global ucb_aug_id
    
    total_num += 1
    num_action[ucb_aug_id] += 1
    return_action[ucb_aug_id].append(rollout_reward_mean)
    qval_action[ucb_aug_id] = np.mean(return_action[ucb_aug_id])

    # select aug
    for i in range(num_aug_types):
        expl_action[i] = ucb_exploration_coef * np.sqrt(np.log(total_num) / num_action[i])
        ucb_action[i] = qval_action[i] + expl_action[i]
    print(ucb_action)
    ucb_aug_id = np.argmax(ucb_action)

    print("select the aug", ucb_aug_id)


def kl_and_loss_stats(policy, train_batch):
    return {
        "cur_kl_coeff": policy.kl_coeff,
        "cur_lr": policy.cur_lr,
        "total_loss": policy.loss_obj.loss,
        "policy_loss": policy.loss_obj.mean_policy_loss,
        "vf_loss": policy.loss_obj.mean_vf_loss,
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.model.value_function(),
            framework="torch"),
        "kl": policy.loss_obj.mean_kl,
        "entropy": policy.loss_obj.mean_entropy,
        "entropy_coeff": policy.entropy_coeff,
    }


def vf_preds_fetches(policy, input_dict, state_batches, model, action_dist):
    """Adds value function outputs to experience train_batches."""
    return {
        SampleBatch.VF_PREDS: policy.model.value_function(),
    }


class KLCoeffMixin:
    def __init__(self, config):
        # KL Coefficient.
        self.kl_coeff = config["kl_coeff"]
        self.kl_target = config["kl_target"]

    def update_kl(self, sampled_kl):
        if sampled_kl > 2.0 * self.kl_target:
            self.kl_coeff *= 1.5
        elif sampled_kl < 0.5 * self.kl_target:
            self.kl_coeff *= 0.5
        return self.kl_coeff


class ValueNetworkMixin:
    def __init__(self, obs_space, action_space, config):
        if config["use_gae"]:

            def value(ob, prev_action, prev_reward, *state):
                model_out, _ = self.model({
                    SampleBatch.CUR_OBS: self._convert_to_tensor([ob]),
                    SampleBatch.PREV_ACTIONS: self._convert_to_tensor(
                        [prev_action]),
                    SampleBatch.PREV_REWARDS: self._convert_to_tensor(
                        [prev_reward]),
                    "is_training": False,
                }, [self._convert_to_tensor(s) for s in state],
                                          self._convert_to_tensor([1]))
                return self.model.value_function()[0]

        else:

            def value(ob, prev_action, prev_reward, *state):
                return 0.0

        self._value = value


def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


PPOTorchPolicy = build_torch_policy(
    name="PPOTorchPolicy",
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    loss_fn=ppo_surrogate_loss,
    update_ucb_values_fn=update_ucb_values,
    stats_fn=kl_and_loss_stats,
    extra_action_out_fn=vf_preds_fetches,
    postprocess_fn=postprocess_ppo_gae,
    extra_grad_process_fn=apply_grad_clipping,
    before_init=setup_config,
    after_init=setup_mixins,
    mixins=[KLCoeffMixin, ValueNetworkMixin])