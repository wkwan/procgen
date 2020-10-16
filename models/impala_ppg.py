from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

import math

import torch as th
from torch import nn
from torch.nn import functional as F

from gym3.types import Real, TensorType
REAL = Real()


import collections
import functools
import itertools
import math
import os
import platform
from contextlib import contextmanager
import re

import numpy as np
import torch.distributed as dist
import torch.distributions as dis
import socket
import time
import random
import multiprocessing as mp

def have_cuda():
    return (
        th.has_cuda and th.cuda.is_available() and not os.getenv("RCALL_NUM_GPU") == "0"
    )

def default_device_type():
    return "cuda" if have_cuda() else "cpu"

def ftensor(*args, **kwargs):
    return th.tensor(*args, **kwargs, device=dev(), dtype=th.float32)


def ltensor(*args, **kwargs):
    return th.tensor(*args, **kwargs, device=dev(), dtype=th.int64)


def zeros(*args, **kwargs):
    return th.zeros(*args, **kwargs, device=dev())


def ones(*args, **kwargs):
    return th.ones(*args, **kwargs, device=dev())


def arange(*args, **kwargs):
    return th.arange(*args, **kwargs, device=dev())


def np2th(nparr):
    dtype = th.float32 if nparr.dtype == np.float64 else None
    return th.from_numpy(nparr).to(device=dev(), dtype=dtype)


def th2np(tharr):
    return tharr.cpu().numpy()


def NormedLinear(*args, scale=1.0, dtype=th.float32, **kwargs):
    """
    nn.Linear but with normalized fan-in init
    """
    dtype = parse_dtype(dtype)
    if dtype == th.float32:
        out = nn.Linear(*args, **kwargs)
    elif dtype == th.float16:
        out = LinearF16(*args, **kwargs)
    else:
        raise ValueError(dtype)
    out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    if kwargs.get("bias", True):
        out.bias.data *= 0
    return out

def NormedConv2d(*args, scale=1, **kwargs):
    """
    nn.Conv2d but with normalized fan-in init
    """
    out = nn.Conv2d(*args, **kwargs)
    out.weight.data *= scale / out.weight.norm(dim=(1, 2, 3), p=2, keepdim=True)
    if kwargs.get("bias", True):
        out.bias.data *= 0
    return out

def flatten_image(x):
    """
    Flattens last three dims
    """
    *batch_shape, h, w, c = x.shape
    return x.reshape((*batch_shape, h * w * c))


def sequential(layers, x, *args, diag_name=None):
    for (i, layer) in enumerate(layers):
        x = layer(x, *args)
    return x

def all_mean_(x, group=dist.group.WORLD):
    dist_all_reduce(x, group=group)
    x /= dist_get_world_size(group=group)
    return x


def all_mean(x):
    return all_mean_(x.clone())


def all_sum_(x, group=dist.group.WORLD):
    dist_all_reduce(x, group=group)
    return x


def all_sum(x):
    return all_sum_(x.clone())


def flatten_tensors(xs, dtype=None, buf=None):
    if buf is None:
        buf = xs[0].new_empty(sum(x.numel() for x in xs), dtype=dtype)
    i = 0
    for x in xs:
        buf[i : i + x.numel()].copy_(x.view(-1))
        i += x.numel()
    return buf


def unflatten_to(newflat, xs):
    start = 0
    for x in xs:
        size = x.numel()
        end = start + size
        x.copy_(newflat[start:end].view(x.shape))
        start = end
    assert start == newflat.numel()

def is_distributed():
    return dist.is_initialized()


def dist_broadcast(*args, **kwargs):
    if not is_distributed():
        return
    dist.broadcast(*args, **kwargs)


def dist_all_reduce(*args, **kwargs):
    if not is_distributed():
        return
    dist.all_reduce(*args, **kwargs)


def dist_get_world_size(group=dist.group.WORLD):
    if not is_distributed():
        return 1
    return dist.get_world_size(group=group)

def sync_grads(
    params, group=dist.group.WORLD, grad_weight=1.0, dtype=None, sync_buffer=None
):
    """
    Sync gradients for the provided params across all members of the specified group
    """
    if not is_distributed():
        assert group is dist.group.WORLD
        return
    if dist.get_world_size(group) == 1:
        return
    grads = [p.grad for p in params if p.grad is not None]
    flatgrad = flatten_tensors(grads, dtype=dtype, buf=sync_buffer)
    if grad_weight != 1.0:
        flatgrad.mul_(grad_weight)
    all_mean_(flatgrad, group=group)
    unflatten_to(flatgrad, grads)

def _numpy_allmean(comm, x):
    out = np.zeros_like(x)
    comm.Allreduce(x, out)
    out /= comm.size
    return out

def parse_dtype(x):
    if isinstance(x, th.dtype):
        return x
    elif isinstance(x, str):
        if x == "float32" or x == "float":
            return th.float32
        elif x == "float64" or x == "double":
            return th.float64
        elif x == "float16" or x == "half":
            return th.float16
        elif x == "uint8":
            return th.uint8
        elif x == "int8":
            return th.int8
        elif x == "int16" or x == "short":
            return th.int16
        elif x == "int32" or x == "int":
            return th.int32
        elif x == "int64" or x == "long":
            return th.int64
        elif x == "bool":
            return th.bool
        else:
            raise ValueError(f"cannot parse {x} as a dtype")
    else:
        raise TypeError(f"cannot parse {type(x)} as dtype")

def sum_nonbatch(x, nbatchdim=2):
    return x.sum(dim=tuple(range(nbatchdim, x.dim()))) if x.dim() > nbatchdim else x

def _process_modelpath(path, stage_index):
    # if we have a pipelined model, the user should specify a path with stage-0 in the filename
    # replace it with the correct stage
    return path.replace("-stage-0", f"-stage-{stage_index}")

class Encoder(nn.Module):
    """
    Takes in seq of observations and outputs sequence of codes

    Encoders can be stateful, meaning that you pass in one observation at a
    time and update the state, which is a separate object. (This object
    doesn't store any state except parameters)
    """

    def __init__(self, obtype, codetype):
        super().__init__()
        self.obtype = obtype
        self.codetype = codetype

    def initial_state(self, batchsize):
        raise NotImplementedError

    def empty_state(self):
        return None

    def stateless_forward(self, obs):
        """
        inputs:
            obs: array or dict, all with preshape (B, T)
        returns:
            codes: array or dict, all with preshape (B, T)
        """
        code, _state = self(obs, None, self.empty_state())
        return code

    def forward(self, obs, first, state_in):
        """
        inputs:
            obs: array or dict, all with preshape (B, T)
            first: float array shape (B, T)
            state_in: array or dict, all with preshape (B,)
        returns:
            codes: array or dict
            state_out: array or dict
        """
        raise NotImplementedError

class CnnBasicBlock(nn.Module):
    """
    Residual basic block (without batchnorm), as in ImpalaCNN
    Preserves channel number and shape
    """

    def __init__(self, inchan, scale=1, batch_norm=False):
        super().__init__()
        self.inchan = inchan
        self.batch_norm = batch_norm
        s = math.sqrt(scale)
        self.conv0 = NormedConv2d(self.inchan, self.inchan, 3, padding=1, scale=s)
        self.conv1 = NormedConv2d(self.inchan, self.inchan, 3, padding=1, scale=s)
        if self.batch_norm:
            self.bn0 = nn.BatchNorm2d(self.inchan)
            self.bn1 = nn.BatchNorm2d(self.inchan)

    def residual(self, x):
        # inplace should be False for the first relu, so that it does not change the input,
        # which will be used for skip connection.
        # getattr is for backwards compatibility with loaded models
        if getattr(self, "batch_norm", False):
            x = self.bn0(x)
        x = F.relu(x, inplace=False)
        x = self.conv0(x)
        if getattr(self, "batch_norm", False):
            x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv1(x)
        return x

    def forward(self, x):
        return x + self.residual(x)


class CnnDownStack(nn.Module):
    """
    Downsampling stack from Impala CNN
    """

    def __init__(self, inchan, nblock, outchan, scale=1, pool=True, **kwargs):
        super().__init__()
        self.inchan = inchan
        self.outchan = outchan
        self.pool = pool
        self.firstconv = NormedConv2d(inchan, outchan, 3, padding=1)
        s = scale / math.sqrt(nblock)
        self.blocks = nn.ModuleList(
            [CnnBasicBlock(outchan, scale=s, **kwargs) for _ in range(nblock)]
        )

    def forward(self, x):
        x = self.firstconv(x)
        if getattr(self, "pool", True):
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for block in self.blocks:
            x = block(x)
        return x

    def output_shape(self, inshape):
        c, h, w = inshape
        assert c == self.inchan
        if getattr(self, "pool", True):
            return (self.outchan, (h + 1) // 2, (w + 1) // 2)
        else:
            return (self.outchan, h, w)


class ImpalaCNN(TorchModelV2, nn.Module):
    name = "ImpalaCNN"  # put it here to preserve pickle compat

    # def __init__(
    #     self, inshape, chans, outsize, scale_ob, nblock, final_relu=True, **kwargs
    # ):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        num_outputs = 15
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)

        nn.Module.__init__(self)

        print("INIT THE IMPALA CNN")
        chans = [16, 32, 32]
        scale_ob = 255.0
        nblock = 2

        self.scale_ob = scale_ob
        h, w, c = obs_space.shape
        curshape = (c, h, w)
        s = 1 / math.sqrt(len(chans))  # per stack scale
        self.stacks = nn.ModuleList()
        for outchan in chans:
            stack = CnnDownStack(
                curshape[0], nblock=nblock, outchan=outchan, scale=s
            )
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)
        self.dense = NormedLinear(intprod(curshape), num_outputs, scale=1.4)
        self.outsize = num_outputs
        self.final_relu = True

    def forward(self, x):
        x = x.to(dtype=th.float32) / self.scale_ob

        b, t = x.shape[:-3]
        x = x.reshape(b * t, *x.shape[-3:])
        x = transpose(x, "bhwc", "bchw")
        x = sequential(self.stacks, x, diag_name=self.name)
        x = x.reshape(b, t, *x.shape[1:])
        x = flatten_image(x)
        x = th.relu(x)
        x = self.dense(x)
        if self.final_relu:
            x = th.relu(x)
        return x


class ImpalaEncoder(Encoder):
    def __init__(
        self,
        inshape,
        outsize=256,
        chans=(16, 32, 32),
        scale_ob=255.0,
        nblock=2,
        **kwargs
    ):
        codetype = TensorType(eltype=REAL, shape=(outsize,))
        obtype = TensorType(eltype=REAL, shape=inshape)
        super().__init__(codetype=codetype, obtype=obtype)
        self.cnn = ImpalaCNN(
            inshape=inshape,
            chans=chans,
            scale_ob=scale_ob,
            nblock=nblock,
            outsize=outsize,
            **kwargs
        )

    def forward(self, x, first, state_in):
        x = self.cnn(x)
        return x, state_in

    def initial_state(self, batchsize):
        return zeros(batchsize, 0)

ModelCatalog.register_custom_model("impala_ppg", ImpalaCNN)
