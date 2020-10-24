import collections
import functools
import itertools
import math
import os
import platform
from contextlib import contextmanager
import re

import numpy as np
import torch as th
import torch.distributed as dist
import torch.distributions as dis
import torch.nn.functional as F
from . import logger
from torch import nn
from . import tree_util
import socket
import time
import random
import multiprocessing as mp

def format_model(mod, rms=False):
    """
    Return a str: a formatted table listing parameters and their sizes
    """
    import pandas

    rows = []
    ntotal = sum(p.numel() for p in mod.parameters())
    for name, param in sorted(mod.named_parameters()):
        shape = tuple(param.shape)
        numel = param.numel()
        std = "%0.5f" % float(param.std())
        crnt = [name, shape, numel, round(numel / ntotal * 100, 1), std, _rms(param)]
        rows.append(crnt)

    for name, module in mod.named_modules():
        numel = sum(p.numel() for p in module.parameters())
        if numel == 0:
            continue
        crnt = [name or "~total", "", numel, round(numel / ntotal * 100, 1), "", ""]
        rows.append(crnt)
    columns = ["path", "shape", "numel", "pct", "std", "rms"]
    if not rms:
        rows = [row[:-1] for row in rows]
        columns = columns[:-1]
    rows.sort(key=lambda x: x[0])
    df = pandas.DataFrame(rows, columns=columns)
    maxlen = df["path"].str.len().max()
    return df.to_string(
        index=False, formatters={"path": "{{:<{}s}}".format(maxlen).format}
    )

def intprod(xs):
    """
    Product of a sequence of integers
    """
    out = 1
    for x in xs:
        out *= x
    return out

def transpose(x, before, after):
    """
    Usage: x_bca = transpose(x_abc, 'abc', 'bca')
    """
    assert sorted(before) == sorted(after), f"cannot transpose {before} to {after}"
    assert x.ndim == len(
        before
    ), f"before spec '{before}' has length {len(before)} but x has {x.ndim} dimensions: {tuple(x.shape)}"
    return x.permute(tuple(before.index(i) for i in after))


def allsame(xs):
    """
    Returns whether all elements of sequence are the same
    """
    assert len(xs) > 0
    return all(x == xs[0] for x in xs[1:])

def batch_len(batch):
    """
    Given nested dict of arrays with same batchsize, return this batchsize
    """
    flatlist, _ = tree_util.tree_flatten(batch)
    if len(flatlist) < 1:
        return 0
    b = flatlist[0].shape[0]
    assert all(
        arr.shape[0] == b for arr in flatlist if th.is_tensor(arr)
    ), "Not all arrays have same batchsize!"
    return b

def param_count(model):
    return sum(p.numel() for p in model.parameters())

def _rms(x):
    return ((x ** 2).mean() ** 0.5).item()


def contextmanager_to_decorator(cm):
    def decorator(fn):
        @functools.wraps(fn)
        def newfn(*args, **kwargs):
            with cm():
                return fn(*args, **kwargs)

        return newfn

    return decorator


def have_cuda():
    return (
        th.has_cuda and th.cuda.is_available() and not os.getenv("RCALL_NUM_GPU") == "0"
    )


def default_device_type():
    return "cuda" if have_cuda() else "cpu"


no_grad = contextmanager_to_decorator(th.no_grad)
DEFAULT_DEVICE = th.device(type=default_device_type())
DEFAULT_COMM = None

def _get_local_rank_size(comm):
    """
    Returns the rank of each process on its machine
    The processes on a given machine will be assigned ranks
        0, 1, 2, ..., N-1,
    where N is the number of processes on this machine.
    Useful if you want to assign one gpu per machine
    """
    this_node = platform.node()
    ranks_nodes = comm.allgather((comm.Get_rank(), this_node))
    node2rankssofar = collections.defaultdict(int)
    local_rank = None
    for (rank, node) in ranks_nodes:
        if rank == comm.Get_rank():
            local_rank = node2rankssofar[node]
        node2rankssofar[node] += 1
    assert local_rank is not None
    return local_rank, node2rankssofar[this_node]

def dev():
    return DEFAULT_DEVICE


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

@functools.lru_cache()  # Just run once
def register_distributions_for_tree_util():
    tree_util.register_pytree_node(
        dis.Categorical,
        lambda d: ((d.logits,), None),
        lambda _keys, xs: dis.Categorical(logits=xs[0]),
    )
    tree_util.register_pytree_node(
        dis.Bernoulli,
        lambda d: ((d.logits,), None),
        lambda _keys, xs: dis.Bernoulli(logits=xs[0]),
    )

@functools.lru_cache()
def warn_no_gradient(model, task):
    for n, p in model.named_parameters():
        if p.grad is None:
            print(f"parameter '{n}' {p.shape} has no gradient for '{task}'")

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

@no_grad
def minibatched_call(fn, mbsize, *args, **kwargs):
    """
    Same result as fn(**kwargs) but breaking up the inputs
    into minibatches of size mbsize to avoid OOM errors
    """
    tensor_list, _ = tree_util.tree_flatten((args, kwargs))
    batchsize = tensor_list[0].shape[0]
    print("before minibatched call")
    mbs = [
        fn(*tree_slice(args, inds), **tree_slice(kwargs, inds))
        for inds in th.arange(batchsize).split(mbsize)
    ]
    return tree_cat(mbs, dim=0)


def tree_stack(trees):
    return tree_util.tree_multimap(lambda *xs: th.stack(xs, dim=0), *trees)


def tree_cat(trees, dim=0):
    return tree_util.tree_multimap(lambda *xs: th.cat(xs, dim=dim), *trees)


def tree_slice(tree, sli):
    return tree_util.tree_map(lambda x: x[sli], tree)


def sum_nonbatch(x, nbatchdim=2):
    return x.sum(dim=tuple(range(nbatchdim, x.dim()))) if x.dim() > nbatchdim else x

def _process_modelpath(path, stage_index):
    # if we have a pipelined model, the user should specify a path with stage-0 in the filename
    # replace it with the correct stage
    return path.replace("-stage-0", f"-stage-{stage_index}")