import numpy as np
import scipy.signal
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import DeveloperAPI

import numpy as np

import torch as th
import torch.distributed as dist
from torch import nn

import os

def flatten_tensors(xs, dtype=None, buf=None):
    if buf is None:
        buf = xs[0].new_empty(sum(x.numel() for x in xs), dtype=dtype)
    i = 0
    for x in xs:
        buf[i : i + x.numel()].copy_(x.view(-1))
        i += x.numel()
    return buf

def have_cuda():
    return (
        th.has_cuda and th.cuda.is_available() and not os.getenv("RCALL_NUM_GPU") == "0"
    )

def is_distributed():
    return dist.is_initialized()

def dist_all_reduce(*args, **kwargs):
    if not is_distributed():
        return
    dist.all_reduce(*args, **kwargs)


def dist_get_world_size(group=dist.group.WORLD):
    if not is_distributed():
        return 1
    return dist.get_world_size(group=group)


def default_device_type():
    return "cuda" if have_cuda() else "cpu"

DEFAULT_DEVICE = th.device(type=default_device_type())

def all_mean_(x, group=dist.group.WORLD):
    dist_all_reduce(x, group=group)
    x /= dist_get_world_size(group=group)
    return x


def all_mean(x):
    return all_mean_(x.clone())

def unflatten_to(newflat, xs):
    start = 0
    for x in xs:
        size = x.numel()
        end = start + size
        x.copy_(newflat[start:end].view(x.shape))
        start = end
    assert start == newflat.numel()


class RunningMeanStd(nn.Module):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(
        self,
        epsilon: "initial count (with mean=0 ,var=1)" = 1e-4,
        shape: "unbatched shape of data" = (),
        distributed: "whether to allreduce stats" = True,
    ):
        super().__init__()
        self.register_buffer("mean", th.zeros(shape))
        self.register_buffer("var", th.ones(shape))
        self.register_buffer("count", th.tensor(epsilon))
        self.distributed = distributed and is_distributed()

    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = th.tensor([x.shape[0]], device=x.device, dtype=th.float32)
        if self.distributed:
            # flatten+unflatten so we just need one allreduce
            flat = flatten_tensors([batch_mean, batch_var, batch_count])
            flat = flat.to(device=DEFAULT_DEVICE)  # Otherwise all_mean_ will fail
            all_mean_(flat)
            unflatten_to(flat, [batch_mean, batch_var, batch_count])
            batch_count *= dist.get_world_size()
        self.update_from_moments(batch_mean, batch_var, batch_count[0])

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        # pylint: disable=attribute-defined-outside-init
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + delta ** 2 * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class RewardNormalizer:
    """
    Pseudocode can be found in https://arxiv.org/pdf/1811.02553.pdf
    section 9.3 (which is based on our Baselines code, haha)
    Motivation is that we'd rather normalize the returns = sum of future rewards,
    but we haven't seen the future yet. So we assume that the time-reversed rewards
    have similar statistics to the rewards, and normalize the time-reversed rewards.
    """

    def __init__(self, num_envs, cliprew=10.0, gamma=0.99, epsilon=1e-8, per_env=False):
        ret_rms_shape = (num_envs,) if per_env else ()
        self.ret_rms = RunningMeanStd(shape=ret_rms_shape)
        self.cliprew = cliprew
        self.gamma = gamma
        self.epsilon = epsilon
        self.per_env = per_env

    def __call__(self, reward, done):
        rets = backward_discounted_sum(
            reward=reward, gamma=self.gamma
        )
        # self.ret = rets[:, -1]
        self.ret_rms.update(rets)
        return self.transform(reward)

    def transform(self, reward):
        return th.clamp(
            reward / th.sqrt(self.ret_rms.var + self.epsilon),
            -self.cliprew,
            self.cliprew,
        )


def backward_discounted_sum(
    *,
    reward: "(th.Tensor[1, float]) reward",
    gamma: "(float)",
):

    # print("backward discounted", reward.shape, first.shape, first)
    # assert first.dim() == 2
    nstep = reward.shape[0]
    ret = th.zeros_like(reward)
    # print("reward init", reward.shape, first.shape)
    prevret = ret[0] = reward[0]
    for t in range(1, nstep):
        prevret = ret[t] = reward[t] * gamma * prevret
        # print("reward at nstep", t, first[t-1])
    return ret



def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class Postprocessing:
    """Constant definitions for postprocessing."""

    ADVANTAGES = "advantages"
    VALUE_TARGETS = "value_targets"


reward_normalizer = RewardNormalizer(1)

@DeveloperAPI
def compute_advantages(rollout,
                       last_r,
                       gamma=0.9,
                       lambda_=1.0,
                       use_gae=True,
                       use_critic=True):
    """
    Given a rollout, compute its value targets and the advantage.

    Args:
        rollout (SampleBatch): SampleBatch of a single trajectory
        last_r (float): Value estimation for last observation
        gamma (float): Discount factor.
        lambda_ (float): Parameter for GAE
        use_gae (bool): Using Generalized Advantage Estimation
        use_critic (bool): Whether to use critic (value estimates). Setting
                           this to False will use 0 as baseline.

    Returns:
        SampleBatch (SampleBatch): Object with experience from rollout and
            processed rewards.
    """

    traj = {}
    trajsize = len(rollout[SampleBatch.ACTIONS])
    print("trajsize is", trajsize)
    for key in rollout:
        traj[key] = np.stack(rollout[key])

    assert SampleBatch.VF_PREDS in rollout or not use_critic, \
        "use_critic=True but values not found"
    assert use_critic or not use_gae, \
        "Can't use gae without using a value function"

    if use_gae:

        print("before normalizer", traj[SampleBatch.REWARDS])
        traj[SampleBatch.REWARDS] = reward_normalizer(th.from_numpy(traj[SampleBatch.REWARDS])).numpy()
        print("after normalizer", traj[SampleBatch.REWARDS])

        vpred_t = np.concatenate(
            [rollout[SampleBatch.VF_PREDS],
             np.array([last_r])])
        delta_t = (
            traj[SampleBatch.REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1])
        # This formula for the advantage comes from:
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        traj[Postprocessing.ADVANTAGES] = discount(delta_t, gamma * lambda_)
        traj[Postprocessing.VALUE_TARGETS] = (
            traj[Postprocessing.ADVANTAGES] +
            traj[SampleBatch.VF_PREDS]).copy().astype(np.float32)
    else:
        rewards_plus_v = np.concatenate(
            [rollout[SampleBatch.REWARDS],
             np.array([last_r])])
        discounted_returns = discount(rewards_plus_v,
                                      gamma)[:-1].copy().astype(np.float32)
        # print("custom post processing", discounted_returns)

        if use_critic:
            traj[Postprocessing.
                 ADVANTAGES] = discounted_returns - rollout[SampleBatch.
                                                            VF_PREDS]
            traj[Postprocessing.VALUE_TARGETS] = discounted_returns
        else:
            traj[Postprocessing.ADVANTAGES] = discounted_returns
            traj[Postprocessing.VALUE_TARGETS] = np.zeros_like(
                traj[Postprocessing.ADVANTAGES])

    traj[Postprocessing.ADVANTAGES] = traj[
        Postprocessing.ADVANTAGES].copy().astype(np.float32)

    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
    return SampleBatch(traj)