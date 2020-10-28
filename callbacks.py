
from typing import Dict

import ray
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks

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
        self.distributed = distributed and tu.is_distributed()

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
        self.ret = th.zeros(num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.per_env = per_env

    def __call__(self, reward, first):
        rets = backward_discounted_sum(
            prevret=self.ret, reward=reward.cpu(), first=first.cpu(), gamma=self.gamma
        )
        self.ret = rets[:, -1]
        self.ret_rms.update(rets if self.per_env else rets.reshape(-1))
        return self.transform(reward)

    def transform(self, reward):
        return th.clamp(
            reward / th.sqrt(self.ret_rms.var + self.epsilon),
            -self.cliprew,
            self.cliprew,
        )


def backward_discounted_sum(
    *,
    prevret: "(th.Tensor[1, float]) value predictions",
    reward: "(th.Tensor[1, float]) reward",
    first: "(th.Tensor[1, bool]) mark beginning of episodes",
    gamma: "(float)",
):
    first = first.to(dtype=th.float32)
    assert first.dim() == 2
    _nenv, nstep = reward.shape
    ret = th.zeros_like(reward)
    for t in range(nstep):
        prevret = ret[:, t] = reward[:, t] + (1 - first[:, t]) * gamma * prevret
    return ret




class CustomCallbacks(DefaultCallbacks):
    """
    Please refer to : 
        https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
        https://docs.ray.io/en/latest/rllib-training.html#callbacks-and-custom-metrics
    for examples on adding your custom metrics and callbacks. 

    This code adapts the documentations of the individual functions from :
    https://github.com/ray-project/ray/blob/master/rllib/agents/callbacks.py

    These callbacks can be used for custom metrics and custom postprocessing.
    """

    def __init__(self, *args, **kwargs):
        super(CustomCallbacks, self).__init__(*args, **kwargs)
        self.reward_normalizer = RewardNormalizer(1)

    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        """Callback run on the rollout worker before each episode starts.
        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            base_env (BaseEnv): BaseEnv running the episode. The underlying
                env object can be gotten by calling base_env.get_unwrapped().
            policies (dict): Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            episode (MultiAgentEpisode): Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """
        pass

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):
        """Runs on each episode step.
        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            base_env (BaseEnv): BaseEnv running the episode. The underlying
                env object can be gotten by calling base_env.get_unwrapped().
            episode (MultiAgentEpisode): Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """
        pass

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy],
                       episode: MultiAgentEpisode, **kwargs):
        """Runs when an episode is done.
        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            base_env (BaseEnv): BaseEnv running the episode. The underlying
                env object can be gotten by calling base_env.get_unwrapped().
            policies (dict): Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            episode (MultiAgentEpisode): Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            kwargs: Forward compatibility placeholder.
        """
        ######################################################################
        # An example of adding a custom metric from the latest observation 
        # from your env
        ######################################################################
        # last_obs_object_from_episode = episode.last_observation_for()
        # We define a dummy custom metric, observation_mean
        # episode.custom_metrics["observation_mean"] = last_obs_object_from_episode.mean()
        pass

    def on_postprocess_trajectory(
            self, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str,
            policies: Dict[str, Policy], postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        """Called immediately after a policy's postprocess_fn is called.
        You can use this callback to do additional postprocessing for a policy,
        including looking at the trajectory data of other agents in multi-agent
        settings.
        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            episode (MultiAgentEpisode): Episode object.
            agent_id (str): Id of the current agent.
            policy_id (str): Id of the current policy for the agent.
            policies (dict): Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            postprocessed_batch (SampleBatch): The postprocessed sample batch
                for this agent. You can mutate this object to apply your own
                trajectory postprocessing.
            original_batches (dict): Mapping of agents to their unpostprocessed
                trajectory data. You should not mutate this object.
            kwargs: Forward compatibility placeholder.
        """
        print("postprocess trajectory", postprocessed_batch[SampleBatch.DONES])
        postprocessed_batch[SampleBatch.PREV_REWARDS] = self.reward_normalizer(postprocessed_batch[SampleBatch.PREV_REWARDS], postprocessed_batch[SampleBatch.DONES])

    def on_sample_end(self, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        """Called at the end RolloutWorker.sample().
        Args:
            worker (RolloutWorker): Reference to the current rollout worker.
            samples (SampleBatch): Batch to be returned. You can mutate this
                object to modify the samples generated.
            kwargs: Forward compatibility placeholder.
        """
        pass

    def on_train_result(self, trainer, result: dict, **kwargs):
        """Called at the end of Trainable.train().
        Args:
            trainer (Trainer): Current trainer instance.
            result (dict): Dict of results returned from trainer.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        # In this case we also print the mean timesteps throughput
        # for easier reference in the logs
        # print("=============================================================")
        # print(" Timesteps Throughput : {} ts/sec".format(TBD))
        # print("=============================================================")
        pass
