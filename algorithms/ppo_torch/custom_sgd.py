"""Utils for minibatch SGD across multiple RLlib policies."""

import numpy as np
import logging
from collections import defaultdict
import random

from ray.util import log_once
from ray.rllib.evaluation.metrics import LEARNER_STATS_KEY
from ray.rllib.policy.sample_batch import SampleBatch, DEFAULT_POLICY_ID, \
    MultiAgentBatch

from .tree_util import tree_map

import torch as th

from . import torch_util as tu

from torch import distributions as td

import itertools

from torch import nn
from .custom_postprocessing import Postprocessing

from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size


def make_minibatches(segs, mbsize):
    """
    Yield one epoch of minibatch over the dataset described by segs
    Each minibatch mixes data between different segs
    """
    # nenv = tu.batch_len(segs[0])
    nenv = 1024
    nseg = len(segs)
    envs_segs = th.tensor(list(itertools.product(range(nenv), range(nseg))))
    for perminds in th.randperm(len(envs_segs)).split(mbsize):
        esinds = envs_segs[perminds]
        # print("for perminds", perminds.shape, esinds.shape)
        yield tu.tree_stack(
            [tu.tree_slice(segs[segind], envind) for (envind, segind) in esinds]
        )


logger = logging.getLogger(__name__)


def averaged(kv, axis=None):
    """Average the value lists of a dictionary.

    For non-scalar values, we simply pick the first value.

    Arguments:
        kv (dict): dictionary with values that are lists of floats.

    Returns:
        dictionary with single averaged float as values.
    """
    out = {}
    for k, v in kv.items():
        if v[0] is not None and not isinstance(v[0], dict):
            out[k] = np.mean(v, axis=axis)
        else:
            out[k] = v[0]
    return out


def standardized(array):
    """Normalize the values in an array.

    Arguments:
        array (np.ndarray): Array of values to normalize.

    Returns:
        array with zero mean and unit standard deviation.
    """
    return (array - array.mean()) / max(1e-4, array.std())


def minibatches(samples, sgd_minibatch_size):
    """Return a generator yielding minibatches from a sample batch.

    Arguments:
        samples (SampleBatch): batch of samples to split up.
        sgd_minibatch_size (int): size of minibatches to return.

    Returns:
        generator that returns mini-SampleBatches of size sgd_minibatch_size.
    """
    if not sgd_minibatch_size:
        yield samples
        return

    if isinstance(samples, MultiAgentBatch):
        raise NotImplementedError(
            "Minibatching not implemented for multi-agent in simple mode")

    if "state_in_0" in samples.data:
        if log_once("not_shuffling_rnn_data_in_simple_mode"):
            logger.warning("Not shuffling RNN data for SGD in simple mode")
    else:
        samples.shuffle()

    i = 0
    slices = []
    while i < samples.count:
        slices.append((i, i + sgd_minibatch_size))
        i += sgd_minibatch_size
    random.shuffle(slices)

    for i, j in slices:
        yield samples.slice(i, j)


nepochs = 0
seg_buf = []


def do_minibatch_sgd(samples, policies, local_worker, num_sgd_iter,
                     sgd_minibatch_size, standardize_fields):
    """Execute minibatch SGD.

    Arguments:
        samples (SampleBatch): batch of samples to optimize.
        policies (dict): dictionary of policies to optimize.
        local_worker (RolloutWorker): master rollout worker instance.
        num_sgd_iter (int): number of epochs of optimization to take.
        sgd_minibatch_size (int): size of minibatches to use for optimization.
        standardize_fields (list): list of sample field names that should be
            normalized prior to optimization.

    Returns:
        averaged info fetches over the last SGD epoch taken.
    """
    # Get batch
    global nepochs
    global seg_buf
    if isinstance(samples, SampleBatch):
        samples = MultiAgentBatch({DEFAULT_POLICY_ID: samples}, samples.count)
    # print("samples batch policy batches", samples.policy_batches['default_policy']['dones'].shape)
    
    fetches = {}
    for policy_id, policy in policies.items():
        model = policy.model

        dist_class = policy.dist_class

        if policy_id not in samples.policy_batches:
            continue

        batch = samples.policy_batches[policy_id]
        for field in standardize_fields:
            batch[field] = standardized(batch[field])

        seg_buf.append(batch)

        for i in range(num_sgd_iter):
            iter_extra_fetches = defaultdict(list)
            #pass the whole batch to the worker, then let it break it down into minibatches
            batch_fetches = (local_worker.learn_on_batch(
                MultiAgentBatch({
                    policy_id: batch
                }, batch.count)))[policy_id]

            for k, v in batch_fetches.get(LEARNER_STATS_KEY, {}).items():
                iter_extra_fetches[k].append(v)

            logger.debug("{} {}".format(i, averaged(iter_extra_fetches)))

        fetches[policy_id] = averaged(iter_extra_fetches)
    
        nepochs += 1
        if nepochs % 16 == 0:
            # print("do auxiliary phase")
            def forward(seg):
                logits, state = model.forward(seg, None, None)
                return logits, state      

            REPLAY_MB_SIZE = 1024

            # #compute presleep outputs for replay buffer (what does this mean?)
            for seg in seg_buf:
                np_data = {}
                np_data["obs"] = th.from_numpy(seg.data["obs"]).to(th.cuda.current_device())
                logits, state = tu.minibatched_call(forward, REPLAY_MB_SIZE, seg=np_data)
                seg.data["oldpd"] = logits.cpu().numpy()

            replay_batch = SampleBatch.concat_samples(seg_buf)
            seg_buf.clear()
            #train on replay buffer
            for i in range(6):    
                for mb in minibatches(replay_batch, REPLAY_MB_SIZE):
                    # mb = tree_map(lambda x: x.to(tu.dev()), mb)
                    pad_batch_to_sequences_of_same_size(mb,
                                                        max_seq_len=20,
                                                        shuffle=False,
                                                        batch_divisibility_req=1)
                    mb["obs"] = th.from_numpy(mb["obs"]).to(th.cuda.current_device())

                    logits, vpredaux = model.forward_aux(mb)

                    oldpd = dist_class(th.from_numpy(mb['oldpd']).to(th.cuda.current_device()))
                    pd = dist_class(logits, model)
                    pol_distance = oldpd.kl(pd).mean()

                    vpredtrue = model.value_function()

                    vtarg = th.from_numpy(mb[Postprocessing.VALUE_TARGETS]).to(th.cuda.current_device())
                    vf_aux = 0.5 * th.mean(th.pow(vpredaux - vtarg, 2.0))
                    vf_true = 0.5 * th.mean(th.pow(vpredtrue - vtarg, 2.0))

                    loss = pol_distance + vf_aux + vf_true
                    # print("losses", pol_distance, vf_aux, vf_true, loss)

                    policy.aux_learn(loss)

    return fetches