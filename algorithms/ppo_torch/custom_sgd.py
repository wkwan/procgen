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



def make_minibatches(segs, mbsize):
    """
    Yield one epoch of minibatch over the dataset described by segs
    Each minibatch mixes data between different segs
    """
    nenv = tu.batch_len(segs[0])
    nseg = len(segs)
    # print("nenv", nenv, "nseg", nseg)
    nenv = 1024 #should be 2048, but then it doesn't work for any smaller minibatches
    envs_segs = th.tensor(list(itertools.product(range(nenv), range(nseg))))
    for perminds in th.randperm(len(envs_segs)).split(mbsize):
        esinds = envs_segs[perminds]
        # print("for perminds", perminds, esinds)
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

    if isinstance(samples, SampleBatch):
        samples = MultiAgentBatch({DEFAULT_POLICY_ID: samples}, samples.count)

    global nepochs
    seg_buf = []
    fetches = {}

    for policy_id, policy in policies.items():

        model = policy.model
        dist_class = policy.dist_class
        print("dist class is", dist_class)

        if policy_id not in samples.policy_batches:
            continue

        batch = samples.policy_batches[policy_id]
        for field in standardize_fields:
            batch[field] = standardized(batch[field])

        for i in range(num_sgd_iter):
            iter_extra_fetches = defaultdict(list)
            #get minibatch

            for minibatch in minibatches(batch, sgd_minibatch_size):
                # print("get minibatch call", nepochs)
                nepochs += 1
                #compute losses and do backprop
                # print("minibatch", minibatch)
                batch_fetches = (local_worker.learn_on_batch(
                    MultiAgentBatch({
                        policy_id: minibatch
                    }, minibatch.count)))[policy_id]
                minibatch.data["vtarg"] = batch_fetches["vtarg"]
                minibatch.data["oldpd"] = batch_fetches["oldpd"]
                minibatch.data["dones"] = batch_fetches["dones"]

                seg_buf.append(tree_map(lambda x: x, minibatch.data))

                for k, v in batch_fetches.get(LEARNER_STATS_KEY, {}).items():
                    iter_extra_fetches[k].append(v)
            logger.debug("{} {}".format(i, averaged(iter_extra_fetches)))
            # print("done one pass of minibatches")
            needed_keys = {"obs", "dones", "oldpd", "vtarg"}

            seg_buf = [{k: seg[k] for k in needed_keys} for seg in seg_buf]
            

            def forward(seg):
                logits, state = model.forward(seg, None, None)
                return logits, state

            def compute_aux_loss(aux, seg):
                vtarg = seg["vtarg"]
                return {
                    "vf_aux": 0.5 * ((aux["vpredaux"] - vtarg) ** 2).mean(),
                    "vf_true": 0.5 * ((aux["vpredtrue"] - vtarg) ** 2).mean(),
                }

            #compute presleep outputs for replay buffer (what does this mean?)
            for seg in seg_buf:
                seg["obs"] = th.from_numpy(seg["obs"]).to(th.cuda.current_device())
                logits, state = tu.minibatched_call(forward, 4, seg=seg)
                seg["oldpd"] = logits
                # print("presleep oldpd", seg["oldpd"])
                # print("calculated old pd", seg["oldpd"])

            #train on replay buffer
            for i in range(9):
                for mb in make_minibatches(seg_buf, 8):
                    mb = tree_map(lambda x: x.to(tu.dev()), mb)
                    print("oldpd", mb['oldpd'])
                    logits, state = model.forward(mb, None, None)
                    print("new pd", logits)
                    # pd = dist_class(logits, model)
                    # print("newpd", pd)
                    # name2loss = {}
                    # name2loss["pol_distance"] = td.kl_divergence(mb["oldpd"], pd).mean()
                    # print("pol dist", name2loss["pol_distance"])
                    # name2loss.update(compute_aux_loss(aux, mb))
            seg_buf.clear()
        fetches[policy_id] = averaged(iter_extra_fetches)
    return fetches